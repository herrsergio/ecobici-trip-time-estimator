"""Train a Keras MLP with start + end station embeddings.

Uses the Apple GPU via tensorflow-metal when available.

Architecture:

* ``start_station`` -> 16-dim Embedding
* ``end_station`` -> 16-dim Embedding (shares vocabulary indexing scheme
  with start_station so a single mapping can be re-used at inference)
* ``gender`` -> 2-dim Embedding
* Numeric features -> StandardScaler
* Concatenate -> Dense(128, relu) -> Dropout -> Dense(64, relu) -> Dropout
  -> Dense(1) (regresses log-duration)

The vocabularies and the scaler are fit on the training split only, then
re-used at inference time. Unknown stations / genders are mapped to
index 0 so prediction is well-defined for categories not seen during
training.

Run::

    python scripts/train_keras.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from features import (
    KERAS_NUMERIC,
    add_features,
    apply_target_encoders,
    compute_target_encoders,
    time_split,
)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MODEL_DIR = ART / "keras_model"
PREP_PATH = ART / "keras_preproc.joblib"

# Index 0 of every embedding is reserved for unseen / out-of-vocabulary
# categories. This guarantees prediction never crashes on a new station.
OOV_IDX = 0


def _encode_inputs(
    df: pd.DataFrame,
    station_to_idx: dict[str, int],
    gender_to_idx: dict[str, int],
    scaler: StandardScaler,
) -> dict[str, np.ndarray]:
    """Convert a DataFrame into the dict of NumPy arrays the model expects.

    The same ``station_to_idx`` lookup is used for both the start and the
    end station, since both columns share the same vocabulary of station
    IDs.

    Args:
        df: DataFrame already passed through ``add_features``.
        station_to_idx: Lookup mapping station id (str) to embedding index.
        gender_to_idx: Lookup mapping gender (str) to embedding index.
        scaler: Fitted StandardScaler for the numeric block.

    Returns:
        Dict with four keys (``start_station``, ``end_station``,
        ``gender``, ``numeric``) wired to the four named inputs of the
        Keras model.
    """
    start_idx = (
        df["start_station"]
        .astype(str)
        .map(station_to_idx)
        .fillna(OOV_IDX)
        .astype("int32")
        .to_numpy()
    )
    end_idx = (
        df["end_station"].astype(str).map(station_to_idx).fillna(OOV_IDX).astype("int32").to_numpy()
    )
    gender_idx = (
        df["gender"].astype(str).map(gender_to_idx).fillna(OOV_IDX).astype("int32").to_numpy()
    )
    numeric = scaler.transform(df[KERAS_NUMERIC].to_numpy()).astype("float32")
    return {
        "start_station": start_idx,
        "end_station": end_idx,
        "gender": gender_idx,
        "numeric": numeric,
    }


def _build_model(
    n_stations: int,
    n_genders: int,
    n_numeric: int,
    station_dim: int = 16,
) -> tf.keras.Model:
    """Define and compile the MLP architecture.

    Args:
        n_stations: Size of the station vocabulary including the OOV slot.
        n_genders: Size of the gender vocabulary including the OOV slot.
        n_numeric: Number of numeric input features.
        station_dim: Dimensionality of the station embedding.

    Returns:
        A compiled ``tf.keras.Model`` ready for ``fit``.
    """
    start_in = tf.keras.Input(shape=(), dtype="int32", name="start_station")
    end_in = tf.keras.Input(shape=(), dtype="int32", name="end_station")
    gender_in = tf.keras.Input(shape=(), dtype="int32", name="gender")
    numeric_in = tf.keras.Input(shape=(n_numeric,), dtype="float32", name="numeric")

    # Both station inputs share the same embedding table so the model can
    # learn a single representation per station id regardless of whether
    # it appears as origin or destination.
    station_embedding = tf.keras.layers.Embedding(n_stations, station_dim, name="station_emb")
    start_emb = tf.keras.layers.Flatten()(station_embedding(start_in))
    end_emb = tf.keras.layers.Flatten()(station_embedding(end_in))

    gender_emb = tf.keras.layers.Embedding(n_genders, 2, name="gender_emb")(gender_in)
    gender_emb = tf.keras.layers.Flatten()(gender_emb)

    x = tf.keras.layers.Concatenate()([start_emb, end_emb, gender_emb, numeric_in])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, name="log_duration")(x)

    model = tf.keras.Model(
        inputs={
            "start_station": start_in,
            "end_station": end_in,
            "gender": gender_in,
            "numeric": numeric_in,
        },
        outputs=out,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main() -> None:
    """Run the Keras training pipeline end to end.

    Side effects: writes ``keras_model/model.keras`` and
    ``keras_preproc.joblib`` into the ``artifacts/`` folder.
    """
    print("TF version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs detected:", gpus if gpus else "none (training on CPU)")

    df = pd.read_parquet(ART / "rides.parquet")
    df = add_features(df)
    splits = time_split(df)
    print(
        f"Train/val/test rows: {len(splits.train):,} / {len(splits.val):,} / {len(splits.test):,}"
    )

    # Fit the same target encoders the LightGBM path uses, so the MLP
    # gets pair_te_mean and the other TE columns as inputs. Encoders are
    # fit on the training split only to preserve the leak-safety property.
    print("Fitting target encoders on training split...")
    encoders = compute_target_encoders(splits.train, smoothing_station=50.0, smoothing_pair=100.0)
    print(
        f"  station encoder: {len(encoders['station']):,} stations, "
        f"{len(encoders['pair']):,} (start, end) pairs"
    )
    print("Applying encoders to train / val / test...")
    splits.train = apply_target_encoders(splits.train, encoders)
    splits.val = apply_target_encoders(splits.val, encoders)
    splits.test = apply_target_encoders(splits.test, encoders)

    # Vocabularies are built from the training set only so we never leak
    # val/test categories into the model's category space. We build a
    # single station vocabulary by unioning origin and destination ids so
    # the shared embedding can map any station regardless of role.
    start_stations = splits.train["start_station"].astype(str).unique()
    end_stations = splits.train["end_station"].astype(str).unique()
    stations = sorted(set(start_stations) | set(end_stations))
    genders = sorted(splits.train["gender"].astype(str).unique())
    station_to_idx = {s: i + 1 for i, s in enumerate(stations)}  # 0 = OOV
    gender_to_idx = {g: i + 1 for i, g in enumerate(genders)}

    scaler = StandardScaler().fit(splits.train[KERAS_NUMERIC].to_numpy())

    X_tr = _encode_inputs(splits.train, station_to_idx, gender_to_idx, scaler)
    X_va = _encode_inputs(splits.val, station_to_idx, gender_to_idx, scaler)
    X_te = _encode_inputs(splits.test, station_to_idx, gender_to_idx, scaler)
    y_tr = splits.train["log_duration"].to_numpy().astype("float32")
    y_va = splits.val["log_duration"].to_numpy().astype("float32")
    y_te = splits.test["log_duration"].to_numpy().astype("float32")

    model = _build_model(
        n_stations=len(station_to_idx) + 1,
        n_genders=len(gender_to_idx) + 1,
        n_numeric=len(KERAS_NUMERIC),
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=4,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=2,
            factor=0.5,
            monitor="val_loss",
        ),
    ]

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=30,
        batch_size=4096,
        callbacks=callbacks,
        verbose=2,
    )

    yhat_te = model.predict(X_te, batch_size=8192, verbose=0).ravel()
    y_sec = np.expm1(y_te)
    p_sec = np.expm1(yhat_te)
    metrics = {
        "mae_sec": float(mean_absolute_error(y_sec, p_sec)),
        "rmse_sec": float(np.sqrt(mean_squared_error(y_sec, p_sec))),
        "r2_log": float(r2_score(y_te, yhat_te)),
        "mape_pct": float(np.mean(np.abs((y_sec - p_sec) / np.maximum(y_sec, 1))) * 100),
    }
    print("\nTest metrics:", json.dumps(metrics, indent=2))

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    model.save(MODEL_DIR / "model.keras")
    joblib.dump(
        {
            "station_to_idx": station_to_idx,
            "gender_to_idx": gender_to_idx,
            "scaler": scaler,
            "numeric_features": KERAS_NUMERIC,
            "encoders": encoders,
            "metrics": metrics,
        },
        PREP_PATH,
    )
    print(f"Saved model to {MODEL_DIR} and preproc to {PREP_PATH}")


if __name__ == "__main__":
    main()
