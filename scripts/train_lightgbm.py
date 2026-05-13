"""Train a LightGBM regressor on log-duration.

Pipeline:

1. Load the cleaned Parquet produced by ``prepare_data.py``.
2. Add features and split chronologically.
3. Fit smoothed station target encoders on the training split.
4. Apply the encoders to all three splits.
5. Train LightGBM with early stopping on the validation split.
6. Persist the model, the encoders, and a meta JSON describing how to
   reproduce the feature pipeline at inference time.

Run::

    python scripts/train_lightgbm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import (
    GBM_FEATURES,
    add_features,
    apply_target_encoders,
    compute_target_encoders,
    time_split,
)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MODEL_PATH = ART / "lgbm_model.txt"
META_PATH = ART / "lgbm_meta.json"
ENCODERS_PATH = ART / "lgbm_encoders.joblib"

# Categorical features handed to LightGBM as native categoricals so the
# library learns optimal partitions of the categories instead of treating
# them as ordinal integers.
CAT_COLS = ["gender", "start_station", "end_station", "hour_bucket"]


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Slice ``df`` down to the GBM feature columns and cast categoricals.

    Args:
        df: DataFrame already passed through ``add_features`` and
            ``apply_target_encoders``.

    Returns:
        A ``(X, y)`` tuple where ``X`` contains only the model features
        (with proper ``category`` dtypes) and ``y`` is the target.
    """
    X = df[GBM_FEATURES].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype("category")
    y = df["log_duration"].to_numpy()
    return X, y


def _metrics(y_log: np.ndarray, yhat_log: np.ndarray) -> dict:
    """Compute regression metrics in both log-space and seconds.

    Args:
        y_log: Ground-truth log-durations on the test split.
        yhat_log: Model predictions in log-space.

    Returns:
        Dict with MAE/RMSE in seconds, R^2 in log-space, and a MAPE
        guarded against division by very small durations.
    """
    y = np.expm1(y_log)
    p = np.expm1(yhat_log)
    return {
        "mae_sec": float(mean_absolute_error(y, p)),
        "rmse_sec": float(np.sqrt(mean_squared_error(y, p))),
        "r2_log": float(r2_score(y_log, yhat_log)),
        "mape_pct": float(np.mean(np.abs((y - p) / np.maximum(y, 1))) * 100),
    }


def main() -> None:
    """Run the full LightGBM training pipeline end to end.

    Side effects: writes ``lgbm_model.txt``, ``lgbm_encoders.joblib``, and
    ``lgbm_meta.json`` into the ``artifacts/`` folder.
    """
    df = pd.read_parquet(ART / "rides.parquet")
    df = add_features(df)
    splits = time_split(df)
    print(
        f"Train/val/test rows: {len(splits.train):,} / {len(splits.val):,} / {len(splits.test):,}"
    )

    print("Fitting target encoders on training split...")
    encoders = compute_target_encoders(splits.train, smoothing_station=50.0, smoothing_pair=100.0)
    print(
        "  station encoder: "
        f"{len(encoders['station']):,} stations, "
        f"{len(encoders['pair']):,} (start, end) pairs, "
        f"global_mean(log_duration)={encoders['global_mean']:.4f}"
    )

    print("Applying encoders to train / val / test...")
    splits.train = apply_target_encoders(splits.train, encoders)
    splits.val = apply_target_encoders(splits.val, encoders)
    splits.test = apply_target_encoders(splits.test, encoders)

    X_tr, y_tr = _prep(splits.train)
    X_va, y_va = _prep(splits.val)
    X_te, y_te = _prep(splits.test)

    train_set = lgb.Dataset(
        X_tr,
        label=y_tr,
        categorical_feature=CAT_COLS,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_va,
        label=y_va,
        categorical_feature=CAT_COLS,
        reference=train_set,
        free_raw_data=False,
    )

    # With the pair encoder now in play, the signal is much stronger and we
    # do not need 255 leaves to fit it. Drop back to 127 for ~2x wall-clock
    # savings; early stopping will trim rounds as soon as val plateaus.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=4000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(75), lgb.log_evaluation(100)],
    )

    yhat_te = model.predict(X_te, num_iteration=model.best_iteration)
    metrics = _metrics(y_te, yhat_te)
    print("\nTest metrics:", json.dumps(metrics, indent=2))

    imp = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "gain": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    print("\nFeature importance (gain):")
    print(imp.to_string(index=False))

    ART.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    joblib.dump(encoders, ENCODERS_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "features": GBM_FEATURES,
                "cat_features": CAT_COLS,
                "metrics": metrics,
                "best_iteration": int(model.best_iteration),
                "station_categories": list(X_tr["start_station"].cat.categories.astype(str)),
                "end_station_categories": list(X_tr["end_station"].cat.categories.astype(str)),
                "gender_categories": list(X_tr["gender"].cat.categories.astype(str)),
                "hour_bucket_categories": [int(c) for c in X_tr["hour_bucket"].cat.categories],
            },
            indent=2,
        )
    )
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved encoders to {ENCODERS_PATH}")


if __name__ == "__main__":
    main()
