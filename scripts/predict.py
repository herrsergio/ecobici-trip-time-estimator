"""CLI for single-row trip-duration prediction with either trained model.

Both prediction paths share the same feature-engineering function as the
training scripts, which keeps train- and serve-time logic in lockstep.

Examples::

    python scripts/predict.py --model lgbm  --gender M --age 32 \
        --start-station 042 --end-station 113 \
        --datetime "2026-05-12 08:30:00"

    python scripts/predict.py --model keras --gender F --age 28 \
        --start-station 113 --end-station 008 \
        --datetime "2026-05-12 18:00:00"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import KERAS_NUMERIC, add_features, apply_target_encoders

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def _row_to_features(args: argparse.Namespace) -> pd.DataFrame:
    """Turn the parsed CLI args into a 1-row DataFrame of features.

    Args:
        args: Parsed argparse Namespace with gender / age / start_station
            / end_station / datetime fields.

    Returns:
        A 1-row DataFrame already augmented by ``add_features``.
    """
    df = pd.DataFrame(
        [
            {
                "gender": args.gender.upper(),
                "age": int(args.age),
                "start_station": str(args.start_station),
                "end_station": str(args.end_station),
                "start_dt": pd.to_datetime(args.datetime),
            }
        ]
    )
    return add_features(df)


def predict_lgbm(df_feat: pd.DataFrame) -> float:
    """Score one feature row using the persisted LightGBM model.

    Loads the model, the encoders, and the meta JSON from
    ``artifacts/``. Applies the target encoders (the heavy-lift signal
    in the route-time-estimator framing) and recreates the original
    training-time category orderings so the LightGBM Booster sees the
    same codes it learned with.

    Args:
        df_feat: 1-row DataFrame from ``_row_to_features``.

    Returns:
        Predicted ride duration in seconds.
    """
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(ART / "lgbm_model.txt"))
    meta = json.loads((ART / "lgbm_meta.json").read_text())
    encoders = joblib.load(ART / "lgbm_encoders.joblib")

    df_feat = apply_target_encoders(df_feat, encoders)

    X = df_feat[meta["features"]].copy()
    X["start_station"] = pd.Categorical(
        X["start_station"].astype(str),
        categories=meta["station_categories"],
    )
    X["end_station"] = pd.Categorical(
        X["end_station"].astype(str),
        categories=meta["end_station_categories"],
    )
    X["gender"] = pd.Categorical(
        X["gender"].astype(str),
        categories=meta["gender_categories"],
    )
    if "hour_bucket" in meta.get("cat_features", []):
        X["hour_bucket"] = pd.Categorical(
            X["hour_bucket"].astype(int),
            categories=meta.get("hour_bucket_categories", [0, 1, 2, 3, 4]),
        )
    yhat_log = model.predict(X, num_iteration=meta["best_iteration"])
    return float(np.expm1(yhat_log)[0])


def predict_keras(df_feat: pd.DataFrame) -> float:
    """Score one feature row using the persisted Keras model.

    Args:
        df_feat: 1-row DataFrame from ``_row_to_features``.

    Returns:
        Predicted ride duration in seconds.
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(ART / "keras_model" / "model.keras")
    prep = joblib.load(ART / "keras_preproc.joblib")

    df_feat = apply_target_encoders(df_feat, prep["encoders"])

    start_idx = (
        df_feat["start_station"]
        .astype(str)
        .map(prep["station_to_idx"])
        .fillna(0)
        .astype("int32")
        .to_numpy()
    )
    end_idx = (
        df_feat["end_station"]
        .astype(str)
        .map(prep["station_to_idx"])
        .fillna(0)
        .astype("int32")
        .to_numpy()
    )
    gender_idx = (
        df_feat["gender"]
        .astype(str)
        .map(prep["gender_to_idx"])
        .fillna(0)
        .astype("int32")
        .to_numpy()
    )
    numeric = prep["scaler"].transform(df_feat[KERAS_NUMERIC].to_numpy()).astype("float32")
    yhat_log = model.predict(
        {
            "start_station": start_idx,
            "end_station": end_idx,
            "gender": gender_idx,
            "numeric": numeric,
        },
        verbose=0,
    ).ravel()
    return float(np.expm1(yhat_log)[0])


def main() -> None:
    """Parse CLI args, build the feature row, and print the prediction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "keras"], default="lgbm")
    parser.add_argument("--gender", required=True, help="M or F")
    parser.add_argument("--age", required=True, type=int)
    parser.add_argument(
        "--start-station",
        required=True,
        help="Origin station id, e.g. 042",
    )
    parser.add_argument(
        "--end-station",
        required=True,
        help="Destination station id, e.g. 113",
    )
    parser.add_argument(
        "--datetime",
        required=True,
        help='Start datetime, e.g. "2026-05-12 08:30:00"',
    )
    args = parser.parse_args()

    df_feat = _row_to_features(args)
    pred_sec = predict_lgbm(df_feat) if args.model == "lgbm" else predict_keras(df_feat)
    print(f"Predicted duration: {pred_sec:.0f} seconds ({pred_sec / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
