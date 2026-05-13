"""Shared feature engineering, time-aware splitting, and station target encoders.

Three responsibilities live here so that train / predict scripts share the
exact same logic and stay aligned at inference time:

1. ``add_features`` derives calendar and cyclic time features from
   ``start_dt``. It is the single source of truth for what a "raw row"
   becomes after feature engineering.
2. ``time_split`` partitions the data chronologically (oldest -> train,
   newest -> test). This matches how the model will be deployed: trained on
   the past, used on the future.
3. ``compute_target_encoders`` / ``apply_target_encoders`` build smoothed
   per-station aggregates of the target (``log_duration``). Encoders are
   fit on the training split only; merging them onto val/test/predict-time
   rows is leak-safe because the lookup tables never contain val/test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Hour-of-day buckets capturing typical CDMX commuter patterns. We use the
# same edges everywhere so that station x hour_bucket encodings line up
# between training and prediction.
HOUR_BIN_EDGES = [-1, 5, 9, 15, 19, 23]
HOUR_BIN_LABELS = ["night", "morning_rush", "midday", "evening_rush", "evening"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar, cyclic, and hour-bucket features from ``start_dt``.

    Adds the regression target ``log_duration`` only when ``duration_sec`` is
    already present, so this function is safe to call on prediction-time
    rows that only know the start side.

    Args:
        df: DataFrame with at least a ``start_dt`` datetime column.

    Returns:
        A new DataFrame (input not mutated) with the engineered columns
        appended.
    """
    out = df.copy()
    dt = out["start_dt"]

    # Plain calendar parts. Trees can use these directly; the cyclic
    # versions below are mainly for the Keras model.
    out["hour"] = dt.dt.hour.astype("int8")
    out["dow"] = dt.dt.dayofweek.astype("int8")
    out["month"] = dt.dt.month.astype("int8")
    out["day"] = dt.dt.day.astype("int8")
    out["year"] = dt.dt.year.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["hour_bucket"] = pd.cut(
        out["hour"],
        bins=HOUR_BIN_EDGES,
        labels=False,
        include_lowest=True,
    ).astype("int8")

    # Cyclic encodings so that "23:00" and "00:00" are neighbours for the
    # neural net. LightGBM does not need these but they are cheap.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)

    if "duration_sec" in out.columns:
        # log1p compresses the long right tail of ride durations and avoids
        # negative predictions after expm1 at inference time.
        out["log_duration"] = np.log1p(out["duration_sec"].astype("float64"))

    return out


@dataclass
class Splits:
    """Container for the three chronological partitions of the dataset."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def time_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Splits:
    """Split ``df`` chronologically: oldest rows -> train, newest -> test.

    Args:
        df: DataFrame that contains a ``start_dt`` column.
        val_frac: Fraction of rows assigned to the validation slice.
        test_frac: Fraction of rows assigned to the test slice.

    Returns:
        A ``Splits`` dataclass with non-overlapping ``train``, ``val``,
        and ``test`` DataFrames in chronological order.
    """
    df = df.sort_values("start_dt").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return Splits(
        train=df.iloc[:n_train].copy(),
        val=df.iloc[n_train : n_train + n_val].copy(),
        test=df.iloc[n_train + n_val :].copy(),
    )


# ---------------------------------------------------------------------------
# Smoothed target encoders
# ---------------------------------------------------------------------------


def _smoothed_mean(
    group: pd.core.groupby.SeriesGroupBy,
    prior: float,
    smoothing: float,
) -> pd.Series:
    """Compute a Bayesian-smoothed mean per group.

    The smoothing pulls small groups toward the global ``prior`` so a
    station with only a handful of rides does not produce a wild estimate.

    Args:
        group: SeriesGroupBy where the underlying values are the target.
        prior: Value to shrink toward (typically the global mean).
        smoothing: Pseudo-count weight given to the prior.

    Returns:
        A Series indexed by group with the smoothed mean.
    """
    means = group.mean()
    counts = group.count()
    return (counts * means + smoothing * prior) / (counts + smoothing)


def compute_target_encoders(
    train: pd.DataFrame,
    smoothing_station: float = 50.0,
    smoothing_pair: float = 100.0,
) -> dict:
    """Fit smoothed encoders of ``log_duration`` on ``train``.

    Five lookup tables are produced, all derived from the training split
    only so they can be merged onto val / test / live-prediction rows
    without leakage:

    * ``station``: smoothed mean, p95, and ``log1p(count)`` per start
      station (kept mostly as a fallback if the pair below is unseen).
    * ``station_weekend``: smoothed mean per (start_station, is_weekend).
    * ``station_hourbucket``: smoothed mean per (start_station, hour_bucket).
    * ``end_station``: smoothed mean per arrival station.
    * ``pair``: smoothed mean, p95 and ``log1p(count)`` per
      (start_station, end_station). This is the dominant signal in the
      route-time-estimator framing.

    The pair table uses a higher smoothing prior because some
    origin/destination pairs see very few rides in the training data.

    Args:
        train: DataFrame containing ``start_station``, ``end_station``,
            ``is_weekend``, ``hour_bucket``, and ``log_duration``.
        smoothing_station: Pseudo-count weight for per-station encoders.
        smoothing_pair: Pseudo-count weight for the (start, end) pair
            encoder; higher because pairs are much sparser than stations.

    Returns:
        A dict with the encoder tables and the global prior used during
        smoothing.
    """
    df = train.copy()
    df["start_station"] = df["start_station"].astype(str)
    df["end_station"] = df["end_station"].astype(str)

    global_mean = float(df["log_duration"].mean())

    station_grp = df.groupby("start_station", observed=True)["log_duration"]
    station_tbl = pd.DataFrame(
        {
            "station_te_mean": _smoothed_mean(station_grp, global_mean, smoothing_station),
            "station_te_p95": station_grp.quantile(0.95),
            "station_count_log": np.log1p(station_grp.count()),
        }
    ).reset_index()

    we_grp = df.groupby(["start_station", "is_weekend"], observed=True)["log_duration"]
    weekend_tbl = (
        _smoothed_mean(we_grp, global_mean, smoothing_station)
        .rename("station_weekend_te_mean")
        .reset_index()
    )

    hb_grp = df.groupby(["start_station", "hour_bucket"], observed=True)["log_duration"]
    hourbucket_tbl = (
        _smoothed_mean(hb_grp, global_mean, smoothing_station)
        .rename("station_hourbucket_te_mean")
        .reset_index()
    )

    end_grp = df.groupby("end_station", observed=True)["log_duration"]
    end_tbl = (
        _smoothed_mean(end_grp, global_mean, smoothing_station)
        .rename("end_station_te_mean")
        .reset_index()
    )

    pair_grp = df.groupby(["start_station", "end_station"], observed=True)["log_duration"]
    pair_tbl = pd.DataFrame(
        {
            "pair_te_mean": _smoothed_mean(pair_grp, global_mean, smoothing_pair),
            "pair_te_p95": pair_grp.quantile(0.95),
            "pair_count_log": np.log1p(pair_grp.count()),
        }
    ).reset_index()

    return {
        "global_mean": global_mean,
        "station": station_tbl,
        "station_weekend": weekend_tbl,
        "station_hourbucket": hourbucket_tbl,
        "end_station": end_tbl,
        "pair": pair_tbl,
    }


TE_FEATURES = [
    "station_te_mean",
    "station_te_p95",
    "station_count_log",
    "station_weekend_te_mean",
    "station_hourbucket_te_mean",
    "end_station_te_mean",
    "pair_te_mean",
    "pair_te_p95",
    "pair_count_log",
]


def apply_target_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Merge the encoder lookup tables onto ``df``.

    Unknown stations or pairs (not seen at training time) receive the
    global prior so prediction is still well-defined for new routes.

    Args:
        df: DataFrame already augmented by ``add_features`` (needs
            ``start_station``, ``end_station``, ``is_weekend``,
            ``hour_bucket``).
        encoders: Output of ``compute_target_encoders``.

    Returns:
        A new DataFrame with the ``TE_FEATURES`` columns appended.
    """
    out = df.copy()
    out["start_station"] = out["start_station"].astype(str)
    out["end_station"] = out["end_station"].astype(str)
    out["is_weekend"] = out["is_weekend"].astype("int8")
    out["hour_bucket"] = out["hour_bucket"].astype("int8")

    prior = encoders["global_mean"]

    station = encoders["station"].copy()
    station["start_station"] = station["start_station"].astype(str)
    out = out.merge(station, on="start_station", how="left")

    weekend = encoders["station_weekend"].copy()
    weekend["start_station"] = weekend["start_station"].astype(str)
    weekend["is_weekend"] = weekend["is_weekend"].astype("int8")
    out = out.merge(weekend, on=["start_station", "is_weekend"], how="left")

    hb = encoders["station_hourbucket"].copy()
    hb["start_station"] = hb["start_station"].astype(str)
    hb["hour_bucket"] = hb["hour_bucket"].astype("int8")
    out = out.merge(hb, on=["start_station", "hour_bucket"], how="left")

    end_station = encoders["end_station"].copy()
    end_station["end_station"] = end_station["end_station"].astype(str)
    out = out.merge(end_station, on="end_station", how="left")

    pair = encoders["pair"].copy()
    pair["start_station"] = pair["start_station"].astype(str)
    pair["end_station"] = pair["end_station"].astype(str)
    out = out.merge(pair, on=["start_station", "end_station"], how="left")

    mean_cols = (
        "station_te_mean",
        "station_te_p95",
        "station_weekend_te_mean",
        "station_hourbucket_te_mean",
        "end_station_te_mean",
        "pair_te_mean",
        "pair_te_p95",
    )
    for col in mean_cols:
        out[col] = out[col].fillna(prior).astype("float32")
    out["station_count_log"] = out["station_count_log"].fillna(0.0).astype("float32")
    out["pair_count_log"] = out["pair_count_log"].fillna(0.0).astype("float32")

    return out


# ---------------------------------------------------------------------------
# Public feature lists used by the training scripts
# ---------------------------------------------------------------------------

#: Columns fed to LightGBM. Includes raw categoricals plus target encodings.
GBM_FEATURES = [
    "gender",
    "age",
    "start_station",
    "end_station",
    "hour",
    "dow",
    "month",
    "day",
    "year",
    "is_weekend",
    "hour_bucket",
    *TE_FEATURES,
]

#: Numeric columns fed to the Keras MLP. Station and gender go through
#: embedding layers and are handled separately in ``train_keras.py``.
#: We include the target-encoder columns so the MLP starts from the same
#: signal LightGBM uses; without them, the network would have to discover
#: pair-level mean durations from scratch through the embedding-MLP
#: interaction.
KERAS_NUMERIC = [
    "age",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    *TE_FEATURES,
]
