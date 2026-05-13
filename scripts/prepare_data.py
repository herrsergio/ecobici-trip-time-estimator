"""Build the Parquet dataset used by all training scripts.

Reads every ``data/*.csv.xz`` file, normalizes columns (the 2024-01 file
quotes ``"Fecha Arribo"`` with a space — we rename it), computes the
``duration_sec`` target by subtracting the start datetime from the end
datetime, drops implausible records, and writes a single Parquet file at
``artifacts/rides.parquet``.

Run::

    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)
OUT_PATH = ART_DIR / "rides.parquet"

# Filters used to drop records that almost certainly do not represent real
# rides. The min/max durations match Ecobici's 45-minute free-ride cap with
# a wide tolerance for edge cases.
MIN_DURATION_SEC = 60  # < 1 min: false starts or immediate re-docks.
MAX_DURATION_SEC = 3 * 3600  # > 3 h: forgotten or lost bikes.
MIN_AGE, MAX_AGE = 16, 90

# 2024-01.csv.xz uses "Fecha Arribo" (with a space, quoted in the header)
# instead of "Fecha_Arribo". Every other file uses the underscore variant.
COLUMN_RENAMES = {"Fecha Arribo": "Fecha_Arribo"}

NEEDED_COLS = [
    "Genero_Usuario",
    "Edad_Usuario",
    "Ciclo_Estacion_Retiro",
    "Ciclo_EstacionArribo",
    "Fecha_Retiro",
    "Hora_Retiro",
    "Fecha_Arribo",
    "Hora_Arribo",
]


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single ``.csv.xz`` file and return only the needed columns.

    Station IDs are read as strings to preserve the leading zeros used by
    the source files (e.g. ``"015"``).

    Args:
        path: Path to a single ``data/YYYY-MM.csv.xz`` file.

    Returns:
        DataFrame with the seven columns listed in ``NEEDED_COLS``.

    Raises:
        ValueError: if the file is missing any of the needed columns.
    """
    df = pd.read_csv(
        path,
        compression="xz",
        dtype={
            "Ciclo_Estacion_Retiro": "string",
            "Ciclo_EstacionArribo": "string",
            "Bici": "string",
        },
        low_memory=False,
    )
    df = df.rename(columns=COLUMN_RENAMES)
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df[NEEDED_COLS].copy()


def _to_datetime(date_str: pd.Series, time_str: pd.Series) -> pd.Series:
    """Combine ``DD/MM/YYYY`` and ``HH:MM:SS`` strings into a datetime Series.

    Args:
        date_str: Series of date strings in ``DD/MM/YYYY`` format.
        time_str: Series of time strings in ``HH:MM:SS`` format.

    Returns:
        A datetime64 Series. Unparseable rows become ``NaT`` and are
        filtered out by the caller.
    """
    combined = date_str.astype("string").str.strip() + " " + time_str.astype("string").str.strip()
    return pd.to_datetime(combined, format="%d/%m/%Y %H:%M:%S", errors="coerce")


def main() -> None:
    """Read all monthly files, clean them, and persist a single Parquet.

    Prints a short summary of how many rows survived filtering, the date
    range covered, ride duration quantiles, and the station vocabulary
    size — useful sanity checks before training.
    """
    files = sorted(DATA_DIR.glob("*.csv.xz"))
    if not files:
        sys.exit(f"No .csv.xz files in {DATA_DIR}")
    print(f"Found {len(files)} files")

    frames: list[pd.DataFrame] = []
    for path in tqdm(files, desc="Reading"):
        try:
            frames.append(_read_one(path))
        except Exception as exc:
            # We intentionally swallow per-file errors so a single corrupt
            # file does not block the rest of the dataset.
            print(f"WARN: failed to read {path.name}: {exc}")

    df = pd.concat(frames, ignore_index=True)
    raw_rows = len(df)
    print(f"Raw rows: {raw_rows:,}")

    df["start_dt"] = _to_datetime(df["Fecha_Retiro"], df["Hora_Retiro"])
    df["end_dt"] = _to_datetime(df["Fecha_Arribo"], df["Hora_Arribo"])
    df["duration_sec"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds()

    df["Edad_Usuario"] = pd.to_numeric(df["Edad_Usuario"], errors="coerce")
    df["Genero_Usuario"] = df["Genero_Usuario"].astype("string").str.upper().str.strip()

    mask = (
        df["start_dt"].notna()
        & df["end_dt"].notna()
        & df["duration_sec"].between(MIN_DURATION_SEC, MAX_DURATION_SEC)
        & df["Edad_Usuario"].between(MIN_AGE, MAX_AGE)
        & df["Genero_Usuario"].isin(["M", "F"])
        & df["Ciclo_Estacion_Retiro"].notna()
        & df["Ciclo_EstacionArribo"].notna()
    )

    df = df.loc[
        mask,
        [
            "Genero_Usuario",
            "Edad_Usuario",
            "Ciclo_Estacion_Retiro",
            "Ciclo_EstacionArribo",
            "start_dt",
            "duration_sec",
        ],
    ].copy()

    df = df.rename(
        columns={
            "Genero_Usuario": "gender",
            "Edad_Usuario": "age",
            "Ciclo_Estacion_Retiro": "start_station",
            "Ciclo_EstacionArribo": "end_station",
        }
    )
    df["age"] = df["age"].astype("int16")
    df["gender"] = df["gender"].astype("category")
    df["start_station"] = df["start_station"].astype("category")
    df["end_station"] = df["end_station"].astype("category")
    df = df.sort_values("start_dt", ignore_index=True)

    kept_ratio = len(df) / raw_rows
    print(f"Kept {len(df):,} of {raw_rows:,} rows ({kept_ratio:.1%})")
    print(f"Date range: {df['start_dt'].min()} to {df['start_dt'].max()}")
    print(
        f"Duration p50/p95: {df['duration_sec'].median():.0f}s / "
        f"{df['duration_sec'].quantile(0.95):.0f}s"
    )
    print(
        f"Unique start/end stations: {df['start_station'].nunique()} / "
        f"{df['end_station'].nunique()}"
    )

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
