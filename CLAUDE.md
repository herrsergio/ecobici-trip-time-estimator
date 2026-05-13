# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Trip-time estimator for the Mexico City public bike-sharing service "Ecobici". Given origin station, destination station, rider demographics, and start datetime, predicts ride duration in seconds. The repo trains two models side-by-side — LightGBM (CPU) and a Keras MLP with shared start/end station embeddings (Apple GPU via tensorflow-metal) — so they can be compared on the same chronological holdout.

An earlier framing used only ride-start features (origin + demographics + time). That model hit a hard structural ceiling at R² ≈ 0.09 because ~90% of duration variance is destination-driven. The current framing includes the destination as an input and is therefore the right model for a route-planner UX where the rider has already chosen a destination.

## Setup and common commands

```bash
# One-time: create .venv with a TF-compatible Python (3.10/3.11/3.12 auto-pick)
./setup_env.sh
source .venv/bin/activate

# End-to-end pipeline
python scripts/prepare_data.py        # data/*.csv.xz -> artifacts/rides.parquet
python scripts/train_lightgbm.py      # writes artifacts/lgbm_*
python scripts/train_keras.py         # writes artifacts/keras_*
python scripts/compare.py             # side-by-side test metrics

# Single-row inference (start and end station are both required)
python scripts/predict.py --model lgbm --gender M --age 32 \
    --start-station 042 --end-station 113 \
    --datetime "2026-05-12 08:30:00"

# Lint / format (dev deps in requirements-dev.txt)
ruff check scripts/
ruff format scripts/
```

`setup_env.sh` deliberately avoids Python 3.13+ because TensorFlow does not ship wheels for it yet. Override the interpreter with `PYTHON=/path/to/python3.11 ./setup_env.sh`.

LightGBM on macOS needs `libomp` at runtime (`brew install libomp`) — it is NOT bundled in the wheel.

## Architecture

The pipeline is a four-stage producer-consumer chain that talks via files in `artifacts/`:

```
data/*.csv.xz
   |
   v  prepare_data.py
artifacts/rides.parquet           <- single cleaned dataset
   |
   +--> train_lightgbm.py  -->  lgbm_model.txt + lgbm_encoders.joblib + lgbm_meta.json
   |
   +--> train_keras.py     -->  keras_model/model.keras + keras_preproc.joblib
   |
   v  compare.py / predict.py
```

`scripts/features.py` is the **single source of truth** for feature engineering and the train/val/test split. Both training scripts and `predict.py` import from it; changes to it ripple through train- and serve-time logic in lockstep. Do not duplicate feature logic into the training scripts.

### Invariants the codebase depends on

These are not obvious from reading any single file. Violating them silently breaks accuracy or leaks the test set:

1. **No-leakage feature set**: the model is meant to score a trip at the moment the rider has just chosen origin and destination. Allowed inputs: gender, age, start_station, end_station, start_dt. **Arrival time and duration must never enter the feature columns.** The `Fecha_Arribo` / `Hora_Arribo` raw columns are used only to compute the target during `prepare_data.py`, then dropped. `Ciclo_EstacionArribo` is kept (as `end_station`) because the destination is assumed known at scoring time.

2. **Target is `log1p(duration_sec)`**, persisted as `log_duration` by `add_features`. Every prediction path must `np.expm1(...)` before reporting seconds. Negative or zero predictions cannot occur because of this transform.

3. **Chronological split, not random**: `time_split` sorts by `start_dt` and slices oldest -> train, newest -> test. Both training scripts call it. Switching to random splitting would leak future-into-past patterns.

4. **Target encoders are fit on `splits.train` only**, then `apply_target_encoders` merges the lookup tables onto val/test. The dominant signal is the (start_station, end_station) pair encoder: it answers "how long do trips on this route typically take?" Per-station fallbacks (single-station, station × weekend, station × hour_bucket, end_station) cover sparse or unseen pairs. The smoothing prior (`global_mean`) handles routes never seen at training time.

5. **Station IDs are strings, not ints**: source files use leading zeros (`"015"`, `"008"`). `prepare_data.py` reads `Ciclo_Estacion_Retiro` and `Ciclo_EstacionArribo` with `dtype="string"`. Coercing to int destroys distinct station identities.

6. **Header quirk in `2024-01.csv.xz`**: that one file uses `"Fecha Arribo"` (quoted, with a space) instead of `Fecha_Arribo`. `prepare_data.COLUMN_RENAMES` normalizes it. Any new file that introduces a similar variant should be added there.

### Inference contract

`predict.py` reconstructs the train-time feature pipeline exactly:

- LightGBM path: loads model + encoders + meta, applies `add_features` -> `apply_target_encoders`, and recreates the original `pd.Categorical` orderings from `meta.json` (`station_categories`, `end_station_categories`, `gender_categories`, `hour_bucket_categories`) so the booster sees the same category codes it learned with.
- Keras path: loads the saved model + the `joblib` preproc bundle (vocab dicts + StandardScaler), maps unknown categories to index 0 (the reserved OOV slot). The same `station_to_idx` is used for both start and end station inputs because the model shares a single station embedding table.

If you add a new feature in `features.py`, you must also:
- Add it to `GBM_FEATURES` (LightGBM) and/or `KERAS_NUMERIC` (Keras).
- If it's categorical, add it to `CAT_COLS` in `train_lightgbm.py` and persist its categories in the meta JSON.
- Update `apply_target_encoders` if the feature is part of an encoder lookup.

## Tooling

Linting / formatting is configured in `pyproject.toml` (ruff). The config enables PEP 8, pyflakes, isort, bugbear, pyupgrade, pandas-vet, pydocstyle (Google), and NumPy rules. `features` is registered as a first-party module for import-sort. `ruff check` and `ruff format` must both pass; CI for this repo (if added) should run both.
