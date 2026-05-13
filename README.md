# ecobici-trip-time-estimator

Trip-time estimator for the **Ecobici** public bike-sharing service in Mexico City. Given an origin station, a destination station, rider demographics, and a start datetime, predicts ride duration in seconds.

Two models are trained side-by-side on the same chronological holdout so they can be compared:

* **LightGBM** (CPU, gradient-boosted trees)
* **Keras MLP** with shared start/end station embeddings (Apple GPU via `tensorflow-metal`)

## Results

Final test-set metrics (chronological 70/15/15 split, ~7 M test rides):

| Metric         | LightGBM | Keras MLP |
|----------------|---------:|----------:|
| R² (log-space) |   0.633  |   0.545   |
| MAE (seconds)  |    257   |    311    |
| RMSE (seconds) |    514   |    559    |
| MAPE           |   28.8 % |   38.1 %  |

LightGBM is the recommended production model. The Keras MLP is kept as a sanity check and to confirm the result is not an artifact of the model class.

## Data

The 28 monthly CSV files in `data/` come from the Mexico City government open-data portal for Ecobici:

* Source: <https://ecobici.cdmx.gob.mx/datos-abiertos/>

Files are LZMA-compressed (`.csv.xz`) and named `YYYY-MM.csv.xz`. Coverage in this repo: **January 2024 through April 2026** (~48.7 M raw rows; ~47 M after cleaning).

Schema (Spanish column names):

| Column                  | Meaning                              |
|-------------------------|--------------------------------------|
| `Genero_Usuario`        | Rider gender (`M` / `F`)             |
| `Edad_Usuario`          | Rider age                            |
| `Bici`                  | Bike id (unused)                     |
| `Ciclo_Estacion_Retiro` | Origin station id (string)           |
| `Fecha_Retiro`          | Start date (`DD/MM/YYYY`)            |
| `Hora_Retiro`           | Start time (`HH:MM:SS`)              |
| `Ciclo_EstacionArribo`  | Destination station id (string)      |
| `Fecha_Arribo`          | Arrival date                         |
| `Hora_Arribo`           | Arrival time                         |

Note: the `2024-01.csv.xz` file uses `"Fecha Arribo"` (quoted, with a space) in the header. `scripts/prepare_data.py` normalizes this. Station ids have leading zeros (`015`, `008`) and are intentionally kept as strings.

## Setup

Requires Python 3.10, 3.11, or 3.12 (TensorFlow does not yet ship wheels for 3.13+). On macOS, LightGBM also needs `libomp` at runtime — install it once with `brew install libomp`.

```bash
./setup_env.sh            # creates .venv and installs runtime deps
source .venv/bin/activate

# Optional, for linting / formatting:
pip install -r requirements-dev.txt
```

`setup_env.sh` auto-picks a TF-compatible Python from your system (or pyenv). To force a specific interpreter:

```bash
PYTHON=$HOME/.pyenv/versions/3.11.14/bin/python3.11 ./setup_env.sh
```

## Usage

```bash
# 1. Build artifacts/rides.parquet (~5 min)
python scripts/prepare_data.py

# 2. Train LightGBM (~30 min on CPU)
python scripts/train_lightgbm.py

# 3. Optional: train Keras MLP (~10 min on Apple GPU)
python scripts/train_keras.py

# 4. Side-by-side metrics
python scripts/compare.py

# 5. Single-row prediction
python scripts/predict.py --model lgbm --gender M --age 32 \
    --start-station 042 --end-station 113 \
    --datetime "2026-05-12 08:30:00"
```

Linting / formatting (after installing `requirements-dev.txt`):

```bash
ruff check scripts/
ruff format scripts/
```

## How it works

The pipeline is a four-stage producer-consumer chain that hands off through files in `artifacts/`:

```
data/*.csv.xz
   │
   ▼  prepare_data.py
artifacts/rides.parquet           ← cleaned dataset
   │
   ├──▶ train_lightgbm.py  →  lgbm_model.txt + lgbm_encoders.joblib + lgbm_meta.json
   │
   ├──▶ train_keras.py     →  keras_model/model.keras + keras_preproc.joblib
   │
   ▼  compare.py / predict.py
```

`scripts/features.py` is the single source of truth for feature engineering. Both training scripts and `predict.py` import from it so train- and serve-time logic stay in lockstep.

Key design decisions:

* **Target**: `log1p(duration_sec)`. Stabilizes the right-skewed distribution and guarantees non-negative predictions after `expm1`.
* **Chronological split**: oldest rows → train, newest → test. Random splitting would leak future-into-past patterns.
* **Smoothed target encoders** fit on the training split only. The dominant signal is the `(start_station, end_station)` pair encoder — essentially "how long do trips on this route usually take?" Single-station and station × time-bucket encoders fall back for sparse / unseen routes.
* **No leakage at scoring time**: arrival time and ride duration never enter the feature columns; only the destination station, which is assumed known when the rider plans a trip.

An earlier framing used only ride-start features (no destination). It hit a hard ceiling at **R² ≈ 0.09** because ~90% of duration variance is destination-driven. Adding the destination as an input lifted R² 7× to 0.63 — see `CLAUDE.md` for more on this pivot.

## Repository layout

```
.
├── data/                  # YYYY-MM.csv.xz files (raw Ecobici data)
├── scripts/
│   ├── prepare_data.py    # data/*.csv.xz  → artifacts/rides.parquet
│   ├── features.py        # shared feature engineering + encoders + split
│   ├── train_lightgbm.py  # LightGBM trainer
│   ├── train_keras.py     # Keras MLP trainer
│   ├── compare.py         # side-by-side test metrics
│   └── predict.py         # single-row inference CLI
├── artifacts/             # generated by the pipeline (gitignored in practice)
├── requirements.txt       # runtime deps
├── requirements-dev.txt   # ruff (dev only)
├── pyproject.toml         # ruff config
├── setup_env.sh           # venv bootstrap with TF-compatible Python pick
├── CLAUDE.md              # guidance for AI coding agents
└── README.md              # this file
```

## Caveats and known limits

* **Predictions are conditional on the route being typical.** The model averages historical durations for each (start, end) pair; one-off events (closures, weather, accidents) are not modeled.
* **Sparse pairs fall back to a global prior.** For routes with very few historical rides, expect predictions close to the network-wide mean duration.
* **Demographics contribute very little.** `gender` and `age` show up in the feature-importance table but their combined gain is < 2 % of `pair_te_mean`. They are kept because they cost nothing and slightly help for atypical riders.
* **Time drift.** The model was trained on data through April 2026. If Ecobici adds or relocates stations afterwards, predictions for the affected routes will degrade and the model should be retrained.
