"""Print side-by-side test metrics for both trained models.

Run AFTER both training scripts have completed::

    python scripts/compare.py

If only one of the models has been trained, the other column displays
``n/a`` so partial runs are still readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

ART = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    """Load both metric files (if present) and print a comparison table."""
    lgbm_meta_path = ART / "lgbm_meta.json"
    keras_prep_path = ART / "keras_preproc.joblib"

    lgbm = json.loads(lgbm_meta_path.read_text())["metrics"] if lgbm_meta_path.exists() else None
    keras = joblib.load(keras_prep_path)["metrics"] if keras_prep_path.exists() else None

    if lgbm is None and keras is None:
        raise SystemExit(
            "No trained models found. Run train_lightgbm.py and/or train_keras.py first."
        )

    keys = ("mae_sec", "rmse_sec", "r2_log", "mape_pct")
    print(f"{'metric':<10} {'LightGBM':>14} {'Keras MLP':>14}")
    print("-" * 42)
    for k in keys:
        lv = f"{lgbm[k]:.4f}" if lgbm else "    n/a"
        kv = f"{keras[k]:.4f}" if keras else "    n/a"
        print(f"{k:<10} {lv:>14} {kv:>14}")


if __name__ == "__main__":
    main()
