"""
train_multihorizon.py
=====================
Train XGBoost return-direction predictors for three horizons:
  - 1d  (next trading day)
  - 5d  (next 5 trading days / weekly)
  - 20d (next 20 trading days / monthly)

Each model is saved to its own directory under models/saved_models/.

Usage:
    python scripts/train_multihorizon.py
    python scripts/train_multihorizon.py --horizons 1d 5d
    python scripts/train_multihorizon.py --min-train 90

Outputs:
    models/saved_models/xgboost_return_1d/   (model.json, meta.json, walk_forward_results.csv)
    models/saved_models/xgboost_return_5d/
    models/saved_models/xgboost_return_20d/
    reports/metrics/multihorizon_summary.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.return_predictor import ReturnPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "processed"
# Auto-detect merged CSV (pick the latest one)
_merged_candidates = sorted(DATA_DIR.glob("merged_*.csv"))
DATA_PATH = _merged_candidates[-1] if _merged_candidates else DATA_DIR / "merged.csv"

# Model output dirs
MODEL_BASE = PROJECT_ROOT / "models" / "saved_models"
REPORT_DIR = PROJECT_ROOT / "reports" / "metrics"

# Per-horizon hyperparameters
# 1d: use Optuna-tuned params from previous tuning run
# 5d/20d: start with sensible defaults (slightly different regularization
#          because longer horizons have smoother targets)
HORIZON_PARAMS = {
    "1d": {
        "n_estimators": 473,
        "max_depth": 5,
        "learning_rate": 0.213,
        "subsample": 0.617,
        "colsample_bytree": 0.863,
        "gamma": 3.398,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.924,
        "min_child_weight": 1,
    },
    "5d": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "gamma": 1.0,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "scale_pos_weight": 1.0,
        "min_child_weight": 3,
    },
    "20d": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "gamma": 2.0,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "scale_pos_weight": 1.0,
        "min_child_weight": 5,
    },
    "60d": {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "gamma": 3.0,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": 1.0,
        "min_child_weight": 7,
    },
}

MIN_TRAIN_DAYS = 60
HORIZONS = ["1d", "5d", "20d", "60d"]
HORIZON_LABELS = {
    "1d": "Daily (next day)",
    "5d": "Weekly (5-day)",
    "20d": "Monthly (20-day)",
    "60d": "Quarterly (60-day)",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train multi-horizon XGBoost models")
    parser.add_argument(
        "--horizons", nargs="+", default=HORIZONS,
        choices=HORIZONS, help="Horizons to train (default: all three)"
    )
    parser.add_argument(
        "--min-train", type=int, default=MIN_TRAIN_DAYS,
        help="Minimum training days for walk-forward (default: 60)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Multi-Horizon XGBoost Training")
    print("  Horizons: " + ", ".join(args.horizons))
    print("=" * 65)
    print()

    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("  Run the data pipeline first (load_merged_dataset).")
        return

    all_results = {}
    total_start = time.time()

    for hz in args.horizons:
        print()
        print("-" * 65)
        print(f"  Training {HORIZON_LABELS[hz]} model (horizon={hz})")
        print("-" * 65)

        params = HORIZON_PARAMS[hz]
        model_dir = MODEL_BASE / f"xgboost_return_{hz}"

        predictor = ReturnPredictor(horizon=hz, **params)

        hz_start = time.time()
        metrics = predictor.train(
            data_path=DATA_PATH,
            min_train_days=args.min_train,
            verbose=True,
        )
        hz_elapsed = time.time() - hz_start

        # Save model
        predictor.save(model_dir)

        # Print summary
        print()
        print(predictor.summary())

        all_results[hz] = {
            "label": HORIZON_LABELS[hz],
            "model_dir": str(model_dir),
            "elapsed_seconds": round(hz_elapsed, 1),
            "metrics": metrics,
        }

    total_elapsed = time.time() - total_start

    # --- Comparison summary ---
    print()
    print("=" * 65)
    print("  MULTI-HORIZON COMPARISON")
    print("=" * 65)
    print()
    print(f"  {'Horizon':<12} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'Time':>8}")
    print("  " + "-" * 62)

    for hz in args.horizons:
        m = all_results[hz]["metrics"]
        t = all_results[hz]["elapsed_seconds"]
        print(
            f"  {HORIZON_LABELS[hz]:<12} "
            f"{m['accuracy']:>9.1%} "
            f"{m['f1']:>9.1%} "
            f"{m['roc_auc']:>9.1%} "
            f"{m['precision']:>9.1%} "
            f"{m['recall']:>9.1%} "
            f"{t:>7.0f}s"
        )

    print()
    print(f"  Total training time: {total_elapsed / 60:.1f} min")

    # --- Save combined report ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "horizons_trained": args.horizons,
        "total_elapsed_minutes": round(total_elapsed / 60, 1),
        "data_path": str(DATA_PATH),
        "min_train_days": args.min_train,
        "results": {},
    }
    for hz in args.horizons:
        r = all_results[hz]
        report["results"][hz] = {
            "label": r["label"],
            "elapsed_seconds": r["elapsed_seconds"],
            "accuracy": r["metrics"]["accuracy"],
            "f1": r["metrics"]["f1"],
            "roc_auc": r["metrics"]["roc_auc"],
            "precision": r["metrics"]["precision"],
            "recall": r["metrics"]["recall"],
            "n_predictions": r["metrics"]["n_predictions"],
            "top_5_features": dict(
                list(
                    sorted(
                        r["metrics"].get("per_ticker", {}).items(),
                        key=lambda x: x[1].get("accuracy", 0),
                        reverse=True,
                    )[:5]
                )
            ) if r["metrics"].get("per_ticker") else {},
            "hyperparameters": HORIZON_PARAMS[hz],
        }

    report_path = REPORT_DIR / "multihorizon_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[SAVED] Summary report -> {report_path}")
    print()
    print("Models saved to:")
    for hz in args.horizons:
        print(f"  {hz}: {MODEL_BASE / f'xgboost_return_{hz}'}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
