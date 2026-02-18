"""
train_xgboost.py
================
Train the XGBoost return-direction predictor and save it.

Usage:
    python scripts/train_xgboost.py

Outputs:
    models/saved_models/xgboost_return/
        model.json                  - XGBoost model
        meta.json                   - metrics + feature importance
        walk_forward_results.csv    - per-day OOS predictions
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.return_predictor import ReturnPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_2025-02-13_to_2026-02-13.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models" / "xgboost_return"
MIN_TRAIN_DAYS = 60


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  XGBoost Return Direction Predictor - Training")
    print("=" * 60)
    print()

    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("  Run the data pipeline first:")
        print("    python scripts/download_prices.py")
        print("    python scripts/download_news_extended.py")
        print("  Then merge with load_merged_dataset(save_processed=True)")
        return

    # Train
    predictor = ReturnPredictor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    metrics = predictor.train(
        data_path=DATA_PATH,
        min_train_days=MIN_TRAIN_DAYS,
        verbose=True,
    )

    # Save
    predictor.save(MODEL_DIR)

    # Print summary
    print()
    print(predictor.summary())

    # Demo: predict next day for each ticker
    print("\n--- Next-Day Predictions (demo) ---")
    import pandas as pd
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    for ticker in df["ticker"].unique():
        try:
            result = predictor.predict_next_day(df, ticker)
            prob = result["prob_up"] if result["direction"] == "UP" else result["prob_down"]
            print(f"  {ticker}: {result['direction']} "
                  f"(confidence: {result['confidence']:.1%}, "
                  f"prob: {prob:.1%}) "
                  f"[based on {result['based_on_date']}]")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    # Save walk-forward results to reports
    reports_dir = PROJECT_ROOT / "reports" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(reports_dir / "xgboost_walkforward_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[SAVED] Metrics -> {reports_dir / 'xgboost_walkforward_metrics.json'}")


if __name__ == "__main__":
    main()
