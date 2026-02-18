"""
tune_xgboost.py
===============
Bayesian hyperparameter optimization for XGBoost return predictor
using Optuna with time-aware walk-forward cross-validation.

Usage:
    python scripts/tune_xgboost.py              # 150 trials (default)
    python scripts/tune_xgboost.py --trials 200 # custom trial count

Outputs:
    models/saved_models/xgboost_return/          - retrained with best params
    reports/metrics/tuning_results.json           - full tuning log
"""

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ---- project imports ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.return_predictor import (
    prepare_features,
    get_feature_columns,
    walk_forward_split,
    ReturnPredictor,
    TARGET,
    MIN_TRAIN_DAYS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_2025-02-13_to_2026-02-13.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models" / "xgboost_return"
REPORT_DIR = PROJECT_ROOT / "reports" / "metrics"

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

_CACHED_DATA = {}


def get_prepared_data():
    """Load and prepare features once, then cache."""
    if "df" not in _CACHED_DATA:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        df = prepare_features(df)
        df = df.dropna(subset=[TARGET])
        feature_cols = get_feature_columns()
        df = df.dropna(subset=feature_cols)
        _CACHED_DATA["df"] = df
        _CACHED_DATA["feature_cols"] = feature_cols
    return _CACHED_DATA["df"], _CACHED_DATA["feature_cols"]


# ---------------------------------------------------------------------------
# Fast Walk-Forward Evaluation (sampled steps for speed)
# ---------------------------------------------------------------------------

def full_walk_forward_eval(params, df, feature_cols):
    """Evaluate params with FULL walk-forward validation.

    Trains a new model at every single time step for maximum accuracy
    in evaluating each hyperparameter configuration.

    Parameters
    ----------
    params : dict
        XGBoost hyperparameters.
    df : pd.DataFrame
        Prepared feature DataFrame.
    feature_cols : list[str]
        Feature column names.

    Returns
    -------
    dict with accuracy, roc_auc, f1
    """
    splits = walk_forward_split(df, min_train=MIN_TRAIN_DAYS)

    all_preds = []
    all_probs = []
    all_true = []

    for train_idx, test_idx in splits:
        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, TARGET]
        X_test = df.loc[test_idx, feature_cols]
        y_test = df.loc[test_idx, TARGET]

        model = XGBClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_true.extend(y_test.values)

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return {
        "accuracy": float(accuracy_score(all_true, all_preds)),
        "roc_auc": float(roc_auc_score(all_true, all_probs)),
        "f1": float(f1_score(all_true, all_preds, zero_division=0)),
        "n_eval_steps": len(splits),
    }


# ---------------------------------------------------------------------------
# Optuna Objective
# ---------------------------------------------------------------------------

def create_objective(df, feature_cols):
    """Create an Optuna objective function (closure over data)."""

    def objective(trial):
        params = {
            # Core tree params
            "n_estimators": trial.suggest_int("n_estimators", 50, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),

            # Sampling
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),

            # Regularization
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),

            # Class balance
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),

            # Fixed
            "random_state": 42,
            "eval_metric": "logloss",
        }

        result = full_walk_forward_eval(params, df, feature_cols)

        # Report intermediate values for pruning
        trial.set_user_attr("accuracy", result["accuracy"])
        trial.set_user_attr("f1", result["f1"])

        # Optimize ROC AUC (better than accuracy for imbalanced binary tasks)
        return result["roc_auc"]

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument("--trials", type=int, default=150,
                        help="Number of Optuna trials (default: 150)")
    args = parser.parse_args()
    n_trials = args.trials

    print("=" * 60)
    print("  XGBoost Hyperparameter Tuning (Optuna)")
    print("=" * 60)

    # --- Load baseline metrics ---
    baseline_meta = MODEL_DIR / "meta.json"
    baseline_metrics = {}
    if baseline_meta.exists():
        with open(baseline_meta) as f:
            meta = json.load(f)
        baseline_metrics = meta.get("metrics", {})
        print(f"\n  Baseline performance:")
        print(f"    Accuracy : {baseline_metrics.get('accuracy', 0):.3f}")
        print(f"    ROC AUC  : {baseline_metrics.get('roc_auc', 0):.3f}")
        print(f"    F1       : {baseline_metrics.get('f1', 0):.3f}")
        print(f"    Params   : {json.dumps(meta.get('model_params', {}), indent=None)}")
    else:
        print("\n  No baseline model found. Will train from scratch.")

    # --- Load data ---
    print(f"\n  Loading data from {DATA_PATH.name} ...")
    df, feature_cols = get_prepared_data()
    print(f"  {len(df)} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].nunique()} dates")

    # --- Run Optuna ---
    print(f"\n  Starting optimization ({n_trials} trials) ...")
    print("  Objective: maximize walk-forward ROC AUC")
    print("  Using FULL walk-forward CV (every step, no shortcuts)\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_return_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t0 = time.time()
    study.optimize(
        create_objective(df, feature_cols),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[_progress_callback],
    )
    elapsed = time.time() - t0

    # --- Results ---
    best = study.best_trial
    print(f"\n{'=' * 60}")
    print(f"  Optimization complete in {elapsed / 60:.1f} minutes")
    print(f"  Best trial #{best.number}")
    print(f"  Best ROC AUC: {best.value:.4f}")
    print(f"  Best accuracy: {best.user_attrs.get('accuracy', 0):.4f}")
    print(f"  Best F1: {best.user_attrs.get('f1', 0):.4f}")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:25s}: {v}")
    print(f"{'=' * 60}")

    # --- Comparison with baseline ---
    if baseline_metrics:
        print(f"\n  Comparison with baseline (full walk-forward):")
        for metric in ["accuracy", "roc_auc", "f1"]:
            old = baseline_metrics.get(metric, 0)
            new = best.user_attrs.get(metric, 0) if metric != "roc_auc" else best.value
            delta = new - old
            sign = "+" if delta >= 0 else ""
            print(f"    {metric:10s}: {old:.4f} -> {new:.4f} ({sign}{delta:.4f})")

    # --- Retrain final model with best params ---
    print("\n  Retraining final model with best hyperparameters ...")

    predictor = ReturnPredictor(
        n_estimators=best.params["n_estimators"],
        max_depth=best.params["max_depth"],
        learning_rate=best.params["learning_rate"],
        min_child_weight=best.params["min_child_weight"],
        gamma=best.params["gamma"],
        subsample=best.params["subsample"],
        colsample_bytree=best.params["colsample_bytree"],
        colsample_bylevel=best.params["colsample_bylevel"],
        reg_alpha=best.params["reg_alpha"],
        reg_lambda=best.params["reg_lambda"],
        scale_pos_weight=best.params["scale_pos_weight"],
    )

    metrics = predictor.train(
        data_path=DATA_PATH,
        min_train_days=MIN_TRAIN_DAYS,
        verbose=True,
    )

    # --- Save ---
    predictor.save(MODEL_DIR)
    print()
    print(predictor.summary())

    # --- Save tuning report ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n_trials": n_trials,
        "elapsed_minutes": round(elapsed / 60, 1),
        "best_trial": best.number,
        "best_params": best.params,
        "best_roc_auc": best.value,
        "best_accuracy": best.user_attrs.get("accuracy", 0),
        "best_f1": best.user_attrs.get("f1", 0),
        "baseline_metrics": baseline_metrics,
        "improvement": {
            metric: round(
                (best.user_attrs.get(metric, 0) if metric != "roc_auc" else best.value)
                - baseline_metrics.get(metric, 0), 4
            )
            for metric in ["accuracy", "roc_auc", "f1"]
        },
        "top_10_trials": [
            {
                "number": t.number,
                "roc_auc": t.value,
                "accuracy": t.user_attrs.get("accuracy", 0),
                "f1": t.user_attrs.get("f1", 0),
                "params": t.params,
            }
            for t in sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:10]
        ],
    }

    report_path = REPORT_DIR / "tuning_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[SAVED] Tuning report -> {report_path}")

    # Save metrics
    metrics_path = REPORT_DIR / "xgboost_walkforward_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[SAVED] Updated metrics -> {metrics_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress_callback(study, trial):
    """Print progress every 10 trials."""
    n = trial.number + 1
    if n % 10 == 0 or n == 1:
        best = study.best_trial
        val = trial.value if trial.value is not None else 0.0
        print(f"  Trial {n:3d} | "
              f"Current AUC: {val:.4f} | "
              f"Best AUC: {best.value:.4f} (trial #{best.number})")


if __name__ == "__main__":
    main()
