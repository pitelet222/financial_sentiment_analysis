"""
Return Direction Predictor — XGBoost + Sentiment Features
==========================================================

Predicts next-day stock return direction (up / down) by combining:
  - Sentiment features (FinBERT scores aggregated daily)
  - Price-based technical features (returns, volatility, gaps, volume)

Architecture
------------
- **Model**: XGBoost binary classifier (return_direction: 1=up, 0=down)
- **Validation**: Walk-forward (expanding window) — trains on all data up
  to day *t*, predicts day *t+1*. No lookahead bias.
- **Features**: Lagged so the model only sees data available *before*
  market open on the prediction day.

Usage
-----
>>> from src.models.return_predictor import ReturnPredictor
>>> predictor = ReturnPredictor()
>>> predictor.train("data/processed/merged_2025-02-13_to_2026-02-13.csv")
>>> predictor.save("models/saved_models/xgboost_return")
>>> predictor = ReturnPredictor.load("models/saved_models/xgboost_return")
>>> pred = predictor.predict_latest(ticker_df)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Features used for prediction (all lagged by 1 day to avoid lookahead)
SENTIMENT_FEATURES = [
    "avg_overall_sentiment",
    "avg_ticker_sentiment",
    "sentiment_std",
    "sentiment_range",
    "pct_positive",
    "pct_negative",
    "article_count",
    "sentiment_rolling_3d",
    "sentiment_rolling_5d",
]

PRICE_FEATURES = [
    "daily_return",
    "intraday_range",
    "gap_pct",
    "volume_change",
]

# Additional engineered features (computed in prepare_features)
ENGINEERED_FEATURES = [
    "return_lag2",
    "return_lag3",
    "volatility_5d",
    "avg_return_5d",
    "sentiment_momentum",  # 3d rolling - 5d rolling
    "news_has_coverage",   # binary: any articles?
]

TARGET = "return_direction"  # 1 = up, 0 = down

# Minimum training window for walk-forward validation
MIN_TRAIN_DAYS = 60


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix from the merged dataset.

    All features are *lagged* so that row t uses information available
    before market open on day t (i.e., yesterday's values).

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset from ``load_merged_dataset()`` — one row per
        ticker per trading day.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with columns ready for XGBoost.
        Rows with insufficient history (first few days) are dropped.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- Fill NaN sentiment with 0 (no news = neutral) ---
    for col in SENTIMENT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- Lag all raw features by 1 day (use yesterday's data) ---
    lag_cols = SENTIMENT_FEATURES + PRICE_FEATURES
    for col in lag_cols:
        if col in df.columns:
            df[col] = df.groupby("ticker")[col].shift(1)

    # --- Engineered features (already lagged via base features) ---
    # Additional return lags
    df["return_lag2"] = df.groupby("ticker")["daily_return"].shift(1)
    df["return_lag3"] = df.groupby("ticker")["daily_return"].shift(2)

    # 5-day volatility and average return
    df["volatility_5d"] = (
        df.groupby("ticker")["daily_return"]
        .transform(lambda s: s.rolling(5, min_periods=3).std())
    )
    df["avg_return_5d"] = (
        df.groupby("ticker")["daily_return"]
        .transform(lambda s: s.rolling(5, min_periods=3).mean())
    )

    # Sentiment momentum: short vs long rolling
    df["sentiment_momentum"] = (
        df["sentiment_rolling_3d"] - df["sentiment_rolling_5d"]
    )

    # Binary: any news coverage?
    df["news_has_coverage"] = (df["article_count"] > 0).astype(int)

    return df


def get_feature_columns() -> List[str]:
    """Return the ordered list of feature column names."""
    return SENTIMENT_FEATURES + PRICE_FEATURES + ENGINEERED_FEATURES


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_split(
    df: pd.DataFrame,
    min_train: int = MIN_TRAIN_DAYS,
) -> List[Tuple[pd.Index, pd.Index]]:
    """Generate walk-forward (expanding window) train/test splits.

    For each ticker, the procedure is:
        - First ``min_train`` days -> training only
        - Day min_train+1 -> test (trained on 1..min_train)
        - Day min_train+2 -> test (trained on 1..min_train+1)
        - ...and so on

    We pool all tickers together for each time step: at each step,
    every ticker up to that date is in train, the next date is test.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-prepared dataset sorted by (ticker, date).
    min_train : int
        Minimum number of trading days per ticker before first prediction.

    Returns
    -------
    list of (train_idx, test_idx) tuples
    """
    splits = []
    dates = sorted(df["date"].unique())

    if len(dates) <= min_train:
        raise ValueError(
            f"Need > {min_train} unique dates, got {len(dates)}"
        )

    for i in range(min_train, len(dates)):
        train_dates = dates[:i]
        test_date = dates[i]

        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"] == test_date

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ---------------------------------------------------------------------------
# Predictor Class
# ---------------------------------------------------------------------------

class ReturnPredictor:
    """XGBoost-based next-day return direction predictor.

    Attributes
    ----------
    model : XGBClassifier | None
        Trained XGBoost model.
    feature_cols : list[str]
        Ordered feature column names.
    metrics : dict
        Walk-forward validation metrics.
    feature_importance : dict
        Feature importance scores (gain-based).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        colsample_bylevel: float = 1.0,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
    ):
        self.model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "eval_metric": "logloss",
        }
        self.model: Optional[XGBClassifier] = None
        self.feature_cols = get_feature_columns()
        self.metrics: Dict = {}
        self.feature_importance: Dict = {}
        self._is_trained = False

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train(
        self,
        data_path: str | Path,
        min_train_days: int = MIN_TRAIN_DAYS,
        verbose: bool = True,
    ) -> Dict:
        """Train with walk-forward validation, then fit final model on all data.

        Parameters
        ----------
        data_path : str or Path
            Path to the merged CSV (output of ``load_merged_dataset``).
        min_train_days : int
            Minimum training window before first prediction.
        verbose : bool
            Print progress updates.

        Returns
        -------
        dict
            Validation metrics dictionary.
        """
        # --- Load & prepare ---
        if verbose:
            print("[1/4] Loading and preparing features ...")
        df = pd.read_csv(data_path, parse_dates=["date"])
        df = prepare_features(df)

        # Drop rows with NaN target or features
        df = df.dropna(subset=[TARGET])
        feature_df = df[self.feature_cols + [TARGET, "date", "ticker"]].copy()
        feature_df = feature_df.dropna(subset=self.feature_cols)

        if verbose:
            print(f"       {len(feature_df)} usable rows "
                  f"({feature_df['ticker'].nunique()} tickers, "
                  f"{feature_df['date'].nunique()} dates)")

        # --- Walk-forward validation ---
        if verbose:
            print("[2/4] Walk-forward validation ...")

        splits = walk_forward_split(feature_df, min_train=min_train_days)
        all_preds = []
        all_probs = []
        all_true = []
        all_dates = []
        all_tickers = []

        for i, (train_idx, test_idx) in enumerate(splits):
            X_train = feature_df.loc[train_idx, self.feature_cols]
            y_train = feature_df.loc[train_idx, TARGET]
            X_test = feature_df.loc[test_idx, self.feature_cols]
            y_test = feature_df.loc[test_idx, TARGET]

            model = XGBClassifier(**self.model_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train, verbose=False)

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(y_test.values)
            all_dates.extend(feature_df.loc[test_idx, "date"].values)
            all_tickers.extend(feature_df.loc[test_idx, "ticker"].values)

            if verbose and (i + 1) % 50 == 0:
                acc = accuracy_score(all_true, all_preds)
                print(f"       Step {i + 1}/{len(splits)} | "
                      f"Running accuracy: {acc:.3f}")

        # --- Compute metrics ---
        all_true = np.array(all_true)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        self.metrics = {
            "accuracy": float(accuracy_score(all_true, all_preds)),
            "precision": float(precision_score(all_true, all_preds, zero_division=0)),
            "recall": float(recall_score(all_true, all_preds, zero_division=0)),
            "f1": float(f1_score(all_true, all_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(all_true, all_probs)),
            "n_predictions": int(len(all_true)),
            "n_correct": int((all_true == all_preds).sum()),
            "class_distribution": {
                "actual_up": int(all_true.sum()),
                "actual_down": int((1 - all_true).sum()),
                "predicted_up": int(all_preds.sum()),
                "predicted_down": int((1 - all_preds).sum()),
            },
            "min_train_days": min_train_days,
        }

        # Per-ticker metrics
        ticker_metrics = {}
        for tkr in feature_df["ticker"].unique():
            mask = np.array(all_tickers) == tkr
            if mask.sum() > 0:
                ticker_metrics[tkr] = {
                    "accuracy": float(accuracy_score(
                        all_true[mask], all_preds[mask]
                    )),
                    "f1": float(f1_score(
                        all_true[mask], all_preds[mask], zero_division=0
                    )),
                    "n_predictions": int(mask.sum()),
                }
        self.metrics["per_ticker"] = ticker_metrics

        if verbose:
            print(f"\n       Walk-Forward Results ({len(splits)} steps):")
            print(f"       Accuracy : {self.metrics['accuracy']:.3f}")
            print(f"       F1       : {self.metrics['f1']:.3f}")
            print(f"       ROC AUC  : {self.metrics['roc_auc']:.3f}")
            print(f"       Precision: {self.metrics['precision']:.3f}")
            print(f"       Recall   : {self.metrics['recall']:.3f}")

        # --- Store walk-forward predictions ---
        self._wf_results = pd.DataFrame({
            "date": all_dates,
            "ticker": all_tickers,
            "actual": all_true,
            "predicted": all_preds,
            "prob_up": all_probs,
        })

        # --- Final model: train on all data ---
        if verbose:
            print("\n[3/4] Training final model on all data ...")

        X_all = feature_df[self.feature_cols]
        y_all = feature_df[TARGET]

        self.model = XGBClassifier(**self.model_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_all, y_all, verbose=False)

        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance = dict(
            sorted(
                zip(self.feature_cols, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        if verbose:
            print("\n[4/4] Top 10 features by importance:")
            for fname, imp in list(self.feature_importance.items())[:10]:
                bar = "#" * int(imp * 50)
                print(f"       {fname:30s} {imp:.4f} {bar}")

        self._is_trained = True
        return self.metrics

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict return direction for prepared feature rows.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with columns matching ``self.feature_cols``.

        Returns
        -------
        predictions : np.ndarray
            Binary predictions (1=up, 0=down).
        probabilities : np.ndarray
            Probability of class 1 (up).
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_aligned = X[self.feature_cols].copy()
        # Fill any remaining NaN with 0 (no news = neutral)
        X_aligned = X_aligned.fillna(0)

        preds = self.model.predict(X_aligned)
        probs = self.model.predict_proba(X_aligned)[:, 1]
        return preds, probs

    def predict_next_day(
        self, df: pd.DataFrame, ticker: str
    ) -> Dict:
        """Predict the next trading day's direction for a single ticker.

        Uses the latest row of prepared features to make a prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Merged dataset (will be run through ``prepare_features``).
        ticker : str
            Ticker symbol to predict.

        Returns
        -------
        dict with keys: ticker, direction, probability, confidence,
             last_date, features_used
        """
        prepared = prepare_features(df)
        ticker_df = prepared[prepared["ticker"] == ticker].copy()

        if ticker_df.empty:
            raise ValueError(f"No data for ticker {ticker}")

        # Take the last row (most recent available data)
        last_row = ticker_df.iloc[[-1]]  # keep as DataFrame
        last_date = last_row["date"].iloc[0]

        # Fill NaN features
        X = last_row[self.feature_cols].fillna(0)

        preds, probs = self.predict(X)
        prob_up = float(probs[0])
        direction = "UP" if preds[0] == 1 else "DOWN"
        confidence = abs(prob_up - 0.5) * 2  # 0..1 scale

        return {
            "ticker": ticker,
            "direction": direction,
            "prediction": int(preds[0]),
            "prob_up": round(prob_up, 4),
            "prob_down": round(1 - prob_up, 4),
            "confidence": round(confidence, 4),
            "based_on_date": str(last_date)[:10],
            "features_used": {
                col: round(float(X[col].iloc[0]), 4)
                for col in self.feature_cols
            },
        }

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save model, metrics, and feature importance to disk.

        Creates:
        - ``model.json`` — XGBoost model
        - ``meta.json`` — metrics, feature importance, parameters
        - ``walk_forward_results.csv`` — per-day predictions from validation
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No trained model to save.")

        # Model
        self.model.save_model(str(directory / "model.json"))

        # Metadata
        meta = {
            "feature_cols": self.feature_cols,
            "model_params": self.model_params,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Walk-forward results
        if hasattr(self, "_wf_results"):
            self._wf_results.to_csv(
                directory / "walk_forward_results.csv", index=False
            )

        print(f"[SAVED] Model -> {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "ReturnPredictor":
        """Load a saved predictor from disk."""
        directory = Path(directory)

        # Load metadata
        with open(directory / "meta.json") as f:
            meta = json.load(f)

        # Reconstruct — filter to only __init__ params
        import inspect
        init_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        saved_params = meta.get("model_params", {})
        valid_params = {k: v for k, v in saved_params.items() if k in init_params}
        predictor = cls(**valid_params)
        predictor.feature_cols = meta.get("feature_cols", get_feature_columns())
        predictor.metrics = meta.get("metrics", {})
        predictor.feature_importance = meta.get("feature_importance", {})

        # Load model
        predictor.model = XGBClassifier()
        predictor.model.load_model(str(directory / "model.json"))
        predictor._is_trained = True

        # Load walk-forward results if available
        wf_path = directory / "walk_forward_results.csv"
        if wf_path.exists():
            predictor._wf_results = pd.read_csv(wf_path, parse_dates=["date"])

        print(f"[LOADED] Model <- {directory}")
        return predictor

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of the model's performance."""
        if not self.metrics:
            return "No metrics available. Train the model first."

        lines = [
            "=" * 55,
            "  XGBoost Return Direction Predictor — Summary",
            "=" * 55,
            f"  Walk-Forward Predictions: {self.metrics.get('n_predictions', '?')}",
            f"  Accuracy  : {self.metrics.get('accuracy', 0):.1%}",
            f"  F1 Score  : {self.metrics.get('f1', 0):.1%}",
            f"  ROC AUC   : {self.metrics.get('roc_auc', 0):.1%}",
            f"  Precision : {self.metrics.get('precision', 0):.1%}",
            f"  Recall    : {self.metrics.get('recall', 0):.1%}",
            "",
        ]

        # Per-ticker breakdown
        per_ticker = self.metrics.get("per_ticker", {})
        if per_ticker:
            lines.append("  Per-Ticker Accuracy:")
            for tkr, tm in per_ticker.items():
                lines.append(
                    f"    {tkr}: {tm['accuracy']:.1%} "
                    f"(F1={tm['f1']:.1%}, n={tm['n_predictions']})"
                )
            lines.append("")

        # Top features
        if self.feature_importance:
            lines.append("  Top 5 Features:")
            for fname, imp in list(self.feature_importance.items())[:5]:
                lines.append(f"    {fname:30s} {imp:.4f}")

        lines.append("=" * 55)
        return "\n".join(lines)
