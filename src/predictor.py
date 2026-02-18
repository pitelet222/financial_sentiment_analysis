"""
Predictor — One-Call Sentiment Inference Facade
=================================================

Provides a single high-level function ``get_sentiment(ticker)`` that:

1. Loads (or returns cached) news for the ticker — either from the
   project's raw CSVs or from live RSS feeds.
2. Loads the fine-tuned FinBERT model (singleton, kept in memory).
3. Runs batch inference on the headlines.
4. Returns a clean DataFrame and a summary dict.

This is the **entry point that the dashboard and any external consumer
should call**.  It hides all wiring between ``data_loader``,
``news_fetcher``, and ``SentimentAnalyzer``.

Typical usage
-------------
>>> from src.predictor import get_sentiment
>>> result = get_sentiment("AAPL")
>>> result["signal"]          # "BULLISH", "BEARISH", or "NEUTRAL"
>>> result["confidence"]      # 0–100 %
>>> result["articles"]        # DataFrame with per-headline scores
>>> result["summary_stats"]   # dict with aggregate numbers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =========================================================================
# Paths
# =========================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FINETUNED_MODEL = _PROJECT_ROOT / "models" / "saved_models" / "finbert_finetuned"
_DEFAULT_MODEL = "ProsusAI/finbert"  # fallback if no fine-tuned checkpoint

# =========================================================================
# Singleton model holder (avoids reloading ~110 M params on every call)
# =========================================================================

_analyzer_instance = None


def _get_analyzer():
    """Return (and lazily create) a singleton SentimentAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        from src.models.sentiment_analyzer import SentimentAnalyzer

        if _FINETUNED_MODEL.exists():
            logger.info("Loading fine-tuned model from %s", _FINETUNED_MODEL)
            _analyzer_instance = SentimentAnalyzer.load(_FINETUNED_MODEL)
        else:
            logger.info("No fine-tuned checkpoint found — using %s", _DEFAULT_MODEL)
            _analyzer_instance = SentimentAnalyzer(model_name=_DEFAULT_MODEL)
    return _analyzer_instance


# =========================================================================
# Internal helpers
# =========================================================================

def _load_cached_news(ticker: str) -> Optional[pd.DataFrame]:
    """Try to load raw news CSVs that already exist on disk."""
    from src.data.data_loader import RAW_DATA_DIR, DEFAULT_START, DEFAULT_END

    fp = RAW_DATA_DIR / f"news_{ticker}_{DEFAULT_START}_to_{DEFAULT_END}.csv"
    if fp.exists():
        from src.data.data_loader import load_news_csv
        return load_news_csv(fp)
    return None


def _load_live_news(ticker: str) -> pd.DataFrame:
    """Fetch live news via RSS with disk caching."""
    from src.data.news_fetcher import fetch_and_cache
    return fetch_and_cache(ticker)


def _compute_signal(avg_score: float) -> tuple[str, float]:
    """Derive a signal label and confidence % from an average score."""
    if avg_score > 0.15:
        label = "BULLISH"
    elif avg_score < -0.15:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    confidence = min(abs(avg_score) / 0.5 * 100, 100.0)
    return label, round(confidence, 1)


# =========================================================================
# Public API
# =========================================================================

def get_sentiment(
    ticker: str,
    *,
    source: str = "auto",
    text_columns: Optional[List[str]] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Run end-to-end sentiment analysis for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    source : ``"auto"`` | ``"cached"`` | ``"live"``
        Where to get news articles:

        - ``"auto"`` (default) — use raw CSVs if available, otherwise
          fetch live RSS.
        - ``"cached"`` — only use existing raw CSVs (raises if missing).
        - ``"live"`` — always fetch fresh RSS headlines.
    text_columns : list of str, optional
        Columns to run inference on (default ``["title", "summary"]``).
    batch_size : int
        Batch size for the model (default 16).

    Returns
    -------
    dict
        ``signal``        — ``"BULLISH"`` / ``"BEARISH"`` / ``"NEUTRAL"``
        ``confidence``    — 0–100 %
        ``avg_score``     — mean continuous score (−1 to +1)
        ``n_articles``    — number of articles analysed
        ``articles``      — ``pd.DataFrame`` with per-headline predictions
        ``summary_stats`` — dict with distribution breakdown

    Examples
    --------
    >>> r = get_sentiment("AAPL")
    >>> print(r["signal"], r["confidence"])
    BULLISH 72.0
    """
    ticker = ticker.upper()
    if text_columns is None:
        text_columns = ["title", "summary"]

    # --- 1. Get news ---
    news: Optional[pd.DataFrame] = None

    if source in ("auto", "cached"):
        news = _load_cached_news(ticker)

    if news is None and source in ("auto", "live"):
        news = _load_live_news(ticker)

    if news is None or news.empty:
        return {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "avg_score": 0.0,
            "n_articles": 0,
            "articles": pd.DataFrame(),
            "summary_stats": {},
        }

    # --- 2. Run model ---
    analyzer = _get_analyzer()

    # Only predict on columns that exist
    cols_to_predict = [c for c in text_columns if c in news.columns]
    if not cols_to_predict:
        cols_to_predict = ["title"]  # fallback

    scored = analyzer.predict_dataframe(
        news, text_columns=cols_to_predict, batch_size=batch_size
    )

    # --- 3. Pick a primary score column ---
    primary_col = f"finbert_{cols_to_predict[0]}_score"
    scores = scored[primary_col].dropna()

    avg_score = float(scores.mean()) if len(scores) else 0.0
    signal, confidence = _compute_signal(avg_score)

    # --- 4. Summary stats ---
    labels_col = f"finbert_{cols_to_predict[0]}_label"
    label_counts = scored[labels_col].value_counts().to_dict() if labels_col in scored.columns else {}

    summary_stats = {
        "avg_score": round(avg_score, 4),
        "median_score": round(float(scores.median()), 4) if len(scores) else 0.0,
        "std_score": round(float(scores.std()), 4) if len(scores) > 1 else 0.0,
        "min_score": round(float(scores.min()), 4) if len(scores) else 0.0,
        "max_score": round(float(scores.max()), 4) if len(scores) else 0.0,
        "pct_positive": round((scores > 0.1).mean() * 100, 1) if len(scores) else 0.0,
        "pct_negative": round((scores < -0.1).mean() * 100, 1) if len(scores) else 0.0,
        "label_counts": label_counts,
    }

    return {
        "signal": signal,
        "confidence": confidence,
        "avg_score": round(avg_score, 4),
        "n_articles": len(scored),
        "articles": scored,
        "summary_stats": summary_stats,
    }


# =========================================================================
# CLI smoke test
# =========================================================================

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Running sentiment analysis for {ticker} …\n")
    result = get_sentiment(ticker)
    print(f"Signal     : {result['signal']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Avg score  : {result['avg_score']}")
    print(f"Articles   : {result['n_articles']}")
    print(f"Stats      : {result['summary_stats']}")
    if not result["articles"].empty:
        print(f"\nSample headlines:")
        cols = ["title", f"finbert_title_score", f"finbert_title_label"]
        avail = [c for c in cols if c in result["articles"].columns]
        print(result["articles"][avail].head(5).to_string(index=False))
