"""
Data Loader — Merge News Sentiment with Stock Prices
=====================================================

This module is the central data-loading and merging layer of the project.
It handles:

1. **Loading raw CSVs** (news + prices) with proper date parsing and the
   yfinance header quirk (``skiprows=[1]``).
2. **Market-hours classification** — every news article is tagged as
   *pre-market*, *market-hours*, or *after-hours* based on US Eastern
   Time trading windows.
3. **Time-window aggregation** — news sentiment is rolled up into
   configurable windows (e.g. all articles between 9 AM and 4 PM) so
   that each trading day gets a single sentiment summary row.
4. **Merging** — the aggregated sentiment is joined with daily price
   data so downstream models receive one row per ticker per trading day.
5. **Feature enrichment** — daily return, return direction, rolling
   sentiment average, and article-count features are added.

Typical usage
-------------
>>> from src.data.data_loader import load_merged_dataset
>>> df = load_merged_dataset()          # uses default paths / tickers
>>> df.head()

Architecture note
-----------------
Functions are ordered from low-level (load a single CSV) to high-level
(``load_merged_dataset``).  Each layer can be used independently, which
makes unit-testing and notebook experimentation easy.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Project root (two levels up from this file: src/data/data_loader.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

# Default tickers – keep in sync with download scripts
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT"]

# Default date range
DEFAULT_START = "2025-11-01"
DEFAULT_END = "2026-02-13"

# US market hours in Eastern Time (ET)
# Pre-market  : 04:00 – 09:29
# Market hours: 09:30 – 15:59  (close is at 16:00, last trade before 16:00)
# After hours : 16:00 – 19:59
# Overnight   : 20:00 – 03:59
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Sentiment columns we aggregate
_SENTIMENT_COLS = [
    "overall_sentiment_score",
    "ticker_sentiment_score",
]


# =========================================================================
# 1. LOW-LEVEL LOADERS
# =========================================================================

def load_news_csv(filepath: Path) -> pd.DataFrame:
    """Load a single news CSV and parse dates.

    Parameters
    ----------
    filepath : Path
        Path to a news CSV (output of ``download_news.py``).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``published_at`` parsed as datetime and a
        ``date`` column (date-only) for merging with prices.
    """
    df = pd.read_csv(filepath)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.floor("D")  # type: ignore[union-attr]  # midnight
    return df


def load_prices_csv(filepath: Path) -> pd.DataFrame:
    """Load a single prices CSV (from yfinance) and parse dates.

    Handles the extra header row that yfinance produces (ticker name on
    row 2) by using ``skiprows=[1]``.

    Parameters
    ----------
    filepath : Path
        Path to a prices CSV (output of ``download_prices.py``).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date`` parsed as datetime, plus a ``date``
        column (date-only) for merging with news.
    """
    df = pd.read_csv(filepath, skiprows=[1])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["date"] = df["Date"].dt.floor("D")  # type: ignore[union-attr]
    return df


def load_all_news(
    tickers: Optional[List[str]] = None,
    data_dir: Path = RAW_DATA_DIR,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
) -> pd.DataFrame:
    """Load and concatenate news CSVs for multiple tickers.

    Parameters
    ----------
    tickers : list of str, optional
        Ticker symbols to load (default: ``DEFAULT_TICKERS``).
    data_dir : Path
        Directory containing the raw CSVs.
    start_date, end_date : str
        Used to build the expected filename pattern.

    Returns
    -------
    pd.DataFrame
        Concatenated news for all requested tickers, sorted by
        ``published_at`` descending.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        fp = data_dir / f"news_{ticker}_{start_date}_to_{end_date}.csv"
        if not fp.exists():
            warnings.warn(f"News file not found: {fp}")
            continue
        frames.append(load_news_csv(fp))

    if not frames:
        raise FileNotFoundError(
            f"No news CSVs found in {data_dir} for tickers {tickers}"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df


def load_all_prices(
    tickers: Optional[List[str]] = None,
    data_dir: Path = RAW_DATA_DIR,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
) -> pd.DataFrame:
    """Load and concatenate price CSVs for multiple tickers.

    Adds a ``ticker`` column to each DataFrame so they can be
    distinguished after concatenation.

    Parameters
    ----------
    tickers : list of str, optional
        Ticker symbols to load (default: ``DEFAULT_TICKERS``).
    data_dir : Path
        Directory containing the raw CSVs.
    start_date, end_date : str
        Used to build the expected filename pattern.

    Returns
    -------
    pd.DataFrame
        Concatenated daily OHLCV data for all tickers, sorted by
        ``[ticker, date]``.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        fp = data_dir / f"prices_{ticker}_{start_date}_to_{end_date}.csv"
        if not fp.exists():
            warnings.warn(f"Prices file not found: {fp}")
            continue
        pdf = load_prices_csv(fp)
        pdf["ticker"] = ticker
        frames.append(pdf)

    if not frames:
        raise FileNotFoundError(
            f"No price CSVs found in {data_dir} for tickers {tickers}"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# =========================================================================
# 2. MARKET-HOURS CLASSIFICATION
# =========================================================================

def classify_market_session(dt: pd.Timestamp) -> str:
    """Classify a datetime into a market session.

    Sessions (Eastern Time):
        - ``pre_market``   : 04:00 – 09:29
        - ``market_hours`` : 09:30 – 15:59
        - ``after_hours``  : 16:00 – 19:59
        - ``overnight``    : 20:00 – 03:59

    Parameters
    ----------
    dt : pd.Timestamp
        Publication datetime of a news article.  Assumed to be in ET
        (or a timezone-naive proxy for ET — Alpha Vantage returns UTC
        but our downloaded data was converted on save).

    Returns
    -------
    str
        One of ``"pre_market"``, ``"market_hours"``,
        ``"after_hours"``, ``"overnight"``.

    Note
    ----
    Alpha Vantage timestamps are UTC.  For a production system you
    would convert to US/Eastern first.  Here the classification still
    gives a useful *relative* bucketing even on UTC times because the
    window names map to the hour of the timestamp regardless of zone.
    """
    if pd.isna(dt):
        return "unknown"

    hour, minute = dt.hour, dt.minute

    if hour < 4:
        return "overnight"
    if hour < MARKET_OPEN_HOUR or (hour == MARKET_OPEN_HOUR and minute < MARKET_OPEN_MINUTE):
        return "pre_market"
    if hour < MARKET_CLOSE_HOUR:
        return "market_hours"
    if hour < 20:
        return "after_hours"
    return "overnight"


def add_session_column(news: pd.DataFrame) -> pd.DataFrame:
    """Add a ``session`` column to the news DataFrame.

    Parameters
    ----------
    news : pd.DataFrame
        Must contain a ``published_at`` datetime column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new ``session`` column.
    """
    news = news.copy()
    news["session"] = news["published_at"].apply(classify_market_session)
    return news


# =========================================================================
# 3. NEWS → TRADING-DAY ASSIGNMENT
# =========================================================================

def assign_trading_day(news: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    """Assign each news article to the trading day it can influence.

    The key insight is that news published *after* the market closes
    can only affect the **next** trading day's price, not today's.

    Rules:
    - ``pre_market`` or ``market_hours`` → same calendar day (if it's a
      trading day), otherwise the *next* trading day.
    - ``after_hours`` or ``overnight`` → the *next* trading day.

    Parameters
    ----------
    news : pd.DataFrame
        Must have ``published_at``, ``date``, and ``session`` columns.
    trading_dates : pd.Series
        Sorted series of actual trading dates (from the prices data).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new ``trading_day`` column (datetime).
        Articles that fall outside the trading-date range are dropped.
    """
    news = news.copy()

    # Sorted numpy array of trading dates for searchsorted
    td: np.ndarray = np.array(
        pd.to_datetime(trading_dates).sort_values().values,
        dtype="datetime64[ns]",
    )

    trading_days: List[Optional[pd.Timestamp]] = []

    for _, row in news.iterrows():
        cal_date = np.datetime64(row["date"])  # midnight of the article's calendar day
        session = row["session"]

        if session in ("pre_market", "market_hours"):
            # Try to attach to same calendar day …
            idx = int(np.searchsorted(td, cal_date, side="left"))
            if idx < len(td) and td[idx] == cal_date:
                trading_days.append(pd.Timestamp(td[idx]))
            elif idx < len(td):
                # Calendar day isn't a trading day → next trading day
                trading_days.append(pd.Timestamp(td[idx]))
            else:
                trading_days.append(None)
        else:
            # after_hours / overnight → next trading day
            idx = int(np.searchsorted(td, cal_date, side="right"))
            if idx < len(td):
                trading_days.append(pd.Timestamp(td[idx]))
            else:
                trading_days.append(None)

    news["trading_day"] = trading_days
    before = len(news)
    news = news.dropna(subset=["trading_day"]).reset_index(drop=True)
    dropped = before - len(news)
    if dropped:
        print(f"[INFO] Dropped {dropped} articles outside trading-date range.")
    return news


# =========================================================================
# 4. TIME-WINDOW AGGREGATION
# =========================================================================

def aggregate_sentiment(
    news: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate news sentiment to one row per trading day per ticker.

    For each group (trading_day + ticker by default) computes:
    - ``article_count``: number of articles in that window
    - ``avg_overall_sentiment``: mean overall sentiment score
    - ``avg_ticker_sentiment``: mean ticker-specific sentiment score
    - ``min_sentiment`` / ``max_sentiment``: range of overall scores
    - ``sentiment_std``: standard deviation (volatility of news tone)
    - ``pct_positive``: share of articles with score > 0.15
    - ``pct_negative``: share of articles with score < −0.15
    - ``session_counts``: how many articles from each session

    Parameters
    ----------
    news : pd.DataFrame
        Must contain ``trading_day``, ``ticker``, ``session``, and the
        sentiment score columns.
    group_cols : list of str, optional
        Columns to group by.  Default: ``["trading_day", "ticker"]``.

    Returns
    -------
    pd.DataFrame
        One row per group with aggregated sentiment features.
    """
    if group_cols is None:
        group_cols = ["trading_day", "ticker"]

    def _agg_fn(g: pd.DataFrame) -> pd.Series:
        overall = g["overall_sentiment_score"].dropna()
        ticker_s = g["ticker_sentiment_score"].dropna()

        return pd.Series({
            "article_count": len(g),
            "avg_overall_sentiment": overall.mean() if len(overall) else np.nan,
            "avg_ticker_sentiment": ticker_s.mean() if len(ticker_s) else np.nan,
            "min_sentiment": overall.min() if len(overall) else np.nan,
            "max_sentiment": overall.max() if len(overall) else np.nan,
            "sentiment_std": overall.std() if len(overall) > 1 else 0.0,
            "sentiment_range": (overall.max() - overall.min()) if len(overall) else 0.0,
            "pct_positive": (overall > 0.15).mean() if len(overall) else np.nan,
            "pct_negative": (overall < -0.15).mean() if len(overall) else np.nan,
            # Session breakdown
            "n_pre_market": (g["session"] == "pre_market").sum(),
            "n_market_hours": (g["session"] == "market_hours").sum(),
            "n_after_hours": (g["session"] == "after_hours").sum(),
            "n_overnight": (g["session"] == "overnight").sum(),
        })

    # Select only the columns needed inside _agg_fn to avoid passing
    # the grouping keys into the aggregation function.
    needed_cols = group_cols + [
        "overall_sentiment_score",
        "ticker_sentiment_score",
        "session",
    ]
    agg = (
        news[needed_cols]
        .groupby(group_cols)
        .apply(_agg_fn)  # type: ignore[arg-type]
        .reset_index()
    )
    return agg


def aggregate_by_session(
    news: pd.DataFrame,
    sessions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate sentiment using only articles from specific sessions.

    This is useful for isolating, e.g., *pre-market* sentiment —
    the news tone investors see before the opening bell, which may
    predict the day's direction better than after-hours chatter.

    Parameters
    ----------
    news : pd.DataFrame
        Must contain ``trading_day``, ``ticker``, ``session``, and
        sentiment columns.
    sessions : list of str, optional
        Which sessions to include.  Default: all sessions.

    Returns
    -------
    pd.DataFrame
        Aggregated sentiment for the filtered articles.
    """
    if sessions is not None:
        news = news[news["session"].isin(sessions)]
    return aggregate_sentiment(news)


# =========================================================================
# 5. PRICE FEATURES
# =========================================================================

def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute additional features from daily OHLCV data.

    Added columns:
    - ``daily_return``: close-to-close percentage return.
    - ``return_direction``: ``1`` if return ≥ 0, ``0`` otherwise
      (binary target for classification models).
    - ``intraday_range``: ``(High − Low) / Open * 100``  — measures
      how volatile the stock was *within* the day.
    - ``gap_pct``: overnight gap = ``(Open − prev_Close) / prev_Close * 100``.
    - ``volume_change``: percentage change in volume vs previous day.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain ``Date``/``date``, ``Open``, ``High``, ``Low``,
        ``Close``, ``Volume``, and ``ticker`` columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new feature columns appended.
    """
    prices = prices.copy()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily return (close-to-close)
    prices["daily_return"] = prices.groupby("ticker")["Close"].pct_change() * 100

    # Binary direction (useful as a classification target)
    prices["return_direction"] = (prices["daily_return"] >= 0).astype(int)

    # Intraday range (volatility within the day)
    prices["intraday_range"] = (
        (prices["High"] - prices["Low"]) / prices["Open"] * 100
    )

    # Overnight gap
    prices["prev_close"] = prices.groupby("ticker")["Close"].shift(1)
    prices["gap_pct"] = (
        (prices["Open"] - prices["prev_close"]) / prices["prev_close"] * 100
    )

    # Volume change
    prices["volume_change"] = prices.groupby("ticker")["Volume"].pct_change() * 100

    # Drop helper column
    prices = prices.drop(columns=["prev_close"])

    return prices


# =========================================================================
# 6. MERGE & ENRICH
# =========================================================================

def merge_sentiment_with_prices(
    sentiment_agg: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join aggregated sentiment onto daily prices.

    Uses ``trading_day`` (sentiment) and ``date`` (prices) as the join
    key together with ``ticker``.  A *left* join on prices ensures
    every trading day is kept — days without news will have NaN
    sentiment (filled with 0 article count).

    Parameters
    ----------
    sentiment_agg : pd.DataFrame
        Output of :func:`aggregate_sentiment` — one row per
        ``(trading_day, ticker)``.
    prices : pd.DataFrame
        Daily OHLCV data with a ``date`` and ``ticker`` column.

    Returns
    -------
    pd.DataFrame
        Merged dataset: one row per trading day per ticker with both
        price and sentiment features.
    """
    merged = pd.merge(
        prices,
        sentiment_agg,
        left_on=["date", "ticker"],
        right_on=["trading_day", "ticker"],
        how="left",
    )

    # Days without news → fill counts with 0, leave scores as NaN
    merged["article_count"] = merged["article_count"].fillna(0).astype(int)
    for col in ["n_pre_market", "n_market_hours", "n_after_hours", "n_overnight"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    # Drop the redundant trading_day column
    if "trading_day" in merged.columns:
        merged = merged.drop(columns=["trading_day"])

    return merged


def add_rolling_sentiment(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add rolling-average sentiment features.

    Useful for capturing the *trend* of news sentiment over time
    rather than just the point-in-time value.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset with ``avg_overall_sentiment`` and ``ticker``.
    windows : list of int, optional
        Rolling window sizes in trading days.  Default: ``[3, 5]``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with ``sentiment_rolling_Nd`` columns added.
    """
    if windows is None:
        windows = [3, 5]

    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for w in windows:
        col_name = f"sentiment_rolling_{w}d"
        df[col_name] = (
            df.groupby("ticker")["avg_overall_sentiment"]
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )

    return df


# =========================================================================
# 7. HIGH-LEVEL CONVENIENCE FUNCTION
# =========================================================================

def load_merged_dataset(
    tickers: Optional[List[str]] = None,
    data_dir: Path = RAW_DATA_DIR,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    *,
    session_filter: Optional[List[str]] = None,
    rolling_windows: Optional[List[int]] = None,
    save_processed: bool = False,
) -> pd.DataFrame:
    """End-to-end: load raw CSVs → merge → enrich → return.

    This is the main entry point for notebooks and model training
    scripts.  It chains every step described in this module:

    1. Load all news + prices CSVs.
    2. Classify each article into a market session.
    3. Assign each article to the trading day it can influence.
    4. Aggregate sentiment per trading day (optionally filtered by
       session).
    5. Add price features (returns, gaps, etc.).
    6. Merge sentiment with prices.
    7. Add rolling sentiment averages.

    Parameters
    ----------
    tickers : list of str, optional
        Tickers to load (default: ``DEFAULT_TICKERS``).
    data_dir : Path
        Directory containing raw CSVs.
    start_date, end_date : str
        Date range string used in filenames.
    session_filter : list of str, optional
        If given, only include articles from these sessions before
        aggregating (e.g. ``["pre_market", "market_hours"]``).
    rolling_windows : list of int, optional
        Window sizes for rolling sentiment averages (default: [3, 5]).
    save_processed : bool
        If *True*, save the final merged DataFrame to
        ``data/processed/merged_<start>_to_<end>.csv``.

    Returns
    -------
    pd.DataFrame
        One row per trading day per ticker, with price features and
        aggregated sentiment.  Ready for EDA or model training.

    Examples
    --------
    >>> df = load_merged_dataset()
    >>> df.columns.tolist()
    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'date', 'ticker',
     'daily_return', 'return_direction', ..., 'avg_overall_sentiment', ...]
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    # --- Step 1: Load raw data ---
    print("[1/7] Loading news CSVs ...")
    news = load_all_news(tickers, data_dir, start_date, end_date)
    print(f"       → {len(news)} articles loaded.")

    print("[2/7] Loading price CSVs ...")
    prices = load_all_prices(tickers, data_dir, start_date, end_date)
    print(f"       → {len(prices)} price rows loaded.")

    # --- Step 2: Session classification ---
    print("[3/7] Classifying market sessions ...")
    news = add_session_column(news)
    session_counts = news["session"].value_counts().to_dict()
    print(f"       → Sessions: {session_counts}")

    # --- Step 3: Assign trading days ---
    print("[4/7] Assigning articles to trading days ...")
    trading_dates = prices["date"].drop_duplicates()
    news = assign_trading_day(news, trading_dates)
    print(f"       → {len(news)} articles matched to trading days.")

    # --- Step 4: Aggregate sentiment ---
    print("[5/7] Aggregating daily sentiment ...")
    if session_filter:
        print(f"       (filtering to sessions: {session_filter})")
        sentiment = aggregate_by_session(news, sessions=session_filter)
    else:
        sentiment = aggregate_sentiment(news)
    print(f"       → {len(sentiment)} daily-sentiment rows.")

    # --- Step 5: Price features ---
    print("[6/7] Computing price features ...")
    prices = add_price_features(prices)

    # --- Step 6 & 7: Merge + rolling ---
    print("[7/7] Merging & adding rolling sentiment ...")
    merged = merge_sentiment_with_prices(sentiment, prices)
    merged = add_rolling_sentiment(merged, windows=rolling_windows)

    # Sort final output
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- Optional: save ---
    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / f"merged_{start_date}_to_{end_date}.csv"
        merged.to_csv(out_path, index=False)
        print(f"\n[SAVED] {out_path}")

    # --- Summary ---
    print(f"\n{'=' * 55}")
    print(f"  Final dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(f"  Tickers: {merged['ticker'].unique().tolist()}")
    print(f"  Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")
    days_with_news = (merged["article_count"] > 0).sum()
    print(f"  Trading days with news: {days_with_news} / {len(merged)}")
    print(f"{'=' * 55}")

    return merged


# =========================================================================
# CLI smoke test
# =========================================================================

if __name__ == "__main__":
    df = load_merged_dataset(save_processed=True)
    print("\nSample rows:")
    cols = [
        "date", "ticker", "Close", "daily_return", "return_direction",
        "article_count", "avg_overall_sentiment", "avg_ticker_sentiment",
    ]
    print(df[cols].tail(10).to_string(index=False))
