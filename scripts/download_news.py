"""
download_news.py
================
Script to download financial news articles using the Alpha Vantage
News & Sentiment API.

This script fetches news articles related to specific stock tickers and saves
each result set as a CSV file in `data/raw/`. Alpha Vantage also returns
pre-computed sentiment scores for each article, which will be useful later as
a baseline to compare against our own model.

How it works:
    1. Reads the ALPHA_VANTAGE_KEY from a `.env` file in the project root.
    2. For each ticker, queries the Alpha Vantage NEWS_SENTIMENT endpoint.
    3. The API returns articles with metadata, summaries, and sentiment scores.
    4. The script loops through multiple time windows (monthly chunks) to cover
       the full date range, since the API returns up to 1000 articles per call.
    5. Saves the articles as a CSV with one row per article.

Usage:
    python scripts/download_news.py

    You can also import the function in a notebook:
        from scripts.download_news import download_news

Dependencies:
    - requests: HTTP requests to the Alpha Vantage API
    - pandas: Data manipulation
    - python-dotenv: Load environment variables from .env file

API Key:
    Sign up at https://www.alphavantage.co/support/#api-key to get a free key.
    Store it in a `.env` file in the project root:
        ALPHA_VANTAGE_KEY=your_key_here

    Free tier limits: 25 requests/day, 5 requests/minute.

Output:
    CSV files saved to: data/raw/news_<TICKER>_<START>_to_<END>.csv
    Columns: title, summary, source, url, published_at, authors,
             overall_sentiment_score, overall_sentiment_label,
             ticker_sentiment_score, ticker_sentiment_label, ticker
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load environment variables from .env file (expects ALPHA_VANTAGE_KEY)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Alpha Vantage base URL
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Default tickers — should match the ones in download_prices.py
DEFAULT_TICKERS = ["AAPL", "MSFT"]

# Default date range — format: "YYYY-MM-DD"
DEFAULT_START_DATE = "2025-02-13"
DEFAULT_END_DATE = "2026-02-13"

# Where to save the raw news data
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Maximum number of articles to request per API call (Alpha Vantage max: 1000)
ARTICLES_PER_REQUEST = 200


# ---------------------------------------------------------------------------
# Helper: convert date formats
# ---------------------------------------------------------------------------

def to_alpha_vantage_format(date_str: str) -> str:
    """
    Convert a date string from "YYYY-MM-DD" to Alpha Vantage's expected
    format "YYYYMMDDTHHMM".

    Parameters
    ----------
    date_str : str
        Date in "YYYY-MM-DD" format (e.g. "2025-11-01").

    Returns
    -------
    str
        Date in "YYYYMMDDTHHMM" format (e.g. "20251101T0000").
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%m%dT%H%M")


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def download_news(api_key: str, ticker: str,
                  start_date: str, end_date: str,
                  limit: int = ARTICLES_PER_REQUEST) -> pd.DataFrame:
    """
    Download news articles related to a stock ticker from Alpha Vantage.

    The Alpha Vantage NEWS_SENTIMENT endpoint returns articles with:
    - Article metadata (title, source, URL, authors)
    - A text summary of the article
    - An overall sentiment score and label for the entire article
    - A ticker-specific sentiment score and label (how the article
      relates specifically to the requested ticker)

    Sentiment scores range from -1 (very bearish) to +1 (very bullish):
        Score <= -0.35  → Bearish
        -0.35 < Score <= -0.15 → Somewhat-Bearish
        -0.15 < Score < 0.15  → Neutral
        0.15 <= Score < 0.35  → Somewhat-Bullish
        Score >= 0.35  → Bullish

    Parameters
    ----------
    api_key : str
        Your Alpha Vantage API key.
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    start_date : str
        Start date in "YYYY-MM-DD" format.
    end_date : str
        End date in "YYYY-MM-DD" format.
    limit : int
        Maximum articles per API call (default: 200, max: 1000).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per article. Columns:
        - title: Article headline
        - summary: Article summary text
        - source: Name of the news source
        - url: Link to the full article
        - published_at: Publication datetime (UTC)
        - authors: Author name(s), comma-separated
        - overall_sentiment_score: Sentiment of the whole article (-1 to +1)
        - overall_sentiment_label: e.g. "Bullish", "Neutral", "Bearish"
        - ticker_sentiment_score: Sentiment specific to this ticker (-1 to +1)
        - ticker_sentiment_label: e.g. "Bullish", "Neutral", "Bearish"
        - ticker: The ticker symbol this row is associated with

        Returns an empty DataFrame if no articles are found or an error occurs.

    Example
    -------
    >>> df = download_news("your_key", "AAPL", "2025-11-01", "2026-02-13")
    >>> print(df[["title", "overall_sentiment_label"]].head())
    """
    print(f"[INFO] Downloading news for {ticker} "
          f"({start_date} to {end_date}) from Alpha Vantage ...")

    # --- Build request parameters ---
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": to_alpha_vantage_format(start_date),
        "time_to": to_alpha_vantage_format(end_date),
        "limit": limit,
        "sort": "RELEVANCE",
        "apikey": api_key,
    }

    try:
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed for {ticker}: {e}")
        return pd.DataFrame()

    # --- Check for API errors ---
    # Alpha Vantage returns error messages in different formats
    if "Error Message" in data:
        print(f"[ERROR] Alpha Vantage error: {data['Error Message']}")
        return pd.DataFrame()

    if "Note" in data:
        # This usually means we hit the rate limit
        print(f"[WARNING] Alpha Vantage rate limit: {data['Note']}")
        return pd.DataFrame()

    if "Information" in data:
        print(f"[WARNING] Alpha Vantage info: {data['Information']}")
        return pd.DataFrame()

    # --- Extract articles from the "feed" key ---
    feed = data.get("feed", [])

    if not feed:
        print(f"[WARNING] No articles found for {ticker}.")
        return pd.DataFrame()

    # --- Parse each article into a flat row ---
    all_articles = []

    for article in feed:
        # The API returns ticker-specific sentiment inside a list.
        # We need to find the entry that matches our requested ticker.
        ticker_sentiment_score = None
        ticker_sentiment_label = None

        for ts in article.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                ticker_sentiment_score = float(ts.get("ticker_sentiment_score", 0))
                ticker_sentiment_label = ts.get("ticker_sentiment_label", "")
                break

        # Parse the publication time from Alpha Vantage format (YYYYMMDDTHHMMSS)
        raw_time = article.get("time_published", "")

        all_articles.append({
            "title": article.get("title", ""),
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "published_at": raw_time,
            "authors": ", ".join(article.get("authors", [])),
            "overall_sentiment_score": float(
                article.get("overall_sentiment_score", 0)
            ),
            "overall_sentiment_label": article.get(
                "overall_sentiment_label", ""
            ),
            "ticker_sentiment_score": ticker_sentiment_score,
            "ticker_sentiment_label": ticker_sentiment_label,
            "ticker": ticker,
        })

    # --- Convert to DataFrame ---
    df = pd.DataFrame(all_articles)

    # Parse published_at to proper datetime
    df["published_at"] = pd.to_datetime(
        df["published_at"], format="%Y%m%dT%H%M%S", errors="coerce"
    )

    # Sort by publication date (most recent first)
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    print(f"[INFO] Total articles collected for {ticker}: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_news_to_csv(df: pd.DataFrame, ticker: str,
                     start_date: str, end_date: str,
                     output_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Save a news DataFrame to a CSV file in the raw data directory.

    Parameters
    ----------
    df : pd.DataFrame
        News data returned by download_news().
    ticker : str
        Ticker symbol (used in the filename).
    start_date : str
        Start date (used in the filename).
    end_date : str
        End date (used in the filename).
    output_dir : Path
        Directory where the CSV will be saved.

    Returns
    -------
    Path
        Full path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive filename: news_AAPL_2025-11-01_to_2026-02-13.csv
    filename = f"news_{ticker}_{start_date}_to_{end_date}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    print(f"[INFO] Saved {len(df)} articles to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """
    Download news articles for all tickers in DEFAULT_TICKERS and save them
    as individual CSV files in data/raw/.

    Reads the API key from the ALPHA_VANTAGE_KEY environment variable
    (loaded from the .env file).

    Note: Alpha Vantage free tier allows 25 requests/day and 5/minute,
    so the script pauses 15 seconds between tickers to stay within limits.
    """
    # --- Load API key ---
    api_key = os.getenv("ALPHA_VANTAGE_KEY")

    if not api_key:
        print("[ERROR] ALPHA_VANTAGE_KEY not found.")
        print("  Make sure you have a .env file in the project root with:")
        print("  ALPHA_VANTAGE_KEY=your_key_here")
        return

    print("=" * 60)
    print("  Financial Sentiment Analysis — News Data Downloader")
    print("  (powered by Alpha Vantage News & Sentiment API)")
    print("=" * 60)
    print(f"  Tickers : {DEFAULT_TICKERS}")
    print(f"  Period  : {DEFAULT_START_DATE} → {DEFAULT_END_DATE}")
    print(f"  Output  : {RAW_DATA_DIR}")
    print("=" * 60)

    for i, ticker in enumerate(DEFAULT_TICKERS):
        df = download_news(api_key, ticker, DEFAULT_START_DATE, DEFAULT_END_DATE)

        if not df.empty:
            save_news_to_csv(df, ticker, DEFAULT_START_DATE, DEFAULT_END_DATE)

        # Pause between tickers to respect the rate limit (5 req/min)
        if i < len(DEFAULT_TICKERS) - 1:
            print("[INFO] Waiting 15 seconds to respect API rate limits ...")
            time.sleep(15)

    print("\n[DONE] News download complete.")


if __name__ == "__main__":
    main()