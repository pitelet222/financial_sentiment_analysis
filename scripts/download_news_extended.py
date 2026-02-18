"""
download_news_extended.py
=========================
Download 1 year of news for AAPL + MSFT in monthly chunks.
Respects Alpha Vantage free tier: 25 req/day, 5 req/min.

Usage:
    python scripts/download_news_extended.py
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"

TICKERS = ["AAPL", "MSFT"]
START_DATE = "2025-02-13"
END_DATE = "2026-02-13"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DELAY_BETWEEN_CALLS = 15  # seconds (5 req/min limit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _av_fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M")


def _month_chunks(start: str, end: str):
    """Yield (chunk_start, chunk_end) datetime pairs, ~1 month each."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=30), e)
        yield s, chunk_end
        s = chunk_end + timedelta(days=1)


def fetch_chunk(ticker: str, dt_from: datetime, dt_to: datetime) -> list[dict]:
    """Fetch one month-chunk of news from Alpha Vantage."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": _av_fmt(dt_from),
        "time_to": _av_fmt(dt_to),
        "limit": 1000,
        "sort": "RELEVANCE",
        "apikey": API_KEY,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [ERROR] Request failed: {exc}")
        return []

    if "Note" in data:
        print(f"  [RATE LIMIT] {data['Note']}")
        return []
    if "Information" in data:
        print(f"  [INFO] {data['Information']}")
        return []
    if "Error Message" in data:
        print(f"  [ERROR] {data['Error Message']}")
        return []

    feed = data.get("feed", [])
    articles = []
    for art in feed:
        # Find ticker-specific sentiment
        ts_score, ts_label = None, None
        for ts in art.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                ts_score = float(ts.get("ticker_sentiment_score", 0))
                ts_label = ts.get("ticker_sentiment_label", "")
                break

        articles.append({
            "title": art.get("title", ""),
            "summary": art.get("summary", ""),
            "source": art.get("source", ""),
            "url": art.get("url", ""),
            "published_at": art.get("time_published", ""),
            "authors": ", ".join(art.get("authors", [])),
            "overall_sentiment_score": float(art.get("overall_sentiment_score", 0)),
            "overall_sentiment_label": art.get("overall_sentiment_label", ""),
            "ticker_sentiment_score": ts_score,
            "ticker_sentiment_label": ts_label,
            "ticker": ticker,
        })
    return articles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("[ERROR] ALPHA_VANTAGE_KEY not found in .env")
        return

    chunks = list(_month_chunks(START_DATE, END_DATE))
    total_calls = len(chunks) * len(TICKERS)
    print("=" * 60)
    print("  Extended News Download — 1 Year of Data")
    print(f"  Tickers : {TICKERS}")
    print(f"  Period  : {START_DATE} -> {END_DATE}")
    print(f"  Chunks  : {len(chunks)} months x {len(TICKERS)} tickers = {total_calls} API calls")
    print(f"  Est time: ~{total_calls * DELAY_BETWEEN_CALLS // 60} minutes")
    print("=" * 60)

    call_count = 0

    for ticker in TICKERS:
        all_articles: list[dict] = []

        for i, (c_start, c_end) in enumerate(chunks, 1):
            call_count += 1
            print(f"  [{call_count}/{total_calls}] {ticker} : "
                  f"{c_start.strftime('%Y-%m-%d')} -> {c_end.strftime('%Y-%m-%d')} ...",
                  end=" ", flush=True)

            arts = fetch_chunk(ticker, c_start, c_end)
            print(f"-> {len(arts)} articles")
            all_articles.extend(arts)

            # Rate limit pause (skip after last call)
            if call_count < total_calls:
                time.sleep(DELAY_BETWEEN_CALLS)

        # Deduplicate by title
        df = pd.DataFrame(all_articles)
        if df.empty:
            print(f"  [WARNING] No articles found for {ticker}")
            continue

        df["published_at"] = pd.to_datetime(
            df["published_at"], format="%Y%m%dT%H%M%S", errors="coerce"
        )
        df = df.drop_duplicates(subset=["title"], keep="first")
        df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

        # Save
        filename = f"news_{ticker}_{START_DATE}_to_{END_DATE}.csv"
        path = RAW_DIR / filename
        df.to_csv(path, index=False)
        print(f"  [SAVED] {len(df)} unique articles -> {path.name}")

    print(f"\n[DONE] All downloads complete. Total API calls: {call_count}")


if __name__ == "__main__":
    main()
