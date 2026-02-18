"""
download_news_extended.py
=========================
Download 1 year of news for all project tickers in monthly chunks.
Respects Alpha Vantage free tier: 25 req/day, 5 req/min.

Supports **resume**: skips tickers that already have a CSV in data/raw/.
Run it once per day to download 2 tickers (~24 API calls).

Usage:
    python scripts/download_news_extended.py              # download next 2 missing tickers
    python scripts/download_news_extended.py --all        # download ALL missing tickers (multi-day)
    python scripts/download_news_extended.py --ticker AMZN # download a specific ticker
"""

import os
import sys
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

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "JPM", "GS", "BAC",
    "JNJ", "UNH", "PFE",
    "TSLA", "WMT", "KO",
    "XOM", "CVX",
    "CAT", "BA",
]
START_DATE = "2025-02-13"
END_DATE = "2026-02-13"

# Max tickers to download per run (free tier: 25 req/day -> ~2 tickers)
MAX_TICKERS_PER_RUN = 2

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

def _ticker_csv_exists(ticker: str) -> bool:
    """Check if news CSV already exists for this ticker."""
    filename = f"news_{ticker}_{START_DATE}_to_{END_DATE}.csv"
    return (RAW_DIR / filename).exists()


def _get_tickers_to_download(specific_ticker: str | None = None,
                              download_all: bool = False) -> list[str]:
    """Determine which tickers still need downloading."""
    if specific_ticker:
        if _ticker_csv_exists(specific_ticker):
            print(f"  [SKIP] {specific_ticker} — CSV already exists. "
                  f"Delete it to re-download.")
            return []
        return [specific_ticker]

    missing = [t for t in TICKERS if not _ticker_csv_exists(t)]

    if not missing:
        print("  [DONE] All tickers already have news CSVs!")
        return []

    if download_all:
        return missing

    # Limit to MAX_TICKERS_PER_RUN for free tier
    batch = missing[:MAX_TICKERS_PER_RUN]
    remaining = len(missing) - len(batch)
    if remaining > 0:
        print(f"  [INFO] Downloading {len(batch)}/{len(missing)} missing tickers "
              f"this run. {remaining} remaining for next run(s).")
    return batch


def main():
    if not API_KEY:
        print("[ERROR] ALPHA_VANTAGE_KEY not found in .env")
        return

    # Parse CLI args
    specific_ticker = None
    download_all = False
    args = sys.argv[1:]
    if "--all" in args:
        download_all = True
    if "--ticker" in args:
        idx = args.index("--ticker")
        if idx + 1 < len(args):
            specific_ticker = args[idx + 1].upper()

    tickers_to_dl = _get_tickers_to_download(specific_ticker, download_all)
    if not tickers_to_dl:
        return

    chunks = list(_month_chunks(START_DATE, END_DATE))
    total_calls = len(chunks) * len(tickers_to_dl)

    # Show already-downloaded tickers
    already_done = [t for t in TICKERS if _ticker_csv_exists(t)]

    print("=" * 60)
    print("  Extended News Download — 1 Year of Data (with Resume)")
    print(f"  All tickers  : {TICKERS}")
    print(f"  Already done : {already_done if already_done else 'none'}")
    print(f"  To download  : {tickers_to_dl}")
    print(f"  Period       : {START_DATE} -> {END_DATE}")
    print(f"  Chunks       : {len(chunks)} months x {len(tickers_to_dl)} tickers "
          f"= {total_calls} API calls")
    print(f"  Est time     : ~{total_calls * DELAY_BETWEEN_CALLS // 60} minutes")
    print("=" * 60)

    call_count = 0

    for ticker in tickers_to_dl:
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

    print(f"\n[DONE] This run complete. Total API calls: {call_count}")

    # Show progress summary
    still_missing = [t for t in TICKERS if not _ticker_csv_exists(t)]
    done_now = [t for t in TICKERS if _ticker_csv_exists(t)]
    print(f"\n  Progress: {len(done_now)}/{len(TICKERS)} tickers downloaded")
    if still_missing:
        print(f"  Still missing: {still_missing}")
        print(f"  Run this script again tomorrow to download the next batch.")
    else:
        print("  All tickers complete! Next steps:")
        print("    1. Rebuild merged dataset:  python -c \"from src.data.data_loader import load_merged_dataset; load_merged_dataset(save_processed=True)\"")
        print("    2. Retrain XGBoost:         python scripts/train_multihorizon.py")


if __name__ == "__main__":
    main()
