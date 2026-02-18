"""
News Fetcher — Free RSS-Based Live News Pipeline with Disk Cache
=================================================================

Provides a **zero-API-key** way to pull recent financial headlines for
any ticker using public RSS feeds from Google News, Yahoo Finance, and
MarketWatch.  Articles are cached on disk so repeated calls (e.g. from
the Streamlit dashboard) don't hammer the feeds.

Typical usage
-------------
>>> from src.data.news_fetcher import fetch_live_news
>>> df = fetch_live_news("AAPL")
>>> df[["published_at", "title", "source"]].head()

The module also exposes ``fetch_and_cache`` which transparently returns
cached data when it is still fresh (configurable TTL).

Architecture
------------
``fetch_live_news``  →  hits RSS feeds  →  normalises to DataFrame
``fetch_and_cache``  →  checks disk cache first, falls back to live

The DataFrame schema intentionally matches the raw news CSVs produced
by ``scripts/download_news.py`` so the rest of the pipeline
(``data_loader``, ``SentimentAnalyzer.predict_dataframe``, dashboard)
can consume it without changes.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import feedparser
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "cache"

# Default cache time-to-live (seconds).  1 hour keeps the dashboard
# responsive while avoiding excessive feed requests.
DEFAULT_CACHE_TTL = 3600

# RSS feed templates — {ticker} is replaced at runtime
_RSS_FEEDS = [
    {
        "name": "Google News",
        "url": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        "source_field": "Google News",
    },
    {
        "name": "Yahoo Finance",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "source_field": "Yahoo Finance",
    },
]

# Timeout for each HTTP request (seconds)
_REQUEST_TIMEOUT = 15


# =========================================================================
# Low-level: fetch one feed
# =========================================================================

def _parse_feed(url: str, source_name: str, ticker: str) -> List[dict]:
    """Parse a single RSS feed and return a list of article dicts."""
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
    except Exception as exc:
        logger.warning("Failed to fetch %s for %s: %s", source_name, ticker, exc)
        return []

    articles: List[dict] = []
    for entry in feed.entries:
        # Published date
        published = None
        for date_field in ("published_parsed", "updated_parsed"):
            parsed_time = getattr(entry, date_field, None)
            if parsed_time:
                published = datetime(*parsed_time[:6], tzinfo=timezone.utc)
                break

        articles.append({
            "title": getattr(entry, "title", ""),
            "summary": getattr(entry, "summary", getattr(entry, "description", "")),
            "source": source_name,
            "url": getattr(entry, "link", ""),
            "published_at": published,
            "authors": getattr(entry, "author", ""),
            "overall_sentiment_score": None,   # to be filled by model
            "overall_sentiment_label": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None,
            "ticker": ticker,
        })

    return articles


# =========================================================================
# yfinance news helper
# =========================================================================

def _fetch_yfinance_news(ticker: str) -> List[dict]:
    """Pull headlines from yfinance Ticker.news (Yahoo Finance API).

    Returns a list of article dicts matching the project schema.
    This complements the RSS feeds with Yahoo Finance's curated news
    feed which often contains analyst notes and earnings coverage
    that RSS misses.
    """
    try:
        tick = yf.Ticker(ticker)
        raw_news = tick.news  # list[dict] | None
    except Exception as exc:
        logger.warning("yfinance .news failed for %s: %s", ticker, exc)
        return []

    if not raw_news:
        return []

    articles: List[dict] = []
    for item in raw_news:
        content = item.get("content", item)  # v0.2.x nests under 'content'

        title = content.get("title", "")
        summary = content.get("summary", content.get("description", ""))

        # Provider / source
        provider = content.get("provider", {})
        source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"

        # URL
        canon = content.get("canonicalUrl", {})
        url = canon.get("url", "") if isinstance(canon, dict) else content.get("url", content.get("link", ""))

        # Published date
        pub_str = content.get("pubDate", content.get("displayTime", ""))
        published = None
        if pub_str:
            try:
                published = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        articles.append({
            "title": title,
            "summary": summary,
            "source": f"{source} (yfinance)",
            "url": url,
            "published_at": published,
            "authors": "",
            "overall_sentiment_score": None,
            "overall_sentiment_label": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None,
            "ticker": ticker,
        })

    return articles


# =========================================================================
# Public: fetch live news
# =========================================================================

def fetch_live_news(
    ticker: str,
    feeds: Optional[List[dict]] = None,
    max_articles: int = 60,
    include_yfinance: bool = True,
) -> pd.DataFrame:
    """Fetch recent headlines for *ticker* from public RSS feeds + yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    feeds : list of dict, optional
        Override the default RSS feed list.  Each dict must have keys
        ``url`` (with ``{ticker}`` placeholder), ``source_field``.
    max_articles : int
        Cap on total articles returned (default 60).
    include_yfinance : bool
        If True (default), also pull headlines from ``yfinance.Ticker.news``.

    Returns
    -------
    pd.DataFrame
        Schema matches ``scripts/download_news.py`` output:
        ``title, summary, source, url, published_at, authors,
        overall_sentiment_score, overall_sentiment_label,
        ticker_sentiment_score, ticker_sentiment_label, ticker``.
    """
    if feeds is None:
        feeds = _RSS_FEEDS

    all_articles: List[dict] = []

    # 1) RSS feeds
    for feed_cfg in feeds:
        url = feed_cfg["url"].format(ticker=ticker)
        source = feed_cfg.get("source_field", feed_cfg.get("name", "RSS"))
        articles = _parse_feed(url, source, ticker)
        all_articles.extend(articles)
        # Be polite — small delay between feeds
        time.sleep(0.3)

    # 2) yfinance curated news
    if include_yfinance:
        yf_articles = _fetch_yfinance_news(ticker)
        all_articles.extend(yf_articles)
        logger.info("yfinance contributed %d articles for %s", len(yf_articles), ticker)

    if not all_articles:
        logger.info("No live articles found for %s", ticker)
        return pd.DataFrame(columns=[
            "title", "summary", "source", "url", "published_at", "authors",
            "overall_sentiment_score", "overall_sentiment_label",
            "ticker_sentiment_score", "ticker_sentiment_label", "ticker",
        ])

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # Drop duplicates by title (same headline from multiple feeds)
    df = df.drop_duplicates(subset=["title"], keep="first")

    # Sort newest first, cap
    df = (
        df.sort_values("published_at", ascending=False)
        .head(max_articles)
        .reset_index(drop=True)
    )

    return df


# =========================================================================
# Public: cached variant
# =========================================================================

def fetch_and_cache(
    ticker: str,
    cache_dir: Path = CACHE_DIR,
    ttl: int = DEFAULT_CACHE_TTL,
    **kwargs,
) -> pd.DataFrame:
    """Fetch live news with transparent disk caching.

    If a cached CSV for *ticker* exists and is fresher than *ttl*
    seconds, it is returned directly.  Otherwise, ``fetch_live_news``
    is called and the result is written to cache.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    cache_dir : Path
        Directory for cached CSVs (created automatically).
    ttl : int
        Cache time-to-live in seconds (default 3600 = 1 h).
    **kwargs
        Forwarded to ``fetch_live_news``.

    Returns
    -------
    pd.DataFrame
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"news_live_{ticker.upper()}.csv"

    # Check cache freshness
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < ttl:
            logger.info("Cache hit for %s (age %.0fs < %ds)", ticker, age, ttl)
            df = pd.read_csv(cache_file)
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            return df

    # Cache miss → fetch
    logger.info("Cache miss for %s — fetching live news …", ticker)
    df = fetch_live_news(ticker, **kwargs)

    if not df.empty:
        df.to_csv(cache_file, index=False)
        logger.info("Cached %d articles for %s", len(df), ticker)

    return df


# =========================================================================
# CLI smoke test
# =========================================================================

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching live news for {ticker} …")
    df = fetch_live_news(ticker)
    print(f"\n{len(df)} articles found:\n")
    if not df.empty:
        print(df[["published_at", "title", "source"]].to_string(index=False))
