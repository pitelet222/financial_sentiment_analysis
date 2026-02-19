"""
News Fetcher — Multi-Source Live News Pipeline with Disk Cache
==============================================================

Provides a **zero-API-key** way to pull recent financial headlines for
any ticker using:

1. **RSS feeds** — Google News, Yahoo Finance
2. **yfinance API** — Yahoo Finance curated news
3. **SEC EDGAR** — 8-K material-event filings (lawsuits, departures,
   impairments, acquisitions) which provide a more **balanced** sentiment
   distribution than news headlines alone.

Articles are cached on disk so repeated calls (e.g. from the Streamlit
dashboard) don't hammer the feeds.

Typical usage
-------------
>>> from src.data.news_fetcher import fetch_live_news
>>> df = fetch_live_news("AAPL")
>>> df[["published_at", "title", "source"]].head()

The module also exposes ``fetch_and_cache`` which transparently returns
cached data when it is still fresh (configurable TTL).

Architecture
------------
``fetch_live_news``  →  hits RSS + yfinance + EDGAR  →  normalises to DataFrame
``fetch_and_cache``  →  checks disk cache first, falls back to live

The DataFrame schema intentionally matches the raw news CSVs produced
by ``scripts/download_news.py`` so the rest of the pipeline
(``data_loader``, ``SentimentAnalyzer.predict_dataframe``, dashboard)
can consume it without changes.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

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

# -------------------------------------------------------------------------
# SEC EDGAR constants
# -------------------------------------------------------------------------

# SEC requires a descriptive User-Agent (name + email).  Update with your
# own details if you fork this project.
_SEC_USER_AGENT = "FinancialSentimentAnalysis/1.0 (contact@example.com)"

# Ticker → CIK mapping for the 19 project tickers.  The SEC identifies
# companies by CIK (Central Index Key), not ticker.  This avoids an
# extra HTTP round-trip on every call.
_TICKER_TO_CIK: Dict[str, str] = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "NVDA":  "0001045810",
    "META":  "0001326801",
    "JPM":   "0000019617",
    "GS":    "0000886982",
    "BAC":   "0000070858",
    "JNJ":   "0000200406",
    "UNH":   "0000731766",
    "PFE":   "0000078003",
    "TSLA":  "0001318605",
    "WMT":   "0000104169",
    "KO":    "0000021344",
    "XOM":   "0000034088",
    "CVX":   "0000093410",
    "CAT":   "0000018230",
    "BA":    "0000012927",
}

# Human-readable descriptions for common 8-K item numbers.  These replace
# the terse SEC codes with text that FinBERT can score meaningfully.
_8K_ITEM_DESCRIPTIONS: Dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy Listing Requirements",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


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
# SEC EDGAR: 8-K filings
# =========================================================================

def _resolve_cik(ticker: str) -> Optional[str]:
    """Return the 10-digit zero-padded CIK for *ticker*.

    Uses the hardcoded mapping first (instant), then falls back to the
    SEC company-tickers JSON endpoint for unknown tickers.
    """
    cik = _TICKER_TO_CIK.get(ticker.upper())
    if cik:
        return cik

    # Fallback: query SEC's ticker→CIK mapping (updated daily)
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _SEC_USER_AGENT})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = _json.loads(resp.read().decode())
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                _TICKER_TO_CIK[ticker.upper()] = cik  # cache for session
                return cik
    except Exception as exc:
        logger.warning("SEC CIK lookup failed for %s: %s", ticker, exc)

    return None


def _fetch_edgar_filings(
    ticker: str,
    max_filings: int = 15,
    form_types: Optional[List[str]] = None,
) -> List[dict]:
    """Fetch recent SEC filings (8-K by default) for *ticker*.

    Uses the official SEC EDGAR submissions API
    (``data.sec.gov/submissions/CIK{cik}.json``), which is free, requires
    no API key, and returns structured JSON.  Only a descriptive
    ``User-Agent`` header is required per SEC policy.

    The filing's *items* field (e.g. ``2.05 — Costs Associated with Exit
    or Disposal Activities``) is translated into a human-readable title
    that FinBERT can score.  This provides **more balanced sentiment**
    than news headlines, which skew bullish.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    max_filings : int
        Maximum number of filings to return (default 15).
    form_types : list of str, optional
        SEC form types to include (default ``["8-K", "8-K/A"]``).

    Returns
    -------
    list of dict
        Article dicts matching the project schema.
    """
    if form_types is None:
        form_types = ["8-K", "8-K/A"]

    cik = _resolve_cik(ticker)
    if not cik:
        logger.warning("No CIK found for %s — skipping EDGAR", ticker)
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _SEC_USER_AGENT})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = _json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("EDGAR submissions fetch failed for %s: %s", ticker, exc)
        return []

    company_name = data.get("name", ticker)
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    primary_descs = recent.get("primaryDocDescription", [])
    items_list = recent.get("items", [])

    articles: List[dict] = []
    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        if len(articles) >= max_filings:
            break

        filing_date_str = dates[i] if i < len(dates) else ""
        accession = accessions[i].replace("-", "") if i < len(accessions) else ""
        accession_dashed = accessions[i] if i < len(accessions) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        items_raw = items_list[i] if i < len(items_list) else ""

        # Build a meaningful title from the 8-K item numbers
        item_numbers = [s.strip() for s in items_raw.split(",") if s.strip()]
        item_descs = [_8K_ITEM_DESCRIPTIONS.get(n, n) for n in item_numbers]

        if item_descs:
            title = f"{company_name} — SEC 8-K: {'; '.join(item_descs)}"
        else:
            desc = primary_descs[i] if i < len(primary_descs) else "Current Report"
            title = f"{company_name} — SEC 8-K Filing: {desc}"

        # Published date
        published = None
        if filing_date_str:
            try:
                published = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        # Direct link to the filing on EDGAR
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik.lstrip('0')}/{accession}/{primary_doc}"
        ) if accession and primary_doc else ""

        articles.append({
            "title": title,
            "summary": f"SEC {form} filing for {company_name}. "
                       f"Items: {', '.join(item_descs) if item_descs else 'N/A'}. "
                       f"Filed {filing_date_str}.",
            "source": "SEC EDGAR",
            "url": filing_url,
            "published_at": published,
            "authors": "",
            "overall_sentiment_score": None,
            "overall_sentiment_label": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None,
            "ticker": ticker,
        })

    logger.info("EDGAR returned %d filings for %s", len(articles), ticker)
    return articles


# =========================================================================
# Public: fetch live news
# =========================================================================

def fetch_live_news(
    ticker: str,
    feeds: Optional[List[dict]] = None,
    max_articles: int = 60,
    include_yfinance: bool = True,
    include_edgar: bool = True,
) -> pd.DataFrame:
    """Fetch recent headlines for *ticker* from RSS + yfinance + SEC EDGAR.

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
    include_edgar : bool
        If True (default), also pull recent 8-K filings from SEC EDGAR.
        These tend to have **neutral/negative** sentiment, which balances
        the positive skew of news headlines.

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

    # 3) SEC EDGAR 8-K filings (balances bullish news bias)
    if include_edgar:
        edgar_articles = _fetch_edgar_filings(ticker, max_filings=15)
        all_articles.extend(edgar_articles)
        logger.info("EDGAR contributed %d filings for %s", len(edgar_articles), ticker)

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
