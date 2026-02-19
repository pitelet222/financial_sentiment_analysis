"""
Download SEC EDGAR 8-K Filings for Training Data
==================================================

Fetches recent 8-K (material event) filings from the SEC EDGAR API for
all 19 project tickers and saves them as CSVs in ``data/raw/`` with the
same schema as the Alpha Vantage news CSVs.  Each filing title is then
scored by FinBERT to produce ``overall_sentiment_score`` and
``ticker_sentiment_score`` columns.

This supplements the news data with **neutrally / negatively-skewed**
content (director departures, impairments, dispositions, etc.) that
balances the positive bias inherent in news headlines.

Usage
-----
    python scripts/download_edgar_filings.py              # all 19 tickers
    python scripts/download_edgar_filings.py --ticker AAPL MSFT
    python scripts/download_edgar_filings.py --max-filings 40

No API key is required — SEC EDGAR is free and allows 10 req/sec.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DEFAULT_START, DEFAULT_END, DEFAULT_TICKERS

# ---------------------------------------------------------------------------
# SEC EDGAR constants
# ---------------------------------------------------------------------------
_SEC_USER_AGENT = "FinancialSentimentAnalysis/1.0 (contact@example.com)"
_REQUEST_TIMEOUT = 15

# Ticker → CIK (zero-padded to 10 digits)
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


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

def resolve_cik(ticker: str) -> Optional[str]:
    """Return the 10-digit CIK for *ticker* (hardcoded or SEC lookup)."""
    cik = _TICKER_TO_CIK.get(ticker.upper())
    if cik:
        return cik

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _SEC_USER_AGENT})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                _TICKER_TO_CIK[ticker.upper()] = cik
                return cik
    except Exception as exc:
        print(f"  [WARN] CIK lookup failed for {ticker}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Fetch filings from EDGAR
# ---------------------------------------------------------------------------

def fetch_edgar_8k(
    ticker: str,
    max_filings: int = 40,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch 8-K filings for *ticker* from SEC EDGAR.

    Returns a DataFrame matching the project's news CSV schema.
    """
    cik = resolve_cik(ticker)
    if not cik:
        print(f"  [SKIP] No CIK for {ticker}")
        return pd.DataFrame()

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _SEC_USER_AGENT})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"  [ERROR] EDGAR fetch failed for {ticker}: {exc}")
        return pd.DataFrame()

    company_name = data.get("name", ticker)
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    primary_descs = recent.get("primaryDocDescription", [])
    items_list = recent.get("items", [])

    # Parse date filters
    dt_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    dt_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    rows: List[dict] = []
    for i, form in enumerate(forms):
        if form not in ("8-K", "8-K/A"):
            continue

        filing_date_str = dates[i] if i < len(dates) else ""
        if not filing_date_str:
            continue

        try:
            filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Filter by date range
        if dt_start and filing_dt < dt_start:
            continue
        if dt_end and filing_dt > dt_end:
            continue

        if len(rows) >= max_filings:
            break

        accession = accessions[i].replace("-", "") if i < len(accessions) else ""
        accession_dashed = accessions[i] if i < len(accessions) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        items_raw = items_list[i] if i < len(items_list) else ""

        # Build human-readable title from item numbers
        item_numbers = [s.strip() for s in items_raw.split(",") if s.strip()]
        item_descs = [_8K_ITEM_DESCRIPTIONS.get(n, n) for n in item_numbers]

        if item_descs:
            title = f"{company_name} — SEC 8-K: {'; '.join(item_descs)}"
        else:
            desc = primary_descs[i] if i < len(primary_descs) else "Current Report"
            title = f"{company_name} — SEC 8-K Filing: {desc}"

        summary = (
            f"SEC {form} filing for {company_name}. "
            f"Items: {', '.join(item_descs) if item_descs else 'N/A'}. "
            f"Filed {filing_date_str}."
        )

        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik.lstrip('0')}/{accession}/{primary_doc}"
        ) if accession and primary_doc else ""

        # Use 06:00 UTC as published_at (pre-market) — SEC filings are
        # typically available before market open.
        published_at = filing_dt.replace(hour=6, minute=0, tzinfo=timezone.utc)

        rows.append({
            "title": title,
            "summary": summary,
            "source": "SEC EDGAR",
            "url": filing_url,
            "published_at": published_at.isoformat(),
            "authors": "",
            "overall_sentiment_score": None,  # filled by FinBERT below
            "overall_sentiment_label": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None,
            "ticker": ticker,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Score with FinBERT
# ---------------------------------------------------------------------------

def score_with_finbert(df: pd.DataFrame) -> pd.DataFrame:
    """Score each filing title with FinBERT and fill sentiment columns."""
    if df.empty:
        return df

    from src.models.sentiment_analyzer import SentimentAnalyzer

    model_path = _PROJECT_ROOT / "models" / "saved_models" / "finbert_finetuned"
    if model_path.exists():
        analyzer = SentimentAnalyzer.load(model_path)
    else:
        analyzer = SentimentAnalyzer()

    titles = df["title"].fillna("").tolist()
    preds = analyzer.predict_batch(titles, batch_size=16, show_progress=True)

    # Map FinBERT continuous score to Alpha Vantage-style labels
    def _score_to_label(score: float) -> str:
        if score >= 0.35:
            return "Bullish"
        elif score >= 0.15:
            return "Somewhat-Bullish"
        elif score <= -0.35:
            return "Bearish"
        elif score <= -0.15:
            return "Somewhat-Bearish"
        else:
            return "Neutral"

    df = df.copy()
    df["overall_sentiment_score"] = [p["score"] for p in preds]
    df["overall_sentiment_label"] = [_score_to_label(p["score"]) for p in preds]
    # For EDGAR, ticker sentiment ≈ overall (the filing IS about the company)
    df["ticker_sentiment_score"] = df["overall_sentiment_score"]
    df["ticker_sentiment_label"] = df["overall_sentiment_label"]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR 8-K filings for XGBoost training data"
    )
    parser.add_argument(
        "--ticker", nargs="+", default=None,
        help="Specific tickers to download (default: all 19)"
    )
    parser.add_argument(
        "--max-filings", type=int, default=40,
        help="Max 8-K filings per ticker (default: 40)"
    )
    parser.add_argument(
        "--start-date", default=DEFAULT_START,
        help=f"Start date for filtering (default: {DEFAULT_START})"
    )
    parser.add_argument(
        "--end-date", default=DEFAULT_END,
        help=f"End date for filtering (default: {DEFAULT_END})"
    )
    parser.add_argument(
        "--no-score", action="store_true",
        help="Skip FinBERT scoring (save time, fill scores later)"
    )
    args = parser.parse_args()

    tickers = args.ticker or DEFAULT_TICKERS
    out_dir = _PROJECT_ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"SEC EDGAR 8-K Downloader")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Date range: {args.start_date} → {args.end_date}")
    print(f"  Max filings/ticker: {args.max_filings}")
    print(f"{'=' * 60}\n")

    all_frames: List[pd.DataFrame] = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Fetching {ticker} ...")
        df = fetch_edgar_8k(
            ticker,
            max_filings=args.max_filings,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if df.empty:
            print(f"  → 0 filings in date range\n")
            continue

        print(f"  → {len(df)} filings found")
        all_frames.append(df)

        # Polite delay (SEC allows 10 req/sec but let's be courteous)
        time.sleep(0.2)

    if not all_frames:
        print("\n[DONE] No filings found. Nothing to save.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Score with FinBERT
    if not args.no_score:
        print(f"\n[SCORING] Running FinBERT on {len(combined)} filing titles ...")
        combined = score_with_finbert(combined)

    # Save per-ticker CSVs (matching the naming convention)
    total_saved = 0
    for ticker in combined["ticker"].unique():
        tk_df = combined[combined["ticker"] == ticker].reset_index(drop=True)
        fname = f"edgar_{ticker}_{args.start_date}_to_{args.end_date}.csv"
        out_path = out_dir / fname
        tk_df.to_csv(out_path, index=False)
        print(f"  Saved {len(tk_df):3d} filings → {out_path.name}")
        total_saved += len(tk_df)

    # Print sentiment summary
    scores = combined["overall_sentiment_score"].dropna()
    if len(scores):
        pos = (scores > 0.15).sum()
        neg = (scores < -0.15).sum()
        neu = len(scores) - pos - neg
        print(f"\n{'=' * 60}")
        print(f"  Total: {total_saved} filings across {combined['ticker'].nunique()} tickers")
        print(f"  Sentiment: positive={pos}, neutral={neu}, negative={neg}")
        print(f"  Avg score: {scores.mean():.4f} (compare: news avg ≈ +0.20)")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
