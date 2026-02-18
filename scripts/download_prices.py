"""
download_prices.py
==================
Script to download historical stock price data using Yahoo Finance (yfinance).

This script fetches Open, High, Low, Close, Volume (OHLCV) data for one or more
stock tickers and saves each as a CSV file in the `data/raw/` directory.

Usage:
    python scripts/download_prices.py

    You can also import the functions in a notebook or another script:
        from scripts.download_prices import download_stock_data

Dependencies:
    - yfinance: Yahoo Finance API wrapper (no API key needed)
    - pandas: Data manipulation

Output:
    CSV files saved to: data/raw/prices_<TICKER>_<START>_to_<END>.csv
    Columns: Date, Open, High, Low, Close, Volume
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default tickers to download — add or remove as needed
DEFAULT_TICKERS = ["AAPL", "MSFT"]

# Default date range — format: "YYYY-MM-DD"
DEFAULT_START_DATE = "2025-02-13"
DEFAULT_END_DATE = "2026-02-13"

# Where to save the raw price data (relative to project root)
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock prices from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL", "MSFT", "GOOGL").
    start_date : str
        Start date in "YYYY-MM-DD" format (inclusive).
    end_date : str
        End date in "YYYY-MM-DD" format (inclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
        Returns an empty DataFrame if the download fails.

    Example
    -------
    >>> df = download_stock_data("AAPL", "2025-11-01", "2026-02-13")
    >>> print(df.head())
    """
    print(f"[INFO] Downloading price data for {ticker} "
          f"from {start_date} to {end_date} ...")

    try:
        # yf.download returns a DataFrame indexed by Date
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            print(f"[WARNING] No data returned for {ticker}. "
                  "Check the ticker symbol and date range.")
            return pd.DataFrame()

        # Reset index so 'Date' becomes a regular column instead of the index
        df = df.reset_index()

        # Keep only the columns we need
        columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[[col for col in columns_to_keep if col in df.columns]]

        print(f"[INFO] Downloaded {len(df)} rows for {ticker}.")
        return df

    except Exception as e:
        print(f"[ERROR] Failed to download data for {ticker}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_prices_to_csv(df: pd.DataFrame, ticker: str,
                       start_date: str, end_date: str,
                       output_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Save a price DataFrame to a CSV file in the raw data directory.

    Parameters
    ----------
    df : pd.DataFrame
        Price data returned by download_stock_data().
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
    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive filename: prices_AAPL_2025-11-01_to_2026-02-13.csv
    filename = f"prices_{ticker}_{start_date}_to_{end_date}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    print(f"[INFO] Saved prices to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """
    Download stock prices for all tickers in DEFAULT_TICKERS and save them
    as individual CSV files in data/raw/.
    """
    print("=" * 60)
    print("  Financial Sentiment Analysis — Price Data Downloader")
    print("=" * 60)
    print(f"  Tickers : {DEFAULT_TICKERS}")
    print(f"  Period  : {DEFAULT_START_DATE} → {DEFAULT_END_DATE}")
    print(f"  Output  : {RAW_DATA_DIR}")
    print("=" * 60)

    for ticker in DEFAULT_TICKERS:
        df = download_stock_data(ticker, DEFAULT_START_DATE, DEFAULT_END_DATE)

        if not df.empty:
            save_prices_to_csv(df, ticker, DEFAULT_START_DATE, DEFAULT_END_DATE)

    print("\n[DONE] Price download complete.")


if __name__ == "__main__":
    main()