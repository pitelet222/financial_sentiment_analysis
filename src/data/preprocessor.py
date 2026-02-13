"""
Text Preprocessing Module for Financial News Sentiment Analysis
================================================================

This module provides a pipeline of text-cleaning functions tailored to
financial news articles.  Each function handles one concern and can be
used independently or composed via ``preprocess_text()`` /
``preprocess_dataframe()`` for batch processing.

Typical usage
-------------
>>> from src.data.preprocessor import preprocess_dataframe
>>> import pandas as pd
>>> df = pd.read_csv("data/raw/news_AAPL_2025-11-01_to_2026-02-13.csv")
>>> df = preprocess_dataframe(df, text_columns=["title", "summary"])
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Comprehensive list of stock tickers used in this project.
# Extend as needed when adding more tickers to the data pipeline.
PROJECT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
                               "NVDA", "JPM", "BAC", "GS"]

# Common financial abbreviations that should NOT be treated as tickers.
_FALSE_TICKER_WORDS = {
    "CEO", "CFO", "CTO", "COO", "IPO", "ETF", "SEC", "FTC", "FDA", "GDP",
    "EPS", "NYSE", "USA", "USD", "AI", "UK", "EU", "API", "ESG", "P",
    "Q1", "Q2", "Q3", "Q4", "FY", "YOY", "QOQ", "B2B", "B2C",
}

# Regex patterns compiled once for performance
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
_TICKER_PATTERN = re.compile(
    r"(?<!\w)"           # not preceded by a word character
    r"\$?[A-Z]{1,5}"    # optional $ + 1-5 uppercase letters
    r"(?!\w)",           # not followed by a word character
)
_MULTIPLE_SPACES = re.compile(r"\s{2,}")
_SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s.,;:!?'\"%()\-$/]")


# =========================================================================
# Individual cleaning functions
# =========================================================================

def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare ``www.`` links from *text*.

    Examples
    --------
    >>> remove_urls("Check https://example.com for details.")
    'Check  for details.'
    """
    return _URL_PATTERN.sub("", text)


def remove_html_tags(text: str) -> str:
    """Strip any residual HTML/XML tags (e.g. ``<b>``, ``<br/>``)."""
    return _HTML_TAG_PATTERN.sub("", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from *text*."""
    return _EMAIL_PATTERN.sub("", text)


def remove_special_characters(text: str, keep_financial: bool = True) -> str:
    """Remove special characters while preserving readable punctuation.

    Parameters
    ----------
    text : str
        Input text to clean.
    keep_financial : bool, optional
        When *True* (default), keep ``$``, ``%``, and ``/`` which are
        common in financial text (e.g. "$150", "52-week", "Q4/2025").

    Returns
    -------
    str
        Cleaned text with only alphanumerics, spaces, and basic
        punctuation retained.
    """
    if keep_financial:
        return _SPECIAL_CHARS.sub("", text)
    # Stricter: remove everything except letters, digits, spaces, and
    # basic sentence punctuation.
    return re.sub(r"[^a-zA-Z0-9\s.,;:!?'\"\-]", "", text)


def normalize_text(text: str, *, lowercase: bool = False) -> str:
    """Normalize whitespace and Unicode characters.

    Steps performed:
    1. Convert Unicode characters to their closest ASCII equivalents
       (e.g. curly quotes → straight quotes, em-dash → hyphen).
    2. Collapse multiple consecutive spaces/tabs/newlines into a single
       space.
    3. Strip leading/trailing whitespace.

    Parameters
    ----------
    text : str
        Raw text to normalize.
    lowercase : bool, optional
        If *True*, convert the entire string to lowercase.  Defaults to
        *False* because casing carries meaning for sentiment analysis
        (e.g. "PLUNGING" vs "plunging") and is needed for ticker
        extraction.

    Returns
    -------
    str
        Normalized text.
    """
    # Decompose Unicode → ASCII approximation, drop non-ASCII leftovers
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Collapse whitespace
    text = _MULTIPLE_SPACES.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def extract_tickers(text: str, known_tickers: Optional[List[str]] = None) -> List[str]:
    """Extract stock ticker symbols mentioned in *text*.

    Uses a regex to find uppercase 1–5 letter tokens (optionally
    preceded by ``$``), then filters out common false positives (e.g.
    "CEO", "IPO", "SEC").

    Parameters
    ----------
    text : str
        Text to scan for tickers.
    known_tickers : list of str, optional
        If provided, only return tickers that appear in this whitelist.
        Otherwise return all candidates that pass the false-positive
        filter.

    Returns
    -------
    list of str
        Deduplicated list of ticker symbols found, in order of first
        appearance.  The leading ``$`` is stripped.

    Examples
    --------
    >>> extract_tickers("Apple (AAPL) and $MSFT beat estimates.")
    ['AAPL', 'MSFT']
    >>> extract_tickers("The CEO said GDP is rising.", known_tickers=["AAPL"])
    []
    """
    candidates = _TICKER_PATTERN.findall(text)
    # Strip leading $ and deduplicate while preserving order
    seen: set[str] = set()
    tickers: List[str] = []
    for raw in candidates:
        symbol = raw.lstrip("$")
        if len(symbol) < 1:
            continue
        if symbol in seen:
            continue
        if symbol in _FALSE_TICKER_WORDS:
            continue
        if known_tickers is not None and symbol not in known_tickers:
            continue
        seen.add(symbol)
        tickers.append(symbol)
    return tickers


def remove_numbers(text: str, keep_percentages: bool = True) -> str:
    """Remove standalone numbers while optionally keeping percentages.

    Parameters
    ----------
    text : str
        Input text.
    keep_percentages : bool, optional
        When *True* (default), preserve patterns like "5%" or "3.2%".

    Returns
    -------
    str
        Text with numbers removed.
    """
    if keep_percentages:
        # Remove numbers that are NOT followed by %
        return re.sub(r"\b\d+\.?\d*(?!%)\b", "", text)
    return re.sub(r"\b\d+\.?\d*%?\b", "", text)


def truncate_text(text: str, max_chars: int = 1000) -> str:
    """Truncate text to *max_chars* characters on a word boundary.

    Useful for limiting input length before feeding into transformer
    models that have token limits.

    Parameters
    ----------
    text : str
        Input text.
    max_chars : int, optional
        Maximum number of characters (default 1000).

    Returns
    -------
    str
        Truncated text ending at the last complete word within the
        limit, with "..." appended if truncation occurred.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "..."


# =========================================================================
# Composite pipeline
# =========================================================================

def preprocess_text(
    text: str,
    *,
    lowercase: bool = False,
    remove_nums: bool = False,
    max_length: Optional[int] = None,
) -> str:
    """Apply the full cleaning pipeline to a single string.

    Pipeline order:
    1. Remove URLs
    2. Remove HTML tags
    3. Remove email addresses
    4. Remove special characters (keeping financial symbols)
    5. Normalize whitespace and Unicode
    6. (Optional) Remove numbers
    7. (Optional) Truncate to *max_length*

    Parameters
    ----------
    text : str
        Raw text to clean.
    lowercase : bool, optional
        Convert to lowercase after cleaning (default *False*).
    remove_nums : bool, optional
        Strip standalone numbers (default *False*).
    max_length : int, optional
        If set, truncate result to this many characters.

    Returns
    -------
    str
        Cleaned text ready for sentiment analysis.

    Examples
    --------
    >>> preprocess_text("Check https://x.com — AAPL is up 5%!")
    'Check  AAPL is up 5%!'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_emails(text)
    text = remove_special_characters(text, keep_financial=True)
    text = normalize_text(text, lowercase=lowercase)

    if remove_nums:
        text = remove_numbers(text)

    if max_length is not None:
        text = truncate_text(text, max_chars=max_length)

    # Final whitespace cleanup after all transformations
    text = _MULTIPLE_SPACES.sub(" ", text).strip()

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    *,
    lowercase: bool = False,
    remove_nums: bool = False,
    max_length: Optional[int] = None,
    extract_ticker_col: bool = False,
    known_tickers: Optional[List[str]] = None,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Apply text preprocessing to one or more columns of a DataFrame.

    For each column in *text_columns*, a new column
    ``<original_name>_clean`` is added with the cleaned text, leaving
    the original column intact for reference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw news data.
    text_columns : list of str, optional
        Columns to clean.  Defaults to ``["title", "summary"]``.
    lowercase : bool, optional
        Convert cleaned text to lowercase (default *False*).
    remove_nums : bool, optional
        Remove standalone numbers (default *False*).
    max_length : int, optional
        Truncate cleaned text to this many characters.
    extract_ticker_col : bool, optional
        If *True*, add a ``mentioned_tickers`` column with tickers
        extracted from the cleaned summary text.
    known_tickers : list of str, optional
        Whitelist for ticker extraction (see :func:`extract_tickers`).
    drop_empty : bool, optional
        Drop rows where all cleaned text columns are empty after
        preprocessing (default *True*).

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with ``*_clean`` columns appended.
    """
    if text_columns is None:
        text_columns = ["title", "summary"]

    df = df.copy()

    for col in text_columns:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        clean_col = f"{col}_clean"
        df[clean_col] = df[col].fillna("").apply(
            lambda t: preprocess_text(
                t,
                lowercase=lowercase,
                remove_nums=remove_nums,
                max_length=max_length,
            )
        )

    # Optionally extract tickers from the cleaned summary
    if extract_ticker_col:
        source_col = "summary_clean" if "summary_clean" in df.columns else "title_clean"
        df["mentioned_tickers"] = df[source_col].apply(
            lambda t: extract_tickers(t, known_tickers=known_tickers)
        )

    # Drop rows where cleaning left everything blank
    if drop_empty:
        clean_cols = [f"{c}_clean" for c in text_columns if f"{c}_clean" in df.columns]
        mask = df[clean_cols].apply(lambda row: all(v == "" for v in row), axis=1)
        dropped = mask.sum()
        if dropped > 0:
            df = df[~mask].reset_index(drop=True)
            print(f"Dropped {dropped} rows with empty text after cleaning.")

    return df


# =========================================================================
# CLI entry point (for quick testing)
# =========================================================================

if __name__ == "__main__":
    # Quick smoke test with sample text
    sample = (
        'Apple (AAPL) stock — up 5%! Check https://finance.yahoo.com '
        '<b>Bold</b> contact info@example.com for más détails™.'
    )
    print("Original :", sample)
    print("Cleaned  :", preprocess_text(sample))
    print("Tickers  :", extract_tickers(sample))
    print("Lowercase:", preprocess_text(sample, lowercase=True))
