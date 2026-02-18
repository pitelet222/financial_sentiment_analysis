# Financial Sentiment Analysis — Complete Project Guide

> **Version**: Phase 1 (multi-ticker, multi-horizon)
> **Python**: 3.13.9 · **Framework stack**: PyTorch, Transformers, XGBoost, Streamlit, Plotly

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Installation & Setup](#3-installation--setup)
4. [End-to-End Pipeline](#4-end-to-end-pipeline)
5. [Source Modules (`src/`)](#5-source-modules-src)
   - 5.1 [data/data_loader.py](#51-datadata_loaderpy)
   - 5.2 [data/news_fetcher.py](#52-datanews_fetcherpy)
   - 5.3 [data/preprocessor.py](#53-datapreprocessorpy)
   - 5.4 [models/sentiment_analyzer.py](#54-modelssentiment_analyzerpy)
   - 5.5 [models/return_predictor.py](#55-modelsreturn_predictorpy)
   - 5.6 [predictor.py](#56-predictorpy)
6. [Scripts (`scripts/`)](#6-scripts-scripts)
   - 6.1 [download_news.py](#61-download_newspy)
   - 6.2 [download_prices.py](#62-download_pricespy)
   - 6.3 [train_xgboost.py](#63-train_xgboostpy)
   - 6.4 [tune_xgboost.py](#64-tune_xgboostpy)
   - 6.5 [train_multihorizon.py](#65-train_multihorizonpy)
7. [Dashboard (`dashboard/app.py`)](#7-dashboard-dashboardapppy)
8. [Notebooks (`notebooks/`)](#8-notebooks-notebooks)
9. [Models & Artifacts](#9-models--artifacts)
10. [Data Files](#10-data-files)
11. [Feature Catalogue](#11-feature-catalogue)
12. [Model Performance Summary](#12-model-performance-summary)
13. [Configuration & Environment](#13-configuration--environment)
14. [How to Extend](#14-how-to-extend)

---

## 1. Project Overview

This project builds a **financial sentiment analysis system** that:

1. **Collects** financial news articles (Alpha Vantage API) and stock prices (Yahoo Finance) for **19 tickers** across 6 sectors.
2. **Scores** each article with the **ProsusAI/FinBERT** transformer model (fine-tuned checkpoint included).
3. **Merges** sentiment aggregates with daily OHLCV price data and **10 technical indicators** (RSI, MACD, Bollinger Bands, ATR, etc.) plus the **VIX** index.
4. **Predicts** return direction at three horizons (**1-day, 5-day, 20-day**) using **XGBoost** classifiers evaluated with strict walk-forward validation.
5. **Visualises** everything in a **10-panel Streamlit dashboard** with live news scoring, rule-based alert simulation, and multi-horizon prediction cards.

### Tickers (19)

| Sector       | Tickers                      |
|--------------|------------------------------|
| Tech         | AAPL, MSFT, GOOGL, AMZN, NVDA, META |
| Finance      | JPM, GS, BAC                |
| Healthcare   | JNJ, UNH, PFE               |
| Consumer     | TSLA, WMT, KO               |
| Energy       | XOM, CVX                     |
| Industrials  | CAT, BA                      |

### Date Range

`2025-02-13` to `2026-02-13` (1 year, ~251 trading days).

---

## 2. Directory Structure

```
financial-sentiment-analysis/
├── configs/                          # (reserved for YAML/JSON config files)
├── dashboard/
│   └── app.py                        # Streamlit dashboard (1,691 lines, 10 panels)
├── data/
│   ├── external/                     # (reserved for third-party datasets)
│   ├── interim/                      # (reserved for intermediate transforms)
│   ├── processed/
│   │   └── merged_2025-..._to_2026-...csv  # Final merged dataset (~4,769 rows × 40 cols)
│   └── raw/
│       ├── news_AAPL_*.csv           # Alpha Vantage news for AAPL
│       ├── news_MSFT_*.csv           # Alpha Vantage news for MSFT
│       ├── prices_<TICKER>_*.csv     # yfinance OHLCV per ticker (19 files + VIX)
│       └── ...
├── deployment/
│   ├── docker/                       # (reserved)
│   └── kubernetes/                   # (reserved)
├── docs/
│   └── PROJECT_GUIDE.md              # ← You are here
├── logs/                             # (reserved for training/run logs)
├── models/
│   ├── checkpoints/                  # (reserved for training checkpoints)
│   └── saved_models/
│       ├── finbert_finetuned/        # Fine-tuned FinBERT (model + tokenizer + meta)
│       ├── xgboost_return/           # Single-horizon XGBoost (1d, Optuna-tuned)
│       ├── xgboost_return_1d/        # Multi-horizon: daily model
│       ├── xgboost_return_5d/        # Multi-horizon: weekly model
│       └── xgboost_return_20d/       # Multi-horizon: monthly model
├── notebooks/
│   ├── evaluation/
│   │   ├── 01_news_exploration.ipynb
│   │   ├── 02_price_exploration.ipynb
│   │   ├── 03_baseline_sentiment.ipynb
│   │   └── 04_finbert_price_evaluation.ipynb
│   ├── exploratory/
│   └── modeling/
│       └── 01_finbert_baseline.ipynb
├── reports/
│   ├── figures/                      # (reserved for exported charts)
│   └── metrics/
│       ├── tuning_results.json       # Optuna tuning report (150 trials)
│       ├── xgboost_walkforward_metrics.json
│       └── multihorizon_summary.json # 1d/5d/20d comparison
├── requirements.txt                  # Python dependencies
├── scripts/
│   ├── download_news.py              # Alpha Vantage news downloader
│   ├── download_prices.py            # yfinance price downloader
│   ├── train_xgboost.py              # Single-horizon XGBoost training
│   ├── tune_xgboost.py               # Optuna hyperparameter optimization
│   └── train_multihorizon.py         # Multi-horizon (1d/5d/20d) training
├── src/
│   ├── __init__.py
│   ├── api/
│   │   └── __init__.py               # (reserved for REST API layer)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Central data pipeline (896 lines)
│   │   ├── news_fetcher.py           # Live news via RSS + yfinance (319 lines)
│   │   └── preprocessor.py           # Text cleaning pipeline (310 lines)
│   ├── features/
│   │   └── __init__.py               # (reserved for feature engineering)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py     # FinBERT wrapper (839 lines)
│   │   └── return_predictor.py       # XGBoost predictor (669 lines)
│   └── visualization/
│       └── __init__.py               # (reserved)
├── tests/
│   ├── __init__.py
│   ├── integration/
│   └── unit/
├── .env                              # ALPHA_VANTAGE_KEY=your_key_here
├── .gitignore
└── README.md
```

---

## 3. Installation & Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd financial-sentiment-analysis

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
# Create a .env file in the project root:
echo ALPHA_VANTAGE_KEY=your_key_here > .env

# 5. Download data
python scripts/download_prices.py    # 19 tickers + VIX
python scripts/download_news.py      # AAPL & MSFT news (Alpha Vantage)

# 6. Build merged dataset (run once, saves to data/processed/)
python -c "from src.data.data_loader import load_merged_dataset; load_merged_dataset(save_processed=True)"

# 7. Train models
python scripts/train_multihorizon.py   # 1d, 5d, 20d XGBoost models

# 8. Launch dashboard
streamlit run dashboard/app.py
```

### Dependencies (`requirements.txt`)

| Category | Packages |
|----------|----------|
| Data | `numpy`, `pandas` |
| ML | `torch`, `transformers`, `scipy`, `xgboost`, `scikit-learn` |
| Financial data | `yfinance` |
| News data | `requests`, `feedparser` |
| Environment | `python-dotenv` |
| Dashboard | `streamlit` |
| Tuning | `optuna` (optional, for hyperparameter search) |
| Charts | `plotly` (installed with streamlit) |

---

## 4. End-to-End Pipeline

The data flows through **9 sequential stages** inside `load_merged_dataset()`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     END-TO-END DATA PIPELINE                        │
├──────────┬──────────────────────────────────────────────────────────┤
│ Stage 1  │ load_all_news()        → Load raw news CSVs (AAPL+MSFT) │
│ Stage 2  │ load_all_prices()      → Load OHLCV for 19 tickers      │
│ Stage 3  │ add_session_column()   → Tag pre/market/after/overnight  │
│ Stage 4  │ assign_trading_day()   → Map articles → effective date   │
│ Stage 5  │ aggregate_sentiment()  → 1 row per date+ticker           │
│ Stage 6  │ add_price_features()   → Return, range, gap, volume      │
│ Stage 7  │ add_technical_indicators() → RSI, MACD, Bollinger, ATR…  │
│ Stage 8  │ merge + add_rolling_sentiment() + load_vix()             │
│ Stage 9  │ save_processed (optional) → CSV output                   │
└──────────┴──────────────────────────────────────────────────────────┘
                              │
                              ▼
                  merged DataFrame (4,769 rows × 40 cols)
                              │
               ┌──────────────┤──────────────┐
               ▼              ▼              ▼
         ReturnPredictor  ReturnPredictor  ReturnPredictor
           (horizon=1d)    (horizon=5d)    (horizon=20d)
               │              │              │
               ▼              ▼              ▼
           model.json     model.json     model.json
           meta.json      meta.json      meta.json
               │              │              │
               └──────────────┤──────────────┘
                              ▼
                    Streamlit Dashboard
                      (10 panels)
```

### Pipeline Detail

| Stage | Function | Input | Output | Key Logic |
|-------|----------|-------|--------|-----------|
| 1 | `load_all_news()` | Raw news CSVs | DataFrame (7,993 rows) | Concatenates per-ticker CSVs; gracefully handles missing files |
| 2 | `load_all_prices()` | Raw price CSVs | DataFrame (19 tickers × ~251 days) | Auto-adds ticker column; `skiprows=[1]` for yfinance header |
| 3 | `add_session_column()` | News DataFrame | +`market_session` column | Classifies by hour into 4 time windows |
| 4 | `assign_trading_day()` | News + trading dates | +`trading_day` column | Pre/market→same day; after/overnight→next trading day |
| 5 | `aggregate_sentiment()` | Per-article rows | 1 row/date/ticker | Computes: article_count, avg/std/min/max/range sentiment, pct_positive, pct_negative, session counts |
| 6 | `add_price_features()` | Prices | +5 columns | daily_return, return_direction, intraday_range, gap_pct, volume_change |
| 7 | `add_technical_indicators()` | Prices | +10 columns | RSI-14, MACD (line/signal/histogram), Bollinger %B, ATR-14, 52w high/low distance, volume z-score + 5d/20d forward return targets |
| 8 | `merge_sentiment_with_prices()` + `add_rolling_sentiment()` + `load_vix()` | Sentiment agg + prices + VIX | Joined DataFrame | Left join on date+ticker; rolling 3d/5d windows; VIX_close column |
| 9 | Optional CSV save | Merged DF | `data/processed/*.csv` | — |

---

## 5. Source Modules (`src/`)

### 5.1 `data/data_loader.py`

**Purpose**: Central data-loading, merging, and feature-engineering pipeline.
**Lines**: 896

#### Constants

| Name | Value | Description |
|------|-------|-------------|
| `_PROJECT_ROOT` | Auto-detected | Three levels up from this file |
| `RAW_DATA_DIR` | `data/raw/` | Where raw CSVs are stored |
| `PROCESSED_DATA_DIR` | `data/processed/` | Where merged CSVs are saved |
| `DEFAULT_TICKERS` | 19 tickers (list) | AAPL, MSFT, GOOGL, ... BA |
| `DEFAULT_START` | `"2025-02-13"` | Dataset start date |
| `DEFAULT_END` | `"2026-02-13"` | Dataset end date |
| `MARKET_OPEN_HOUR/MINUTE` | 9:30 | US market open (ET) |
| `MARKET_CLOSE_HOUR/MINUTE` | 16:00 | US market close (ET) |

#### Functions (ordered low-level → high-level)

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `load_news_csv(filepath)` | `filepath: Path` | `DataFrame` | Load one news CSV, parse `published_at` to datetime |
| `load_prices_csv(filepath)` | `filepath: Path` | `DataFrame` | Load one price CSV with `skiprows=[1]` (yfinance header) |
| `load_all_news(tickers, start, end, data_dir)` | Defaults to project constants | `DataFrame` | Concatenates all `news_<TICKER>_*.csv` files; `warnings.warn` for missing |
| `load_all_prices(tickers, start, end, data_dir)` | Defaults to project constants | `DataFrame` | Concatenates all `prices_<TICKER>_*.csv`; adds `ticker` column |
| `classify_market_session(dt)` | `dt: datetime` | `str` | Returns `"pre_market"`, `"market_hours"`, `"after_hours"`, or `"overnight"` |
| `add_session_column(news)` | `news: DataFrame` | `DataFrame` | Adds `market_session` column via `classify_market_session` |
| `assign_trading_day(news, trading_dates)` | `news: DataFrame`, `trading_dates: list` | `DataFrame` | Maps articles to effective trading day (accounts for weekends/holidays) |
| `aggregate_sentiment(news, group_cols)` | `news: DataFrame`, `group_cols: list` | `DataFrame` | Produces: `article_count`, `avg_overall_sentiment`, `avg_ticker_sentiment`, `sentiment_std`, `sentiment_min`, `sentiment_max`, `sentiment_range`, `pct_positive`, `pct_negative`, per-session counts |
| `aggregate_by_session(news, sessions)` | `news: DataFrame`, `sessions: list` | `DataFrame` | Filtered aggregation for specific market sessions |
| `add_price_features(prices)` | `prices: DataFrame` | `DataFrame` | Adds: `daily_return` (% change), `return_direction` (1=up/0=down), `intraday_range` (High-Low)/Close, `gap_pct` (Open vs prev Close), `volume_change` |
| `add_technical_indicators(prices)` | `prices: DataFrame` | `DataFrame` | Adds per-ticker: `rsi_14`, `macd`, `macd_signal`, `macd_histogram`, `bb_pct_b`, `atr_14`, `distance_52w_high`, `distance_52w_low`, `volume_zscore`, `return_5d` (forward target), `return_20d` (forward target) |
| `load_vix(start, end, data_dir)` | Defaults | `DataFrame` | Loads VIX price CSV, returns `date` + `VIX_close` |
| `merge_sentiment_with_prices(sentiment_agg, prices)` | Two DataFrames | `DataFrame` | Left join on `date` + `ticker`; fills missing sentiment with 0/NaN |
| `add_rolling_sentiment(df, windows)` | `df: DataFrame`, `windows: [3, 5]` | `DataFrame` | Adds `sentiment_rolling_3d`, `sentiment_rolling_5d` (per-ticker rolling mean) |
| `load_merged_dataset(tickers, start, end, save_processed)` | All defaulted | `DataFrame` | **MAIN ENTRY POINT** — executes the full 9-stage pipeline |

---

### 5.2 `data/news_fetcher.py`

**Purpose**: Fetch live financial news from free RSS feeds and yfinance, with disk caching.
**Lines**: 319

#### Constants

| Name | Value | Description |
|------|-------|-------------|
| `CACHE_DIR` | `data/interim/live_news_cache/` | Disk cache directory |
| `DEFAULT_CACHE_TTL` | `3600` (1 hour) | Cache expiry in seconds |
| `_RSS_FEEDS` | dict | Google News and Yahoo Finance RSS URLs (with `{ticker}` placeholder) |

#### Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `_parse_feed(url, source_name, ticker)` | URL string, source label, ticker | `list[dict]` | Parses a single RSS feed via `feedparser`; extracts title, summary, url, published_at, source, ticker |
| `_fetch_yfinance_news(ticker)` | `ticker: str` | `list[dict]` | Pulls news from `yfinance.Ticker(ticker).news`; normalises to same schema |
| `fetch_live_news(ticker, feeds, max_articles, include_yfinance)` | `ticker: str`, rest defaulted | `DataFrame` | Combines RSS + yfinance news, **deduplicates by title**, caps at `max_articles` (60). Returns DataFrame matching raw news CSV schema |
| `fetch_and_cache(ticker, cache_dir, ttl)` | `ticker: str`, rest defaulted | `DataFrame` | Transparent disk caching layer: checks file freshness → returns cached or fetches fresh, saves as CSV |

**Data flow**: `fetch_and_cache()` → `fetch_live_news()` → `_parse_feed()` + `_fetch_yfinance_news()`

---

### 5.3 `data/preprocessor.py`

**Purpose**: Text-cleaning pipeline for financial news articles.
**Lines**: 310

#### Constants

| Name | Value | Description |
|------|-------|-------------|
| `PROJECT_TICKERS` | 19 tickers | Known ticker symbols (for extraction) |
| `_FALSE_TICKER_WORDS` | `{"CEO", "CFO", "IPO", "ETF", "SEC", "GDP", ...}` | Words that look like tickers but aren't |
| Compiled regex patterns | Various | `URL_PATTERN`, `HTML_PATTERN`, `EMAIL_PATTERN`, `TICKER_PATTERN` |

#### Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `remove_urls(text)` | `text: str` | `str` | Strip HTTP/HTTPS URLs |
| `remove_html_tags(text)` | `text: str` | `str` | Strip `<tag>` elements |
| `remove_emails(text)` | `text: str` | `str` | Strip email addresses |
| `remove_special_characters(text, keep_financial)` | `text: str`, `keep_financial: bool=True` | `str` | Remove non-alphanumeric chars; optionally preserves `$`, `%`, `.`, `+`, `-` |
| `normalize_text(text, lowercase)` | `text: str`, `lowercase: bool=False` | `str` | Collapse whitespace, optional lowercasing |
| `remove_numbers(text, keep_percentages)` | `text: str`, `keep_percentages: bool=True` | `str` | Remove digits; optionally keep `42%` patterns |
| `truncate_text(text, max_chars)` | `text: str`, `max_chars: int=1000` | `str` | Hard truncation (FinBERT has 512-token limit) |
| `extract_tickers(text, known_tickers)` | `text: str`, `known_tickers: set` | `list[str]` | Regex `\b[A-Z]{1,5}\b` + filter against `_FALSE_TICKER_WORDS` + optional whitelist |
| `preprocess_text(text, ...)` | `text: str` + booleans for each step | `str` | **7-step composite pipeline**: URLs → HTML → emails → special chars → normalize → numbers → truncate |
| `preprocess_dataframe(df, text_columns, ...)` | `df: DataFrame`, `text_columns: list` | `DataFrame` | Batch processing: adds `*_clean` columns, optional ticker extraction column |

---

### 5.4 `models/sentiment_analyzer.py`

**Purpose**: Production-ready wrapper around the ProsusAI/FinBERT transformer for sentiment scoring.
**Lines**: 839

#### Class: `SentimentAnalyzer`

##### Constructor

```python
SentimentAnalyzer(
    model_name: str = "ProsusAI/finbert",
    device: str | None = None,      # auto-detects GPU
    max_length: int = 512,
    label_map: dict = {0: "positive", 1: "negative", 2: "neutral"}
)
```

Loads the model and tokenizer from HuggingFace (or a local path). Automatically selects CUDA if available.

##### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `predict(text)` | `text: str` | `dict` | Single text → `{label, positive, negative, neutral, confidence, score}`. Score is `positive - negative` in [-1, +1] |
| `predict_batch(texts, batch_size, show_progress)` | `texts: list[str]`, `batch_size: int=16` | `list[dict]` | Batched inference; same output schema as `predict` per item |
| `predict_dataframe(df, text_columns)` | `df: DataFrame`, `text_columns: list[str]` | `DataFrame` | Adds columns: `finbert_{col}_label`, `finbert_{col}_score`, `finbert_{col}_conf`, `finbert_{col}_pos`, `finbert_{col}_neg`, `finbert_{col}_neu` |
| `aggregate_daily_sentiment(news_with_preds)` | `news: DataFrame` | `DataFrame` | FinBERT-specific daily rollup (article count, avg/std/min/max scores, pct distributions) |
| `evaluate_price_correlation(merged_df)` | `df: DataFrame` | `dict` | Computes Pearson/Spearman correlation, direction accuracy, lag analysis (t+0, t+1, t+2), and quintile return analysis |
| `fine_tune(train_df, val_df, text_col, label_col, epochs, lr, batch_size, warmup_ratio)` | Training config | `dict` | Fine-tunes FinBERT with AdamW optimizer + linear warmup schedule; optional validation loop; returns loss/accuracy history |
| `save(path)` | `path: str\|Path` | `None` | Saves model weights + tokenizer + `analyzer_meta.json` (label_map, model_name, max_length) |
| `load(path)` (classmethod) | `path: str\|Path` | `SentimentAnalyzer` | Reconstructs from saved artifacts |

##### Internal

| Item | Description |
|------|-------------|
| `_SentimentDataset` | PyTorch `Dataset` subclass for fine-tuning (tokenizes on-the-fly) |
| `_probs_to_result(probs, text)` | Converts softmax probabilities to the output dict |
| `_validate(model, dataloader, device)` | Validation loop returning avg loss + accuracy |

---

### 5.5 `models/return_predictor.py`

**Purpose**: XGBoost binary classifier for predicting return direction (up/down) at multiple horizons.
**Lines**: 669

#### Constants

| Name | Count | Contents |
|------|-------|----------|
| `SENTIMENT_FEATURES` | 9 | avg_overall_sentiment, avg_ticker_sentiment, sentiment_std, sentiment_range, pct_positive, pct_negative, article_count, sentiment_rolling_3d, sentiment_rolling_5d |
| `PRICE_FEATURES` | 4 | daily_return, intraday_range, gap_pct, volume_change |
| `TECHNICAL_FEATURES` | 10 | rsi_14, macd, macd_signal, macd_histogram, bb_pct_b, atr_14, distance_52w_high, distance_52w_low, volume_zscore, VIX_close |
| `ENGINEERED_FEATURES` | 6 | return_lag2, return_lag3, volatility_5d, avg_return_5d, sentiment_momentum, news_has_coverage |
| **Total features** | **29** | — |
| `TARGETS` | dict | `{"1d": "return_direction", "5d": "return_5d", "20d": "return_20d"}` |
| `MIN_TRAIN_DAYS` | 60 | Minimum training window for walk-forward |

#### Top-Level Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `prepare_features(df)` | `df: DataFrame` | `DataFrame` | **Lag all features by 1 day** (per ticker) to prevent lookahead bias; compute 6 engineered features: `return_lag2/3` (2nd/3rd day lags), `volatility_5d` (5-day std of returns), `avg_return_5d` (5-day mean), `sentiment_momentum` (rolling_3d − rolling_5d), `news_has_coverage` (1 if article_count > 0) |
| `get_feature_columns()` | — | `list[str]` | Returns the ordered list of all 29 feature column names |
| `walk_forward_split(df, min_train)` | `df: DataFrame`, `min_train: int` | `list[tuple]` | Generates expanding-window train/test index pairs over unique dates |

#### Class: `ReturnPredictor`

##### Constructor

```python
ReturnPredictor(
    horizon: str = "1d",          # "1d", "5d", or "20d"
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0.0,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 1.0,
    min_child_weight: int = 1,
    colsample_bylevel: float = 1.0,
)
```

Creates an XGBClassifier with the given hyperparameters. The `horizon` parameter selects the target column (`return_direction`, `return_5d`, or `return_20d`).

##### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `train(data_path, min_train_days, verbose)` | `data_path: str\|Path`, `min_train_days: int=60` | `dict` (metrics) | Full pipeline: load CSV → `prepare_features()` → `walk_forward_split()` → evaluate each fold → train final model on all data → compute feature importance + per-ticker breakdown |
| `predict(X)` | `X: DataFrame` | `dict` | Returns `{predictions: array, probabilities: array}` for a feature matrix |
| `predict_next_day(df, ticker)` | `df: DataFrame`, `ticker: str` | `dict` | End-to-end: filters to ticker → prepares features → uses last row → returns `{direction: "UP"/"DOWN", prob_up, prob_down, confidence, based_on_date}` |
| `save(directory)` | `directory: str\|Path` | `None` | Saves `model.json` (XGBoost), `meta.json` (metrics, horizon, params, feature importance), `walk_forward_results.csv` |
| `load(directory)` (classmethod) | `directory: str\|Path` | `ReturnPredictor` | Reconstructs with correct horizon and target column |
| `summary()` | — | `str` | Formatted text report: overall metrics + per-ticker breakdown |

##### Stored State (after `train`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.model` | `XGBClassifier` | Trained model |
| `self.metrics` | `dict` | accuracy, f1, roc_auc, precision, recall, n_predictions, per_ticker |
| `self.feature_importance` | `dict` | Feature name → gain importance (sorted descending) |
| `self._wf_results` | `DataFrame` | All walk-forward predictions (date, ticker, actual, predicted, prob) |
| `self.horizon` | `str` | `"1d"`, `"5d"`, or `"20d"` |
| `self.target_col` | `str` | Derived from TARGETS dict |

---

### 5.6 `predictor.py`

**Purpose**: High-level facade module for one-call sentiment inference (used by scripts and notebooks).
**Lines**: 233

#### Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `_get_analyzer()` | — | `SentimentAnalyzer` | Singleton: lazy-loads the model; prefers fine-tuned checkpoint at `models/saved_models/finbert_finetuned/`, falls back to base ProsusAI/finbert |
| `_load_cached_news(ticker)` | `ticker: str` | `DataFrame` | Loads from `data/raw/news_<TICKER>_*.csv`; parses dates |
| `_load_live_news(ticker)` | `ticker: str` | `DataFrame` | Fetches via `fetch_and_cache(ticker)` from RSS/yfinance |
| `_compute_signal(avg_score)` | `avg_score: float` | `str` | `> 0.15` → `"BULLISH"`, `< -0.15` → `"BEARISH"`, else `"NEUTRAL"` |
| `get_sentiment(ticker, source)` | `ticker: str`, `source: str="auto"` | `dict` | **Main API**: returns `{signal, confidence, avg_score, n_articles, articles: DataFrame, summary_stats: dict}`. Source can be `"cached"`, `"live"`, or `"auto"` (tries live first, falls back to cached) |

---

## 6. Scripts (`scripts/`)

### 6.1 `download_news.py`

**Lines**: 343 · **API**: Alpha Vantage NEWS_SENTIMENT

Downloads financial news articles for specified tickers. Free tier: 25 requests/day, 5/minute.

| Function | Description |
|----------|-------------|
| `to_alpha_vantage_format(date_str)` | Converts `"YYYY-MM-DD"` → `"YYYYMMDDTHHMM"` |
| `download_news(api_key, ticker, start_date, end_date, limit)` | Single-ticker download; parses `ticker_sentiment` array to find ticker-specific scores; returns DataFrame with: title, summary, source, url, published_at, authors, overall_sentiment_score/label, ticker_sentiment_score/label, ticker |
| `save_news_to_csv(df, ticker, start_date, end_date)` | Saves to `data/raw/news_<TICKER>_<START>_to_<END>.csv` |
| `main()` | Loops over `DEFAULT_TICKERS` (AAPL, MSFT), downloads + saves, 15s pause between tickers |

**Requires**: `.env` file with `ALPHA_VANTAGE_KEY`.

---

### 6.2 `download_prices.py`

**Lines**: ~180 · **API**: Yahoo Finance (yfinance, no key needed)

Downloads OHLCV price data for all 19 tickers + VIX.

| Function | Description |
|----------|-------------|
| `download_stock_data(ticker, start_date, end_date)` | `yf.download()` → reset index → keep Date/Open/High/Low/Close/Volume |
| `save_prices_to_csv(df, ticker, ...)` | Saves to `data/raw/prices_<TICKER>_<START>_to_<END>.csv` |
| `main()` | Loops over 19 tickers, downloads + saves |

---

### 6.3 `train_xgboost.py`

**Lines**: 103

Simple single-horizon training script using default XGBoost parameters.

```python
predictor = ReturnPredictor(n_estimators=100, max_depth=3, ...)
predictor.train(DATA_PATH)
predictor.save(MODEL_DIR)
```

Also runs a demo per-ticker prediction loop and saves metrics to `reports/metrics/`.

---

### 6.4 `tune_xgboost.py`

**Lines**: 341 · **Optimizer**: Optuna (TPE sampler)

Bayesian hyperparameter optimization with **full walk-forward CV** per trial.

| Function | Description |
|----------|-------------|
| `get_prepared_data()` | Loads + prepares features once (cached in module-level dict) |
| `full_walk_forward_eval(params, df, feature_cols)` | Trains at every time step; returns accuracy, roc_auc, f1, n_eval_steps |
| `create_objective(df, feature_cols)` | Closure returning Optuna objective function |
| `_progress_callback(study, trial)` | Prints progress every 10 trials |
| `main()` | Runs optimization → retrains final model with best params → saves model + tuning report |

**Search space** (12 hyperparameters):

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 50–600 |
| `max_depth` | 2–10 |
| `learning_rate` | 0.005–0.3 (log) |
| `min_child_weight` | 1–20 |
| `gamma` | 0.0–5.0 |
| `subsample` | 0.4–1.0 |
| `colsample_bytree` | 0.3–1.0 |
| `colsample_bylevel` | 0.3–1.0 |
| `reg_alpha` | 1e-5–10.0 (log) |
| `reg_lambda` | 0.01–20.0 (log) |
| `scale_pos_weight` | 0.5–2.0 |

**Objective**: Maximize walk-forward ROC AUC.

---

### 6.5 `train_multihorizon.py`

**Lines**: 234

Trains XGBoost models for **three horizons** with per-horizon hyperparameters.

| Horizon | Key Hyperparameters | Target Column |
|---------|---------------------|---------------|
| 1d | Optuna-tuned: n_estimators=473, max_depth=5, lr=0.213, gamma=3.398 | `return_direction` |
| 5d | Moderate defaults: n_estimators=300, max_depth=4, lr=0.1 | `return_5d` |
| 20d | Conservative: n_estimators=200, max_depth=3, lr=0.05, gamma=2.0 | `return_20d` |

Saves each model to `models/saved_models/xgboost_return_{hz}/` and a combined report to `reports/metrics/multihorizon_summary.json`.

---

## 7. Dashboard (`dashboard/app.py`)

**Lines**: 1,691 · **Framework**: Streamlit 1.54 + Plotly

Launch: `streamlit run dashboard/app.py`

### Cached Loaders

| Function | Cache Type | Description |
|----------|-----------|-------------|
| `get_merged()` | `@st.cache_data` | Loads full merged dataset via `load_merged_dataset()` |
| `get_news()` | `@st.cache_data` | Loads raw news with session column |
| `load_xgb_model()` | `@st.cache_resource` | Loads single-horizon XGBoost from `xgboost_return/` |
| `load_multihorizon_models()` | `@st.cache_resource` | Loads 1d/5d/20d models; returns dict |
| `load_model()` | `@st.cache_resource` | Loads FinBERT (prefers fine-tuned checkpoint) |
| `get_live_news(ticker)` | `@st.cache_data(ttl=3600)` | Fetches live RSS news |
| `score_live_headlines(ticker)` | `@st.cache_data(ttl=3600)` | Fetches + scores with FinBERT |

### Sidebar Controls

- **Ticker selector**: dropdown of all 19 `DEFAULT_TICKERS`
- **Date range**: date input picker bounded by dataset range

### Panel Reference

| # | Title | Description | Key Visualisations |
|---|-------|-------------|-------------------|
| 1 | **Signal Card + Metrics** | Latest sentiment signal (BULLISH/BEARISH/NEUTRAL), last close price, avg daily return, total articles, 5-day rolling sentiment | HTML signal cards, metric cards |
| 2 | **Price & Sentiment Over Time** | Dual-axis time series: close price (line) + daily sentiment (bars) + 5-day rolling (dashed orange) | Plotly subplots (2 rows, shared x-axis) |
| 3 | **News Feed + Distribution** | Colour-coded article table (score, label) + sentiment donut chart + score histogram | Styled DataFrame, Pie chart, Histogram |
| 4 | **Sentiment–Price Relationship** | Scatter plot (sentiment vs. return) with trend line + Pearson r, Spearman ρ, direction accuracy | Scatter + metrics cards |
| 5 | **Live Headlines** | Real-time RSS/yfinance headlines scored by FinBERT; live signal card; top-5 impact bar chart; sortable table with Impact metric | Impact = |score| × confidence |
| 6 | **Multi-Ticker Comparison** | Side-by-side signal cards for 2–6 tickers; grouped bar chart (avg score + pos/neg %); comparison table | Cards + grouped bars |
| 7 | **Alert Simulation** | Rule-based signal engine with 4 configurable thresholds (min score, min confidence, min dominance, min articles) + strong-signal bonus rules → BUY/SELL/HOLD/STRONG BUY/STRONG SELL | Signal card, rule breakdown, gauge chart |
| 8 | **XGBoost Prediction** (Panel 9) | Next-day prediction card (UP/DOWN + probability); probability gauge; top-10 feature importance bar chart; walk-forward rolling accuracy timeline; per-ticker metric cards | Prediction card, gauge, bars, timeline |
| 9 | **Multi-Horizon Prediction** (Panel 10) | 1d/5d/20d prediction cards side-by-side; grouped P(Up)/P(Down) bar chart; model comparison table; per-ticker accuracy heatmap across horizons | Cards, grouped bars, table, heatmap |

### Custom CSS Classes

| Class | Purpose |
|-------|---------|
| `.signal-card` + `.bullish` / `.bearish` / `.neutral-card` | Gradient signal cards |
| `.metric-card` + `.metric-label` / `.metric-value` / `.metric-delta` | KPI metric cards |
| `.pred-card` + `.pred-up` / `.pred-down` / `.pred-neutral` | XGBoost prediction cards |
| `.alert-card` + `.alert-strong-buy` / `.alert-buy` / `.alert-hold` / `.alert-sell` / `.alert-strong-sell` | Alert simulation cards |
| `.live-badge` + `.badge-pos` / `.badge-neg` / `.badge-neu` | Live headline sentiment badges |
| `.feature-bar` | Feature importance row layout |
| `.rule-row` + `.rule-pass` / `.rule-fail` | Alert rule breakdown rows |

---

## 8. Notebooks (`notebooks/`)

### Evaluation

| Notebook | Purpose |
|----------|---------|
| `01_news_exploration.ipynb` | EDA on raw news data: article counts, sources, temporal distribution, sentiment label breakdown |
| `02_price_exploration.ipynb` | EDA on price data: return distributions, volatility, correlation between tickers |
| `03_baseline_sentiment.ipynb` | Compares Alpha Vantage sentiment scores (API-provided) vs. FinBERT predictions |
| `04_finbert_price_evaluation.ipynb` | Full evaluation: sentiment–price correlation, direction accuracy, lag analysis, quintile returns |

### Modeling

| Notebook | Purpose |
|----------|---------|
| `01_finbert_baseline.ipynb` | FinBERT baseline: loads model, runs predictions, analyses output distributions |

---

## 9. Models & Artifacts

### FinBERT (fine-tuned)

**Location**: `models/saved_models/finbert_finetuned/`

| File | Description |
|------|-------------|
| `model.safetensors` | Fine-tuned FinBERT weights (safetensors format) |
| `config.json` | HuggingFace model config |
| `tokenizer.json` | Fast tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |
| `analyzer_meta.json` | Label map, model name, max_length |

### XGBoost Models

**Locations**: `models/saved_models/xgboost_return_{1d,5d,20d}/`

Each directory contains:

| File | Description |
|------|-------------|
| `model.json` | XGBoost model (JSON format) |
| `meta.json` | Metrics (accuracy, AUC, F1, precision, recall), hyperparameters, feature importance, horizon, per-ticker breakdown |
| `walk_forward_results.csv` | Per-day out-of-sample predictions: date, ticker, actual, predicted, prob_up, prob_down |

---

## 10. Data Files

### Raw Data (`data/raw/`)

| Pattern | Source | Count | Description |
|---------|--------|-------|-------------|
| `news_AAPL_*.csv` | Alpha Vantage | ~4,000 articles | Financial news with sentiment scores |
| `news_MSFT_*.csv` | Alpha Vantage | ~4,000 articles | Financial news with sentiment scores |
| `prices_<TICKER>_*.csv` | Yahoo Finance | 19 tickers × ~251 rows | OHLCV daily price data |
| `prices_^VIX_*.csv` | Yahoo Finance | ~251 rows | VIX volatility index |

**News CSV columns**: title, summary, source, url, published_at, authors, overall_sentiment_score, overall_sentiment_label, ticker_sentiment_score, ticker_sentiment_label, ticker

**Price CSV columns**: Date, Open, High, Low, Close, Volume (note: yfinance adds an extra header row → `skiprows=[1]`)

### Processed Data (`data/processed/`)

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `merged_2025-02-13_to_2026-02-13.csv` | ~4,769 | ~40 | Full merged dataset: 19 tickers × ~251 days, with sentiment, price features, technical indicators, VIX, and forward return targets |

---

## 11. Feature Catalogue

The XGBoost models use **29 features**, organized into 4 groups. All features are **lagged by 1 day** (via `prepare_features()`) to prevent lookahead bias.

### Sentiment Features (9)

| Feature | Source | Description |
|---------|--------|-------------|
| `avg_overall_sentiment` | Alpha Vantage → aggregate | Mean sentiment score of all articles for that day |
| `avg_ticker_sentiment` | Alpha Vantage → aggregate | Mean ticker-specific sentiment score |
| `sentiment_std` | Computed | Standard deviation of article scores (disagreement measure) |
| `sentiment_range` | Computed | max − min sentiment score for the day |
| `pct_positive` | Computed | % of articles with positive sentiment |
| `pct_negative` | Computed | % of articles with negative sentiment |
| `article_count` | Computed | Number of news articles on that day |
| `sentiment_rolling_3d` | Computed | 3-day rolling mean of avg_overall_sentiment |
| `sentiment_rolling_5d` | Computed | 5-day rolling mean of avg_overall_sentiment |

### Price Features (4)

| Feature | Source | Description |
|---------|--------|-------------|
| `daily_return` | yfinance → computed | `(Close − prev_Close) / prev_Close × 100` |
| `intraday_range` | yfinance → computed | `(High − Low) / Close` |
| `gap_pct` | yfinance → computed | `(Open − prev_Close) / prev_Close × 100` |
| `volume_change` | yfinance → computed | `Volume / prev_Volume − 1` |

### Technical Features (10)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `rsi_14` | 14-day RSI | Relative Strength Index (0–100; >70 overbought, <30 oversold) |
| `macd` | 12/26 EMA diff | MACD line |
| `macd_signal` | 9-day EMA of MACD | Signal line |
| `macd_histogram` | MACD − signal | Momentum histogram |
| `bb_pct_b` | 20-day Bollinger | Position within bands (0=lower, 1=upper) |
| `atr_14` | 14-day ATR | Average True Range / Close (volatility) |
| `distance_52w_high` | Rolling 252-day max | `(Close − 52w_high) / 52w_high` |
| `distance_52w_low` | Rolling 252-day min | `(Close − 52w_low) / 52w_low` |
| `volume_zscore` | 20-day z-score | Volume anomaly detection |
| `VIX_close` | CBOE VIX index | Market-wide fear gauge |

### Engineered Features (6)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `return_lag2` | `daily_return.shift(2)` | Return 2 days ago |
| `return_lag3` | `daily_return.shift(3)` | Return 3 days ago |
| `volatility_5d` | `daily_return.rolling(5).std()` | 5-day return standard deviation |
| `avg_return_5d` | `daily_return.rolling(5).mean()` | 5-day average return |
| `sentiment_momentum` | `rolling_3d − rolling_5d` | Short vs. medium-term sentiment trend |
| `news_has_coverage` | `article_count > 0` → 1/0 | Binary: was there any news? |

### Target Variables

| Target | Column | Description |
|--------|--------|-------------|
| 1-day | `return_direction` | 1 if today's close > yesterday's close, else 0 |
| 5-day | `return_5d` | 1 if close in 5 days > today's close, else 0 |
| 20-day | `return_20d` | 1 if close in 20 days > today's close, else 0 |

---

## 12. Model Performance Summary

### Multi-Horizon Results (Walk-Forward Validation)

| Horizon | Accuracy | F1 | AUC | Precision | Recall |
|---------|----------|-----|------|-----------|--------|
| **1d** (Daily) | 52.9% | 62.8% | 51.7% | 53.5% | 75.9% |
| **5d** (Weekly) | 60.4% | 67.8% | 63.6% | 61.9% | 74.9% |
| **20d** (Monthly) | 71.4% | 81.0% | 71.0% | 73.3% | 90.6% |

### Key Observations

- **Longer horizons are easier to predict**: 20d achieves 71.4% accuracy vs. 52.9% for 1d. This is expected — daily noise averages out over longer windows.
- **Walk-forward validation prevents overfitting**: Every split trains only on past data and tests on the next unseen day(s). No data from the future ever leaks into training.
- **Optuna tuning** (150 trials, 53.6 min) improved 1d AUC from 52.1% → 55.0% and F1 from 58.1% → 61.4%.
- **Feature importance**: Technical indicators (RSI, MACD, ATR) and price features (daily_return, volume_change) tend to rank higher than sentiment features, suggesting that with only 2 tickers having news data, the sentiment signal is diluted across 19 tickers.

---

## 13. Configuration & Environment

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ALPHA_VANTAGE_KEY` | For news download only | Free API key from [alphavantage.co](https://www.alphavantage.co/support/#api-key) |

### Key Configuration Points

| Setting | Location | Current Value |
|---------|----------|---------------|
| Tickers list | `src/data/data_loader.py` → `DEFAULT_TICKERS` | 19 tickers |
| Date range | `src/data/data_loader.py` → `DEFAULT_START/END` | 2025-02-13 to 2026-02-13 |
| FinBERT model | `src/models/sentiment_analyzer.py` | ProsusAI/finbert (or fine-tuned at `models/saved_models/finbert_finetuned/`) |
| XGBoost hyperparams | `scripts/train_multihorizon.py` → `HORIZON_PARAMS` | Per-horizon tuned |
| Walk-forward min training | `MIN_TRAIN_DAYS` | 60 days |
| Signal thresholds | `src/predictor.py` / `dashboard/app.py` | ±0.15 for BULLISH/BEARISH |
| RSS feeds | `src/data/news_fetcher.py` → `_RSS_FEEDS` | Google News, Yahoo Finance |
| Live news cache TTL | `src/data/news_fetcher.py` → `DEFAULT_CACHE_TTL` | 3600s (1 hour) |
| Max live articles | `news_fetcher.py` → `fetch_live_news()` | 60 |

### Platform Notes

- **Windows encoding**: All `print()` statements use ASCII only (no Unicode arrows/dashes) to avoid `UnicodeEncodeError` on Windows consoles.
- **Streamlit 1.54**: Uses `width="stretch"` instead of deprecated `use_container_width=True`. Uses HTML `<div>` cards instead of `st.metric()` (which renders blank in some environments).
- **XGBoost 3.2**: The `use_label_encoder` parameter has been removed; `eval_metric` is passed inside the model params dict.

---

## 14. How to Extend

### Add More Tickers

1. Add the ticker symbol to `DEFAULT_TICKERS` in `src/data/data_loader.py`
2. Add it to `DEFAULT_TICKERS` in `scripts/download_prices.py`
3. Run `python scripts/download_prices.py`
4. If you have Alpha Vantage quota, also add to `scripts/download_news.py`
5. Rebuild the merged dataset and retrain models

### Add a New Technical Indicator

1. Implement computation in `add_technical_indicators()` inside `src/data/data_loader.py`
2. Add the column name to `TECHNICAL_FEATURES` in `src/models/return_predictor.py`
3. Retrain models

### Add a New Prediction Horizon

1. Add a forward return column in `add_technical_indicators()` (e.g., `return_10d`)
2. Add the mapping to `TARGETS` in `src/models/return_predictor.py`
3. Add horizon params to `HORIZON_PARAMS` in `scripts/train_multihorizon.py`
4. Update the dashboard's Panel 10 to include the new horizon

### Replace the Sentiment Model

1. Create a new class following the `SentimentAnalyzer` interface (must have `predict()`, `predict_batch()`)
2. Update `src/predictor.py` → `_get_analyzer()` to load the new model
3. The rest of the pipeline (data_loader, return_predictor, dashboard) will work unchanged

### Add a Paid News API

1. Create a new fetcher function in `src/data/news_fetcher.py` (e.g., `_fetch_newsapi()`)
2. Integrate it into `fetch_live_news()`
3. Re-download news for all 19 tickers
4. With 19× more sentiment data, model performance should improve significantly

---

## Quick Command Reference

```bash
# Download fresh data
python scripts/download_prices.py
python scripts/download_news.py

# Build merged dataset
python -c "from src.data.data_loader import load_merged_dataset; load_merged_dataset(save_processed=True)"

# Train single-horizon model
python scripts/train_xgboost.py

# Hyperparameter tuning (150 trials, ~1 hour)
python scripts/tune_xgboost.py --trials 150

# Train all three horizons
python scripts/train_multihorizon.py

# Launch dashboard
streamlit run dashboard/app.py

# Quick sentiment check (CLI)
python -c "from src.predictor import get_sentiment; print(get_sentiment('AAPL'))"
```

---

*Last updated: Phase 1 — multi-ticker, multi-horizon, 10-panel dashboard.*
