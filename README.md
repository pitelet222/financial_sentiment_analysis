# Financial Sentiment Analysis for Trading Signals

A machine learning project that analyses financial news sentiment using **FinBERT** (fine-tuned) and predicts next-day stock return direction with **XGBoost**, presented through an interactive **Streamlit** dashboard.

## Features

- **FinBERT Sentiment Analysis** — Fine-tuned ProsusAI/FinBERT model scores headlines as positive / negative / neutral with calibrated confidence
- **XGBoost Return Predictor** — Predicts next-day return direction (UP / DOWN) using lagged sentiment + price features, validated with walk-forward cross-validation (no lookahead bias)
- **Live News Pipeline** — Aggregates headlines from 3 sources (Google News RSS, Yahoo Finance RSS, yfinance API) with disk caching
- **Interactive Dashboard** — Streamlit app with 9 panels:
  1. Signal card + key metrics (bullish / bearish / neutral)
  2. Dual-axis price + sentiment chart (Plotly)
  3. Historical news feed with sentiment scores
  4. Sentiment distribution (donut + histogram)
  5. Correlation analysis (scatter + Pearson / Spearman)
  6. Live headlines scored by FinBERT in real time
  7. Multi-ticker comparison (up to 6 tickers side-by-side)
  8. Alert simulation panel (rule-based BUY / SELL / HOLD engine)
  9. XGBoost prediction panel (next-day direction, feature importance, walk-forward accuracy timeline)
- **Impact Ranking** — Headlines ranked by `|score| × confidence` with top-5 chart
- **Multi-Ticker Comparison** — Compare sentiment across AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META
- **Rule-Based Alert Engine** — 4 configurable thresholds, 6 signal levels (STRONG BUY → STRONG SELL), transparent rule breakdown

## Project Structure

```
financial-sentiment-analysis/
├── dashboard/
│   └── app.py              # Streamlit dashboard (main entry point)
├── data/
│   ├── cache/              # Disk-cached live news (auto-generated)
│   ├── processed/          # Merged price + sentiment CSVs
│   └── raw/                # Raw news & price CSVs (AAPL, MSFT)
├── models/
│   └── saved_models/
│       ├── finbert_finetuned/  # Fine-tuned FinBERT checkpoint
│       └── xgboost_return/     # Trained XGBoost predictor
├── notebooks/
│   ├── evaluation/         # News/price exploration, baseline, FinBERT eval
│   └── modeling/           # FinBERT fine-tuning notebook
├── scripts/
│   ├── download_news.py    # Fetch historical news (Alpha Vantage)
│   ├── download_news_extended.py  # Monthly-chunked 1yr news download
│   ├── download_prices.py  # Fetch historical prices (yfinance)
│   └── train_xgboost.py    # Train XGBoost return predictor
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Load, merge & feature-engineer datasets
│   │   ├── news_fetcher.py     # Live RSS + yfinance news pipeline
│   │   └── preprocessor.py     # Text cleaning & normalisation
│   ├── models/
│   │   ├── sentiment_analyzer.py  # FinBERT wrapper (predict, fine-tune, save/load)
│   │   └── return_predictor.py    # XGBoost return direction predictor
│   └── predictor.py        # High-level get_sentiment() facade
├── tests/
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- ~500 MB disk space for the FinBERT model weights

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd financial-sentiment-analysis

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at [http://localhost:8501](http://localhost:8501). The FinBERT model loads automatically on first run (~10s).

### Train the XGBoost Predictor

```bash
# Ensure the merged dataset exists first
python scripts/train_xgboost.py
```

Trains with walk-forward validation (expanding window, no lookahead). Saves to `models/saved_models/xgboost_return/`.

### Model Setup

The fine-tuned FinBERT checkpoint (`models/saved_models/finbert_finetuned/`) is excluded from git due to its size (~440 MB). On first run the code **automatically falls back** to the base [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) model from Hugging Face, which is downloaded and cached automatically. To use the fine-tuned checkpoint, place it in the path above or re-run the fine-tuning notebook (`notebooks/modeling/01_finbert_baseline.ipynb`).

### API Keys

Copy `.env.example` to `.env` and fill in your keys. Only needed for fetching historical news:

```bash
cp .env.example .env
```

### Quick Sentiment Check (CLI)

```python
from src.predictor import get_sentiment
result = get_sentiment("AAPL")
print(result["signal"], result["confidence"], result["avg_score"])
```

## Data

| Dataset | Source | Tickers | Period |
|---------|--------|---------|--------|
| Historical news | Alpha Vantage API | AAPL, MSFT | Feb 2025 – Feb 2026 |
| Historical prices | Yahoo Finance | AAPL, MSFT | Feb 2025 – Feb 2026 |
| Live headlines | Google News RSS, Yahoo Finance RSS, yfinance API | Any ticker | Real-time |

## Models

### FinBERT (Sentiment)

- **Base**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (BERT fine-tuned on financial text)
- **Fine-tuned checkpoint**: `models/saved_models/finbert_finetuned/`
- **Parameters**: 109.5M
- **Output**: `{ label, score, confidence, positive, negative, neutral }`

### XGBoost (Return Prediction)

- **Task**: Binary classification — next-day return direction (UP / DOWN)
- **Features**: 15 features (9 sentiment + 4 price + 6 engineered), all lagged by 1 day
- **Validation**: Walk-forward (expanding window) — no lookahead bias
- **Saved to**: `models/saved_models/xgboost_return/`

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Sentiment model | PyTorch + Hugging Face Transformers || Return predictor | XGBoost + scikit-learn || Dashboard | Streamlit 1.54 + Plotly |
| Data pipeline | pandas, yfinance, feedparser |
| Live news | RSS feeds + yfinance API (zero API keys) |
| Historical news | Alpha Vantage (API key required) |

## Disclaimer

⚠️ This project is for **educational purposes only**. The trading signals are simulated and do not constitute financial advice. Always do your own research before making investment decisions.

