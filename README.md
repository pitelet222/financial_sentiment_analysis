# Financial Sentiment Analysis for Trading Signals

A machine learning project that analyses financial news sentiment using **FinBERT** (fine-tuned) and generates rule-based trading signals through an interactive **Streamlit** dashboard.

## Features

- **FinBERT Sentiment Analysis** — Fine-tuned ProsusAI/FinBERT model scores headlines as positive / negative / neutral with calibrated confidence
- **Live News Pipeline** — Aggregates headlines from 3 sources (Google News RSS, Yahoo Finance RSS, yfinance API) with disk caching
- **Interactive Dashboard** — Streamlit app with 8 panels:
  1. Signal card + key metrics (bullish / bearish / neutral)
  2. Dual-axis price + sentiment chart (Plotly)
  3. Historical news feed with sentiment scores
  4. Sentiment distribution (donut + histogram)
  5. Correlation analysis (scatter + Pearson / Spearman)
  6. Live headlines scored by FinBERT in real time
  7. Multi-ticker comparison (up to 6 tickers side-by-side)
  8. Alert simulation panel (rule-based BUY / SELL / HOLD engine)
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
│       └── finbert_finetuned/  # Fine-tuned FinBERT checkpoint
├── notebooks/
│   ├── evaluation/         # News/price exploration, baseline, FinBERT eval
│   └── modeling/           # FinBERT fine-tuning notebook
├── scripts/
│   ├── download_news.py    # Fetch historical news (Alpha Vantage)
│   └── download_prices.py  # Fetch historical prices (yfinance)
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Load, merge & feature-engineer datasets
│   │   ├── news_fetcher.py     # Live RSS + yfinance news pipeline
│   │   └── preprocessor.py     # Text cleaning & normalisation
│   ├── models/
│   │   └── sentiment_analyzer.py  # FinBERT wrapper (predict, fine-tune, save/load)
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

### Quick Sentiment Check (CLI)

```python
from src.predictor import get_sentiment
result = get_sentiment("AAPL")
print(result["signal"], result["confidence"], result["avg_score"])
```

## Data

| Dataset | Source | Tickers | Period |
|---------|--------|---------|--------|
| Historical news | Alpha Vantage API | AAPL, MSFT | Nov 2025 – Feb 2026 |
| Historical prices | Yahoo Finance | AAPL, MSFT | Nov 2025 – Feb 2026 |
| Live headlines | Google News RSS, Yahoo Finance RSS, yfinance API | Any ticker | Real-time |

## Model

- **Base**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (BERT fine-tuned on financial text)
- **Fine-tuned checkpoint**: `models/saved_models/finbert_finetuned/`
- **Parameters**: 109.5M
- **Output**: `{ label, score, confidence, positive, negative, neutral }`

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Sentiment model | PyTorch + Hugging Face Transformers |
| Dashboard | Streamlit 1.54 + Plotly |
| Data pipeline | pandas, yfinance, feedparser |
| Live news | RSS feeds + yfinance API (zero API keys) |
| Historical news | Alpha Vantage (API key required) |

## Disclaimer

⚠️ This project is for **educational purposes only**. The trading signals are simulated and do not constitute financial advice. Always do your own research before making investment decisions.

