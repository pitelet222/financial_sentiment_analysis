---
language: en
license: apache-2.0
tags:
  - finance
  - sentiment-analysis
  - finbert
  - text-classification
datasets:
  - financial_phrasebank
  - custom-alpha-vantage-news
pipeline_tag: text-classification
base_model: ProsusAI/finbert
---

# FinBERT Fine-Tuned for Financial News Sentiment

A fine-tuned version of [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) optimized for scoring financial news headlines as **positive**, **negative**, or **neutral**.

## Model Details

- **Base model**: ProsusAI/finbert (BERT-base, 109.5M parameters)
- **Fine-tuning data**: Financial news headlines from Alpha Vantage API (AAPL, MSFT — ~8,000 articles, Feb 2025 – Feb 2026)
- **Task**: 3-class text classification (positive / negative / neutral)
- **Output**: Label + score (−1 to +1) + per-class probabilities + calibrated confidence

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "YOUR_USERNAME/finbert-financial-news"  # Update after upload
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

headline = "Apple reports record quarterly revenue, beating analyst expectations"
inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)

labels = ["positive", "negative", "neutral"]
pred_idx = probs.argmax().item()
print(f"Prediction: {labels[pred_idx]} ({probs[0][pred_idx]:.1%})")
```

## Or use the project wrapper

```python
from src.models.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer.load("models/saved_models/finbert_finetuned/")
result = analyzer.predict("Tesla surges 8% after strong delivery numbers")
# {'label': 'positive', 'score': 0.42, 'confidence': 0.89,
#  'positive': 0.89, 'negative': 0.03, 'neutral': 0.08}
```

## Part of a Larger Pipeline

This model is the sentiment scoring component of a full **FinBERT + XGBoost** pipeline that predicts stock return direction across 3 horizons (daily / weekly / monthly) for 19 tickers. See the [full project on GitHub](https://github.com/YOUR_USERNAME/financial-sentiment-analysis).

## Limitations

- Trained primarily on English-language financial news
- Optimized for headline-length text (< 128 tokens)
- Sentiment scores reflect market-moving language, not fundamental analysis
- Fine-tuned on a specific time period (Feb 2025 – Feb 2026) — may not generalize to all market regimes

## Citation

```bibtex
@misc{finbert-finetuned-news,
  title={FinBERT Fine-Tuned for Financial News Sentiment},
  author={Your Name},
  year={2026},
  url={https://github.com/YOUR_USERNAME/financial-sentiment-analysis}
}
```
