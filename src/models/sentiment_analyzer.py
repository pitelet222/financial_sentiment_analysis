"""
Sentiment Analyzer — Reusable FinBERT Wrapper for Production Use
=================================================================

This module provides a ``SentimentAnalyzer`` class that wraps the
ProsusAI/finbert model (or any compatible Hugging Face sequence-
classification model) with a clean, production-ready API.

It encapsulates:
- Model and tokenizer loading (with optional GPU support)
- Single-text and batched inference
- Text preprocessing via the project's ``preprocessor`` module
- Continuous score computation: P(positive) − P(negative)
- Integration helpers for merging predictions into DataFrames
- **Evaluation** of sentiment–price correlation on merged datasets
- **Fine-tuning** on domain-specific labelled data
- **Save / load** for checkpointed or fine-tuned models

Typical usage
-------------
>>> from src.models.sentiment_analyzer import SentimentAnalyzer
>>> analyzer = SentimentAnalyzer()              # loads ProsusAI/finbert
>>> analyzer.predict("Apple reports record revenue")
{'label': 'positive', 'positive': 0.95, 'negative': 0.02, 'neutral': 0.03,
 'confidence': 0.95, 'score': 0.93}

>>> import pandas as pd
>>> df = pd.read_csv("data/raw/news_AAPL_2025-11-01_to_2026-02-13.csv")
>>> df = analyzer.predict_dataframe(df, text_columns=["title", "summary"])

Fine-tuning
-----------
>>> analyzer.fine_tune(train_df, text_col="summary", label_col="label",
...                    epochs=3, lr=2e-5)
>>> analyzer.save("models/saved_models/finbert_finetuned")

Architecture note
-----------------
The class is stateful (holds the model in memory) and thread-safe for
inference (``model.eval()`` + ``torch.no_grad()``).  It is designed to
be instantiated once and reused across the application lifecycle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Project imports
from src.data.preprocessor import preprocess_text

logger = logging.getLogger(__name__)


# =========================================================================
# Constants
# =========================================================================

DEFAULT_MODEL_NAME = "ProsusAI/finbert"

# FinBERT's label mapping (from the model's config.json)
_LABEL_MAP: Dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}


# =========================================================================
# SentimentAnalyzer class
# =========================================================================

class SentimentAnalyzer:
    """High-level wrapper around a Hugging Face sentiment model.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model identifier or path to a local checkpoint.
        Default: ``"ProsusAI/finbert"``.
    device : str, optional
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        If *None* (default), auto-detects CUDA availability.
    max_length : int, optional
        Maximum token length for the tokenizer (default 512).
    label_map : dict, optional
        Mapping from model output index to label string.
        Default: ``{0: "positive", 1: "negative", 2: "neutral"}``.

    Attributes
    ----------
    model : AutoModelForSequenceClassification
        The loaded transformer model in eval mode.
    tokenizer : AutoTokenizer
        The corresponding tokenizer.
    device : torch.device
        Device the model is running on.
    label_map : dict
        Index → label mapping.

    Examples
    --------
    >>> analyzer = SentimentAnalyzer()
    >>> result = analyzer.predict("Stocks surge on strong earnings")
    >>> result["label"]
    'positive'
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = 512,
        label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        # Resolve device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_name = model_name
        self.max_length = max_length
        self.label_map = label_map or dict(_LABEL_MAP)

        # Load tokenizer + model
        print(f"[SentimentAnalyzer] Loading '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[SentimentAnalyzer] Ready — {n_params:,} parameters, "
              f"labels: {list(self.label_map.values())}")

    # -----------------------------------------------------------------
    # Single-text prediction
    # -----------------------------------------------------------------

    def predict(self, text: str, preprocess: bool = True) -> Dict[str, Union[str, float]]:
        """Predict sentiment for a single text.

        Parameters
        ----------
        text : str
            Raw or pre-cleaned financial text.
        preprocess : bool, optional
            If *True* (default), apply ``preprocess_text()`` before
            tokenization.

        Returns
        -------
        dict
            Keys:
            - ``label`` : str — predicted class (positive/negative/neutral)
            - ``positive`` : float — P(positive)
            - ``negative`` : float — P(negative)
            - ``neutral`` : float — P(neutral)
            - ``confidence`` : float — probability of the predicted class
            - ``score`` : float — P(positive) − P(negative), range [-1, +1]
        """
        if preprocess:
            # Approximate max chars: ~4 chars/token is a safe heuristic
            text = preprocess_text(text, max_length=(self.max_length - 2) * 4)

        if not text or not text.strip():
            return self._neutral_result()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        return self._probs_to_result(probs)

    # -----------------------------------------------------------------
    # Batch prediction
    # -----------------------------------------------------------------

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        preprocess: bool = True,
        show_progress: bool = True,
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict sentiment for a list of texts using batched inference.

        Parameters
        ----------
        texts : list of str
            Raw text strings.
        batch_size : int, optional
            Texts per forward pass (default 16).  Reduce if running
            out of memory on GPU.
        preprocess : bool, optional
            Apply text cleaning before tokenization (default *True*).
        show_progress : bool, optional
            Print progress counter (default *True*).

        Returns
        -------
        list of dict
            One result dict per input text (same format as ``predict``).
        """
        results: List[Dict[str, Union[str, float]]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            # Preprocess
            if preprocess:
                cleaned = [
                    preprocess_text(t, max_length=(self.max_length - 2) * 4)
                    if isinstance(t, str) else ""
                    for t in batch_texts
                ]
            else:
                cleaned = [t if isinstance(t, str) else "" for t in batch_texts]

            # Tokenize
            inputs = self.tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            for j in range(probs.shape[0]):
                results.append(self._probs_to_result(probs[j]))

            if show_progress:
                done = min(i + batch_size, len(texts))
                print(f"  Processed {done}/{len(texts)} texts", end="\r")

        if show_progress:
            print()  # newline after progress

        return results

    # -----------------------------------------------------------------
    # DataFrame integration
    # -----------------------------------------------------------------

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        batch_size: int = 16,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """Run sentiment prediction on one or more text columns.

        For each column in *text_columns*, adds the following columns:
        - ``finbert_<col>_label``  — predicted label
        - ``finbert_<col>_score``  — continuous score [-1, +1]
        - ``finbert_<col>_conf``   — confidence of predicted label
        - ``finbert_<col>_pos``    — P(positive)
        - ``finbert_<col>_neg``    — P(negative)
        - ``finbert_<col>_neu``    — P(neutral)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with text columns to analyse.
        text_columns : list of str, optional
            Columns to predict on.  Default: ``["title", "summary"]``.
        batch_size : int, optional
            Batch size for inference (default 16).
        preprocess : bool, optional
            Apply text cleaning (default *True*).

        Returns
        -------
        pd.DataFrame
            Copy of the input with prediction columns appended.
        """
        if text_columns is None:
            text_columns = ["title", "summary"]

        df = df.copy()

        for col in text_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available: {list(df.columns)}"
                )

            print(f"[SentimentAnalyzer] Predicting on '{col}' "
                  f"({len(df)} texts, batch_size={batch_size}) ...")

            preds = self.predict_batch(
                df[col].tolist(),
                batch_size=batch_size,
                preprocess=preprocess,
            )

            prefix = f"finbert_{col}"
            df[f"{prefix}_label"] = [p["label"] for p in preds]
            df[f"{prefix}_score"] = [p["score"] for p in preds]
            df[f"{prefix}_conf"] = [p["confidence"] for p in preds]
            df[f"{prefix}_pos"] = [p["positive"] for p in preds]
            df[f"{prefix}_neg"] = [p["negative"] for p in preds]
            df[f"{prefix}_neu"] = [p["neutral"] for p in preds]

        return df

    # -----------------------------------------------------------------
    # Aggregate FinBERT sentiment per trading day
    # -----------------------------------------------------------------

    def aggregate_daily_sentiment(
        self,
        news_with_preds: pd.DataFrame,
        score_column: str = "finbert_summary_score",
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate FinBERT scores to one row per trading day per ticker.

        Similar to ``data_loader.aggregate_sentiment()`` but operates on
        FinBERT's continuous score instead of Alpha Vantage scores.

        Parameters
        ----------
        news_with_preds : pd.DataFrame
            Must contain ``trading_day``, ``ticker``, and the column
            specified by *score_column*.
        score_column : str
            Column with continuous FinBERT scores (default:
            ``"finbert_summary_score"``).
        group_cols : list of str, optional
            Columns to group by (default: ``["trading_day", "ticker"]``).

        Returns
        -------
        pd.DataFrame
            Aggregated FinBERT sentiment with columns:
            ``fb_article_count``, ``fb_avg_sentiment``,
            ``fb_min_sentiment``, ``fb_max_sentiment``,
            ``fb_sentiment_std``, ``fb_pct_positive``,
            ``fb_pct_negative``.
        """
        if group_cols is None:
            group_cols = ["trading_day", "ticker"]

        def _agg(g: pd.DataFrame) -> pd.Series:
            scores = g[score_column].dropna()
            return pd.Series({
                "fb_article_count": len(g),
                "fb_avg_sentiment": scores.mean() if len(scores) else np.nan,
                "fb_min_sentiment": scores.min() if len(scores) else np.nan,
                "fb_max_sentiment": scores.max() if len(scores) else np.nan,
                "fb_sentiment_std": scores.std() if len(scores) > 1 else 0.0,
                "fb_pct_positive": (scores > 0.1).mean() if len(scores) else np.nan,
                "fb_pct_negative": (scores < -0.1).mean() if len(scores) else np.nan,
            })

        needed = group_cols + [score_column]
        agg = (
            news_with_preds[needed]
            .groupby(group_cols)
            .apply(_agg, include_groups=False)
            .reset_index()
        )
        return agg

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _probs_to_result(self, probs: torch.Tensor) -> Dict[str, Union[str, float]]:
        """Convert a probability tensor to a result dictionary."""
        scores = {
            self.label_map[i]: round(probs[i].item(), 4)
            for i in range(len(self.label_map))
        }
        predicted_idx = probs.argmax().item()
        predicted_label = self.label_map[predicted_idx]

        return {
            "label": predicted_label,
            "positive": scores["positive"],
            "negative": scores["negative"],
            "neutral": scores["neutral"],
            "confidence": round(probs[predicted_idx].item(), 4),
            "score": round(scores["positive"] - scores["negative"], 4),
        }

    @staticmethod
    def _neutral_result() -> Dict[str, Union[str, float]]:
        """Return a neutral result for empty/invalid text."""
        return {
            "label": "neutral",
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "confidence": 1.0,
            "score": 0.0,
        }

    # -----------------------------------------------------------------
    # Evaluation: sentiment vs price correlation
    # -----------------------------------------------------------------

    def evaluate_price_correlation(
        self,
        merged_df: pd.DataFrame,
        sentiment_col: str = "fb_avg_sentiment",
        return_col: str = "daily_return",
        direction_col: str = "return_direction",
        ticker_col: str = "ticker",
    ) -> Dict[str, Any]:
        """Evaluate how well FinBERT sentiment correlates with price moves.

        Computes:
        - Pearson & Spearman correlations (overall and per-ticker).
        - Direction accuracy — does positive sentiment predict positive
          returns?
        - Lag analysis — do correlations improve at t+1 or t+2?
        - Quintile returns — average return per sentiment quintile.

        Parameters
        ----------
        merged_df : pd.DataFrame
            Must contain *sentiment_col*, *return_col*, *direction_col*,
            and *ticker_col*.  Rows where sentiment is NaN are dropped.
        sentiment_col : str
            Column with continuous FinBERT sentiment score.
        return_col : str
            Column with daily percentage return.
        direction_col : str
            Column with binary return direction (1 = up, 0 = down).
        ticker_col : str
            Column with the stock ticker.

        Returns
        -------
        dict
            Nested dictionary with correlation metrics keyed by
            ``"overall"`` and per-ticker.
        """
        from scipy import stats

        df = merged_df.dropna(subset=[sentiment_col, return_col]).copy()
        if df.empty:
            logger.warning("No rows with both sentiment and returns.")
            return {}

        results: Dict[str, Any] = {}

        # ---------- overall ----------
        pearson_r, pearson_p = stats.pearsonr(df[sentiment_col], df[return_col])
        spearman_r, spearman_p = stats.spearmanr(df[sentiment_col], df[return_col])

        # Direction accuracy: sentiment > 0 → predict up (1)
        pred_dir = (df[sentiment_col] > 0).astype(int)
        dir_acc = (pred_dir == df[direction_col]).mean()

        results["overall"] = {
            "n": len(df),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "direction_accuracy": round(dir_acc, 4),
        }

        # ---------- per ticker ----------
        results["per_ticker"] = {}
        for ticker, gdf in df.groupby(ticker_col):
            if len(gdf) < 5:
                continue
            pr, pp = stats.pearsonr(gdf[sentiment_col], gdf[return_col])
            sr, sp = stats.spearmanr(gdf[sentiment_col], gdf[return_col])
            d_acc = ((gdf[sentiment_col] > 0).astype(int) == gdf[direction_col]).mean()
            results["per_ticker"][ticker] = {
                "n": len(gdf),
                "pearson_r": round(pr, 4),
                "pearson_p": round(pp, 4),
                "spearman_r": round(sr, 4),
                "spearman_p": round(sp, 4),
                "direction_accuracy": round(d_acc, 4),
            }

        # ---------- lag analysis ----------
        lag_results = {}
        for lag in [0, 1, 2]:
            shifted = df.groupby(ticker_col)[return_col].shift(-lag)
            valid = df[[sentiment_col]].assign(future_ret=shifted).dropna()
            if len(valid) < 5:
                continue
            pr, pp = stats.pearsonr(valid[sentiment_col], valid["future_ret"])
            lag_results[f"t+{lag}"] = {
                "pearson_r": round(pr, 4),
                "pearson_p": round(pp, 4),
                "n": len(valid),
            }
        results["lag_analysis"] = lag_results

        # ---------- quintile analysis ----------
        try:
            df["sent_quintile"] = pd.qcut(df[sentiment_col], 5, labels=False, duplicates="drop")
            quintile = (
                df.groupby("sent_quintile")[return_col]
                .agg(["mean", "median", "count"])
                .rename(columns={"mean": "avg_return", "median": "med_return", "count": "n"})
            )
            results["quintile_returns"] = quintile.to_dict(orient="index")
        except ValueError:
            results["quintile_returns"] = {}

        return results

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Save the model, tokenizer, and metadata to *path*.

        Parameters
        ----------
        path : str or Path
            Directory to save into (created if it doesn't exist).

        Returns
        -------
        Path
            The save directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        meta = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "label_map": {str(k): v for k, v in self.label_map.items()},
        }
        (path / "analyzer_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[SentimentAnalyzer] Saved to {path}")
        return path

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "SentimentAnalyzer":
        """Load a previously saved ``SentimentAnalyzer`` from *path*.

        Parameters
        ----------
        path : str or Path
            Directory produced by ``save()``.
        device : str, optional
            Override device (default: auto-detect).

        Returns
        -------
        SentimentAnalyzer
        """
        path = Path(path)
        meta_file = path / "analyzer_meta.json"
        meta: Dict[str, Any] = {}
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())

        label_map = {int(k): v for k, v in meta.get("label_map", _LABEL_MAP).items()}
        return cls(
            model_name=str(path),
            device=device,
            max_length=meta.get("max_length", 512),
            label_map=label_map,
        )

    # -----------------------------------------------------------------
    # Fine-tuning
    # -----------------------------------------------------------------

    def fine_tune(
        self,
        train_df: pd.DataFrame,
        text_col: str = "summary",
        label_col: str = "label",
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 8,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[float]]:
        """Fine-tune the underlying model on a labelled DataFrame.

        The labels in *label_col* must be the string names that appear
        in ``self.label_map`` (e.g. ``"positive"``, ``"negative"``,
        ``"neutral"``).

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with text and label columns.
        text_col : str
            Column containing text (default ``"summary"``).
        label_col : str
            Column containing labels (default ``"label"``).
        val_df : pd.DataFrame, optional
            Validation data.  If provided, validation loss and accuracy
            are reported after each epoch.
        epochs : int
            Number of training epochs (default 3).
        lr : float
            Peak learning rate (default 2e-5).
        batch_size : int
            Batch size (default 8).
        warmup_ratio : float
            Fraction of total steps used for linear warmup (default 0.1).
        weight_decay : float
            AdamW weight decay (default 0.01).
        max_grad_norm : float
            Gradient clipping norm (default 1.0).
        save_path : str or Path, optional
            If given, save the fine-tuned model after training.

        Returns
        -------
        dict
            Training history with ``"train_loss"`` (and optionally
            ``"val_loss"``, ``"val_accuracy"``) lists per epoch.
        """
        # Reverse label map: label string → index
        label_to_idx = {v: k for k, v in self.label_map.items()}

        # Build datasets
        train_ds = _SentimentDataset(
            texts=train_df[text_col].tolist(),
            labels=[label_to_idx[l] for l in train_df[label_col]],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_df is not None:
            val_ds = _SentimentDataset(
                texts=val_df[text_col].tolist(),
                labels=[label_to_idx[l] for l in val_df[label_col]],
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        self.model.train()
        history: Dict[str, List[float]] = {"train_loss": []}
        if val_loader:
            history["val_loss"] = []
            history["val_accuracy"] = []

        print(f"\n[Fine-tune] {epochs} epochs, lr={lr}, batch_size={batch_size}, "
              f"train_samples={len(train_ds)}")

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            self.model.train()

            for step, batch in enumerate(train_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            history["train_loss"].append(round(avg_train_loss, 4))

            msg = f"  Epoch {epoch}/{epochs} — train_loss={avg_train_loss:.4f}"

            # Validation
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history["val_loss"].append(round(val_loss, 4))
                history["val_accuracy"].append(round(val_acc, 4))
                msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.2%}"

            print(msg)

        # Put model back in eval mode
        self.model.eval()
        print("[Fine-tune] Done.")

        if save_path:
            self.save(save_path)

        return history

    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one validation pass and return (loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(loader), correct / total if total else 0.0

    # -----------------------------------------------------------------
    # Dunder methods
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SentimentAnalyzer(model='{self.model_name}', "
            f"device={self.device}, max_length={self.max_length})"
        )


# =========================================================================
# Helper Dataset for fine-tuning
# =========================================================================

class _SentimentDataset(Dataset):
    """PyTorch Dataset for fine-tuning a sequence-classification model."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# =========================================================================
# CLI quick test
# =========================================================================

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    test_texts = [
        "Apple reports record quarterly revenue of $124 billion",
        "Stocks plunge amid fears of global recession",
        "The company held its annual general meeting on Tuesday",
        "Revenue missed analyst expectations, guidance cut significantly",
        "Strong demand drives 40% growth in cloud computing division",
    ]

    print("\n" + "=" * 65)
    print("  Quick Sentiment Test")
    print("=" * 65)

    for text in test_texts:
        result = analyzer.predict(text)
        print(f"\n  Text : {text}")
        print(f"  Label: {result['label']} "
              f"(conf={result['confidence']:.2%}, score={result['score']:+.3f})")
        print(f"  P(pos)={result['positive']:.3f}  "
              f"P(neg)={result['negative']:.3f}  "
              f"P(neu)={result['neutral']:.3f}")
