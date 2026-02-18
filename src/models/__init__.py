"""
Models sub-package.

Exposes the ``SentimentAnalyzer`` class for convenient imports::

    from src.models import SentimentAnalyzer
"""

try:
    from src.models.sentiment_analyzer import SentimentAnalyzer
    __all__ = ["SentimentAnalyzer"]
except ImportError:
    # torch / transformers not installed — skip SentimentAnalyzer
    __all__ = []
