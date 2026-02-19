"""
Financial Sentiment Dashboard — Streamlit App
==============================================

Interactive dashboard that overlays FinBERT sentiment analysis on stock
price data for AAPL and MSFT.  Imports directly from the project's
``src/`` modules.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src.*` imports work when
# Streamlit is launched from the repo root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scipy import stats as sp_stats

import datetime as _dt

from src.data.data_loader import (
    load_all_news,
    load_merged_dataset,
    add_session_column,
    DEFAULT_TICKERS,
    DEFAULT_START,
    DEFAULT_END,
)
from src.data.news_fetcher import fetch_and_cache
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.return_predictor import ReturnPredictor, prepare_features

# =========================================================================
# Page configuration
# =========================================================================

st.set_page_config(
    page_title="Financial Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# Custom CSS
# =========================================================================

st.markdown(
    """
    <style>
    /* Signal cards */
    .signal-card {
        padding: 1.2rem;
        border-radius: 0.75rem;
        text-align: center;
        color: white;
        font-weight: 600;
    }
    .bullish  { background: linear-gradient(135deg, #00c853, #009624); }
    .bearish  { background: linear-gradient(135deg, #ff1744, #c4001d); }
    .neutral-card { background: linear-gradient(135deg, #ffc107, #ff8f00); }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 0.75rem;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.3rem;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .metric-card .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    .delta-pos { color: #00c853; }
    .delta-neg { color: #ff1744; }
    .delta-neutral { color: #666; }

    /* Positive / negative sentiment colours in dataframes */
    .pos-sent { color: #00c853; font-weight: 600; }
    .neg-sent { color: #ff1744; font-weight: 600; }

    /* Live news headline rows */
    .live-row {
        padding: 0.6rem 0;
        border-bottom: 1px solid #eee;
    }
    .live-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    .badge-pos  { background: #00c853; }
    .badge-neg  { background: #ff1744; }
    .badge-neu  { background: #ffc107; color: #333; }

    /* Alert simulation cards */
    .alert-card {
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .alert-strong-buy  { background: linear-gradient(135deg, #00e676, #00c853); }
    .alert-buy         { background: linear-gradient(135deg, #69f0ae, #00c853); }
    .alert-hold        { background: linear-gradient(135deg, #ffc107, #ff8f00); }
    .alert-sell         { background: linear-gradient(135deg, #ff5252, #ff1744); }
    .alert-strong-sell  { background: linear-gradient(135deg, #ff1744, #b71c1c); }
    .alert-no-data     { background: linear-gradient(135deg, #757575, #424242); }

    .rule-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid #eee;
        font-size: 0.9rem;
    }
    .rule-pass { color: #00c853; font-weight: 600; }
    .rule-fail { color: #ff1744; font-weight: 600; }
    .rule-neutral { color: #888; }

    /* XGBoost prediction cards */
    .pred-card {
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .pred-up   { background: linear-gradient(135deg, #00c853, #009624); }
    .pred-down { background: linear-gradient(135deg, #ff1744, #c4001d); }
    .pred-neutral { background: linear-gradient(135deg, #ffc107, #ff8f00); }
    .feature-bar {
        display: flex;
        align-items: center;
        padding: 0.25rem 0;
        font-size: 0.85rem;
    }
    .feature-bar .fname {
        width: 45%;
        text-align: right;
        padding-right: 0.8rem;
        color: #555;
    }
    .feature-bar .fval {
        flex: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# Sidebar — Controls
# =========================================================================

with st.sidebar:
    st.title("📊 Controls")
    st.markdown("---")

    # Ticker selector
    selected_ticker = st.selectbox(
        "**Select Ticker**",
        options=DEFAULT_TICKERS,
        index=0,
        help="Choose a stock ticker to analyse.",
    )

    st.markdown("---")

    # Date range — derive bounds dynamically from the actual data
    # so the picker always covers whatever CSVs exist on disk.
    @st.cache_data(show_spinner=False)
    def _data_date_bounds():
        """Return (min_date, max_date) from the merged CSV."""
        try:
            _tmp = load_merged_dataset()
            _tmp["date"] = pd.to_datetime(_tmp["date"])
            return _tmp["date"].min(), _tmp["date"].max()
        except Exception:
            return pd.Timestamp(DEFAULT_START), pd.Timestamp(DEFAULT_END)

    _data_min, _data_max = _data_date_bounds()

    st.subheader("Date Range")
    date_range = st.date_input(
        "Select range",
        value=(_data_min.date(), _data_max.date()),
        min_value=_data_min.date(),
        max_value=_data_max.date(),
        help="Filter data within the available date window.",
    )

    # Normalise tuple (user might pick a single date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        sel_start, sel_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        sel_start, sel_end = _data_min, _data_max

    st.markdown("---")

    # --- Data freshness badge ---
    _csv_path = _PROJECT_ROOT / "data" / "processed" / f"merged_{DEFAULT_START}_to_{DEFAULT_END}.csv"
    if _csv_path.exists():
        import os as _os
        _mtime = _dt.datetime.fromtimestamp(_os.path.getmtime(_csv_path))
        _hours_ago = (_dt.datetime.now() - _mtime).total_seconds() / 3600
        if _hours_ago < 1:
            _fresh_txt = "just now"
        elif _hours_ago < 24:
            _fresh_txt = f"{_hours_ago:.0f}h ago"
        else:
            _fresh_txt = f"{_hours_ago / 24:.0f}d ago"
        _fresh_color = "#00c853" if _hours_ago < 24 else "#ff9800" if _hours_ago < 72 else "#f44336"
        st.markdown(
            f'<div style="background:{_fresh_color}22; border:1px solid {_fresh_color};'
            f' border-radius:8px; padding:6px 10px; text-align:center; margin-bottom:6px;">'
            f'<span style="color:{_fresh_color}; font-weight:600;">📡 Data updated: {_fresh_txt}</span><br>'
            f'<span style="font-size:0.75rem; opacity:0.8;">{_mtime.strftime("%Y-%m-%d %H:%M")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Merged dataset not found — run the data pipeline first.")

    st.caption("Data: Alpha Vantage + Yahoo Finance")
    st.caption("Model: ProsusAI/FinBERT (fine-tuned)")
    st.caption("XGBoost: Return direction predictor")

# =========================================================================
# Data loading (cached)
# =========================================================================

@st.cache_data(show_spinner="Loading merged dataset …")
def get_merged() -> pd.DataFrame:
    """Load the full merged dataset once and cache it."""
    df = load_merged_dataset()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="Loading news articles …")
def get_news() -> pd.DataFrame:
    """Load raw news with session info."""
    news = load_all_news()
    news = add_session_column(news)
    news["published_at"] = pd.to_datetime(news["published_at"])
    news["date"] = news["published_at"].dt.floor("D")
    return news


@st.cache_resource(show_spinner="Loading XGBoost model …")
def load_xgb_model():
    """Load the trained XGBoost return predictor (multi-horizon 1d model)."""
    model_dir = _PROJECT_ROOT / "models" / "saved_models" / "xgboost_return_1d"
    if not (model_dir / "model.json").exists():
        return None
    return ReturnPredictor.load(model_dir)


@st.cache_resource(show_spinner="Loading multi-horizon models …")
def load_multihorizon_models():
    """Load XGBoost models for 1d, 5d, and 20d horizons.

    Returns dict mapping horizon -> ReturnPredictor (or None if missing).
    """
    models = {}
    for hz in ("1d", "5d", "20d"):
        model_dir = _PROJECT_ROOT / "models" / "saved_models" / f"xgboost_return_{hz}"
        if (model_dir / "model.json").exists():
            models[hz] = ReturnPredictor.load(model_dir)
        else:
            models[hz] = None
    return models

merged_all = get_merged()
news_all = get_news()

# =========================================================================
# Filter by ticker + date range
# =========================================================================

mask_merged = (
    (merged_all["ticker"] == selected_ticker)
    & (merged_all["date"] >= sel_start)
    & (merged_all["date"] <= sel_end)
)
df = merged_all.loc[mask_merged].copy().sort_values("date").reset_index(drop=True)

mask_news = (
    (news_all["ticker"] == selected_ticker)
    & (news_all["date"] >= sel_start)
    & (news_all["date"] <= sel_end)
)
news_df = news_all.loc[mask_news].copy().sort_values("published_at", ascending=False).reset_index(drop=True)

# =========================================================================
# Header
# =========================================================================

st.title(f"📈 {selected_ticker} — Financial Sentiment Dashboard")
st.markdown(
    f"Analysing **{len(df)}** trading days and **{len(news_df)}** news articles "
    f"from **{sel_start.date()}** to **{sel_end.date()}**."
)
st.markdown("---")

# =========================================================================
# Row 1 — Signal Card + Key Metrics
# =========================================================================

if not df.empty:
    latest = df.iloc[-1]
    latest_sentiment = latest.get("avg_overall_sentiment", 0)
    if pd.isna(latest_sentiment):
        latest_sentiment = 0.0

    # Determine signal
    if latest_sentiment > 0.15:
        signal_label = "BULLISH"
        signal_css = "bullish"
    elif latest_sentiment < -0.15:
        signal_label = "BEARISH"
        signal_css = "bearish"
    else:
        signal_label = "NEUTRAL"
        signal_css = "neutral-card"

    confidence_pct = min(abs(latest_sentiment) / 0.5 * 100, 100)  # scale to 0-100

    col_signal, col_price, col_return, col_articles, col_rolling = st.columns(
        [1.4, 1, 1, 1, 1]
    )

    with col_signal:
        st.markdown(
            f"""
            <div class="signal-card {signal_css}">
                <div style="font-size:0.85rem;opacity:0.9;">Latest Signal</div>
                <div style="font-size:2rem;">{signal_label}</div>
                <div style="font-size:0.9rem;">Confidence: {confidence_pct:.0f}%</div>
                <div style="font-size:0.75rem;opacity:0.8;">
                    Sentiment: {latest_sentiment:+.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_price:
        daily_ret = latest.get('daily_return', 0)
        delta_val = f"{daily_ret:.2f}%" if not pd.isna(daily_ret) else ""
        delta_css = "delta-pos" if daily_ret >= 0 else "delta-neg"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Last Close</div>'
            f'<div class="metric-value">${latest["Close"]:.2f}</div>'
            f'<div class="metric-delta {delta_css}">{delta_val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_return:
        avg_ret = df["daily_return"].mean()
        ret_css = "delta-pos" if avg_ret >= 0 else "delta-neg"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Avg Daily Return</div>'
            f'<div class="metric-value {ret_css}">{avg_ret:+.3f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_articles:
        total_articles = int(df["article_count"].sum())
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Total Articles</div>'
            f'<div class="metric-value">{total_articles}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_rolling:
        roll_5 = latest.get("sentiment_rolling_5d", np.nan)
        roll_str = f"{roll_5:+.3f}" if not pd.isna(roll_5) else "N/A"
        roll_css = "delta-pos" if (not pd.isna(roll_5) and roll_5 >= 0) else "delta-neg" if (not pd.isna(roll_5) and roll_5 < 0) else "delta-neutral"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">5-Day Sentiment</div>'
            f'<div class="metric-value {roll_css}">{roll_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =====================================================================
    # Row 2 — Sentiment × Price Chart (dual axis)
    # =====================================================================

    st.subheader("📉 Stock Price & Sentiment Over Time")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Stock Price (Close)", "Daily Sentiment Score"),
    )

    # --- Price line ---
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["Close"],
            mode="lines+markers",
            name="Close Price",
            line=dict(color="#1976d2", width=2),
            marker=dict(size=4),
            hovertemplate="$%{y:.2f}<extra>Close</extra>",
        ),
        row=1,
        col=1,
    )

    # --- Sentiment bar ---
    sent = df["avg_overall_sentiment"].fillna(0)
    colors = ["#00c853" if v >= 0 else "#ff1744" for v in sent]

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=sent,
            name="Avg Sentiment",
            marker_color=colors,
            opacity=0.75,
            hovertemplate="%{y:+.3f}<extra>Sentiment</extra>",
        ),
        row=2,
        col=1,
    )

    # Rolling sentiment overlay
    if "sentiment_rolling_5d" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["sentiment_rolling_5d"],
                mode="lines",
                name="5-Day Rolling",
                line=dict(color="#ff9800", width=2, dash="dash"),
                hovertemplate="%{y:+.3f}<extra>5d Rolling</extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=550,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=50, r=30, t=40, b=50),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1)

    st.plotly_chart(fig, width="stretch")

    # =====================================================================
    # Row 3 — News Feed + Sentiment Distribution (side by side)
    # =====================================================================

    col_news, col_dist = st.columns([3, 2])

    with col_news:
        st.subheader("📰 News Feed")

        if news_df.empty:
            st.info("No articles found for this ticker and date range.")
        else:
            display_df = news_df[["published_at", "title", "source",
                                   "overall_sentiment_score",
                                   "overall_sentiment_label"]].copy()
            display_df.columns = ["Published", "Headline", "Source",
                                   "Score", "Sentiment"]

            display_df["Published"] = display_df["Published"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )

            def _color_sentiment(val):
                if val == "Bullish":
                    return "color: #00c853; font-weight: 600"
                elif val == "Bearish":
                    return "color: #ff1744; font-weight: 600"
                return "color: #ff8f00; font-weight: 600"

            def _color_score(val):
                try:
                    v = float(val)
                    if v > 0.15:
                        return "color: #00c853; font-weight: 600"
                    elif v < -0.15:
                        return "color: #ff1744; font-weight: 600"
                    return "color: #ff8f00"
                except (ValueError, TypeError):
                    return ""

            styled = (
                display_df.style
                .map(_color_sentiment, subset=["Sentiment"])
                .map(_color_score, subset=["Score"])
                .format({"Score": "{:+.3f}"})
            )
            st.dataframe(styled, width="stretch", height=420)

    with col_dist:
        st.subheader("📊 Sentiment Distribution")

        if not news_df.empty:
            # Donut chart of sentiment labels
            label_counts = news_df["overall_sentiment_label"].value_counts()
            donut_colors = {
                "Bullish": "#00c853",
                "Bearish": "#ff1744",
                "Neutral": "#ffc107",
                "Somewhat-Bullish": "#66bb6a",
                "Somewhat-Bearish": "#ef5350",
            }
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=label_counts.index,
                        values=label_counts.values,
                        hole=0.45,
                        marker=dict(
                            colors=[
                                donut_colors.get(l, "#90a4ae")
                                for l in label_counts.index
                            ]
                        ),
                        textinfo="label+percent",
                        hovertemplate="%{label}: %{value} articles<extra></extra>",
                    )
                ]
            )
            fig_donut.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_donut, width="stretch")

            # Histogram of sentiment scores
            st.markdown("**Score Histogram**")
            fig_hist = go.Figure(
                data=[
                    go.Histogram(
                        x=news_df["overall_sentiment_score"].dropna(),
                        nbinsx=25,
                        marker_color="#1976d2",
                        opacity=0.8,
                        hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
                    )
                ]
            )
            fig_hist.update_layout(
                height=200,
                xaxis_title="Sentiment Score",
                yaxis_title="Count",
                template="plotly_white",
                margin=dict(l=40, r=20, t=10, b=40),
            )
            st.plotly_chart(fig_hist, width="stretch")
        else:
            st.info("No articles to display.")

    st.markdown("---")

    # =====================================================================
    # Row 4 — Correlation & Performance Metrics
    # =====================================================================

    st.subheader("📐 Sentiment–Price Relationship")

    col_scatter, col_metrics = st.columns([3, 2])

    with col_scatter:
        valid = df.dropna(subset=["avg_overall_sentiment", "daily_return"])
        if len(valid) >= 5:
            fig_scatter = go.Figure(
                data=[
                    go.Scatter(
                        x=valid["avg_overall_sentiment"],
                        y=valid["daily_return"],
                        mode="markers",
                        marker=dict(
                            color=valid["daily_return"],
                            colorscale="RdYlGn",
                            size=8,
                            line=dict(width=0.5, color="white"),
                            showscale=True,
                            colorbar=dict(title="Return %"),
                        ),
                        hovertemplate=(
                            "Sentiment: %{x:+.3f}<br>"
                            "Return: %{y:+.2f}%<br>"
                            "<extra></extra>"
                        ),
                    )
                ]
            )
            # Trend line
            z = np.polyfit(valid["avg_overall_sentiment"], valid["daily_return"], 1)
            x_line = np.linspace(
                valid["avg_overall_sentiment"].min(),
                valid["avg_overall_sentiment"].max(),
                50,
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_line,
                    y=np.polyval(z, x_line),
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.4)", dash="dash", width=2),
                    name="Trend",
                    showlegend=False,
                )
            )
            fig_scatter.update_layout(
                xaxis_title="Avg Daily Sentiment",
                yaxis_title="Daily Return (%)",
                height=350,
                template="plotly_white",
                margin=dict(l=50, r=30, t=20, b=50),
            )
            st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info("Not enough data points for a scatter plot.")

    with col_metrics:
        valid = df.dropna(subset=["avg_overall_sentiment", "daily_return"])
        if len(valid) >= 5:
            pearson_r, pearson_p = sp_stats.pearsonr(
                valid["avg_overall_sentiment"], valid["daily_return"]
            )
            spearman_r, spearman_p = sp_stats.spearmanr(
                valid["avg_overall_sentiment"], valid["daily_return"]
            )
            pred_dir = (valid["avg_overall_sentiment"] > 0).astype(int)
            dir_acc = (pred_dir == valid["return_direction"]).mean()

            st.markdown(
                '<div class="metric-card" style="margin-bottom:0.75rem;">'
                '<div class="metric-label">Pearson r</div>'
                f'<div class="metric-value">{pearson_r:+.4f}</div>'
                f'<div class="metric-delta delta-neutral">p = {pearson_p:.4f}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-card" style="margin-bottom:0.75rem;">'
                '<div class="metric-label">Spearman ρ</div>'
                f'<div class="metric-value">{spearman_r:+.4f}</div>'
                f'<div class="metric-delta delta-neutral">p = {spearman_p:.4f}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            acc_css = "delta-pos" if dir_acc > 0.5 else "delta-neg"
            st.markdown(
                '<div class="metric-card" style="margin-bottom:0.75rem;">'
                '<div class="metric-label">Direction Accuracy</div>'
                f'<div class="metric-value {acc_css}">{dir_acc:.1%}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Direction accuracy = % of days where sentiment sign "
                "matches return direction."
            )
        else:
            st.info("Not enough data for correlation analysis.")

else:
    st.markdown(
        '<div style="background:#23272e; border:1px solid #444; border-radius:12px;'
        ' padding:2.5rem; text-align:center; margin:1rem 0;">'
        '<div style="font-size:3rem;">📉</div>'
        '<div style="font-size:1.2rem; margin-top:0.5rem;">No price data found for '
        f'<b>{selected_ticker}</b> in this date range</div>'
        '<div style="font-size:0.85rem; opacity:0.7; margin-top:0.4rem;">'
        'Try adjusting the date range in the sidebar, or make sure price data has been '
        'downloaded with <code>python scripts/download_prices.py</code></div>'
        '</div>',
        unsafe_allow_html=True,
    )

# =========================================================================
# Live Headlines Section
# =========================================================================

st.markdown("---")
st.subheader("🔴 Live Headlines — Real-Time Sentiment")
st.caption(
    "Fresh headlines from Google News, Yahoo Finance RSS, yfinance API & "
    "SEC EDGAR 8-K filings, scored by FinBERT in real time.  "
    "EDGAR filings provide a more balanced sentiment (neutral/negative) "
    "to offset the bullish bias of news headlines.  Cache refreshes every hour."
)


@st.cache_resource(show_spinner="Loading FinBERT model …")
def load_model() -> SentimentAnalyzer:
    """Load the sentiment model once (singleton across reruns)."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "saved_models" / "finbert_finetuned"
    if model_path.exists():
        return SentimentAnalyzer.load(model_path)
    return SentimentAnalyzer()  # fallback to base ProsusAI/finbert


@st.cache_data(show_spinner="Fetching live news …", ttl=3600)
def get_live_news(ticker: str) -> pd.DataFrame:
    """Fetch live RSS news with 1-hour cache."""
    return fetch_and_cache(ticker, ttl=3600)


@st.cache_data(show_spinner="Scoring headlines with FinBERT …", ttl=3600)
def score_live_headlines(ticker: str) -> pd.DataFrame:
    """Fetch live headlines and score them with FinBERT."""
    raw = get_live_news(ticker)
    if raw.empty:
        return raw

    analyzer = load_model()

    # Run FinBERT on titles (headlines carry most signal)
    titles = raw["title"].fillna("").tolist()
    preds = analyzer.predict_batch(titles, batch_size=16, show_progress=False)

    raw = raw.copy()
    raw["sentiment_score"] = [p["score"] for p in preds]
    raw["sentiment_label"] = [p["label"] for p in preds]
    raw["sentiment_conf"] = [p["confidence"] for p in preds]
    raw["prob_positive"] = [p["positive"] for p in preds]
    raw["prob_negative"] = [p["negative"] for p in preds]
    raw["prob_neutral"] = [p["neutral"] for p in preds]
    return raw


live_df = score_live_headlines(selected_ticker)

if live_df.empty:
    st.markdown(
        '<div style="background:#23272e; border:1px solid #444; border-radius:12px;'
        ' padding:2rem; text-align:center; margin:1rem 0;">'
        '<div style="font-size:2.5rem;">📭</div>'
        f'<div style="font-size:1.1rem; margin-top:0.5rem;">No live headlines found for <b>{selected_ticker}</b></div>'
        '<div style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">'
        'RSS feeds may be temporarily unavailable or this ticker may not have recent coverage.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    # ---- Live signal card row ----
    live_scores = live_df["sentiment_score"].dropna()
    live_avg = float(live_scores.mean()) if len(live_scores) else 0.0

    if live_avg > 0.15:
        lbl, css = "BULLISH", "bullish"
    elif live_avg < -0.15:
        lbl, css = "BEARISH", "bearish"
    else:
        lbl, css = "NEUTRAL", "neutral-card"
    live_conf = min(abs(live_avg) / 0.5 * 100, 100)

    lc1, lc2, lc3 = st.columns([1.4, 1, 1])
    with lc1:
        st.markdown(
            f'<div class="signal-card {css}">'
            f'<div style="font-size:0.85rem;opacity:0.9;">Live Signal</div>'
            f'<div style="font-size:2rem;">{lbl}</div>'
            f'<div style="font-size:0.9rem;">Confidence: {live_conf:.0f}%</div>'
            f'<div style="font-size:0.75rem;opacity:0.8;">'
            f'Avg score: {live_avg:+.3f} · {len(live_df)} articles</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with lc2:
        pct_pos = (live_scores > 0.1).mean() * 100
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Positive %</div>'
            f'<div class="metric-value delta-pos">{pct_pos:.0f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with lc3:
        pct_neg = (live_scores < -0.1).mean() * 100
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Negative %</div>'
            f'<div class="metric-value delta-neg">{pct_neg:.0f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---- Build table with Impact column ----
    tbl = live_df[["published_at", "title", "source",
                    "sentiment_score", "sentiment_label",
                    "sentiment_conf"]].copy()
    tbl.columns = ["Published", "Headline", "Source",
                    "Score", "Sentiment", "Confidence"]
    tbl["Published"] = pd.to_datetime(tbl["Published"]).dt.strftime("%Y-%m-%d %H:%M")
    tbl["Confidence"] = (tbl["Confidence"] * 100).round(1)

    # Impact = |score| × confidence  (0–100 scale)
    # Headlines with strong sentiment AND high model certainty rank highest
    tbl["Impact"] = (tbl["Score"].abs() * tbl["Confidence"]).round(1)

    # ---- Sort controls ----
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort headlines by",
            ["Impact (strongest first)", "Newest first", "Score (most positive)", "Score (most negative)"],
            index=0,
            label_visibility="collapsed",
        )

    sort_map = {
        "Impact (strongest first)": ("Impact", False),
        "Newest first": ("Published", False),
        "Score (most positive)": ("Score", False),
        "Score (most negative)": ("Score", True),
    }
    col, asc = sort_map[sort_by]
    tbl = tbl.sort_values(col, ascending=asc).reset_index(drop=True)

    # ---- Top-5 impact bar chart ----
    top5 = tbl.head(5).copy()
    if sort_by == "Impact (strongest first)" and len(top5) > 0:
        # Truncate long headlines for the chart
        top5["Short"] = top5["Headline"].str[:60].where(
            top5["Headline"].str.len() <= 60,
            top5["Headline"].str[:57] + "…"
        )
        top5["Color"] = top5["Score"].apply(
            lambda s: "#00c853" if s > 0.1 else ("#ff1744" if s < -0.1 else "#ff8f00")
        )

        fig_impact = go.Figure(go.Bar(
            x=top5["Impact"].values[::-1],
            y=top5["Short"].values[::-1],
            orientation="h",
            marker_color=top5["Color"].values[::-1],
            text=[f"{s:+.2f}" for s in top5["Score"].values[::-1]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.1f}<br>Score: %{text}<extra></extra>",
        ))
        fig_impact.update_layout(
            title="Top 5 Most Impactful Headlines",
            xaxis_title="Impact (|score| × confidence)",
            yaxis_title="",
            height=260,
            margin=dict(l=20, r=20, t=40, b=30),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_impact, key="impact_chart", width="stretch")

    # ---- Color-coded headline table ----
    def _clr_label(val):
        colors = {"positive": "#00c853", "negative": "#ff1744", "neutral": "#ff8f00"}
        c = colors.get(str(val).lower(), "#666")
        return f"color: {c}; font-weight: 600"

    def _clr_score(val):
        try:
            v = float(val)
            if v > 0.1:
                return "color: #00c853; font-weight: 600"
            elif v < -0.1:
                return "color: #ff1744; font-weight: 600"
            return "color: #ff8f00"
        except (ValueError, TypeError):
            return ""

    def _clr_impact(val):
        try:
            v = float(val)
            if v > 30:
                return "color: #e040fb; font-weight: 700"
            elif v > 15:
                return "color: #7c4dff; font-weight: 600"
            return "opacity: 0.7"
        except (ValueError, TypeError):
            return ""

    styled_live = (
        tbl.style
        .map(_clr_label, subset=["Sentiment"])
        .map(_clr_score, subset=["Score"])
        .map(_clr_impact, subset=["Impact"])
        .format({"Score": "{:+.3f}", "Confidence": "{:.1f}%", "Impact": "{:.1f}"})
    )
    st.dataframe(styled_live, width="stretch", height=420)

# =========================================================================
# Multi-Ticker Comparison
# =========================================================================

st.markdown("---")
st.subheader("📊 Multi-Ticker Sentiment Comparison")
st.caption(
    "Compare live FinBERT sentiment across multiple tickers.  "
    "Select up to 6 tickers below."
)

# Ticker input — default to project tickers + a few popular ones
_COMPARISON_DEFAULTS = ["AAPL", "MSFT", "GOOGL", "NVDA"]
_POPULAR_TICKERS = DEFAULT_TICKERS

compare_tickers = st.multiselect(
    "Tickers to compare",
    options=_POPULAR_TICKERS,
    default=_COMPARISON_DEFAULTS,
    max_selections=6,
    help="Pick 2–6 tickers. Each one fetches live news + runs FinBERT.",
)

if len(compare_tickers) >= 2:
    # Fetch & score each ticker (reuses the cached functions above)
    comp_rows: list[dict] = []
    with st.spinner("Fetching & scoring headlines for all tickers …"):
        for tk in compare_tickers:
            tk_df = score_live_headlines(tk)
            if tk_df.empty:
                comp_rows.append({
                    "Ticker": tk, "Articles": 0, "Avg Score": 0.0,
                    "Positive %": 0.0, "Negative %": 0.0,
                    "Top Impact": 0.0, "Signal": "NO DATA",
                })
                continue
            scores = tk_df["sentiment_score"].dropna()
            avg = float(scores.mean())
            ppos = float((scores > 0.1).mean() * 100)
            pneg = float((scores < -0.1).mean() * 100)
            impact_vals = (scores.abs() * tk_df["sentiment_conf"].dropna()).dropna()
            top_imp = float(impact_vals.max() * 100) if len(impact_vals) else 0.0
            if avg > 0.15:
                sig = "BULLISH"
            elif avg < -0.15:
                sig = "BEARISH"
            else:
                sig = "NEUTRAL"
            comp_rows.append({
                "Ticker": tk, "Articles": len(tk_df),
                "Avg Score": round(avg, 4),
                "Positive %": round(ppos, 1),
                "Negative %": round(pneg, 1),
                "Top Impact": round(top_imp, 1),
                "Signal": sig,
            })

    comp_df = pd.DataFrame(comp_rows)

    # ---- Summary cards row ----
    card_cols = st.columns(len(compare_tickers))
    for idx, row in comp_df.iterrows():
        sig_css = {
            "BULLISH": "bullish", "BEARISH": "bearish",
            "NEUTRAL": "neutral-card", "NO DATA": "neutral-card",
        }.get(row["Signal"], "neutral-card")
        with card_cols[idx]:
            st.markdown(
                f'<div class="signal-card {sig_css}" style="padding:0.8rem;">'
                f'<div style="font-size:1.2rem;font-weight:700;">{row["Ticker"]}</div>'
                f'<div style="font-size:1.4rem;">{row["Signal"]}</div>'
                f'<div style="font-size:0.8rem;opacity:0.9;">'
                f'Score: {row["Avg Score"]:+.3f} · {row["Articles"]} articles</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ---- Grouped bar chart: Avg Score per ticker ----
    bar_colors = [
        "#00c853" if s > 0.1 else ("#ff1744" if s < -0.1 else "#ff8f00")
        for s in comp_df["Avg Score"]
    ]

    fig_comp = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Average Sentiment Score", "Positive vs Negative %"],
        horizontal_spacing=0.12,
    )

    # Left: avg score bars
    fig_comp.add_trace(
        go.Bar(
            x=comp_df["Ticker"], y=comp_df["Avg Score"],
            marker_color=bar_colors,
            text=[f"{s:+.3f}" for s in comp_df["Avg Score"]],
            textposition="outside",
            name="Avg Score",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Right: stacked positive / negative %
    fig_comp.add_trace(
        go.Bar(
            x=comp_df["Ticker"], y=comp_df["Positive %"],
            marker_color="#00c853", name="Positive %",
            text=[f"{v:.0f}%" for v in comp_df["Positive %"]],
            textposition="inside",
        ),
        row=1, col=2,
    )
    fig_comp.add_trace(
        go.Bar(
            x=comp_df["Ticker"], y=comp_df["Negative %"],
            marker_color="#ff1744", name="Negative %",
            text=[f"{v:.0f}%" for v in comp_df["Negative %"]],
            textposition="inside",
        ),
        row=1, col=2,
    )

    fig_comp.update_layout(
        height=350,
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    fig_comp.update_yaxes(title_text="Score", row=1, col=1)
    fig_comp.update_yaxes(title_text="%", row=1, col=2)

    st.plotly_chart(fig_comp, key="comp_chart", width="stretch")

    # ---- Comparison table ----
    def _clr_signal(val):
        c = {"BULLISH": "#00c853", "BEARISH": "#ff1744", "NEUTRAL": "#ff8f00",
             "NO DATA": "#666"}.get(val, "#666")
        return f"color: {c}; font-weight: 700"

    styled_comp = (
        comp_df.style
        .map(_clr_signal, subset=["Signal"])
        .format({
            "Avg Score": "{:+.4f}",
            "Positive %": "{:.1f}%",
            "Negative %": "{:.1f}%",
            "Top Impact": "{:.1f}",
        })
    )
    st.dataframe(styled_comp, width="stretch", hide_index=True)

elif len(compare_tickers) == 1:
    st.markdown(
        '<div style="background:#23272e; border:1px solid #444; border-radius:12px;'
        ' padding:1.5rem; text-align:center;">'
        '<div style="font-size:1.5rem;">👆</div>'
        '<div style="font-size:0.95rem; margin-top:0.3rem;">Select at least <b>2 tickers</b> to compare side-by-side.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="background:#23272e; border:1px solid #444; border-radius:12px;'
        ' padding:1.5rem; text-align:center;">'
        '<div style="font-size:1.5rem;">📊</div>'
        '<div style="font-size:0.95rem; margin-top:0.3rem;">Select tickers above to see a live sentiment comparison.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# =========================================================================
# Alert Simulation Panel
# =========================================================================

st.markdown("---")
st.subheader("🚨 Alert Simulation — Rule-Based Signal Engine")
st.caption(
    "**How it works:** Live news headlines are fetched from Google News RSS and scored by FinBERT "
    "(sentiment model). The engine then aggregates those scores and checks them against the "
    "thresholds you set below. If all rules pass → BUY/SELL signal; otherwise → HOLD.  \n"
    "This is **not** an ML prediction — it is a transparent, deterministic rule engine. "
    "No XGBoost or black-box model is involved; you control every threshold."
)

# ---- Pre-compute live data stats so sliders can show hints ----
alert_df = score_live_headlines(selected_ticker)

_has_alert_data = not (alert_df.empty or len(alert_df) == 0)
if _has_alert_data:
    _a_scores = alert_df["sentiment_score"].dropna()
    _a_confs = alert_df["sentiment_conf"].dropna()
    _a_avg = float(_a_scores.mean())
    _a_avg_conf = float(_a_confs.mean()) * 100
    _a_pct_pos = float((_a_scores > 0.1).mean()) * 100
    _a_pct_neg = float((_a_scores < -0.1).mean()) * 100
    _a_count = len(alert_df)
    _a_dominant_pct = max(_a_pct_pos, _a_pct_neg)
    _a_direction = "positive" if _a_pct_pos >= _a_pct_neg else "negative"
    _hint_score = f"Live: {abs(_a_avg):.3f}"
    _hint_conf = f"Live: {_a_avg_conf:.0f}%"
    _hint_dom = f"Live: {_a_dominant_pct:.0f}%"
    _hint_count = f"Live: {_a_count}"
else:
    _hint_score = _hint_conf = _hint_dom = _hint_count = "No data"

# ---- Threshold controls with live data hints ----
alert_c1, alert_c2, alert_c3, alert_c4 = st.columns(4)

with alert_c1:
    thresh_score = st.slider(
        "Min |avg score| for signal",
        min_value=0.01, max_value=0.50, value=0.10, step=0.01,
        help="How strong must the average sentiment be to trigger a BUY/SELL?",
        key="alert_thresh_score",
    )
    st.caption(f"📊 {_hint_score}")
with alert_c2:
    thresh_conf = st.slider(
        "Min confidence %",
        min_value=10, max_value=90, value=35, step=5,
        help="Minimum avg per-headline confidence to trust the signal.",
        key="alert_thresh_conf",
    )
    st.caption(f"📊 {_hint_conf}")
with alert_c3:
    thresh_dominance = st.slider(
        "Min dominant % (pos or neg)",
        min_value=20, max_value=90, value=45, step=5,
        help="What % of headlines must lean the same way?",
        key="alert_thresh_dom",
    )
    st.caption(f"📊 {_hint_dom}")
with alert_c4:
    thresh_articles = st.slider(
        "Min article count",
        min_value=1, max_value=50, value=5, step=1,
        help="Require at least this many articles for a signal.",
        key="alert_thresh_articles",
    )
    st.caption(f"📊 {_hint_count}")

st.markdown("")

# Strong signal requires higher thresholds
strong_score = thresh_score * 2
strong_dominance = min(thresh_dominance + 15, 95)

# ---- Evaluate rules against live data for the selected ticker ----

if not _has_alert_data:
    st.markdown(
        '<div class="alert-card alert-no-data">'
        '<div style="font-size:2.5rem;">NO DATA</div>'
        '<div style="font-size:0.9rem;opacity:0.9;">'
        f'No live headlines available for {selected_ticker}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    a_avg = _a_avg
    a_avg_conf = _a_avg_conf
    a_pct_pos = _a_pct_pos
    a_pct_neg = _a_pct_neg
    a_count = _a_count
    a_dominant_pct = _a_dominant_pct
    a_direction = _a_direction

    # ---- Rule evaluation ----
    rules = [
        {
            "name": f"Avg sentiment |score| ≥ {thresh_score:.2f}",
            "value": f"{abs(a_avg):.3f}",
            "passed": abs(a_avg) >= thresh_score,
        },
        {
            "name": f"Avg confidence ≥ {thresh_conf}%",
            "value": f"{a_avg_conf:.1f}%",
            "passed": a_avg_conf >= thresh_conf,
        },
        {
            "name": f"Dominant sentiment ≥ {thresh_dominance}%",
            "value": f"{a_dominant_pct:.1f}% {a_direction}",
            "passed": a_dominant_pct >= thresh_dominance,
        },
        {
            "name": f"Article count ≥ {thresh_articles}",
            "value": f"{a_count}",
            "passed": a_count >= thresh_articles,
        },
    ]

    # Strong-signal extra rules
    strong_rules = [
        {
            "name": f"Strong: |score| ≥ {strong_score:.2f}",
            "value": f"{abs(a_avg):.3f}",
            "passed": abs(a_avg) >= strong_score,
        },
        {
            "name": f"Strong: dominance ≥ {strong_dominance}%",
            "value": f"{a_dominant_pct:.1f}%",
            "passed": a_dominant_pct >= strong_dominance,
        },
    ]

    base_pass = all(r["passed"] for r in rules)
    strong_pass = base_pass and all(r["passed"] for r in strong_rules)
    n_passed = sum(r["passed"] for r in rules)

    # Determine signal
    if not base_pass:
        signal_label = "HOLD"
        signal_css = "alert-hold"
        signal_reason = f"{n_passed}/4 rules passed — insufficient conviction."
    elif strong_pass and a_avg > 0:
        signal_label = "STRONG BUY"
        signal_css = "alert-strong-buy"
        signal_reason = "All base + strong rules passed with positive bias."
    elif strong_pass and a_avg < 0:
        signal_label = "STRONG SELL"
        signal_css = "alert-strong-sell"
        signal_reason = "All base + strong rules passed with negative bias."
    elif a_avg > 0:
        signal_label = "BUY"
        signal_css = "alert-buy"
        signal_reason = "All 4 base rules passed — sentiment leans positive."
    else:
        signal_label = "SELL"
        signal_css = "alert-sell"
        signal_reason = "All 4 base rules passed — sentiment leans negative."

    # ---- Signal card ----
    sig_col, detail_col = st.columns([1, 1.6])

    with sig_col:
        st.markdown(
            f'<div class="alert-card {signal_css}">'
            f'<div style="font-size:1rem;opacity:0.9;">'
            f'{selected_ticker} — Alert Signal</div>'
            f'<div style="font-size:2.8rem;margin:0.3rem 0;">'
            f'{signal_label}</div>'
            f'<div style="font-size:0.85rem;opacity:0.9;">{signal_reason}</div>'
            f'<div style="font-size:0.75rem;opacity:0.7;margin-top:0.4rem;">'
            f'Based on {a_count} live headlines</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- Rule breakdown ----
    with detail_col:
        st.markdown("**Rule Breakdown**")

        # Show a progress indicator for how many rules pass
        st.progress(n_passed / 4, text=f"{n_passed}/4 base rules passed")

        all_rules = rules + strong_rules
        breakdown_html = '<div style="border:1px solid #e0e0e0; border-radius:0.5rem; overflow:hidden;">'
        for r in all_rules:
            icon = "✅" if r["passed"] else "❌"
            css_cls = "rule-pass" if r["passed"] else "rule-fail"
            # Add "BLOCKING" badge for failed base rules
            blocking = ""
            if not r["passed"] and not r["name"].startswith("Strong"):
                blocking = (
                    ' <span style="background:#ff1744;color:white;'
                    'font-size:0.6rem;padding:0.1rem 0.35rem;border-radius:0.3rem;'
                    'margin-left:0.4rem;vertical-align:middle;">BLOCKING</span>'
                )
            breakdown_html += (
                f'<div class="rule-row">'
                f'<span>{icon} {r["name"]}{blocking}</span>'
                f'<span class="{css_cls}">{r["value"]}</span>'
                f'</div>'
            )
        breakdown_html += '</div>'
        st.markdown(breakdown_html, unsafe_allow_html=True)

        if not base_pass:
            # Hint about which slider to adjust
            failing = [r["name"].split("≥")[0].strip() for r in rules if not r["passed"]]
            st.caption(
                f"💡 Adjust thresholds for: {', '.join(failing)} — "
                f"or wait for more headlines."
            )

    st.markdown("")

    # ---- Gauge chart: signal strength ----
    # Map signal to a 0-100 scale for a gauge visualization
    signal_strength = min(abs(a_avg) / 0.5 * 100, 100)
    gauge_color = {
        "STRONG BUY": "#00e676", "BUY": "#00c853",
        "HOLD": "#ff8f00",
        "SELL": "#ff5252", "STRONG SELL": "#ff1744",
    }.get(signal_label, "#888")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=signal_strength,
        title={"text": "Signal Strength", "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 28}},
        delta={"reference": 50, "increasing": {"color": "#00c853"},
               "decreasing": {"color": "#ff1744"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": gauge_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(255,23,68,0.15)"},
                {"range": [25, 50], "color": "rgba(255,193,7,0.15)"},
                {"range": [50, 75], "color": "rgba(105,240,174,0.15)"},
                {"range": [75, 100], "color": "rgba(0,200,83,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": signal_strength,
            },
        },
    ))
    fig_gauge.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=40, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(fig_gauge, key="alert_gauge", width="stretch")
    with g2:
        # Summary metrics
        st.markdown(
            '<div class="metric-card" style="margin-bottom:0.6rem;">'
            '<div class="metric-label">Avg Sentiment Score</div>'
            f'<div class="metric-value">{a_avg:+.4f}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-card" style="margin-bottom:0.6rem;">'
            '<div class="metric-label">Avg Model Confidence</div>'
            f'<div class="metric-value">{a_avg_conf:.1f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Dominant Direction</div>'
            f'<div class="metric-value">{a_dominant_pct:.0f}% {a_direction}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ---- Disclaimer ----
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem; "
        "margin-top:1rem;'>⚠️ This is a <b>simulation</b> for educational "
        "purposes only. Not financial advice. Always do your own research "
        "before making investment decisions.</div>",
        unsafe_allow_html=True,
    )

# =========================================================================
# Panel 9 — XGBoost Return Prediction
# =========================================================================

st.markdown("---")
st.header("🤖 XGBoost — Next-Day Return Prediction")

xgb_model = load_xgb_model()

if xgb_model is None:
    st.markdown(
        '<div style="background:#23272e; border:1px solid #444; border-radius:12px;'
        ' padding:2rem; text-align:center; margin:1rem 0;">'
        '<div style="font-size:2.5rem;">🤖</div>'
        '<div style="font-size:1.1rem; margin-top:0.5rem;">XGBoost model not trained yet</div>'
        '<div style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">'
        'Run <code>python scripts/train_multihorizon.py</code> to train the return predictor.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    # --- Run prediction for selected ticker ---
    try:
        pred_result = xgb_model.predict_next_day(merged_all, selected_ticker)
        direction = pred_result["direction"]
        prob_up = pred_result["prob_up"]
        prob_down = pred_result["prob_down"]
        confidence = pred_result["confidence"]
        based_on = pred_result["based_on_date"]

        # --- Prediction card ---
        xp1, xp2 = st.columns([1, 2])

        with xp1:
            if direction == "UP":
                card_cls = "pred-up"
                arrow = "▲"
                prob_shown = prob_up
            else:
                card_cls = "pred-down"
                arrow = "▼"
                prob_shown = prob_down

            st.markdown(
                f'<div class="pred-card {card_cls}">'
                f'<div style="font-size:2.5rem;">{arrow}</div>'
                f'<div style="font-size:1.8rem;">PREDICT {direction}</div>'
                f'<div style="font-size:1rem; opacity:0.9; margin-top:0.3rem;">'
                f'Probability: {prob_shown:.1%}</div>'
                f'<div style="font-size:0.85rem; opacity:0.75; margin-top:0.2rem;">'
                f'Confidence: {confidence:.1%} · Based on {based_on}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Model performance card
            acc = xgb_model.metrics.get("accuracy", 0)
            auc = xgb_model.metrics.get("roc_auc", 0)
            f1 = xgb_model.metrics.get("f1", 0)
            n_preds = xgb_model.metrics.get("n_predictions", 0)

            st.markdown(
                '<div class="metric-card" style="margin-top:0.8rem;">'
                '<div class="metric-label">Walk-Forward Performance</div>'
                f'<div class="metric-value">{acc:.1%} acc</div>'
                f'<div class="metric-delta delta-neutral">'
                f'AUC {auc:.3f} · F1 {f1:.1%} · n={n_preds}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with xp2:
            # --- Probability gauge ---
            fig_prob = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_up * 100,
                title={"text": "P(Up) vs P(Down)", "font": {"size": 14}},
                number={"suffix": "% up", "font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100]},
                    "bar": {
                        "color": "#00c853" if prob_up > 0.5 else "#ff1744",
                        "thickness": 0.3,
                    },
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(255,23,68,0.12)"},
                        {"range": [50, 100], "color": "rgba(0,200,83,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffc107", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_prob.update_layout(
                height=200,
                margin=dict(l=30, r=30, t=40, b=10),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_prob, key="xgb_prob_gauge", width="stretch")

            # --- Feature importance chart ---
            if xgb_model.feature_importance:
                top_n = 10
                feat_names = list(xgb_model.feature_importance.keys())[:top_n]
                feat_vals = list(xgb_model.feature_importance.values())[:top_n]

                fig_imp = go.Figure(go.Bar(
                    x=feat_vals[::-1],
                    y=[n.replace("_", " ").title() for n in feat_names[::-1]],
                    orientation="h",
                    marker_color=[
                        "#00c853" if n in (
                            "avg_overall_sentiment", "avg_ticker_sentiment",
                            "sentiment_rolling_3d", "sentiment_rolling_5d",
                            "sentiment_momentum", "pct_positive", "pct_negative",
                            "article_count", "sentiment_std", "sentiment_range",
                        ) else "#2196f3"
                        for n in feat_names[::-1]
                    ],
                ))
                fig_imp.update_layout(
                    title="Feature Importance (top 10)",
                    xaxis_title="Importance (gain)",
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=30),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_imp, key="xgb_feat_imp", width="stretch")

    except Exception as e:
        st.error(f"Prediction error: {e}")

    # --- Walk-forward accuracy over time ---
    if hasattr(xgb_model, "_wf_results") and xgb_model._wf_results is not None:
        wf = xgb_model._wf_results.copy()
        wf["date"] = pd.to_datetime(wf["date"])
        wf["correct"] = (wf["actual"] == wf["predicted"]).astype(int)

        # Rolling accuracy (20-day window)
        wf_sorted = wf.sort_values("date")
        wf_sorted["rolling_acc"] = (
            wf_sorted["correct"].rolling(20, min_periods=5).mean()
        )

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            x=wf_sorted["date"],
            y=wf_sorted["rolling_acc"] * 100,
            mode="lines",
            name="20-day rolling accuracy",
            line=dict(color="#2196f3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        ))
        fig_wf.add_hline(
            y=50, line_dash="dash", line_color="#ff8f00",
            annotation_text="Coin flip (50%)",
        )
        fig_wf.update_layout(
            title="Walk-Forward Validation: Rolling Accuracy Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[20, 80]),
            height=300,
            margin=dict(l=10, r=10, t=40, b=30),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_wf, key="xgb_wf_accuracy", width="stretch")

        # Per-ticker metrics — compact grid (5 per row)
        per_ticker = xgb_model.metrics.get("per_ticker", {})
        if per_ticker:
            tickers_sorted = sorted(
                per_ticker.items(), key=lambda x: x[1]["accuracy"], reverse=True
            )
            COLS_PER_ROW = 5
            for row_start in range(0, len(tickers_sorted), COLS_PER_ROW):
                row_items = tickers_sorted[row_start : row_start + COLS_PER_ROW]
                tk_cols = st.columns(COLS_PER_ROW)
                for i, (tkr, tm) in enumerate(row_items):
                    acc = tm["accuracy"]
                    # colour-code: green ≥55%, red <50%, amber otherwise
                    if acc >= 0.55:
                        border_color = "#00c853"
                    elif acc < 0.50:
                        border_color = "#ff1744"
                    else:
                        border_color = "#ffc107"
                    with tk_cols[i]:
                        st.markdown(
                            f'<div style="background:#f8f9fa; border-radius:0.5rem; '
                            f'padding:0.5rem 0.6rem; text-align:center; '
                            f'border-left:3px solid {border_color}; '
                            f'margin-bottom:0.4rem;">'
                            f'<div style="font-size:0.7rem; color:#666;">{tkr}</div>'
                            f'<div style="font-size:1.1rem; font-weight:700;">'
                            f'{acc:.1%}</div>'
                            f'<div style="font-size:0.65rem; color:#888;">'
                            f'F1={tm["f1"]:.1%} · n={tm["n_predictions"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem; "
        "margin-top:1rem;'>XGBoost predicts next-day return direction "
        "using lagged sentiment + price features. Walk-forward validated "
        "(no lookahead bias). Not financial advice.</div>",
        unsafe_allow_html=True,
    )

# =========================================================================
# Panel 10 — Multi-Horizon Prediction (1d / 5d / 20d)
# =========================================================================

st.markdown("---")
st.header("Multi-Horizon Prediction (Daily / Weekly / Monthly)")

hz_models = load_multihorizon_models()
hz_available = {k: v for k, v in hz_models.items() if v is not None}

if not hz_available:
    st.markdown(
        '<div style="background:#23272e; border:1px solid #444; border-radius:12px;'
        ' padding:2rem; text-align:center; margin:1rem 0;">'
        '<div style="font-size:2.5rem;">🤖</div>'
        '<div style="font-size:1.1rem; margin-top:0.5rem;">Multi-horizon models not trained yet</div>'
        '<div style="font-size:0.85rem; opacity:0.7; margin-top:0.3rem;">'
        'Run <code>python scripts/train_multihorizon.py</code> to train 1d / 5d / 20d predictors.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    # --- Predict for each horizon ---
    HORIZON_META = {
        "1d": {"label": "Daily (1-day)", "icon": "1D", "color_up": "#00c853", "color_dn": "#ff1744"},
        "5d": {"label": "Weekly (5-day)", "icon": "5D", "color_up": "#00e676", "color_dn": "#ff5252"},
        "20d": {"label": "Monthly (20-day)", "icon": "20D", "color_up": "#69f0ae", "color_dn": "#ff8a80"},
    }

    hz_preds = {}
    for hz, mdl in hz_available.items():
        try:
            hz_preds[hz] = mdl.predict_next_day(merged_all, selected_ticker)
        except Exception as e:
            hz_preds[hz] = {"error": str(e)}

    # --- Prediction cards side by side ---
    cols = st.columns(len(hz_available))
    for i, (hz, meta) in enumerate(
        [(h, HORIZON_META[h]) for h in hz_available if h in hz_preds]
    ):
        pred = hz_preds[hz]
        mdl = hz_available[hz]
        with cols[i]:
            if "error" in pred:
                st.markdown(
                    f'<div class="pred-card pred-neutral">'
                    f'<div style="font-size:1.8rem;">{meta["icon"]}</div>'
                    f'<div style="font-size:1rem;">{meta["label"]}</div>'
                    f'<div style="font-size:0.8rem; margin-top:0.5rem;">Error: {pred["error"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                direction = pred["direction"]
                prob_up = pred["prob_up"]
                prob_down = pred["prob_down"]
                confidence = pred["confidence"]
                card_cls = "pred-up" if direction == "UP" else "pred-down"
                arrow = "^" if direction == "UP" else "v"
                prob_shown = prob_up if direction == "UP" else prob_down

                st.markdown(
                    f'<div class="pred-card {card_cls}">'
                    f'<div style="font-size:1.6rem;">{meta["icon"]} {arrow}</div>'
                    f'<div style="font-size:1.4rem;">PREDICT {direction}</div>'
                    f'<div style="font-size:0.9rem; opacity:0.9; margin-top:0.3rem;">'
                    f'P(Up): {prob_up:.1%} · P(Down): {prob_down:.1%}</div>'
                    f'<div style="font-size:0.8rem; opacity:0.75; margin-top:0.2rem;">'
                    f'Confidence: {confidence:.1%}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Model performance
                acc = mdl.metrics.get("accuracy", 0)
                auc = mdl.metrics.get("roc_auc", 0)
                f1 = mdl.metrics.get("f1", 0)
                st.markdown(
                    f'<div class="metric-card" style="margin-top:0.5rem;">'
                    f'<div class="metric-label">{meta["label"]} Accuracy</div>'
                    f'<div class="metric-value">{acc:.1%}</div>'
                    f'<div class="metric-delta delta-neutral">'
                    f'AUC {auc:.3f} | F1 {f1:.1%}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # --- Multi-horizon probability comparison chart ---
    valid_preds = {h: p for h, p in hz_preds.items() if "error" not in p}
    if valid_preds:
        fig_mh = go.Figure()

        hz_labels = [HORIZON_META[h]["label"] for h in valid_preds]
        prob_ups = [valid_preds[h]["prob_up"] * 100 for h in valid_preds]
        prob_downs = [valid_preds[h]["prob_down"] * 100 for h in valid_preds]

        fig_mh.add_trace(go.Bar(
            x=hz_labels,
            y=prob_ups,
            name="P(Up)",
            marker_color="#00c853",
            text=[f"{p:.1f}%" for p in prob_ups],
            textposition="inside",
        ))
        fig_mh.add_trace(go.Bar(
            x=hz_labels,
            y=prob_downs,
            name="P(Down)",
            marker_color="#ff1744",
            text=[f"{p:.1f}%" for p in prob_downs],
            textposition="inside",
        ))

        fig_mh.update_layout(
            title=f"Multi-Horizon Outlook: {selected_ticker}",
            barmode="group",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 100]),
            height=320,
            margin=dict(l=10, r=10, t=40, b=30),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        fig_mh.add_hline(y=50, line_dash="dash", line_color="#ffc107", annotation_text="50%")
        st.plotly_chart(fig_mh, key="multihorizon_bar", width="stretch")

    # --- Model comparison table ---
    if hz_available:
        comparison_data = []
        for hz, mdl in hz_available.items():
            m = mdl.metrics
            comparison_data.append({
                "Horizon": HORIZON_META[hz]["label"],
                "Accuracy": f"{m.get('accuracy', 0):.1%}",
                "F1": f"{m.get('f1', 0):.1%}",
                "AUC": f"{m.get('roc_auc', 0):.3f}",
                "Precision": f"{m.get('precision', 0):.1%}",
                "Recall": f"{m.get('recall', 0):.1%}",
                "Predictions": m.get("n_predictions", 0),
            })
        st.markdown("**Model Comparison (Walk-Forward Validation)**")
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, width=900)

    # --- Per-ticker accuracy heatmap ---
    if len(hz_available) >= 2:
        ticker_accs = {}
        for hz, mdl in hz_available.items():
            per_tkr = mdl.metrics.get("per_ticker", {})
            for tkr, tm in per_tkr.items():
                if tkr not in ticker_accs:
                    ticker_accs[tkr] = {}
                ticker_accs[tkr][HORIZON_META[hz]["label"]] = tm.get("accuracy", 0)

        if ticker_accs:
            hm_df = pd.DataFrame(ticker_accs).T
            hm_df = hm_df.sort_index()

            fig_hm = go.Figure(data=go.Heatmap(
                z=hm_df.values * 100,
                x=hm_df.columns.tolist(),
                y=hm_df.index.tolist(),
                colorscale=[
                    [0, "#ff1744"],
                    [0.5, "#ffc107"],
                    [1, "#00c853"],
                ],
                zmin=35,
                zmax=65,
                text=[[f"{v*100:.1f}%" for v in row] for row in hm_df.values],
                texttemplate="%{text}",
                textfont={"size": 11},
                colorbar=dict(title="Accuracy %"),
            ))
            fig_hm.update_layout(
                title="Per-Ticker Accuracy by Horizon",
                xaxis_title="Horizon",
                yaxis_title="Ticker",
                height=max(300, len(ticker_accs) * 30 + 100),
                margin=dict(l=10, r=10, t=40, b=30),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hm, key="multihorizon_heatmap", width="stretch")

    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem; "
        "margin-top:1rem;'>Multi-horizon models predict return direction over "
        "1, 5, and 20 trading days. Each model uses walk-forward validation "
        "with lagged technical + sentiment features. Not financial advice.</div>",
        unsafe_allow_html=True,
    )

# =========================================================================
# Footer
# =========================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; opacity:0.5; font-size:0.85rem;'>"
    "Financial Sentiment Analysis Dashboard · Built with Streamlit + Plotly · "
    "Model: ProsusAI/FinBERT (fine-tuned) + XGBoost (multi-horizon)"
    "</div>",
    unsafe_allow_html=True,
)
