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

from src.data.data_loader import (
    load_all_news,
    load_all_prices,
    load_merged_dataset,
    add_session_column,
    assign_trading_day,
    RAW_DATA_DIR,
    DEFAULT_TICKERS,
    DEFAULT_START,
    DEFAULT_END,
)

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

    # Date range
    st.subheader("Date Range")
    start_date = pd.Timestamp(DEFAULT_START)
    end_date = pd.Timestamp(DEFAULT_END)
    date_range = st.date_input(
        "Select range",
        value=(start_date.date(), end_date.date()),
        min_value=start_date.date(),
        max_value=end_date.date(),
        help="Filter data within the available date window.",
    )

    # Normalise tuple (user might pick a single date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        sel_start, sel_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        sel_start, sel_end = start_date, end_date

    st.markdown("---")
    st.caption("Data: Alpha Vantage + Yahoo Finance")
    st.caption("Model: ProsusAI/FinBERT (fine-tuned)")

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
        from scipy import stats as sp_stats

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
    st.warning("No data found for the selected ticker and date range.")

# =========================================================================
# Footer
# =========================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; opacity:0.5; font-size:0.85rem;'>"
    "Financial Sentiment Analysis Dashboard · Built with Streamlit + Plotly · "
    "Model: ProsusAI/FinBERT (fine-tuned)"
    "</div>",
    unsafe_allow_html=True,
)
