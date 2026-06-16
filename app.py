import streamlit as st
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Trader Sentiment Analysis Dashboard", layout="wide", page_icon="📈")

st.title("Trader Performance vs Market Sentiment")
st.markdown("Explore how BTC Market Sentiment (Fear/Greed) influences different trader segments.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    out_dir = "outputs"
    try:
        trader_summary = pd.read_csv(os.path.join(out_dir, "trader_summary.csv"))
        perf_sent = pd.read_csv(os.path.join(out_dir, "performance_by_sentiment.csv"))
        beh_sent = pd.read_csv(os.path.join(out_dir, "behavior_by_sentiment.csv"))
        return trader_summary, perf_sent, beh_sent
    except FileNotFoundError:
        return None, None, None

trader_summary, perf_sent, beh_sent = load_data()

if trader_summary is None:
    st.error("Data files not found in the `outputs/` directory. Please run `analysis.py` first.")
    st.stop()

# --- SIDEBAR NAV ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["1. Sentiment Overviews", "2. Trader Dashboards", "3. Archetype Clustering"])

if page == "1. Sentiment Overviews":
    st.header("1. Sentiment Regime Overviews")
    st.markdown("Macro sentiment is a core predictor of success. Performance and trading frequency both scale independently with market greed regimes.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Shift")
        st.dataframe(
            perf_sent.style.format({
                "avg_pnl": "${:,.2f}",
                "median_pnl": "${:,.2f}",
                "avg_wr": "{:.2%}",
                "avg_dd": "${:,.2f}"
            }),
            width="stretch"
        )
        st.image("charts/fig1_performance.png", width="stretch")

    with col2:
        st.subheader("Behavior Shift")
        st.dataframe(
            beh_sent.style.format({
                "avg_trades": "{:.1f}",
                "avg_size_usd": "${:,.0f}",
                "avg_ls_ratio": "{:.3f}"
            }),
            width="stretch"
        )
        st.image("charts/fig3_behavior.png", width="stretch")

    st.subheader("Drawdown Profile")
    st.image("charts/fig2_pnl_violin.png")

elif page == "2. Trader Dashboards":
    st.header("2. Segment Level Behaviors")
    st.markdown("Different types of traders handle sentiment swings uniquely.")

    st.subheader("Top Performers (Net PnL)")
    st.image("charts/fig5_top10_traders.png")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("By Account Size")
        st.image("charts/fig6_size_segment.png", width="stretch")

    with col2:
        st.subheader("By Historical Win Rate")
        st.image("charts/fig8_perf_segment.png", width="stretch")

    st.subheader("Volatility Amplification (Frequency & Leverage)")
    col_a, col_b = st.columns(2)

    with col_a:
        st.image("charts/fig7_freq_heatmap.png")

    with col_b:
        st.image("charts/fig4_leverage_distribution.png")

elif page == "3. Archetype Clustering":
    st.header("3. Clustering & Predictive Model")

    st.subheader("Behavioral Archetypes (K=3)")
    st.markdown("A K-Means algorithm grouped traders into three profiles based on frequency, win-rate, size, leverage, and Sharpe proxy.")
    st.image("charts/fig10_clustering.png")

    # Only use columns that actually exist in the CSV
    desired_cols = ['trades_per_day', 'avg_size_usd', 'overall_win_rate', 'avg_daily_pnl', 'sharpe_proxy']
    available_cols = [c for c in desired_cols if c in trader_summary.columns]
    cluster_stats = trader_summary.groupby('cluster')[available_cols].mean().round(2)
    cluster_stats['Count'] = trader_summary['cluster'].value_counts().sort_index()

    # Only format columns that exist
    format_map = {
        "avg_size_usd": "${:,.0f}",
        "overall_win_rate": "{:.2%}",
        "sharpe_proxy": "{:.2f}",
        "avg_daily_pnl": "${:,.2f}",
        "trades_per_day": "{:.1f}"
    }
    active_format = {k: v for k, v in format_map.items() if k in cluster_stats.columns}
    st.dataframe(cluster_stats.style.format(active_format))

    st.divider()

    st.subheader("Predictive Model Feature Importance")
    st.markdown("A Gradient Boosting Classifier predicted whether a user's day would end in profit based on their behavior and market sentiment. Accuracy on test: 91.3%.")
    st.image("charts/fig11_feature_importance.png")
