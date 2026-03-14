# Trader Performance vs Market Sentiment (Fear/Greed)
### Primetrade.ai — Data Science Intern Assignment

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook streamlit
```

---

## Data files

Place both files in the `data/` folder exactly as named below:

| File | Description |
|------|-------------|
| `data/fear_greed_sentiment.csv` | Bitcoin Fear/Greed index — columns: `Date`, `Classification` |
| `data/historical_data.csv` | Hyperliquid trade history — columns: `Account`, `Coin`, `Execution Price`, `Size Tokens`, `Size USD`, `Side`, `Timestamp IST`, `Start Position`, `Direction`, `Closed PnL`, `Fee`, and more |

---

## Run

```bash
# Option A — plain Python script (recommended for quick run)
python analysis.py

# Option B — Jupyter notebook (recommended for step-by-step review)
jupyter notebook trader_sentiment_analysis.ipynb

# Option C — Interactive Streamlit Dashboard (Data exploration)
streamlit run app.py
```

---

## Output

| Folder | Contents |
|--------|----------|
| `charts/` | 11 PNG charts — fig1 through fig11 |
| `outputs/` | `trader_summary.csv`, `performance_by_sentiment.csv`, `behavior_by_sentiment.csv` |

### Charts generated

| File | Description |
|------|-------------|
| `fig1_performance.png` | Mean PnL, Win Rate, Drawdown — Fear vs Greed |
| `fig2_pnl_violin.png` | PnL distribution violin plot |
| `fig3_behavior.png` | Trade frequency, size, long ratio by sentiment |
| `fig4_leverage_distribution.png` | Leverage distribution and avg leverage by sentiment |
| `fig5_top10_traders.png` | Top 10 traders by net PnL |
| `fig6_size_segment.png` | Large vs Small size traders x sentiment |
| `fig7_freq_heatmap.png` | Frequent vs Infrequent traders x sentiment heatmap |
| `fig8_perf_segment.png` | Winner / Neutral / Loser segments x sentiment |
| `fig9_monthly_pnl.png` | Monthly total PnL split by Fear/Greed |
| `fig10_clustering.png` | K-Means trader clustering (k=3) |
| `fig11_feature_importance.png` | Predictive model feature importances |

---

## Project Structure

```
trader_sentiment_analysis/
├── analysis.py                       # standalone end-to-end script
├── app.py                            # Streamlit dashboard for interactive exploration
├── trader_sentiment_analysis.ipynb   # Jupyter notebook (same analysis, step by step)
├── README.md
├── WRITE_UP.md                       # 1-page methodology + insights + strategies
├── data/
│   ├── fear_greed_sentiment.csv
│   └── historical_data.csv
├── charts/                           # generated on run (11 PNGs)
└── outputs/                          # generated on run (3 CSVs)
```

---

## Methodology (brief)

1. Data prep — column normalisation, timestamp parsing (dd-mm-yyyy HH:MM), deduplication, inner-join on daily date
2. Feature engineering — daily PnL, win rate, trade count, avg trade size (USD), long ratio, drawdown proxy per account per day
3. Segmentation — trade size (large/small), frequency (frequent/infrequent), performance (winners/neutral/losers)
4. Analysis — group comparisons across Fear vs Greed regimes with chart evidence
5. Clustering — K-Means (k=3) on scaled trader-level features to surface behavioral archetypes
6. Predictive model — Gradient Boosting classifier predicting next-day profitability (5-fold CV ~82%)

See WRITE_UP.md for key insights and strategy recommendations.
