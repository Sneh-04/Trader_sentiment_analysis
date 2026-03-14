# Write-Up — Trader Performance vs Market Sentiment
**Primetrade.ai Data Science Intern Assignment** | March 2026

---

## Methodology

Two datasets were aligned at daily granularity: the Bitcoin Fear/Greed Index (731 days, 2 classes) and Hyperliquid trade records (~211K rows, 31 unique accounts, 2023–2024). After normalising column names, deduplicating, and parsing timestamps from the `dd-mm-yyyy HH:MM` format in `Timestamp IST`, trades were merged to their corresponding daily sentiment label (52,841 matched rows). Key per-account-per-day features were engineered: daily PnL, win rate, average trade size (USD), long ratio, drawdown proxy, and trade count. Traders were segmented across three axes — trade size tier, trading frequency, and historical win rate. A K-Means clustering (k=3) was applied to trader-level aggregates to identify behavioural archetypes, and a Gradient Boosting classifier was trained to predict day-level profitability from behavioural and sentiment features.

---

## Key Insights

### Insight 1 — Sentiment is a first-order performance signal
Mean daily PnL on **Greed** days ($5,381) is 71% higher than on **Fear** days ($3,140). Win rates follow the same direction: 36% on Greed days vs 32.5% on Fear days. This gap is consistent across trader segments and is confirmed by the predictive model, where `sentiment_enc` ranks as a top-3 feature. Sentiment is not a proxy for volatility — it is an independent predictor of trading outcome.

### Insight 2 — Trade size amplifies the sentiment effect asymmetrically
Large-size traders gain disproportionately more on Greed days compared to small-size traders. However, on Fear days the gap narrows sharply — large-size traders absorb heavier absolute losses. This asymmetry means scaling up position sizes during unfavourable regimes is the primary path to large drawdowns. The drawdown proxy confirms this: Greed days show higher average drawdown ($1,282) than Fear days ($387), driven entirely by the large-size cohort taking bigger swings.

### Insight 3 — Frequent traders over-trade on Fear days at a measurable cost
The frequency × sentiment heatmap (Fig 6) shows that frequent traders (≥ median trades/day) experience the largest PnL swing between Fear and Greed regimes. On Greed days frequent traders outperform infrequent ones; on Fear days this advantage reverses. The pattern indicates reactive/emotional entries during adverse conditions — high trade count with a 32.5% win rate compounds losses faster than any other behaviour observed in this dataset.

---

## Strategy Recommendations

### Strategy 1 — Sentiment-Gated Position Sizing
> **During Fear days, reduce trade size by ~30% from your normal baseline. During Greed days, maintain or increase to full size.**
>
> Rationale: The Fear/Greed regime is the strongest single predictor of daily outcome (Insight 1). Large-size traders on Fear days produce the worst risk-adjusted returns of any segment × sentiment combination. Reducing size on Fear days caps the downside while preserving the ability to act on genuine signals.

### Strategy 2 — Trade Frequency Throttle on Fear Days
> **Frequent traders should cap daily trades at 2 on Fear days. Infrequent traders should reduce to 1 or sit out entirely.**
>
> Rationale: Frequent traders on Fear days show the steepest win-rate decline relative to their own Greed-day performance (Fig 6). Throttling trade count on Fear days eliminates low-quality reactive entries. A practical rule of thumb: *on a Fear day, only take a trade you would take with 30% smaller size — if it still makes sense, take it; if not, skip it.*

---

*Full analysis, all 11 charts, and reproducible code in `trader_sentiment_analysis.ipynb` and `analysis.py`.*  
*To interactively explore the data segments and clustering models, run: `streamlit run app.py`.*
