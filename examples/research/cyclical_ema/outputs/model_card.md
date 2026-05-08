# Model Card — Cyclical EMA v0

## Verdict: ❌ HYPOTHESIS REJECTED (at v0 configuration)

The hypothesis "ML gating beats pure EMA on cyclical stocks by ≥ +0.3 Sharpe"
is **REJECTED** in this v0 configuration:

| Metric | Strategy | Pure EMA Baseline | Gap | Target |
|---|---|---|---|---|
| Sharpe | **1.952** | **3.289** | **-1.336** | ≥ +0.30 |
| Annual Return | 103.15% | 102.54% | — | — |
| Max Drawdown | -23.577% | -13.149% | — | ≤ -25% |
| Trades | 286 | 2313 | — | — |

Both strategy and baseline have absolute Sharpe well above the ≥1.0 floor —
this just means the universe is profitable. The KEY question (does ML add
value?) gets a NO at v0: the model's aggressive 0.55 score
threshold cuts ~88% of
baseline trades, losing positive convexity faster than it filters losers.

## Model
- **Algorithm**: LightGBM binary classifier (gating on top of EMA strategy)
- **Trained on**: 10,864 samples (A:B:C = 7,799:2,599:466)
- **Tested on**:  2,313 closed samples (post-2021-05-07)
- **Features**:   28 factors (see `feature_importance.csv`, `shap_summary.png`)

PR-AUC (test): 0.2149 (random baseline ≈ 0.179)
Precision@top10%: 0.2511

## Best hyperparameters (Optuna best of 50 trials)

```json
{
  "num_leaves": 28,
  "learning_rate": 0.037619132057241315,
  "min_child_samples": 106,
  "feature_fraction": 0.8380365045328141,
  "bagging_fraction": 0.767752701654755,
  "lambda_l2": 4.93770512439691
}
```

## Known limitations / caveats

- **Survivorship bias (medium)** —— training data uses yfinance + Wikipedia
  historical S&P 500 removals; not strict PIT. Test set only includes tickers
  still listed on 2026-05-07.
- **Universe filter is snapshot-anchored** at 2026-05-07, not PIT
  per training sample.
- **No fundamental factors (L5)** —— yfinance.info has no PIT publish dates;
  L5 deferred to v0.5. Quality proxies (`sharpe_5y` / `max_dd_5y`) were
  included but dropped at runtime due to >30% NaN.
- **Hurst threshold recalibrated 0.40 → 0.45** during integration: real
  US-stock Hurst(100d) distribution centers near 0.52, std 0.04; original
  0.40 yielded zero A-group tickers. 0.45 corresponds to top ~5% mean-
  reverting tail.
- **Backtest uses daily mark-to-market** with TopN normalized weights
  (no rotation; cash drag implicit only when zero positions are open).

## v0.5 directions (to overturn the verdict)

1. **Threshold sweep** —— x ∈ [0.40, 0.65] step 0.025; current x=0.55 is
   likely too aggressive. A lower threshold should keep more trades and
   recover convexity.
2. **Regression head** —— predict gross_return continuously instead of
   binary classification at 5%. Loss function aligned with what we care
   about (P&L, not classification accuracy).
3. **Hurst window** —— try 60d instead of 100d (closer to half-cycle
   length, may give cleaner mean-reversion signal).
4. **L5 fundamental factors** —— PIT-clean from Polygon/Sharadar.
5. **Walk-forward retraining** —— each quarter re-train; current static
   2021-trained model is stale for 2025-26 trades.

## Migration to live trading

Required before any production deployment:
- Paid PIT data source (Polygon / Sharadar) for survivorship-clean
  replication
- A v0.5 configuration that beats the baseline by ≥ +0.30 Sharpe (not yet
  achieved at v0)
- Walk-forward retraining schedule
- Daily MTM verification on a fresh out-of-sample period

## Top features (by SHAP |mean|)

See `shap_summary.png` and `feature_importance.csv`.

## Live snapshot

See `candidates.csv` for the 2026-05-07 snapshot of universe.A
tickers with active EMA-entry signal + their model scores. The 39 rows
include sector / Hurst / vol / ret_3m for inspection.

---
Generated: 2026-05-08T14:04:57.689684
