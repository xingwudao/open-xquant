---
name: evaluate-cross-sectional
description: >-
  Evaluate cross-sectional factors with IC, Rank IC, ICIR, decay, and turnover
  in open-xquant; use for stock selection and multi-asset ranking factors.
---

# Evaluate Cross-Sectional Factor

Use this when the factor ranks assets on the same date.

## Minimal SDK Pattern

```python
import pandas as pd

from oxq.data.market import LocalMarketDataProvider
from oxq.factor_eval.metrics import (
    compute_decay,
    compute_ic,
    compute_icir,
    compute_rank_ic,
    compute_turnover,
)

symbols = ["SPY", "QQQ", "IWM"]
market = LocalMarketDataProvider(data_dir="/path/to/parquet")

factor_df = pd.DataFrame()
prices_df = pd.DataFrame()
for sym in symbols:
    bars = market.get_bars(sym, "2018-01-01", "2024-12-31")
    prices_df[sym] = bars["close"]
    factor_df[sym] = bars["close"].pct_change(20)

factor_df = factor_df.dropna()
prices_df = prices_df.loc[factor_df.index]
forward_returns = prices_df.pct_change(1).shift(-1).dropna()

common = factor_df.index.intersection(forward_returns.index)
factor_df = factor_df.loc[common]
forward_returns = forward_returns.loc[common]

ic = compute_ic(factor=factor_df, forward_returns=forward_returns)
rank_ic = compute_rank_ic(factor=factor_df, forward_returns=forward_returns)
icir = compute_icir(ic["mean"], ic["std"])
decay = compute_decay(factor=factor_df, prices=prices_df, horizons=[1, 3, 5, 10, 20])
turnover = compute_turnover(factor=factor_df)
```

## Review Checklist

- enough symbols for cross-sectional claims
- no forward-return leakage
- factor and returns aligned on common dates
- NaN rows removed deliberately
- multiple horizons checked
- turnover considered with likely trading cost

## Interpretation

Use thresholds as heuristics, not rules:

- IC mean above `0.03`: promising
- IC mean between `0.01` and `0.03`: weak but worth inspecting
- ICIR below `0.1`: unstable
- high turnover: likely cost-sensitive even with positive IC
- fast decay: signal may require execution assumptions the backtest cannot meet

## Report Shape

Include:

- factor name and formula
- symbols and date range
- number of dates and symbols
- horizon results
- IC mean, IC std, ICIR, Rank IC
- turnover
- decay pattern
- pass, weak, or fail conclusion with caveats

## Red Lines

- Do not use cross-sectional IC as primary evidence for fewer than 10 symbols.
- Do not report IC mean without IC std or ICIR.
- Do not ignore turnover and decay.
