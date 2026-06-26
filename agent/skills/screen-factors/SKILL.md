---
name: screen-factors
description: >-
  Screen symbols with open-xquant price, financial, and custom factors; use
  when users ask for value, quality, momentum, or multi-factor candidate lists.
---

# Factor Screening

You build candidate lists from data. Screening is not a backtest.

## Confirm Inputs

Ask for:

- market and symbols
- factor definitions
- rebalance date or date range
- thresholds or ranking rules
- required data provider
- how to handle missing values

## Inspect Available Indicators

```bash
uv run python - <<'PY'
import oxq
print(sorted(oxq.list_indicators()))
PY
```

Financial indicator classes include names such as `PE`, `PB`, `BP`, `EP`,
`ROEChange`, `NetProfitMargin`, `AccrualRatio`, `CashFlowRatio`, `MarketCap`,
`TurnoverRate`, and `PowerRatio`. Verify required input columns before
computing them.

## Price-Based Screening Pattern

```python
import pandas as pd

from oxq.data.market import LocalMarketDataProvider
from oxq.indicators import NdayReturn, RollingVolatility

symbols = ["AAPL", "MSFT", "GOOGL"]
market = LocalMarketDataProvider(data_dir="/path/to/parquet")

rows = []
for sym in symbols:
    bars = market.get_bars(sym, "2020-01-01", "2024-12-31")
    momentum = NdayReturn().compute(bars, column="close", period=60).iloc[-1]
    volatility = RollingVolatility().compute(bars, column="close", period=20).iloc[-1]
    rows.append({"symbol": sym, "momentum_60": momentum, "vol_20": volatility})

screen = pd.DataFrame(rows).dropna()
screen["score"] = screen["momentum_60"].rank(pct=True) - screen["vol_20"].rank(pct=True)
candidates = screen.sort_values("score", ascending=False).head(10)
```

## Financial Screening

For financial fields, use the factor data layer and inspect returned columns.
Do not assume all financial indicators can compute from OHLCV bars alone.

## Red Lines

- Do not call a screened list a validated strategy.
- Do not ignore missing factor values.
- Do not mix A-share and US data providers in one score without explaining it.
- Do not use future financial statement publication dates.
