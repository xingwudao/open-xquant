---
name: explore-data
description: >-
  Inspect, download, and validate open-xquant market, macro, and financial
  data; use before backtests, factor studies, or any task that depends on local
  parquet data.
---

# Data Explorer

You prepare data before research. Do not assume data exists or is complete.

## Market Data

List local parquet files:

```bash
uv run python - <<'PY'
from oxq.tools.data import list_symbols
print(list_symbols(data_dir="/path/to/market"))
PY
```

Inspect one symbol:

```bash
uv run python - <<'PY'
from oxq.tools.data import inspect_symbol
print(inspect_symbol("SPY", data_dir="/path/to/market"))
PY
```

Download with `yfinance`:

```bash
uv run python - <<'PY'
from pathlib import Path
from oxq.data.loaders import YFinanceDownloader

data_dir = Path("/tmp/oxq_agent_data")
YFinanceDownloader().download_many(
    symbols=["SPY", "QQQ"],
    start="2018-01-01",
    end="2026-01-01",
    dest_dir=data_dir,
)
print(data_dir)
PY
```

For A-share market bars, use `AkShareDownloader` and install the `akshare`
extra first.

## Required Data Shape

Every market parquet must have:

- file name `<SYMBOL>.parquet`
- tz-aware `DatetimeIndex`
- columns `open`, `high`, `low`, `close`, `volume`
- enough history for indicator warmup and train/test periods

If `LocalMarketDataProvider` says `No data for '<SYMBOL>'`, stop and fix data
or ask the user for a valid `--data-dir`.

## Macro Data

Use `WorldBankFetcher` with indicator name, year range, and countries:

```python
from oxq.data.factors import WorldBankFetcher

fetcher = WorldBankFetcher()
gdp = fetcher.fetch(
    target="gdp",
    start="2018",
    end="2024",
    countries=["USA", "CHN"],
)
```

Available macro aliases include `gdp`, `gdp_per_capita`, `gdp_growth`, and
`cpi`.

## Financial Data

For A-share fundamentals, use `EastMoneyFetcher` through the factor data layer
and verify available fields before screening. Network and provider coverage can
fail; report provider errors directly.

## Red Lines

- Do not run a formal backtest on unknown or missing data.
- Do not mix data providers inside one run unless the user approves and the
  report states it.
- Do not treat generated mock data or demo downloads as production research
  evidence.
