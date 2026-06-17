---
name: strategy-builder
description: >-
  Build and validate open-xquant strategy_spec.yaml files from user strategy
  ideas; use for new strategy design, spec edits, and audited CLI backtest
  workflows.
---

# Strategy Builder

You convert a user strategy idea into a validated `strategy_spec.yaml` and
an auditable research run.

## Ground Rules

- Treat `strategy_spec.yaml` as the single source of truth.
- Do not invent the user's hypothesis, universe, cost, dates, or objective.
- Prefer current stable CLI behavior before SDK customization.
- Stop when `oxq spec validate` fails.
- Do not present an unaudited backtest as a research result.

## Current Stable Spec Path

Use this path first:

- `universe.type: static`
- `data.provider: local`
- `market.calendar: XNYS`
- `signal.signal_time: close_t`
- `execution.trade_time: next_open`
- `execution.fill_price_mode: next_open`
- positive `cost.fee_rate`
- positive `cost.slippage_rate`
- `portfolio.type: EqualWeight` when `signal.rules` is present

Do not promise these are directly supported by the audited CLI compiler:

- `index` or `filter` universe
- non-`local` data provider in `oxq backtest run`
- multiple `Crossover` rules in one spec
- `Peak` signal for causal audited backtests
- `Timestamp` `month_end` or `quarter_end`
- signal rules combined with non-`EqualWeight` portfolio
- arbitrary rules declared under a top-level `rules:` YAML section

## Phase 0: Confirm Constraints

Ask for or confirm:

- tradable symbols
- train and test periods
- initial cash
- fee rate and slippage rate
- benchmark
- success metric
- exit or risk requirement

If the user wants a quick demo, state that it is an environment check, not an
investment conclusion.

## Phase 1: Create Spec

```bash
uv run oxq spec init "<strategy idea>" --out strategy_spec.yaml
```

Edit the generated spec. Keep the safe timing model:

```yaml
signal:
  signal_time: close_t
  indicators:
    sma_fast:
      type: SMA
      params: { column: close, period: 10 }
    sma_slow:
      type: SMA
      params: { column: close, period: 50 }
  rules:
    golden_cross:
      type: Crossover
      params: { fast: sma_fast, slow: sma_slow }

execution:
  trade_time: next_open
  fill_price_mode: next_open
  initial_cash: 100000

cost:
  fee_rate: 0.001
  slippage_rate: 0.001
```

For non-SMA strategies, inspect the registry before choosing names:

```bash
uv run python - <<'PY'
import oxq
print("Indicators:", sorted(oxq.list_indicators()))
print("Signals:", sorted(oxq.list_signals()))
print("Portfolios:", sorted(oxq.list_portfolio_optimizers()))
PY
```

## Phase 2: Validate

```bash
uv run oxq spec validate strategy_spec.yaml
```

Fatal issues to fix before continuing:

- empty `research.hypothesis`
- unsupported universe or data provider
- missing local data semantics
- same-bar signal and fill timing
- zero or negative fee/slippage
- missing OOS test period
- train/test period overlap
- signal parameter references an unknown column
- compiler dry-run failure

Warnings are not fatal, but must be reported later.

## Phase 3: Prepare Data

`oxq backtest run` reads local parquet files. If data is missing, prepare it
or ask the user for a data directory.

```bash
uv run python - <<'PY'
from pathlib import Path
from oxq.data.loaders import YFinanceDownloader

data_dir = Path("/tmp/oxq_agent_data")
YFinanceDownloader().download_many(
    symbols=["SPY"],
    start="2018-01-01",
    end="2026-01-01",
    dest_dir=data_dir,
)
print(data_dir)
PY
```

Use `--data-dir /path/to/parquet` when the data is not in the default
`~/.oxq/data/market` directory.

## Phase 4: Compile And Backtest

```bash
uv run oxq strategy compile strategy_spec.yaml
uv run oxq backtest run strategy_spec.yaml --data-dir /path/to/parquet --out runs/auto
```

Capture the printed run directory. If needed:

```bash
RUN_DIR=$(find runs -mindepth 2 -maxdepth 2 -type d | sort | tail -1)
echo "$RUN_DIR"
```

## Phase 5: Audit And Report

```bash
uv run oxq audit reproducibility "$RUN_DIR"
uv run oxq audit research "$RUN_DIR"
uv run oxq robustness run "$RUN_DIR"
uv run oxq report write "$RUN_DIR"
uv run oxq experiment add "$RUN_DIR"
```

If research audit has fatal findings, mark the strategy rejected. If robustness
returns `WARN`, keep the warning in the final report.

## Red Lines

- Do not skip validation.
- Do not run formal research on missing or unconfirmed data.
- Do not use zero-cost assumptions.
- Do not edit a validated spec after seeing results just to improve metrics.
- Do not recommend paper or live trading without audit and report artifacts.
