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
- Resolve the open-xquant runner before the first command. In command examples
  below, `uv run oxq` means the resolved runner.

## Runner Resolution

In a new research directory, `uv run oxq` may fail because open-xquant is
installed as long-lived Agent capability, not as a package in that directory.
Before running commands:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search the user's home
directory for unrelated open-xquant checkouts, and do not switch to a different
source tree just because it contains an `agent/skills/` directory.

## Current Stable Spec Path

Use this path first:

- `universe.type: static`
- `data.provider: local`
- `market.calendar: XNYS`, `ARCX`, `XSHG`, or `XSHE`
- `signal.signal_time: close_t`
- `execution.trade_time: next_open`
- `execution.fill_price_mode: next_open`
- explicit execution semantics:
  `execution.order_timing: next_session_open`,
  `execution.price_bar: next_session`,
  `execution.price_type: open`
- `execution.cash_annual_return` and `execution.lot_size_config`
- `metrics.profile: open_xquant_default` unless the user asks for another
  supported metrics profile
- positive `cost.fee_rate`
- positive `cost.slippage_rate`
- `portfolio.type: EqualWeight` only for boolean signal filters
- `ROC` + `ROCTiming` + `SignalToPosition` for single-symbol timing strategies
  that need explicit `BUY` / `SELL` / `HOLD` and HOLD-maintains-position
  semantics

Do not promise these are directly supported by the audited CLI compiler:

- `index` or `filter` universe
- non-`local` data provider in `oxq backtest run`
- multiple `Crossover` rules in one spec
- `Peak` signal for causal audited backtests
- `Timestamp` `month_end` or `quarter_end`
- signal rules combined with portfolio types other than `EqualWeight` or
  `SignalToPosition`
- `BUY` / `SELL` / `HOLD` categorical signal rules with `EqualWeight`
- arbitrary rules declared under a top-level `rules:` YAML section

## Phase 0: Confirm Constraints

### Experiment Lifecycle Check

Before creating or editing a spec, inspect `runs/` when it exists:

```bash
ls runs/
```

For each run directory, classify completion by checking:

- `metrics.json`
- `research_report.md`
- `research_bias_audit.json`
- `robustness.json`

Skip robustness sub-runs such as `<run_id>_cost_x2` and parameter-perturbation
siblings when deciding whether a user experiment is unfinished. Treat those
directories as child artifacts of the parent run, not resumable experiments.

If a run has `metrics.json` but lacks `research_report.md`, ask whether to
resume that unfinished experiment or abandon it and start a new run. Abandoning
does not delete the run; record the status in `experiments.jsonl` when an
experiment registry exists. If all prior experiments are complete, summarize
the recent runs and continue.

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
  order_timing: next_session_open
  price_bar: next_session
  price_type: open
  initial_cash: 100000
  cash_annual_return: 0.0
  lot_size_config:
    default: 1
    by_symbol: {}

cost:
  fee_rate: 0.001
  slippage_rate: 0.001

metrics:
  profile: open_xquant_default
  risk_free_rate: 0.0
  return_type: simple
  annualization_days: 252
  calmar_denominator: max_drawdown
  evaluation_window: full

robustness:
  cost_multiplier: [1.0, 2.0]
  parameter_perturbation: {}
  regime_analysis: false
```

Categorical custom signals must declare their output domain as rule metadata,
not as compute params:

```yaml
signal:
  rules:
    timing:
      type: CustomTiming
      output_domain: [BUY, SELL, HOLD]
      params:
        column: close
portfolio:
  type: SignalToPosition
  params:
    signal: timing
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
- unsupported market calendar
- missing local data semantics
- same-bar signal and fill timing
- conflicting legacy and explicit execution semantics
- invalid lot-size configuration
- unsupported metrics profile or metric assumption
- negative fee/slippage
- zero fee/slippage unless this is an explicit replay-style validation spec;
  preserve the warning when zero costs are accepted
- missing OOS test period
- train/test period overlap
- signal parameter references an unknown column
- compiler dry-run failure

Warnings are not fatal, but must be reported later.

After validation passes, use `spec-auditor` before any backtest. The auditor
must classify material fields as confirmed, default, or unconfirmed. Any
unconfirmed field blocks `oxq backtest run` until the user confirms or changes
the value, then validation and the auditor gate run again.

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
uv run oxq backtest run strategy_spec.yaml --data-dir /path/to/parquet --out runs/auto --json > backtest_result.json
RUN_DIR=$(uv run python - <<'PY'
import json
payload = json.load(open("backtest_result.json"))
if payload["status"] != "pass":
    raise SystemExit(payload)
print(payload["run_dir"])
PY
)
test -n "$RUN_DIR"
echo "$RUN_DIR"
```

Use `artifacts.target_weights_csv` from `backtest_result.json` for baseline
target-weight comparisons. Do not parse human stdout for `run_dir`.

## Phase 5: Audit And Report

```bash
uv run oxq audit reproducibility "$RUN_DIR"
uv run oxq audit research "$RUN_DIR"
uv run oxq robustness run "$RUN_DIR"
uv run oxq experiment add "$RUN_DIR"
```

If research audit has fatal findings, mark the strategy rejected. If robustness
returns `WARN`, keep the warning in the final report. When `robustness.json`
exists, summarize cost stress, IS/OOS metric diff, parameter perturbation, and
regime analysis instead of only quoting baseline Sharpe.
Use `research-report-writer` to write `research_report.md` and render
`research_report.html` from that final Markdown.

When the user accepts a completed run as the selected version, ask whether to
mark it as final. If yes, update `runs/final` as a lightweight pointer:

- copy the selected run's `strategy_spec.yaml` to `runs/final/strategy_spec.yaml`
- write `runs/final/selected.json` with `run_id`, `selected_at`, reason,
  `metrics_snapshot`, and `previous_run_id` when replacing an older selection
- write `runs/final/README.md` with the selected reason and next steps
- mark the run status as `final` in `experiments.jsonl` when present

## Red Lines

- Do not skip validation.
- Do not run formal research on missing or unconfirmed data.
- Do not use zero-cost assumptions.
- Do not edit a validated spec after seeing results just to improve metrics.
- Do not recommend paper or live trading without audit and report artifacts.
