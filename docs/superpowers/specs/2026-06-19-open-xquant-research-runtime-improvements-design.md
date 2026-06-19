# open-xquant Research Runtime Improvements Design

## Purpose

This iteration improves open-xquant as an open-source research runtime. The
goal is not to couple open-xquant to any production system. The recent
`prod_global_rotation_etf` comparison was used as a practical capability check:
it showed that the current project can express the strategy idea, but cannot
cleanly express several important research assumptions.

The improvements are split into three independent PRs so each one can be
reviewed, tested, and merged without forcing a large cross-cutting change.

## Current Gaps

The current stable path is intentionally conservative:

- `signal_time: close_t`
- `execution.trade_time: next_open`
- `execution.fill_price_mode: next_open`
- positive fee and slippage
- `market.calendar: XNYS`
- one default metrics profile
- robustness placeholders for IS/OOS comparison, parameter perturbation, and
  regime analysis

That path is useful for safe research, but it is too narrow for an open-source
research runtime. The project needs to represent execution assumptions,
calendar choices, metric definitions, and validation classifications more
explicitly.

## Design Principles

- Preserve backward compatibility for existing specs.
- Treat `strategy_spec.yaml` as the single source of truth.
- Separate causal validity from conservative research defaults.
- Make all non-conservative assumptions visible in validation and reports.
- Keep production-compatible profiles generic and optional.
- Prefer small, reviewable PRs over a single large migration.

## PR 1: Execution Semantics, Calendars, and Layered Validation

### Goal

Extend spec execution fields and validator output so open-xquant can describe
execution assumptions precisely while preserving the conservative default path.

### Scope

This PR covers:

- `execution.cash_annual_return`
- `execution.lot_size_config`
- `execution.price_type`
- support for `XNYS`, `ARCX`, `XSHG`, `XSHE`
- four validation dimensions:
  - `causal`
  - `executable`
  - `conservative`
  - `production_consistent`

### Non-Goals

- Do not remove `execution.fill_price_mode`.
- Do not require users to adopt production-style execution assumptions.
- Do not implement full intraday execution simulation.
- Do not add broker-specific order book modeling.

### Spec Shape

Existing fields remain valid:

```yaml
execution:
  trade_time: next_open
  fill_price_mode: next_open
  lot_size: 1
  initial_cash: 100000
```

New fields add explicit semantics:

```yaml
execution:
  order_timing: next_session_open
  price_bar: next_session
  price_type: open
  cash_annual_return: 0.0
  lot_size_config:
    default: 100
    by_symbol:
      513100.SS: 100
```

Compatibility rules:

- If only `fill_price_mode` exists, derive the new fields internally.
- If both old and new fields exist, validator checks they agree.
- `lot_size` remains a shorthand for `lot_size_config.default`.

### Price Type Semantics

Supported first-pass values:

- `open`
- `close`
- `mid`
- `avg`

Definitions:

- `open`: bar open.
- `close`: bar close.
- `mid`: project-defined midpoint. First pass should use the existing broker
  behavior unless changed deliberately.
- `avg`: arithmetic OHLC-derived average. The exact formula must be recorded
  in `metrics.json` or `environment.json`.

Future extension:

- `random`
- `vwap`
- `twap`

These require explicit seed, distribution, or intraday data and are out of
scope for PR 1.

### Validator Dimensions

The validator should continue returning a top-level `status`, `errors`, and
`warnings`, but each finding should also carry one or more dimensions.

Example:

```json
{
  "severity": "warning",
  "check": "execution_conservatism",
  "dimensions": ["conservative"],
  "message": "next_session mid fill is causal but not conservative"
}
```

Dimension meanings:

- `causal`: The signal does not depend on data that would be unknown when the
  order decision is made.
- `executable`: The requested execution model can be simulated by the current
  compiler and broker.
- `conservative`: The execution and cost assumptions are suitable for default
  audited research.
- `production_consistent`: The spec intentionally declares assumptions that
  may be valid for replaying an external system, even if they are not
  conservative.

### Validation Rules

Fatal examples:

- same-bar close signal filled with same-bar close or same-bar midpoint
- signal references unknown columns
- fill price requested from a bar that cannot exist in loaded data
- random execution without a reproducible seed
- unknown calendar
- non-finite cost or cash fields

Warning examples:

- next-session midpoint fill
- next-session close fill
- zero fee or zero slippage
- synthetic OHLC-derived price
- static universe without point-in-time metadata

Pass examples:

- close signal, next-session open fill
- explicit positive fee and slippage
- supported exchange calendar
- deterministic lot sizing

### Calendar Support

Allow these calendars in spec validation:

- `XNYS`
- `ARCX`
- `XSHG`
- `XSHE`

Implementation should continue using `exchange_calendars` where available.
If a requested calendar cannot be resolved at runtime, compilation should fail
with a clear message.

### Artifacts

`environment.json` or a new `execution_assumptions.json` should record:

- resolved calendar
- effective price type
- effective price bar
- effective lot size config
- cash annual return
- whether compatibility fields were derived

### Tests

Add tests for:

- old specs still validate and compile
- `cash_annual_return` flows into `Engine.run`
- `lot_size_config.default` is used when set
- `lot_size` fallback still works
- supported calendars pass validation
- unknown calendars fail validation
- next-session midpoint is causal but not conservative
- same-bar midpoint remains fatal
- zero cost becomes a warning when explicitly declared as replay-style

## PR 2: Metrics Profiles and Report Transparency

### Goal

Make metric assumptions explicit so results from different systems or research
profiles are not compared as if they used the same formula.

### Scope

This PR covers:

- `metrics.profile`
- `metrics.risk_free_rate`
- return type
- annualization days
- Calmar denominator
- evaluation window
- report sections that display these assumptions

### Non-Goals

- Do not change existing metric numbers for existing specs unless a metrics
  section opts into new behavior.
- Do not make any external system the default profile.
- Do not add portfolio accounting changes.

### Spec Shape

```yaml
metrics:
  profile: open_xquant_default
  risk_free_rate: 0.0
  return_type: simple
  annualization_days: 252
  calmar_denominator: max_drawdown
  evaluation_window: full
```

Built-in profiles:

- `open_xquant_default`
- `xquant_production`

`xquant_production` is a generic compatibility profile used to demonstrate
that open-xquant can express another metric convention. It should not create a
hard dependency on any proprietary runtime.

### Profile Definitions

`open_xquant_default`:

- return type: simple
- risk-free rate: `0.0`
- annualization days: `252`
- Calmar denominator: absolute max drawdown
- evaluation window: full run unless report section is OOS-specific

`xquant_production`:

- return type: log
- risk-free rate: `0.02`
- annualization days: `252`
- Calmar denominator: absolute max drawdown
- evaluation window: declared metric window

### Metrics Engine

Create a focused metrics module that can compute named profiles from a
`RunResult` or equity curve.

The module should return both values and assumptions:

```json
{
  "metrics_profile": "open_xquant_default",
  "metric_assumptions": {
    "return_type": "simple",
    "risk_free_rate": 0.0,
    "annualization_days": 252,
    "calmar_denominator": "max_drawdown",
    "evaluation_window": "full"
  },
  "annualized_return": 0.123,
  "sharpe_ratio": 1.2
}
```

### Report Changes

Add report sections:

- Metrics Profile
- Metric Assumptions
- Execution Assumptions
- Validation Classification
- IS/OOS Metrics

The report should explicitly state when a run uses non-default metrics.

### Tests

Add tests for:

- default specs produce existing metric behavior
- `xquant_production` computes log-return annualization
- risk-free rate changes Sharpe
- annualization days affect annualized return and volatility
- report prints metric assumptions
- invalid metrics profile fails validation
- profile values are serialized to `metrics.json`

## PR 3: Robustness Implementation and Agent Dependency Guidance

### Goal

Replace robustness placeholders with real checks and make first-time agent
setup less fragile.

### Scope

This PR covers:

- IS/OOS metric diff
- parameter perturbation reruns
- regime segmented statistics
- agent guide dependency instructions
- optional dependency isolation for tools

### Non-Goals

- Do not implement advanced walk-forward optimization in robustness.
- Do not add external macro regime providers.
- Do not require all optional extras for the core CLI to run.

### IS/OOS Metric Diff

Use the run's spec and effective data directory.

Run or derive metrics for:

- train period
- test period

Output:

- IS total return
- IS Sharpe
- IS max drawdown
- IS Calmar
- OOS total return
- OOS Sharpe
- OOS max drawdown
- OOS Calmar
- relative degradation for comparable metrics

Status:

- `pass`: OOS remains within configured degradation thresholds.
- `warn`: OOS degrades materially or thresholds are missing.
- `fail`: OOS Sharpe is negative or drawdown breaches configured policy.

### Parameter Perturbation

Read:

```yaml
robustness:
  parameter_perturbation:
    mom.period: [15, 20, 30]
    vol.period: [15, 20, 30]
```

First implementation:

- clone the spec per target/value
- rerun the backtest
- collect metrics
- compare each perturbed run to baseline

Do not implement full Cartesian grid unless explicitly configured later.
One-at-a-time perturbation is easier to explain and review.

### Regime Analysis

First implementation should be deterministic and provider-free.

Segment by realized market behavior:

- `uptrend`
- `downtrend`
- `high_vol`
- `low_vol`

Inputs:

- strategy equity curve
- default benchmark if present

Outputs per regime:

- date count
- total return
- annualized return
- Sharpe
- max drawdown
- trade count

### Artifacts

Write:

```text
robustness.json
```

into the run directory.

Report should read `robustness.json` when present instead of only echoing
configured robustness fields.

### Agent Dependency Guidance

Update `docs/agent-guide.md`:

- recommend `uv sync --all-extras` for first-time full agent installation
- recommend at least:
  - `uv sync --extra yfinance --extra scipy --extra chart`
- explain that missing optional packages should disable only their feature
  area, not the whole tool layer

Update `oxq doctor` to check and report:

- `pyarrow`
- `exchange_calendars`
- `scipy`
- optional data/chart packages when relevant

### Tool Import Isolation

The `oxq.tools` package should not fail to import all tools just because one
optional feature is missing a package such as `scipy`.

Implementation approach:

- move optional imports into tool functions, or
- catch optional import failures at module registration boundaries and expose
  clear error messages only for affected tools.

### Tests

Add tests for:

- robustness writes `robustness.json`
- IS/OOS comparison produces numeric output
- parameter perturbation reruns at least one changed spec
- invalid perturbation path reports an error for that target only
- regime analysis returns all configured regime buckets
- report reads `robustness.json`
- importing `oxq.tools.engine` does not require `scipy`
- `doctor` reports missing optional dependencies without failing core checks

## Rollout Plan

1. Merge PR 1 first.
   It defines the execution and validation semantics that later reports and
   robustness checks will display.

2. Merge PR 2 second.
   It creates metric profiles and report transparency so later robustness
   output uses explicit formulas.

3. Merge PR 3 last.
   It depends on the earlier spec and metric definitions to avoid duplicate
   implementation work.

## Compatibility

Existing specs should keep working without changes.

Backward-compatible defaults:

- missing `execution.cash_annual_return` means `0.0`
- missing `execution.lot_size_config` falls back to `execution.lot_size`
- missing `execution.price_type` is derived from `fill_price_mode`
- missing `metrics` uses `open_xquant_default`
- existing validator status remains `pass` or `fail`

## Open Questions

1. Should `mid` keep the current open-xquant definition or be renamed if its
   formula changes?
2. Should zero explicit cost require a new explicit flag such as
   `cost.assumption: zero_cost_replay`?
3. Should `ARCX` be treated as a separate trading calendar in all contexts, or
   should it alias to `XNYS` where exchange calendar support is incomplete?
4. Should one-at-a-time perturbation be the only default, with Cartesian grid
   behind a future flag?

## Acceptance Criteria

The full iteration is complete when:

- Existing specs validate and run unchanged.
- A spec can declare cash return, lot sizing, supported calendars, and explicit
  execution price assumptions.
- Validator output distinguishes causal, executable, conservative, and
  production-consistent concerns.
- Metrics JSON and report include metric profile assumptions.
- Robustness no longer reports placeholder warnings for IS/OOS, parameter
  perturbation, or regime analysis.
- Agent guide and doctor make optional dependency expectations clear.
- Missing optional packages do not break unrelated tool imports.
