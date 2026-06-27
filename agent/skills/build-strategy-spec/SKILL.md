---
name: build-strategy-spec
description: >-
  Build open-xquant strategy_spec.yaml files from user strategy ideas for
  multi-Agent systems; stops after deterministic validation and writes a builder
  phase result for downstream orchestration.
---

# Strategy Builder

Build or edit a `strategy_spec.yaml`. This skill is for multi-Agent systems
where strategy construction, audit, execution, monitoring, and reporting may be
handled by separate Agents.

## Scope

Do:

- convert the user's strategy idea into `strategy_spec.yaml`
- use the component catalog and canonical recipes before choosing components
- preserve explicit user requirements and documented defaults
- write concise `spec_build_notes.md` when component or recipe choices matter
- run deterministic spec validation
- write `builder_phase_result.json` for the downstream orchestrator

Do not:

- produce `spec_audit.json`
- approve assumptions on behalf of the user
- call audit skills
- download market data
- run `oxq strategy compile`
- run `oxq backtest run`
- attach provenance
- run monitoring, robustness, report writing, or report review
- describe an unaudited spec as ready for formal research

## Runner Resolution

In a new research directory, `uv run oxq` may fail because open-xquant is
installed as a long-lived Agent capability, not as a package in that directory.
Before running commands:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

## Stable Spec Defaults

Use current stable CLI behavior first as proposed defaults. Record each default
in build notes so the spec auditor can ask for grouped user confirmation before
any formal backtest:

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
- explicit data warmup policy through `data.min_start_date` when indicators or
  rules need lookback data before the evaluated interval
- `portfolio.type: EqualWeight` only for boolean signal filters
- `ROC` + `ROCTiming` + `SignalToPosition` for single-symbol timing strategies
  that need explicit `BUY` / `SELL` / `HOLD` and HOLD-maintains-position
  semantics

If the user supplies one complete backtest period and does not ask for an
IS/OOS split, encode the full period as `validation.test_period` with
`validation.required_oos: false`. Do not split the full period into train/test
or set `required_oos: true` unless the user confirms that validation plan.

For lookback indicators, define data warmup deliberately:

- If the user wants a true full-interval evaluation from the first test date,
  set `data.min_start_date` earlier than the evaluation start so the largest
  lookback has enough prior bars.
- If no pre-window data is available or the user accepts first-window warmup
  NaNs/cash behavior, leave `data.min_start_date` empty only after recording
  that policy in `spec_build_notes.md`.
- Do not silently let two otherwise identical specs differ only because one
  Agent fetched warmup history and another did not.

## Component Catalog Gate

Before editing `strategy_spec.yaml`, run the component catalog gate. This gate
is mandatory for every new spec and every material component edit:

1. Load `component_catalog.json` from the research directory when it exists.
   If it is missing or stale, create it with:

   ```bash
   uv run oxq registry export --out component_catalog.json
   ```

2. Search exact names and aliases in the catalog for every requested
   indicator, signal, portfolio optimizer, and rule. Prefer `source: builtin`
   components over custom components whenever they satisfy the user request.
3. Search `recipes` before composing custom indicator chains or portfolio
   structures. Match exact recipe names, aliases, and definitions against the
   user's requested semantics.
4. If a recipe matches, use its `canonical_spec` structure and fill only the
   placeholders the user supplied or confirmed, such as `$period`, `$score_col`,
   or `$n`.
5. If no built-in component or recipe matches a requested semantic component,
   do not invent a component name and do not route to component code creation.
   Mark the builder phase as blocked with `needs_custom_component`.
6. Record selected components, selected recipes, `catalog_hash`, and concise
   reasons in `spec_build_notes.md` or equivalent task-local build notes.

Examples:

- User asks for "20日收益率 / 20日波动率": use the
  `volatility_adjusted_momentum` recipe, not an invented
  `RiskAdjustedMomentum` component.
- User asks for "SMA 金叉": use the `sma_golden_cross` recipe.
- User asks for "ROC timing": use the `roc_timing` recipe.
- User asks for "TopN 正动量轮动": use the
  `top_n_positive_momentum_rotation` recipe.
- User asks to "取 TopN，归一化权重": use `TopNRanking`, not `EqualWeight`.

## Build Flow

Initialize when no spec exists:

```bash
uv run oxq spec init "<strategy idea>" --out strategy_spec.yaml
```

Edit `strategy_spec.yaml` so it contains user-confirmed values and explicit
documented defaults. Defaults are only proposals at this phase; the downstream
spec auditor must confirm every effective field, including parser/runtime
defaults injected by OpenXQuant, before formal backtest.

For categorical custom signals, declare output domain as rule metadata:

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

For rebalance throttling, use the built-in rule:

```yaml
portfolio:
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 10
```

Do not also set a conflicting `execution.rebalance.interval_days` value.

## Validate And Output

Run deterministic validation:

```bash
uv run oxq spec validate strategy_spec.yaml
```

Fix fatal validation errors before completing the builder phase. Warnings are
allowed, but record them for the downstream orchestrator.

Write `builder_phase_result.json` after validation:

```json
{
  "status": "pass | blocked | fail",
  "strategy_spec": "strategy_spec.yaml",
  "component_catalog": "component_catalog.json",
  "spec_build_notes": "spec_build_notes.md",
  "validation": {
    "status": "pass | fail",
    "spec_hash": "sha256:<hash>",
    "warnings": [],
    "errors": []
  },
  "selected_components": [],
  "selected_recipes": [],
  "data_warmup_policy": {
    "status": "confirmed | default | blocked",
    "min_start_date": "",
    "reason": ""
  },
  "needs_custom_component": [],
  "next_required_phase": "audit | component_authoring"
}
```

The builder phase output is:

- `strategy_spec.yaml`
- `component_catalog.json`
- `spec_build_notes.md` or equivalent build notes
- `builder_phase_result.json`

This builder skill stops after writing those artifacts. The downstream
multi-Agent system decides which audit, compile, execution, monitor, or report
worker runs next.

## Red Lines

- Do not skip the component catalog gate.
- Do not invent component names when a catalog component or recipe exists.
- Do not convert a full backtest interval into required OOS without user
  confirmation.
- Do not write `spec_audit.json`.
- Do not call audit, compile, backtest, monitor, robustness, experiment, or
  report skills from this builder skill.
- Do not call component creation skills from this builder skill. When a custom
  component is required, write `needs_custom_component` and stop.
- Do not run formal backtests from this skill.
