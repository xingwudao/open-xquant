---
name: build-rule
description: >-
  Configure and reason about open-xquant trading rules, exits, risk holds, and
  rebalance limits; use when the user asks for stop loss, take profit, drawdown
  guards, holding limits, or rule components.
---

# Rule Builder

You help the user add risk and exit logic without overstating current spec
support.

## Current Execution Model

Rules are Python objects passed to `Engine.run(..., rules=[...])` or added by
the spec compiler internally. The audited CLI spec path does not support a
generic top-level `rules:` YAML section or `oxq backtest run --rules`, but it
does support a narrow whitelist under `portfolio.rules`.

In current `Engine.step()`:

- pre-trade rule effects consumed: `RuleResult.weights`, `RuleResult.hold`
- post-trade rule effects consumed: `RuleResult.target_positions`
- declared but not currently consumed by `Engine`: `RuleResult.constraints`

Do not tell the user that a `constraints` field will affect execution unless
you have verified the engine path being used consumes it.

## Built-in Rules

Inspect the live registry:

```bash
uv run python - <<'PY'
import oxq
print(sorted(oxq.list_rules()))
PY
```

Common built-ins:

- `ExitRule`
- `StopLossRule`
- `TakeProfitRule`
- `TrailingStopRule`
- `MaxDrawdownRisk`
- `DailyLossLimitRisk`
- `MaxHoldingsRule`
- `RebalanceFrequencyRule`
- `CalendarRebalanceRule`
- `BlacklistRule`

## Safe CLI Path

For `Crossover` specs, `compile_run()` automatically adds `ExitRule` using the
declared `fast` and `slow` columns. For every explicit `portfolio.rules` item,
all required params must be present; do not rely on runtime constructor
defaults.

For rebalance throttling by trading-session count, use:

```yaml
execution:
  rebalance:
    frequency: daily
    interval_days: 5
```

or:

```yaml
portfolio:
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 5
```

For calendar rebalance, use the execution schedule. The compiler emits
`CalendarRebalanceRule` internally:

```yaml
execution:
  rebalance:
    frequency: monthly
    schedule: month_start
```

Supported audited `portfolio.rules` entries are:

```yaml
portfolio:
  rules:
    stop_loss:
      type: StopLossRule
      params:
        threshold: 0.05
    take_profit:
      type: TakeProfitRule
      params:
        threshold: 0.20
    trailing_stop:
      type: TrailingStopRule
      params:
        trail_pct: 0.08
    max_drawdown:
      type: MaxDrawdownRisk
      params:
        max_drawdown: 0.15
    daily_loss:
      type: DailyLossLimitRisk
      params:
        max_daily_loss: 0.03
    max_holdings:
      type: MaxHoldingsRule
      params:
        max_holdings: 10
```

For portfolio target constraints, use:

```yaml
portfolio:
  constraints:
    max_weight: 0.2
    min_weight: 0.01
    max_holdings: 10
    cash_reserve: 0.05
```

`portfolio.constraints.min_position_value` is parsed but blocked by validation
because the audited runtime cannot execute it yet.

Use SDK execution for rules outside the whitelist until the SPEC validator,
compiler, runtime, and artifact audit support them.

## SDK Pattern

```python
from oxq.core import Engine
from oxq.rules import StopLossRule, TakeProfitRule

engine = Engine()
result = engine.run(
    strategy=strategy,
    market=market,
    broker=broker,
    start="2020-01-01",
    end="2024-12-31",
    rules=[
        StopLossRule(threshold=0.05),
        TakeProfitRule(threshold=0.20),
    ],
)
```

After SDK runs, make sure the user still gets equivalent artifacts or a clear
statement that the run is exploratory rather than the standard CLI artifact
pipeline.

## When A New Rule Is Needed

Load `agent/skills/create-component/SKILL.md`, then route to
`agent/skills/create-rule/SKILL.md`.

Before creating a rule, confirm:

- pre-trade or post-trade category
- trigger condition
- exact `RuleResult` field to use
- whether internal state is required
- how it will be tested with hand-crafted portfolio and bar data

## Red Lines

- Do not document unsupported YAML `rules:` as working CLI behavior.
- Do not claim `constraints` changes execution unless verified in code.
- Do not mutate `Portfolio` inside `evaluate()`.
- Do not treat a stop-loss rule as a substitute for research audit.
