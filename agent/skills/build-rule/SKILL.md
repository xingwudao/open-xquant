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
the spec compiler internally. The audited CLI spec path does not currently
support a generic top-level `rules:` YAML section or `oxq backtest run --rules`.

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
- `BlacklistRule`

## Safe CLI Path

For `Crossover` specs, `compile_run()` automatically adds `ExitRule` using the
declared `fast` and `slow` columns. For rebalance throttling, use:

```yaml
execution:
  rebalance:
    frequency: daily
    interval_days: 5
```

For other rules, use SDK execution until generic rule YAML support exists.

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
