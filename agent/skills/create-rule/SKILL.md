---
name: create-rule
description: >-
  Create a new open-xquant Rule with portfolio-state tests and registry wiring;
  use after create-component confirms no existing rule satisfies the requested
  risk or exit behavior.
---

# Create Rule

You create bar-by-bar risk, hold, weight override, or exit logic.

## Scope

Default built-in paths:

- source: `src/oxq/rules/{snake_name}.py`
- tests: `tests/rules/test_{snake_name}.py`
- package export: `src/oxq/rules/__init__.py`
- built-in registry: `src/oxq/core/registry.py`

## Phase 1: Read Existing Patterns

Read before editing:

- `src/oxq/core/types.py`
- `src/oxq/core/engine.py` rule handling in `step()`
- `src/oxq/rules/constraint.py`
- `src/oxq/rules/order.py`
- one existing test in `tests/rules/`
- `src/oxq/rules/__init__.py`
- the rule registration block in `src/oxq/core/registry.py`

Important current engine behavior:

- pre-trade consumes `RuleResult.weights`
- pre-trade consumes `RuleResult.hold`
- post-trade consumes `RuleResult.target_positions`
- `RuleResult.constraints` exists in the type but is not currently applied by
  `Engine.step()`

## Phase 2: Define Behavior

State before coding:

- pre-trade or post-trade
- trigger condition
- fields returned in `RuleResult`
- constructor parameters
- internal state, if any
- reset behavior, if any
- exact no-trigger result

Ask the user if risk thresholds or trigger semantics are ambiguous.

## Phase 3: Test First

Write tests with hand-built portfolio and bar rows:

- protocol compliance with `Rule`
- trigger scenario
- no-trigger scenario returns empty `RuleResult()`
- correct `reason` when activated
- no mutation of `Portfolio`
- stateful behavior if relevant
- `name` value

Run the new test and confirm the missing implementation fails before coding.

```bash
uv run pytest tests/rules/test_{snake_name}.py -v
```

## Phase 4: Implement

Skeleton:

```python
"""Short description rule."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, RuleResult


class ClassName:
    """Short behavior description."""

    name = "ClassName"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        """Return a RuleResult when the rule activates."""
        ...
```

Rules:

- read `portfolio`; never mutate it
- use `prices` when current bar valuation is needed
- return `RuleResult()` when inactive
- include `reason` when active
- use `target_positions={symbol: 0.0}` for exits
- use `hold=True` for full-bar trading freeze

## Phase 5: Register

For a built-in component:

- import it in `src/oxq/rules/__init__.py`
- add it to `__all__`
- add it to `_load_builtins()` in `src/oxq/core/registry.py`

Verify:

```bash
uv run python - <<'PY'
import oxq
assert "ClassName" in oxq.list_rules()
print("registered")
PY
uv run pytest tests/rules/test_{snake_name}.py -v
```

Run engine-level tests when the rule affects order generation or lifecycle.

## Red Lines

- Do not use `constraints` for execution-critical behavior unless engine
  support is implemented and tested.
- Do not mutate `Portfolio` inside `evaluate()`.
- Do not omit no-trigger tests.
- Do not register before rule behavior tests pass.
