---
name: create-portfolio-optimizer
description: >-
  Create a new open-xquant PortfolioOptimizer with weight-invariant tests and
  registry wiring; use after create-component confirms no existing optimizer
  satisfies the allocation request.
---

# Create PortfolioOptimizer

You create allocation logic that returns target weights.

## Scope

Default built-in paths:

- source: `src/oxq/portfolio/{snake_name}.py`
- tests: `tests/portfolio/test_{snake_name}.py`
- package export: `src/oxq/portfolio/__init__.py`
- built-in registry: `src/oxq/core/registry.py`

Existing built-ins live in `src/oxq/portfolio/optimizers.py`; read that file
before choosing whether to add a new module or extend the existing built-in
module. Prefer a new module for a new component unless project maintainers ask
otherwise.

## Phase 1: Read Existing Patterns

Read before editing:

- `src/oxq/core/types.py`
- `src/oxq/portfolio/optimizers.py`
- one existing test in `tests/portfolio/`
- `src/oxq/portfolio/__init__.py`
- the portfolio registration block in `src/oxq/core/registry.py`

## Phase 2: Define Behavior

State before coding:

- allocation formula
- constructor parameters
- whether it reads `signals`, `indicators`, or both
- required indicator columns
- fallback when no valid inputs exist
- whether weights can include `CASH`
- max/min weight constraints
- whether the optimizer is stateful. If it consumes categorical signals such as
  `BUY`, `SELL`, and `HOLD`, define how `HOLD` preserves or resets prior
  target weights. `SignalToPosition` is the built-in reference pattern.

Ask the user if allocation logic is ambiguous.

## Phase 3: Test First

Write tests with deterministic DataFrames:

- protocol compliance with `PortfolioOptimizer`
- empty input returns `{"CASH": 1.0}`
- weights sum to `1.0`
- multi-symbol behavior
- hand-calculated allocation
- invalid or NaN input behavior
- `name` value

Run the new test and confirm the missing implementation fails before coding.

```bash
uv run pytest tests/portfolio/test_{snake_name}.py -v
```

## Phase 4: Implement

Skeleton:

```python
"""Short description portfolio optimizer."""

from __future__ import annotations

import pandas as pd


class ClassName:
    """Short allocation description."""

    name = "ClassName"

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Return target weights that sum to 1.0."""
        ...
```

Rules:

- every path returns a non-empty dict
- invalid input returns `{"CASH": 1.0}`
- weights must sum to `1.0`
- include `CASH` for unused capital
- do not mutate input DataFrames
- use only latest available row unless design says otherwise

## Phase 5: Register

For a built-in component:

- import it in `src/oxq/portfolio/__init__.py`
- add it to `__all__`
- add it to `_load_builtins()` in `src/oxq/core/registry.py`

Verify:

```bash
uv run python - <<'PY'
import oxq
assert "ClassName" in oxq.list_portfolio_optimizers()
print("registered")
PY
uv run pytest tests/portfolio/test_{snake_name}.py -v
```

Run engine-level tests when optimizer behavior changes order generation.

## Red Lines

- Do not return weights that sum above or below `1.0`.
- Do not return an empty dict.
- Do not ignore invalid data silently; fall back to `CASH` or exclude the
  symbol deliberately.
- Do not register before invariant tests pass.
