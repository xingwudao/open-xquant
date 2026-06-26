---
name: create-signal
description: >-
  Create a new open-xquant Signal with deterministic output-domain tests and
  registry wiring; use after create-component confirms no existing signal
  satisfies the request.
---

# Create Signal

You create a vectorized trading-intent component.

## Scope

Default built-in paths:

- source: `src/oxq/signals/{snake_name}.py`
- tests: `tests/signals/test_{snake_name}.py`
- package export: `src/oxq/signals/__init__.py`
- built-in registry: `src/oxq/core/registry.py`

## Phase 1: Read Existing Patterns

Read before editing:

- `src/oxq/core/types.py`
- `src/oxq/signals/crossover.py`
- `src/oxq/signals/threshold.py`
- one existing test in `tests/signals/`
- `src/oxq/signals/__init__.py`
- the signal registration block in `src/oxq/core/registry.py`

Confirm the requested output is a Signal, not an Indicator.

## Phase 2: Define Output Semantics

State before coding:

- when the signal fires
- parameters and defaults
- whether output is boolean or categorical
- exact meaning of each output value
- NaN and boundary behavior
- whether it is causal

If the signal needs future rows to identify peaks, centered windows, or
month-end rows, warn that it may be unsuitable for audited causal specs.

## Phase 3: Test First

Write tests with hand-crafted data:

- protocol compliance with `Signal`
- non-empty `name`
- output domain is boolean or the declared categorical set
- categorical trading-intent signals must use exact uppercase labels such as
  `BUY`, `SELL`, and `HOLD`
- if a categorical custom signal is used from spec, declare
  `signal.rules.<name>.output_domain: [BUY, SELL, HOLD]` as rule metadata;
  do not place `output_domain` in `params` because `params` are passed to
  `Signal.compute()`
- trigger scenario
- no-trigger scenario
- NaN or insufficient-history behavior when relevant
- causal threshold behavior; rolling thresholds must not include the current
  row when they are used to classify that row

Run the new test and confirm the missing implementation fails before coding.

```bash
uv run pytest tests/signals/test_{snake_name}.py -v
```

## Phase 4: Implement

Skeleton:

```python
"""Short description signal."""

from __future__ import annotations

import pandas as pd


class ClassName:
    """True when the declared condition is met."""

    name = "ClassName"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
    ) -> pd.Series:
        """Return boolean series where True means enter or activate."""
        ...
```

Rules:

- use explicit keyword parameters, not `**kwargs`
- no I/O or state mutation
- preserve index alignment
- fill or document NaN behavior
- return a `pd.Series`

## Phase 5: Register

For a built-in component:

- import it in `src/oxq/signals/__init__.py`
- add it to `__all__`
- add it to `_load_builtins()` in `src/oxq/core/registry.py`

Verify:

```bash
uv run python - <<'PY'
import oxq
assert "ClassName" in oxq.list_signals()
print("registered")
PY
uv run pytest tests/signals/test_{snake_name}.py -v
```

Run broader tests when registry or compiler behavior changes.

## Red Lines

- Do not return floats from a Signal unless the output is deliberately
  categorical and documented.
- Do not introduce future-data bias without making it explicit.
- Do not register before output-domain tests pass.
- Do not modify existing signals for a new behavior.
