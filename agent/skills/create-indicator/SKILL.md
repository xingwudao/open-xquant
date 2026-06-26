---
name: create-indicator
description: >-
  Create a new open-xquant Indicator with tests and registry wiring; use after
  create-component confirms no existing indicator satisfies the request.
---

# Create Indicator

You create a pure numeric time-series component.

## Scope

Default built-in paths:

- source: `src/oxq/indicators/{snake_name}.py`
- tests: `tests/indicators/test_{snake_name}.py`
- package export: `src/oxq/indicators/__init__.py`
- built-in registry: `src/oxq/core/registry.py`

If the user is building a third-party extension instead of a built-in, use
dynamic `oxq.register_indicator()` in that package and state that it is not
persistently built into open-xquant.

## Phase 1: Read Existing Patterns

Read before editing:

- `src/oxq/core/types.py`
- `src/oxq/indicators/sma.py`
- one similar indicator module
- one existing test in `tests/indicators/`
- `src/oxq/indicators/__init__.py`
- the indicator registration block in `src/oxq/core/registry.py`

Confirm:

- class name is PascalCase
- file name is snake_case
- `name` equals the registry name
- `compute()` returns `pd.Series`
- computation is pure and deterministic

## Phase 2: Define Behavior

State the design before coding:

- formula
- input columns
- parameters and defaults
- output units and sign
- NaN behavior
- constant-price behavior
- dependency requirements

Ask the user if the formula or defaults are ambiguous.

## Phase 3: Test First

Write tests with hand-calculated expectations. Include:

- protocol compliance with `Indicator`
- output is a `pd.Series` with same index
- at least one hand-calculated value
- insufficient-history or NaN behavior
- constant-price or zero-denominator edge case when relevant
- `name` value

Run the new test and confirm it fails for the missing implementation before
writing source when this is a new built-in.

```bash
uv run pytest tests/indicators/test_{snake_name}.py -v
```

## Phase 4: Implement

Use pandas, numpy, and stdlib by default. For optional third-party dependency:

- add or reuse an optional dependency group in `pyproject.toml`
- import lazily or raise a clear install message
- validate with the extra installed

Implementation skeleton:

```python
"""Short description indicator."""

from __future__ import annotations

import pandas as pd


class ClassName:
    """Short behavior description."""

    name = "ClassName"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        period: int = 20,
    ) -> pd.Series:
        """Return the indicator series."""
        ...
```

Composite indicators should compute dependencies internally unless the existing
engine dependency pattern is explicitly being used.

## Phase 5: Register

For a built-in component:

- import it in `src/oxq/indicators/__init__.py`
- add it to `__all__`
- add it to `_load_builtins()` in `src/oxq/core/registry.py`
- add metadata if appropriate

Verify:

```bash
uv run python - <<'PY'
import oxq
assert "ClassName" in oxq.list_indicators()
print("registered")
PY
uv run pytest tests/indicators/test_{snake_name}.py -v
```

Run broader tests when registry or shared behavior changes.

## Red Lines

- Do not copy expected values from the implementation.
- Do not add side effects, I/O, random data, or mutable global state.
- Do not modify unrelated indicators.
- Do not register before tests pass.
