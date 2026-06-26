---
name: build-universe
description: >-
  Define open-xquant strategy universes and explain survivorship/PIT
  constraints; use when users choose symbols, indexes, dynamic filters, or
  tradable pools.
---

# Universe Builder

You help the user define the tradable symbol pool.

## Current Stable Spec Path

The audited CLI compiler currently supports:

```yaml
universe:
  type: static
  symbols: ["SPY", "QQQ"]
  point_in_time: false
  survivorship_bias_policy: warn
```

Treat `index` and `filter` universe as SDK or future/extended paths unless you
verify a specific runtime path supports them.

## Required Checks

Before validating or backtesting:

- ensure `symbols` is not empty
- reject unsafe symbols containing `/`, `\`, `.`, `..`, or absolute paths
- make sure local parquet data exists for every symbol
- make benchmark symbols available in the same data directory
- explain survivorship risk when `point_in_time: false`

Check local data:

```bash
uv run python - <<'PY'
from pathlib import Path

data_dir = Path("/path/to/parquet")
symbols = ["SPY", "QQQ"]
missing = [s for s in symbols if not (data_dir / f"{s}.parquet").exists()]
print("missing:", missing)
PY
```

## SDK Path

Use SDK only when the user explicitly needs dynamic universe logic:

```python
from oxq.universe.static import StaticUniverse

universe = StaticUniverse(("SPY", "QQQ", "IWM"))
```

If using `FilterUniverse` or `IndexUniverse`, read the implementation and tests
first, then state that this is outside the current audited CLI spec path.

## Red Lines

- Do not promise point-in-time index membership unless the data source proves it.
- Do not hide survivorship warnings.
- Do not mix unrelated markets or calendars in one spec without explaining the
  data and currency implications.
