---
name: evaluate-time-series
description: >-
  Evaluate time-series factors with hit rate, decay curve, profit/loss ratio,
  cash-period behavior, and tearsheets in open-xquant; use for timing or
  rotation signals.
---

# Evaluate Time-Series Factor

Use this when the factor predicts direction through time for one asset or a
small rotation set.

## Minimal SDK Pattern

```python
from pathlib import Path

from oxq.factor_eval.bundle import create_bundle
from oxq.factor_eval.tearsheet import generate_tearsheet

factor_series = factor_df.stack().rename_axis(["date", "asset"]).rename("factor")
bundle = create_bundle(
    factor_values=factor_series,
    prices=prices_df,
    forward_periods=[1, 5, 20],
)

out = Path("/tmp/oxq_tearsheet")
result = generate_tearsheet(
    bundle=bundle,
    forward_periods=[1, 5, 20],
    output_dir=str(out),
)
```

If `matplotlib` is missing, install the chart extra:

```bash
uv sync --extra chart
```

## Review Checklist

- t-day factor is evaluated against future returns, not same-day returns
- factor timestamps and price timestamps are aligned
- hit rate and P/L ratio are both reviewed
- decay curve is checked across multiple horizons
- cash or no-position periods are counted
- impossible trading days are excluded when that data exists

## Interpretation

- hit rate above `55%` can be useful only if P/L ratio is acceptable
- high hit rate with low P/L ratio can still lose money
- fast decay implies execution timing matters
- long cash periods reduce capital usage and should be reported

## Red Lines

- Do not report hit rate alone.
- Do not skip T+1 alignment.
- Do not treat a tearsheet image as an audit.
