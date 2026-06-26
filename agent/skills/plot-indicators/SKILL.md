---
name: plot-indicators
description: >-
  Render open-xquant run charts and indicator overlays; use when users ask to
  visualize price, indicators, signals, or chart artifacts.
---

# Chart Indicator

You create visual checks. Charts do not replace validation or audit.

## Current Tool Signature

`oxq.tools.chart.chart_indicator` plots indicator columns from a stored
`RunResult` in the tool session:

```python
from oxq.tools.chart import chart_indicator

result = chart_indicator(
    run_id="run_1",
    symbol="SPY",
    columns=["sma_fast", "sma_slow"],
    overlay=True,
)
print(result)
```

It does not accept raw `data=...`, `indicators=...`, or `output=...`
arguments. If you need a chart from raw bars, either run the strategy through
the tool/session flow first or write a one-off exploratory script and label it
as non-standard.

## Dependency

Install chart dependencies before rendering:

```bash
uv sync --extra chart
```

If using pip:

```bash
python -m pip install -e ".[chart]"
```

## What To Check

- requested columns exist in the run's symbol data
- NaN warmup regions are expected
- indicator scale matches overlay choice
- signal events line up with intended dates
- chart output path exists

## Red Lines

- Do not use charts as proof of profitability.
- Do not hide missing indicator columns by plotting a different column.
- Do not infer causality from visual overlap alone.
