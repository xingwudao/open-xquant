---
name: report-chart-builder
description: >-
  Use when users want charts, figures, visual evidence, or notebook-like
  report assets for an open-xquant experiment report.
---

# Report Chart Builder

Use this skill after a run exists and the user wants charts in the experiment
report. The Agent should discuss chart requirements, write plotting Python when
needed, save generated figures as experiment assets, and register them before
handing the final narrative to `research-report-writer`.

## Workflow

1. Confirm the run directory and required artifacts exist.
   - Read `metrics.json`, `equity_curve.csv`, `benchmark_curve.csv`,
     `trades.csv`, `positions.csv`, and `target_weights.csv` only if present.
   - Do not modify metrics.
   - Do not modify audit artifacts.

2. Discuss chart requirements with the user.
   - Ask what decision the chart should support.
   - Clarify chart type, time range, benchmark, grouping, and labels.
   - Explain when the requested chart cannot be produced from available data.

3. Write plotting Python.
   - Prefer a small script under `report_assets/scripts`.
   - Read run artifacts from the run directory.
   - Write figure outputs under `report_assets/figures`.
   - Keep plotting deterministic and local; do not download new data unless the
     user explicitly asks.

4. Register generated assets.

```bash
oxq report asset add runs/<run_id>/ runs/<run_id>/report_assets/figures/<figure>.png \
  --id <stable_id> \
  --title "<human title>" \
  --caption "<data source and interpretation limits>" \
  --section results \
  --order 10 \
  --source-script runs/<run_id>/report_assets/scripts/<script>.py \
  --source-artifact equity_curve.csv
```

5. Generate the evidence brief and hand off final writing.

```bash
oxq report write runs/<run_id>/ --lang zh --format markdown --out runs/<run_id>/report_evidence.md
```

Then use `research-report-writer` to write the final Markdown report and render
HTML from that final Markdown. The expected outputs are:

- `report_evidence.md`
- `research_report.md`
- `research_report.html`
- `report_assets/manifest.json`
- `report_assets/figures/<figure>.png`
- `report_assets/scripts/<script>.py`

## Common Charts

- Equity curve vs benchmark.
- Drawdown curve.
- Monthly or yearly returns.
- Position exposure over time.
- Turnover or trade count by period.
- Cost impact summary.
- IS/OOS metric comparison.

## Red Lines

- Do not invent chart data.
- Do not edit backtest artifacts to make a chart look better.
- Do not treat a chart as proof of profitability.
- Do not silently scan random image files; only registered assets enter the
  report.
- Do not overwrite a user script without reading it first.
