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
   - scan the run directory for available data assets before recommending
     charts.
   - Do not modify metrics.
   - Do not modify audit artifacts.

2. Recommend a chart set.
   - Use the Chart Applicability Matrix below.
   - List the recommended chart set sorted by rotation-strategy value and data
     availability.
   - Ask the user to confirm the batch before generating charts.
   - Explain when a useful chart cannot be produced from available data.

3. Write plotting Python.
   - Prefer a small script under `report_assets/scripts`.
   - Read run artifacts from the run directory.
   - Write figure outputs under `report_assets/figures`.
   - Keep plotting deterministic and local; do not download new data unless the
     user explicitly asks.
   - For Chinese chart labels, explicitly configure an available CJK font using
     `matplotlib.font_manager.FontProperties(fname=...)` and
     `fontManager.addfont(...)` with a real font file such as
     `/System/Library/Fonts/STHeiti Light.ttc`, PingFang, or Noto Sans CJK.
     If a CJK font cannot be verified, default to English labels instead of
     risking missing glyphs.

4. Register generated assets.

For one asset, use `asset add`:

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

When one plotting script regenerates multiple already-registered figures, use a
batch JSON file and `asset add-batch` so all replaced asset hashes update in one
manifest write:

```json
[
  {
    "id": "equity_curve",
    "file_path": "runs/<run_id>/report_assets/figures/equity_curve.png",
    "title": "Equity curve vs benchmark",
    "caption": "Generated from equity_curve.csv and benchmark_curve.csv.",
    "section": "results",
    "order": 10,
    "source_script": "runs/<run_id>/report_assets/scripts/plot_report_charts.py",
    "source_artifacts": ["equity_curve.csv", "benchmark_curve.csv"]
  },
  {
    "id": "drawdown",
    "file_path": "runs/<run_id>/report_assets/figures/drawdown.png",
    "title": "Drawdown curve",
    "caption": "Generated from equity_curve.csv.",
    "section": "risk",
    "order": 20,
    "source_script": "runs/<run_id>/report_assets/scripts/plot_report_charts.py",
    "source_artifacts": ["equity_curve.csv"]
  }
]
```

```bash
oxq report asset add-batch runs/<run_id>/ runs/<run_id>/report_assets/assets.json
```

After registration, verify every generated figure:

- The image file is non-empty.
- The image dimensions are readable and positive.
- The path is under `report_assets/figures`.
- The figure is present in `report_assets/manifest.json`.
- The manifest hash matches the current file.
- Chinese charts either use a verified CJK font or default to English labels.

Use `oxq report qa runs/<run_id>/` after final Markdown and HTML exist to
re-check image references, fonts, and manifest state.

5. Hand off final writing to `research-report-writer`.

Use `research-report-writer` to read run artifacts, audits, robustness output,
metrics, and registered assets, then write the final Markdown report and render
HTML from that final Markdown. The expected outputs are:

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

## Chart Applicability Matrix

- Equity Curve
  - Data: `equity_curve.csv`, optional `benchmark_curve.csv`
  - Rotation-strategy value: core
- Drawdown
  - Data: `equity_curve.csv`
  - Rotation-strategy value: core
- Monthly Returns Heatmap
  - Data: `equity_curve.csv` with at least three months
  - Rotation-strategy value: high
- IS/OOS Bar Chart
  - Data: `metrics.json` with IS/OOS fields
  - Rotation-strategy value: high
- Cost Sensitivity
  - Data: `robustness.json` with cost stress results
  - Rotation-strategy value: high
- Position Exposure
  - Data: `target_weights.csv`
  - Rotation-strategy value: high
- Trade Distribution
  - Data: non-empty `trades.csv`
  - Rotation-strategy value: medium
- Violin Plot
  - Data: per-symbol return data for at least two assets
  - Rotation-strategy value: high
- Pair Plot
  - Data: per-symbol return data for at least three assets
  - Rotation-strategy value: high
- Parameter Perturbation
  - Data: `robustness.json` with parameter perturbation results
  - Rotation-strategy value: medium
- Regime Analysis
  - Data: `robustness.json` with regime analysis results
  - Rotation-strategy value: medium
- Trade PnL Distribution
  - Data: `trades.csv` with `closed_pnl`
  - Rotation-strategy value: low

## Red Lines

- Do not invent chart data.
- Do not edit backtest artifacts to make a chart look better.
- Do not treat a chart as proof of profitability.
- Do not silently scan random image files; only registered assets enter the
  report.
- Do not overwrite a user script without reading it first.
