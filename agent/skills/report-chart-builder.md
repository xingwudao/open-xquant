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
   - If the user does not give a chart list, propose the Default Professional
     Chart Pack and ask whether to build the full pack, a smaller subset, or a
     custom set.

3. Write plotting Python.
   - Prefer a small script under `report_assets/scripts`.
   - Read run artifacts from the run directory.
   - Write figure outputs under `report_assets/figures`.
   - Keep plotting deterministic and local; do not download new data unless the
     user explicitly asks.
   - For Chinese chart labels, explicitly configure an available CJK font such
     as `Noto Sans CJK`, `Microsoft YaHei`, `PingFang`, or `SimHei`. If a CJK
     font cannot be verified, default to English labels instead of risking
     missing glyphs.

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
- The chart is not blank or visually empty.
- The caption names the source artifact and the interpretation limit.

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

## Default Professional Chart Pack

Use this pack when the user wants a professional report but does not specify
charts. Skip any chart whose source artifact is unavailable, and say why.

- equity curve vs benchmark: source artifact `equity_curve.csv` and
  `benchmark_curve.csv`; use a message title that states whether the strategy
  outperformed, tracked, or lagged the benchmark.
- drawdown: source artifact `equity_curve.csv`; show depth and recovery
  behavior, not just the maximum drawdown number.
- monthly return heatmap or monthly return bars: source artifact
  `equity_curve.csv`; show positive/negative month distribution and clustering.
- IS/OOS comparison: source artifact `metrics.json` and facts API values; show
  whether out-of-sample evidence supports the in-sample thesis.
- cost sensitivity: source artifact `robustness.json`; show the effect of
  `cost_multiplier` scenarios when present.
- parameter perturbation: source artifact `robustness.json`; show whether
  nearby parameters preserve or destroy the thesis when
  `parameter_perturbation` exists.
- regime analysis: source artifact `robustness.json`; show performance by
  market regime when `regime_analysis` is available.
- position exposure: source artifact `positions.csv` or `target_weights.csv`;
  show concentration, cash exposure, and large allocation shifts.
- trade PnL distribution: source artifact `trades.csv`; show whether results
  depend on a few outliers when closed-trade PnL is available.

Every professional chart must have a message title, a caption, and registered
metadata that names each source artifact. A chart may make the report more
readable, but it does not replace artifact-backed evidence.

## Red Lines

- Do not invent chart data.
- Do not edit backtest artifacts to make a chart look better.
- Do not treat a chart as proof of profitability.
- Do not silently scan random image files; only registered assets enter the
  report.
- Do not overwrite a user script without reading it first.
