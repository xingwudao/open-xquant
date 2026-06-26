---
name: build-report-charts
description: >-
  Use when users want charts, figures, visual evidence, or notebook-like
  report assets for an open-xquant experiment report.
---

# Report Chart Builder

Use this skill after a run exists and the user wants charts in the experiment
report. The Agent should discuss chart requirements, write plotting Python when
needed, save generated figures as experiment assets, and register them before
handing the final narrative to `write-research-report`.

## Workflow

1. Confirm the run directory and required artifacts exist.
   - Read `metrics.json`, `equity_curve.csv`, `benchmark_curve.csv`,
     `trades.csv`, `orders.csv`, `positions.csv`, and `target_weights.csv`
     only if present.
   - scan the run directory for available data assets before recommending
     charts.
   - Do not modify metrics.
   - Do not modify audit artifacts.

2. Discuss chart requirements and recommend a chart set.
   - Use the Chart Applicability Matrix below.
   - Ask what decision the chart should support.
   - Clarify chart type, time range, benchmark, grouping, and labels.
   - List the recommended chart set sorted by rotation-strategy value and data
     availability.
   - If the user does not give a chart list, propose the Default Professional
     Chart Pack with trade curve as the first/default recommendation, then ask
     whether to build the full pack, a smaller subset, or a custom set.
   - Ask the user to confirm the batch before generating charts.
   - Explain when a useful or requested chart cannot be produced from available
     data.

3. Write plotting Python.
   - Prefer a small script under `report_assets/scripts`.
   - Read run artifacts from the run directory.
   - Write figure outputs under `report_assets/figures`.
   - Keep plotting deterministic and local; do not download new data unless the
     user explicitly asks.
   - Prefer `seaborn` for professional chart styling, statistical plots,
     palettes, and grid treatment when it is installed. If `seaborn` is not
     installed, fall back to direct `matplotlib` plotting without failing the
     chart task.
   - Use a small deterministic plotting import block in generated scripts:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

if sns is not None:
    sns.set_theme(style="whitegrid", context="talk")
else:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
```

   - Prefer English chart labels. Use local-language labels only when the final
     rendered image is already known to be readable in the current environment.

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
    "id": "trade_curve",
    "file_path": "runs/<run_id>/report_assets/figures/trade_curve.png",
    "title": "Trade curve",
    "caption": "Generated from equity_curve.csv and trades.csv; markers show fills, not intraday paths.",
    "section": "results",
    "order": 10,
    "source_script": "runs/<run_id>/report_assets/scripts/plot_report_charts.py",
    "source_artifacts": ["equity_curve.csv", "trades.csv"]
  },
  {
    "id": "equity_curve",
    "file_path": "runs/<run_id>/report_assets/figures/equity_curve.png",
    "title": "Equity curve vs benchmark",
    "caption": "Generated from equity_curve.csv and benchmark_curve.csv.",
    "section": "results",
    "order": 20,
    "source_script": "runs/<run_id>/report_assets/scripts/plot_report_charts.py",
    "source_artifacts": ["equity_curve.csv", "benchmark_curve.csv"]
  },
  {
    "id": "drawdown",
    "file_path": "runs/<run_id>/report_assets/figures/drawdown.png",
    "title": "Drawdown curve",
    "caption": "Generated from equity_curve.csv.",
    "section": "risk",
    "order": 30,
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
- Charts default to English labels unless the rendered image proves the chosen
  local-language labels are readable.
- The chart is not blank or visually empty.
- The caption names the source artifact and the interpretation limit.

Use `oxq report qa runs/<run_id>/` after final Markdown and HTML exist to
re-check deterministic report artifacts: image references, dates, and manifest
state. Numeric claim review is semantic/advisory; route it through
`review-research-report` or an explicitly advisory QA pass rather than
treating the CLI command as proof that all numeric claims are sourced.

5. Hand off final writing to `write-research-report`.

Use `write-research-report` to read run artifacts, audits, robustness output,
metrics, and registered assets, then write the final Markdown report and render
HTML from that final Markdown. The expected outputs are:

- `research_report.md`
- `research_report.html`
- `report_assets/manifest.json`
- `report_assets/figures/<figure>.png`
- `report_assets/scripts/<script>.py`

## Common Charts

- Trade curve with buy/sell history.
- Equity curve vs benchmark.
- Drawdown curve.
- Monthly or yearly returns.
- Position exposure over time.
- Turnover or trade count by period.
- Cost impact summary.
- IS/OOS metric comparison.

## Default Professional Chart Pack

Use this pack when the user wants a professional report but does not specify
charts. The trade curve is the default choice because it connects portfolio
performance to the recorded buy/sell history. Skip any chart whose source
artifact is unavailable, and say why.

- trade curve: source artifacts `equity_curve.csv` and non-empty `trades.csv`;
  optional source artifacts `orders.csv`, `target_weights.csv`, and
  `benchmark_curve.csv`; show the portfolio equity curve with buy/sell markers
  by symbol so the user can inspect when each holding was entered, reduced, or
  exited. Use distinct marker shapes or colors for BUY and SELL, keep symbol
  legends readable, and use an event rug, symbol lane, or small multiples when
  dense multi-symbol trades would clutter a single curve. Label only major
  events or the highest-turnover symbols unless the full label set remains
  readable. The message title should state how trading activity aligns with
  equity inflections. The caption must name the fill/order artifacts and state
  that markers represent recorded fills, not intraday execution paths unless
  such data is available.
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

## Chart Applicability Matrix

- Trade Curve
  - Data: `equity_curve.csv`, non-empty `trades.csv`; optional `orders.csv`,
    `target_weights.csv`, `benchmark_curve.csv`
  - Rotation-strategy value: core/default
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
