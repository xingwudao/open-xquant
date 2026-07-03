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
     Chart Pack in the Canonical Report Chart Order, then ask whether to build
     the full pack, a smaller subset, or a custom set.
   - Ask the user to confirm the batch before generating charts.
   - Explain when a useful or requested chart cannot be produced from available
     data.

3. Write plotting Python.
   - Prefer a small script under `report_assets/scripts`.
   - Read run artifacts from the run directory.
   - Write figure outputs under `report_assets/figures`.
   - Keep plotting deterministic and local; do not download new data unless the
     user explicitly asks.
   - Require `seaborn` for the default OpenXQuant report chart style. The
     project `chart` extra includes `seaborn>=0.13`; if import fails, treat it
     as an environment problem and fix the chart environment or block with a
     clear message; do not silently downgrade to arbitrary Matplotlib defaults.
   - In a source worktree, run plotting scripts with
     `uv run --extra chart python runs/<run_id>/report_assets/scripts/<script>.py`
     so the `chart` extra is active. In an installed SDK bundle, verify
     `import seaborn` works in the runner environment before plotting.
   - Use the OpenXQuant Report Chart Style below in every generated script
     before plotting any figure.

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

OXQ_REPORT_STYLE = {
    "figure.figsize": (12, 6.75),
    "figure.dpi": 160,
    "savefig.dpi": 180,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D8DEE9",
    "axes.grid": True,
    "grid.color": "#E6EAF0",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "PingFang SC",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ],
    "axes.unicode_minus": False,
}

OXQ_PALETTE = [
    "#2563EB",  # strategy / primary
    "#6B7280",  # benchmark / neutral
    "#0891B2",  # exposure
    "#D97706",  # cost / stress
    "#7C3AED",  # secondary factor
    "#374151",  # text-adjacent series
]

sns.set_theme(style="whitegrid", context="notebook", palette=OXQ_PALETTE, rc=OXQ_REPORT_STYLE)
plt.rcParams.update(OXQ_REPORT_STYLE)

def market_return_colors(market_region: str) -> tuple[str, str]:
    """Return positive and negative colors for bar/heatmap returns."""
    if market_region == "cn":
        return "#D62728", "#2CA02C"  # red-up / green-down
    return "#059669", "#DC2626"
```

   - Set figure text language from the report language. Default to Chinese
     labels when the report language is Chinese; use the font fallback above and
     inspect the final PNG for readable CJK text. If CJK rendering is unavailable
     after environment repair, use concise English labels and keep Chinese
     captions in the report.
   - For a custom chart requested by the user, keep the same `OXQ_REPORT_STYLE`,
     palette, figure size, title weight, grid, source-artifact caption, and
     registration rules unless the user explicitly asks for a different style.
   - For return bars and heatmaps, use `market.region == cn` to select
     red-up / green-down colors; use green-up / red-down outside China.

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
    "section": "results",
    "order": 20,
    "source_script": "runs/<run_id>/report_assets/scripts/plot_report_charts.py",
    "source_artifacts": ["equity_curve.csv"]
  },
  {
    "id": "trade_curve",
    "file_path": "runs/<run_id>/report_assets/figures/trade_curve.png",
    "title": "Trade curve",
    "caption": "Generated from equity_curve.csv and trades.csv; markers show fills.",
    "section": "results",
    "order": 30,
    "source_script": "runs/<run_id>/report_assets/scripts/plot_report_charts.py",
    "source_artifacts": ["equity_curve.csv", "trades.csv"]
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
- Chart labels follow the report language when the rendered image proves the
  font output is readable; otherwise use concise English labels with local
  language captions.
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

## Canonical Report Chart Order

Use this order for the final report unless the user explicitly requests a
different order or a chart's source artifact is unavailable:

1. `equity_curve`: performance versus benchmark.
2. `drawdown`: depth, duration, and recovery behavior.
3. `trade_curve`: buy/sell fills over the equity curve.
4. `position_exposure`: allocation, concentration, and cash exposure.
5. `monthly_returns`: return distribution and clustering by month.
6. `cost_sensitivity`: realistic fee/slippage stress when available.
7. `is_oos_comparison`: in-sample/out-of-sample evidence when available.
8. `parameter_perturbation`: parameter stability when available.
9. `regime_analysis`: market-regime behavior when available.
10. `trade_pnl_distribution`: dependence on outlier trades when available.

Keep the manifest `order` values aligned with this sequence. If the user asks
for a custom chart, place it after the closest related canonical chart unless
the user gives a specific location. When documenting or registering a canonical
sequence that relies on these global `order` values, keep those figures in the
same manifest `section` because report assets sort by `section`, `order`, then
`id`.

## Default Professional Chart Pack

Use this pack when the user wants a professional report but does not specify
charts. Use this order unless the user explicitly requests a different order.
Skip any chart whose source artifact is unavailable, and say why.

- equity curve vs benchmark: source artifact `equity_curve.csv` and
  `benchmark_curve.csv`; use a message title that states whether the strategy
  outperformed, tracked, or lagged the benchmark.
- drawdown: source artifact `equity_curve.csv`; show depth and recovery
  behavior, not just the maximum drawdown number.
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
- position exposure: source artifact `positions.csv` or `target_weights.csv`;
  show concentration, cash exposure, and large allocation shifts.
- monthly return heatmap or monthly return bars: source artifact
  `equity_curve.csv`; show positive/negative month distribution and clustering.
- cost sensitivity: source artifact `robustness.json`; show the effect of
  `cost_multiplier` scenarios when present. Do not treat `2x` of zero or
  near-zero costs as real cost robustness; add a realistic fee/slippage scenario
  when the artifacts support it, or state that the chart is not informative.
- IS/OOS comparison: source artifact `metrics.json` and facts API values; show
  whether out-of-sample evidence supports the in-sample thesis.
- parameter perturbation: source artifact `robustness.json`; show whether
  nearby parameters preserve or destroy the thesis when
  `parameter_perturbation` exists.
- regime analysis: source artifact `robustness.json`; show performance by
  market regime when `regime_analysis` is available.
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
