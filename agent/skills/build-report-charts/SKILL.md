---
name: build-report-charts
description: >-
  Use when report chart assets, figures, visual evidence, plotting scripts, or
  notebook-like assets are required for an open-xquant experiment report.
---

## Phase Path Containment Preflight

Before any phase artifact read, write, directory creation, command, or handoff,
run this preflight completely and block on any failure:

1. Read `.open-xquant/workspace.yaml`. Resolve `version_root` from
   `paths.versions_dir`, using `versions` only when that key is absent.
   Require `version_root` to be a safe workspace-relative path: reject an
   absolute path or any `..` path segment, resolve it canonically with
   symlinks, and require the result to stay inside the workspace.
2. Read `current.json`. For normal phase work, set `expected_version_id` to
   `current.json.active_version`; it must exist. Only a contract that
   explicitly owns cross-version inspection may instead set
   `expected_version_id` from the referenced version id for that historical
   read. This exception never permits active-version work to consume another
   version.
3. Set the intended version directory to
   `<version_root>/<expected_version_id>/` and resolve it canonically. Require
   the intended version directory to remain inside the canonical version root
   and workspace; otherwise treat it as a symlink escape. Read
   `version_manifest.json` only from that exact directory. The manifest
   `version_id` must equal `expected_version_id`; for normal phase work it
   therefore must also equal `current.json.active_version`.
4. Before using each required `phase_paths` value, require a non-empty
   workspace-relative string. Reject an absolute path and any `..` path
   segment. Resolve `<workspace>/<phase_path>` canonically, including existing
   symlink ancestors even when the leaf will be created, and require the target
   to be the intended version directory or a descendant of it. A symlink escape
   outside that directory is invalid.
5. On any identity or path failure, stop before phase artifact reads, writes,
   directory creation, commands, or handoffs. Do not normalize an unsafe path
   into acceptance and do not fall back to a default phase path.

Block examples when `expected_version_id` is `v001` include
`strategy_store/v001/../v002/04_spec_build`,
`strategy_store/v002/04_spec_build`, `/tmp/04_spec_build`, and
`strategy_store/v001/escape/04_spec_build` when `escape` is a symlink
whose target is outside the intended version directory. An allowed custom
nested phase path is
`strategy_store/v001/custom/phases/04_spec_build` when its canonical
target remains under the intended version directory.

For a new-version bootstrap, only `manage-strategy-version` may proceed before
the new manifest exists or the new id becomes active. It must apply the same
workspace-relative, traversal, canonical-containment, and symlink checks to
every constructed phase path before directory creation, then write a matching
manifest before publishing `current.json` last.

## Version Path Resolution

Before using any `<phase_paths.*>` or `<version_root>` placeholder, read
`.open-xquant/workspace.yaml`. Resolve `version_root` from
`paths.versions_dir`; use `versions` only when that key is absent. Require a
safe relative path whose resolved target stays inside the workspace. Then read
`<version_root>/<version_id>/version_manifest.json` and use its exact
`phase_paths` entry for each phase. For example, a configured root of
`research_versions` must resolve the spec-build phase to
`research_versions/v003/04_spec_build`; never redirect it to a default-root phase path.

# Report Chart Builder

Round 26: `default_professional_chart_pack` is versioned with a canonical
requested set; caller subsets fail exact requested equality. Omitted charts are
generated or recorded with a closed skip reason. Use an unsealed report revision
attempt, fresh `report_revision_id` on chart retry, and seal only after chart
completion.

Use this skill after a run exists and the experiment report needs charts. Final
research reports need charts by default: build the Default Professional Chart
Pack automatically, save generated figures as experiment assets, and register
them before handing the final narrative to `write-research-report`.

For an explicit ad hoc chart request that is not part of final report writing,
the Agent may discuss chart requirements before plotting. Do not use that
interactive mode to delay or skip the default report chart pack.

## Version-Governed Chart Package

Before writing chart artifacts, read `current.json` and use `active_version` as
`version_id`. Read source run artifacts from:

```text
<phase_paths.09_backtests>/<run_id>/
```

For current production, stage chart assets for the supplied immutable report
revision and publish them under:

```text
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/report_assets/manifest.json
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/report_assets/figures/<figure>.png
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/report_assets/scripts/<script>.py
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/chart_build_result.json
```

Do not write root-level `report_assets/`. Do not place final report assets
outside the version's `10_reports/<run_id>/` package.

## Workflow

1. Confirm the run directory and required artifacts exist.
   - Read `metrics.json`, `equity_curve.csv`, `benchmark_curve.csv`,
     `trades.csv`, `orders.csv`, `positions.csv`, and `target_weights.csv`
     only if present.
   - scan the run directory for available data assets before recommending
     charts.
   - Do not modify metrics.
   - Do not modify audit artifacts.

2. Resolve chart mode and recommend a chart set.
   - Default Report Mode: when invoked by `oxq-report-writer-worker`,
     `write-research-report`, the coordinator's post-run auto-advance, or any
     final report task without an explicit chart list, set
     `chart_decision: default_professional_chart_pack` and build the Default Professional Chart Pack automatically.
   - Do not ask the user to confirm the default report chart batch.
   - Do not omit charts merely because the user did not request them.
   - Use the Chart Applicability Matrix below to decide which default charts are
     applicable from available artifacts.
   - List the recommended chart set sorted by rotation-strategy value and data
     availability.
   - Skip an individual default chart only when its source artifact is missing,
     empty, or structurally insufficient. Record the skipped chart and reason in
     the chart result or manifest notes; do not treat the whole report as
     chart-free.
   - For explicit ad hoc chart tasks outside report auto-advance, discuss chart
     requirements, ask what decision the chart should support, clarify chart
     type, time range, benchmark, grouping, and labels, then ask the user to
     confirm that custom batch before generating charts.
   - Explain when a useful or requested chart cannot be produced from available
     data.

3. Write plotting Python.
   - Build a small script in temporary staging whose final relative key will be
     `report_assets/scripts/<script>.py`.
   - Read run artifacts from the run directory.
   - Render figure outputs into temporary staging. Their final relative keys
     will be under `report_assets/figures/`; do not render directly into the
     governed report package.
   - Keep plotting deterministic and local; do not download new data unless the
     user explicitly asks.
   - Require `seaborn` for the default open-xquant report chart style. The
     project `chart` extra includes `seaborn>=0.13`; if import fails, treat it
     as an environment problem and fix the chart environment or block with a
     clear message; do not silently downgrade to arbitrary Matplotlib defaults.
   - In a source worktree, run the staged plotting script with
     `uv run --extra chart python <temporary_staging>/<script>.py` so the
     `chart` extra is active. In an installed SDK bundle, verify
     `import seaborn` works in the runner environment before plotting.
   - Use the open-xquant Report Chart Style below in every generated script
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
   - For return bars and heatmaps, resolve a string region such as
     `market_region = market.get("region") or spec.market.region`, then use
     `market_region == "cn"` to select red-up / green-down colors; use
     green-up / red-down outside China.

4. Publish generated assets as one batch.

Generate every figure and source script in temporary staging, construct the
complete next manifest from those exact bytes, and route the figure, script,
obsolete-file deletion, and `report_assets/manifest.json` set through
`publish_report_artifacts(report_dir, artifacts, *, lock_subject=None)`. The
mapping uses safe relative keys and complete `bytes`; `None` deletes an obsolete
asset. Use a callable builder when the manifest depends on the current package:
the callable builder executes under the final-selection lock, performs its
baseline check there, and commits an atomic all-or-rollback batch. Do not write
any target directly and do not invoke an asset CLI path that publishes files or
the manifest outside this batch.

Keep the two relative-path namespaces distinct. Manifest asset `path` values
are relative to the `report_assets/manifest.json` package, so figure values use
`figures/<name>.png`, and `source.script` values use `scripts/<name>.py`; a
manifest value is never `report_assets/figures/<name>.png`. Publisher mapping
keys remain relative to the report directory and therefore use
`report_assets/manifest.json`, `report_assets/figures/<name>.png`, and
`report_assets/scripts/<name>.py`. In `research_report.md` and rendered HTML,
the report image URL remains `report_assets/figures/<name>.png` because those
documents are also relative to the report directory.

The manifest `title` and `caption` values must use the resolved
`report_language`; the default fragment below uses Chinese because the default
report language is `中文`:

```json
{
  "schema_version": 2,
  "assets": [
    {
      "id": "equity_curve",
      "kind": "figure",
      "path": "figures/equity_curve.png",
      "title": "策略净值与基准对比",
      "caption": "来自 equity_curve.csv 和 benchmark_curve.csv；用于比较策略与基准净值走势。",
      "section": "results",
      "order": 10,
      "mime_type": "image/png",
      "sha256": "<sha256-of-exact-figure-bytes>",
      "source": {
        "script": "scripts/plot_report_charts.py",
        "script_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "input_artifacts": ["equity_curve.csv", "benchmark_curve.csv"]
      }
    },
    {
      "id": "drawdown",
      "kind": "figure",
      "path": "figures/drawdown.png",
      "title": "回撤曲线",
      "caption": "来自 equity_curve.csv；展示回撤深度与恢复过程。",
      "section": "results",
      "order": 20,
      "mime_type": "image/png",
      "sha256": "<sha256-of-exact-figure-bytes>",
      "source": {
        "script": "scripts/plot_report_charts.py",
        "script_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "input_artifacts": ["equity_curve.csv"]
      }
    },
    {
      "id": "trade_curve",
      "kind": "figure",
      "path": "figures/trade_curve.png",
      "title": "交易曲线",
      "caption": "来自 equity_curve.csv 和 trades.csv；标记展示成交记录，不代表日内路径。",
      "section": "results",
      "order": 30,
      "mime_type": "image/png",
      "sha256": "<sha256-of-exact-figure-bytes>",
      "source": {
        "script": "scripts/plot_report_charts.py",
        "script_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "input_artifacts": ["equity_curve.csv", "trades.csv"]
      }
    }
  ]
}
```

After registration, verify every generated figure:

- The image file is non-empty.
- The image dimensions are readable and positive.
- The manifest `path` is under `figures/` and resolves under the
  `report_assets/` package; the published file and report URL are under
  `report_assets/figures/` relative to the report directory.
- The figure is present in `report_assets/manifest.json`.
- The manifest hash matches the current file.
- Chart labels follow the report language when the rendered image proves the
  font output is readable; otherwise use concise English labels with local
  language captions.
- The chart is not blank or visually empty.
- The caption names the source artifact and the interpretation limit.

### Governed Publication Lock

The runtime publisher owns lock discovery and final-lock acquisition. When
`report_dir` is an export outside the governed workspace, pass
`lock_subject=source_run_dir`; this binds publication to the source run's
workspace instead of silently publishing unlocked. For an in-workspace package,
the report directory may remain the default subject. Malformed governed
configuration fails closed.

If chart construction needs a coherent run lock, wrap publication with
`run_digest_transaction(source_run_dir)` rather than taking locks manually.
The runtime protocol acquires the run lock first and the final-selection lock
second. Never pre-acquire the final lock, call a direct writer, or publish one
file at a time around `publish_report_artifacts`.

Run deterministic report QA only with a command or wrapper that checks the
split report package
`<phase_paths.10_reports>/<run_id>/` against the source run package
`<phase_paths.09_backtests>/<run_id>/`. Do not run old run-directory QA
against an empty report package. Numeric claim review is semantic/advisory;
route it through `review-research-report` or an explicitly advisory QA pass
rather than treating the CLI command as proof that all numeric claims are
sourced.

5. Hand off final writing to `write-research-report`.

Use `write-research-report` to read run artifacts, audits, robustness output,
metrics, and registered assets, then write the final Markdown report and render
HTML from that final Markdown. The expected outputs are:

- `<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.md`
- `<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.html`
- `<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/report_assets/manifest.json`
- `<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/report_assets/figures/<figure>.png`
- `<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/report_assets/scripts/<script>.py`
- `<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/chart_build_result.json`

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

Use this pack for final professional reports. Use this order unless the user explicitly requests a different order.
Build the Default Professional Chart Pack automatically in Default Report Mode.
Do not ask the user to confirm the default report chart batch. Do not omit charts merely because the user did not request them.
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

## Current Immutable Report-Revision Contract

Current report work is revision-scoped. The coordinator supplies a fresh
`report_revision_id` matching
`\Areport_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z`; the complete candidate is
published under
`<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/`. Build in
temporary staging and publish figures, scripts, the asset manifest, and
`chart_build_result.json` as one batch. The writer later publishes
`candidate_manifest.json` last and thereby seals the immutable report revision.
Create the candidate directory exclusively. An existing id is a collision;
never overwrite, delete, rename, merge, or repair it, especially when its
evidence is reachable from any prior selection.

Every schema-version-2 asset entry has exactly one safe package-relative
`source.script` under `scripts/`, a full lowercase `source.script_sha256`, and
`input_artifacts`. Reject absolute paths, `..`, symlinks, aliases, duplicate
canonical targets, unregistered scripts, and registered scripts not referenced
by an asset. Recompute the script SHA-256 from exact current bytes before
manifest publication and at every writer, review, lineage, comparison, and
selection consumption gate. A script mutation invalidates the manifest,
`chart_build_result.json`, candidate manifest, review revision, lineage,
comparison, and selection even when every PNG is unchanged.

`chart_build_result.json` is the durable writer handoff. Its
requested/applicable/generated/skipped inventory uses canonical chart order,
unique ids, closed skip reason codes, and an exact `{path, sha256}` manifest
reference. The closed skip reason codes are `missing_optional_input`,
`empty_optional_input`, `structurally_insufficient_input`, and
`not_applicable_to_strategy`. Environment, rendering, publication, or required
input failures are blockers, not skips. Validate the set invariants:
`generated.ids == applicable`, generated and skipped are disjoint, and their
union equals requested. A `complete` result has no blockers and every generated
asset hash equals both the manifest and exact figure bytes.

```json
{
  "schema_version": 1,
  "status": "complete",
  "version_id": "v001",
  "run_id": "runA",
  "report_revision_id": "report_20260712_181000",
  "chart_decision": "default_professional_chart_pack",
  "hash_algorithm": "sha256-file-bytes-v1",
  "requested": ["equity_curve", "drawdown", "cost_sensitivity"],
  "applicable": ["equity_curve", "drawdown"],
  "generated": [
    {
      "id": "equity_curve",
      "asset": {
        "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/report_assets/figures/equity_curve.png",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
      }
    },
    {
      "id": "drawdown",
      "asset": {
        "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/report_assets/figures/drawdown.png",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
      }
    }
  ],
  "skipped": [
    {
      "id": "cost_sensitivity",
      "reason_code": "missing_optional_input",
      "input_artifacts": ["robustness.json:cost_stress"]
    }
  ],
  "manifest": {
    "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/report_assets/manifest.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
  },
  "blocking_findings": []
}
```

For `candidate_scoped_historical_report_revision`, resolve the explicit
inactive version through its own manifest and require the exact current-state
guard before and after publication. Use a fresh `report_revision_id` and the
handoff's fresh `review_revision_id`; must not reactivate the inactive version,
change `current.json`, phase state, lineage state, or active run, and must not
overwrite any prior revision. A successful chart batch is only the write step
of the guarded `write -> review -> lineage -> comparison -> reselection`
workflow; prior revision bytes remain reachable.

Accept only the coordinator's closed schema-version-1 historical handoff; reject
aliases, extra keys, missing base/guard fields, and unknown reason codes.
