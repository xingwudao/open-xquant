---
name: compare-experiments
description: Compare two completed open-xquant experiment runs.
---

# Experiment Comparator

Use this skill when the user asks to compare two completed runs. The comparison
is cross-experiment metadata, so do not write outputs inside either run
directory. Resolve the comparison output root from `.open-xquant/workspace.yaml`
when it exists: use `paths.comparisons_dir` for the comparison directory root
and `paths.comparison_registry` for the summary registry. Fall back to
`comparisons/` and `comparisons/comparisons.jsonl` only when no workspace config
value is present.

## Preconditions

Both runs must contain:

- `strategy_spec.yaml`
- `metrics.json`
- `equity_curve.csv`
- `execution_assumptions.json`
- `research_bias_audit.json`
- `reproducibility_audit.json`

If a run lacks `metrics.json`, say that the experiment has not completed a
backtest and cannot be compared yet.
If either run lacks audit or execution-assumption artifacts, stop and explain
that audited comparison requires those artifacts before naming winners or
writing spec-impact conclusions. If an audit contains fatal findings, do not
present that run as comparable to an audited candidate without making the audit
failure the primary result.

## Workflow

1. Read both `strategy_spec.yaml` files and generate `spec_diff.yaml`.
   - Compare fields recursively.
   - Include path, run A value, run B value, and likely impact.
2. Read `execution_assumptions.json` and both audit artifacts for each run.
   - Verify execution assumptions are comparable before interpreting metric
     differences.
   - Treat reproducibility or research-bias failures as blockers for
     winner-style conclusions.
3. Read both `metrics.json` files and generate `metrics_comparison.json`.
   - Include key metrics for each run.
   - Include deltas and winners by return, Sharpe, and drawdown where present.
4. Generate figures under `<comparisons_dir>/<comparison_id>/figures/`.
   - `equity_overlay.png`
   - `drawdown_overlay.png`
   - `metrics_bar.png`
5. Write `comparison_report.md`.
   - Explain which spec differences plausibly drove the metric differences.
   - Do not claim causality when the evidence only supports association.
6. Append a summary row to `<comparison_registry>`.

## Output Layout

With the default fallback paths:

```text
comparisons/
├── comparisons.jsonl
└── <comparison_id>/
    ├── spec_diff.yaml
    ├── metrics_comparison.json
    ├── comparison_report.md
    └── figures/
        ├── equity_overlay.png
        ├── drawdown_overlay.png
        └── metrics_bar.png
```

## Red Lines

- Do not modify either run's artifacts.
- Do not compare unaudited assumptions as if they were equivalent.
- Do not hide metric trade-offs by naming a single winner without context.
