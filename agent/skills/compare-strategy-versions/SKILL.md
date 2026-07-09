---
name: compare-strategy-versions
description: >-
  Use when comparing open-xquant runs or strategy versions, especially when
  distinguishing within-version reproducibility from cross-version strategy
  evidence.
---

# Strategy Version Comparator

This skill extends `compare-experiments` for version-governed workspaces. It
compares completed run packages without writing into either run directory.

## Comparison Types

Within-version comparison:

- same version_id
- same confirmed SPEC unless the comparison is explicitly a robustness variant
- use `oxq backtest compare-runs` as a strict comparability gate
- failures usually block winner-style conclusions

Cross-version comparison:

- different version_id values
- different `spec_hash` is expected, not automatically fatal
- produce a spec diff and explain likely impact
- do not name a winner from metrics alone

## Inputs

- Two or more candidate run packages.
- `experiments.jsonl` with version_id, run_id, run_path, and decision fields.
- Optional existing comparison artifacts.
- Optional user-confirmed comparison scope.

## Workflow

1. Classify the comparison as `within-version` or `cross-version`.
2. Run or inspect strict comparability checks for execution, cost, validation,
   metrics profile, data, audit, and runtime assumptions.
3. For cross-version comparisons, write `spec_diff.yaml` and treat spec
   differences as evidence rather than an error.
4. Compare metrics only after audit and comparability context is explicit.
5. Write a comparison report that separates association from causality.

## Outputs

```text
comparisons/<comparison_id>/
  comparison_manifest.json
  comparability_audit.json
  spec_diff.yaml
  metrics_comparison.json
  comparison_report.md
  figures/
```

Do not leave `figures/` empty. If no figure will be generated, do not create the
directory. If the directory is created, write at least one referenced comparison
figure such as `metrics_bar.png`, `equity_overlay.png`, or
`drawdown_overlay.png`, and register it in `comparison_manifest.json`.

Append a summary row to `comparisons/comparisons.jsonl`.

## Red Lines

- Do not modify run artifacts.
- Do not hide non-comparable assumptions.
- Do not select the final version.
- Do not present an unaudited candidate as comparable to an audited candidate.

## Result

Return comparison status, comparison_id, comparable fields, blocking
differences, metric deltas, and whether the compared candidates are eligible
for `select-final-version`.
