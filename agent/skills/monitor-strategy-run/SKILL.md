---
name: monitor-strategy-run
description: >-
  Audit, monitor, and record open-xquant strategy runs; use after backtests to
  run reproducibility checks, research bias checks, robustness, reports, and
  experiment logs.
---

# Strategy Monitor

You inspect completed run directories and preserve the audit trail.

## Required Run Directory

A valid CLI run should contain:

- `strategy_spec.yaml`
- `spec_hash.txt`
- `compiled_plan.json`
- `component_catalog_hash.txt`
- `recipe_catalog_hash.txt`
- `spec_audit.json`
- `conversation_hash.txt`
- `environment.json`
- `data_manifest.json`
- `artifact_hashes.json`
- `execution_assumptions.json`
- `metrics.json`
- `equity_curve.csv`
- `benchmark_curve.csv` when benchmarks are available
- `trades.csv`
- `target_weights.csv`: per-date raw and adjusted target weights, suitable for
  external baseline comparison without importing open-xquant internals.
- `positions.csv`
- `orders.csv`
- `run_log.jsonl`

If these files are missing, state that the run is not a standard audited CLI
artifact set.

Before interpreting performance, validate the semantic audit artifact shape:

```bash
uv run oxq spec-audit validate runs/<run_id>/spec_audit.json
```

This validation is deterministic schema validation only. Still read
`spec_audit.json` and preserve any blocking findings, unconfirmed defaults,
component provenance warnings, recipe selections, missing user requirements,
agent-added fields, and contradictions in the monitoring summary.

Also read `runtime_audit.json` and `compiled_plan.json` before performance
interpretation. Runtime semantics must already have been audited by
`audit-runtime-semantics`.

If `runtime_audit.json` is missing, blocked, failed, or inconsistent with the
run's `spec_hash.txt`, reject the run before report handoff. Do not recreate
the runtime audit here; route that phase back to `audit-runtime-semantics`.

## Audit

```bash
uv run oxq audit reproducibility runs/<run_id>/
uv run oxq audit research runs/<run_id>/
```

Interpretation:

- reproducibility `FAIL`: investigate hashes, environment, or data manifest
  before any performance discussion
- research audit fatal: reject the run
- research audit warning: keep the warning in the report and explain its
  impact
- spec audit `block` or `fail`: do not discuss performance as an approved
  experiment; resolve the semantic audit first
- runtime audit missing, blocked, failed, or hash-mismatched: reject the run
  even when provenance and reproducibility checks pass

Common warnings include survivorship risk, low OOS trade count, high missing
data ratio, parameter count, benchmark gaps, and drawdown tail risk.

## Robustness

```bash
uv run oxq robustness run runs/<run_id>/
```

`WARN` can mean robustness is incomplete, not that the command failed. Preserve
warnings such as missing parameter perturbation or regime analysis.

After the command finishes, inspect `runs/` and explicitly tell the user when a
created sub-run directory such as `<run_id>_cost_x2` appears. That `_cost_x2`
directory is a parallel cost-stress backtest and should be referenced as a
robustness artifact, not mistaken for an unrelated experiment.

When `robustness.json` exists, inspect and report:

- cost stress results
- IS/OOS metric diff
- parameter perturbation results
- regime segmented statistics
- any fragile, warning, or error status

Do not promote a run when robustness artifacts are missing, untracked, or fail
reproducibility checks.

## Report And Experiment Log

```bash
uv run oxq experiment add runs/<run_id>/
```

Use `write-research-report` to write `research_report.md` from verified run
artifacts, then render `research_report.html` from that final Markdown. The
executive decision is research guidance, not permission to trade. Explain any
audit warnings beside the decision. Include `spec_audit.json` conclusions,
including recipe choices and unconfirmed/default assumptions, in the report
handoff.

Do not write the report directly from the artifacts inside this skill. The
report narrative must be written through `write-research-report`.

## SDK Monitoring

Use SDK monitoring only when you have a `RunResult` object:

```python
from oxq.observe.detector import MarketStateDetector
from oxq.observe.monitor import StrategyMonitor

monitor = StrategyMonitor(result, benchmark="SPY", roll_window=63)
print(monitor.summary())

detector = MarketStateDetector(result, symbols=("SPY",))
print(detector.performance_by_state(result))
```

## Red Lines

- Do not edit artifacts to make audits pass.
- Do not summarize a failed audit as "mostly fine".
- Do not register only successful experiments while omitting failed runs.
