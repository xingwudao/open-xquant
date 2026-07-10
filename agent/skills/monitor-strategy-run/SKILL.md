---
name: monitor-strategy-run
description: >-
  Audit, monitor, and record open-xquant strategy runs; use after backtests to
  run reproducibility checks, research bias checks, robustness, reports, and
  experiment logs.
---

# Strategy Monitor

You inspect completed run directories and preserve the audit trail.

## Version-Governed Run Package

The valid monitored run path is:

```text
versions/<version_id>/09_backtests/<run_id>/
```

Write or verify post-run audit outputs inside that run package, including:

```text
versions/<version_id>/09_backtests/<run_id>/reproducibility_audit.json
versions/<version_id>/09_backtests/<run_id>/research_bias_audit.json
versions/<version_id>/09_backtests/<run_id>/robustness.json
```

Append experiment registry rows with `version_id`, `run_id`, `run_path`,
`run_role`, audit status, and decision. Do not write monitor artifacts at the
workspace root.

Monitoring is not a separate version phase name in the workspace manifest.
Monitor outputs live inside `versions/<version_id>/09_backtests/<run_id>/`.
Do not set `current.json.active_phase`,
`versions/<version_id>/phase_state.json.current_phase`, or
`version_manifest.json.active_phase` to `monitor`. Keep the active phase at
`09_backtests` until the report package is written under `10_reports`.

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
uv run oxq spec-audit validate versions/<version_id>/09_backtests/<run_id>/spec_audit.json
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
uv run oxq audit reproducibility versions/<version_id>/09_backtests/<run_id>/ --json \
  > versions/<version_id>/09_backtests/<run_id>/reproducibility_audit.json
uv run oxq audit research versions/<version_id>/09_backtests/<run_id>/ --json \
  > versions/<version_id>/09_backtests/<run_id>/research_bias_audit.json
```

stdout-only audit output is not sufficient. After each command, verify that the
JSON file exists, is non-empty, and parses as a JSON object before report
handoff.

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
uv run oxq robustness run versions/<version_id>/09_backtests/<run_id>/ --json \
  > versions/<version_id>/09_backtests/<run_id>/robustness.json
```

`WARN` can mean robustness is incomplete, not that the command failed. Preserve
warnings such as missing parameter perturbation or regime analysis.
Verify that `robustness.json` exists, is non-empty, and parses as a JSON object
before handing the run to report writing.

After the command finishes, inspect `versions/<version_id>/09_backtests/` and
explicitly tell the user when a created sub-run directory such as
`<run_id>_cost_x2` appears. That `_cost_x2` directory is a parallel cost-stress
backtest and should be referenced as a robustness artifact, not mistaken for
an unrelated experiment.

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
uv run oxq experiment add versions/<version_id>/09_backtests/<run_id>/
```

Use `write-research-report` to write `research_report.md` from verified run
artifacts, then render `research_report.html` from that final Markdown. The
executive decision is research guidance, not permission to trade. Explain any
audit warnings beside the decision. Include `spec_audit.json` conclusions,
including recipe choices and unconfirmed/default assumptions, in the report
handoff.

Do not write the report directly from the artifacts inside this skill. The
report narrative must be written through `write-research-report`.

After reproducibility audit, research-bias audit, robustness, and experiment
logging pass or produce non-fatal warnings, return a phase result with
`next_phase: oxq-report-writer-worker`. Do not stop after monitoring pass. The
coordinator should immediately hand the verified run package to report writing
unless monitoring is blocked or failed.

Write or return a monitor phase result:

```json
{
  "status": "pass | blocked | fail",
  "version_id": "<version_id>",
  "run_id": "<run_id>",
  "run_dir": "versions/<version_id>/09_backtests/<run_id>",
  "reproducibility_audit": "versions/<version_id>/09_backtests/<run_id>/reproducibility_audit.json",
  "research_bias_audit": "versions/<version_id>/09_backtests/<run_id>/research_bias_audit.json",
  "robustness": "versions/<version_id>/09_backtests/<run_id>/robustness.json",
  "experiment_registry": "experiments.jsonl",
  "next_phase": "oxq-report-writer-worker",
  "blocking_reason": "",
  "errors": []
}
```

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
