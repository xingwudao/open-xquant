---
name: strategy-monitor
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
- `environment.json`
- `data_manifest.json`
- `artifact_hashes.json`
- `execution_assumptions.json`
- `metrics.json`
- `equity_curve.csv`
- `benchmark_curve.csv` when benchmarks are available
- `trades.csv`
- `positions.csv`
- `orders.csv`
- `run_log.jsonl`

If these files are missing, state that the run is not a standard audited CLI
artifact set.

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

Common warnings include survivorship risk, low OOS trade count, high missing
data ratio, parameter count, benchmark gaps, and drawdown tail risk.

## Robustness

```bash
uv run oxq robustness run runs/<run_id>/
```

`WARN` can mean robustness is incomplete, not that the command failed. Preserve
warnings such as missing parameter perturbation or regime analysis.

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
uv run oxq report write runs/<run_id>/
uv run oxq experiment add runs/<run_id>/
```

The report's executive decision is a framework output, not permission to trade.
Explain any audit warnings beside the decision.

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
