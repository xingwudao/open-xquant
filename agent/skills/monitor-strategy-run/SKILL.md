---
name: monitor-strategy-run
description: >-
  Audit, monitor, and record open-xquant strategy runs; use after backtests to
  run reproducibility checks, research bias checks, robustness, reports, and
  experiment logs.
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
Resolve `<experiment_registry>` from `paths.experiment_registry`, using
`experiments.jsonl` only when that key is absent.

# Strategy Monitor

You inspect completed run directories and preserve the audit trail.

## Version-Governed Run Package

The valid monitored run path is:

```text
<phase_paths.09_backtests>/<run_id>/
```

Write or verify post-run audit outputs inside that run package, including:

```text
<phase_paths.09_backtests>/<run_id>/reproducibility_audit.json
<phase_paths.09_backtests>/<run_id>/research_bias_audit.json
<phase_paths.09_backtests>/<run_id>/robustness.json
```

Append experiment registry rows with `version_id`, `run_id`, `run_path`,
`run_role`, audit status, and decision. Do not write monitor artifacts at the
workspace root.

Monitoring is not a separate version phase name in the workspace manifest.
Monitor outputs live inside `<phase_paths.09_backtests>/<run_id>/`.
Do not set `current.json.active_phase`,
`<version_root>/<version_id>/phase_state.json.current_phase`, or
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

The `<phase_paths.06_spec_audit>/spec_audit.json` is the canonical SPEC audit.
The `spec_audit.json` inside the run directory is an attached provenance copy,
not the authoritative phase artifact. Before interpreting performance, resolve
the exact manifest-owned phase-06 path and validate the semantic audit artifact
shape there:

```bash
uv run oxq spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json
```

This validation is deterministic schema validation only. Still read
the canonical `spec_audit.json` and preserve any blocking findings, unconfirmed defaults,
component provenance warnings, recipe selections, missing user requirements,
agent-added fields, and contradictions in the monitoring summary.

Verify that `<phase_paths.09_backtests>/<run_id>/spec_audit.json` exists as a
regular, non-symlink file and is an attached copy of the canonical audit:
compare its bytes and its SHA-256 hash with
`<phase_paths.06_spec_audit>/spec_audit.json`. Verify any corresponding
`artifact_hashes.json` entry also names the run copy and matches that hash.
Reject the run when the attached copy or its recorded hash is missing or
mismatched. Read and interpret the phase-06 artifact; use the run copy only to
verify attached provenance.

Also read `runtime_audit.json` and `compiled_plan.json` before performance
interpretation. Runtime semantics must already have been audited by
`audit-runtime-semantics`.

If `runtime_audit.json` is missing, blocked, failed, or inconsistent with the
run's `spec_hash.txt`, reject the run before report handoff. Do not recreate
the runtime audit here; route that phase back to `audit-runtime-semantics`.

## Audit

Resolve the governed run directory once and use these exact commands:

```bash
RUN_DIR="<phase_paths.09_backtests>/<run_id>"
uv run oxq audit reproducibility "$RUN_DIR" --json --publish
uv run oxq audit research "$RUN_DIR" --json --publish
```

`--json` is response formatting only. It controls the CLI response and does not
publish a governed artifact. `--publish` is the audit publication contract: it
atomically publishes and binds the command's canonical audit file in `RUN_DIR`
and refreshes the current run digest. Shell redirection into governed artifacts
is invalid because it bypasses that publication contract and can leave
`artifact_hashes.json` or `run_digests.jsonl` stale.

After each command, verify that the canonical JSON file exists, is non-empty,
parses as a JSON object, is bound in `artifact_hashes.json`, and leaves
`require_current_run_digest(RUN_DIR)` current before continuing.

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
uv run oxq robustness run "$RUN_DIR" --json
```

Here too, `--json` is response formatting. Robustness self-publishes and binds
`robustness.json`; robustness needs no redirection or extra publish flag.
Shell redirection into governed artifacts is invalid.

For every canonical governed run mutation, the runtime publisher acquires the
final-selection lock centrally. The Agent must not pre-acquire it around the
CLI command or API. Runtime discovery canonicalizes the subject, finds the
nearest ancestor `.open-xquant/workspace.yaml`, and treats a valid non-governed
subject as no-lock while malformed or unsafe governed configuration fails
closed. Runtime lock order is `run_digests.jsonl.lock` followed by
`final-selection.lock` innermost, with both held through recovery, mutation,
publication, digest refresh, and validation. The Agent must not manually nest
or reverse those locks.

`WARN` can mean robustness is incomplete, not that the command failed. Preserve
warnings such as missing parameter perturbation or regime analysis.
Verify that `robustness.json` exists, is non-empty, and parses as a JSON object
and that `require_current_run_digest(RUN_DIR)` remains current before handing
the run to report writing.

After the command finishes, inspect `<phase_paths.09_backtests>/` and
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

## Integrity Refresh Ordering

Apply this monitor mutation order exactly:

1. Publish `reproducibility_audit.json` with the exact reproducibility audit
   command and `--publish`.
2. Publish `research_bias_audit.json` with the exact research audit command and
   `--publish`.
3. Run robustness, which self-publishes `robustness.json`.
4. Run `oxq experiment add` as the final run-package mutation.

Experiment registration validates `metrics.json`, `strategy_spec.yaml`, and
`spec_hash.txt` identity before it rewrites the research-bias audit or appends
the registry. It then performs the canonical post-monitor integrity refresh.
The current hashes for `reproducibility_audit.json`,
`research_bias_audit.json`, and `robustness.json` are written to
`artifact_hashes.json`, followed by exactly one current entry in
`<phase_paths.09_backtests>/run_digests.jsonl` for the monitored run.

If any monitored input or output must be regenerated, repeat steps 1 through 4
in this order. Never regenerate a governed monitor artifact by redirecting
stdout. Require the current run digest after each of the three canonical
publishers and again after experiment registration.

Do not write another run artifact after this refresh. Final verification and
report handoff must be read-only; otherwise the artifact manifest and run
digest become stale and the refresh must be repeated before report QA.

## Report And Experiment Log

```bash
uv run oxq experiment add <phase_paths.09_backtests>/<run_id>/
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
unless monitoring is blocked or failed. Handoff only after the canonical
post-monitor integrity refresh has completed and
`require_current_run_digest(RUN_DIR)` confirms the final registered state is
current.

Write or return a monitor phase result:

```json
{
  "status": "pass | blocked | fail",
  "version_id": "<version_id>",
  "run_id": "<run_id>",
  "run_dir": "<phase_paths.09_backtests>/<run_id>",
  "reproducibility_audit": "<phase_paths.09_backtests>/<run_id>/reproducibility_audit.json",
  "research_bias_audit": "<phase_paths.09_backtests>/<run_id>/research_bias_audit.json",
  "robustness": "<phase_paths.09_backtests>/<run_id>/robustness.json",
  "experiment_registry": "<experiment_registry>",
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
