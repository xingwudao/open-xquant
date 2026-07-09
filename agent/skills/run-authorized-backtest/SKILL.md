---
name: run-authorized-backtest
description: >-
  Run an authorized open-xquant backtest from gated artifacts without editing
  strategy specs, audits, runtime audits, or reports.
---

# Authorized Backtest Runner

Use this skill only after `build-strategy-spec`, `audit-strategy-spec`, and
`audit-runtime-semantics` have produced passing artifacts and the invoking system has
written `backtest_authorization.json`.

This skill is an execution worker. It does not build or edit
`strategy_spec.yaml`, produce `spec_audit.json`, produce `runtime_audit.json`,
download data unless the authorization explicitly points to an approved data
directory, run post-run audits, monitor robustness, register experiments, or
write/review reports.
It must not run `oxq registry export` or create a new
`component_catalog.json`. Use the authorized
`versions/<version_id>/04_spec_build/component_catalog.json`; if that file is
missing or stale, block the runner phase instead of regenerating catalog
artifacts in a later phase directory.

## Inputs

- `backtest_authorization.json`
- `strategy_spec.yaml`
- `spec_audit.json`
- `runtime_audit.json`
- `component_catalog.json` when provenance attachment is required
- `component_manifest.json` or manifest paths when workspace-local custom components are used
- Approved local market data directory or data manifest

## Version-Governed Run Gate

Before reading authorization or running any command, read `current.json` and
use `active_version` as `version_id`. If no active version exists, block and
return to `manage-strategy-version`.

Read gated inputs from:

```text
versions/<version_id>/04_spec_build/strategy_spec.yaml
versions/<version_id>/06_spec_audit/spec_audit.json
versions/<version_id>/08_runtime_audit/runtime_audit.json
versions/<version_id>/08_runtime_audit/backtest_authorization.json
```

Formal run outputs must be written only under:

```text
versions/<version_id>/09_backtests/<run_id>/strategy_spec.yaml
versions/<version_id>/09_backtests/<run_id>/runner_result.json
```

Do not write formal run outputs to root `runs/`.

## Authorization Gate

Read `backtest_authorization.json` before running any command. It should include
at least:

```json
{
  "status": "authorized",
  "strategy_spec": "versions/<version_id>/04_spec_build/strategy_spec.yaml",
  "spec_audit": "versions/<version_id>/06_spec_audit/spec_audit.json",
  "runtime_audit": "versions/<version_id>/08_runtime_audit/runtime_audit.json",
  "component_catalog": "versions/<version_id>/04_spec_build/component_catalog.json",
  "component_manifests": ["components/bundles/<bundle_id>/component_manifest.json"],
  "data_dir": "data",
  "run_out": "versions/<version_id>/09_backtests",
  "spec_hash": "sha256:<hash>",
  "spec_audit_hash": "sha256:<hash>",
  "runtime_audit_hash": "sha256:<hash>"
}
```

If authorization is missing, not `authorized`, or hash fields do not match the
referenced files, stop and write `runner_result.json` with `status: blocked`.
Do not repair the inputs.
The top-level fields above are required. A file that only records nested
diagnostic hashes such as `canonical_hashes.strategy_spec.yaml` is not
authorized, even if those hashes are correct.
When recomputing JSON hashes for authorization checks or diagnostics, pass
`Path` objects to `_hash_json_file`; do not pass strings.
When `component_manifests` is non-empty, `runtime_audit.json` must include the
same `component_bundle_hashes`; the formal run gate rejects missing or stale
bundle hashes.

## Run

Run the formal backtest with both pre-run gates:

```bash
uv run oxq backtest run versions/<version_id>/04_spec_build/strategy_spec.yaml \
  --spec-audit versions/<version_id>/06_spec_audit/spec_audit.json \
  --runtime-audit versions/<version_id>/08_runtime_audit/runtime_audit.json \
  --component-catalog versions/<version_id>/04_spec_build/component_catalog.json \
  --component-manifest components/bundles/<bundle_id>/component_manifest.json \
  --data-dir data \
  --out versions/<version_id>/09_backtests \
  --json
```

Omit `--component-manifest` only when the authorization explicitly contains no
workspace-local custom component manifests. Pass one `--component-manifest`
option for each authorized manifest path.

The formal run command attaches `spec_audit.json`, `runtime_audit.json`,
`conversation_hash.txt`, `component_catalog_hash.txt`, and
`recipe_catalog_hash.txt` into the run directory after rechecking the final
`strategy_spec.yaml` and `compiled_plan.json`. Use
`oxq backtest attach-provenance` only for legacy runs that predate this gate.
The legacy shorthand `--spec-audit spec_audit.json` means the active
`versions/<version_id>/06_spec_audit/spec_audit.json`; do not use it as a root
path in version-governed workspaces.
The legacy shorthand `--runtime-audit runtime_audit.json` means the active
`versions/<version_id>/08_runtime_audit/runtime_audit.json`; do not use it as a
root path in version-governed workspaces.
The legacy shorthand `--component-catalog component_catalog.json` means the
active `versions/<version_id>/04_spec_build/component_catalog.json`; do not use
it as a root path in version-governed workspaces.

Do not run `oxq audit reproducibility`, `oxq audit research`,
`oxq robustness run`, or `oxq experiment add` in this skill. Those commands
belong to `monitor-strategy-run` and `oxq-monitor-worker` after the runner
has produced a standard run package.

If the formal backtest command fails, stop the runner phase and record the
failure in `runner_result.json`.

## Output

Write `runner_result.json`:

```json
{
  "status": "pass | blocked | fail",
  "run_dir": "versions/<version_id>/09_backtests/<run_id>",
  "spec_hash": "sha256:<hash>",
  "spec_audit_hash": "sha256:<hash>",
  "runtime_audit_hash": "sha256:<hash>",
  "next_phase": "oxq-monitor-worker",
  "artifacts": {},
  "checks": [],
  "errors": []
}
```

## Red Lines

- Do not edit `strategy_spec.yaml`.
- Do not edit `spec_audit.json`.
- Do not edit `runtime_audit.json`.
- Do not run `oxq registry export`.
- Do not write `component_catalog.json` outside
  `versions/<version_id>/04_spec_build/`.
- Do not change report files.
- Do not reinterpret user intent.
- Do not continue after a failed authorization, spec audit, runtime audit, or
  formal backtest command.
- Do not run reproducibility, research-bias, robustness, experiment, or report
  commands.
