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
directory, monitor robustness, or write/review reports.

## Inputs

- `backtest_authorization.json`
- `strategy_spec.yaml`
- `spec_audit.json`
- `runtime_audit.json`
- `component_catalog.json` when provenance attachment is required
- `component_manifest.json` or manifest paths when workspace-local custom components are used
- Approved local market data directory or data manifest

## Authorization Gate

Read `backtest_authorization.json` before running any command. It should include
at least:

```json
{
  "status": "authorized",
  "strategy_spec": "strategy_spec.yaml",
  "spec_audit": "spec_audit.json",
  "runtime_audit": "runtime_audit.json",
  "component_catalog": "component_catalog.json",
  "component_manifests": ["component_manifest.json"],
  "data_dir": "data",
  "run_out": "runs/auto",
  "spec_hash": "sha256:<hash>",
  "spec_audit_hash": "sha256:<hash>",
  "runtime_audit_hash": "sha256:<hash>"
}
```

If authorization is missing, not `authorized`, or hash fields do not match the
referenced files, stop and write `runner_result.json` with `status: blocked`.
Do not repair the inputs.
When `component_manifests` is non-empty, `runtime_audit.json` must include the
same `component_bundle_hashes`; the formal run gate rejects missing or stale
bundle hashes.

## Run

Run the formal backtest with both pre-run gates:

```bash
uv run oxq backtest run strategy_spec.yaml \
  --spec-audit spec_audit.json \
  --runtime-audit runtime_audit.json \
  --component-catalog component_catalog.json \
  --component-manifest component_manifest.json \
  --data-dir data \
  --out runs/auto \
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

Then run deterministic post-run checks that do not mutate the spec, audits, or
report:

```bash
uv run oxq audit reproducibility runs/<run_id> --json
uv run oxq audit research runs/<run_id> --json
uv run oxq robustness run runs/<run_id> --json
uv run oxq experiment add runs/<run_id>
```

The post-run `audit research` and `robustness run` commands reload the
component manifests recorded in `runs/<run_id>/component_manifests.json` before
validating or compiling. If those recorded manifests are missing, invalid, or
unloadable, stop and record a failed `runner_result.json`; do not rerun the
checks without the workspace-local custom components.

If any required command fails, stop the runner phase and record the failure in
`runner_result.json`.

## Output

Write `runner_result.json`:

```json
{
  "status": "pass | blocked | fail",
  "run_dir": "runs/<run_id>",
  "spec_hash": "sha256:<hash>",
  "spec_audit_hash": "sha256:<hash>",
  "runtime_audit_hash": "sha256:<hash>",
  "artifacts": {},
  "checks": [],
  "errors": []
}
```

## Red Lines

- Do not edit `strategy_spec.yaml`.
- Do not edit `spec_audit.json`.
- Do not edit `runtime_audit.json`.
- Do not change report files.
- Do not reinterpret user intent.
- Do not continue after a failed authorization, spec audit, runtime audit, or
  reproducibility gate.
