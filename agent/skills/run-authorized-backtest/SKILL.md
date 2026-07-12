---
name: run-authorized-backtest
description: >-
  Run an authorized open-xquant backtest from gated artifacts without editing
  strategy specs, audits, runtime audits, or reports.
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

Resolve `components_dir` independently from `paths.components_dir`; use
`components` only when that key is absent. Require a safe relative path whose
resolved target stays inside the workspace. Use the resolved `<components_dir>`
for every workspace-local component bundle path.

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
`<phase_paths.04_spec_build>/component_catalog.json`; if that file is
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
<phase_paths.04_spec_build>/strategy_spec.yaml
<phase_paths.06_spec_audit>/spec_audit.json
<phase_paths.08_runtime_audit>/runtime_audit.json
<phase_paths.08_runtime_audit>/backtest_authorization.json
```

Formal run outputs must be written only under:

```text
<phase_paths.09_backtests>/<run_id>/strategy_spec.yaml
<phase_paths.09_backtests>/<run_id>/runner_result.json
```

Do not write formal run outputs to root `runs/`.

## Authorization Gate

Read `backtest_authorization.json` before running any command. It should include
at least:

```json
{
  "status": "authorized",
  "strategy_spec": "<phase_paths.04_spec_build>/strategy_spec.yaml",
  "spec_audit": "<phase_paths.06_spec_audit>/spec_audit.json",
  "runtime_audit": "<phase_paths.08_runtime_audit>/runtime_audit.json",
  "component_catalog": "<phase_paths.04_spec_build>/component_catalog.json",
  "component_manifests": ["<components_dir>/bundles/<bundle_id>/component_manifest.json"],
  "data_dir": "data",
  "run_out": "<phase_paths.09_backtests>",
  "spec_hash": "sha256:<hash>",
  "spec_audit_hash": "sha256:<hash>",
  "runtime_audit_hash": "sha256:<hash>",
  "strategy_source_presentation": {
    "path": "<conversations_dir>/<conversation_id>/runtime-source-presentations.jsonl",
    "line_number": 1,
    "event_hash": "sha256:<full raw JSONL line hash>",
    "event_id": "<event id>",
    "phase": "runtime_source_presentation",
    "presentation": "complete_strategy_source",
    "presented_by_role": "coordinator",
    "strategy_source_path": "<phase_paths.07_compile_preview>/strategy.py",
    "strategy_source_hash": "sha256:<full raw strategy.py hash>",
    "runtime_audit_path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
    "runtime_audit_hash": "sha256:<canonical runtime audit hash>",
    "compiled_plan_hash": "sha256:<canonical compiled plan hash>",
    "version_id": "<active version>",
    "active_run": null,
    "run_out": "<phase_paths.09_backtests>"
  }
}
```

If authorization is missing, not `authorized`, or hash fields do not match the
referenced files, stop and write `runner_result.json` with `status: blocked`.
Do not repair the inputs.
The top-level fields above are required. A file that only records nested
diagnostic hashes such as `canonical_hashes.strategy_spec.yaml` is not
authorized, even if those hashes are correct.
Require `hash_type` only on structured references whose schema defines that
field. The authorization's required scalar `spec_hash`, `spec_audit_hash`, and
`runtime_audit_hash` fields encode the algorithm as `sha256:<hex>` and do not
require sibling hash-type fields.
Before running, read the referenced `spec_audit.json` and block unless it has
`status: pass`, `audit_conclusion: all_pass`,
`user_confirmation_status: confirmed`, and a valid `confirmation_event`
reference. The `confirmation_event` must include `path`, `event_id`,
`decision: confirmed`, `line_number`, `event_hash`, `artifact_path`,
`artifact_hash`, `spec_audit_path`, and `spec_audit_hash`; the referenced JSONL
line must carry the same affirmative decision and bind the same full SPEC table
and pre-confirmation `spec_audit.json` hash. A
pending all-pass audit is not authorized for a formal run.
When recomputing JSON hashes for authorization checks or diagnostics, pass
`Path` objects to `_hash_json_file`; do not pass strings.
When `component_manifests` is non-empty, `runtime_audit.json` must include the
same `component_bundle_hashes`; the formal run gate rejects missing or stale
bundle hashes.

The runner must validate `strategy_source_presentation` before component
imports or formal output. Resolve its JSONL `path` only inside configured
`paths.conversations_dir`, hash the referenced raw line, and require the line
to record `presented_by_role: coordinator` and
`presentation: complete_strategy_source`. Recompute the full source-file SHA-256
and bind it to runtime audit schema version 2
`strategy_source_path`/`strategy_source_hash`, the canonical runtime audit
hash, `compiled_plan_hash`, `current.json.active_version`, the current
`active_run` when present, and exact `<phase_paths.09_backtests>` run output.
A runtime-worker boolean, path-only handoff, stale event, mismatched event, or
event outside `<conversations_dir>` does not authorize execution.

## Runner Resolution

In a research workspace, do not assume `uv run oxq` is installed locally.
Before running the formal backtest command:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq` or bare `oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

## Run

Run the formal backtest with both pre-run gates:

```bash
<resolved_runner> backtest run <phase_paths.04_spec_build>/strategy_spec.yaml \
  --spec-audit <phase_paths.06_spec_audit>/spec_audit.json \
  --runtime-audit <phase_paths.08_runtime_audit>/runtime_audit.json \
  --component-catalog <phase_paths.04_spec_build>/component_catalog.json \
  --component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json \
  --data-dir data \
  --out <phase_paths.09_backtests> \
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
`<phase_paths.06_spec_audit>/spec_audit.json`; do not use it as a root
path in version-governed workspaces.
The legacy shorthand `--runtime-audit runtime_audit.json` means the active
`<phase_paths.08_runtime_audit>/runtime_audit.json`; do not use it as a
root path in version-governed workspaces.
The legacy shorthand `--component-catalog component_catalog.json` means the
active `<phase_paths.04_spec_build>/component_catalog.json`; do not use
it as a root path in version-governed workspaces.

For governed run publication, the runtime publisher acquires the
final-selection lock centrally. The Agent must not pre-acquire it around the
formal command. Runtime discovery canonicalizes the run subject, uses the
nearest ancestor `.open-xquant/workspace.yaml`, returns no lock for a valid
non-governed workspace, and fails closed for malformed or unsafe governed
configuration. Runtime lock order is `run_digests.jsonl.lock` followed by
`final-selection.lock` innermost, held together through recovery, mutation,
publication, digest refresh, and validation.

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
  "run_dir": "<phase_paths.09_backtests>/<run_id>",
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
  `<phase_paths.04_spec_build>/`.
- Do not change report files.
- Do not reinterpret user intent.
- Do not continue after a failed authorization, spec audit, runtime audit, or
  formal backtest command.
- Do not run reproducibility, research-bias, robustness, experiment, or report
  commands.
