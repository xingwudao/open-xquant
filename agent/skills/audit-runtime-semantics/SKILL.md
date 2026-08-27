---
name: audit-runtime-semantics
description: >-
  Compile an open-xquant strategy spec preview and audit that material SPEC
  execution semantics are preserved in compiled_plan.json before backtests.
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

Resolve `conversations_dir` independently from `paths.conversations_dir`; use
`conversations` only when that key is absent. Require a safe relative path
whose resolved target stays inside the workspace. Reject absolute paths,
traversal outside the workspace, and symlink escapes whose resolved target
leaves the workspace. Use the resolved `<conversations_dir>` for every
confirmation-event read.

# Runtime Auditor

Use this skill after a confirmed `spec_audit.json` from `audit-strategy-spec`
and before any formal `oxq backtest run`. Its job is deterministic artifact
consistency:
`strategy_spec.yaml` must compile into a `compiled_plan.json` that preserves
material execution semantics.

This skill does not audit conversation provenance, ask the user to confirm
assumptions, edit `strategy_spec.yaml`, run a backtest, monitor a run, or write
a report.

## Inputs

- `strategy_spec.yaml`
- `spec_audit.json`
- `component_catalog.json` when available
- `backtest_authorization.json` or the intended formal run inputs when
  available, including `data_dir` and `component_manifest` paths
- Optional prior `<phase_paths.07_compile_preview>/compiled_plan.json`

## Version-Governed Runtime Gate

Before compiling, read `current.json` and use `active_version` as
`version_id`. If no active version exists, block and return to
`manage-strategy-version`.

Read the confirmed SPEC package from:

```text
<phase_paths.04_spec_build>/strategy_spec.yaml
<phase_paths.06_spec_audit>/spec_audit.json
<phase_paths.04_spec_build>/component_catalog.json
```

Write compile preview artifacts only under:

```text
<phase_paths.07_compile_preview>/compiled_plan.json
<phase_paths.07_compile_preview>/strategy.py
<phase_paths.07_compile_preview>/spec_hash.txt
```

Write the runtime audit only to:

```text
<phase_paths.08_runtime_audit>/runtime_audit.json
```

Do not write root-level `compile_preview/`, `compiled_plan.json`, or
`runtime_audit.json`.

## Confirmed Spec Audit Gate

Before compiling, read `spec_audit.json` and block unless all of these are
true:

- `status: pass`
- `schema_version: 4`
- `spec_provenance_pass: true`
- `audit_conclusion: all_pass`
- `user_confirmation_status: confirmed`
- `confirmation_event` exists, points to
  `<conversations_dir>/<conversation_id>/confirmations.jsonl`, records the full
  SPEC table path/hash, and records `spec_audit_path` plus `spec_audit_hash`
  for the pre-confirmation `spec_audit.json`. The JSONL line must match the
  audit reference for `path`, `event_id`, `decision: confirmed`, `line_number`,
  `artifact_path`, `artifact_hash`, `spec_audit_path`, and `spec_audit_hash`.
  A rejected or missing machine-readable decision blocks strict confirmation.

`confirmation_event.event_hash` is the SHA-256 hash of the raw JSONL line
content referenced by `path` and `line_number`. Do not compare it with a nested
`event_hash` field inside the JSONL payload if one exists; that payload field
is not part of the formal gate. Likewise, `confirmation_event.spec_audit_hash`
is the canonical pre-confirmation audit hash, not the current final
post-confirmation `spec_audit.json` hash. The deterministic
`spec-audit validate --strict-confirmed --json` command is the source of truth
for confirmation-event binding.

Before compiling, run deterministic validation and block unless it exits 0:

```bash
<resolved_runner> spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json \
  --spec <phase_paths.04_spec_build>/strategy_spec.yaml \
  --strict-confirmed \
  --json
```

An `audit_conclusion: all_pass` audit with
`user_confirmation_status: pending` is still blocked. Do not treat it as
authorization for runtime audit, compile preview, or backtest work.

## Runner Resolution

In a research workspace, do not assume `uv run oxq` is installed locally.
Before running deterministic validation or compile commands:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq` or bare `oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

## Compile Preview

If no current compile preview exists for the current `spec_hash`, run:

```bash
<resolved_runner> strategy compile <phase_paths.04_spec_build>/strategy_spec.yaml \
  --data-dir data \
  --component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json \
  --out <phase_paths.07_compile_preview>
```

Use the same `data_dir` and every `component_manifest` path that the formal
`oxq backtest run` will use. Omit `--data-dir` only when the formal run will
also omit it. Omit `--component-manifest` when no workspace-local custom
components are authorized; repeat it for each authorized bundle manifest. The
preview `compiled_plan.json` includes the resolved effective `data_dir`, so a
preview made with different run inputs is not a valid runtime gate.
When component manifests are used, record their authorized `bundle_hash` values
in `runtime_audit.json` as `component_bundle_hashes`.

Read `<phase_paths.07_compile_preview>/compiled_plan.json` and
`<phase_paths.07_compile_preview>/spec_hash.txt`. The `spec_hash` in
the compile preview must match `strategy_spec.yaml` and `spec_audit.json`.

Read the compiled plan's `open_xquant_version` and compare it with
`strategy_spec.yaml` field `required_oxq_version`. When a prior or preview
runtime artifact includes `environment.json`, use it as additional evidence for
the same version comparison. A compile preview produced by a different
open-xquant package version than the SPEC requires is not a valid runtime gate.

The compile preview must also include
`<phase_paths.07_compile_preview>/strategy.py`. Read the complete generated file
and compute `strategy_source_hash` as the full SHA-256 of its raw bytes. Record
the path and hash in `runtime_audit.json` and return the path, hash, and complete
source text to the coordinator. Use a handoff field named
`strategy_source_code`; a path-only handoff is not enough.

The runtime auditor owns generated-source identity, not user-facing
presentation. It must not write source-presentation evidence, set a
presentation boolean, or claim that the coordinator relayed the source. The
coordinator displays the exact `strategy_source_code` in a fenced `python`
block to print the complete `strategy.py` source and records durable
presentation evidence only after that relay.

## Material Runtime Fields

Compare material SPEC fields against `compiled_plan.json`. At minimum audit:

- rebalance interval and runtime rebalance source
- calendar rebalance frequency and schedule
- execution timing and fill price mode
- fee, side-specific fees, sell-side tax, minimum fee, and slippage
- universe type, index metadata, point-in-time policy, and constituent symbols
- data filters, `suspension_policy`, and required filter columns
- indicator component classes, params, and `lag_bars`
- cross-sectional indicator preservation, including `RPS` when present
- portfolio optimizer params that affect allocation, including
  `TopNRanking.pre_filter_signal`, `weighting`, and `ascending`
- validation train/test periods and `required_oos`
- rule components that affect orders or positions
- portfolio constraints that affect target weights
- initial cash, cash return, benchmark, and metrics profile when represented in
  runtime artifacts
- `required_oxq_version` against the open-xquant `open_xquant_version` recorded
  by compile/runtime artifacts such as `compiled_plan.json` and
  `environment.json` when available

If the compiled plan omits a supported material field, or contradicts the
SPEC, block the backtest. For example, if `strategy_spec.yaml` says
`portfolio.rules.rebalance.params.interval_days: 10` but
`compiled_plan.json` shows daily or `interval_days: 1`, this is a blocking
runtime mismatch.

If the engine cannot represent a material field in `compiled_plan.json`, block
and report that the runtime artifact is insufficient for formal execution. Do
not infer success from a missing field.

The formal backtest gate requires exact once-only audit coverage for these
material SPEC sections and compiled-plan paths:

- `market` -> `market`
- `universe` -> `universe`
- `data` -> `data`
- `signal` -> `signals`
- `portfolio` -> `portfolio`
- `execution` -> `execution`
- `cost` -> `cost`
- `benchmark` -> `benchmark`
- `validation` -> `validation`
- `metrics` -> `metrics`

For every row, copy the full effective SPEC section into `spec_value` and the
full compiled-plan section into `runtime_value`; use `status: preserved` and
`blocking: false` only after the semantic comparison passes. Empty coverage,
missing sections, duplicate `field_path` rows, incorrect path pairs, and stale
or mismatched values cannot authorize a formal backtest. Standalone schema
validation remains a shape check; formal execution applies this stricter
SPEC-and-compiled-plan coverage gate.

## `runtime_audit.json`

Write `runtime_audit.json` before authorizing a formal backtest:

Hash fields must use the same semantic/canonical hash boundaries as the
formal backtest gate:

- `spec_hash`: `StrategySpec.from_yaml(...).compute_hash()` or the `Spec Hash`
  printed by `oxq spec validate`. Do not use raw `shasum strategy_spec.yaml`.
- `spec_audit_hash`: canonical JSON hash of `spec_audit.json`, matching
  open-xquant's `_hash_json_file` helper. Do not use raw file bytes.
- `compiled_plan_hash`: canonical JSON hash of the compiled plan payload that
  will be used by the formal run, matching open-xquant's `_hash_json_payload`
  or `_hash_json_file` helper. Do not use raw file bytes.

When computing hashes from Python, use the installed open-xquant helpers exactly:

```python
import hashlib
from pathlib import Path

from oxq.spec.schema import StrategySpec
from oxq.spec.compiler import _hash_json_file

spec_path = Path("<phase_paths.04_spec_build>/strategy_spec.yaml")
spec_audit_path = Path("<phase_paths.06_spec_audit>/spec_audit.json")
compiled_plan_path = Path("<phase_paths.07_compile_preview>/compiled_plan.json")

spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
spec_audit_hash = _hash_json_file(spec_audit_path)
compiled_plan_hash = _hash_json_file(compiled_plan_path)
strategy_source_path = Path("<phase_paths.07_compile_preview>/strategy.py")
strategy_source_hash = f"sha256:{hashlib.sha256(strategy_source_path.read_bytes()).hexdigest()}"
```

Do not import non-existent helpers from `oxq.core.hashing`.

```json
{
  "schema_version": 2,
  "status": "pass | block | fail",
  "runtime_semantics_pass": true,
  "strategy_source_path": "<phase_paths.07_compile_preview>/strategy.py",
  "strategy_source_hash": "sha256:<full raw strategy.py hash>",
  "spec_hash": "sha256:<StrategySpec.compute_hash()>",
  "spec_audit_hash": "sha256:<canonical spec_audit JSON hash>",
  "compiled_plan_hash": "sha256:<canonical compiled_plan JSON hash>",
  "component_bundle_hashes": ["sha256:<hash>"],
  "compiled_plan_path": "<phase_paths.07_compile_preview>/compiled_plan.json",
  "material_field_audits": [
    {
      "field_path": "portfolio.rules.rebalance",
      "spec_value": {
        "type": "RebalanceFrequencyRule",
        "params": {"interval_days": 10}
      },
      "runtime_path": "execution.rebalance",
      "runtime_value": {
        "frequency": "every_n_sessions",
        "interval_days": 10,
        "source": "portfolio.rules.rebalance"
      },
      "status": "preserved | missing | mismatch | not_applicable",
      "evidence": ["compiled_plan.json preserves interval_days: 10"],
      "blocking": false
    }
  ],
  "blocking_findings": [{"message": "..."}]
}
```

After writing `runtime_audit.json`, run:

```bash
<resolved_runner> runtime-audit validate <phase_paths.08_runtime_audit>/runtime_audit.json
```

Schema validation only proves artifact shape. The comparison still belongs to
this skill.

## Output

Report a compact summary:

- compile preview path
- path to `<phase_paths.07_compile_preview>/strategy.py`
- full raw-byte `strategy_source_hash`
- `strategy_source_code` containing the complete source text when returning
  through a worker or subagent handoff
- `spec_hash`
- material fields preserved
- material fields missing or mismatched
- path to `<phase_paths.08_runtime_audit>/runtime_audit.json`
- whether formal backtest remains blocked

Return both the `strategy.py` path and the complete source text. The upstream
coordinator must be able to show the user the source code itself, not only a
file location.

Do not run a formal backtest while `runtime_audit.json` is missing, blocked,
failed, or mismatched.
