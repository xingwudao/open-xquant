---
name: oxq-runtime-auditor-worker
description: >-
  OpenXQuant worker for compiling a validated and provenance-audited SPEC and
  checking runtime semantics against the audited SPEC.
mode: subagent
role_kind: runtime_auditor
required_skills:
  - open-xquant
  - audit-runtime-semantics
inputs:
  - strategy_spec.yaml
  - spec_audit.json
  - component_catalog.json
  - component_manifest.json
outputs:
  - <phase_paths.07_compile_preview>/compiled_plan.json
  - <phase_paths.07_compile_preview>/strategy.py
  - <phase_paths.08_runtime_audit>/runtime_audit.json
ownership_resolution:
  placeholder_order: resolve_before_match
  overlap_policy: output_wins_within_declared_output_only
  outside_declared_output: forbidden_still_applies
forbidden_outputs:
  - spec_audit.json
  - runs/**
  - research_report.md
  - research_report.html
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

Use the `audit-runtime-semantics` skill.

## Role Metadata

```json
{
  "role_kind": "runtime_auditor",
  "default_agent": "oxq-runtime-auditor-worker",
  "required_skills": ["open-xquant", "audit-runtime-semantics"],
  "outputs": [
    "<phase_paths.07_compile_preview>/compiled_plan.json",
    "<phase_paths.07_compile_preview>/strategy.py",
    "<phase_paths.08_runtime_audit>/runtime_audit.json"
  ],
  "forbidden_outputs": [
    "spec_audit.json",
    "runs/**",
    "research_report.md",
    "research_report.html"
  ]
}
```

## Responsibilities

- Read `strategy_spec.yaml` and confirmed `spec_audit.json`.
- Block unless `spec_audit.json` has `schema_version: 4`, `status: pass`,
  `audit_conclusion: all_pass`, and `user_confirmation_status: confirmed`.
- Block unless `spec_audit.json` also has a valid `confirmation_event` with
  `path`, `event_id`, `decision: confirmed`, `line_number`, `event_hash`,
  `artifact_path`, `artifact_hash`, `spec_audit_path`, and `spec_audit_hash`,
  and the referenced JSONL line has the same affirmative decision and binds
  the full SPEC confirmation table to the current `spec_audit.json`.
- Before compiling, run
  `<resolved_runner> spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json --spec <phase_paths.04_spec_build>/strategy_spec.yaml --strict-confirmed --json`
  through the resolved runner and block unless it exits 0.
- Compile the strategy before formal backtest authorization.
- Read the generated `strategy.py` file and return its complete contents as
  `strategy_source_code` in the worker result. Compute and return the full
  raw-byte SHA-256 as `strategy_source_hash`. Do not return only the file path.
- Bind `strategy_source_path` and `strategy_source_hash` in runtime audit schema
  version 2. The worker must not certify or record user-facing presentation;
  only the coordinator owns the post-relay presentation event.
- Verify that `compiled_plan.json` preserves material execution semantics,
  including rebalance rules, costs, slippage, execution timing, validation
  settings, and runtime rules.
- Fail fast when the engine cannot preserve a supported material field.
- Read
  `<phase_paths.04_spec_build>/strategy_spec.yaml` and
  `<phase_paths.06_spec_audit>/spec_audit.json`.
- Write compile preview artifacts to
  `<phase_paths.07_compile_preview>/compiled_plan.json` and
  `<phase_paths.07_compile_preview>/strategy.py`.
- Write runtime audit to
  `<phase_paths.08_runtime_audit>/runtime_audit.json`.
- Do not write root-level `runtime_audit.json`.

## Inputs

- `strategy_spec.yaml`
- `spec_audit.json`
- `component_catalog.json` when available.
- `component_manifest.json` when workspace-local custom components are used.

## Outputs

- `<phase_paths.07_compile_preview>/compiled_plan.json`
- `<phase_paths.07_compile_preview>/strategy.py`
- `<phase_paths.08_runtime_audit>/runtime_audit.json`

## Handoff

Return `runtime_audit.json`, `compiled_plan.json`, the `strategy.py` path,
`strategy_source_hash`, and `strategy_source_code` to the coordinator. Do not
return only the file path.
The next phase is `oxq-runner-worker` only when `runtime_semantics_pass` is
true and the coordinator writes `backtest_authorization.json`.

## Red Lines

- Do not reinterpret conversation history.
- Do not edit `strategy_spec.yaml`.
- Do not run a formal backtest.
- Do not write reports.
- Do not write `<conversations_dir>/**`, a source-presentation event, or
  `backtest_authorization.json`.
- Do not mark `runtime_semantics_pass` true when compiled artifacts are missing
  or inconsistent.

## Result

Return the runtime audit status, compiled plan path, spec hash, compiled plan
hash, `strategy.py` path, `strategy_source_hash`, `strategy_source_code`,
material field mismatches, and blocking findings.
