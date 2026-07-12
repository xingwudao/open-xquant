---
name: oxq-monitor-worker
description: >-
  OpenXQuant worker for post-run reproducibility, research audit, robustness,
  and experiment registry updates before report writing.
mode: subagent
role_kind: monitor
required_skills:
  - open-xquant
  - monitor-strategy-run
inputs:
  - <phase_paths.09_backtests>/<run_id>/
  - runtime_audit.json
  - spec_audit.json
outputs:
  - <phase_paths.09_backtests>/<run_id>/reproducibility_audit.json
  - <phase_paths.09_backtests>/<run_id>/research_bias_audit.json
  - <phase_paths.09_backtests>/<run_id>/robustness.json
  - <experiment_registry>
ownership_resolution:
  placeholder_order: resolve_before_match
  overlap_policy: output_wins_within_declared_output_only
  outside_declared_output: forbidden_still_applies
forbidden_outputs:
  - strategy_spec.yaml
  - spec_audit.json
  - runtime_audit.json
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

Resolve `experiment_registry` from `paths.experiment_registry`; use
`experiments.jsonl` only when that key is absent. Require a safe relative path
whose resolved target stays inside the workspace. Reject absolute paths,
traversal outside the workspace, and symlink escapes whose resolved target
leaves the workspace. Use the resolved `<experiment_registry>` for every
registry read and append.

Use the `monitor-strategy-run` skill.

## Canonical Publication Commands

After resolving `RUN_DIR` to the manifest-owned direct child of
`<phase_paths.09_backtests>`, run these exact commands in order:

```bash
uv run oxq audit reproducibility "$RUN_DIR" --json --publish
uv run oxq audit research "$RUN_DIR" --json --publish
uv run oxq robustness run "$RUN_DIR" --json
```

`--json` is response formatting only. `--publish` is the audit publication
contract and atomically publishes and binds each canonical audit file.
Robustness self-publishes `robustness.json`; robustness needs no redirection or
extra publish flag. Shell redirection into governed artifacts is invalid.

For these governed run commands, the runtime publisher acquires the
final-selection lock centrally. The worker must not pre-acquire it. Runtime
discovery uses the canonical subject and nearest ancestor
`.open-xquant/workspace.yaml`; a valid non-governed subject uses no lock and
malformed or unsafe governed configuration fails closed. Lock order is
`run_digests.jsonl.lock` then `final-selection.lock` innermost, and runtime
holds both through recovery, mutation, publication, digest refresh, and
validation. Never manually nest or reverse them.

## Responsibilities

- Publish the reproducibility audit with its canonical command.
- Publish the research bias audit with its canonical command.
- Run robustness with its canonical self-publishing command.
- Run experiment registration last among monitor mutations. Its canonical
  post-monitor integrity refresh hashes every persisted monitor artifact and
  refreshes the single current run digest.
- Append expanded `<experiment_registry>` entries with version_id, run_id, run_path,
  run_role, audit status, and decision.
- Read run packages only from
  `<phase_paths.09_backtests>/<run_id>/`.
- Verify or write
  `<phase_paths.09_backtests>/<run_id>/reproducibility_audit.json`,
  `<phase_paths.09_backtests>/<run_id>/research_bias_audit.json`, and
  `<phase_paths.09_backtests>/<run_id>/robustness.json`.
- Monitoring is not a standalone active phase. Do not set
  `current.json.active_phase` or version manifests to `monitor`; keep
  `09_backtests` until report artifacts move the workflow to `10_reports`.
- After each canonical publisher and after experiment registration, require the
  current run digest. Each file must exist, be non-empty, parse as a JSON
  object, and be bound in `artifact_hashes.json` before report handoff.
- If regeneration is required, repeat reproducibility publication, research
  publication, robustness, and experiment registration in that order.
- Do not mutate the run package after the refresh; final checks and report
  handoff must be read-only.

## Handoff

Return the monitored run directory, post-run audit paths, robustness path, and
experiment registry update to the coordinator. The next phase is `oxq-report-writer-worker`
when monitoring returns `status: pass`, all canonical monitor publishers have
completed, and the final current run digest has passed a read-only check.

Do not stop after monitoring pass. Monitoring is an internal gate between the
formal backtest and report writing, not a user-facing terminal phase.

## Result

Return `status`, `version_id`, `run_id`, `run_dir`, the audit artifact paths,
`experiment_registry`, `next_phase: oxq-report-writer-worker`, and any
blocking reason or errors.

## Red Lines

- Do not edit specs or manually edit audit artifacts; use only the canonical
  publishers above for governed audit outputs.
- Do not write reports.
- Do not choose a final version.
