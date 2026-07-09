---
name: govern-research-workspace
description: >-
  Use when an open-xquant research workspace looks disorganized, has root-level
  phase artifacts, or must be checked before phase handoff, comparison,
  migration, cleanup, or final selection.
---

# Research Workspace Governor

This skill audits workspace artifact governance. It checks whether the
strategy family directory follows the version-governed layout and whether
phase artifacts are in the directory owned by the role that produced them.

## Inputs

- `.open-xquant/workspace.yaml`
- `workflow_manifest.json`
- `current.json`
- `lineage.json`
- `versions/**`
- `comparisons/**`
- `final/**`

`.open-xquant/workspace.yaml` is configuration only. `current.json`,
`lineage.json`, and `experiments.jsonl` live at the workspace root. Do not probe
`.open-xquant/current.json` or other hidden-directory manifest paths when
checking active version, lineage, or experiment registry state.

## Checks

- root-level phase artifacts must be flagged as layout pollution unless
  explicitly marked as legacy staging. This includes
  `strategy_idea_brief.json`, `strategy_idea_audit.json`,
  `data_inspection_result.json`, `data_availability_report.md`,
  `strategy_spec.yaml`, `component_request.json`, `component_manifest.json`,
  `component_catalog.json`, `spec_build_notes.md`, `spec_mapping_notes.md`,
  `spec_mapping_contract.json`, `builder_phase_result.json`,
  `spec_audit.json`, `audit_notes.md`, `spec_confirmation_table.md`,
  `compile_preview/`, `runtime_audit.json`, `compiled_plan.json`,
  `backtest_authorization.json`, `runner_result.json`, `result.json`,
  `research_report.md`, `research_report.html`, `writer_result.json`,
  `report_review.json`, and root-level `report_assets/`.
- A root-level `strategy_spec.yaml` is layout pollution in a version-governed
  workspace.
- Every phase artifact must live under its version phase directory.
- Backtest run artifacts must live under
  `versions/<version_id>/09_backtests/<run_id>/`.
- In version-governed workspaces, root-level `runs/` is not required when
  `paths.default_output_dir` resolves to
  `versions/<active_version>/09_backtests`.
- `versions/<version_id>/09_backtests/run_digests.jsonl` and its lock file are
  version-local run registries; they are not root-level pollution.
- Robustness-created sibling runs such as `<run_id>_cost_x2` are not root-level
  pollution when they live under `versions/<version_id>/09_backtests/` and are
  referenced by the primary run's `robustness.json`.
- Final report artifacts must live under
  `versions/<version_id>/10_reports/<run_id>/`.
- `spec_confirmation_table.md` paths referenced by spec audit artifacts must
  exist.
- `compiled_plan.json` must be a file under `07_compile_preview/`, not a
  directory name.
- All cross-artifact references must include path, hash, and hash_type.
- `workflow_manifest.json`, `current.json`, and `lineage.json` must agree.

## Outputs

Write only governance audit artifacts:

```text
governance/workspace_audit.json
governance/workspace_audit.md
```

`workspace_audit.json` should include `status`, `blocking_findings`,
`warnings`, `layout_version`, and `next_required_phase`.

## Red Lines

- Do not repair or move files unless the coordinator explicitly asks for a
  migration step.
- Do not edit strategy specs, audits, runtime audits, runs, metrics, or reports.
- Do not mark a workspace clean when root-level phase artifact pollution remains.

## Result

Return the workspace status, blocking layout findings, stale or misplaced phase
artifacts, and the next governance action.
