---
name: manage-strategy-version
description: >-
  Use when an open-xquant research conversation starts, resumes, or changes
  strategy meaning and the Agent must decide whether to continue a phase,
  create a new strategy version, or append a run.
---

# Strategy Version Manager

This skill governs the strategy family -> strategy version -> run attempt
boundary. It does not build specs, audit specs, compile, run backtests, write
reports, compare experiments, or select a final version.

## Inputs

- User request or coordinator handoff.
- `workflow_manifest.json`, `current.json`, and `lineage.json` when present.
- The active `versions/<version_id>/version_manifest.json` when present.
- Current phase artifacts when the coordinator provides them.

## Decision Rules

- Continue the current version when the user is still clarifying an incomplete
  brainstorm phase.
- Create a new version when the user makes a semantic change after idea audit
  pass, after spec audit confirmation, or after seeing a report/comparison.
- Append a new run when the same confirmed SPEC is rerun, stress-tested, or
  robustness-tested without changing strategy meaning.
- Treat phase completion as a governance-only update when a worker returns a
  passing artifact for the current version. Update `current_phase`,
  `active_phase`, `completed_phases`, and `active_run` without changing
  strategy meaning or creating a new version.
- Do not create a new version for formatting, report edits, or artifact
  cleanup.

Semantic change means a material change to hypothesis, universe, benchmark,
Indicator definitions, signal rules, portfolio construction, execution, cost,
validation, risk constraints, metrics, robustness, or decision policy.

## Outputs

Write only version governance artifacts:

```text
versions/vNNN/version_manifest.json
versions/vNNN/phase_state.json
lineage.json
current.json
```

`version_manifest.json` must include:

- `version_id`
- `strategy_family_id`
- `parent_version_id`
- `created_reason`
- `status`
- active phase
- source conversation reference

`lineage.json` must record every new version and the reason it was created.
`current.json` must point to the active version, phase, and optional active run.
For phase completion updates, `versions/vNNN/phase_state.json.current_phase`,
`versions/vNNN/version_manifest.json.active_phase`, and
`current.json.active_phase` must match the latest accepted phase. After a
passing `report_review.json`, set the phase to `10_reports` and set
`current.json.active_run` to the reviewed run id.

## Red Lines

- Do not write or edit `strategy_spec.yaml`.
- Do not write audit artifacts.
- Do not run `oxq`.
- Do not write report files.
- Do not select a final version.
- Do not infer user confirmation for a candidate/default value.

## Result

Return whether the workflow should continue the current version, create a new
version, or append a run. Include the paths of updated `lineage.json`,
`current.json`, and `version_manifest.json`.
