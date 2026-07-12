---
name: manage-strategy-version
description: >-
  Use when an open-xquant research conversation starts, resumes, or changes
  strategy meaning and the Agent must decide whether to continue a phase,
  create a new strategy version, or append a run.
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
`current.json` to distinguish an existing-version resolution from a
new-version bootstrap. For an existing version, read
`<version_root>/<version_id>/version_manifest.json` and use its exact
`phase_paths` entry for each phase. For a new-version bootstrap, create
`<version_root>/<version_id>/` first and construct canonical `phase_paths`
from `<version_root>/<version_id>/01_brainstorm` through
`<version_root>/<version_id>/10_reports`; the new version manifest does not
exist yet and must not be treated as an input prerequisite. For example, a configured root of
`research_versions` must resolve the spec-build phase to
`research_versions/v003/04_spec_build`; never redirect it to a default-root phase path.

# Strategy Version Manager

This skill governs the strategy family -> strategy version -> run attempt
boundary. It does not build specs, audit specs, compile, run backtests, write
reports, compare experiments, or select a final version.

## Inputs

- User request or coordinator handoff.
- `workflow_manifest.json`, `current.json`, and `lineage.json` when present.
- The active `<version_root>/<version_id>/version_manifest.json` when present.
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
- A candidate-scoped historical re-review is evidence regeneration, not phase
  completion. It applies to any explicit inactive candidate whose review is
  missing or stale, including a stale current-schema review. When
  `review-research-report` re-reviews that version/run while another version is
  active, do not update `current.json`,
  `phase_state.json`, `version_manifest.json`, `lineage.json`, or active-run
  state. The reviewer owns no reactivation transaction.

Semantic change means a material change to hypothesis, universe, benchmark,
Indicator definitions, signal rules, portfolio construction, execution, cost,
validation, risk constraints, metrics, robustness, or decision policy.

## Outputs

Write only version governance artifacts:

```text
<version_root>/<version_id>/version_manifest.json
<version_root>/<version_id>/phase_state.json
lineage.json
current.json
<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json
```

`version_manifest.json` must include:

- `schema_version`
- `version_id`
- `strategy_family_id`
- `parent_version_id`
- `created_reason`
- `status`
- `active_phase`
- `source_conversation`
- canonical `phase_paths` under `<version_root>/<version_id>`

For a new version under a configured `research_versions` root, the canonical
paths are:

```text
research_versions/v003/01_brainstorm
research_versions/v003/02_idea_audit
research_versions/v003/03_component_authoring
research_versions/v003/04_spec_build
research_versions/v003/05_data_inspection
research_versions/v003/06_spec_audit
research_versions/v003/07_compile_preview
research_versions/v003/08_runtime_audit
research_versions/v003/09_backtests
research_versions/v003/10_reports
```

Create every directory named by `phase_paths` before publishing
`version_manifest.json`. Publish the new-version governance state as one
coherent bootstrap: add the version entry to `lineage.json.versions`, set
`phase_state.json.current_phase` and `version_manifest.json.active_phase` to
`01_brainstorm`, and set `current.json.active_version` to the new version with
`current.json.active_phase` set to `01_brainstorm`. The lineage entry must use
the same `version_id`, `parent_version_id`, `created_reason`, and `status` as
the manifest.

Treat a `v001 -> v002 bootstrap transaction` as one atomic governance update.
Before publishing the new active pointer, stage all four governance payloads
and verify their combined post-transaction state:

- set the prior `v001` lineage entry status to `superseded`
- set the prior `v001` version_manifest.json status to `superseded`
- set the new `v002` lineage entry and version_manifest.json status to `active`
- require exactly one `active` lineage entry, which must match
  `current.json.active_version`
- require every other lineage entry and its version manifest to be
  `superseded`

Write staged sibling temporary files, replace the prior manifest,
`lineage.json`, new `phase_state.json`, and new manifest only after every
payload validates, then publish `current.json` last. If any write or validation
fails, do not publish the new current pointer and restore the prior coherent
governance state. A bootstrap that leaves both `v001` and `v002` active, or
that omits the prior lineage entry or prior manifest update, is invalid.

`lineage.json` must record every new version and the reason it was created.
`current.json` must point to the active version, phase, and optional active run.
For phase completion updates,
`<version_root>/<version_id>/phase_state.json.current_phase`,
`<version_root>/<version_id>/version_manifest.json.active_phase`, and
`current.json.active_phase` must match the latest accepted phase. After a
passing normal active-version `report_review.json`, set the phase to
`10_reports` and set `current.json.active_run` to the reviewed run id. Do not
apply this update to a historical re-review; its guarded `current.json` must
remain byte-for-byte unchanged.

## Journaled Governance Publication

Every bootstrap and every multi-file version-governance publication is one
journaled, atomic/recoverable logical transaction. Persist its exclusive
journal at
`<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json`.
The journal records exact target paths, baseline existence and SHA-256, staged
path/hash, durable backup path/hash, replacement order, commit index, and state
`prepared -> committing -> committed`. Fsync staged files, backups, journal,
and each containing directory before entering `committing`; never treat a set
of unjournaled sibling temporaries as a transaction.

```json
{
  "schema_version": 1,
  "transaction_id": "governance-bootstrap-v002-20260712T180000Z",
  "operation": "bootstrap_version",
  "state": "prepared",
  "commit_index": 0,
  "targets": [
    {
      "path": "<version_root>/v001/version_manifest.json",
      "kind": "governance_state",
      "baseline": {
        "exists": true,
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
      },
      "staged": {
        "path": ".open-xquant/transactions/governance/staged/target-0",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
      },
      "backup": {
        "path": ".open-xquant/transactions/governance/backups/target-0",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
      }
    },
    {
      "path": "current.json",
      "kind": "commit_pointer",
      "baseline": {
        "exists": true,
        "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
      },
      "staged": {
        "path": ".open-xquant/transactions/governance/staged/target-1",
        "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
      },
      "backup": {
        "path": ".open-xquant/transactions/governance/backups/target-1",
        "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
      }
    }
  ],
  "replacement_order": [
    "<version_root>/v001/version_manifest.json",
    "current.json"
  ]
}
```

The real bootstrap lists every affected old/new manifest, phase state,
`lineage.json`, and `current.json`; the shortened example shows field shape,
not permission to omit targets. `commit_index` equals the number of completed
ordered replacements and is fsynced after each step.

Canonicalize all paths, acquire persistent `workspace-governance.lock`, then
acquire `final-selection.lock` last under the global lock order. Hold both
through baseline validation, recovery, mutation, directory fsync, post-state
validation, and journal commit. Before every replacement perform an
unchanged-byte check against the journaled baseline or the exact bytes written
by this transaction. A mismatch is concurrent mutation and must not be
overwritten.

Recovery runs before a new transaction while holding the same locks. At the
failure boundary before the first replacement, remove only disposable staging
and leave every baseline unchanged. At the failure boundary after a non-pointer
replacement but before `current.json`, roll back each replaced target from its
durable backup, restoring exact prior bytes or prior absence, and fsync parents.
At the failure boundary after `current.json` replacement, inspect every target:
roll forward and complete fsync/validation only when all exact staged bytes are
present; otherwise roll back the entire set, including `current.json`, from
durable backups. Never guess from journal state alone. Keep the journal and
backups until the committed state and all parent-directory fsyncs are durable.

`candidate_scoped_historical_report_revision` is not a governance-state
transaction. It targets an explicit inactive version with a current-state
guard, fresh `report_revision_id`, and fresh `review_revision_id`; it must not
reactivate that version and must not overwrite a prior revision. The version
manager performs no write for the guarded
`write -> review -> lineage -> comparison -> reselection` route; prior revision
bytes remain reachable.

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
