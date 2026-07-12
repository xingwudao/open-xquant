---
name: govern-research-workspace
description: >-
  Use when an open-xquant research workspace looks disorganized, has root-level
  phase artifacts, or must be checked before phase handoff, comparison,
  migration, cleanup, or final selection.
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

## Workspace Path Resolution

Before using any `<phase_paths.*>` or `<version_root>` placeholder, read
`.open-xquant/workspace.yaml`. Resolve `version_root` from
`paths.versions_dir`; use `versions` only when that key is absent. Require a
safe relative path whose resolved target stays inside the workspace. Then read
`<version_root>/<version_id>/version_manifest.json` and use its exact
`phase_paths` entry for each phase. For example, a configured root of
`research_versions` must resolve the spec-build phase to
`research_versions/v003/04_spec_build`; never redirect it to a default-root phase path.

Resolve each key independently from `paths`: use `paths.comparisons_dir`,
`paths.final_dir`, and `paths.governance_dir`. Use `comparisons`, `final`, or
`governance`, respectively, only when that key is absent; never derive one key
from another configured key. Each configured value must be a safe relative path
whose resolved target stays inside the workspace. Reject absolute paths,
traversal outside the workspace, and symlink escapes whose resolved target
leaves the workspace. Call the resolved paths `<comparisons_dir>`, `<final_dir>`,
and `<governance_dir>` below.

Resolve role placeholders before comparing owned outputs with forbidden
outputs. A declared output wins an overlap only for that exact file or declared
output subtree; the broader forbidden rule still applies to its parent,
siblings, and every path outside that output. Therefore a safe configured root
such as `paths.final_dir: runs/final` is valid for the final selector despite a
broader `runs/**` prohibition, but no sibling under `runs/` becomes writable.

## Historical Version Audit

This governor explicitly owns cross-version inspection. Before comparison or
migration, build the complete version inventory from the governed lineage and
selection scope plus direct version-root children as follows:

1. Enumerate version ids from `lineage.json` and every selection artifact in
   the governed `<final_dir>` scope, including candidate sets, final decisions,
   and `current_final.json`.
2. Enumerate every direct child directory of the canonical `<version_root>`
   with `lstat`; do not recurse to discover version roots and do not follow a
   symlink before recording and validating that direct child.
3. Record all raw source inventories, deduplicate ids, validate every id and
   child basename against `^v[0-9]{3,}$`, and construct
   `target_version_ids` as the set union of lineage ids, selection ids, and
   validly named direct-child directory ids. Sort `target_version_ids`
   lexicographically before inspection and sort every emitted source/id list.

Never omit a direct child because lineage or selection artifacts do not
reference it. A valid direct child with a valid matching manifest but absent
from both reference sources is `unreferenced_version` and remains a full audit
target. A validly named `v004` direct child with a missing or invalid manifest
is `orphaned_version`; report it and block rather than skipping its phases. A
`draft` direct child that does not match `^v[0-9]{3,}$` is
`invalid_version_directory_name`; inventory and report it but do not treat its
contents as governed phases. A referenced id without a direct child is
`missing_version_directory`. For example, a `v003` direct child with a valid
matching manifest and no lineage/selection reference is an unreferenced
version, not an invisible directory.

For every target version:

1. Set `target_version_id` from the enumerated ID and resolve
   `<version_root>/<target_version_id>/` canonically. Require that target
   version directory to remain inside the canonical version root and workspace.
2. Load only
   `<version_root>/<target_version_id>/version_manifest.json` for that target.
   The manifest `version_id` must equal `target_version_id`; reject a missing or
   mismatched manifest rather than substituting the active version's manifest.
3. Load that manifest's exact `phase_paths`. Validate every inspected phase path
   as a non-empty, safe workspace-relative path: reject an absolute path or any
   `..` path segment, resolve it canonically with existing symlink ancestors,
   and require it to be the target version directory or a descendant of that
   directory. Reject cross-version paths and symlink escapes, including paths
   that remain in the workspace but resolve under a different version.
4. Use only those per-version resolved phase paths for the audit. Do not fall
   back to the active manifest, a default phase path, or a phase path resolved
   for another target version.

Historical inspection is read-only. The only permitted writes are the
configured `<governance_dir>/workspace_audit.json` and
`<governance_dir>/workspace_audit.md` outputs.

`invalid_version_directory_name`, `orphaned_version`,
`unreferenced_version`, and `missing_version_directory` are blocking governance
findings. Do not return a clean status while any remains.

# Research Workspace Governor

This skill audits workspace artifact governance. It checks whether the
strategy family directory follows the version-governed layout and whether
phase artifacts are in the directory owned by the role that produced them.

## Inputs

- `.open-xquant/workspace.yaml`
- `workflow_manifest.json`
- `current.json`
- `lineage.json`
- `<version_root>/**`
- `<phase_paths.09_backtests>/**`
- `<comparisons_dir>/**`
- `<final_dir>/**`

`.open-xquant/workspace.yaml` is configuration only. `current.json` and
`lineage.json` live at the workspace root. Resolve the experiment registry from
`paths.experiment_registry`, defaulting to `experiments.jsonl` only when absent.
Do not probe `.open-xquant/current.json` or other hidden-directory manifest
paths when checking active version, lineage, or experiment registry state.

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
  `<phase_paths.09_backtests>/<run_id>/`.
- In version-governed workspaces, root-level `runs/` is not required when
  `workflow.default_output_dir` resolves to `<phase_paths.09_backtests>`.
- `<phase_paths.09_backtests>/run_digests.jsonl` and its lock file are
  version-local run registries; they are not root-level pollution.
- Robustness-created sibling runs such as `<run_id>_cost_x2` are not root-level
  pollution when they live under `<phase_paths.09_backtests>/` and are
  referenced by the primary run's `robustness.json`.
- Final report artifacts must live under
  `<phase_paths.10_reports>/<run_id>/`.
- `spec_confirmation_table.md` paths referenced by spec audit artifacts must
  exist.
- `compiled_plan.json` must be a file under `07_compile_preview/`, not a
  directory name.
- All cross-artifact references must include the path/hash fields required by
  their producer schema. Require `hash_type` only on structured references whose schema defines that field.
  Scalar `*_hash`, `artifact_hash`, and
  `event_hash` fields use their required `sha256:<hex>` prefix and do not own a
  sibling `hash_type`.
- `workflow_manifest.json`, `current.json`, and `lineage.json` must agree.

## Outputs

Write only governance audit artifacts:

```text
<governance_dir>/workspace_audit.json
<governance_dir>/workspace_audit.md
```

The shared transaction publisher persists its mandatory journal at
`<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json`.
That journal is internal recovery state, not a third governance-audit handoff
artifact, and may be created or changed only by the journaled publisher defined
below.

`workspace_audit.json` uses schema version 2 and exactly this top-level and
inventory-entry field inventory. The example intentionally covers direct-child
orphan and unreferenced cases:

```json
{
  "schema_version": 2,
  "status": "blocked",
  "blocking_findings": [
    "invalid_version_directory_name:draft",
    "unreferenced_version:v003",
    "orphaned_version:v004"
  ],
  "warnings": [],
  "layout_version": "version-governed-v1",
  "next_required_phase": "governance_remediation",
  "inventory_sources": {
    "lineage_version_ids": ["v001"],
    "selection_version_ids": ["v002"],
    "version_root_direct_children": ["draft", "v001", "v002", "v003", "v004"]
  },
  "version_inventory": [
    {
      "version_id": "draft",
      "path": "<version_root>/draft",
      "sources": ["directory"],
      "naming_status": "invalid",
      "manifest_status": "not_inspected",
      "reference_status": "unreferenced",
      "findings": ["invalid_version_directory_name"]
    },
    {
      "version_id": "v001",
      "path": "<version_root>/v001",
      "sources": ["directory", "lineage"],
      "naming_status": "valid",
      "manifest_status": "valid",
      "reference_status": "referenced",
      "findings": []
    },
    {
      "version_id": "v002",
      "path": "<version_root>/v002",
      "sources": ["directory", "selection"],
      "naming_status": "valid",
      "manifest_status": "valid",
      "reference_status": "referenced",
      "findings": []
    },
    {
      "version_id": "v003",
      "path": "<version_root>/v003",
      "sources": ["directory"],
      "naming_status": "valid",
      "manifest_status": "valid",
      "reference_status": "unreferenced",
      "findings": ["unreferenced_version"]
    },
    {
      "version_id": "v004",
      "path": "<version_root>/v004",
      "sources": ["directory"],
      "naming_status": "valid",
      "manifest_status": "missing",
      "reference_status": "unreferenced",
      "findings": ["orphaned_version"]
    }
  ]
}
```

Every inventory entry requires exactly `version_id`, `path`, `sources`,
`naming_status`, `manifest_status`, `reference_status`, and `findings`.
`sources` is a sorted subset of `directory`, `lineage`, and `selection`.
Validate the complete output schema and require each raw source item to have one
inventory entry before writing either governance output.

Schema-version-1 `workspace_audit.json` is historical output only. Rerun the
governor to produce schema version 2 from the live lineage, selection, and
direct-child sources. Never synthesize `version_inventory` from the old audit,
and do not repair or move workspace artifacts during regeneration.

## Journaled Governance Publication

Publish `workspace_audit.json` and `workspace_audit.md`, and any separately
authorized governance migration, as one journaled atomic/recoverable logical
transaction. The exclusive journal path is
`<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json`;
it records target baseline existence/hash, staged path/hash, durable backup,
replacement order, commit index, and state
`prepared -> committing -> committed`. Fsync all staged files, backups, the
journal, and their parent directories before replacement.

Use the exact journal schema from `manage-strategy-version`: top-level
`schema_version`, `transaction_id`, `operation`, `state`, `commit_index`,
`targets`, and `replacement_order`; each target has `path`, `kind`, `baseline`,
`staged`, and `backup`. Set `operation: workspace_audit_publication` for the
audit pair and include both outputs.

Acquire persistent `workspace-governance.lock`, then
`final-selection.lock` last under the global lock order and hold both through
recovery, unchanged-byte checks, replacement, validation, fsync, and journal
commit. Before every replacement require exact baseline bytes or exact bytes
already written by this transaction.

Recover before new publication under the same locks. Before the first
replacement, discard only staging. After a non-pointer replacement and before
the logical pointer/last target, roll back exact prior bytes or absence from
each durable backup. After `current.json` replacement in a separately
authorized migration, roll forward only when every target equals exact staged
bytes; otherwise roll back the full set including `current.json`. Fsync parent
directories and retain recovery material until `committed` is durable.

## Red Lines

- Do not repair, move, or migrate inspected files; report the required migration
  as a governance finding for a separately authorized owner.
- Do not edit strategy specs, audits, runtime audits, runs, metrics, or reports.
- Do not mark a workspace clean when root-level phase artifact pollution remains.

## Result

Return the workspace status, blocking layout findings, stale or misplaced phase
artifacts, and the next governance action.
