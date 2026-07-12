---
name: audit-artifact-lineage
description: >-
  Use when open-xquant artifacts must be checked for version/run/final
  traceability before comparison, report review, migration, or final selection.
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

Read `.open-xquant/workspace.yaml` before inspecting version or run artifacts.
Resolve `version_root` from `paths.versions_dir`; use `versions` only when that
key is absent, and require the resolved root to remain inside the workspace.
Resolve every referenced version independently: read
`<version_root>/<referenced_version_id>/version_manifest.json` and use that
manifest's exact `phase_paths`, including `<phase_paths.09_backtests>`, for
artifacts belonging to that version. Reserve `current.json` for active-state checks;
do not use the active version's manifest to resolve another candidate. For
example, when the active version is `v003` but candidates reference `v001` and
`v002`, read `<version_root>/v001/version_manifest.json` and
`<version_root>/v002/version_manifest.json` respectively.

## Workspace Path Resolution

Read the same `.open-xquant/workspace.yaml` before inspecting or writing
governance artifacts. Resolve each key independently from `paths`: use
`paths.governance_dir`, `paths.comparisons_dir`, and `paths.final_dir`. Use
`governance`, `comparisons`, or `final`, respectively, only when that key is
absent; never derive one key from another configured key. Each configured value
must be a safe workspace-relative path. Reject absolute paths, traversal
outside the workspace, and symlink escapes whose resolved target leaves the
workspace.

Call the resolved paths `<governance_dir>`, `<comparisons_dir>`, and
`<final_dir>` below. This does not alter `version_root` or per-version
`phase_paths` resolution.

# Artifact Lineage Auditor

Round 26: lineage schema is `3`. Historical refresh consumes new report and
review evidence before fresh lineage, selection, and comparison artifacts; the
old selection is never reused. The route is `write -> review -> lineage ->
prepare new selection -> comparison -> resume` with fresh selection and
candidate hashes.

This skill audits artifact lineage across a strategy version, run package,
comparison, or final selection. It does not create versions, compare metrics,
write reports, or choose the winner.

## Inputs

- `workflow_manifest.json`
- `current.json`
- `lineage.json`
- version manifests under `<version_root>/**`
- run manifests and artifact hashes under `<phase_paths.09_backtests>/**`
- comparison manifests under `<comparisons_dir>/**`
- final selection artifacts under `<final_dir>/**`

## Checks

- Every run must reference one confirmed SPEC, one confirmed spec audit, one
  runtime audit, and one compiled plan from the same version.
- Every report must reference the version_id and run_id it discusses.
- Every final selection must reference only an eligible candidate.
- Every cross-artifact reference must carry the path/hash fields required by
  its producer schema. Require `hash_type` only on structured references whose schema defines that field,
  and validate it against that schema. Scalar hash
  fields such as `spec_hash`, `spec_audit_hash`, `runtime_audit_hash`,
  `artifact_hash`, and `event_hash` identify the algorithm through their
  required `sha256:<hex>` value and do not require a sibling `hash_type`.
- A final decision is blocked when lineage is missing, stale, or impossible to
  recompute.

Eligible candidate means the primary run has confirmed spec audit, passing
runtime audit, passing reproducibility audit, no fatal research audit, and an
eligible report review. A report review is eligible if and only if it has
status exactly `pass`, verdict exactly `consistent`, `blocking_findings`
exactly empty, `required_report_edits` exactly empty, and `errors` exactly
empty. Treat these five checks as one invariant. A non-pass status, a
non-consistent verdict, or any non-empty blocker, edit, or error list makes the
candidate ineligible even when every path and hash is valid. This is the same
invariant enforced by the report-review producer and final selector. No
advisory exception is allowed.

## Transitive Current-Evidence Validation

Before evidence consumption and again immediately before publication,
independently invoke `validate_run_artifact_inventory(run_dir)` independent of
the digest-row check. Treat its immutable return value as authoritative and
require
`profile.contract_schema_version == RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION == 1`.
Select the exactly one current digest row separately and require
`digest_row.artifact_inventory == {"schema_version": 1, "profile": profile.name}`.
The profile is derived only from `artifact_hashes.json.schema_version`: accept
the runtime-defined `artifact_hashes_v0_legacy` through `artifact_hashes_v5`
profiles and reject omission, unknown or unbound extensions, aliases,
duplicates, unsafe or stale bindings, and profile downgrade or mismatch. A
digest-row pass never substitutes for this executable inventory call.

`require_current_run_digest()` is not the complete current-evidence gate.
Regardless of helper implementation, callers must independently validate the
producer-required manifest inventory and transitive bindings. A lineage audit must perform
full manifest-entry integrity validation before reading candidate evidence and
again immediately before publication. Parse the manifest as an object, derive
every required entry from its producer schema, and validate every non-metadata
entry. Each key must be a safe relative path resolving to one current regular
file under the exact direct run, with no duplicate canonical target or symlink
escape. Recompute each value with the producer's artifact-specific algorithm,
including its canonical-JSON exclusions where defined, and require equality.
Reject a missing required manifest entry, a missing or extra governed target,
an invalid digest, or a stale entry.

A mutation without a manifest refresh leaves the manifest digest unchanged but
must fail manifest-entry integrity. A mutation followed by a valid manifest and
run-digest refresh passes that primitive only after all entries are current, but
it invalidates any older report-review or lineage binding. Neither case may be
accepted from the digest-row result alone.

After validating the nine lineage inputs below, recursively validate the bound
`report_review.json`: exact version/run identity, exactly one current run-digest
row, full manifest-entry integrity, exact five `decision_inputs` with current
file hashes, exact four `reviewed_artifacts` with current file hashes, and exact
registered report figure/source-script set and hashes. Repeat this transitive
validation immediately before publishing `status: pass`. Do not trust a review
status, a lineage status, or a containing file hash as a substitute for its
current-input validation.

## Lineage Audit Schema

Audit exactly one candidate version/run per lineage result. Resolve every
candidate artifact through that candidate's exact `version_manifest.json` and
its manifest-owned `phase_paths`; never mix active-version defaults into a
historical candidate audit. A schema-version-2 JSON result has exactly this
field inventory:

```json
{
  "schema_version": 2,
  "status": "pass",
  "scope": {
    "version_id": "v002",
    "run_id": "run_20260712_173012"
  },
  "hash_algorithm": "sha256-file-bytes-v1",
  "input_hashes": [
    {
      "path": "<version_root>/v002/version_manifest.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
      "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    {
      "path": "<phase_paths.06_spec_audit>/spec_audit.json",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    {
      "path": "<phase_paths.07_compile_preview>/compiled_plan.json",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    {
      "path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
      "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
    },
    {
      "path": "<phase_paths.09_backtests>/run_20260712_173012/artifact_hashes.json",
      "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    {
      "path": "<phase_paths.09_backtests>/run_20260712_173012/reproducibility_audit.json",
      "sha256": "sha256:8888888888888888888888888888888888888888888888888888888888888888"
    },
    {
      "path": "<phase_paths.09_backtests>/run_20260712_173012/research_bias_audit.json",
      "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
    },
    {
      "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ],
  "checked_artifacts": [
    "strategy_spec.yaml",
    "runtime_audit.json",
    "reproducibility_audit.json",
    "report_review.json"
  ],
  "blocking_findings": [],
  "warnings": [],
  "next_required_phase": "final_selection"
}
```

For schema version 2, the mandatory logical inputs are a closed set. These nine
paths are the complete mandatory inventory; they are not a producer-selected
material-input list. Derive these exact candidate-owned paths from the candidate's validated version
manifest and direct run/report directories before constructing `input_hashes`:

- `<version_root>/<version_id>/version_manifest.json`
- `<phase_paths.04_spec_build>/strategy_spec.yaml`
- `<phase_paths.06_spec_audit>/spec_audit.json`
- `<phase_paths.07_compile_preview>/compiled_plan.json`
- `<phase_paths.08_runtime_audit>/runtime_audit.json`
- `<phase_paths.09_backtests>/<run_id>/artifact_hashes.json`
- `<phase_paths.09_backtests>/<run_id>/reproducibility_audit.json`
- `<phase_paths.09_backtests>/<run_id>/research_bias_audit.json`
- `<phase_paths.10_reports>/<run_id>/report_review.json`

`input_hashes` must have exact set equality with those derived paths. Each
entry has exactly `path` and `sha256`. Reject omissions, additions, a wrong
phase, a wrong run, a duplicate recorded path or duplicate canonical target,
an unsafe path, a symlink escape, and a path that does not resolve to the exact
derived regular file. Recompute full SHA-256 over every exact current file
immediately before publication and reject stale hashes. `checked_artifacts` is
only a human-readable summary and never defines or relaxes the mandatory set.
`status: pass` also requires exact candidate scope and empty
`blocking_findings`; producer self-attestation is not evidence of completeness.

Schema-version-1 lineage audits are historical only. Do not backfill `scope` or
`input_hashes`, infer them from directory names, or mutate an old audit in
place. Rerun artifact lineage audit against the exact current manifest-owned
inputs to publish schema version 2.

An incomplete schema-version-2 lineage audit is invalid under the closed-set
contract. Do not append missing hashes or repair paths in place. Rerun artifact
lineage audit from the independently derived exact current inputs and publish a
new result. Regenerate the audit after any mandatory input or transitive
report-review binding changes.

## Selection-Bound Comparison Validation

For any current comparison offered to final selection, accept only a
schema-version-3 comparison manifest. Require its exact `selection_id` and exact
`{path, sha256}` candidate-set reference to equal the enclosing selection and
current schema-version-3 candidate-set bytes. Require its exact
`selection_policy` reference to equal the candidate-set policy binding and
current schema-version-3 policy bytes, including the coordinator confirmation
event and exact selection id; reject stale or cross-selection policy. Candidate evidence must equal the exact ordered
projection of complete candidate-set entries selected by the comparison
population, including ordinal, identity, `primary_run`, and complete
`lineage_audit`. Re-run the complete lineage-v2 and current run-inventory
checks for every projected entry. Reject cross-selection substitution even when
two selections contain identical identities.

Schema-version-1 comparison manifests are historical only; schema-version-2
comparison manifests are also historical for current final-selection
consumption. Do not infer or backfill missing
revision/confirmation bindings. Regenerate from the exact current candidate set
and transitive evidence as schema version 3. The phrase schema-version-2
comparison manifest refers only to the historical compatibility validator.

Before replacing a lineage audit that a prepared selection can consume,
participate in the workspace selection-lock protocol. Resolve the canonical
subject with `governing_workspace_root(subject)` and precompute
`final_selection_lock_path(subject)`. Discovery uses the canonical subject and
nearest ancestor `.open-xquant/workspace.yaml`; a valid non-governed subject
uses no lock, while malformed or unsafe governed configuration must fail
closed. Release run and registry locks, then use
`hold_final_selection_lock(precomputed_path)` as the last lock acquired,
publish immutable bytes atomically, and release without unlinking canonical
`<workspace_root>/.open-xquant/locks/final-selection.lock`. Never acquire
another lock while holding it.

## Outputs

```text
<governance_dir>/lineage_audit_<timestamp>.json
<governance_dir>/lineage_audit_<timestamp>.md
```

The JSON must conform to the schema and invariants above.

## Red Lines

- Do not rewrite hashes to make lineage pass.
- Do not compare strategy performance.
- Do not select a final version.
- Do not infer that missing audit artifacts passed.

## Result

Return whether artifact lineage is pass, block, or fail, and list the exact
version/run/final references that need repair or regeneration.

## Round 25 Revision Lineage

Current lineage output is schema version 3. In addition to the schema-version-2
mandatory inputs above, it contains an exact `{path, sha256}` report-revision
reference to `candidate_manifest.json` and an exact `{path, sha256}`
review-revision reference to the immutable schema-version-2
`report_review.json`. The review's embedded report reference must equal the
lineage report reference exactly. Hash both current files into `input_hashes`
and expose the same two references as top-level `report_revision` and
`report_review`. Schema version 2 is historical for new selection because it
cannot distinguish immutable report and review revisions.

The exact top-level keys are `schema_version`, `status`, `scope`,
`hash_algorithm`, `report_revision`, `report_review`, `input_hashes`,
`checked_artifacts`, `blocking_findings`, `warnings`, and
`next_required_phase`. The input-hash set is the old mandatory nine paths with
the direct review path replaced by the immutable review-revision path, plus the
candidate manifest path; no mutable direct report path is accepted.

Treat each target as an immutable report revision and immutable review
revision. Create every new lineage audit with a fresh output id; never overwrite,
repair, or substitute evidence reachable from any prior selection. Revalidate
the candidate manifest's `chart_build_result.json`: its
requested/applicable/generated/skipped set invariants, closed skip reason codes,
exact `{path, sha256}` manifest reference, and generated asset hashes. For every
asset require a safe package-relative `source.script`, full lowercase
`source.script_sha256`, and recompute the script SHA-256. Script mutation makes
lineage stale even when the containing manifest hash was copied correctly.

For `candidate_scoped_historical_report_revision`, audit only the explicit
inactive version/run under the current-state guard after the writer and reviewer
publish fresh `report_revision_id` and fresh `review_revision_id` values. The
auditor must not reactivate the inactive version and must not overwrite the old
lineage or revision. Return a fresh lineage audit for
`write -> review -> lineage -> comparison -> reselection`; prior revision bytes
remain reachable.

`final_decision.json` is the sole canonical decision artifact. Lineage never
requires, emits, or validates a second decision-format file.
