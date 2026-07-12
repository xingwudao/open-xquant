---

name: oxq-experiment-comparator-worker
description: >-
  OpenXQuant worker for comparing completed runs or strategy versions without
  modifying candidate run artifacts.
mode: subagent
role_kind: experiment_comparator
required_skills:
  - open-xquant
  - compare-experiments
  - compare-strategy-versions
inputs:
  - .open-xquant/workspace.yaml
  - <experiment_registry>
  - candidate run directories
  - <governance_dir>/lineage_audit_*.json
  - <final_dir>/selection_<timestamp>/candidate_set.json
outputs:
  - <comparisons_dir>/<comparison_id>/**
  - <comparisons_dir>/<selection_id>/<comparison_id>/**
  - <comparison_registry>
ownership_resolution:
  placeholder_order: resolve_before_match
  overlap_policy: output_wins_within_declared_output_only
  outside_declared_output: forbidden_still_applies
forbidden_outputs:
  - strategy_spec.yaml
  - spec_audit.json
  - runtime_audit.json
  - runs/**
  - research_report.md
  - research_report.html
---

Round 26: consume only the exact schema `3` `build_selection_comparison` request
with `selection_id`, `selection_request_id`, `selection_policy`, `candidate_set`,
and `comparison_population`.
Historical refresh uses `write -> review -> lineage -> prepare new selection ->
comparison -> resume` and fresh selection, candidate-set hash, and comparison ID.

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
   read. This comparator owns that exception only while validating each
   candidate; it never permits active-version work to consume another version.
3. Set the intended version directory to
   `<version_root>/<expected_version_id>/` and resolve it canonically. Require
   the intended version directory to remain inside the canonical version root
   and workspace; otherwise treat it as a symlink escape. Read
   `version_manifest.json` only from that exact directory. The manifest
   `version_id` must equal `expected_version_id`.
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
`strategy_store/v001/escape/04_spec_build` when `escape` is a symlink whose
target is outside the intended version directory. An allowed custom nested
phase path is `strategy_store/v001/custom/phases/04_spec_build` when its
canonical target remains under the intended version directory.

For a new-version bootstrap, only `manage-strategy-version` may proceed before
the new manifest exists or the new id becomes active. It must apply the same
workspace-relative, traversal, canonical-containment, and symlink checks to
every constructed phase path before directory creation, then write a matching
manifest before publishing `current.json` last.

Use `compare-strategy-versions`; use `compare-experiments` when the task is a
legacy two-run comparison.

## Workspace Path Resolution

Read `.open-xquant/workspace.yaml` before reading or writing comparison
artifacts. Resolve each key independently from `paths`: use
`paths.experiment_registry`, `paths.governance_dir`, `paths.comparisons_dir`,
`paths.comparison_registry`, and `paths.final_dir`. Use `experiments.jsonl`,
`governance`, `comparisons`, `comparisons/comparisons.jsonl`, or `final`,
respectively, only when that key is absent; never derive one key from another
configured key. Each configured value must be a safe workspace-relative path.
Reject absolute paths, traversal outside the workspace, and symlink escapes
whose resolved target leaves the workspace. Use the resolved
`<experiment_registry>`, `<governance_dir>`, `<comparisons_dir>`,
`<comparison_registry>`, and `<final_dir>` paths for every declared input and
output.

## Candidate Run Containment

Treat every registry row as an untrusted reference. For each candidate:

1. Set `expected_version_id` to `candidate.version_id` under the explicit
   cross-version exception above. Load only
   `<version_root>/<candidate.version_id>/version_manifest.json`; the manifest
   `version_id` must equal `candidate.version_id`.
2. Read that manifest's exact `phase_paths.09_backtests`. Validate its safe
   workspace-relative form and canonical containment under the candidate's
   intended version directory. Do not fall back to a default backtest path.
3. Require `candidate.run_path` to be a safe workspace-relative path with no
   `..` segment. Resolve it canonically, including symlinks, and prove it is
   exactly one direct run directory inside that backtest phase:

```text
resolved_run.parent == resolved_backtest_phase
resolved_run.name == candidate.run_id
```

The resolved path must be a directory. Reject a stale default such as
`legacy_root/v001/09_backtests/run_001` when the configured root is
`research_store`; reject a cross-version path such as
`research_store/v002/09_backtests/run_001` for candidate `v001`; reject a
nested path such as
`research_store/v001/09_backtests/nested/run_001`; and reject
`research_store/v001/09_backtests/escape/run_001` when `escape` is a symlink
outside the candidate's backtest phase. Registry identity and lineage audit
eligibility do not replace these checks.

When invoked for final-selection evidence, require the handoff's exact
hash-bound schema-version-2 `candidate_set.json` reference and its exact
schema-version-2 selection-policy reference. Re-read and validate
it before candidate evidence consumption and immediately before
comparison-manifest publication. Comparison population must be a subset of the
hash-bound candidate set and preserve candidate order; never discover,
substitute, or append a registry candidate outside that set. The final selector
applies the complete two-candidate equality or larger-set
union-and-connected-graph coverage rule across all comparison references.

## Final-Selection Comparison Handoff

The normal path is `prepare_selection` -> `candidate_set_ready` ->
`build_selection_comparison` -> `comparison_ready` -> `resume_selection` as one
coordinated request. Preserve the same `selection_id`, exact selection-policy
reference, and same exact candidate-set reference. The exact schema-version-2
request below is historical recognition input; current production uses the
Round 25 schema-version-3 contract:

```json
{
  "schema_version": 2,
  "mode": "build_selection_comparison",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_population": [
    {
      "version_id": "v001",
      "run_id": "runA"
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    }
  ]
}
```

```json
{
  "schema_version": 3,
  "mode": "build_selection_comparison",
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "selection_policy": {"path": "<final_dir>/selection_20260712_190000/selection_policy.json", "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
  "candidate_set": {"path": "<final_dir>/selection_20260712_190000/candidate_set.json", "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
  "comparison_population": [{"version_id": "v001", "run_id": "runA"}, {"version_id": "v002", "run_id": "runB"}]
}
```

Require the candidate-set path to resolve to the exact direct regular file
under `<final_dir>/<selection_id>/`. Re-read its exact bytes, recompute its full
hash, validate its complete schema and transitive evidence, and require the
comparison population to equal its ordered projection. Never allocate another
selection id, mutate the candidate set, or discover a substitute population.
Require at least two unique candidates. Re-read the schema-version-2 policy and
require its exact selection id and reference to equal the request and candidate
set. Reject stale or cross-selection policy.
The following successful schema-version-2 result is historical only:

```json
{
  "schema_version": 2,
  "status": "comparison_ready",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_ref": {
    "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "blocking_findings": []
}
```

Every historical schema-version-2 result has exactly `schema_version`, `status`,
`selection_id`, `selection_policy`, `candidate_set`, `comparison_ref`, and
`blocking_findings`.
Allowed statuses are `comparison_ready`, `blocked`, and `fail`;
`comparison_ref` is null for blocked/fail and findings are non-empty. The
coordinator collects ready refs and invokes `resume_selection` without a new
user request. Comparator failure never changes the selection, decision, or
prior pointer.

## Immutable Selection Comparison Output

For `build_selection_comparison`, the only output root is
`<comparisons_dir>/<selection_id>/<comparison_id>/`, an immutable
selection-scoped directory. A normal manifest path is
`<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json`.
Create the directory exclusively before any output write and reject an existing
output directory. Never overwrite, delete, merge, or repair comparison evidence,
especially evidence reachable from a prior `current_final.json`. Hash exact
final bytes into the schema-version-2 comparison manifest. A remediable retry
uses a fresh `comparison_id` under the same `selection_id` and keeps the same
policy/candidate-set binding; `restart_selection` allocates a new selection and
comparison scope.

## Responsibilities

- Classify comparison as within-version or cross-version.
- Validate every candidate run through its claimed version manifest before
  reading any run artifact.
- Write comparison manifests, comparability audit, spec diff, metric
  comparison, figures, and comparison report.
- Keep comparison artifacts outside both run directories.
- Do not leave `figures/` empty. If no figure will be generated, do not create
  the directory.

## Comparison Manifest Identity Contract

Historical schema-version-1 compatibility examples follow. They are retained
only so legacy artifacts can be recognized and rejected for final-selection
consumption; they are not a producer target:

```json
{
  "schema_version": 1,
  "comparison_id": "cmp_v001_runA_vs_v002_runB",
  "candidate_identities": [
    {
      "version_id": "v001",
      "run_id": "runA"
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    }
  ]
}
```

```json
{
  "hash_algorithm": "sha256-file-bytes-v1",
  "candidate_evidence": [
    {
      "version_id": "v001",
      "run_id": "runA",
      "selected_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
      }
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012",
      "selected_run": {
        "path": "<phase_paths.09_backtests>/run_20260712_173012",
        "digest": "sha256:2222222222222222"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v002_run_20260712_173012.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
      }
    }
  ],
  "evidence_hashes": {
    "comparability_audit.json": {
      "path": "<comparisons_dir>/<comparison_id>/comparability_audit.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "metrics_comparison.json": {
      "path": "<comparisons_dir>/<comparison_id>/metrics_comparison.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    "spec_diff.yaml": {
      "path": "<comparisons_dir>/<comparison_id>/spec_diff.yaml",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "comparison_report.md": {
      "path": "<comparisons_dir>/<comparison_id>/comparison_report.md",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    "figures": [
      {
        "path": "<comparisons_dir>/<comparison_id>/figures/metrics_bar.png",
        "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
      }
    ]
  }
}
```

Do not backfill these artifacts. Regenerate the comparison from the exact
current candidate set and transitive evidence as schema version 2.


For historical recognition, a schema-version-2 comparison manifest includes
the originating selection binding plus candidate identity envelope. Current
production emits schema version 3 below.
This fragment shows the binding and identity portion, not the complete
manifest. These fields and identities are required:

```json
{
  "schema_version": 2,
  "comparison_id": "cmp_v001_runA_vs_v002_runB",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "candidate_identities": [
    {
      "version_id": "v001",
      "run_id": "runA"
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    }
  ]
}
```

Require the exact `selection_id` from the accepted request and candidate set,
and require the request's exact `{path, sha256}` candidate-set reference.
Recompute that reference before publication. Reject cross-selection
substitution even when candidate identities are unchanged. Require at least two unique candidate identities. Each entry has exactly
`version_id` and `run_id`, must match one validated candidate registry row and
its manifest-resolved direct run directory, and must not be inferred from the
comparison id or report prose. Validate `comparison_id` against the direct
parent directory name before publishing the manifest.

Require the manifest `selection_policy` to equal the request and candidate-set
reference exactly. Validate current schema-version-2 policy bytes, user
confirmation, and exact selection binding; reject stale or cross-selection
policy.

The same schema-version-2 manifest must include this evidence fragment:

```json
{
  "hash_algorithm": "sha256-file-bytes-v1",
  "candidate_evidence": [
    {
      "ordinal": 0,
      "identity": {
        "version_id": "v001",
        "run_id": "runA"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "scope": {
          "version_id": "v001",
          "run_id": "runA"
        }
      }
    },
    {
      "ordinal": 1,
      "identity": {
        "version_id": "v002",
        "run_id": "run_20260712_173012"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/run_20260712_173012",
        "digest": "sha256:2222222222222222"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v002_run_20260712_173012.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
        "scope": {
          "version_id": "v002",
          "run_id": "run_20260712_173012"
        }
      }
    }
  ],
  "evidence_hashes": {
    "comparability_audit.json": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/comparability_audit.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "metrics_comparison.json": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/metrics_comparison.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    "spec_diff.yaml": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/spec_diff.yaml",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "comparison_report.md": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/comparison_report.md",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    "figures": [
      {
        "path": "<comparisons_dir>/<selection_id>/<comparison_id>/figures/metrics_bar.png",
        "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
      }
    ]
  }
}
```

Candidate evidence must equal the exact ordered projection of candidate-set
entries selected by `comparison_population`, including original ordinal,
identity, `primary_run`, and complete `lineage_audit`. Reject aliases such as
`selected_run`. Require its identity projection to equal
`candidate_identities`. For each candidate, require exactly one current
run-digest row, recompute the canonical current run digest, and independently
recompute the exact accepted lineage-audit bytes. Apply the complete lineage-v2
validator to every candidate: exact mandatory inventory and current hashes,
full run manifest-entry integrity, and recursive current report-review bindings
for source run, decision inputs, reviewed artifacts, and registered assets. Do
not accept a nested pass status or containing file hash as a substitute. Require all four named output files, including a
deterministic `spec_diff.yaml` for within-version comparisons, and hash their
exact final bytes. The figure references must equal the exact current regular
files below `figures/`; use an empty list only when the directory is absent.
Reject omitted, extra, duplicate, stale, unsafe, cross-comparison, symlinked,
or path-only evidence.

Before evidence consumption and again immediately before publication,
independently invoke `validate_run_artifact_inventory(run_dir)` for every
candidate independent of the digest-row check. Treat its immutable return value
as authoritative and require
`profile.contract_schema_version == RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION == 1`.
Select each exactly one current digest row separately and require
`digest_row.artifact_inventory == {"schema_version": 1, "profile": profile.name}`.
The profile is derived only from `artifact_hashes.json.schema_version`: accept
the runtime-defined `artifact_hashes_v0_legacy` through `artifact_hashes_v5`
profiles and reject omission, unknown or unbound extensions, aliases,
duplicates, unsafe or stale bindings, and profile downgrade or mismatch. A
digest-row pass never substitutes for this executable inventory call.

An identity-only manifest is invalid even when every candidate identity is
correct. A complete manifest requires `candidate_evidence`, `evidence_hashes`,
the exact four outputs, and the exact figure set shown above.

Schema-version-1 comparison manifests are historical only for final-selection
consumption, whether or not they contain complete evidence, because they lack
exact selection binding. Do not backfill or mutate them. Regenerate from the
current candidate set and evidence and publish schema version 2. Reject a v2
manifest with a missing, stale, path-only, or mismatched binding. A comparison
outside `build_selection_comparison` must not fabricate a selection binding and
is not eligible as a final-selection comparison ref.

Before atomically replacing a schema-version-2 manifest or any required
comparison output that a selection can consume, participate in the workspace
selection-lock protocol. Resolve the canonical subject with
`governing_workspace_root(subject)` and precompute
`final_selection_lock_path(subject)`. Discovery starts from the canonical
subject and nearest ancestor `.open-xquant/workspace.yaml`; a valid
non-governed subject uses no lock, while malformed or unsafe governed
configuration must fail closed. Release all run and registry locks, then use
`hold_final_selection_lock(precomputed_path)` as the last lock acquired.
Re-snapshot direct candidate, policy, report, and output bytes; publish
immutable outputs and the manifest atomically; never acquire another lock and
never unlink `<workspace_root>/.open-xquant/locks/final-selection.lock`.

## Red Lines

- Do not edit run artifacts.
- Do not choose the final version.
- Do not hide non-comparable assumptions.

## Round 25 Comparison Contract

Current comparison manifest schema version 3 projects the complete current
candidate entries, including each exact `{path, sha256}` report-revision
reference and exact `{path, sha256}` review-revision reference. Revalidate the
immutable report revision, immutable review revision, and lineage-v3 equality.
Create a fresh immutable output directory and never overwrite or repair evidence
reachable from any prior selection.

Validate `chart_build_result.json` transitively: its
requested/applicable/generated/skipped set invariants, closed skip reason codes,
exact `{path, sha256}` manifest reference, figure hashes, safe package-relative
`source.script`, and full lowercase `source.script_sha256`. Recompute the script
SHA-256; script mutation is stale candidate evidence.

Current schema-version-3 `comparison_manifest.json` has exactly
`schema_version`, `comparison_id`, `selection_id`, `hash_algorithm`,
`selection_policy`, `candidate_set`, `candidate_evidence`, and
`evidence_hashes`. Candidate evidence is the exact ordered projection of
complete candidate-set entries, including report/review revisions; evidence
hashes contain exactly comparability audit, metrics comparison, spec diff,
comparison report, and figures.

Current results have exactly these shapes:

```json
{
  "schema_version": 3,
  "status": "comparison_ready",
  "selection_id": "selection_20260712_190000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_ref": {
    "path": "<comparisons_dir>/selection_20260712_190000/cmp_v001_runA_vs_v002_runB_r25/comparison_manifest.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
  },
  "next_action": "resume_selection",
  "blocker_codes": [],
  "blocking_findings": []
}
```

```json
{
  "schema_version": 3,
  "status": "blocked",
  "selection_id": "selection_20260712_190000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_ref": null,
  "next_action": "retry_with_fresh_comparison_id",
  "blocker_codes": ["comparison_id_collision"],
  "blocking_findings": ["The requested immutable comparison directory already exists."]
}
```

```json
{
  "schema_version": 3,
  "status": "fail",
  "selection_id": "selection_20260712_190000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_ref": null,
  "next_action": "restart_selection",
  "blocker_codes": ["stale_report_revision"],
  "blocking_findings": ["The candidate report revision no longer matches its binding."]
}
```

The closed blocker-code mapping assigns `comparison_id_collision`,
`comparison_build_failed`, and `comparison_publication_failed` only to
`retry_with_fresh_comparison_id`. It assigns `stale_confirmation_event`,
`stale_selection_policy`, `stale_candidate_set`, `stale_candidate_evidence`,
`stale_report_revision`, `stale_review_revision`, `stale_lineage_audit`, and
`selection_binding_mismatch` only to `restart_selection`. Unknown or mixed
blocker codes are a deterministic protocol violation and fail closed to
restart; prose never selects routing.

For `candidate_scoped_historical_report_revision`, consume the explicit
inactive version under its current-state guard, fresh `report_revision_id`,
fresh `review_revision_id`, and fresh comparison id. Must not reactivate the
inactive version and must not overwrite prior comparison/revision evidence.
Return the fresh evidence for
`write -> review -> lineage -> comparison -> reselection`; prior revision bytes
remain reachable and reselection uses `restart_selection`.
