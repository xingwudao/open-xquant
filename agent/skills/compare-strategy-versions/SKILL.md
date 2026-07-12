---
name: compare-strategy-versions
description: >-
  Use when comparing open-xquant runs or strategy versions, especially when
  distinguishing within-version reproducibility from cross-version strategy
  evidence.
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

# Strategy Version Comparator

Round 26: comparison schema `3`; the exact closed request fields are
`schema_version`, `mode`, `selection_id`, `selection_request_id`,
`selection_policy`, `candidate_set`, and `comparison_population`. Historical
refresh is `write -> review -> lineage -> prepare new selection -> comparison ->
resume` with fresh selection, candidate-set hash, and comparison ID.

This skill extends `compare-experiments` for version-governed workspaces. It
compares completed run packages without writing into either run directory.

## Workspace Path Resolution

Read `.open-xquant/workspace.yaml` before reading or writing comparison
artifacts. Resolve each key independently from `paths`: use
`paths.experiment_registry`, `paths.governance_dir`, `paths.comparisons_dir`,
`paths.comparison_registry`, and `paths.final_dir`. Use `experiments.jsonl`,
`governance`, `comparisons`, `comparisons/comparisons.jsonl`, or `final`,
respectively, only when that key is absent; never derive one key from another
configured key. Each configured value must be a safe workspace-relative path.
Reject absolute paths, traversal outside the workspace, and symlink escapes
whose resolved target leaves the workspace.

Call the resolved paths `<experiment_registry>`, `<governance_dir>`,
`<comparisons_dir>`, `<comparison_registry>`, and `<final_dir>` below.

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

When the comparator is invoked for final-selection evidence, the handoff must
include the exact hash-bound schema-version-2 `candidate_set.json` reference
and its exact schema-version-2 selection-policy reference.
Re-read and validate that candidate set before candidate evidence consumption
and immediately before comparison-manifest publication. Comparison population
must be a subset of the hash-bound candidate set and preserve candidate order;
never discover, substitute, or append a registry candidate outside that set.
The final selector applies the complete two-candidate equality or larger-set
union-and-connected-graph coverage rule across all comparison references.

## Final-Selection Comparison Handoff

The normal final-selection path is `prepare_selection` ->
`candidate_set_ready` -> `build_selection_comparison` -> `comparison_ready` ->
`resume_selection`. This is one coordinated request and uses the same
`selection_id`, same exact selection-policy reference, and same exact
candidate-set reference throughout. The exact schema-version-2 request below is
retained for historical recognition; current production uses the Round 25
schema-version-3 contract:

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

Resolve the candidate-set path as the exact direct regular file under
`<final_dir>/<selection_id>/`, re-read its exact bytes, recompute its full hash,
validate its complete schema and transitive evidence, and require
`comparison_population` to equal its ordered projection. Do not accept a
path-only ref, allocate a selection id, mutate the candidate set, or discover a
replacement population. On successful manifest publication return exactly:

Require at least two unique candidates. Re-read the schema-version-2 policy
artifact and require its exact selection id and `{path, sha256}` reference to
equal the request and candidate-set binding. Reject stale or cross-selection
policy.

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

Every historical schema-version-2 comparator result has exactly `schema_version`, `status`,
`selection_id`, `selection_policy`, `candidate_set`, `comparison_ref`, and
`blocking_findings`.
`status` is `comparison_ready`, `blocked`, or `fail`; `comparison_ref` is null
for blocked/fail and `blocking_findings` is then non-empty. The coordinator
collects one or more ready refs and invokes `resume_selection` without a new
user request. Comparator failure never changes the selection id, candidate set,
decision, or prior pointer.

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

## Comparison Types

Within-version comparison:

- same version_id
- same confirmed SPEC unless the comparison is explicitly a robustness variant
- use `oxq backtest compare-runs` as a strict comparability gate
- failures usually block winner-style conclusions

Cross-version comparison:

- different version_id values
- different `spec_hash` is expected, not automatically fatal
- produce a spec diff and explain likely impact
- do not name a winner from metrics alone

## Inputs

- Two or more candidate run packages.
- `<experiment_registry>` with version_id, run_id, run_path, and decision
  fields.
- Optional existing `<comparisons_dir>/<comparison_id>/` artifacts only for a
  standalone, non-selection comparison. `build_selection_comparison` rejects
  every existing output directory instead of treating it as optional input.
- `<governance_dir>/lineage_audit_*.json` for candidate lineage eligibility.
- Optional user-confirmed comparison scope.

## Workflow

1. Classify the comparison as `within-version` or `cross-version`.
2. Run or inspect strict comparability checks for execution, cost, validation,
   metrics profile, data, audit, and runtime assumptions.
3. For cross-version comparisons, write `spec_diff.yaml` and treat spec
   differences as evidence rather than an error.
4. Compare metrics only after audit and comparability context is explicit.
5. Write a comparison report that separates association from causality.

## Outputs

For standalone comparisons only:

```text
<comparisons_dir>/<comparison_id>/
  comparison_manifest.json
  comparability_audit.json
  spec_diff.yaml
  metrics_comparison.json
  comparison_report.md
  figures/
```

For `build_selection_comparison`, prepend the immutable selection scope and
use `<comparisons_dir>/<selection_id>/<comparison_id>/` instead.

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
production emits schema version 3 as defined below.
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

Require `selection_id` to equal the exact `selection_id` in the accepted
`build_selection_comparison` request and candidate set. Require `candidate_set`
to equal that request's exact `{path, sha256}` candidate-set reference and
recompute its bytes before publication. This binding is mandatory even when a
later selection has identical candidate identities; reject cross-selection
substitution. Require at least two unique candidate identities. Each entry has exactly
`version_id` and `run_id`, must match one validated candidate registry row and
its manifest-resolved direct run directory, and must not be inferred from the
comparison id or report prose. Validate `comparison_id` against the direct
parent directory name before publishing the manifest.

Require `selection_policy` to equal the request and candidate-set exact policy
reference. Re-read its current bytes, validate schema version 2, user
confirmation, and exact selection binding, and reject stale or cross-selection
policy. The policy reference is part of the comparison manifest's schema-v2
selection binding.

The same schema-version-2 manifest must also contain the following exact
evidence fields. This fragment shows their required shape:

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

Candidate evidence must equal the exact ordered projection of the hash-bound
candidate-set entries selected by `comparison_population`, including each
entry's original ordinal, identity, `primary_run`, and complete
`lineage_audit`; aliases such as `selected_run` are invalid. Its identity
projection must equal `candidate_identities` in the same order and without
duplicates. For each
candidate, require exactly one current run-digest registry row, recompute the
canonical current run digest, and independently recompute the exact accepted
candidate-scoped lineage audit bytes. Apply the complete lineage-v2 validator
to every candidate: exact mandatory inventory and current hashes, full run
manifest-entry integrity, and recursive current report-review bindings for the
source run, decision inputs, reviewed artifacts, and registered assets. Do not
accept a nested pass status or containing file hash as a substitute. These bindings cover every candidate input used by the
comparison; do not accept path-only or self-attested input evidence.

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

All four named comparison outputs are mandatory for every comparison. For a
within-version comparison, `spec_diff.yaml` records the deterministic no-change
or robustness-variant result rather than being omitted. Hash exact final bytes
immediately before publishing the manifest. `evidence_hashes.figures` must
equal the exact current regular-file set below `figures/`, with one unique
path/hash entry per file; it is empty only when the directory is absent. Reject
omitted, extra, duplicate, stale, unsafe, cross-comparison, or symlinked
evidence. No evidence used or produced by the comparison may remain path-only
or outside these transitive bindings.

An identity-only manifest is invalid even when every candidate identity is
correct. A complete manifest requires `candidate_evidence`, `evidence_hashes`,
the exact four outputs, and the exact figure set shown above.

Schema-version-1 comparison manifests are historical only for final-selection
consumption, whether or not they contain complete evidence, because they lack
the exact selection binding. Do not backfill hashes or mutate them in place.
Regenerate the comparison from the current candidate set and current evidence,
then publish a complete schema-version-2 manifest. A schema-version-2 manifest
with a missing, stale, path-only, or mismatched binding is also invalid. A
comparison outside `build_selection_comparison` must not fabricate a selection
binding and is not eligible as a final-selection comparison ref.

Before atomically replacing a schema-version-2 manifest or any required
comparison output that a selection can consume, participate in the workspace
selection-lock protocol. Resolve the canonical subject with
`governing_workspace_root(subject)` and precompute
`final_selection_lock_path(subject)` before taking any publication lock.
Discovery uses the canonical subject and nearest ancestor
`.open-xquant/workspace.yaml`; a valid non-governed subject uses no lock, while
malformed or unsafe governed configuration must fail closed. Complete and
release run and registry locks, then call
`hold_final_selection_lock(precomputed_path)` as the last lock acquired.
Re-snapshot direct candidate, policy, lineage, report, and output bytes, publish
immutable comparison outputs, and atomically replace the manifest last. Release
the lock without unlinking the canonical
`<workspace_root>/.open-xquant/locks/final-selection.lock`; never acquire
another lock while holding it.

Do not leave `figures/` empty. If no figure will be generated, do not create the
directory and record `figures: []`. If the directory is created, write at least
one referenced comparison figure such as `metrics_bar.png`,
`equity_overlay.png`, or `drawdown_overlay.png`, and hash-register every file
in `comparison_manifest.json`.

Append a summary row to `<comparison_registry>`.

## Red Lines

- Do not modify run artifacts.
- Do not hide non-comparable assumptions.
- Do not select the final version.
- Do not present an unaudited candidate as comparable to an audited candidate.

## Result

For a standalone comparison, return comparison status, comparison id,
comparable fields, blockers, metric deltas, and selection eligibility. For
`build_selection_comparison`, return the exact result envelope above so the
coordinator can resume the same selection.

## Round 25 Comparison Contract

Current final-selection comparison manifests use schema version 3 and project
complete schema-version-3 candidate entries, including each exact
`{path, sha256}` report-revision reference and exact `{path, sha256}`
review-revision reference. Revalidate the immutable report revision, immutable
review revision, lineage-v3 equality, and current bytes before consumption and
publication. Create a fresh comparison directory exclusively; never overwrite,
repair, or substitute evidence reachable from any prior selection.

Recursively validate `chart_build_result.json`: the
requested/applicable/generated/skipped set invariants, closed skip reason codes,
exact `{path, sha256}` manifest reference, and generated figure bytes. Require
every safe package-relative `source.script`, full lowercase
`source.script_sha256`, and recompute the script SHA-256. Script mutation is
stale candidate evidence, not a remediable comparison-build error.

Current `comparison_manifest.json` uses exactly this schema-version-3 field
inventory. `candidate_evidence` is an exact ordered projection of complete
candidate-set entries, and `evidence_hashes` is the closed output inventory:

```json
{
  "schema_version": 3,
  "comparison_id": "cmp_v001_runA_vs_v002_runB_r25",
  "selection_id": "selection_20260712_190000",
  "hash_algorithm": "sha256-file-bytes-v1",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
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
      "report_revision": {
        "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/candidate_manifest.json",
        "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "report_review": {
        "path": "<phase_paths.10_reports>/runA/reviews/review_20260712_181500/report_review.json",
        "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA_r25.json",
        "sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
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
        "run_id": "runB"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/runB",
        "digest": "sha256:2222222222222222"
      },
      "report_revision": {
        "path": "<phase_paths.10_reports>/runB/candidates/report_20260712_182000/candidate_manifest.json",
        "sha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
      },
      "report_review": {
        "path": "<phase_paths.10_reports>/runB/reviews/review_20260712_182500/report_review.json",
        "sha256": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v002_runB_r25.json",
        "sha256": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "scope": {
          "version_id": "v002",
          "run_id": "runB"
        }
      }
    }
  ],
  "evidence_hashes": {
    "comparability_audit.json": {
      "path": "<comparisons_dir>/selection_20260712_190000/cmp_v001_runA_vs_v002_runB_r25/comparability_audit.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "metrics_comparison.json": {
      "path": "<comparisons_dir>/selection_20260712_190000/cmp_v001_runA_vs_v002_runB_r25/metrics_comparison.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    "spec_diff.yaml": {
      "path": "<comparisons_dir>/selection_20260712_190000/cmp_v001_runA_vs_v002_runB_r25/spec_diff.yaml",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "comparison_report.md": {
      "path": "<comparisons_dir>/selection_20260712_190000/cmp_v001_runA_vs_v002_runB_r25/comparison_report.md",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    "figures": []
  }
}
```

Every current comparator result uses schema version 3 and exactly the nine keys
below. `next_action` is a closed union and `blocker_codes` uses the closed
blocker-code mapping after these examples:

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

`comparison_ready` requires `resume_selection`, an empty blocker list, and a
non-null reference. `retry_with_fresh_comparison_id` is valid only for
`comparison_id_collision`, `comparison_build_failed`, or
`comparison_publication_failed`; keep the same selection and candidate set but
allocate a fresh `comparison_id`. `restart_selection` is required for
`stale_confirmation_event`, `stale_selection_policy`, `stale_candidate_set`,
`stale_candidate_evidence`, `stale_report_revision`, `stale_review_revision`,
`stale_lineage_audit`, or `selection_binding_mismatch`. Unknown or mixed blocker
codes are a deterministic protocol violation and fail closed to
`restart_selection`; do not guess from prose.

For `candidate_scoped_historical_report_revision`, compare the fresh lineage,
report, and review refs for the explicit inactive version under the current-state
guard. Use the supplied fresh `report_revision_id`, fresh
`review_revision_id`, and a fresh comparison id. Must not reactivate the
inactive version and must not overwrite old comparison or revision evidence.
This is the comparison step in
`write -> review -> lineage -> comparison -> reselection`; prior revision bytes
remain reachable and the coordinator next uses `restart_selection`.
