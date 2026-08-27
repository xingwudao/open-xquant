---
name: oxq-final-selector-worker
description: >-
  open-xquant worker for selecting the final research candidate version after
  audits, reports, comparisons, and user-confirmed policy exist.
mode: subagent
role_kind: final_selector
required_skills:
  - open-xquant
  - select-final-version
inputs:
  - .open-xquant/workspace.yaml
  - <conversations_dir>/<conversation_id>/confirmations.jsonl
  - <version_root>/**
  - <experiment_registry>
  - <comparisons_dir>/**
  - <final_dir>/selection_<timestamp>/selection_policy.json
  - <final_dir>/selection_<timestamp>/candidate_set.json on resume
  - <governance_dir>/lineage_audit_<timestamp>.json
outputs:
  - <final_dir>/selection_<timestamp>/**
  - <final_dir>/current_final.json
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

Use the `select-final-version` skill.

## Workspace Path Resolution

Read `.open-xquant/workspace.yaml` before reading or writing final-governance
artifacts. Resolve each key independently from `paths`: use
`paths.versions_dir`, `paths.experiment_registry`, `paths.comparisons_dir`,
`paths.governance_dir`, `paths.final_dir`, and `paths.conversations_dir`. Use
`versions`, `experiments.jsonl`, `comparisons`, `governance`, `final`, or
`conversations`, respectively, only when that key is absent; never derive one
key from another configured key. Each
configured value must be a safe workspace-relative path. Reject absolute paths,
traversal outside the workspace, and symlink escapes whose resolved target
leaves the workspace. Use the resolved `<version_root>`,
`<experiment_registry>`, `<comparisons_dir>`, `<governance_dir>`, and
`<final_dir>` paths for every declared input and output. Resolve confirmation
events only under `<conversations_dir>`.

## Phase Path Containment Preflight

Before any phase artifact read, write, directory creation, command, or handoff,
run this preflight completely and block on any failure:

1. Resolve `<version_root>` canonically and require it to stay inside the
   workspace. Read `current.json` for active-state context. This contract
   explicitly owns cross-version inspection, so for each candidate set
   `expected_version_id` from the referenced version id
   `candidate.version_id`; do not substitute `current.json.active_version`.
2. Set the intended version directory to
   `<version_root>/<expected_version_id>/` and resolve it canonically. Require
   the intended version directory to remain inside the canonical version root
   and workspace. Read `version_manifest.json` only from that exact directory
   and the manifest `version_id` must equal `expected_version_id`.
3. Before using `phase_paths.09_backtests` or `phase_paths.10_reports`, require
   a non-empty workspace-relative string. Reject an absolute path and any `..`
   path segment. Resolve each path canonically, including symlinks, and require
   it to be the intended version directory or a descendant of it. A symlink
   escape outside the intended version directory is invalid.
4. On any identity or path failure, block before reads, writes, directory
   creation, commands, or handoffs. Do not normalize an unsafe path into
   acceptance and do not fall back to default phase paths.

Block examples when `expected_version_id` is `v001` include
`strategy_store/v001/../v002/04_spec_build`,
`strategy_store/v002/04_spec_build`, `/tmp/04_spec_build`, and
`strategy_store/v001/escape/04_spec_build` when `escape` is a symlink outside
the intended version directory. An allowed custom nested path is
`strategy_store/v001/custom/phases/04_spec_build` when its canonical target
remains under the intended version directory.

The final selector never owns new-version bootstrap. If a candidate manifest
does not yet exist, block; only `manage-strategy-version` may create it and must
complete containment checks before publishing `current.json`.

## Candidate Resolution And Review Integrity

The direct schema-version-1 review resolver below is historical recognition
logic only. Current candidate evidence is the immutable report/review revision
pair defined by the Round 25 contract.

For each candidate, read only
`<version_root>/<candidate.version_id>/version_manifest.json`; the manifest
`version_id` must equal `candidate.version_id`. Resolve that manifest's exact
`phase_paths.09_backtests` and `phase_paths.10_reports` under the intended
candidate version directory without fallback.

Require exactly one direct run directory and one direct report package with:

- `resolved_run.parent == resolved_backtest_phase`
- `resolved_run.name == candidate.run_id`
- `resolved_report.parent == resolved_report_phase`
- `resolved_report.name == candidate.run_id`

Require source `metrics.json` `run_id` and `strategy_id`, plus
`writer_result.json` `version_id`, `run_id`, `strategy_id`, and
`source_run_dir`, to match those exact candidate paths. Read only
`resolved_report/report_review.json`. It must have schema version 1, matching
top-level version/run identity, `hash_algorithm: sha256-file-bytes-v1`, and
required `source_run`, `decision_inputs`, and `reviewed_artifacts` entries for
`research_report.md`, `research_report.html`, `writer_result.json`, and source
`metrics.json`, each with the producer-defined path and digest fields. Require
`source_run` to be
the exact direct candidate run and recompute its canonical current digest from
exactly one matching registry row. Derive and rehash all five external
decision inputs, validate every registered figure and source-script hash, and
rehash all four reviewed artifacts. A robustness mutation followed by an
integrity refresh invalidates the old review because its run digest is stale.

Missing identity, copied or stale reviews, cross-version paths, extra nesting,
and custom-path mismatches block eligibility. Reject
`legacy_root/v001/09_backtests/run_001`,
`research_store/v002/custom/backtests/run_001`,
`research_store/v001/custom/backtests/nested/run_001`, and
`research_store/v001/custom/reports/run_001/copied_report_review.json`.
`research_store/v001/custom/reports/run_001/report_review.json` is acceptable
only when the `v001` manifest resolves the exact custom phases and every
identity/hash check passes.

For every candidate before ranking and again immediately before publication,
perform full manifest-entry integrity validation in addition to the digest-row
check. `require_current_run_digest()` is not the complete current-evidence gate.
Regardless of helper implementation, callers must independently validate the
producer-required manifest inventory and transitive bindings. Parse
`artifact_hashes.json`, derive required entries from its producer schema,
validate every non-metadata entry, and reject unsafe paths, duplicate canonical
targets, symlink escapes, missing required entries, missing or extra governed
targets, malformed digests, and stale entries. Recompute each value with the
producer's artifact-specific algorithm. A mutation without a manifest refresh
must block; a refreshed mutation invalidates old review, lineage, comparison,
and selection bindings.

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

## Responsibilities

- Require a coordinator-journal-confirmed selection policy.
- Check eligible candidates and comparison references.
- Reject copied, stale, cross-version, nested, and custom-path-mismatched report
  reviews before ranking.
- Write final governance artifacts and update `<final_dir>/current_final.json`.

## Report Review Eligibility

An eligible review has status exactly `pass`, verdict exactly `consistent`,
`blocking_findings` exactly empty, `required_report_edits` exactly empty, and
`errors` exactly empty. Treat these fields as one cross-field invariant. Reject
`status: pass` with `verdict: needs_revision` or `verdict: inconsistent`,
`status: pass` with non-empty `blocking_findings`, `status: pass` with non-empty
`required_report_edits`, and `status: pass` with non-empty `errors`, even when
every recorded hash matches. Matching hashes prove identity, not semantic
eligibility.

## Staged Selection Orchestration

The schema-version-2 policy and candidate orchestration in this section is
historical recognition logic. Current production uses the journal-bound schema
version 3 contract below.

The worker owns one schema-version-2 `selection_policy.json` inside the
generated selection directory. It has exactly `schema_version`, `selection_id`,
`hash_algorithm`, `policy_source`, and `policy_payload`. `policy_source` records
either `source: confirmed_payload` with `reference: null` or
`source: hash_bound_reference` with the exact source `{path, sha256}`.
`policy_payload` is exact object equality with the user-confirmed payload and
must retain `confirmed_by_user: true`, confirmation evidence, eligibility,
ranking, and tie breakers. The worker validates the source and then atomically
publishes it inside the generated selection directory; it must not infer policy
fields or publish policy elsewhere.

```json
{
  "schema_version": 2,
  "selection_id": "selection_20260712_180000",
  "hash_algorithm": "sha256-file-bytes-v1",
  "policy_source": {
    "source": "confirmed_payload",
    "reference": null
  },
  "policy_payload": {
    "schema_version": 1,
    "confirmed_by_user": true,
    "confirmation": {
      "source_conversation": "conversation://final-selection-policy",
      "confirmed_at": "2026-07-12T18:00:00Z"
    },
    "eligible_if": {
      "spec_audit": "confirmed",
      "runtime_audit": "pass",
      "reproducibility_audit": "pass",
      "research_audit_fatal": 0,
      "report_review": "pass"
    },
    "rank_by": ["oos_sharpe_ratio", "max_drawdown", "robustness_status", "trade_count"],
    "tie_breakers": ["simpler_spec", "lower_turnover", "lower_cost_sensitivity"]
  }
}
```

The following schema-version-2 envelope is historical recognition input only;
current `prepare_selection` accepts the Round 25 schema-version-3 envelope:

```json
{
  "schema_version": 2,
  "mode": "prepare_selection",
  "selection_id_policy": {
    "source": "generated",
    "selection_id": null
  },
  "selection_policy": {
    "source": "confirmed_payload",
    "payload": {
      "schema_version": 1,
      "confirmed_by_user": true,
      "confirmation": {
        "source_conversation": "conversation://final-selection-policy",
        "confirmed_at": "2026-07-12T18:00:00Z"
      },
      "eligible_if": {
        "spec_audit": "confirmed",
        "runtime_audit": "pass",
        "reproducibility_audit": "pass",
        "research_audit_fatal": 0,
        "report_review": "pass"
      },
      "rank_by": ["oos_sharpe_ratio", "max_drawdown", "robustness_status", "trade_count"],
      "tie_breakers": ["simpler_spec", "lower_turnover", "lower_cost_sensitivity"]
    },
    "reference": null
  },
  "candidate_population": [
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
  ]
}
```

The envelope and every nested object use exactly the shown keys. There is no
implicit default for `selection_id_policy.source`. `source: generated`
requires `selection_id: null`; allocate one new collision-safe id.
`source: provided` requires a valid non-empty id whose selection directory does
not exist and is not a resume alias. Reject every other combination.

Selection ids have one normative grammar:
`\Aselection_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z`. Generated and provided ids use
the same grammar and form one normal direct-child component. Separators,
backslashes, dot segments, absolute forms, drive-qualified forms, Unicode, and
longer values are invalid. Canonicalize `<final_dir>` without symlink
components before allocation; the candidate's resolved parent must equal the
canonical `<final_dir>` exactly, and there may be no symlink parent. Allocate
with exclusive atomic `mkdir`, never an existence check followed by creation.
A provided collision is rejected. A generated collision retries with a fresh
generated id under the final-selection lock and never opens, resumes, removes,
or reuses
the existing directory.

Require
one or more unique candidates, zero-based ordinals, exact identity/scope
equality, direct run and lineage paths, current producer digest, and full
lineage hash. The population has exact ordered equality with the candidates
written to `candidate_set.json`; validation may verify it but must not
rediscover, add, omit, replace, deduplicate, or reorder candidates.

The policy input has exactly `source`, `payload`, and `reference`.
`source: confirmed_payload` carries the exact user-confirmed payload above and
requires `reference: null`; `source: hash_bound_reference` requires
`payload: null` and a current exact `{path, sha256}` reference to bytes with the
same payload schema. The worker must not infer policy fields. Reject unsafe,
malformed, unconfirmed, stale, or cross-selection policy sources.
The `{path, sha256}` form is the hash-bound source reference, and the selector
atomically publishes it inside the generated selection directory after binding.

The only alternative policy input shape is:

```json
{
  "source": "hash_bound_reference",
  "payload": null,
  "reference": {
    "path": "<conversations_dir>/<conversation_id>/confirmations.jsonl",
    "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
  }
}
```


Final selection is one coordinated request with two selector invocations when
multiple candidates do not yet have valid comparison coverage. On
`prepare_selection`, allocate one `selection_id`, validate and atomically
atomically publish the schema-version-2 `selection_policy.json` and immutable
schema-version-2 candidate set inside the generated selection directory, hash
their exact bytes, and
return exactly:

```json
{
  "schema_version": 2,
  "status": "candidate_set_ready",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_refs": [],
  "next_action": "compare_then_resume",
  "blocking_findings": []
}
```

`candidate_set_ready` is a normal nonterminal handoff, not `blocked` or `fail`.
For exactly one candidate, use `next_action: resume_selection` and
`comparison_refs: []`; the coordinator must not invoke the comparator and must
directly resume only after this worker has durably persisted the singleton
comparison-reference ledger. For two or more candidates, use
`next_action: compare_then_resume` and require comparator coverage.
The selector must not write `final_decision.json` and must not update
`current_final.json` in this state. The coordinator routes
`build_selection_comparison` with the same `selection_id` and same exact
candidate-set reference, receives `comparison_ready`, and, without a new user
request, returns this exact handoff:

```json
{
  "schema_version": 2,
  "mode": "resume_selection",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_refs": [
    {
      "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ]
}
```

`resume_selection` must reuse the same selection directory and must not
allocate a new selection id, rewrite the candidate set, or rediscover the
population. It must persist and re-read the exact comparison-reference ledger
before ranking or decision publication. Every selector result has exactly `schema_version`, `status`,
`selection_id`, `selection_policy`, `candidate_set`, `comparison_refs`,
`next_action`, and `blocking_findings`. The allowed statuses are `candidate_set_ready`,
`selected`, `blocked`, and `fail`; allowed next actions are
`compare_then_resume`, `resume_selection`, `restart_selection`, and `none`.

Missing or incomplete comparison coverage returns `status: blocked` with
`next_action: resume_selection`, the same immutable candidate set, and the
validated comparison refs. Stale or invalid candidate-set or transitive
candidate evidence returns `status: blocked` with
`next_action: restart_selection`; never mutate that selection. For every
nonterminal, blocked, or failed result, the prior `current_final.json` remains
unchanged.


The same schema supports a complete one-candidate direct-resume handoff and
complete two-candidate and three-candidate handoffs
from request through pointer publication. Two candidates use the full ordered
projection. Three candidates may use one full comparison or connected
projections such as A-B and B-C; every stage preserves request order,
selection id, candidate-set reference, and exact candidate evidence.

## Comparison Reference Ledger Publication

The Final Selector is the sole producer of `comparison_refs.json`. The
coordinator and router never write `comparison_refs.json`; they only validate
and route handoff values. Write the ledger in the existing selection directory
at `<final_dir>/<selection_id>/comparison_refs.json`. This producer step must
not allocate a new selection id, redirect to another selection, or let an
unlocked coordinator filesystem write stand in for selector publication.

Use these two branches exactly:

1. For exactly one candidate, `prepare_selection` must persist the literal
   UTF-8 bytes `[]`, with no BOM, whitespace, or trailing newline, before
   returning `candidate_set_ready`. Do this after the policy and candidate set
   are validated and published, under `final-selection.lock`, and before the
   coordinator sends the direct `resume_selection` request.
2. For multiple candidates, the coordinator collects the exact
   `comparison_ref` objects from validated `comparison_ready` results. It uses
   comparator dispatch order, not completion order, and requires every result
   to carry the same `selection_id`, selection-policy reference, and exact
   candidate-set reference. It passes that exact ordered array without
   alteration in `resume_selection`. Before ranking or writing
   `final_decision.json`, validate every reference and the complete coverage
   rule against that same `selection_id` and exact candidate-set reference,
   then persist that exact ordered array under `final-selection.lock`. Block an
   incomplete or invalid array before creating a ledger.

The singleton ledger is exactly:

```json
[]
```

The multi-candidate ledger for the running example is exactly:

```json
[
  {
    "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  }
]
```

Encode a non-empty ledger as compact UTF-8 JSON with no BOM or trailing
newline, preserving array order and each entry's `path`, then `sha256`, key
order. Before reading or publishing, revalidate the canonical `<final_dir>`,
the selection directory as its non-symlink direct child, and the target as the
one safe workspace-relative direct regular file named `comparison_refs.json`.
If the target exists, use `lstat` and reject a symlink, non-regular file,
different canonical parent, or aliased target.

While holding the precomputed workspace `final-selection.lock`, create a
restrictively permissioned same-directory temporary regular file with
exclusive no-follow creation, write all canonical bytes, flush and `fsync` the
temporary file, perform atomic `os.replace`, and `fsync` the selection
directory before reporting publication. Clean up only the disposable temporary
file on failure and leave `current_final.json` unchanged.

Retries are immutable and idempotent. If the target is absent, publish it as
above. If it already contains the exact expected canonical bytes, fully
revalidate it, sync the file and parent directory, and treat the retry as an
idempotent no-op. If it contains a stale or different array, malformed bytes,
or even a semantically equal but non-canonical encoding, block or fail that
selection; the selector must not overwrite it and must not allocate a new
selection id as part of the retry.

After either branch publishes or accepts the ledger, `resume_selection` must
re-read its persisted bytes from the validated direct regular file, require
the bytes to be canonical, parse and fully validate the array, and require
request `comparison_refs` to equal the persisted array exactly. Repeat this
read and equality check immediately before decision publication.
`final_decision.comparison_refs` must equal that same persisted array exactly.

On `resume_selection`, acquire or reacquire `final-selection.lock` before this
ledger read and hold the same lock continuously through decision and pointer
publication. Do not release it between multi-candidate ledger publication or
idempotent acceptance, the required re-read, final-decision publication, and
atomic pointer replacement.

Pointer-time validation re-reads `comparison_refs.json` and requires its bytes,
parsed array, and transitive comparison evidence to remain current and to equal
the validated decision array. `current_final.json` does not duplicate the
array; its hash-bound `final_decision` reference transitively binds the same
array only after this equality check passes. A mismatch blocks pointer
replacement and leaves the prior pointer byte-for-byte unchanged.

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

## Candidate Set Schema And Publication

Historical `candidate_set.json` uses schema version 2 and exactly this top-level and
per-candidate field inventory:

```json
{
  "schema_version": 2,
  "selection_id": "selection_20260712_180000",
  "hash_algorithm": "sha256-file-bytes-v1",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidates": [
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
  ]
}
```

Require at least one candidate. `ordinal` is zero-based and must equal the
array position. Each candidate has exactly `ordinal`, `identity`, `primary_run`,
and `lineage_audit`; each nested object has exactly the fields shown. Candidate
identities are unique and have exact ordered equality with the coordinator's
explicit ordered selection population. Registry discovery validates that
population but may not omit, add, replace, deduplicate, or reorder it.
`primary_run` must identify the manifest-resolved direct run and its current
producer digest. `lineage_audit.scope` must equal `identity`, and its direct
governance path and exact current file hash must pass the complete lineage-v2
validator.

The policy reference must equal the preparation result and current policy
artifact exactly, and the policy exact selection id must equal the candidate
set selection id. Reject stale or cross-selection policy. Every resume handoff
must carry the same policy reference.

Validate the complete payload and all transitive evidence, publish it
atomically before ranking, re-read its exact bytes, and compute its full
SHA-256. Both `final_decision.json` and `current_final.json` must contain the
same exact `{path, sha256}` candidate-set reference. Revalidate
`candidate_set.json` immediately before final-decision publication, then
revalidate it again immediately before current-pointer publication; in both
gates require the current bytes, schema, selection id, order, identities,
primary runs, lineage scopes, and transitive hashes to remain unchanged. Do not
mutate or backfill an existing candidate set. Missing, stale, path-only, aliased,
reordered, or post-hash-mutated candidate sets block publication.

## Current Lineage Audit Gate

Build `candidate_set.json` as the selection manifest before ranking. Every
candidate entry must contain exactly one lineage audit reference, with no
fallback glob or latest-timestamp inference. Require exactly one lineage audit
reference for the candidate and require its path to resolve to one direct
regular file under the canonical `<governance_dir>` without a symlink escape.
Reject zero or multiple matching lineage audits, duplicate references, and any
path not owned by the candidate-set manifest.
This is the one unambiguous candidate-set-manifest-owned audit path.

Read the referenced schema-version-2 audit and validate its exact field set,
`hash_algorithm: sha256-file-bytes-v1`, `status: pass`, empty
`blocking_findings`, and exact candidate scope. Before trusting `input_hashes`,
independently derive the closed mandatory set from the candidate manifest and
direct run/report directories: version manifest, strategy spec, spec audit,
compiled plan, runtime audit, artifact hashes, reproducibility audit,
research-bias audit, and report review. Require exact set equality. Reject an
omission, addition, wrong phase, wrong run, duplicate recorded path, duplicate
canonical target, stale hash, unsafe path, symlink escape, or non-exact file.
Reject a lineage audit whose scope does not exactly equal the selected
version_id/run_id. After exact set equality is proven, recompute SHA-256 over
every current `input_hashes` file as the mandatory-input check.

Apply the complete lineage-v2 validator rather than accepting the audit-file
hash or `input_hashes` alone. Recursively revalidate the bound
`report_review.json`, its exact identity and eligibility invariant, current
source-run digest, full manifest-entry integrity validation, exact current five
`decision_inputs`, exact current four `reviewed_artifacts`, and exact current
registered report figure/source-script set and hashes. Do not trust
`status: pass` or the lineage-audit file hash as proof of transitive currency.

Hash the exact current lineage audit bytes after those checks. Copy `scope` and
`input_hashes` exactly into `final_decision.lineage_audit` together with that
single path and hash.

## Final Decision Schema And Publication

The historical decision uses schema version 3 and the historical pointer uses
schema version 2. The following compatibility examples are
historical compatibility examples only. They must not be emitted, selected, or
backfilled under the current contract:

```json
{
  "schema_version": 3,
  "selection_id": "selection_20260712_180000",
  "status": "selected",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "selected_as": "final_research_candidate",
  "hash_algorithm": "sha256-file-bytes-v1",
  "selected_run": {
    "path": "<phase_paths.09_backtests>/run_20260712_173012",
    "digest": "sha256:1111111111111111"
  },
  "report_artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.md",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.html",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/writer_result.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "report_review": {
    "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "lineage_audit": {
    "path": "<governance_dir>/lineage_audit_20260712_175500.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    "scope": {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    },
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
    ]
  },
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "comparison_refs": [
    {
      "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ],
  "blocked_candidates": [],
  "blocking_findings": [],
  "created_by_role": "oxq-final-selector-worker"
}
```

```json
{
  "schema_version": 2,
  "selection_id": "selection_20260712_180000",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "final_decision": {
    "path": "<final_dir>/selection_20260712_180000/final_decision.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

Historical selected `final_decision.json` uses schema version 4 and exactly this field
inventory; do not add aliases or path-only duplicates:

```json
{
  "schema_version": 4,
  "selection_id": "selection_20260712_180000",
  "status": "selected",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "selected_as": "final_research_candidate",
  "hash_algorithm": "sha256-file-bytes-v1",
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "selected_run": {
    "path": "<phase_paths.09_backtests>/run_20260712_173012",
    "digest": "sha256:1111111111111111"
  },
  "report_artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.md",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.html",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/writer_result.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "report_review": {
    "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "lineage_audit": {
    "path": "<governance_dir>/lineage_audit_20260712_175500.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    "scope": {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    },
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
    ]
  },
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "comparison_refs": [
    {
      "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ],
  "blocked_candidates": [],
  "blocking_findings": [],
  "created_by_role": "oxq-final-selector-worker"
}
```

Require `selection_id` and the candidate-set reference to equal the published
candidate set. The selected `{version_id, run_id}` identity must appear exactly
once in that candidate set; `selected_run` must equal that entry's
`primary_run`, and `lineage_audit` must equal the complete validated audit
identified by that entry. A selected identity or evidence object may not be
derived from policy, comparison prose, or an unbound registry row.

Read `<phase_paths.09_backtests>/run_digests.jsonl`, parse every valid entry,
and require exactly one valid matching `run_id` row. Do not use file order or
choose a last matching row; the former "last matching `run_id` entry" rule is
invalid. Require its `artifact_hashes` value to match
`sha256:[0-9a-f]{16}`, recompute the selected run digest from current
`artifact_hashes.json` with the producer's canonical JSON algorithm, and apply
the same semantics as `require_current_run_digest()`. Require equality with
`selected_run.digest` and require its path to resolve to the exact direct run
directory. Recompute the selected run digest only from that validated current
manifest.

Copy the three `report_artifacts` references from the accepted review and
recompute the full sha-256 over the exact current file bytes for each artifact.
After validating review identity, cross-field eligibility, and artifact hashes,
hash the exact current `report_review.json` bytes. After validating the
confirmed policy, hash the exact current `selection_policy.json` bytes. All
file-byte hashes are full lowercase `sha256:<64 hex>` values under
`sha256-file-bytes-v1`.

Validate `comparison_refs.json` as the canonical ledger produced above and
require request `comparison_refs`, the persisted array, and
`final_decision.comparison_refs` to equal exactly. Every
entry has exactly `path` and `sha256`. Require `path` to be a non-empty safe
workspace-relative path with no `..` segment, resolve it canonically, and
require it to resolve to the exact direct
`<comparisons_dir>/<selection_id>/<comparison_id>/comparison_manifest.json` regular file
without a symlink escape. Recompute SHA-256 over the exact current file bytes
and require equality with the entry's full lowercase `sha256:<64 hex>` value.
Validate the referenced `comparison_manifest.json` producer schema;
final-selection consumption requires schema version 2. Validate it as a schema-version-2
comparison manifest. Require its `comparison_id` to equal the direct parent
directory, its exact `selection_id` to equal the resumed selection, and its
exact `{path, sha256}` candidate-set reference to equal the resume request and
validated candidate set. Reject cross-selection substitution even when
identities match. Require at least two unique exact `{version_id, run_id}`
candidate identities. Candidate evidence must equal the exact ordered
projection of complete candidate-set entries, including ordinal, identity,
`primary_run`, and `lineage_audit`; aliases and partial projections are
invalid. For every `candidate_evidence`
entry, including non-selected candidates, invoke the complete lineage-v2
validator and recursively revalidate the bound `report_review.json`, full
manifest-entry integrity validation, decision inputs, reviewed artifacts, and
registered report assets. Do not trust `status: pass` or the lineage-audit file
hash. Then recompute every candidate run digest and lineage-audit file hash.
Recompute the four mandatory output file
hashes and require figure references to equal the exact current regular-file
set under `figures/`. Reject omitted, extra, duplicate, stale, non-candidate,
unsafe, symlinked, or path-only evidence. Reject reports, figures,
comparability audits, malformed manifests, and manifests for other candidates
as top-level comparison references.

Comparison population must be a subset of the hash-bound candidate set, and
each comparison's ordered identities must equal their projection from candidate
order. comparison_refs must be non-empty whenever candidate_set has multiple
candidates. With exactly two candidates, every referenced comparison must
exactly equal the two-candidate ordered population. With more than two
candidates, every referenced population must contain at least two unique
candidates, the union must exactly equal the complete candidate-set population,
and the comparison coverage graph must be connected, with candidates as nodes
and co-membership in a comparison as edges. Reject an omitted candidate, an
unrelated replacement, a duplicate, an order mismatch, a disconnected group,
or any comparison population not derived from the current candidate set. A
single-candidate set alone may use `comparison_refs: []`. The selected identity
must appear exactly once in the union of validated comparison populations. The
selector must not require the selected identity in every referenced comparison;
a connected A-B/B-C chain may select A, B, or C when all other gates pass.
The old requirement that the selected version_id/run_id identity appears
exactly once in each referenced comparison is invalid under this rule.

Reject a legacy comparison manifest that lacks complete candidate or output
evidence even when its own manifest hash matches. Do not infer or backfill the
missing bindings; regenerate the comparison from current evidence before
selection can cite it.

Schema-version-1 comparison manifests are historical only because they cannot
bind a selection id and candidate-set bytes. Do not infer or backfill that
binding. Regenerate from the exact current candidate set and transitive
evidence as a new schema-version-2 comparison manifest before selection can
cite it.

For `blocked` or `fail`, use the same exact top-level keys. Set
`selected_version_id`, `selected_run_id`, `selected_as`, `selected_run`, and
`report_review` to JSON `null`, set `lineage_audit` to JSON `null`, set
`report_artifacts` to `{}`, require a
non-empty `blocking_findings`, and retain a hash-bound `selection_policy` when
one exists; it may be JSON `null` only when no policy artifact exists. Retain
the hash-bound `candidate_set` when a valid schema-version-2 candidate set was
published; it may be JSON `null` only when no candidate set could be safely
published. Never
write or update `current_final.json` for a blocked or failed decision.

Validate the complete in-memory `final_decision.json` payload before any write,
including exact key sets, types, identities, paths, digest formats,
lineage-audit equality, comparison-reference equality, and hashes; then
publish it atomically.

## Workspace-Scoped Selection Publication Lock

Use `governing_workspace_root(subject)` with the canonical subject and nearest
ancestor `.open-xquant/workspace.yaml`, then precompute
`final_selection_lock_path(subject)`. A valid non-governed subject has no lock;
malformed or unsafe governed configuration must fail closed. Use
`hold_final_selection_lock(precomputed_path)` for the canonical
`<workspace_root>/.open-xquant/locks/final-selection.lock`.
Resolve `<workspace_root>` first, create only the real `.open-xquant/locks`
directory with restrictive permissions, reject symlinked path components, and
take an exclusive advisory file lock on the persistent regular lock file. Never
unlink the lock file. The Final Selector owns the lock lifecycle,
comparison-reference ledger, and pointer publication. All governed writers of
selection-transitive evidence must take
this same exclusive lock before replacing policy, candidate-set, lineage,
run-inventory, comparison, comparison-output, comparison-reference ledger,
review, report, decision, or pointer bytes that an active selection can
consume.

The selection lock is the last lock acquired in the workspace-wide lock order.
A participant must release every run and registry lock before
`final-selection.lock` and must not acquire another lock while holding it. The
selector itself acquires only this precomputed lock for comparison-reference
ledger publication and final decision/pointer publication. The coordinator and
router never acquire it; they
serialize handoffs and require workers to honor the protocol.

Before taking it, complete `validate_run_artifact_inventory(run_dir)`,
`require_current_run_digest(run_dir)`, and registry validation, retain their
immutable results, and release their locks. For pointer publication, acquire it
before direct schema-version-5 transitive revalidation and hold it continuously
through all validation, unchanged-byte
checks, comparison-ledger equality, pointer construction, validation, and
atomic `current_final.json`
replacement. During the full validator, snapshot the exact bytes at each
dependency read, including policy, candidate set, `comparison_refs.json`, every lineage and run
inventory input, every comparison manifest and output, every review and report
artifact, and the published decision. After validation, perform an
unchanged-byte sweep: re-read every dependency in that closed snapshot set and
require byte-for-byte equality. This detects mutation after every dependency
read, not only mutation of the decision or last-read artifact.

Before any pointer work, retain the previous pointer byte-for-byte for recovery
classification. Build and validate the new pointer in a same-directory
temporary regular file, flush and `fsync` it, atomically replace
`current_final.json`, then `fsync(<final_dir>)` after atomic replacement before
reporting success. A pre-rename failure leaves the prior pointer unchanged. A
post-rename directory-sync failure makes the publication outcome indeterminate
and enters the recovery protocol below; it must not claim that the prior
pointer is unchanged. Keep the advisory lock until success or recovery state is
durably classified, and never unlink the lock file.


Immediately before `current_final.json` publication, run full schema-version-5
decision validation again against the published decision and current evidence.
Inside the final-selection lock, perform only direct byte snapshots, direct
parsing/hash/path checks, the unchanged-byte sweep, and atomic pointer
replacement. It must not invoke `validate_run_artifact_inventory`, must not
invoke `require_current_run_digest`, and must not invoke run-locking APIs.
Re-read and validate the selection policy and its user-confirmed binding;
`comparison_refs.json`, whose canonical parsed array must equal
`final_decision.comparison_refs` exactly; all comparison manifests and every required comparison
output, including the exact figure set; the schema-version-3 candidate set;
every schema-version-3 comparison reference; every candidate's
complete lineage-v3 validator and report-review bindings; every directly
snapshotted current run digest and the preflight inventory result checked
against direct bytes; and
the selected report review and report artifacts. Recompute all schema, identity,
path, population, inventory, output, and hash invariants.

After the full gate passes, re-read the published `final_decision.json` bytes.
This must re-read the decision bytes after that validation; require them to be
byte-for-byte unchanged from the bytes just validated, and only then compute
their full SHA-256. A policy, comparison manifest, required
comparison output, candidate, lineage, inventory, report, or decision mutation
blocks publication. Build exactly this pointer:

```json
{
  "schema_version": 3,
  "selection_id": "selection_20260712_180000",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "final_decision": {
    "path": "<final_dir>/selection_20260712_180000/final_decision.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

Require pointer identities to equal the decision identities, the decision
path/hash to identify that exact decision, and the candidate-set reference to
equal the validated decision reference exactly. Require the pointer's
hash-bound decision bytes and pointer-time ledger validation to transitively
bind the same persisted comparison-reference array. Validate and atomically publish
the pointer. Write `current_final.json` last. If any pointer-time validation,
hash, or write fails, do not update `current_final.json`; leave the prior
`current_final.json` unchanged and do not expose a partially validated pointer.

## Migration

Schema-version-1 prepare requests and candidate sets are historical only. They
lack selection-bound policy evidence and must not be backfilled or resumed.
Use `restart_selection` after remediation: send a new schema-version-2 prepare
request only for historical-validator reproduction. Current remediation sends
a schema-version-3 prepare request with a fresh coordinator confirmation event,
allocates a new selection id, publishes policy/candidate-set v3, and regenerates
comparison-v3 evidence.

Under the previous schema-version-3 contract, schema-version-1 or
schema-version-2 `final_decision.json` artifacts are historical only.
Treat schema-version-1 through schema-version-3 `final_decision.json`, any
decision without a hash-bound schema-version-2 `candidate_set` and exact
selection-bound policy, path-only
policy references, and schema-version-1 or schema-version-2 current pointers as
historical only. Never infer or backfill their missing hashes in place. In
particular, do not backfill `lineage_audit` in place. Schema-version-4 decisions
and schema-version-3 pointers that bind candidate-set v1 or unbound policy are
historical despite their unchanged outer versions. Rerun final selection from
the exact current candidate-scoped lineage audit, immutable report/review
revisions, comparison manifests, and journal-confirmed policy to publish a
schema-version-5 decision and schema-version-4 current pointer.
If regeneration blocks or fails, leave the existing current pointer unchanged
until `current_final.json` is replaced last after full validation.

## Red Lines

The schema-version-2 selector result envelope above is a historical
compatibility example only. Current selector result handoffs use schema
version 3, while a selected result publishes the schema-version-5
`final_decision.json` described below. `restart_selection` allocates a new
selection id and must not reuse or overwrite the failed selection directory.

- Do not run backtests.
- Do not modify candidate artifacts.
- Do not select without a validated coordinator-owned confirmation journal event.
- Do not infer eligibility for a legacy review missing current identity hashes.
- Do not call the selected candidate investable.

## Round 25 Current Selection Contract

The schema-version-2 prepare/candidate examples and schema-version-4 decision
above are historical compatibility evidence. Current preparation is schema
version 3. The Coordinator is the sole producer of selection-policy events in
the append-only `confirmations.jsonl`; it writes under the persistent sibling
`confirmations.jsonl.lock`. The event hash covers exact raw JSONL line bytes
excluding the terminating LF. Caller self-attestation is invalid, including an
inline `confirmed_by_user` boolean.

Validate the exact journal `path`, `event_id`, one-based `line_number`,
`event_hash`, `decision`, `selection_request_id`, and `policy_hash`. Reject a
fabricated event, stale request/event, mismatched policy hash or decision,
duplicate/conflicting event id, malformed line, replacement, or non-coordinator
producer. Release the journal lock before `final-selection.lock`; direct
pointer-time validation re-reads that same line unchanged.

```json
{
  "schema_version": 3,
  "mode": "prepare_selection",
  "selection_request_id": "selection-request-20260712-1",
  "selection_id_policy": {
    "source": "generated",
    "selection_id": null
  },
  "selection_policy": {
    "payload": {
      "schema_version": 2,
      "eligible_if": {
        "spec_audit": "confirmed",
        "runtime_audit": "pass",
        "reproducibility_audit": "pass",
        "research_audit_fatal": 0,
        "report_review": "pass"
      },
      "rank_by": ["oos_sharpe_ratio", "max_drawdown", "robustness_status", "trade_count"],
      "tie_breakers": ["simpler_spec", "lower_turnover", "lower_cost_sensitivity"]
    },
    "policy_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "confirmation_event": {
      "path": "<conversations_dir>/<conversation_id>/confirmations.jsonl",
      "event_id": "selection-policy-confirmation-1",
      "line_number": 1,
      "event_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      "decision": "confirmed",
      "selection_request_id": "selection-request-20260712-1",
      "policy_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    }
  },
  "candidate_population": [
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
    }
  ]
}
```

The canonical policy hash uses `sha256-canonical-json-v1` and must equal the
request and event. Publish a schema-version-3 policy that binds the event, then
publish exactly this current candidate set:

The policy artifact has exactly `schema_version`, `selection_id`,
`selection_request_id`, `hash_algorithm`, `policy_hash_algorithm`,
`policy_hash`, `confirmation_event`, and `policy_payload`; its event and payload
equal the accepted request exactly.

```json
{
  "schema_version": 3,
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "hash_algorithm": "sha256-file-bytes-v1",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  },
  "candidates": [
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
    }
  ]
}
```

Require the exact `{path, sha256}` report-revision reference and exact
`{path, sha256}` review-revision reference at every gate. These identify one
immutable report revision and immutable review revision. Never overwrite or
substitute evidence reachable from any prior selection. Recursively validate
`chart_build_result.json`: requested/applicable/generated/skipped set
invariants, closed skip reason codes, exact `{path, sha256}` manifest reference,
figure hashes, safe package-relative `source.script`, full lowercase
`source.script_sha256`, and recomputed script SHA-256. Script mutation requires
`restart_selection`.

At each gate, recompute the script SHA-256 from exact current bytes.

`final_decision.json` is the sole canonical decision artifact. Current decision
schema version 5 and pointer schema version 4 bind the selection request,
confirmation-backed policy, candidate set, exact report/review revisions,
lineage, and schema-version-3 comparisons. No companion decision file is
produced.

```json
{
  "schema_version": 5,
  "status": "selected",
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "selected_version_id": "v001",
  "selected_run_id": "runA",
  "selected_as": "final_research_candidate",
  "selected_run": {
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
  },
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_refs": [],
  "blocking_findings": []
}
```

```json
{
  "schema_version": 4,
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "selected_version_id": "v001",
  "selected_run_id": "runA",
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "final_decision": {
    "path": "<final_dir>/selection_20260712_190000/final_decision.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
  }
}
```

For `candidate_scoped_historical_report_revision`, use the explicit inactive
version, current-state guard, fresh `report_revision_id`, fresh
`review_revision_id`, fresh lineage audit, and fresh comparison id. Must not
reactivate the inactive version and must not overwrite prior evidence. Route
`write -> review -> lineage -> comparison -> reselection`, then
`restart_selection`; prior revision bytes remain reachable.

After rename, `fsync(<final_dir>)` after atomic replacement is mandatory. A
post-rename directory-sync failure means publication outcome is indeterminate
and must not claim that the prior pointer is unchanged. Recover under
`final-selection.lock`: exact new pointer bytes are revalidated and directory
synced, exact prior pointer bytes trigger a full retry, and any other bytes
block as corruption. Parent fsync is required, never prohibited.

Atomic replacement is not the final fallible publication operation because
the required parent fsync follows it. Every pre-rename failure must leave the
prior `current_final.json` byte-for-byte unchanged; post-rename sync failure is
classified separately.
