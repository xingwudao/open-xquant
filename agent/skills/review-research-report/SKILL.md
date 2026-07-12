---
name: review-research-report
description: >-
  Use when reviewing a completed open-xquant research report for decision
  consistency, artifact fidelity, audit/robustness interpretation, chart
  narrative quality, and semantic issues that deterministic report QA should not
  judge.
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

# Research Report Reviewer

Round 26: review consumes sealed revisions only. Missing-chart retry uses a fresh
unsealed report revision attempt and must seal only after chart completion.
Historical refresh follows `write -> review -> lineage -> prepare new selection ->
comparison -> resume` with fresh selection and candidate hashes.

Use this skill after `write-research-report` has sealed the supplied immutable
report revision, HTML has been rendered from the same Markdown, and
deterministic artifact QA has run against that exact candidate directory.

`oxq report qa` checks files, dates, registered image references, manifest
integrity, hashes, and dimensions. This skill reviews the research meaning:
whether the report's decision, explanation, risk language, and chart narrative
are faithful to the artifacts.

## Version-Governed Review Gate

For normal phase work, read `current.json` and use `active_version` as
`version_id`. Read the source run package from:

```text
<phase_paths.09_backtests>/<run_id>/
```

Read final report artifacts from:

```text
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.md
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.html
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/writer_result.json
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/candidate_manifest.json
```

Write the semantic review only to:

```text
<phase_paths.10_reports>/<run_id>/reviews/<review_revision_id>/report_review.json
```

Do not write root-level `report_review.json`.
The old direct path `<phase_paths.10_reports>/<run_id>/report_review.json` is a
historical recognition target only and must not be replaced.

### Historical Schema-1 Handoff Recognition

Any explicit inactive candidate whose review is missing or stale may be
reviewed while another version is active, including a candidate with a stale
current-schema `report_review.json`. The coordinator must use exactly this
field inventory only when recognizing an old handoff. It is not a current
producer request and must be translated to
`candidate_scoped_historical_report_revision` before any write:

```json
{
  "mode": "candidate_scoped_historical_rereview",
  "version_id": "v001",
  "run_id": "runA",
  "current_state_guard": {
    "path": "current.json",
    "active_version": "v002",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "reason": "stale_report_review",
  "requested_by_role": "oxq-coordinator"
}
```

Validate the exact handoff fields before any candidate read. Require `reason`
to be `missing_report_review` or `stale_report_review` and require direct
evidence that the exact target review is absent or fails the normal current
identity, schema, inventory, digest, decision-input, reviewed-artifact, or
report-asset checks. Artifact age and schema age are not eligibility
conditions. Require the guard path to be exactly `current.json`, hash its exact
initial bytes, and require its active version and hash to match the handoff.
Set `expected_version_id` and
`run_id` only from the explicit handoff, then resolve
`<version_root>/v001/version_manifest.json` and that manifest's exact run and
report phase paths. Apply the same canonical containment, direct-child,
identity, full-integrity, and pre/post review checks as normal mode. In the
illustrated active-v002/historical-v001 case, the report reviewer must not
change `current.json`, any version/phase state, or the active run; this mode
does not reactivate `v001`.

Rerun deterministic report QA against the exact historical report package, but
do not publish or replace the direct `report_review.json`. Publish a fresh
schema-version-2 review revision against a fresh immutable report revision as
defined below; do not backfill fields or hashes into the old payload. Re-read
and rehash `current.json` immediately before and after publication and require
the guard to remain unchanged. This remains a candidate-scoped historical
re-review even though its current producer writes immutable candidate and
review revisions. The coordinator then follows this regeneration order:
rerun artifact lineage audit for `v001/runA`, regenerate every comparison that
cites its old review or lineage evidence, and rerun final selection into a new
selection directory. Historical re-review completion is evidence regeneration,
not active phase completion, so it must not trigger version-manager state
updates.

## Inputs

Read:

- `research_report.md` and `research_report.html`
- `strategy_spec.yaml`, especially `decision_policy`
- `spec_audit.json`, including field provenance, recipe matches,
  component audits, missing requirements, agent-added fields, contradictions,
  and blocking findings
- `compiled_plan.json` and `runtime_audit.json`
- `metrics.json` and `execution_assumptions.json`
- `research_bias_audit.json` and `reproducibility_audit.json`
- `robustness.json`, including perturbation and regime analysis sections
- `report_assets/manifest.json`, captions, source scripts, and figure files
- `equity_curve.csv`, `benchmark_curve.csv`, `trades.csv`, `positions.csv`,
  and `target_weights.csv` when needed to verify a claim
- the deterministic `oxq report qa` output
- optional/advisory numeric QA output when available, especially any
  `numeric_claim_unverified` warnings
- `writer_result.json`, especially the resolved `language`

Do not rewrite metrics, audits, robustness output, backtest artifacts, Markdown,
or HTML. This skill reviews the report and writes `report_review.json`; it does
not edit the report. If a report edit is needed, return a blocking finding for
the upstream writer phase.

## Review Artifact Identity

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

The review is valid only for the exact final evidence it inspected. Before
semantic review, require exactly one valid matching `run_id` row in
`<phase_paths.09_backtests>/run_digests.jsonl`, recompute the producer's
canonical current run digest from the direct run's current
`artifact_hashes.json`, and require equality. This has the same row-cardinality
and current-manifest semantics as `require_current_run_digest()`.

`require_current_run_digest()` is not the complete current-evidence gate.
Regardless of helper implementation, callers must independently validate the
producer-required manifest inventory and transitive bindings. Before semantic review and again immediately before publication,
perform full manifest-entry integrity validation. Parse `artifact_hashes.json`
as an object; derive all required entries from its producer schema; and validate
every non-metadata entry. Each entry must name a safe relative current regular
file below the exact direct run with no duplicate canonical target or symlink
escape. Recompute every value with the producer's artifact-specific algorithm,
including canonical-JSON exclusions, and require equality. Reject missing
required entries, missing or extra governed targets, malformed digests, and
stale entries.

A mutation without a manifest refresh must block even though the canonical
manifest digest still matches the registry. A mutation followed by an integrity
refresh changes the manifest and run digest and therefore invalidates the old
review. Both the manifest-entry check and the digest-row check are mandatory at
the pre-review and post-review gates; neither substitutes for the other.

The canonical current run digest binds decision-critical files inside the run
package, including metrics, execution assumptions, reproducibility,
research-bias, and robustness evidence. Hash every decision-critical input
outside that package separately under `decision_inputs`: the exact candidate
strategy spec, spec audit, compiled plan, runtime audit, and report-assets
manifest. Validate the manifest's registered figure and source-script hashes
against every current registered file; reject an omitted, extra, duplicate,
missing, or stale registered asset.

After all semantic checks and immediately before writing `report_review.json`,
repeat the run-digest check, re-read every `decision_inputs` file and registered
asset, and re-read the final bytes of the Markdown, HTML,
`writer_result.json`, and source `metrics.json`. Compute lowercase full SHA-256
over exact file bytes, with no JSON normalization, for `decision_inputs` and
`reviewed_artifacts`. Record the validated run path/digest as `source_run`.
File-byte hashing uses `sha256-file-bytes-v1`.

Set top-level `version_id` and `run_id` from the resolved version and direct
report/run directories, not from report content. Each recorded `path` must be
the exact safe workspace-relative path that was read. Any evidence change
after deterministic QA or during review blocks and restarts QA/review. In
particular, a robustness mutation followed by an integrity refresh changes the
canonical run digest and invalidates the old review even when the report,
writer result, and metrics bytes did not change.

## Governed Publication Lock

Publish the fresh `reviews/<review_revision_id>/report_review.json` through
`publish_report_artifacts(report_dir, artifacts, *, lock_subject=None)`, where
`report_dir` is the run-level report directory and the mapping key contains the
revision path. The
mapping uses safe relative keys and complete `bytes`; `None` deletes a target.
When review output depends on current report bytes, pass a callable:
the callable builder executes under the final-selection lock, performs the
baseline check, and commits an atomic all-or-rollback batch. Direct path writes
and shell redirection are forbidden.

For an export whose `report_dir` is outside the governed workspace, pass
`lock_subject=source_run_dir`. If review construction needs a coherent run
lock, wrap publication with `run_digest_transaction(source_run_dir)`; the
runtime acquires the run lock first and the final-selection lock second. Do not
pre-acquire the final lock or invoke a second run-locking API from the callable
builder.

## Review Checklist

1. Decision consistency.
   - Extract the executive decision: `REJECT`, `NO EVIDENCE`, `WATCHLIST`, or
     `PAPER TRADING CANDIDATE`.
   - Evaluate `decision_policy` against `metrics.json`; for example, if
     `reject_if.max_drawdown_lt` is breached, the report must not conclude
     `WATCHLIST` or `PAPER TRADING CANDIDATE`.
   - Treat fatal audits, failed reproducibility, missing OOS evidence, or
     fragile/error robustness as blockers unless the report explicitly rejects
     or withholds promotion.

2. Audit and robustness fidelity.
   - Confirm reproducibility status, research-bias fatal/warning counts, and
     robustness status are stated accurately.
   - Confirm material warnings are explained, not summarized away as
     "风险可控" or "审计通过".
   - Check that perturbation and regime analysis findings are represented when
     they affect the decision.
   - Confirm `spec_audit.json` conclusions are represented, including
     unconfirmed defaults, component provenance warnings, selected recipes, and
     any unresolved blocking questions.

3. Numeric warning triage.
   - Do not treat every `numeric_claim_unverified` as a report error.
   - Classify each important warning as one of:
     `likely report error`, `facts registry gap`, `precision/format issue`, or
     `low-impact appendix claim`.
   - Escalate executive-decision, core metric table, drawdown, Sharpe, return,
     benchmark, and OOS trade-count mismatches above appendix parameter values.

4. Evidence chain and structure.
   - The report must put conclusion first, then evidence, then risks and next
     actions.
   - It must include strategy context, configured end date, effective last
     trading day, costs, benchmark-relative behavior, drawdown, trade count,
     audit status, and robustness interpretation.
   - Claims like "OOS 优于 IS 表明没有过拟合", "方向正确", "风险可控", or
     "适合 paper trading" require explicit artifact-backed reasoning.
   - Confirm the report language matches `writer_result.json.language`. If no
     explicit user language was requested, the report must be Chinese and must
     not unexpectedly switch to an all-English report.

5. Chart review.
   - A final research report without registered chart assets is a blocker;
     return the report to `build-report-charts` before approving it.
   - Inspect registered figures when they support key conclusions.
   - Check that charts are not blank, nearly blank, unreadable, mislabeled, or
     purely decorative.
   - Check that figure order follows the Canonical Report Chart Order from
     `build-report-charts` unless the user explicitly requested a different
     sequence.
   - Check that chart style is consistent across figures: same light report
     background, grid treatment, typography scale, palette family, and
     source-artifact caption discipline.
   - Check that local-language chart labels are readable when the report
     language is local-language; concise English figure labels are acceptable
     only when CJK font rendering is unavailable and captions remain local.
   - Captions must identify source artifacts and avoid unsupported conclusions.

6. Markdown/HTML semantic consistency.
   - Confirm HTML appears to represent the same report as Markdown.
   - Check that section order, figure order, captions, and key paragraphs are
     not missing or reordered in a way that changes meaning.

## Historical Schema-1 Output

The direct-path schema-version-1 JSON below is retained for historical
recognition. Current output uses the schema-version-2 immutable review revision
defined later:

```json
{
  "schema_version": 1,
  "version_id": "v001",
  "run_id": "run_001",
  "hash_algorithm": "sha256-file-bytes-v1",
  "source_run": {
    "path": "<phase_paths.09_backtests>/<run_id>",
    "digest": "sha256:1111111111111111"
  },
  "status": "pass",
  "verdict": "consistent",
  "findings": [],
  "blocking_findings": [],
  "required_report_edits": [],
  "reviewed_artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/<run_id>/research_report.md",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/<run_id>/research_report.html",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/<run_id>/writer_result.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "metrics.json": {
      "path": "<phase_paths.09_backtests>/<run_id>/metrics.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "decision_inputs": {
    "strategy_spec.yaml": {
      "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
      "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    "spec_audit.json": {
      "path": "<phase_paths.06_spec_audit>/spec_audit.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    },
    "compiled_plan.json": {
      "path": "<phase_paths.07_compile_preview>/compiled_plan.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "runtime_audit.json": {
      "path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    "report_assets/manifest.json": {
      "path": "<phase_paths.10_reports>/<run_id>/report_assets/manifest.json",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    }
  },
  "errors": []
}
```

All shown top-level fields are required. `source_run` has exactly `path` and
`digest`; `decision_inputs` has exactly the five shown keys and each value has
exactly `path` and `sha256`. `status` is one of `pass`, `blocked`,
or `fail`; `verdict` is one of `consistent`, `inconsistent`, or
`needs_revision`. The four `reviewed_artifacts` keys and each nested `path` and
`sha256` field are required for every result. For an unavailable file, retain
its expected path, set `sha256` to JSON `null`, put the details in `errors`, and
never substitute a stale digest. An unavailable artifact by itself requires
`status: blocked`; use `fail` only when the completed evidence independently
proves a semantic inconsistency. A passing review requires four full 64-hex
digests.

## Result Invariants

`status: pass` if and only if `verdict: consistent`, `blocking_findings` is
empty, `required_report_edits` is empty, `errors` is empty, the source run has
one canonical current digest, all five decision inputs and all registered
assets are current, and all four reviewed artifacts have current full digests.
Advisory `findings` may be non-empty, but they must not describe a blocking
report defect or required edit.

`status: blocked` requires `verdict: needs_revision` and at least one of
`blocking_findings`, `required_report_edits`, or `errors` to be non-empty. Use
this combination when the report can be revised or required evidence/inputs
must be restored before semantic review can pass. Every null artifact digest
requires a corresponding `errors` entry.

`status: fail` requires `verdict: inconsistent` and non-empty
`blocking_findings`. Use this combination for a completed review that proves a
decision-critical semantic inconsistency. `required_report_edits` may describe
the necessary correction and `errors` remains empty unless artifact inspection
itself also failed.

Validate these field and cross-field invariants before writing
`report_review.json`. Never emit `pass` with `needs_revision` or `inconsistent`,
never emit `blocked` or `fail` with `consistent`, and never rely on matching
hashes to excuse a contradictory status/verdict/list combination.

Existing schema-version-1 report reviews are not grandfathered around these
invariants. A version-1 review remains usable only when it already contains the
complete current `source_run`, `decision_inputs`, reviewed-artifact, registered
asset, status, verdict, and list evidence required by this contract. Rerun
semantic review for a missing or contradictory result; do not rewrite its
fields or infer eligibility downstream.

Include:

- Final decision verdict: `consistent`, `inconsistent`, or `needs revision`.
- Blocking issues first, each tied to artifact evidence.
- Warning triage summary, separating report defects from facts registry gaps.
- Required report edits, if any.
- Follow-up framework improvements, such as facts registry fields that should
  later move into deterministic QA.

## Red Lines

- Do not approve a report whose decision conflicts with `decision_policy`.
- Do not approve positive risk language that omits fatal or material warnings.
- Do not approve a report that omits unresolved `spec_audit.json` blockers or
  presents non-canonical component choices as confirmed.
- Do not let many low-value numeric warnings hide a decision-critical mismatch.
- Do not use chart aesthetics as evidence of strategy quality.
- Do not claim chart text is readable solely because plotting code configured
  styling; inspect the final rendered figure when readability matters.
- Do not edit `research_report.md` or `research_report.html`; write
  `report_review.json` instead.

## Current Immutable Review Revision

The selectable producer target is an immutable review revision at
`<phase_paths.10_reports>/<run_id>/reviews/<review_revision_id>/report_review.json`.
It reviews exactly one immutable report revision identified by the exact
`{path, sha256}` report-revision reference to
`candidates/<report_revision_id>/candidate_manifest.json`. The coordinator
supplies fresh ids matching `\Areport_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z` and
`\Areview_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z`. Create the review directory
exclusively and publish once. Never overwrite, delete, rename, merge, or repair
an immutable report revision, an immutable review revision, or evidence
reachable from any prior selection.

Before semantic review and again inside the publication callable, recursively
validate the candidate manifest, `chart_build_result.json`, and asset manifest.
Require requested/applicable/generated/skipped set invariants, closed skip
reason codes, the exact `{path, sha256}` manifest reference, every generated
figure hash, each safe package-relative `source.script`, and its full lowercase
`source.script_sha256`. Recompute the script SHA-256; a script mutation blocks
the review. Review paths must resolve beneath the exact report revision, never a
mutable run-level working head.

Current schema-version-2 `report_review.json` uses these required fields; the
existing schema-version-1 direct-path example is historical only and must not
be overwritten or selected by a current workflow:

```json
{
  "schema_version": 2,
  "version_id": "v001",
  "run_id": "runA",
  "review_revision_id": "review_20260712_181500",
  "report_revision": {
    "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/candidate_manifest.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "hash_algorithm": "sha256-file-bytes-v1",
  "source_run": {
    "path": "<phase_paths.09_backtests>/runA",
    "digest": "sha256:1111111111111111"
  },
  "status": "pass",
  "verdict": "consistent",
  "findings": [],
  "blocking_findings": [],
  "required_report_edits": [],
  "reviewed_artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/research_report.md",
      "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/research_report.html",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/writer_result.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "metrics.json": {
      "path": "<phase_paths.09_backtests>/runA/metrics.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    }
  },
  "decision_inputs": {
    "strategy_spec.yaml": {
      "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "spec_audit.json": {
      "path": "<phase_paths.06_spec_audit>/spec_audit.json",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    "compiled_plan.json": {
      "path": "<phase_paths.07_compile_preview>/compiled_plan.json",
      "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
    },
    "runtime_audit.json": {
      "path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
      "sha256": "sha256:8888888888888888888888888888888888888888888888888888888888888888"
    },
    "chart_build_result.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/chart_build_result.json",
      "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
    },
    "report_assets/manifest.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/report_assets/manifest.json",
      "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }
  },
  "errors": []
}
```

`candidate_scoped_historical_report_revision` supersedes the old re-review
mode for new writes. It accepts an explicit inactive version/run, an exact
current-state guard, a base report-revision reference when one exists, a fresh
`report_revision_id`, and a fresh `review_revision_id`. The writer first seals
the new candidate; this reviewer then writes the new review and must not
reactivate the inactive version, mutate active governance state, and must not overwrite
any old revision. Return both exact references for the guarded
`write -> review -> lineage -> comparison -> reselection` workflow; prior
revision bytes remain reachable.

Accept only the coordinator's closed schema-version-1 historical handoff; reject
aliases, extra keys, missing base/guard fields, and unknown reason codes.
