# SDD ledger — plan: docs/superpowers/plans/2026-08-26-freeze-quant-operator-contract-v1.md

## Pre-flight scan

| Task | Produces / consumes | Self-consistency | Ruling |
| --- | --- | --- | --- |
| Task 1 | Produces frozen schemas, fixtures, policy, and tests from the existing Markdown specification. | The tests exercise JSON Schema behavior and duplicate-key behavior; file paths and fixture values match the schemas required by the task. | Clean; proceed. |

Ruling: The plan changes human prose and machine-readable contract files in one task because they form one indivisible versioned contract surface. Splitting them would allow an intermediate commit where prose and schemas disagree. Cost if wrong: a larger single review diff.

Ruling: The eQuant SMA example uses operator version `1.0.0`, not the original brief's `0.2.0`, because the normalized certified distribution starts a stable semantic contract and the frozen rules require a major version for changed output/default semantics. Cost if wrong: eQuant must issue a new major version before its first certification.

Task 1: minor (deferred): Add focused negative tests for every repaired frozen-schema boundary; final review must verify whether the fix tests make this obsolete.

Task 1: fix round 1/5 started. Open findings: manifest cannot express required input mutation/data requirements/availability derivation/explicit-fill details; SemVer regex admits numeric prerelease leading zeroes; identity patterns do not precisely separate dotted operator IDs from Python module/callable syntax; parameter constraints/defaults are under-governed; QuantPanel reference checks allow record fields absent from `columns`.

Ruling: Python `module` names may be dotted import paths whose individual segments are lowercase snake_case; `callable` is one lowercase snake_case identifier; `operator_id` is lowercase dot-separated kebab-style segments without underscores. Cost if wrong: a provider using underscore operator-ID segments or a non-dotted module restriction will require a contract-major change or explicit migration.

Task 1: fix round 1/5 (2 addressed, 3 open; commits e7c4761..ee1b5a9). Open: constant fill can omit its value; SemVer accepts a final newline under validator regex behavior; module/callable patterns accept trailing or consecutive underscores.

Task 1: fix round 2/5 (3 addressed, 0 open; commits ee1b5a9..876ef27).

Task 1: minor resolved: fix rounds expanded the negative suite to 35 focused contract tests covering the repaired boundaries.

Task 1: complete (commits 5effd4e..876ef27, review clean).

Final review: 0 Critical, 6 Important, 1 Minor. One final fix wave started for published cross-field validation, sorting declarations, parameter/request constraints, digest profiles, random seed binding, v1 compatibility direction, and full source commit identity.

Ruling: Publish the semantic validator under `contracts/quant-operators/`, outside the `oxq` SDK. Contract tests may execute it, but no `oxq`, SDK, backtest, or report code may run. Cost if wrong: consumers must vendor a Python reference implementation or reimplement the normative conformance behavior in another language.

Ruling: Remove `manifest_digest` from the manifest body and bind the SHA-256 of exact UTF-8 manifest file bytes in the external compatibility/certification record. Cost if wrong: semantically identical but byte-different manifest serialization has a different digest and must be treated as a distinct binding artifact.

Ruling: v1 compatibility is validator-backward compatibility: every instance accepted by an older v1 schema release remains accepted by newer v1 releases. New optional fields require a new schema release/digest and are not forward-compatible with older consumers; bindings pin that release. Operator formula/default/output changes follow provider SemVer and recertification, not contract-major versioning. Cost if wrong: older consumers may require explicit schema upgrades before accepting newer provider manifests.

Final fix wave: complete in `d7b6e987d7b09f376c61c2167c9c209250ba5f96`; five recorded RED/GREEN groups and final `118 passed`. Final report: `final-fix-report.md`.
