---
name: audit-artifact-lineage
description: >-
  Use when open-xquant artifacts must be checked for version/run/final
  traceability before comparison, report review, migration, or final selection.
---

# Artifact Lineage Auditor

This skill audits artifact lineage across a strategy version, run package,
comparison, or final selection. It does not create versions, compare metrics,
write reports, or choose the winner.

## Inputs

- `workflow_manifest.json`
- `current.json`
- `lineage.json`
- version manifests under `versions/**`
- run manifests and artifact hashes under `09_backtests/**`
- comparison manifests under `comparisons/**`
- final selection artifacts under `final/**`

## Checks

- Every run must reference one confirmed SPEC, one confirmed spec audit, one
  runtime audit, and one compiled plan from the same version.
- Every report must reference the version_id and run_id it discusses.
- Every final selection must reference only an eligible candidate.
- Every cross-artifact reference must include `hash_type`.
- A final decision is blocked when lineage is missing, stale, or impossible to
  recompute.

Eligible candidate means the primary run has confirmed spec audit, passing
runtime audit, passing reproducibility audit, no fatal research audit, and a
report review that is pass or explicitly non-blocking.

## Outputs

```text
governance/lineage_audit_<timestamp>.json
governance/lineage_audit_<timestamp>.md
```

The JSON must include `status`, `scope`, `checked_artifacts`,
`blocking_findings`, `warnings`, and `next_required_phase`.

## Red Lines

- Do not rewrite hashes to make lineage pass.
- Do not compare strategy performance.
- Do not select a final version.
- Do not infer that missing audit artifacts passed.

## Result

Return whether artifact lineage is pass, block, or fail, and list the exact
version/run/final references that need repair or regeneration.
