---
name: select-final-version
description: >-
  Use when a user asks to choose, confirm, promote, or mark a final
  open-xquant strategy version after audited runs, reports, and comparisons
  exist.
---

# Final Version Selector

This skill performs final version governance. It chooses a final research
candidate only from eligible audited candidates and only after the user
confirms the selection policy. It is not investment advice.

## Inputs

- Candidate versions and primary runs from `experiments.jsonl`.
- `report_review.json` for each candidate.
- `comparisons/<comparison_id>/` artifacts when available.
- `governance/lineage_audit_*.json`.
- User-confirmed `selection_policy.json`.

## Policy Gate

Agent-suggested policy is only a candidate. It must not become final until the
user confirms it.

`selection_policy.json` must include:

- `confirmed_by_user: true`
- eligibility gates
- ranking fields
- tie breakers
- source conversation reference

If no confirmed policy exists, write a blocked result and ask the coordinator
to get user confirmation.

## Eligibility Gate

A candidate is eligible only when:

- spec audit is confirmed
- runtime audit is pass
- reproducibility audit is pass
- research audit has no fatal findings
- report review is pass or explicitly non-blocking
- artifact lineage audit is pass

## Outputs

```text
final/selection_<timestamp>/
  candidate_set.json
  selection_policy.json
  comparison_refs.json
  final_decision.json
  final_decision.md

final/current_final.json
```

`final_decision.json` must include selected_version_id, selected_run_id,
selection_policy, comparison_refs, blocked_candidates, and created_by_role.

## Red Lines

- Do not run backtests.
- Do not modify run artifacts, reports, metrics, audits, or comparisons.
- Do not select a final version without confirmed_by_user policy evidence.
- Do not call the result investable or live-trading ready.

## Result

Return `selected`, `blocked`, or `fail`, the selected version/run when any, and
the exact blocker preventing final selection.
