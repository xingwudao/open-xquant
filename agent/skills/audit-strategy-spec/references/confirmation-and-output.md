# Confirmation And Output

Use this reference before writing `spec_audit.json`, confirmation tables,
default confirmation checklists, hashes, or strict validation commands.

## Contents

- Two-Step Spec Audit Gate Details
- Default Confirmation Checklist
- `spec_audit.json`
- Hashes And Validation
- Output

## Two-Step Spec Audit Gate Details

The SPEC audit has two semantic states before downstream runtime work:

- `audit_conclusion: all_pass`: the auditor found no remaining provenance,
  calibration, default, component, recipe, or mapping blockers.
- `user_confirmation_status: confirmed`: the user explicitly confirmed the
  complete SPEC table shown by the coordinator.

These are not the same. When the auditor reaches `audit_conclusion: all_pass`
but the user has not confirmed the table, write `spec_audit.json` with:

- `status: block`
- `spec_provenance_pass: true`
- `audit_conclusion: all_pass`
- `user_confirmation_status: pending`
- `next_required_phase: user_spec_confirmation`
- `blocking_findings: []`

Use `status: block`, `user_confirmation_status: pending`, and
`next_required_phase: user_spec_confirmation` as the administrative gate until
user confirmation. Do not add the pending confirmation itself to
`blocking_findings`, because `audit_conclusion: all_pass` means there are no
remaining SPEC provenance, calibration, component, recipe, or mapping blockers.

The coordinator must then show a Full Spec Confirmation Table to the user.
Use a Markdown table and include the complete SPEC, not just exceptions:

```markdown
| Section | Field path | Spec value | Source | Audit status | Impact |
| --- | --- | --- | --- | --- | --- |
| execution | execution.fill_price_mode | next_open | User confirmed execution group | confirmed | Changes fill timing and slippage realism |
```

Do not summarize only blockers. Include every material field and component
choice needed to understand the whole strategy: research, market, universe,
data, validation, benchmark, indicators, signals, portfolio, rules, execution,
cost, risk constraints, metrics, robustness, and decision policy when present.

Only write `spec_confirmation_table.md` when the audit reaches
`audit_conclusion: all_pass`, including the `user_confirmation_status: pending`
administrative gate, or when recording the final
`user_confirmation_status: confirmed` audit. If the audit is blocked for
provenance, mapping, calibration, component, data, default, or contradiction
reasons, do not write a placeholder confirmation table and do not include a
fake table hash.

Only after the user explicitly confirms the full Markdown table may the audit
become confirmed. Then update or write `spec_audit.json` with:

- `status: pass`
- `spec_provenance_pass: true`
- `audit_conclusion: all_pass`
- `user_confirmation_status: confirmed`
- `confirmation_event` referencing the durable confirmation log event
- `next_required_phase: runtime_audit`

If the user does not confirm the full table, the audit remains blocked and no
downstream worker may run. If the user rejects or edits any row, return to the
earliest required phase: `brainstorm` for idea changes or `build` for SPEC
translation changes. The SPEC auditor must not edit the YAML directly.

Do not use the pending Full SPEC Confirmation Table itself as evidence that the
user has already authorized the strategy. Pending table rows are for user
review; the top-level `user_confirmation_status: pending` is the downstream
gate. However, for a pending all-pass audit, `field_audits` must already
satisfy strict-confirmed coverage because the CLI validates every effective
StrategySpec field before the user is asked to confirm the table.
Do not leave any effective field row as `status: default`, `unconfirmed`, or `agent_added`
when `audit_conclusion: all_pass`.

For an effective field that was selected by runtime, catalog, parser, or
framework behavior rather than an earlier user turn, use
`status: confirmed`, `blocking: false`, and evidence that describes strict
coverage without claiming final user authorization, for example:
"Strict confirmed coverage row for the Full SPEC Confirmation Table; final
user authorization is tracked by top-level `user_confirmation_status`."
Avoid evidence wording that says the user did not specify or confirm the field,
because the deterministic validator rejects `status: confirmed` rows whose
evidence contains those negative phrases. Do not write stale evidence such as
"Field value included in full SPEC confirmation table for user approval" if it
leaves ambiguity about whether the user has already authorized the table.

After confirmation, the next worker must generate and print the complete `strategy.py` source
for user review before any formal backtest authorization.

## Default Confirmation Checklist

When default fields are the only remaining blockers, do not ask one question
per field. Present one compact confirmation checklist grouped by assumption
area. Use short grouped bullets when tables would be hard to read.

Each checklist item must include:

- group
- field path
- value
- why this value exists
- runtime or research impact

Use these default groups when applicable:

- `validation`: full-period backtest, `required_oos`, train/test windows
- `execution`: trade time, fill price, rebalance default, lot size
- `cost`: fee rate, side-specific fees, sell-side tax, minimum fee, slippage
- `cash`: initial cash and cash return
- `metrics`: risk-free rate, annualization days, evaluation window
- `data`: provider, data directory, adjustment, warmup policy
- `empty policy`: empty dictionaries or lists that disable optional behavior

Ask one grouped question after the checklist, such as:

```text
Please confirm whether you accept all default assumptions in this checklist.
You can also reject or override any row by field path.
```

If the user confirms the whole checklist or a group, convert every covered
field to `confirmed` in `field_audits`. The same batch confirmation may be used
as evidence for multiple rows, for example:

```json
{
  "field_path": "execution.fill_price_mode",
  "spec_value": "next_open",
  "status": "confirmed",
  "evidence": [
    "User confirmed the Default Confirmation Checklist execution group."
  ],
  "blocking": false
}
```

Do not mark fields outside the confirmed checklist or confirmed group as
`confirmed`. If the user rejects or edits any row, return to the builder; do
not update `strategy_spec.yaml` from the auditor phase. The builder must rerun
deterministic validation before the audit repeats.

Before emitting a final passing audit, validate the StrategySpec hash and
effective field coverage:

```bash
oxq spec-audit validate versions/<version_id>/06_spec_audit/spec_audit.json \
  --spec versions/<version_id>/04_spec_build/strategy_spec.yaml \
  --component-catalog versions/<version_id>/04_spec_build/component_catalog.json \
  --mapping-contract versions/<version_id>/04_spec_build/spec_mapping_contract.json \
  --strict-confirmed
```

Do not report `status: pass` unless this command passes.

## `spec_audit.json`

Write `spec_audit.json` before approving a backtest. It is semantic output from
this skill, not from a deterministic CLI. Use the existing schema version:

```json
{
  "schema_version": 4,
  "status": "pass | block | fail",
  "spec_provenance_pass": true,
  "audit_conclusion": "all_pass | blocked | fail",
  "user_confirmation_status": "pending | confirmed | rejected",
  "spec_hash": "sha256:<StrategySpec.compute_hash()>",
  "conversation_hash": "sha256:<hash>",
  "catalog_hash": "<component_catalog.catalog_hash>",
  "strategy_idea_brief": "versions/<version_id>/01_brainstorm/strategy_idea_brief.json",
  "strategy_idea_audit": "versions/<version_id>/02_idea_audit/strategy_idea_audit.json",
  "strategy_idea_brief_hash": "sha256:<hash>",
  "strategy_idea_audit_hash": "sha256:<hash>",
  "next_required_phase": "user_spec_confirmation | runtime_audit | build | brainstorm | data_inspection",
  "recipe_matches": [
    {
      "recipe": "volatility_adjusted_momentum",
      "status": "used | available_but_not_used | not_applicable",
      "evidence": ["..."],
      "canonical": true
    }
  ],
  "field_audits": [
    {
      "field_path": "execution.initial_cash",
      "spec_value": 100000,
      "material_category": "execution_assumption",
      "status": "confirmed | default | unconfirmed | contradiction | agent_added",
      "evidence": ["..."],
      "blocking": false
    }
  ],
  "component_audits": [
    {
      "component_path": "signal.indicators.ret_n.type",
      "component_type": "NdayReturn",
      "status": "catalog | recipe | missing | non_canonical",
      "recipe": "volatility_adjusted_momentum",
      "evidence": ["..."],
      "blocking": false
    }
  ],
  "missing_user_requirements": [{"message": "...", "evidence": ["..."]}],
  "agent_added_fields": [{"message": "...", "field_path": "..."}],
  "contradictions": [{"message": "...", "field_path": "...", "evidence": ["..."]}],
  "unsupported_mappings": [
    {
      "source_field": "portfolio.constraints.min_position_value",
      "requested_semantic": "minimum notional position size",
      "reason": "current SPEC parses this field but the audited runtime cannot execute it",
      "disposition": "blocked",
      "blocking": true
    }
  ],
  "spec_confirmation_table": {
    "format": "markdown",
    "path": "versions/<version_id>/06_spec_audit/spec_confirmation_table.md",
    "hash": "sha256:<markdown file hash>",
    "hash_type": "sha256"
  },
  "confirmation_event": {
    "path": "conversations/<conversation_id>/confirmations.jsonl",
    "event_id": "<stable confirmation event id>",
    "line_number": 1,
    "event_hash": "sha256:<confirmation jsonl line hash>",
    "artifact_path": "versions/<version_id>/06_spec_audit/spec_confirmation_table.md",
    "artifact_hash": "sha256:<markdown file hash>",
    "spec_audit_path": "versions/<version_id>/06_spec_audit/spec_audit.json",
    "spec_audit_hash": "sha256:<pre-confirmation spec_audit hash>",
    "phase": "spec_confirmation",
    "field_scope": "full_spec_table"
  },
  "blocking_findings": [{"message": "...", "question": "..."}]
}
```

`spec_confirmation_table` is conditional. It is required only when the SPEC has
no audit blockers and the workflow is at `audit_conclusion: all_pass`,
`user_confirmation_status: pending`, or
`user_confirmation_status: confirmed`. For `audit_conclusion: blocked`, omit
`spec_confirmation_table` or set it to `null`; do not write a placeholder table
to satisfy schema.

`confirmation_event` is conditional separately. It is required only when
`user_confirmation_status: confirmed` or `status: pass`. It must point to the
line in `conversations/<conversation_id>/confirmations.jsonl` where the user
confirmed the full SPEC table. The JSONL line and audit reference must both
include the same `event_id`, `artifact_path`, `artifact_hash`,
`spec_audit_path`, and `spec_audit_hash`. The JSONL line must also include
`phase: spec_confirmation` and `field_scope: full_spec_table`; the
`confirmation_event` object in `spec_audit.json` must mirror both values.
`spec_audit_hash` is the pre-confirmation `spec_audit.json` hash.
Pending all-pass audits must not
invent a confirmation event.

Write the JSONL confirmation line first without a nested `event_hash` field.
The `confirmation_event.event_hash` field in `spec_audit.json` is the SHA-256
hash of that raw JSONL line content. It is not the hash of the parsed JSON
object, and it is not a value that must appear inside the JSONL payload.

Compute `confirmation_event.spec_audit_hash` from the canonical
pre-confirmation audit payload: remove `confirmation_event`, set
`status: block`, set `user_confirmation_status: pending`, and use the variant
whose `next_required_phase` is `user_spec_confirmation`. Serialize that
candidate with `json.dumps(candidate, sort_keys=True, default=str)` and hash
those bytes with SHA-256; use either the framework-short or full
`sha256:<hex>` digest accepted by the validator. This is not the final
post-confirmation `spec_audit.json` hash and not a raw file hash.

`field_audits` are also conditional in a different way: they describe only
effective StrategySpec fields from `StrategySpec.from_yaml(...).to_effective_dict()`.
If a user-confirmed value appears only in a YAML-only or non-operative path,
audit the effective field as `contradiction` and record the raw YAML location
as source evidence:

```json
{
  "field_path": "execution.initial_cash",
  "spec_value": 100000.0,
  "material_category": "execution_assumption",
  "status": "contradiction",
  "evidence": [
    "User confirmed initial_cash = 1000000.",
    "YAML source path portfolio.initial_cash contains 1000000 but is not effective."
  ],
  "blocking": true
}
```

Record the correction request outside `field_audits`:

```json
{
  "message": "initial_cash was placed under a non-operative YAML path",
  "effective_field_path": "execution.initial_cash",
  "source_yaml_path": "portfolio.initial_cash",
  "expected_value": 1000000,
  "effective_value": 100000.0,
  "builder_required_fix": "move the value to execution.initial_cash and remove portfolio.initial_cash"
}
```

Do not write `portfolio.initial_cash` as a `field_audits` row. It is source
evidence for a builder mapping fix, not an effective SPEC field.

When `audit_conclusion: all_pass` or `status: pass`, `blocking_findings` must
be an empty list. Do not keep resolved historical blockers in
`blocking_findings`; move that explanation to `audit_notes.md`, field evidence,
or a non-blocking resolution note. Any non-empty `blocking_findings` keeps the
SPEC blocked for formal backtest.

Likewise, a passing audit must have empty `missing_user_requirements`,
`agent_added_fields`, and `contradictions` lists. Once the user confirms an
agent-added or default assumption in the full SPEC table, record it as a
`field_audits` row with `status: confirmed` and evidence, plus optional
`audit_notes.md` context; do not keep the resolved item in
`agent_added_fields`.

## Hashes And Validation

When a confirmation table is required, compute the
`spec_confirmation_table.hash` with the framework file hash helper, not with `shasum`.
The formal backtest gate accepts full SHA-256 hashes for
compatibility, but Agents should write the framework short hash:

```bash
PYTHON="$(dirname "$RUNNER")/python"
"$PYTHON" - <<'PY'
from pathlib import Path

from oxq.spec.compiler import _hash_file

table_path = Path("versions/<version_id>/06_spec_audit/spec_confirmation_table.md")
print(_hash_file(table_path))
PY
```

Compute `spec_hash` from the parsed strategy SPEC semantics, not from raw file
bytes. Use
`oxq spec validate versions/<version_id>/04_spec_build/strategy_spec.yaml` and
copy its `Spec Hash`, or equivalently use
`StrategySpec.from_yaml(...).compute_hash()`. Do not use
`shasum strategy_spec.yaml`; the file SHA cannot satisfy the backtest gate.

Compute `conversation_hash` from the same canonical raw conversation body used
by `brainstorm-strategy-idea` and `audit-strategy-idea`. If the input contains
`CONVERSATION_HISTORY_RAW:`, hash the body after that marker after stripping
only leading and trailing whitespace. The value must match
`strategy_idea_brief.json.conversation_hash` and
`strategy_idea_audit.json.conversation_hash`; do not hash the entire transcript
file including the marker, attachments, or later confirmation notes.

After writing a blocked or failed `spec_audit.json`, run non-strict
schema, hash, catalog, and mapping-contract validation:

```bash
oxq spec-audit validate versions/<version_id>/06_spec_audit/spec_audit.json \
  --spec versions/<version_id>/04_spec_build/strategy_spec.yaml \
  --component-catalog versions/<version_id>/04_spec_build/component_catalog.json \
  --mapping-contract versions/<version_id>/04_spec_build/spec_mapping_contract.json
```

Do not use `--strict-confirmed` for `audit_conclusion: blocked`; blocked audits
are expected to contain `default`, `unconfirmed`, `agent_added`, and
`contradiction` rows.

Run `--strict-confirmed` before returning any
`audit_conclusion: all_pass`, including the user-confirmation-pending state.
This proves the Full Spec Confirmation Table and `field_audits` cover every
effective SPEC field before the user is asked to confirm it. A strict PASS on
`status: block` / `user_confirmation_status: pending` is only a coverage check;
it is not downstream runtime or backtest authorization. After the user confirms
the full SPEC table, run the same strict command again before final
`status: pass`.

For a pending all-pass audit, `field_audits` must already satisfy strict-confirmed coverage.
The Full SPEC Confirmation Table must be a single continuous Markdown table
with the exact columns `Section`, `Field path`, `Spec value`, `Source`,
`Audit status`, and `Impact`.
Use the top-level effective field prefix as the `Section` value, such as `data` for
`data.filters.exclude_st` and `portfolio` for `portfolio.params.n`; for
top-level scalar fields such as `schema_version`, use the field name itself as
the section. Represent list and dict values with deterministic JSON
serialization.
Represent empty strings as an empty cell, not `(empty)`,
`null`, or explanatory text, so the table matches the effective StrategySpec
value.

Schema validation only proves the artifact shape. It does not prove the
semantic audit is correct; that responsibility stays in this skill.

This skill does not compile the strategy and does not compare
`strategy_spec.yaml` with `compiled_plan.json`. That is the
`audit-runtime-semantics` skill's boundary.
