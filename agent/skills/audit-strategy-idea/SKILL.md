---
name: audit-strategy-idea
description: >-
  Audit strategy_idea_brief.json and the brainstorm conversation before
  strategy_spec.yaml construction begins.
---

# Strategy Idea Auditor

Use this skill after `brainstorm-strategy-idea` writes
`strategy_idea_brief.json` and before `build-strategy-spec` is allowed to run.

This skill audits the brainstorm process. It does not judge whether the
strategy will work, and it does not translate the idea into a SPEC.

## Inputs

- `strategy_idea_brief.json`
- Agent-provided raw conversation history. Do not assume a filename or path.
  The invoking Agent must supply the exact user/agent conversation text or
  Studio-provided object in a task-local variable such as:

  ```text
  CONVERSATION_HISTORY_RAW:
  <paste or load the exact brainstorm conversation text>
  ```

If `CONVERSATION_HISTORY_RAW` is missing, empty, summarized, or not exact, the
audit must block. Do not use the SHA-256 of an empty string as a placeholder
conversation hash unless the actual raw conversation is truly empty, which is
not a valid strategy brainstorm. The hash must be computed from the exact raw
conversation supplied by the coordinator.

Canonical hash rule: take the text after the `CONVERSATION_HISTORY_RAW:`
marker, strip only leading and trailing whitespace from that body, then compute
SHA-256 over the resulting UTF-8 text while preserving all interior content,
line breaks, and order. Compare this canonical hash to
`strategy_idea_brief.json.conversation_hash`.

## Version-Governed Artifact Gate

Before reading or writing audit artifacts, read `current.json` and use
`active_version` as `version_id`. The version-governed input brief is:

```text
versions/<version_id>/01_brainstorm/strategy_idea_brief.json
```

Write `strategy_idea_audit.json` only to:

```text
versions/<version_id>/02_idea_audit/strategy_idea_audit.json
```

The initial concrete output path is:

```text
versions/v001/02_idea_audit/strategy_idea_audit.json
```

Do not write root-level `strategy_idea_audit.json`.

## Scope

Do:

- audit the strategy brainstorm workflow
- verify that every confirmed value in `strategy_idea_brief.json` has user
  evidence in the raw conversation
- verify that every phase was explained before the brainstormer asked for
  values
- verify that defaults and candidates were explicitly confirmed by the user
- verify that `strategy_idea_brief.json.conversation_hash` equals the SHA-256
  hash computed from the supplied exact raw conversation
- write `strategy_idea_audit.json`

Do not:

- Do not read, write, or edit `strategy_spec.yaml`
- Do not run `oxq`
- Do not export `component_catalog.json`
- Do not repair `strategy_idea_brief.json`
- Do not fill missing values on behalf of the user
- Do not call `build-strategy-spec`

## Strategy Idea Workflow Audit

Check that:

- every required brainstorm phase is present
- phases appear and complete in the required order
- the brainstormer explained the phase before asking for values
- the brainstormer redirected the user to the earliest incomplete phase when
  the user skipped ahead
- default or candidate values were explicitly confirmed by the user
- every confirmed phase has user evidence from `CONVERSATION_HISTORY_RAW`
- no phase marks inferred, defaulted, or example values as confirmed
- `strategy_idea_brief.json.conversation_hash` is present, non-placeholder,
  not the hash of an empty string, and exactly equals the hash computed from
  the supplied exact raw conversation

Required phase names:

- `research intent and hypothesis`
- `market, universe, and benchmark`
- `data and evaluation window`
- `Indicator definitions`
- `Signal rule definitions`
- `Portfolio construction`
- `execution, costs, rebalance, and risk constraints`
- `metrics, robustness, and decision policy`

Audit phase coverage against SPEC field groups:

- research phase: `strategy_id`, `name`, `research.*`
- market phase: `market.*`, `universe.*`, `benchmark.symbols`
- data phase: `data.*`, `validation.*`
- Indicator definitions phase: `signal.indicators.*`
- signal phase: `signal.signal_time`, `signal.rules.*`
- portfolio phase: `portfolio.type`, `portfolio.params`
- execution phase: `execution.*`, `cost.*`, `portfolio.rules.*`
- decision phase: `metrics.*`, `robustness.*`, `decision_policy.*`

Missing, skipped, unconfirmed, or out-of-order brainstorm phases block the
audit. A phase also blocks when the brainstormer did not explain its purpose
before asking for values, when candidate defaults were not explicitly accepted,
or when the brainstormer failed to redirect the user back into the workflow
after a skip.

A missing, placeholder, empty-string, summarized-conversation, or mismatched
`conversation_hash` also blocks the audit. The audit artifact must record the
computed conversation hash and the found brief hash value in
`blocking_findings`.

Indicator definitions require special attention. The audit must verify that
the user described each Indicator as an Indicator, not only as a final signal,
and that the brief maps those definitions to `signal.indicators.*`.

## Default Confirmation Rule

Defaults and examples are never confirmed by existence. A value passes only
when the user explicitly gave it, accepted a candidate, or confirmed a grouped
candidate checklist.

If a value is useful but not confirmed, block the audit and ask the coordinator
to return to `brainstorm-strategy-idea`. Do not make the confirmation yourself.

## Output Artifact

Write `strategy_idea_audit.json`:

```json
{
  "schema_version": 1,
  "status": "pass | block | fail",
  "idea_workflow_pass": true,
  "strategy_idea_brief": "versions/<version_id>/01_brainstorm/strategy_idea_brief.json",
  "strategy_idea_brief_hash": "sha256:<hash>",
  "conversation_hash": "sha256:<hash>",
  "phase_audits": [
    {
      "phase": "Indicator definitions",
      "status": "confirmed | missing | skipped | unconfirmed | out_of_order",
      "order": 4,
      "purpose_explained": true,
      "spec_fields": ["signal.indicators.*"],
      "evidence": ["..."],
      "blocking": false
    }
  ],
  "default_confirmation_audits": [],
  "blocking_findings": [],
  "next_required_phase": "brainstorm | build"
}
```

Set `status: block`, `idea_workflow_pass: false`, and
`next_required_phase: brainstorm` when any workflow or confirmation check
fails.

Set `status: pass`, `idea_workflow_pass: true`, and `next_required_phase:
build` only when the entire brainstorm brief is complete, ordered, explained,
and backed by user evidence.

## Result

Return the audit status, `strategy_idea_audit.json` path, blocking findings,
and the next required phase. If the audit blocks, ask for a return to
`brainstorm-strategy-idea`, not to `build-strategy-spec`.
