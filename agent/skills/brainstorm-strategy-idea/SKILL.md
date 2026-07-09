---
name: brainstorm-strategy-idea
description: >-
  Guide a user through the open-xquant pre-spec strategy idea workflow and
  produce strategy_idea_brief.json before any strategy_spec.yaml work begins.
---

# Strategy Idea Brainstorm

Use this skill when the user has a strategy idea, asks to design a new
strategy, or tries to jump directly into SPEC creation without a completed
strategy idea brief.

This is a user-facing elicitation skill. Its job is to make the user describe
the whole strategy idea in a strict phase order, with no unconfirmed defaults,
before a builder Agent can write `strategy_spec.yaml`.

## Scope

Do:

- actively guide the user through the pre-spec brainstorm workflow
- explain each phase before asking the user for values
- ask only for the values needed by the current phase
- Pull the user back to the earliest incomplete phase when they skip ahead
- record explicit user evidence for every confirmed value
- label any proposed value as a candidate until the user confirms it
- write `strategy_idea_brief.json`

Do not:

- Do not run `oxq`
- Do not import the open-xquant SDK
- Do not export `component_catalog.json`
- Do not write or edit `strategy_spec.yaml`
- Do not validate a spec
- Do not audit the brief
- Do not treat a template default, Agent preference, or example value as a
  user-confirmed value

## Version-Governed Output Gate

Before writing any artifact, read `current.json` and use
`active_version` as `version_id`. If `active_version` is missing, block and
return to `manage-strategy-version`; do not create a root-level staging file.

Write `strategy_idea_brief.json` only to:

```text
versions/<version_id>/01_brainstorm/strategy_idea_brief.json
```

The initial concrete path is:

```text
versions/v001/01_brainstorm/strategy_idea_brief.json
```

Do not write root-level `strategy_idea_brief.json`.

## Conversation Rule

Work one phase at a time. If the user gives information for a later phase,
record it as candidate context and return to the earliest incomplete phase.

For every phase:

1. Explain the phase before asking for values.
2. Ask only for values needed to complete that phase.
3. Record confirmed values, candidate values, unconfirmed values, and user
   evidence.
4. If the user gives a partial answer, ask the smallest next question for that
   phase.
5. Any default or candidate value must be explicitly confirmed by the user.

Never say a value is confirmed because it is common, reasonable, in a template,
or inferred from a prior example. Even when you propose a default, ask the user
to accept or replace it.

## Required Phases

The brief is incomplete until all eight phases are covered in this order.

1. `research intent and hypothesis`
   - Explain: this defines what market behavior the strategy claims to exploit
     and what would make the idea testable.
   - Confirm: `strategy_id`, `name`, `research.hypothesis`,
     `research.rationale`.

2. `market, universe, and benchmark`
   - Explain: this defines the tradable scope and comparison baseline.
   - Confirm: `market.*`, `universe.*`, `benchmark.symbols`,
     survivorship policy, and point-in-time policy.

3. `data and evaluation window`
   - Explain: this defines what data can support the idea, how bars are read,
     and what period is evaluated.
   - Confirm: `data.provider`, `data.data_dir`, `data.price_adjustment`,
     `data.required_columns`, `data.min_start_date`,
     `validation.train_period`, `validation.test_period`,
     `validation.required_oos`.

4. `Indicator definitions`
   - Explain: Indicators are deterministic derived columns computed from raw
     market data. They are stored under `signal.indicators.*` and are not the
     final trading decision by themselves.
   - Confirm each Indicator output column, component type, input column,
     lookback/window, parameters, and warmup implication.
   - Confirm: `signal.indicators.*` and any `data.min_start_date` needed by
     Indicator lookbacks.

5. `Signal rule definitions`
   - Explain: Signal rules convert raw data or Indicator columns into boolean
     signals or categorical `BUY` / `SELL` / `HOLD` intent.
   - Confirm: `signal.signal_time`, `signal.rules.*`,
     `signal.rules.*.output_domain` when categorical.

6. `Portfolio construction`
   - Explain: this converts per-symbol signals and Indicator scores into target
     portfolio weights.
   - Confirm: `portfolio.type`, `portfolio.params`, filters, ranking, TopN,
     weight normalization, max weights, and cash fallback behavior.

7. `execution, costs, rebalance, and risk constraints`
   - Explain: this defines when signals are observed, when orders fill, what
     trading costs apply, and which runtime constraints are executable.
   - Confirm: `execution.*`, `cost.*`, supported `portfolio.rules.*`,
     rebalance interval, lot size, initial cash, cash return, fee, slippage,
     and unsupported risk or exit requests that must block or route elsewhere.

8. `metrics, robustness, and decision policy`
   - Explain: this defines how the experiment will be judged and what evidence
     is required before any promotion decision.
   - Confirm: `metrics.*`, `robustness.*`, `decision_policy.*`, benchmark
     success metric, reject thresholds, and promotion thresholds.

## Brief Artifact

Write or update `strategy_idea_brief.json` as the single output of this skill:

```json
{
  "schema_version": 1,
  "status": "complete | blocked",
  "current_phase": "Indicator definitions",
  "next_required_phase": "brainstorm | audit_idea",
  "spec_readiness": "ready_for_idea_audit | blocked",
  "phases": [
    {
      "id": "indicator_definitions",
      "name": "Indicator definitions",
      "order": 4,
      "purpose_explained": true,
      "status": "confirmed | blocked",
      "spec_fields": ["signal.indicators.*", "data.min_start_date"],
      "confirmed_values": {},
      "candidate_values": {},
      "unconfirmed_values": {},
      "user_evidence": ["..."],
      "blocking_questions": []
    }
  ],
  "unresolved_defaults": [],
  "conversation_hash": "sha256:<hash>"
}
```

Hash rule:

- Compute `conversation_hash` from the exact raw brainstorm conversation that
  backs the brief, including the phase explanations, user answers, and user
  confirmations that appear in `user_evidence`.
- If the coordinator supplies a `CONVERSATION_HISTORY_RAW` block, hash the
  canonical raw conversation body: take the text after the
  `CONVERSATION_HISTORY_RAW:` marker, strip only leading and trailing
  whitespace from that body, then compute SHA-256 over the resulting UTF-8
  text while preserving all interior content, line breaks, and order.
- If no exact raw conversation is available, set `status: blocked`,
  `spec_readiness: blocked`, and ask the coordinator for exact raw
  conversation history.
- Never write `sha256:placeholder`, an empty-string hash, a made-up hash, or a
  hash of a summary.

Use `status: blocked` until every phase has `purpose_explained: true`, explicit
user evidence, and no unconfirmed material value. When blocked, return the next
question for the earliest incomplete phase and set `next_required_phase:
brainstorm`.

When complete, set `next_required_phase: audit_idea`. The next skill is
`audit-strategy-idea`, not `build-strategy-spec`.

## Red Lines

- Do not let the user skip the workflow by asking for a backtest, a quick
  draft, or sensible assumptions.
- Do not ask the builder to infer missing values.
- Do not call `build-strategy-spec` before `strategy_idea_brief.json` is
  complete and later passes `audit-strategy-idea`.
- Do not collapse Indicator definitions into signal rules. Indicators are
  SPEC `signal.indicators.*`; signal rules are SPEC `signal.rules.*`.

## Result

Return the current phase, the updated `strategy_idea_brief.json` path, the
remaining blocking questions, and whether the next required phase is
`brainstorm` or `audit_idea`.
