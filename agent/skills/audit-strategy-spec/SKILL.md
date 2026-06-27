---
name: audit-strategy-spec
description: Audit strategy_spec.yaml field provenance before backtests.
---

# Spec Auditor

Use this skill after `oxq spec validate strategy_spec.yaml` passes and before
any `oxq backtest run`. Its job is to prevent unapproved strategy assumptions
from entering a formal experiment.

## Inputs

- `strategy_spec.yaml`
- Agent-provided raw conversation history. Do not assume a filename or path.
  The invoking Agent must supply the source text or Studio-provided object in a
  task-local variable such as:

  ```text
  CONVERSATION_HISTORY_RAW:
  <paste or load the exact user/agent conversation text for this experiment>
  ```

  If Studio provides a `conversation.json` object or path variable, use that
  provided value. Do not hardcode `conversation.json` as a required path.
- `component_catalog.json`, or a freshly exported catalog

## Source Boundary

Use the start of the current experiment as the source boundary. A just-finished
`oxq spec validate` is only a checkpoint proving the current YAML is
well-formed; it is not the boundary for tracing user confirmations made while
building that spec.

When resuming after a prior run, use the latest relevant prior run containing
`spec_hash.txt` and `environment.json` to identify the previous experiment's
timestamp, then trace user confirmations made after that point. If no prior
validated run exists, trace from the start of the current conversation.

## Field Classification

Classify material fields as:

- `confirmed`: the user explicitly gave the value or an equivalent meaning.
- `default`: the value matches a documented build-strategy-spec template default
  that is proposed but not yet confirmed.
- `unconfirmed`: the value is not a template default and no user source exists.
- `agent_added`: the Agent introduced the value by inference, convention, or a
  workflow preference that the user did not explicitly approve.

Never mark a field `confirmed` when its evidence says the user did not specify,
did not confirm, or that the Agent inferred the value. If the evidence says
"用户未指定", "未确认", "Agent 将", "Agent inferred", or equivalent wording, the
status must be `agent_added` or `unconfirmed`, and the field must block the
backtest unless it is a documented template default accepted by the user.

`default` is not a passing final status. A default may appear while the audit is
blocked and asking grouped confirmation questions, but every effective SPEC
field must become `confirmed` before `spec_audit.json` can pass. This includes
fields omitted from YAML but injected by OpenXQuant defaults, such as execution,
cost, cash, validation, metrics, empty dictionaries, and empty lists.

Material fields include:

- train/test periods
- `validation.required_oos`
- symbols and benchmark
- execution timing and fill price fields
- data warmup policy and `data.min_start_date`
- initial cash and cash return
- fee rate and slippage rate
- risk-free rate and metrics profile
- exit, risk, and rebalance constraints

If the user supplied only one full backtest date range, do not treat an
Agent-created IS/OOS split as confirmed. Classify `validation.train_period`,
`validation.test_period`, and `validation.required_oos` as `agent_added` or
`unconfirmed` until the user confirms the split or explicitly accepts the
default validation plan.

When the user supplied one full backtest date range and did not request OOS
validation, `validation.required_oos: false` with that full range in
`validation.test_period` is a valid full-interval backtest representation. Do
not force `required_oos: true`, and do not require a train/test split unless
the user confirms it.

Audit data warmup as a material field. If the spec uses indicators, signals,
rules, or recipes with lookback periods, verify whether the user or builder
notes confirmed one of these policies:

- pre-window data is loaded through `data.min_start_date`
- the first lookback window may remain NaN/cash until enough bars exist
- no warmup is required because the strategy has no lookback dependency

Block the audit when lookback behavior exists but `data.min_start_date` or an
explicit no-warmup policy is missing. This is material because it can change
early-period exposure and make runs incomparable.

For each material field, record source evidence:

- `field_path`
- current spec value
- classification: `confirmed`, `default`, `unconfirmed`, `contradiction`, or
  `agent_added`
- evidence snippets or message references from `CONVERSATION_HISTORY_RAW`
- whether the item blocks backtest

## Component Provenance

Before approving a spec for backtest, audit component provenance against the
same catalog used while building the spec:

1. Load `component_catalog.json` from the research directory. If it is missing,
   run:

   ```bash
   uv run oxq registry export --out component_catalog.json
   ```

2. Record the catalog's `catalog_hash` and compare it with any hash recorded in
   the spec build notes, Studio task metadata, or prior audit artifact. A hash
   mismatch blocks the backtest until the spec is rechecked against the current
   catalog or the user explicitly accepts the catalog change.
3. Verify every `signal.indicators.*.type`, `signal.rules.*.type`,
   `portfolio.type`, and any documented rule component exists in the catalog.
   Components absent from the catalog are blocking unless a separate component
   authoring workflow has already registered the custom component in the
   catalog.
4. Search catalog aliases and `recipes` for more standard canonical structures
   than the spec currently uses. Block when the user request matches a recipe
   but the spec decomposes it differently or uses an invented shortcut name.
5. Block Agent-created names that look semantic but are not registered, such as
   `RiskAdjustedMomentum`, when equivalent built-in components or recipes exist.
6. Check semantic coverage: each user-requested indicator, signal, portfolio
   optimizer, and rule must appear either as a selected catalog component or as
   a selected recipe. Missing user-requested semantics block the backtest.

Also check:

- SPEC fields that exist but were never requested or confirmed by the user.
- User-stated requirements that are missing from the SPEC.
- SPEC values that contradict the conversation history.
- Component choices that deviate from a matched recipe's `canonical_spec`.

Examples:

- If the user says "20日收益率 / 20日波动率" and the spec uses
  `RiskAdjustedMomentum`, block it when the catalog has no such component.
  Require the canonical `volatility_adjusted_momentum` recipe:
  `NdayReturn + RollingVolatility + Ratio`.
- If the user says "取 TopN，归一化权重" and the spec uses `EqualWeight`, block
  it because `TopNRanking` is the catalog component matching that portfolio
  semantic.

## Gate

Any `default`, `unconfirmed`, or `agent_added` effective field blocks formal
backtest until the user confirms the grouped assumption or provides a
replacement value. Any blocking component provenance issue blocks backtest.
Train/test splits and required OOS settings are material and must not pass
silently.
Always group related fields instead of asking one question per YAML key:

- data warmup and local data coverage
- execution assumptions
- cost assumptions
- train/test split
- cash and risk-free assumptions
- benchmark and success metric
- component/catalog provenance

Ask the user to either confirm the group or provide replacement values. After
any change, re-run `oxq spec validate strategy_spec.yaml` and repeat this
auditor gate.

## Default Confirmation Checklist

When default fields are the only remaining blockers, do not ask one question
per field. Present one compact confirmation checklist grouped by assumption
area. Use a Markdown table only when the Agent surface supports it cleanly;
otherwise use short grouped bullets with the same columns.

Each checklist row must include:

- group
- field path
- value
- why this value exists
- runtime or research impact

Use these default groups when applicable:

- `validation`: full-period backtest, `required_oos`, train/test windows
- `execution`: trade time, fill price, rebalance default, lot size
- `cost`: fee rate, minimum fee, slippage
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
`confirmed`. If the user rejects or edits any row, update `strategy_spec.yaml`,
rerun deterministic validation, then repeat the audit.

Before emitting a final passing audit, validate effective field coverage:

```bash
uv run oxq spec-audit validate spec_audit.json --spec strategy_spec.yaml --strict-confirmed
```

Do not report `status: pass` unless this command passes.

## `spec_audit.json`

Write `spec_audit.json` before approving a backtest. It is semantic output from
this skill, not from a deterministic CLI. Use this schema:

```json
{
  "schema_version": 3,
  "status": "pass | block | fail",
  "spec_provenance_pass": true,
  "spec_hash": "sha256:<hash>",
  "conversation_hash": "sha256:<hash>",
  "catalog_hash": "sha256:<hash>",
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
  "blocking_findings": [{"message": "...", "question": "..."}]
}
```

Compute `conversation_hash` from the exact raw conversation input supplied to
this skill. After writing `spec_audit.json`, run:

```bash
uv run oxq spec-audit validate spec_audit.json --spec strategy_spec.yaml --strict-confirmed
```

Schema validation only proves the artifact shape. It does not prove the
semantic audit is correct; that responsibility stays in this skill.

This skill does not compile the strategy and does not compare
`strategy_spec.yaml` with `compiled_plan.json`. That is the `audit-runtime-semantics`
skill's boundary.

## Output

Report a compact summary:

- confirmed fields
- default fields awaiting checklist confirmation
- unconfirmed fields that block progress
- selected catalog components and recipes
- component provenance issues, including catalog hash mismatch, missing
  components, non-canonical recipe decomposition, or missing user-requested
  semantics
- path to `spec_audit.json`
- Default Confirmation Checklist when defaults need user approval
- blocking confirmation questions

Do not run or approve a backtest while blocking fields remain. After this skill
passes, the next formal gate is `audit-runtime-semantics`.
