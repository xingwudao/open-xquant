---
name: audit-strategy-spec
description: Audit strategy_spec.yaml field provenance before backtests.
---

# Spec Auditor

Use this skill after `build-strategy-spec` validates `strategy_spec.yaml` and
before any `oxq backtest run`. Its job is to prevent unapproved strategy
assumptions from entering a formal experiment.

This is a SPEC calibration audit. The separate `audit-strategy-idea` skill owns
the brainstorm workflow audit.

## Inputs

- `strategy_spec.yaml`
- `strategy_idea_brief.json`
- `strategy_idea_audit.json`
- `builder_phase_result.json`
- `spec_build_notes.md` when available
- `spec_mapping_notes.md` when available
- `spec_mapping_contract.json` when available
- `data_inspection_result.json` when the builder cites data coverage, column
  coverage, common date range, price adjustment, or `latest_available`
  resolution evidence
- Agent-provided raw conversation history. Do not assume a filename or path.
  The invoking Agent must supply the source text or Studio-provided object in a
  task-local variable such as:

  ```text
  CONVERSATION_HISTORY_RAW:
  <paste or load the exact user/agent conversation text for this experiment>
  ```

  If Studio provides a `conversation.json` object or path variable, use that
  provided value. Do not hardcode `conversation.json` as a required path.
- `component_catalog.json` produced by the builder after the audited idea gate
  passes

## Version-Governed Artifact Gate

Before auditing, read `current.json` and use `active_version` as `version_id`.
If no active version exists, block and return to `manage-strategy-version`.

Read phase artifacts from these paths:

```text
versions/<version_id>/01_brainstorm/strategy_idea_brief.json
versions/<version_id>/02_idea_audit/strategy_idea_audit.json
versions/<version_id>/04_spec_build/strategy_spec.yaml
versions/<version_id>/04_spec_build/builder_phase_result.json
versions/<version_id>/04_spec_build/component_catalog.json
versions/<version_id>/04_spec_build/spec_build_notes.md
versions/<version_id>/04_spec_build/spec_mapping_notes.md
versions/<version_id>/04_spec_build/spec_mapping_contract.json
versions/<version_id>/05_data_inspection/data_inspection_result.json
```

Write SPEC audit artifacts only under:

```text
versions/<version_id>/06_spec_audit/spec_audit.json
versions/<version_id>/06_spec_audit/audit_notes.md
versions/<version_id>/06_spec_audit/spec_confirmation_table.md  # only for all_pass / user confirmation pending or confirmed
```

The coordinator must show
`versions/<version_id>/06_spec_audit/spec_confirmation_table.md` to the user
when the audit reaches all pass but confirmation is pending.
Do not create a placeholder `spec_confirmation_table.md` for
`audit_conclusion: blocked`; blocked audits may omit `spec_confirmation_table`
from `spec_audit.json` or set it to `null`.

Do not write root-level `spec_audit.json`, `audit_notes.md`, or
`spec_confirmation_table.md`.
Do not write `versions/<version_id>/04_spec_build/component_catalog.json` or
any other builder phase artifact.

## Read-Only SPEC Boundary

The SPEC auditor is read-only for builder outputs. Do not write, patch, rewrite, normalize, or repair
`versions/<version_id>/04_spec_build/strategy_spec.yaml` or any other
`versions/<version_id>/04_spec_build/` artifact during this skill.

If the audit discovers that a user-confirmed value was misplaced, ignored, dropped, or mistranslated
in YAML, block with `next_required_phase: build`.
Examples include a confirmed `execution.initial_cash` value placed under an
ignored section, a confirmed lot-size value represented only by a non-operative
field, or a user-confirmed rebalance value hidden behind an effective default.
Do not convert that finding into a confirmation-table row and do not ask the
user to confirm the wrong effective value. The builder must make the YAML change,
rerun `oxq spec validate`, refresh build notes or mapping artifacts
when needed, and return to `build-strategy-spec` before this audit resumes.

### User-confirmed source value vs effective value check

Use `StrategySpec.from_yaml(...).to_effective_dict()` as the field-path source
for effective SPEC auditing. Do not treat a YAML-only key that is absent from
the effective dictionary as a valid confirmed field. For example,
`portfolio.initial_cash` is not an effective StrategySpec field; audit `execution.initial_cash`, not `portfolio.initial_cash`.
`field_audits` must contain only effective StrategySpec field paths.
Do not write YAML-only paths such as `portfolio.initial_cash` as `field_audits` rows.
Put those paths in `evidence`, `source_yaml_path`, and
`builder_required_fix` fields on `contradictions` rows.

Before starting audit completion, before any Default Confirmation Checklist, compare every user-confirmed source value
from the audited idea to the effective StrategySpec value that will
actually run. If they differ, the SPEC is mistranslated and cannot be repaired by user confirmation
in the auditor phase.

Concrete blocker: if the audited idea confirms starting cash of
`1000000`, the YAML contains `portfolio.initial_cash: 1000000`, and the
effective StrategySpec still shows `execution.initial_cash: 100000.0`, the
value was placed under an ignored or non-operative field. In that case, write a
blocked audit with `next_required_phase: build`, do not write `audit_conclusion: all_pass`,
and do not create or show a Full SPEC
Confirmation Table for that value. The same rule applies to lot size,
rebalance, execution timing, data filters, costs, validation periods, and any
other user-confirmed material field whose effective value differs from the
confirmed source.

## Runner Resolution

In a new research directory, `uv run oxq` may fail because open-xquant is
installed as a long-lived Agent capability, not as a package in that directory.
Before running deterministic validation commands:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq` or bare `oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

## Audited Idea Gate

First fast-fail if the brainstorm audit is missing or blocked. Before catalog
work, field provenance, or grouped confirmation questions, check:

- `strategy_idea_brief.json` exists and its hash matches the builder result
  when a hash is available.
- `strategy_idea_audit.json` exists.
- `strategy_idea_audit.json` has `status: pass`,
  `idea_workflow_pass: true`, and `next_required_phase: build`.

If any check fails, do not continue the SPEC audit. Write or return a blocked
result with:

- `status: block`
- `spec_provenance_pass: false`
- `next_required_phase: brainstorm`
- a blocking finding that names the missing or failed idea artifact

Only continue when the audited idea gate passes.

## Spec Calibration Audit

After the audited idea gate passes, verify the spec faithfully maps the audited idea.
This audit answers whether `strategy_spec.yaml` is a correct, confirmed, and
catalog-backed implementation of the already audited strategy description.

Check that:

- every material `strategy_spec.yaml` value is confirmed by
  `strategy_idea_brief.json` or by later explicit user evidence
- no value depends only on a template, parser, runtime, or Agent-chosen default
- every confirmed phase in the idea brief appears in the SPEC field group it
  controls
- user-stated requirements are not missing from the SPEC
- SPEC values do not contradict the audited idea or raw conversation
- Indicator definitions from the idea brief map to `signal.indicators.*`
- signal rules map to `signal.rules.*`
- portfolio, execution, cost, metrics, robustness, and decision policy fields
  map to their corresponding audited idea sections

When the SPEC faithfully maps the idea but effective defaults still need user
approval, block with grouped confirmation questions. When the SPEC mistranslates
the audited idea, block with `next_required_phase: build`. When the audited
idea itself is incomplete or unconfirmed, block with `next_required_phase:
brainstorm`.

A SPEC mistranslation includes any effective StrategySpec value that contradicts
the audited idea because the YAML field was misplaced, ignored, dropped, or
mapped to a non-operative field. This is a builder error, not a user
confirmation problem.

Use these next-phase labels in findings:

- `next_required_phase: brainstorm`
- `next_required_phase: build`
- `next_required_phase: data_inspection`
- `next_required_phase: user_spec_confirmation`
- `next_required_phase: runtime_audit`

## OpenXQuant Version Provenance

Audit `required_oxq_version` as a material provenance field. OpenXQuant version
provenance is not a user strategy assumption, so the user does not choose a
default version value during brainstorm. The builder must still record it from
the resolved runner/package version, and the coordinator must show it in the
full SPEC confirmation table.

OpenXQuant version provenance must remain visible from builder output through
SPEC audit and runtime audit.

Check that:

- `strategy_spec.yaml` contains a non-empty `required_oxq_version`.
- `builder_phase_result.json` records the same `required_oxq_version`.
- `schema_version` remains the SPEC schema version and was not changed to the
  OpenXQuant package version.
- the recorded version is compatible with the runner or package metadata used
  by the current build phase when that metadata is available.

A missing, empty, contradictory, or stale `required_oxq_version` blocks formal
backtest and must return to `next_required_phase: build`. If the value is
present and internally consistent, mark it `confirmed` as generated provenance
with evidence from the builder result and runner metadata, then include it in
`field_audits`.

## Source Boundary

Use the start of the current experiment as the source boundary. A just-finished
`oxq spec validate` is only a checkpoint proving the current YAML is
well-formed; it is not the boundary for tracing user confirmations made while
building that spec.

When resuming after a prior run, use the latest relevant prior run containing
`spec_hash.txt` and `environment.json` to identify the previous experiment's
timestamp, then trace user confirmations made after that point. If no prior
validated run exists, trace from the start of the current conversation.

## Data Boundary Audit

Data coverage, column coverage, common date ranges, price adjustment evidence,
provider readiness, and `latest_available` resolution must come from a prior
`versions/<version_id>/05_data_inspection/data_inspection_result.json` or from
explicit user-confirmed data snapshot evidence. Do not accept
builder-authored notes saying it inspected parquet files, listed data
directories, or inferred the latest date as sufficient evidence.

Block with `next_required_phase: data_inspection` when a material SPEC value
depends on data facts and no data-inspection artifact or user-confirmed
snapshot supports it. Block with `next_required_phase: build` when the builder
removed a confirmed data or tradability requirement because local data was
missing or unverified.

## Field Classification

Classify material fields as:

- `confirmed`: the user explicitly gave the value or an equivalent meaning.
- `default`: the value matches a documented build-strategy-spec template
  default or runtime default that is visible but not yet confirmed.
- `unconfirmed`: the value is not a template default and no user source exists.
- `agent_added`: the Agent introduced the value by inference, convention, or a
  workflow preference that the user did not explicitly approve.
- `contradiction`: the SPEC value conflicts with the audited idea or raw
  conversation evidence.

Never mark a field `confirmed` when its evidence says the user did not specify,
did not confirm, or that the Agent inferred the value. If the evidence says
"用户未指定", "未确认", "Agent 将", "Agent inferred", or equivalent wording, the
status must be `agent_added` or `unconfirmed`, and the field must block the
backtest unless it is a documented default accepted by the user.

Never mark a field `confirmed` when the only evidence is a framework, parser,
runtime, template, or OpenXQuant default. Evidence such as "Framework default",
"runtime default", "parser default", "template default", "框架默认", "运行时默认",
or "系统默认" is not user confirmation. Likewise, do not reword a default as
"Effective StrategySpec default value", "Documented for full SPEC coverage", or
"absent from YAML" and then mark it `confirmed`; those phrases still mean the
user has not confirmed the value. It must remain blocked until the user
confirms that value or the builder writes a SPEC value that matches the audited
idea. For inherited version values, cite the actual user confirmation such as
"User confirmed the v002 Full SPEC table and confirmed v003 inherits all v002
confirmed values except TopNRanking n=2", not the default mechanism.

Market metadata defaults are material. For China ETF or A-share strategies,
`market.region: us`, `market.currency: USD`, or a generic US equity default
must block unless explicitly confirmed by the user. Return to
`next_required_phase: build` when the audited idea confirms China assets but
the effective SPEC still contains US market defaults.

Effective execution defaults are material. If the audited idea confirms a
10-trading-day rebalance but the effective SPEC also contains
`execution.rebalance.interval_days: 1`, block and return to build unless the
SPEC explicitly aligns execution rebalance semantics with the portfolio
rebalance rule or the user confirms the runtime default is intentionally
non-operative.

`default` is not a passing final status. A default may appear while the audit is
blocked and asking grouped confirmation questions, but every effective SPEC
field must become `confirmed` before `spec_audit.json` can pass. This includes
fields omitted from YAML but injected by OpenXQuant defaults, such as
execution, cost, cash, validation, metrics, empty dictionaries, and empty
lists.

Material fields include:

- train/test periods
- `validation.required_oos`
- `required_oxq_version`
- symbols, index metadata, point-in-time policy, and benchmark
- data filters and the required columns they depend on
- `data.filters.suspension_policy` and suspended-position behavior
- execution timing and fill price fields
- data warmup policy and `data.min_start_date`
- indicator parameters and `signal.indicators.*.lag_bars`
- rank-select portfolio fields such as `TopNRanking.score_col`, `n`,
  `filter_negative`, `max_weight`, `pre_filter_signal`, `weighting`, and
  `ascending`
- initial cash and cash return
- fee rate, side-specific fees, sell-side tax, minimum fee, and slippage rate
- risk-free rate and metrics profile
- exit, risk, rebalance rules, and portfolio constraints

If the user supplied only one full backtest date range and the audited idea did
not confirm OOS validation, do not treat an Agent-created IS/OOS split as
confirmed. Classify `validation.train_period`, `validation.test_period`, and
`validation.required_oos` as `agent_added` or `unconfirmed` until the user
confirms the split or explicitly accepts the default validation plan.

When the audited idea requires all close-based data to lag at least one bar,
audit every indicator that directly reads `close` for explicit `lag_bars`.
For derived indicators that read intermediate columns, require builder evidence
showing whether the inputs were already lagged or whether an additional
`lag_bars` value was confirmed. A silent default `lag_bars: 0` on a material
derived field remains blocked until explained and confirmed.

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
- `material_category`
- classification: `confirmed`, `default`, `unconfirmed`, `contradiction`, or
  `agent_added`
- evidence snippets or message references from `CONVERSATION_HISTORY_RAW`
- whether the item blocks backtest

`material_category` separates strategy logic from research and execution
assumptions. Use these values:

- `strategy_logic`: indicators, signals, rules, and strategy decision logic
- `portfolio_construction`: optimizer, ranking, weights, rebalance, constraints
- `execution_assumption`: order timing, fill price, lot size, cash handling,
  and effective `execution.initial_cash`
- `backtest_assumption`: general simulation settings not covered elsewhere
- `data_assumption`: provider, columns, adjustment, filters, warmup, universe,
  market metadata, benchmark, and point-in-time policy
- `cost_assumption`: fees, tax, minimum fee, and slippage
- `validation_assumption`: train/test windows and OOS requirements
- `risk_assumption`: risk controls and robustness settings
- `metric_assumption`: risk-free rate, metrics profile, annualization, and
  decision metrics
- `system_provenance`: OpenXQuant version and generated provenance fields

## Component Provenance

Before approving a spec for backtest, audit component provenance against the
same catalog used while building the spec:

1. Load `versions/<version_id>/04_spec_build/component_catalog.json` from the
   active version. If it is missing or stale, block with
   `next_required_phase: build`. The builder must export or refresh the catalog;
   the auditor must not write into `04_spec_build/`.

2. Record `component_catalog.json`'s internal `catalog_hash` field and compare
   it with any catalog hash recorded in `builder_phase_result.json`,
   `spec_build_notes.md`, Studio task metadata, or prior audit artifact. Do not
   use raw file SHA or canonical JSON file hash as `catalog_hash`; the formal
   backtest gate compares `spec_audit.json.catalog_hash` with
   `component_catalog.json.catalog_hash`. A hash mismatch blocks the backtest
   until the spec is rechecked against the current catalog or the user
   explicitly accepts the catalog change.
3. Verify every `signal.indicators.*.type`, `signal.rules.*.type`,
   `portfolio.type`, and any documented rule component exists in the catalog.
   Components absent from the catalog are blocking unless a separate component
   authoring workflow has already registered the custom component in the
   catalog.
4. Search catalog aliases and `recipes` for more standard canonical structures
   than the spec currently uses. Block when the audited idea matches a recipe
   but the spec decomposes it differently or uses an invented shortcut name.
5. Block Agent-created names that look semantic but are not registered, such as
   `RiskAdjustedMomentum`, when equivalent built-in components or recipes
   exist.
6. Check semantic coverage: each audited idea indicator, signal, portfolio
   optimizer, and rule must appear either as a selected catalog component or as
   a selected recipe. Missing audited semantics block the backtest.

Also check:

- SPEC fields that exist but were never requested or confirmed by the user.
- User-stated requirements that are missing from the SPEC.
- SPEC values that contradict the conversation history.
- Component choices that deviate from a matched recipe's `canonical_spec`.
- Source fields that the builder omitted, excluded, or marked unsupported.

## Unsupported Mapping Disclosure

Read `spec_mapping_notes.md` and `builder_phase_result.json` before deciding
the SPEC audit conclusion. Audit whether every user-requested source field that
does not appear in executable SPEC semantics is disclosed in
`unsupported_mappings`.

If `spec_mapping_contract.json` exists, validate it with
the Python API `oxq.spec.validate_mapping_contract` before approving the SPEC.
This is not a CLI command; do not run `oxq spec validate_mapping_contract`.
If it is missing for a builder output that claims mapping completeness, block
and return to `next_required_phase: build`. The contract must not classify strategy
semantics as `excluded_non_material`; such rows must be `mapped`,
`needs_user_confirmation`, `unsupported`, or `blocked`.

When running deterministic `spec-audit validate`, pass the mapping contract via
`--mapping-contract versions/<version_id>/04_spec_build/spec_mapping_contract.json`
so machine validation can reject unsupported strategy semantics that are marked
non-blocking.

For a passing audit, deterministic validation also enforces the builder-pass
mapping gate: every `semantic: strategy` row must be mapped and non-blocking.
Do not return `audit_conclusion: all_pass` while any strategy row is
`blocked`, `unsupported`, `needs_user_confirmation`, or `blocking: true`.

Each unsupported mapping row must include:

- `source_field`
- `requested_semantic`
- `reason`
- `disposition`
- `blocking`

Allowed dispositions are `blocked`, `deferred_framework`,
`excluded_non_material`, and `not_applicable`.

Rules:

- If `builder_phase_result.json` has `unmapped_source_fields`, every item must
  either map to a SPEC field or appear in `unsupported_mappings`.
- If any `unsupported_mappings` item has `blocking: true`, the audit must set
  `status: block`, `audit_conclusion: blocked`, and the earliest
  `next_required_phase`.
- If mapping-contract validation fails, block with
  `next_required_phase: build`; do not continue to runtime or user SPEC
  confirmation. Unsupported `strategy` semantics with `blocking: false` are an
  invalid builder handoff.
- Strategy rows with `status: needs_user_confirmation` and `blocking: false`
  are also invalid builder handoffs. Strategy assumptions needing user
  confirmation remain blocked until the user confirms them.
- If an unsupported item is non-material, explain why it is excluded from
  strategy hash semantics and set `disposition: excluded_non_material`.
- If no unsupported source fields were found, write
  `"unsupported_mappings": []` and state that no unsupported source fields were
  found in `audit_notes.md`.
- Do not let a SPEC pass when A-share data filters, stamp tax, calendar
  rebalance, cash reserve, unsupported portfolio constraints, or unsupported
  risk controls were requested but omitted without disclosure.

Examples:

- If the user says "20日收益率 / 20日波动率" and the spec uses
  `RiskAdjustedMomentum`, block it when the catalog has no such component.
  Require the canonical `volatility_adjusted_momentum` recipe:
  `NdayReturn + RollingVolatility + Ratio`.
- If the user says "取 TopN，归一化权重" and the spec uses `EqualWeight`, block
  it because `TopNRanking` is the catalog component matching that portfolio
  semantic.
- If the user says "先阈值过滤，再按分数排名取 TopN", require a boolean
  `signal.rules.*` gate and `portfolio.params.pre_filter_signal` pointing to
  that gate. Audit `weighting` and `ascending` as material allocation fields.
- If the user says "RPS / 相对强弱 / 横截面强弱排名", require registered `RPS`
  or an equivalent catalog-backed cross-sectional indicator. Audit `period`,
  `scale`, `min_symbols`, and `lag_bars`.

## Cross-Sectional Component Feasibility

Audit cross-sectional logic as a runtime feasibility issue, not only as a
component name lookup. Cross-sectional logic includes same-date winsorization,
ranking, z-score, percentile rank, neutralization, or clipping across symbols.

Use `PortfolioOptimizer first` for cross-sectional allocation transforms, but
do not force `PortfolioOptimizer` for every cross-sectional request:

- A SPEC must not claim a cross-sectional transform is implemented as a
  normal single-symbol `Indicator` unless the component is registered and
  exposes first-class cross-sectional semantics, such as builtin `RPS`, and
  preserves all-symbol same-date semantics.
- If the SPEC uses an optimizer-internal transform, verify that the audited
  idea allows the transform to live inside portfolio construction rather than
  as a reusable factor column.
- If `TopNRanking` is used, mark ranking, positive filtering, normalization,
  and `max_weight` as covered only when present. Do not mark winsorization,
  neutralization, or z-scoring as covered by `max_weight`.
- If the builder or component author forced a `PortfolioOptimizer` but the
  optimizer cannot receive required inputs, preserve required runtime evidence,
  or produce the required semantics, block with `framework_unsupported`.
- If no current component kind can faithfully represent the cross-sectional
  behavior, block with `next_required_phase: build` or component authoring
  notes that request explicit framework development before formal backtest.

## Gate

Any `default`, `unconfirmed`, `contradiction`, or `agent_added` effective field
blocks formal backtest until the user confirms the grouped assumption or
provides a replacement value. Any blocking component provenance issue blocks
backtest. Train/test splits and required OOS settings are material and must not
pass silently.

Always group related fields instead of asking one question per YAML key:

- data warmup and local data coverage
- execution assumptions
- cost assumptions
- train/test split
- cash and risk-free assumptions
- benchmark and success metric
- component/catalog provenance

Ask the user to either confirm the group or provide replacement values. If the
user provides replacement values, do not edit `strategy_spec.yaml` in this
audit. Write a blocked audit with `next_required_phase: build`; the builder
must apply the change, re-run
`oxq spec validate versions/<version_id>/04_spec_build/strategy_spec.yaml`, and
then return for another audit pass.

## Two-Step Spec Audit Gate

The SPEC audit has two semantic states before downstream runtime work:

- `audit_conclusion: all_pass`: the auditor found no remaining provenance,
  calibration, default, component, or recipe blockers.
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
`confirmed`. If the user rejects or edits any row, return to the builder; do not
update `strategy_spec.yaml` from the auditor phase. The builder must rerun
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
    "spec_audit_hash": "sha256:<pre-confirmation spec_audit hash>"
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
`spec_audit_path`, and `spec_audit_hash`. `spec_audit_hash` is the
pre-confirmation `spec_audit.json` hash. Pending all-pass audits must not
invent a confirmation event.

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

Compute `conversation_hash` from the exact raw conversation input supplied to
this skill. After writing a blocked or failed `spec_audit.json`, run non-strict
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

Schema validation only proves the artifact shape. It does not prove the
semantic audit is correct; that responsibility stays in this skill.

This skill does not compile the strategy and does not compare
`strategy_spec.yaml` with `compiled_plan.json`. That is the
`audit-runtime-semantics` skill's boundary.

## Output

Report a compact summary:

- audited idea artifact hashes
- confirmed fields
- default fields awaiting checklist confirmation
- unconfirmed fields that block progress
- selected catalog components and recipes
- spec calibration findings
- component provenance issues, including catalog hash mismatch, missing
  components, non-canonical recipe decomposition, or missing user-requested
  semantics
- unsupported_mappings disclosure and whether no unsupported source fields were
  found
- path to `spec_audit.json`
- Full Spec Confirmation Table when `audit_conclusion: all_pass` is reached
  but `user_confirmation_status` is not confirmed
- Default Confirmation Checklist when defaults need user approval
- blocking confirmation questions

Do not run or approve a backtest while blocking fields remain. After this skill
is confirmed by the user, the next formal gate is `audit-runtime-semantics`.
