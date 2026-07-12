# Component And Mapping Audit

Use this reference for component provenance, unsupported mapping disclosure,
mapping-contract validation, and cross-sectional feasibility.

## Contents

- Component Provenance
- Unsupported Mapping Disclosure
- Cross-Sectional Component Feasibility

## Component Provenance

Before approving a spec for backtest, audit component provenance against the
same catalog used while building the spec:

1. Load `<phase_paths.04_spec_build>/component_catalog.json` from the
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

If `spec_mapping_contract.json` exists, validate it with the Python API
`oxq.spec.validate_mapping_contract` before approving the SPEC. This is not a
CLI command; do not run `oxq spec validate_mapping_contract`. If it is missing
for a builder output that claims mapping completeness, block and return to
`next_required_phase: build`. The contract must not classify strategy
semantics as `excluded_non_material`; such rows must be `mapped`,
`needs_user_confirmation`, `unsupported`, or `blocked`.

When running deterministic `spec-audit validate`, pass the mapping contract via
`--mapping-contract <phase_paths.04_spec_build>/spec_mapping_contract.json`
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

- A SPEC must not claim a cross-sectional transform is implemented as a normal
  single-symbol `Indicator` unless the component is registered and exposes
  first-class cross-sectional semantics, such as builtin `RPS`, and preserves
  all-symbol same-date semantics.
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
