# Source Mapping, Catalog, And Feasibility

Use this reference before component selection, recipe selection, unsupported
mapping disclosure, mapping-contract validation, or cross-sectional feasibility
decisions.

## Contents

- Source Mapping And Unsupported Disclosure
- Component Catalog Gate
- Cross-Sectional Component Feasibility
- YAML Patterns

## Source Mapping And Unsupported Disclosure

Build from the audited idea as a source mapping exercise, not as a free-form
YAML drafting task. Before completing the builder phase, write
`<phase_paths.04_spec_build>/spec_mapping_notes.md` with:

- every audited idea or source config field that materially affected the SPEC
- its target `strategy_spec.yaml` field path, selected component, or selected
  canonical recipe
- any `unmapped_source_fields` that were requested by the user but cannot be
  represented by the current SPEC schema, validator, catalog, or runtime
- any fields excluded because they are run/report/workspace configuration
  rather than material strategy semantics
- whether each excluded or unsupported item is blocking

Also copy the structured unsupported summary into `builder_phase_result.json`
as `unsupported_mappings`. Each item must include `source_field`,
`requested_semantic`, `reason`, `disposition`, and `blocking`.

Unsupported strategy semantics are blocking by default. If a user-requested
strategy behavior cannot be faithfully represented by current SPEC schema,
validator, catalog, or runtime, every related `unsupported_mappings` row and
mapping-contract row must use `blocking: true`, and the builder phase must
return `status: blocked`. Do not mark a material strategy semantic as
non-blocking merely because there is an approximate proxy, partial coverage, or
future framework path.

Treat these as material blocking examples unless the exact supported behavior
is explicitly mapped and confirmed:

- calendar-aware last trading day/week-end/month-end rebalance when only a
  bar-count proxy such as `interval_days` is available
- same-bar or before-close fills when the executable timing is `next_open`
- non-tradable policies beyond the confirmed supported suspension fields
- unresolved `latest_available` dates, price adjustment mode, required data
  columns, or provider/source semantics
- cross-sectional winsorization, clipping, neutralization, or z-scoring not
  implemented by a registered cross-sectional component or optimizer
- any close-derived indicator missing the user-confirmed `lag_bars` policy

Also write `<phase_paths.04_spec_build>/spec_mapping_contract.json`
and validate it with the Python API `oxq.spec.validate_mapping_contract`. This
is not a CLI command; do not run `oxq spec validate_mapping_contract`. The
contract must include `schema_version: 1`, `source_format`, a non-empty
`source_fields` inventory, and `field_mappings`. Build `source_fields` by
flattening every field present in the source artifact before mapping begins.
Every inventory field must appear in exactly one mapping row, and every mapping
row must name a field from that inventory. Missing, duplicate, or invented
source fields block builder pass. Each mapping row must identify:

- `source_field`
- `semantic`: `strategy`, `run`, `report`, `studio`, `metadata`, or
  `unsupported`
- `status`: `mapped`, `needs_user_confirmation`, `unsupported`,
  `excluded_non_material`, or `blocked`
- `target_field` when mapped to `strategy_spec.yaml`
- `confirmation_required`
- `blocking`
- `reason`

Every `field_mappings` row must have a non-empty `reason`.
Do not leave `reason` as an empty string, `null`, `n/a`, or a placeholder such
as `mapped`.
The reason should explain why the source field maps to that target, why it is
outside `strategy_spec.yaml`, or why it blocks/needs confirmation.

Do not label run, report, studio, or metadata source fields as `semantic: strategy` merely because they came from the same source brief.
Use `excluded_non_material` only with `semantic: run`, `report`, `studio`, or `metadata`; it is invalid for strategy semantics. For source fields that belong
to the strategy but cannot be represented, use `status: unsupported` or
`status: blocked` with `semantic: strategy`, `blocking: true`, and a concrete
reason.

Before writing `target_field` values for `semantic: strategy` rows, derive the
allowed field paths from the effective StrategySpec, not from conceptual YAML
paths:

```python
from oxq.spec import StrategySpec

effective = StrategySpec.from_yaml("<phase_paths.04_spec_build>/strategy_spec.yaml").to_effective_dict()
```

Flatten that effective dictionary and use only those field paths, such as
`execution.rebalance.frequency`, `execution.lot_size_config.default`,
`portfolio.params.n`, `signal.indicators.<name>.lag_bars`, or
`universe.symbols`. Copy the flattened path exactly; for example use
`signal.indicators.<name>.lag_bars`, not `signal.indicators.<name>.params.lag_bars`,
when the parser normalizes `lag_bars` out of `params`. Do not use parent container paths
like `portfolio`, conceptual absent paths like `execution.leverage.allowed`, or
YAML-only paths that are not present in the effective dictionary. If a source
semantic maps only to an absent or non-operative path, return to the SPEC shape
and fix the YAML or mark the source semantic unsupported; do not make the
mapping contract pass by inventing a target path.

Strategy semantics must not be hidden as `excluded_non_material`. If a
strategy source field cannot be represented, mark it `unsupported` or
`blocked` with a reason and set `blocking: true`.

If a strategy source field is marked `needs_user_confirmation`, it is still
blocking until the user confirms it. The mapping-contract row must use
`confirmation_required: true` and `blocking: true`; the builder phase must not
return `status: pass` while any strategy row needs user confirmation.

Validate the mapping contract before completing the builder phase. In an
installed research workspace, use the resolved runner's virtualenv Python:

```bash
RUNNER="/path/to/resolved/oxq"
PYTHON="$(dirname "$RUNNER")/python"
"$PYTHON" - <<'PY'
from oxq.spec import validate_mapping_contract_file

result = validate_mapping_contract_file("<phase_paths.04_spec_build>/spec_mapping_contract.json")
print(result)
raise SystemExit(0 if result["status"] == "pass" else 1)
PY
```

When returning `builder_phase_result.json` with `status: pass`, also run the
builder-pass gate:

```bash
RUNNER="/path/to/resolved/oxq"
PYTHON="$(dirname "$RUNNER")/python"
"$PYTHON" - <<'PY'
from oxq.spec import validate_mapping_contract_for_builder_pass_file

result = validate_mapping_contract_for_builder_pass_file("<phase_paths.04_spec_build>/spec_mapping_contract.json")
print(result)
raise SystemExit(0 if result["status"] == "pass" else 1)
PY
```

The base mapping contract validator checks structure and may allow
`status: blocked` as a legal handoff state. The builder-pass gate is stricter:
`source_fields` must be non-empty and have exact once-only `field_mappings`
coverage, and every `semantic: strategy` row must be mapped and non-blocking
before the builder may return `status: pass`.
Run both validators again after every mapping-contract edit. Do not write `builder_phase_result.json` with
`status: pass` until both validator outputs are `status: pass`.

Use these dispositions:

- `blocked`: required strategy behavior cannot be represented now
- `deferred_framework`: requires open-xquant framework/runtime development
- `excluded_non_material`: belongs outside `strategy_spec.yaml`
- `not_applicable`: source field did not apply after user-confirmed changes

If any `unsupported_mappings` row has `blocking: true`, or any strategy
mapping-contract row has `status: blocked`, `status: unsupported`,
`status: needs_user_confirmation`, or `blocking: true`, the builder phase must
return `status: blocked` and `next_required_phase: brainstorm`, `build`,
`data_inspection`, `runtime_audit`, or `component_authoring` as appropriate.
Do not silently drop source fields such as A-share data filters, stamp tax,
calendar rebalance, cash reserve, unsupported portfolio constraints, or
unsupported risk controls.

## Component Catalog Gate

After the audited idea gate passes and before editing `strategy_spec.yaml`, run
the component catalog gate. This gate is mandatory for every new spec and every
material component edit:

1. Load `component_catalog.json` from the research directory when it exists.
   If it is missing or stale, create it with:

   ```bash
   uv run oxq registry export --out <phase_paths.04_spec_build>/component_catalog.json
   ```

   Do not read the full catalog into the model context. Query only the needed
   names, aliases, recipes, and hashes with structured tools such as `jq`:

   ```bash
   jq -r '.catalog_hash' <phase_paths.04_spec_build>/component_catalog.json
   jq '.indicators[] | select(.name=="NdayReturn" or .name=="RollingVolatility" or .name=="Ratio" or .name=="RPS")' <phase_paths.04_spec_build>/component_catalog.json
   jq '.signals[] | select(.name=="Threshold")' <phase_paths.04_spec_build>/component_catalog.json
   jq '.portfolios[] | select(.name=="TopNRanking")' <phase_paths.04_spec_build>/component_catalog.json
   jq '.recipes[] | select(.name=="volatility_adjusted_momentum" or .name=="threshold_then_rank_top_n" or .name=="rps_top_n_rotation")' <phase_paths.04_spec_build>/component_catalog.json
   ```

2. Search exact names and aliases in the catalog for every requested
   indicator, signal, portfolio optimizer, and rule. Prefer `source: builtin`
   components over custom components whenever they satisfy the audited idea.
3. Search `recipes` before composing custom indicator chains or portfolio
   structures. Match exact recipe names, aliases, and definitions against the
   audited idea semantics.
4. If a recipe matches, use its `canonical_spec` structure and fill only the
   placeholders confirmed in the audited idea, such as `$period`, `$score_col`,
   or `$n`.
5. If no built-in component or recipe matches a requested semantic component,
   do not invent a component name and do not route to component code creation.
   Mark the builder phase as blocked with `needs_custom_component`.
6. Record selected components, selected recipes, and the catalog payload's
   `catalog_hash` field in `spec_build_notes.md` or equivalent task-local
   build notes. Do not replace `catalog_hash` with a raw file SHA or canonical
   JSON file hash; if file integrity is needed, record it separately as
   `component_catalog_file_hash`.

Examples:

- User asks for "20日收益率 / 20日波动率": use the
  `volatility_adjusted_momentum` recipe, not an invented
  `RiskAdjustedMomentum` component.
- User asks for "SMA 金叉": use the `sma_golden_cross` recipe.
- User asks for "ROC timing": use the `roc_timing` recipe.
- User asks for "TopN 正动量轮动": use the
  `top_n_positive_momentum_rotation` recipe.
- User asks for "先阈值过滤，再按分数排名取 TopN": match
  `threshold_then_rank_top_n`, then use `TopNRanking.pre_filter_signal` to
  connect the confirmed boolean filter rule to the ranking optimizer.
- User asks for "RPS / 相对强弱 / 横截面强弱排名": use the `RPS` indicator or
  the `rps_top_n_rotation` recipe when the portfolio is rank-select TopN.
- User asks to "取 TopN，归一化权重": use `TopNRanking`, not `EqualWeight`.

## Cross-Sectional Component Feasibility

When an audited idea asks for cross-sectional logic such as same-date
winsorization, ranking, z-score, percentile rank, neutralization, or clipping
across symbols, first decide which runtime phase can see the required data.
Most `Indicator` and `Signal` components receive a single symbol's DataFrame.
Cross-sectional indicators are supported only when the component explicitly
implements `compute_cross_section(dict[str, DataFrame])`, as builtin `RPS`
does. `PortfolioOptimizer` components receive all-symbol same-date inputs
through `dict[str, DataFrame]`.

Use `PortfolioOptimizer first` as a feasibility preference, not as a forced
solution:

- If the requested cross-sectional value is a reusable factor column such as
  RPS percentile rank, prefer a registered cross-sectional `Indicator` that
  exposes `compute_cross_section`.
- If the cross-sectional behavior can be expressed as allocation logic using
  existing indicator columns, mark a custom component request with
  `suggested_kind`: `PortfolioOptimizer` and `feasibility_status`: `candidate`.
- If the behavior must produce a reusable factor column before portfolio
  construction, requires history across prior allocation states, needs data the
  optimizer does not receive, or would not be visible in audited runtime
  artifacts, do not force `PortfolioOptimizer`. Mark the request as
  `feasibility_status`: `unsupported` with `blocked_reason:
  framework_unsupported`.
- If an existing builtin optimizer such as `TopNRanking` covers only part of
  the behavior, record the covered and missing semantics separately. Do not
  claim winsorization or neutralization is implemented by a max-weight cap.

For example, a request for "risk-adjusted momentum cross-sectional
winsorization before TopN allocation" should first be represented as a custom
`PortfolioOptimizer` candidate such as `WinsorizedTopNRanking`. If the exact
semantics cannot be implemented without a first-class cross-sectional factor
stage, block with `framework_unsupported` rather than inventing an
`Indicator`.

## YAML Patterns

For categorical custom signals, declare output domain as rule metadata:

```yaml
signal:
  rules:
    timing:
      type: CustomTiming
      output_domain: [BUY, SELL, HOLD]
      params:
        column: close
portfolio:
  type: SignalToPosition
  params:
    signal: timing
```

For rebalance throttling, use the built-in rule:

```yaml
portfolio:
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 10
```

Do not also set a conflicting `execution.rebalance.interval_days` value.

For calendar rebalance, use execution schedule:

```yaml
execution:
  rebalance:
    frequency: monthly
    schedule: month_start
```

For data filters, map confirmed requirements explicitly:

```yaml
data:
  required_columns: [open, high, low, close, volume, is_st, is_suspended]
  filters:
    exclude_st: true
    exclude_suspended: true
    suspension_policy: hold_existing
    lookahead_policy: point_in_time_required
```
