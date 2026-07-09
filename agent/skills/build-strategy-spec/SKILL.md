---
name: build-strategy-spec
description: >-
  Build open-xquant strategy_spec.yaml files from audited strategy idea briefs
  for multi-Agent systems; stops after deterministic validation and writes a
  builder phase result for downstream orchestration.
---

# Strategy Builder

Build or edit a `strategy_spec.yaml` from an audited strategy idea brief. This
skill is for multi-Agent systems where strategy brainstorming, idea audit,
SPEC construction, SPEC audit, execution, monitoring, and reporting may be
handled by separate Agents.

## Version-Governed Artifact Gate

Before any catalog export, SPEC initialization, SPEC edit, or validation, read
`current.json` and use `active_version` as `version_id`. If no active version
exists, block and return to `manage-strategy-version`.

The required audited idea inputs are:

```text
versions/<version_id>/01_brainstorm/strategy_idea_brief.json
versions/<version_id>/02_idea_audit/strategy_idea_audit.json
```

This builder may write only inside:

```text
versions/<version_id>/04_spec_build/
```

Required outputs are:

```text
versions/<version_id>/04_spec_build/strategy_spec.yaml
versions/<version_id>/04_spec_build/component_catalog.json
versions/<version_id>/04_spec_build/spec_build_notes.md
versions/<version_id>/04_spec_build/spec_mapping_notes.md
versions/<version_id>/04_spec_build/spec_mapping_contract.json
versions/<version_id>/04_spec_build/builder_phase_result.json
```

Do not write root-level `strategy_spec.yaml`, `component_catalog.json`,
`spec_build_notes.md`, `spec_mapping_notes.md`,
`spec_mapping_contract.json`, or
`builder_phase_result.json`.

## Scope

Do:

- require `strategy_idea_brief.json` and passing `strategy_idea_audit.json`
  before any SPEC or catalog work
- convert the audited strategy idea into `strategy_spec.yaml`
- use the component catalog and canonical recipes before choosing components
- preserve explicit user requirements and user-confirmed candidate values
- write concise `spec_build_notes.md` when component or recipe choices matter
- write `spec_mapping_notes.md` that maps audited idea fields to SPEC fields
  and discloses unsupported or deliberately excluded source fields
- run deterministic spec validation
- write `builder_phase_result.json` for the downstream orchestrator

Do not:

- produce `spec_audit.json`
- approve assumptions on behalf of the user
- call audit skills
- ask the user to complete brainstorm phases
- inspect, list, or read market data files, parquet files, manifests, or
  provider directories
- download market data
- run `oxq strategy compile`
- run `oxq backtest run`
- attach provenance for runs, backtests, monitoring, or reports
- run monitoring, robustness, report writing, or report review
- describe an unaudited spec as ready for formal research

## Audited Idea Input Gate

Before doing any SPEC work, require both artifacts:

- `strategy_idea_brief.json`
- `strategy_idea_audit.json`

The idea audit must have `status: pass`, `idea_workflow_pass: true`, and
`next_required_phase: build`. If either artifact is missing, stale, blocked,
or failing, stop and return a blocked `builder_phase_result.json` with
`next_required_phase: brainstorm`.

Hard gate:

- Do not run `oxq spec init` before `strategy_idea_audit.json` passes.
- Do not run `oxq registry export` before `strategy_idea_audit.json` passes.
- Do not write or edit `strategy_spec.yaml` before `strategy_idea_audit.json`
  passes.

If the audited brief contains unresolved defaults, candidates, contradictions,
or missing phase evidence, do not choose values. Return the issue to
`brainstorm-strategy-idea` through the coordinator.

## Runner Resolution

In a new research directory, `uv run oxq` may fail because open-xquant is
installed as a long-lived Agent capability, not as a package in that directory.
Before running commands after the audited idea gate passes:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

When this skill needs a Python SDK snippet in an installed research workspace,
use the resolved runner's virtualenv Python from the same `bin` directory as
the runner. `oxq run python` does not exist, and `uv run python` is only
appropriate inside a source checkout where `oxq` is installed for that project.

## OpenXQuant Version Provenance

Every generated `strategy_spec.yaml` must distinguish the SPEC schema version
from the OpenXQuant package version:

- `schema_version` is the SPEC schema version. It changes only when the
  strategy configuration schema changes.
- `required_oxq_version` is the OpenXQuant package version used by the
  resolved runner for this builder phase.

Do not change `schema_version` to the package version. When writing or
materially editing `strategy_spec.yaml`, write `required_oxq_version` from the
resolved OpenXQuant runner/package version used by this builder phase. If the
runner cannot report a version, block the builder phase rather than emitting a
SPEC with an empty `required_oxq_version`.

Read the package version with the resolved runner's virtualenv Python. Do not
call `oxq --version`, `oxq version`, or `oxq spec show`; those are not the
builder version source. In an installed research workspace:

```bash
RUNNER="/path/to/resolved/oxq"
PYTHON="$(dirname "$RUNNER")/python"
"$PYTHON" - <<'PY'
import oxq

print(oxq.__version__)
PY
```

`required_oxq_version` is provenance, not a strategy default chosen for the
user. Do not ask the user to invent or confirm it before writing the SPEC, but
ensure the later SPEC confirmation table shows it because a package-version
change affects reproducibility. Record the resolved version in
`builder_phase_result.json`.

## Current SPEC Template Shape

Use the current `StrategySpec` YAML shape, not the mapping contract schema or
any older Studio YAML shape:

- `schema_version` must be the string `"0.1"`, not integer `1`.
- Use top-level `strategy_id` and `name`; do not create a nested `strategy.id`
  or `strategy.name` object.
- `validation.train_period` and `validation.test_period` are two-item lists
  such as `["2021-01-01", "2025-12-31"]`, not `{start, end}` maps.
- `cost` is a top-level section; do not nest `cost` under `execution`.
- `market.asset_class`, `market.region`, `market.currency`, and
  `market.calendar` are scalar fields, not `market.regions`.

If uncertain, use `oxq spec init` only as a mechanical shape reference after
the audited idea gate passes, then replace candidate values with audited
values. Do not keep initializer defaults as confirmations.

Never run `oxq spec init` without an explicit `--out` path under
`versions/<version_id>/04_spec_build/`. The default command writes
root-level `strategy_spec.yaml`, which is a workspace layout violation even if
the file is deleted later. Do not create, read, or use root-level
`strategy_spec.yaml` as a template reference. If such a file is accidentally
created, delete it, record `layout_violation` in `builder_phase_result.json`,
and keep the builder result blocked.

## Data Inspection Boundary

This builder does not own data coverage or data quality investigation. Do not
list `data_dir`, read parquet files, inspect columns, call `list_symbols`,
call `inspect_symbol`, or resolve date coverage from local market data. The
`explore-data` skill and `oxq-data-inspection-worker` own those actions and
write:

```text
versions/<version_id>/05_data_inspection/data_inspection_result.json
versions/<version_id>/05_data_inspection/data_availability_report.md
```

The builder may use a data fact only when it is already present in the audited
idea, explicitly confirmed by the user, or recorded in a prior
`data_inspection_result.json` for the active version. Cite that artifact in
`spec_build_notes.md` and `spec_mapping_contract.json`.

If `latest_available`, required column coverage, common symbol date range,
price adjustment evidence, provider readiness, or suspension/tradability data
availability is unknown, do not infer it and do not inspect the files. Return
`status: blocked` with `next_required_phase: data_inspection`. If a prior data
inspection artifact says required data is unavailable, preserve the requested
strategy semantic in `spec_mapping_contract.json` and block instead of
removing the requirement from the SPEC.

## SPEC Audit Repair Handoff

When a blocked SPEC audit result returns `next_required_phase: build`, read
`versions/<version_id>/06_spec_audit/spec_audit.json` and `audit_notes.md`
before editing the SPEC. The auditor is read-only and does not repair YAML.
The builder owns the repair.

For contradictions that include `effective_field_path`, `source_yaml_path`, and
`builder_required_fix`:

- move the value to the effective field path that OpenXQuant actually parses
- remove the non-operative YAML path when it is not part of
  `StrategySpec.from_yaml(...).to_effective_dict()`
- preserve the original user-confirmed value and cite the audit finding in
  `spec_build_notes.md`
- update `spec_mapping_notes.md`, `spec_mapping_contract.json`, and
  `builder_phase_result.json` so the handoff records the repair
- rerun `oxq spec validate` on
  `versions/<version_id>/04_spec_build/strategy_spec.yaml`

Example: if the SPEC audit says `source_yaml_path: portfolio.initial_cash`,
`effective_field_path: execution.initial_cash`, and
`builder_required_fix: move the value to execution.initial_cash and remove
portfolio.initial_cash`, write the confirmed value under
`execution.initial_cash` and remove `portfolio.initial_cash`. Do not preserve a
known non-operative field merely as documentation.

## Stable Candidate Spec Values

Use current stable CLI behavior only as implementation candidates after the
audited idea gate passes. Each value must trace back to a confirmed value in
`strategy_idea_brief.json` or to an explicit candidate accepted by
`strategy_idea_audit.json`. Parser, runtime, or template defaults are not
confirmed merely because they exist:

- `universe.type: static`
- `universe.type: index` only when the audited idea confirms an index identity
  and provides a local constituent snapshot in `universe.symbols`; record
  `index_key` or `index_code`, `point_in_time`, and
  `survivorship_bias_policy`
- `data.provider: local`
- `market.calendar: XNYS`, `ARCX`, `XSHG`, or `XSHE`
- For China ETF or A-share ideas, write confirmed market metadata explicitly:
  `market.asset_class: etf` or the confirmed asset class,
  `market.region: cn`, `market.currency: CNY`, and `market.calendar: XSHG`
  or `XSHE` as confirmed. Do not leave framework defaults such as
  `region: us` or `currency: USD` in an effective SPEC for China assets.
- `signal.signal_time: close_t`
- `signal.indicators.<name>.lag_bars` when the audited idea confirms that an
  indicator must be delayed to avoid same-bar availability or publication lag
- When the audited idea says all close-based data must lag at least one bar,
  every indicator that directly reads `close` must write an explicit
  `lag_bars` value. For derived indicators such as `Ratio` that read already
  lagged intermediate columns, either write the confirmed extra lag or record in
  `spec_build_notes.md` why no additional lag is needed; do not leave the
  default unexplained.
- `execution.trade_time: next_open`, `execution.fill_price_mode: next_open`,
  and explicit execution semantics:
  `execution.order_timing: next_session_open`,
  `execution.price_bar: next_session`,
  `execution.price_type: open`
  only when the audited idea confirms next-session open execution. If the
  audited idea says same-day, before-close, market-on-close, or another timing
  not executable by the current runtime, block the builder phase instead of
  substituting `next_open`.
- `execution.cash_annual_return` and `execution.lot_size_config`
- `metrics.profile: open_xquant_default` when confirmed by the idea brief
- positive `cost.fee_rate`
- positive `cost.slippage_rate`
- `cost.buy_fee_rate`, `cost.sell_fee_rate`, `cost.sell_tax_rate`, or
  `cost.stamp_tax` only when side-specific fees or sell-side tax were
  confirmed; do not infer them from market convention without confirmation
- `data.required_columns` when the user confirmed required bar columns.
- `data.filters.exclude_st`, `exclude_suspended`,
  `exclude_new_listed_days`, `limit_up_policy`, `limit_down_policy`, and
  `suspension_policy`, and `lookahead_policy` when confirmed by the audited
  idea. Add every required filter column to `data.required_columns`.
- For a confirmed suspended/non-tradable policy of "no new positions, hold
  existing positions, and scale remaining tradable targets", map the supported
  suspension portion explicitly to `data.filters.exclude_suspended: true`,
  `data.filters.suspension_policy: hold_existing`, and add `is_suspended` to
  `data.required_columns`. If data inspection has not confirmed that
  `is_suspended` exists, keep the requirement visible and block with
  `next_required_phase: data_inspection`; do not drop the column because it is
  unverified or missing. If the user requested broader non-tradability that
  also needs limit-up, limit-down, missing quotes, exchange halt status, or
  custom tradable masks, map only confirmed supported columns and block the
  unsupported remainder instead of calling it covered.
- `execution.lot_size` and `execution.lot_size_config.default` when the user
  confirmed lot size; keep them consistent.
- `execution.rebalance.frequency` and `execution.rebalance.interval_days`
  when the effective runtime SPEC otherwise injects a conflicting daily
  rebalance default. Keep this aligned with any
  `portfolio.rules.rebalance.params.interval_days`.
- `execution.rebalance.frequency: weekly` with `schedule: week_start`, or
  `frequency: monthly` with `schedule: month_start`, when the audited idea
  confirms calendar rebalance. Do not write month-end or week-end schedules.
- `portfolio.rules` only for the audited runtime whitelist:
  `RebalanceFrequencyRule`, `StopLossRule`, `TakeProfitRule`,
  `TrailingStopRule`, `MaxDrawdownRisk`, `DailyLossLimitRisk`, and
  `MaxHoldingsRule`. Every required param must be explicit.
- `portfolio.constraints.max_weight`, `min_weight`, `max_holdings`, and
  `cash_reserve` when confirmed. Do not write
  `portfolio.constraints.min_position_value`; it is parsed but blocked because
  the current runtime cannot execute it.
- `robustness.parameter_perturbation`, `robustness.cost_multiplier`, and
  `robustness.regime_analysis` when the audited idea confirmed robustness
  requirements that the SPEC schema can represent.
- `decision_policy.promote_if` and `decision_policy.reject_if` when the audited
  idea confirmed promotion or rejection thresholds. Do not invent unsupported
  keys, but do not write ad hoc keys such as `decision_policy.pass.conditions`
  because the schema drops them from the effective SPEC.
- explicit data warmup policy through `data.min_start_date` when indicators or
  rules need lookback data before the evaluated interval
- `portfolio.type: EqualWeight` only for boolean signal filters
- `portfolio.type: TopNRanking` for rank-select strategies. Confirm and write
  `score_col`, `n`, `filter_negative`, `max_weight`, `weighting`, and
  `ascending`. Use `pre_filter_signal` only when it references a confirmed
  boolean `signal.rules.*` gate.
- `signal.indicators.<name>.type: RPS` for confirmed cross-sectional relative
  price strength percentile rank. Confirm `period`, `scale`, `min_symbols`,
  and any `lag_bars`; do not replace RPS with a single-symbol rank.
- `ROC` + `ROCTiming` + `SignalToPosition` for single-symbol timing strategies
  that need explicit `BUY` / `SELL` / `HOLD` and HOLD-maintains-position
  semantics

If the audited idea supplies one complete backtest period and does not confirm
an IS/OOS split, encode the full period as `validation.test_period` with
`validation.required_oos: false`. Do not split the full period into train/test
or set `required_oos: true` unless the audited idea explicitly confirms that
validation plan.

Do not replace `latest_available` with the current calendar date by inference.
Do not inspect market data to resolve it inside this builder phase. Resolve it
only from a prior `data_inspection_result.json` or a user-confirmed data
snapshot, record the resolution evidence, and return a blocked builder result
with `next_required_phase: data_inspection` when the effective end date is
still unresolved.

Do not choose an arbitrary calendar date to make `oxq spec validate` pass when
the audited idea says `latest_available`. A validation error caused by an
unresolved data-inspection dependency is a blocked handoff, not permission to
invent a date.

For lookback indicators, define data warmup deliberately:

- If the audited idea requires a true full-interval evaluation from the first
  test date, set `data.min_start_date` earlier than the evaluation start so the
  largest lookback has enough prior bars.
- If the audited idea confirms first-window warmup NaNs or cash behavior,
  leave `data.min_start_date` empty only after recording that policy in
  `spec_build_notes.md`.
- Do not silently let two otherwise identical specs differ only because one
  Agent fetched warmup history and another did not.

## Source Mapping And Unsupported Disclosure

Build from the audited idea as a source mapping exercise, not as a free-form
YAML drafting task. Before completing the builder phase, write
`versions/<version_id>/04_spec_build/spec_mapping_notes.md` with:

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

Also write `versions/<version_id>/04_spec_build/spec_mapping_contract.json`
and validate it with the Python API `oxq.spec.validate_mapping_contract`. This
is not a CLI command; do not run `oxq spec validate_mapping_contract`. The
contract must include `schema_version: 1`, `source_format`, and
`field_mappings`. Each
mapping row must identify:

- `source_field`
- `semantic`: `strategy`, `run`, `report`, `studio`, `metadata`, or
  `unsupported`
- `status`: `mapped`, `needs_user_confirmation`, `unsupported`,
  `excluded_non_material`, or `blocked`
- `target_field` when mapped to `strategy_spec.yaml`
- `confirmation_required`
- `blocking`
- `reason`

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

result = validate_mapping_contract_file("versions/<version_id>/04_spec_build/spec_mapping_contract.json")
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

result = validate_mapping_contract_for_builder_pass_file("versions/<version_id>/04_spec_build/spec_mapping_contract.json")
print(result)
raise SystemExit(0 if result["status"] == "pass" else 1)
PY
```

The base mapping contract validator checks structure and may allow
`status: blocked` as a legal handoff state. The builder-pass gate is stricter:
every `semantic: strategy` row must be mapped and non-blocking before the
builder may return `status: pass`.

Use these dispositions:

- `blocked`: required strategy behavior cannot be represented now
- `deferred_framework`: requires OpenXQuant framework/runtime development
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
   uv run oxq registry export --out versions/<version_id>/04_spec_build/component_catalog.json
   ```

   Do not read the full catalog into the model context. Query only the needed
   names, aliases, recipes, and hashes with structured tools such as `jq`:

   ```bash
   jq -r '.catalog_hash' versions/<version_id>/04_spec_build/component_catalog.json
   jq '.indicators[] | select(.name=="NdayReturn" or .name=="RollingVolatility" or .name=="Ratio" or .name=="RPS")' versions/<version_id>/04_spec_build/component_catalog.json
   jq '.signals[] | select(.name=="Threshold")' versions/<version_id>/04_spec_build/component_catalog.json
   jq '.portfolios[] | select(.name=="TopNRanking")' versions/<version_id>/04_spec_build/component_catalog.json
   jq '.recipes[] | select(.name=="volatility_adjusted_momentum" or .name=="threshold_then_rank_top_n" or .name=="rps_top_n_rotation")' versions/<version_id>/04_spec_build/component_catalog.json
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

## Build Flow

Initialize when no spec exists, but only after the audited idea gate and
component catalog gate pass:

```bash
uv run oxq spec init "<audited strategy idea>" --out versions/<version_id>/04_spec_build/strategy_spec.yaml
```

This command is only a mechanical initializer. Replace any template values
that are not backed by `strategy_idea_brief.json` and
`strategy_idea_audit.json`.

Do not run the initializer with its default output path. The only acceptable
initializer target is `versions/<version_id>/04_spec_build/strategy_spec.yaml`
or a temporary file inside the same `04_spec_build` directory.

Edit `strategy_spec.yaml` so it contains only audited strategy values and
documented implementation choices. The downstream spec auditor must confirm
every effective field, including parser/runtime defaults injected by
OpenXQuant, before formal backtest.

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

## Validate And Output

Run deterministic validation:

```bash
uv run oxq spec validate versions/<version_id>/04_spec_build/strategy_spec.yaml
```

Fix fatal validation errors before completing the builder phase only when they
are mechanical SPEC-shape or translation errors. If a fatal validation error
comes from an intentionally unresolved blocked dependency such as
`latest_available`, unconfirmed required data columns, or unavailable
tradability data, do not invent replacement values. Keep `validation.status:
fail`, return `status: blocked`, set `next_required_phase: data_inspection`
or the earliest owning phase, and explain the blocker in
`builder_phase_result.json`. Warnings are allowed, but record them for the
downstream orchestrator.

Write `builder_phase_result.json` after validation:

```json
{
  "status": "pass | blocked | fail",
  "strategy_spec": "versions/<version_id>/04_spec_build/strategy_spec.yaml",
  "strategy_idea_brief": "versions/<version_id>/01_brainstorm/strategy_idea_brief.json",
  "strategy_idea_audit": "versions/<version_id>/02_idea_audit/strategy_idea_audit.json",
  "strategy_idea_brief_hash": "sha256:<hash>",
  "strategy_idea_audit_hash": "sha256:<hash>",
  "required_oxq_version": "0.1.0",
  "component_catalog": "versions/<version_id>/04_spec_build/component_catalog.json",
  "catalog_hash": "<component_catalog.catalog_hash>",
  "spec_build_notes": "versions/<version_id>/04_spec_build/spec_build_notes.md",
  "spec_mapping_notes": "versions/<version_id>/04_spec_build/spec_mapping_notes.md",
  "spec_mapping_contract": "versions/<version_id>/04_spec_build/spec_mapping_contract.json",
  "validation": {
    "status": "pass | fail",
    "spec_hash": "sha256:<hash>",
    "warnings": [],
    "errors": []
  },
  "selected_components": [],
  "selected_recipes": [],
  "unmapped_source_fields": [],
  "unsupported_mappings": [
    {
      "source_field": "portfolio.constraints.min_position_value",
      "requested_semantic": "minimum notional position size",
      "reason": "current SPEC parses this field but the audited runtime cannot execute it",
      "disposition": "blocked",
      "blocking": true
    }
  ],
  "data_warmup_policy": {
    "status": "confirmed | blocked",
    "min_start_date": "",
    "reason": ""
  },
  "needs_custom_component": [],
  "next_required_phase": "audit | brainstorm | build | data_inspection | component_authoring"
}
```

The builder phase output is:

- `versions/<version_id>/04_spec_build/strategy_spec.yaml`
- `versions/<version_id>/04_spec_build/component_catalog.json`
- `versions/<version_id>/04_spec_build/spec_build_notes.md` or equivalent
  build notes
- `versions/<version_id>/04_spec_build/spec_mapping_notes.md`
- `versions/<version_id>/04_spec_build/spec_mapping_contract.json`
- `versions/<version_id>/04_spec_build/builder_phase_result.json`

This builder skill stops after writing those artifacts. The downstream
multi-Agent system decides which audit, compile, execution, monitor, or report
worker runs next.

## Red Lines

- Do not skip the Audited Idea Input Gate.
- Do not skip the component catalog gate.
- Do not run catalog export, `oxq spec init`, or SPEC writes before the idea
  audit passes.
- Do not run `oxq spec init` without `--out` under
  `versions/<version_id>/04_spec_build/`, and do not create or consult
  root-level `strategy_spec.yaml`.
- Do not invent component names when a catalog component or recipe exists.
- Do not omit `spec_mapping_notes.md`, `unmapped_source_fields`, or
  `unsupported_mappings` when the source idea contains fields beyond the
  executable SPEC surface.
- Do not omit `spec_mapping_contract.json` or hide strategy semantics as
  non-material Studio/report metadata.
- Do not convert a full backtest interval into required OOS without user
  confirmation in the audited idea.
- Do not treat template defaults as confirmed values.
- Do not write `spec_audit.json`.
- Do not list or read market data files, parquet files, data manifests, or
  provider directories; route data coverage and column questions to
  `explore-data`.
- Do not call audit, compile, backtest, monitor, robustness, experiment, or
  report skills from this builder skill.
- Do not call component creation skills from this builder skill. When a custom
  component is required, write `needs_custom_component` and stop.
- Do not run formal backtests from this skill.
