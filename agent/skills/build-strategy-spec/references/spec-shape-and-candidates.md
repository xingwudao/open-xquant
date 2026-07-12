# SPEC Shape And Candidate Values

Use this reference when writing or repairing `strategy_spec.yaml`.

## Contents

- Current SPEC Template Shape
- Stable Candidate Spec Values
- Full-Interval Validation
- `latest_available`
- Data Warmup

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
`<phase_paths.04_spec_build>/`. The default command writes root-level
`strategy_spec.yaml`, which is a workspace layout violation even if the file is
deleted later. Do not create, read, or use root-level `strategy_spec.yaml` as a
template reference. If such a file is accidentally created, delete it, record
`layout_violation` in `builder_phase_result.json`, and keep the builder result
blocked.

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
  confirms calendar rebalance. Do not combine a calendar schedule with
  `interval_days > 1` or an interval-based portfolio rebalance rule. Do not
  write month-end or week-end schedules.
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
  requirements that the SPEC schema can represent. Use schema-shaped values:
  `cost_multiplier: [2.0]`, `parameter_perturbation: {indicator.param:
  [values...]}`, and `regime_analysis: true | false`. Do not write boolean
  `true` for `cost_multiplier` or `parameter_perturbation`; that passes weak
  YAML parsing but breaks the robustness runner.
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

## Full-Interval Validation

If the audited idea supplies one complete backtest period and does not confirm
an IS/OOS split, encode the full period as `validation.test_period` with
`validation.required_oos: false`. Do not split the full period into train/test
or set `required_oos: true` unless the audited idea explicitly confirms that
validation plan.

## `latest_available`

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

## Data Warmup

For lookback indicators, define data warmup deliberately:

- If the audited idea requires a true full-interval evaluation from the first
  test date, set `data.min_start_date` earlier than the evaluation start so the
  largest lookback has enough prior bars.
- If the audited idea confirms first-window warmup NaNs or cash behavior,
  leave `data.min_start_date` empty only after recording that policy in
  `spec_build_notes.md`.
- Do not silently let two otherwise identical specs differ only because one
  Agent fetched warmup history and another did not.
