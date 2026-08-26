# Field Classification

Use this reference when auditing effective StrategySpec fields, default
assumptions, source boundaries, and data warmup.

## Contents

- Effective Field Audit
- Source Boundary
- Field Classification
- Material Fields
- Data Warmup
- Material Categories

## User-confirmed source value vs effective value check

Use `StrategySpec.from_yaml(...).to_effective_dict()` as the field-path source
for effective SPEC auditing. Do not treat a YAML-only key that is absent from
the effective dictionary as a valid confirmed field. For example,
`portfolio.initial_cash` is not an effective StrategySpec field;
audit `execution.initial_cash`, not `portfolio.initial_cash`.
`field_audits` must contain only effective StrategySpec field paths.
Do not write YAML-only paths such as `portfolio.initial_cash` as `field_audits` rows.
Put those paths in `evidence`, `source_yaml_path`, and `builder_required_fix`
fields on `contradictions` rows.

Before starting audit completion, before any Default Confirmation Checklist,
compare every user-confirmed source value from the audited idea to the
effective StrategySpec value that will actually run. If they differ, the SPEC
is mistranslated and cannot be repaired by user confirmation in the auditor
phase.

Concrete blocker: if the audited idea confirms starting cash of `1000000`, the
YAML contains `portfolio.initial_cash: 1000000`, and the effective StrategySpec
still shows `execution.initial_cash: 100000.0`, the value was placed under an
ignored or non-operative field. In that case, write a blocked audit with
`next_required_phase: build`, do not write `audit_conclusion: all_pass`, and do
not create or show a Full SPEC Confirmation Table for that value. The same rule
applies to lot size, rebalance, execution timing, data filters, costs,
validation periods, and any other user-confirmed material field whose effective
value differs from the confirmed source.

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
runtime, template, or open-xquant default. Evidence such as "Framework default",
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
fields omitted from YAML but injected by open-xquant defaults, such as
execution, cost, cash, validation, metrics, empty dictionaries, and empty
lists.

## Material Fields

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

## Data Warmup

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

## Material Categories

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
- `system_provenance`: open-xquant version and generated provenance fields
