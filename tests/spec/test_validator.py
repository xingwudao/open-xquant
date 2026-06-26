from __future__ import annotations

import pandas as pd
import pytest
import yaml

import oxq.core.registry as registry
from oxq.spec.schema import IndicatorDef, PortfolioRuleDef, RebalanceDef, SignalRuleDef, StrategySpec, _dataclass_to_canonical_dict
from oxq.spec.validator import validate


def _finding(findings: list[dict], check: str) -> dict:
    return next(finding for finding in findings if finding["check"] == check)


def test_validate_accepts_supported_metrics_profile(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: metrics_profile_supported
research:
  hypothesis: metric assumptions should be explicit
market:
  calendar: XNYS
universe:
  type: static
  symbols: [SPY]
data:
  provider: local
  price_adjustment: adjusted
  required_columns: [open, high, low, close, volume]
signal:
  signal_time: close_t
portfolio:
  type: EqualWeight
execution:
  trade_time: next_open
  fill_price_mode: next_open
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  train_period: ["2020-01-01", "2020-12-31"]
  test_period: ["2021-01-01", "2021-12-31"]
  required_oos: true
metrics:
  profile: xquant_production
  risk_free_rate: 0.02
  return_type: log
  annualization_days: 252
  calmar_denominator: max_drawdown
  evaluation_window: oos
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    result = validate(spec)

    assert result.status == "pass"
    assert spec.metrics.profile == "xquant_production"
    assert spec.metrics.risk_free_rate == 0.02
    assert spec.metrics.return_type == "log"
    assert spec.metrics.annualization_days == 252
    assert spec.metrics.calmar_denominator == "max_drawdown"
    assert spec.metrics.evaluation_window == "oos"


def test_validate_rejects_unknown_metrics_profile(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: metrics_profile_unknown
research:
  hypothesis: unknown metric profiles should fail validation
universe:
  type: static
  symbols: [SPY]
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2021-01-01", "2021-12-31"]
metrics:
  profile: unsupported_profile
""",
        encoding="utf-8",
    )

    result = validate(StrategySpec.from_yaml(spec_path))

    assert result.status == "fail"
    assert any(error["check"] == "metrics_profile_unsupported" for error in result.errors)


def test_validate_rejects_reversed_test_period() -> None:
    spec = StrategySpec.template(strategy_id="bad_dates", hypothesis="date ranges must be ordered")
    spec.validation.test_period = ["2024-12-31", "2024-01-01"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "validation_period_order" for error in result.errors)


def test_validate_rejects_empty_or_unsafe_strategy_id() -> None:
    empty = StrategySpec.template(strategy_id="", hypothesis="strategy id is required")
    unsafe = StrategySpec.template(strategy_id="../outside", hypothesis="strategy id must be path safe")

    empty_result = validate(empty)
    unsafe_result = validate(unsafe)

    assert empty_result.status == "fail"
    assert unsafe_result.status == "fail"
    assert any(error["check"] == "strategy_id_invalid" for error in empty_result.errors)
    assert any(error["check"] == "strategy_id_invalid" for error in unsafe_result.errors)


def test_validate_rejects_min_start_date_after_requested_start() -> None:
    spec = StrategySpec.template(strategy_id="bad_min_start", hypothesis="min_start_date must not truncate requested range")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-12-31"]
    spec.data.min_start_date = "2024-02-01"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "data_min_start_date" for error in result.errors)


def test_validate_rejects_negative_costs() -> None:
    spec = StrategySpec.template(strategy_id="negative_costs", hypothesis="costs cannot improve fills")
    spec.cost.fee_rate = -0.001
    spec.cost.slippage_rate = -0.001
    spec.cost.fee_min = -1.0

    result = validate(spec)

    assert result.status == "fail"
    checks = {error["check"] for error in result.errors}
    assert "cost_model_missing" in checks
    assert "fee_min_negative" in checks


def test_from_yaml_rejects_scalar_universe_symbols(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: scalar_symbols
research:
  hypothesis: symbols must be a list
universe:
  type: static
  symbols: SPY
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="universe.symbols"):
        StrategySpec.from_yaml(spec_path)


def test_from_yaml_normalizes_null_params_to_empty_mapping(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: null_params
research:
  hypothesis: null params should not crash validation
signal:
  indicators:
    sma:
      type: SMA
      params: null
  rules:
    threshold:
      type: Threshold
      params: null
portfolio:
  type: EqualWeight
  params: null
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.signal.indicators["sma"].params == {}
    assert spec.signal.rules["threshold"].params == {}
    assert spec.portfolio.params == {}
    assert validate(spec).status == "fail"


def test_from_yaml_rejects_non_mapping_params(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: bad_params
research:
  hypothesis: params must be mappings
signal:
  indicators:
    sma:
      type: SMA
      params: ["period", 5]
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="signal.indicators.sma.params"):
        StrategySpec.from_yaml(spec_path)


def test_from_yaml_coerces_quoted_point_in_time_false(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: quoted_point_in_time
research:
  hypothesis: quoted booleans should not bypass audits
universe:
  type: static
  symbols: ["SPY"]
  point_in_time: "false"
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    result = validate(spec)

    assert spec.universe.point_in_time is False
    warning = _finding(result.warnings, "static_universe_survivorship")
    assert warning["dimensions"] == ["conservative"]


def test_from_yaml_rejects_invalid_point_in_time_bool(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: invalid_point_in_time
research:
  hypothesis: invalid booleans should fail parsing
universe:
  type: static
  symbols: ["SPY"]
  point_in_time: maybe
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="universe.point_in_time"):
        StrategySpec.from_yaml(spec_path)


def test_validate_rejects_unsafe_universe_symbols() -> None:
    spec = StrategySpec.template(strategy_id="unsafe_symbol", hypothesis="symbols are data filenames")
    spec.universe.symbols = ["../outside"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "universe_symbol_unsafe" for error in result.errors)


def test_from_yaml_coerces_quoted_decision_policy_thresholds(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: policy_thresholds
research:
  hypothesis: quoted policy thresholds should parse
decision_policy:
  reject_if:
    oos_sharpe_lt: "0.5"
  promote_if:
    max_drawdown_gte: "-0.2"
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.decision_policy.reject_if["oos_sharpe_lt"] == 0.5
    assert spec.decision_policy.promote_if["max_drawdown_gte"] == -0.2


def test_from_yaml_rejects_invalid_decision_policy_thresholds(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: bad_policy_threshold
research:
  hypothesis: policy thresholds must be numeric
decision_policy:
  promote_if:
    oos_sharpe_gte: high
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="decision_policy"):
        StrategySpec.from_yaml(spec_path)


def test_from_yaml_normalizes_unquoted_validation_dates(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: unquoted_dates
research:
  hypothesis: unquoted YAML dates should be accepted
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  train_period: [2020-01-01, 2021-12-31]
  test_period: [2022-01-01, 2024-12-31]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.validation.train_period == ["2020-01-01", "2021-12-31"]
    assert spec.validation.test_period == ["2022-01-01", "2024-12-31"]


def test_from_yaml_coerces_quoted_numeric_fields(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: quoted_numbers
research:
  hypothesis: quoted numeric fields should parse
universe:
  type: static
  symbols: [SPY]
cost:
  fee_rate: "0.001"
  fee_min: "1.5"
  slippage_rate: "0.002"
execution:
  trade_time: next_open
  fill_price_mode: next_open
  rebalance:
    frequency: daily
    interval_days: "5"
  lot_size: "100"
  initial_cash: "250000"
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    result = validate(spec)

    assert spec.cost.fee_rate == 0.001
    assert spec.cost.fee_min == 1.5
    assert spec.cost.slippage_rate == 0.002
    assert spec.execution.rebalance.interval_days == 5
    assert spec.execution.lot_size == 100
    assert spec.execution.initial_cash == 250000.0
    assert result.status == "pass"


def test_from_yaml_preserves_portfolio_rebalance_rule(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: portfolio_rebalance_rule
research:
  hypothesis: portfolio rebalance rules should remain material
universe:
  type: static
  symbols: [SPY]
portfolio:
  type: EqualWeight
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 10
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.portfolio.rules["rebalance"].type == "RebalanceFrequencyRule"
    assert spec.portfolio.rules["rebalance"].params["interval_days"] == 10
    assert spec.to_dict()["portfolio"]["rules"]["rebalance"]["params"]["interval_days"] == 10


def test_from_yaml_coerces_quoted_portfolio_rebalance_interval(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: quoted_portfolio_rebalance
research:
  hypothesis: quoted portfolio rebalance interval should parse
universe:
  type: static
  symbols: [SPY]
portfolio:
  type: EqualWeight
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: "10"
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.portfolio.rules["rebalance"].params["interval_days"] == 10
    assert validate(spec).status == "pass"


def test_validate_rejects_unsupported_portfolio_rule() -> None:
    spec = StrategySpec.template(strategy_id="bad_portfolio_rule", hypothesis="unsupported rules must fail fast")
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(type="WeeklyCalendarRule", params={"interval_days": 5})

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "portfolio_rebalance_rule_unsupported" for error in result.errors)


def test_validate_rejects_rebalance_rule_conflict() -> None:
    spec = StrategySpec.template(strategy_id="rebalance_conflict", hypothesis="conflicting rebalance settings fail")
    spec.execution.rebalance.interval_days = 5
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(type="RebalanceFrequencyRule", params={"interval_days": 10})

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "rebalance_interval_conflict" for error in result.errors)


def test_validate_rejects_constructor_supplied_rebalance_rule_conflict() -> None:
    spec = StrategySpec.template(strategy_id="rebalance_constructor_conflict", hypothesis="constructor intervals are explicit")
    spec.execution.rebalance = RebalanceDef(interval_days=5)
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(type="RebalanceFrequencyRule", params={"interval_days": 10})

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "rebalance_interval_conflict" for error in result.errors)


def test_validate_rejects_constructor_supplied_daily_rebalance_rule_conflict() -> None:
    spec = StrategySpec.template(strategy_id="rebalance_constructor_daily_conflict", hypothesis="explicit daily is material")
    spec.execution.rebalance = RebalanceDef(interval_days=1)
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(type="RebalanceFrequencyRule", params={"interval_days": 10})

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "rebalance_interval_conflict" for error in result.errors)


def test_validate_rejects_unsupported_rebalance_rule_params() -> None:
    spec = StrategySpec.template(strategy_id="rebalance_extra_params", hypothesis="unsupported params must fail fast")
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(
        type="RebalanceFrequencyRule",
        params={"interval_days": 10, "anchor": "month_end"},
    )

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "portfolio_rebalance_params_unsupported" for error in result.errors)


def test_validate_rejects_explicit_daily_rebalance_rule_conflict(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: explicit_daily_rebalance_conflict
research:
  hypothesis: explicit daily execution must not be overwritten by portfolio rules
universe:
  type: static
  symbols: [SPY]
portfolio:
  type: EqualWeight
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 10
execution:
  rebalance:
    frequency: daily
    interval_days: 1
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    result = validate(StrategySpec.from_yaml(spec_path))

    assert result.status == "fail"
    assert any(error["check"] == "rebalance_interval_conflict" for error in result.errors)


def test_to_dict_preserves_explicit_daily_rebalance_conflict(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    round_trip_path = tmp_path / "round_trip.yaml"
    spec_path.write_text(
        """
strategy_id: explicit_daily_rebalance_round_trip
research:
  hypothesis: explicit daily rebalance remains material after serialization
universe:
  type: static
  symbols: [SPY]
portfolio:
  type: EqualWeight
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 10
execution:
  rebalance:
    frequency: daily
    interval_days: 1
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    serialized = spec.to_dict()
    round_trip_path.write_text(yaml.dump(serialized, sort_keys=False), encoding="utf-8")
    round_tripped = StrategySpec.from_yaml(round_trip_path)

    assert serialized["execution"]["rebalance"]["interval_days"] == 1
    assert validate(spec).status == "fail"
    result = validate(round_tripped)
    assert result.status == "fail"
    assert any(error["check"] == "rebalance_interval_conflict" for error in result.errors)


def test_hash_distinguishes_explicit_daily_rebalance_from_implicit_default(tmp_path) -> None:
    implicit_path = tmp_path / "implicit.yaml"
    explicit_path = tmp_path / "explicit.yaml"
    base_yaml = """
strategy_id: explicit_daily_rebalance_hash
research:
  hypothesis: explicit daily rebalance is material provenance
universe:
  type: static
  symbols: [SPY]
portfolio:
  type: EqualWeight
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 10
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
"""
    implicit_path.write_text(base_yaml, encoding="utf-8")
    explicit_path.write_text(
        base_yaml
        + """
execution:
  rebalance:
    frequency: daily
    interval_days: 1
""",
        encoding="utf-8",
    )

    implicit_spec = StrategySpec.from_yaml(implicit_path)
    explicit_spec = StrategySpec.from_yaml(explicit_path)

    assert validate(implicit_spec).status == "pass"
    assert validate(explicit_spec).status == "fail"
    assert implicit_spec.compute_hash() != explicit_spec.compute_hash()


def test_from_yaml_rejects_non_finite_numeric_fields(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: non_finite
research:
  hypothesis: non-finite values are invalid
cost:
  fee_rate: .nan
  slippage_rate: .inf
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        StrategySpec.from_yaml(spec_path)


def test_validate_rejects_invalid_execution_numeric_ranges() -> None:
    spec = StrategySpec.template(strategy_id="bad_execution_numbers", hypothesis="execution numbers need ranges")
    spec.execution.lot_size = 0
    spec.execution.rebalance.interval_days = 0

    result = validate(spec)

    assert result.status == "fail"
    checks = {error["check"] for error in result.errors}
    assert "lot_size_invalid" in checks
    assert "rebalance_interval_invalid" in checks


def test_validate_rejects_invalid_lot_size_config_default() -> None:
    spec = StrategySpec.template(strategy_id="bad_lot_config_default", hypothesis="lot config default needs range")
    spec.execution.lot_size_config.default = 0

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "lot_size_config_invalid")
    assert error["dimensions"] == ["executable"]


def test_validate_rejects_invalid_lot_size_config_by_symbol() -> None:
    spec = StrategySpec.template(strategy_id="bad_lot_config_symbol", hypothesis="lot config symbols need ranges")
    spec.execution.lot_size_config.by_symbol = {"SPY": 0}

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "lot_size_config_invalid")
    assert error["dimensions"] == ["executable"]


@pytest.mark.parametrize("cash_annual_return", ["0.01", float("nan"), -0.01])
def test_validate_rejects_invalid_cash_annual_return(cash_annual_return) -> None:
    spec = StrategySpec.template(strategy_id="bad_cash_return", hypothesis="cash return must be non-negative finite")
    spec.execution.cash_annual_return = cash_annual_return

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "cash_annual_return_invalid")
    assert error["dimensions"] == ["executable"]


@pytest.mark.parametrize("cash_annual_return", [0.0, 0.025])
def test_validate_accepts_valid_cash_annual_return(cash_annual_return: float) -> None:
    spec = StrategySpec.template(strategy_id="valid_cash_return", hypothesis="cash return accepts non-negative finite")
    spec.execution.cash_annual_return = cash_annual_return

    result = validate(spec)

    assert "cash_annual_return_invalid" not in {error["check"] for error in result.errors}


def test_validate_rejects_rebalance_frequency_when_interval_is_omitted(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: weekly_rebalance
research:
  hypothesis: calendar frequency without explicit interval is ambiguous
universe:
  symbols: ["SPY"]
execution:
  rebalance:
    frequency: weekly
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    result = validate(spec)

    assert spec.execution.rebalance.interval_days == 1
    assert result.status == "fail"
    assert any(error["check"] == "rebalance_frequency_unsupported" for error in result.errors)


def test_validate_accepts_non_daily_rebalance_with_explicit_interval() -> None:
    spec = StrategySpec.template(strategy_id="explicit_rebalance", hypothesis="explicit N-bar rebalance is supported")
    spec.execution.rebalance.frequency = "weekly"
    spec.execution.rebalance.interval_days = 5

    result = validate(spec)

    assert result.status == "pass"


def test_validate_accepts_non_daily_frequency_with_portfolio_rule_interval(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: weekly_portfolio_rule_rebalance
research:
  hypothesis: portfolio rebalance rule supplies the effective interval
universe:
  type: static
  symbols: [SPY]
portfolio:
  type: EqualWeight
  rules:
    rebalance:
      type: RebalanceFrequencyRule
      params:
        interval_days: 5
execution:
  rebalance:
    frequency: weekly
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    result = validate(StrategySpec.from_yaml(spec_path))

    assert result.status == "pass"


def test_validate_accepts_timestamp_signal_dry_run() -> None:
    spec = StrategySpec.template(strategy_id="timestamp_signal", hypothesis="timestamp signals use datetime indexes")
    spec.signal.rules = {"monday": SignalRuleDef(type="Timestamp", params={"rule": "weekday:0"})}

    result = validate(spec)

    assert result.status == "pass"


def test_validate_rejects_timestamp_end_period_rules() -> None:
    spec = StrategySpec.template(strategy_id="timestamp_month_end", hypothesis="timestamp end rules need calendar support")
    spec.signal.rules = {"month_end": SignalRuleDef(type="Timestamp", params={"rule": "month_end"})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "timestamp_end_rule_unsupported" for error in result.errors)


def test_validate_rejects_invalid_timestamp_rules() -> None:
    spec = StrategySpec.template(strategy_id="bad_timestamp_rule", hypothesis="timestamp rules must be explicit")
    spec.signal.rules = {
        "typo": SignalRuleDef(type="Timestamp", params={"rule": "weekdy:0"}),
        "weekend": SignalRuleDef(type="Timestamp", params={"rule": "weekday:6"}),
    }

    result = validate(spec)

    assert result.status == "fail"
    errors = [error for error in result.errors if error["check"] == "timestamp_rule_invalid"]
    assert len(errors) == 2


def test_validate_rejects_unsupported_price_adjustment() -> None:
    spec = StrategySpec.template(strategy_id="raw_prices", hypothesis="price adjustment semantics must be explicit")
    spec.data.price_adjustment = "raw"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "price_adjustment_unsupported" for error in result.errors)


def test_validate_rejects_required_columns_without_close() -> None:
    spec = StrategySpec.template(strategy_id="missing_close", hypothesis="close is required for valuation")
    spec.data.required_columns = ["open", "high", "low", "volume"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "required_columns_missing_close" for error in result.errors)


def test_validate_rejects_next_open_required_columns_without_open() -> None:
    spec = StrategySpec.template(strategy_id="missing_open", hypothesis="open is required for next-open fills")
    spec.data.required_columns = ["high", "low", "close", "volume"]
    spec.execution.fill_price_mode = "next_open"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "required_columns_missing_open" for error in result.errors)


def test_validate_rejects_explicit_next_session_open_without_open() -> None:
    spec = StrategySpec.template(strategy_id="missing_explicit_open", hypothesis="open is required for explicit open fills")
    spec.data.required_columns = ["high", "low", "close", "volume"]
    spec.execution.fill_price_mode = ""
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "required_columns_missing_open" for error in result.errors)


def test_validate_rejects_explicit_next_session_mid_without_open() -> None:
    spec = StrategySpec.template(strategy_id="missing_explicit_mid_open", hypothesis="mid requires open and close")
    spec.data.required_columns = ["high", "low", "close", "volume"]
    spec.execution.fill_price_mode = ""
    spec.execution.order_timing = "next_session_mid"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "mid"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "required_columns_missing_open" for error in result.errors)


def test_validate_rejects_explicit_next_session_avg_without_ohlc() -> None:
    spec = StrategySpec.template(strategy_id="missing_explicit_avg_ohlc", hypothesis="avg requires OHLC")
    spec.data.required_columns = ["close", "volume"]
    spec.execution.fill_price_mode = ""
    spec.execution.order_timing = "next_session_avg"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "avg"

    result = validate(spec)

    assert result.status == "fail"
    checks = {error["check"] for error in result.errors}
    assert "required_columns_missing_open" in checks
    assert "required_columns_missing_high" in checks
    assert "required_columns_missing_low" in checks


def test_validate_rejects_unsupported_data_provider() -> None:
    spec = StrategySpec.template(strategy_id="remote_provider", hypothesis="compiler only supports local data")
    spec.data.provider = "yfinance"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "data_provider_unsupported" for error in result.errors)


@pytest.mark.parametrize("calendar", ["XNYS", "ARCX", "XSHG", "XSHE"])
def test_validate_accepts_supported_market_calendars(calendar: str) -> None:
    spec = StrategySpec.template(strategy_id=f"calendar_{calendar.lower()}", hypothesis="supported calendars compile")
    spec.market.calendar = calendar

    result = validate(spec)

    assert not any(error["check"] == "market_calendar_unsupported" for error in result.errors)


def test_validate_rejects_unknown_market_calendar() -> None:
    spec = StrategySpec.template(strategy_id="bad_calendar", hypothesis="calendar must resolve deterministically")
    spec.market.calendar = "BAD"

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "market_calendar_unsupported")
    assert error["dimensions"] == ["executable"]


def test_from_yaml_rejects_fractional_integer_fields(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
strategy_id: fractional_int
research:
  hypothesis: integer fields cannot truncate
execution:
  rebalance:
    interval_days: 1.5
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer"):
        StrategySpec.from_yaml(spec_path)


def test_validate_rejects_overlapping_train_and_test_periods() -> None:
    spec = StrategySpec.template(strategy_id="overlap_dates", hypothesis="OOS must not overlap train")
    spec.validation.train_period = ["2020-01-01", "2023-01-01"]
    spec.validation.test_period = ["2022-12-31", "2024-01-01"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "validation_period_order" for error in result.errors)


def test_validate_rejects_present_periods_without_exactly_two_dates() -> None:
    spec = StrategySpec.template(strategy_id="bad_period_lengths", hypothesis="periods need start and end")
    spec.validation.train_period = ["2020-01-01"]
    spec.validation.test_period = ["2021-01-01", "2021-12-31"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "validation_period_length" for error in result.errors)


def test_validate_rejects_required_oos_without_two_date_train_period() -> None:
    spec = StrategySpec.template(strategy_id="bad_required_oos", hypothesis="required OOS needs full periods")
    spec.validation.required_oos = True
    spec.validation.train_period = ["2020-01-01"]
    spec.validation.test_period = ["2021-01-01", "2021-12-31"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "oos_incomplete" for error in result.errors)


def test_validate_accepts_full_period_backtest_without_required_oos() -> None:
    spec = StrategySpec.template(strategy_id="full_period", hypothesis="full period backtest is not forced OOS")
    spec.validation.required_oos = False
    spec.validation.train_period = []
    spec.validation.test_period = ["2025-01-01", "2026-06-18"]

    result = validate(spec)

    assert result.status == "pass"


def test_validate_labels_missing_full_period_when_oos_not_required() -> None:
    spec = StrategySpec.template(strategy_id="missing_full_period", hypothesis="full period still needs dates")
    spec.validation.required_oos = False
    spec.validation.train_period = []
    spec.validation.test_period = []

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "backtest_period_missing" for error in result.errors)


def test_validate_rejects_next_high_and_next_low_fill_modes() -> None:
    spec = StrategySpec.template(strategy_id="bad_fill", hypothesis="unsupported next extrema fills fail")
    spec.execution.fill_price_mode = "next_high"

    result = validate(spec)

    assert result.status == "fail"
    checks = [error["check"] for error in result.errors]
    assert checks.count("fill_price_mode_invalid") == 1
    assert "execution_semantics_invalid" not in checks


@pytest.mark.parametrize("fill_price_mode", ["next_close", "next_mid", "next_avg", "next_hl2"])
def test_validate_accepts_executable_next_session_legacy_fill_modes(fill_price_mode: str) -> None:
    spec = StrategySpec.template(strategy_id=f"legacy_{fill_price_mode}", hypothesis="legacy executable fill modes work")
    spec.execution.trade_time = "next_open"
    spec.execution.fill_price_mode = fill_price_mode

    result = validate(spec)

    assert result.status == "pass"
    assert not any(error["check"] == "fill_price_mode_invalid" for error in result.errors)


def test_validate_rejects_trade_time_fill_price_mismatch() -> None:
    spec = StrategySpec.template(strategy_id="mismatch", hypothesis="execution declaration must match fill mode")
    spec.execution.trade_time = "next_open"
    spec.execution.fill_price_mode = "close"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "execution_timing_mismatch" for error in result.errors)


def test_validate_warns_for_next_session_mid_explicit_execution_conservatism() -> None:
    spec = StrategySpec.template(strategy_id="next_mid", hypothesis="next-session mid is causal but optimistic")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_mid"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "mid"

    result = validate(spec)

    assert result.status == "pass"
    warning = _finding(result.warnings, "execution_conservatism")
    assert warning["dimensions"] == ["conservative"]


def test_validate_warns_for_next_session_close_explicit_execution_conservatism() -> None:
    spec = StrategySpec.template(strategy_id="next_close", hypothesis="next-session close is causal but optimistic")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_close"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "close"

    result = validate(spec)

    assert result.status == "pass"
    warning = _finding(result.warnings, "execution_conservatism")
    assert warning["dimensions"] == ["conservative"]


def test_validate_warns_for_next_session_avg_explicit_execution_conservatism() -> None:
    spec = StrategySpec.template(strategy_id="next_avg", hypothesis="next-session avg is causal but optimistic")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_avg"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "avg"

    result = validate(spec)

    assert result.status == "pass"
    warning = _finding(result.warnings, "execution_conservatism")
    assert warning["dimensions"] == ["conservative"]


def test_validate_explicit_next_session_avg_maps_to_next_avg() -> None:
    from oxq.spec.execution import derive_execution_semantics

    spec = StrategySpec.template(strategy_id="next_avg_map", hypothesis="avg maps to executable next avg")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_avg"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "avg"

    result = validate(spec)
    effective = derive_execution_semantics(spec.execution)

    assert result.status == "pass"
    assert effective.fill_price_mode == "next_avg"


def test_validate_explicit_next_session_hl2_maps_to_next_hl2() -> None:
    from oxq.spec.execution import derive_execution_semantics

    spec = StrategySpec.template(strategy_id="next_hl2_map", hypothesis="hl2 maps to executable next hl2")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_hl2"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "hl2"

    result = validate(spec)
    effective = derive_execution_semantics(spec.execution)

    assert result.status == "pass"
    assert effective.fill_price_mode == "next_hl2"


def test_validate_does_not_warn_for_next_session_open_explicit_execution() -> None:
    spec = StrategySpec.template(strategy_id="next_open_explicit", hypothesis="next open is conservative")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "pass"
    assert not any(warning["check"] == "execution_conservatism" for warning in result.warnings)


def test_validate_rejects_same_session_mid_explicit_execution_lag() -> None:
    spec = StrategySpec.template(strategy_id="same_mid", hypothesis="same-session mid is not causal")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "close_t"
    spec.execution.order_timing = "same_session_close"
    spec.execution.price_bar = "same_session"
    spec.execution.price_type = "mid"

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "execution_lag")
    assert error["dimensions"] == ["causal"]


def test_validate_rejects_legacy_mid_as_same_session_execution_lag() -> None:
    spec = StrategySpec.template(strategy_id="legacy_mid", hypothesis="legacy mid maps to same-session")
    spec.execution.fill_price_mode = "mid"
    spec.execution.trade_time = "close_t"

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "execution_lag")
    assert error["dimensions"] == ["causal"]


def test_validate_reports_invalid_derived_execution_semantics() -> None:
    spec = StrategySpec.template(strategy_id="bad_execution_semantics", hypothesis="execution semantics must derive")
    spec.execution.fill_price_mode = ""
    spec.execution.order_timing = "next_session_mid"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "execution_semantics_invalid")
    assert error["dimensions"] == ["executable"]


def test_validate_reports_partial_explicit_execution_as_invalid_semantics() -> None:
    spec = StrategySpec.template(strategy_id="partial_execution", hypothesis="partial explicit fields are invalid")
    spec.execution.fill_price_mode = ""
    spec.execution.order_timing = "next_session_open"

    result = validate(spec)

    assert result.status == "fail"
    checks = {error["check"] for error in result.errors}
    assert "execution_semantics_invalid" in checks
    assert "fill_price_mode_missing" not in checks


def test_validate_warns_for_zero_cost_with_explicit_replay_style_execution() -> None:
    spec = StrategySpec.template(strategy_id="zero_cost_replay", hypothesis="zero costs may be replay-style")
    spec.cost.fee_rate = 0.0
    spec.cost.slippage_rate = 0.0
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "pass"
    warning = _finding(result.warnings, "cost_model_zero")
    assert set(warning["dimensions"]) == {"conservative", "production_consistent"}


def test_validate_rejects_zero_cost_without_explicit_execution_fields() -> None:
    spec = StrategySpec.template(strategy_id="zero_cost_legacy", hypothesis="legacy zero costs are missing costs")
    spec.cost.fee_rate = 0.0
    spec.cost.slippage_rate = 0.0

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "cost_model_missing" for error in result.errors)


def test_validate_rejects_negative_costs_with_explicit_execution_fields() -> None:
    spec = StrategySpec.template(strategy_id="negative_explicit_costs", hypothesis="negative costs are invalid")
    spec.cost.fee_rate = -0.001
    spec.cost.slippage_rate = -0.001
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "cost_model_missing" for error in result.errors)
    assert not any(warning["check"] == "cost_model_zero" for warning in result.warnings)


def test_validate_rejects_zero_fee_even_with_explicit_execution_fields() -> None:
    spec = StrategySpec.template(strategy_id="zero_fee", hypothesis="fee must be positive unless all costs are replay zero")
    spec.cost.fee_rate = 0.0
    spec.cost.slippage_rate = 0.001
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "fee_missing" for error in result.errors)


def test_validate_rejects_zero_slippage_even_with_explicit_execution_fields() -> None:
    spec = StrategySpec.template(
        strategy_id="zero_slippage",
        hypothesis="slippage must be positive unless all costs are replay zero",
    )
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.0
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "slippage_missing" for error in result.errors)


def test_validate_rejects_lot_size_config_by_symbol_until_executable() -> None:
    spec = StrategySpec.template(strategy_id="by_symbol_lot", hypothesis="by-symbol lot sizes must execute")
    spec.execution.lot_size_config.by_symbol = {"SPY": 1}

    result = validate(spec)

    assert result.status == "fail"
    error = _finding(result.errors, "lot_size_config_by_symbol_unsupported")
    assert error["dimensions"] == ["executable"]


def test_validate_non_string_market_calendar_returns_finding() -> None:
    spec = StrategySpec.template(strategy_id="bad_calendar_type", hypothesis="calendar must validate")
    spec.market.calendar = ["XNYS"]  # type: ignore[assignment]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "market_calendar_unsupported" for error in result.errors)


def test_validate_rejects_unsupported_signal_time() -> None:
    spec = StrategySpec.template(strategy_id="unsupported_signal_time", hypothesis="runtime only supports close signals")
    spec.signal.signal_time = "open_t"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_time_unsupported" for error in result.errors)


def test_validate_rejects_multiple_crossover_rules() -> None:
    spec = StrategySpec.template(strategy_id="multi_cross", hypothesis="multiple crossovers are ambiguous")
    spec.signal.indicators = {
        "fast_a": IndicatorDef(type="SMA", params={"period": 2}),
        "slow_a": IndicatorDef(type="SMA", params={"period": 3}),
        "fast_b": IndicatorDef(type="SMA", params={"period": 4}),
        "slow_b": IndicatorDef(type="SMA", params={"period": 5}),
    }
    spec.signal.rules = {
        "cross_a": SignalRuleDef(type="Crossover", params={"fast": "fast_a", "slow": "slow_a"}),
        "cross_b": SignalRuleDef(type="Crossover", params={"fast": "fast_b", "slow": "slow_b"}),
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "multiple_crossover_rules" for error in result.errors)


def test_validate_rejects_peak_signal_future_data() -> None:
    spec = StrategySpec.template(strategy_id="peak_signal", hypothesis="future data signals must not run")
    spec.signal.rules = {"peak": SignalRuleDef(type="Peak", params={"column": "close"})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "peak_future_data" for error in result.errors)


def test_validate_rejects_unknown_indicator_type() -> None:
    spec = StrategySpec.template(strategy_id="unknown_indicator", hypothesis="registry names must resolve")
    spec.signal.indicators = {"bad": IndicatorDef(type="NoSuchIndicator")}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "compile_dry_run_failed" for error in result.errors)


def test_validate_rejects_bad_indicator_compute_params() -> None:
    spec = StrategySpec.template(strategy_id="bad_indicator_params", hypothesis="indicator params must bind")
    spec.signal.indicators = {"sma": IndicatorDef(type="SMA", params={"window": 20})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "compute_dry_run_failed" for error in result.errors)


def test_validate_rejects_unknown_signal_type() -> None:
    spec = StrategySpec.template(strategy_id="unknown_signal", hypothesis="registry names must resolve")
    spec.signal.rules = {"bad": SignalRuleDef(type="NoSuchSignal")}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "compile_dry_run_failed" for error in result.errors)


def test_validate_rejects_bad_signal_compute_params() -> None:
    spec = StrategySpec.template(strategy_id="bad_signal_params", hypothesis="signal params must execute")
    spec.signal.rules = {
        "threshold": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": "0.5"}),
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "compute_dry_run_failed" for error in result.errors)


def test_validate_rejects_unknown_portfolio_type() -> None:
    spec = StrategySpec.template(strategy_id="unknown_portfolio", hypothesis="registry names must resolve")
    spec.portfolio.type = "NoSuchOptimizer"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "compile_dry_run_failed" for error in result.errors)


def test_validate_rejects_pct_equity_with_signal_rules() -> None:
    spec = StrategySpec.template(strategy_id="pct_signal", hypothesis="PctEquity cannot consume unfiltered signals")
    spec.portfolio.type = "PctEquity"
    spec.signal.rules = {"threshold": SignalRuleDef(type="Threshold", params={"column": "score", "threshold": 0.0})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_portfolio_unsupported" for error in result.errors)


def test_validate_rejects_missing_optimizer_constructor_params() -> None:
    spec = StrategySpec.template(strategy_id="missing_optimizer_param", hypothesis="optimizer params must instantiate")
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "compile_dry_run_failed" for error in result.errors)


def test_validate_rejects_optimizer_references_to_unknown_columns() -> None:
    spec = StrategySpec.template(strategy_id="missing_score_col", hypothesis="optimizer inputs must be declared")
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {"score_col": "score", "n": 2}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "optimizer_column_missing" for error in result.errors)


def test_validate_rejects_invalid_top_n_ranking_params() -> None:
    spec = StrategySpec.template(strategy_id="bad_top_n_params", hypothesis="ranking optimizer params must be valid")
    spec.signal.indicators = {"score": IndicatorDef(type="SMA", params={"period": 3})}
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {"score_col": "score", "n": 0, "max_weight": 1.5}

    result = validate(spec)

    assert result.status == "fail"
    errors = [error for error in result.errors if error["check"] == "optimizer_param_invalid"]
    assert len(errors) == 2


def test_validate_rejects_risk_parity_missing_volatility_col() -> None:
    spec = StrategySpec.template(strategy_id="bad_risk_parity", hypothesis="risk parity volatility input must exist")
    spec.portfolio.type = "RiskParity"
    spec.portfolio.params = {"volatility_col": "volatility"}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "optimizer_column_missing" for error in result.errors)


def test_validate_rejects_kelly_missing_input_cols_and_bad_fraction() -> None:
    spec = StrategySpec.template(strategy_id="bad_kelly", hypothesis="kelly inputs must be declared")
    spec.signal.indicators = {"win_rate": IndicatorDef(type="SMA", params={"period": 3})}
    spec.portfolio.type = "Kelly"
    spec.portfolio.params = {
        "win_rate_col": "win_rate",
        "avg_win_col": "avg_win",
        "avg_loss_col": "avg_loss",
        "fraction": 0,
    }

    result = validate(spec)

    assert result.status == "fail"
    checks = [error["check"] for error in result.errors]
    assert "optimizer_column_missing" in checks
    assert "optimizer_param_invalid" in checks


def test_validate_rejects_signal_references_to_unknown_columns() -> None:
    spec = StrategySpec.template(strategy_id="missing_signal_columns", hypothesis="signal inputs must be declared")
    spec.signal.rules = {"entry": SignalRuleDef(type="Crossover", params={"fast": "fast_missing", "slow": "slow_missing"})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_column_missing" for error in result.errors)


def test_validate_rejects_indicator_and_signal_names_that_overwrite_data_columns() -> None:
    spec = StrategySpec.template(strategy_id="name_collision", hypothesis="derived columns must not overwrite prices")
    spec.signal.indicators = {"close": IndicatorDef(type="SMA", params={"period": 3})}
    spec.signal.rules = {"open": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0})}

    result = validate(spec)

    assert result.status == "fail"
    errors = [error for error in result.errors if error["check"] == "signal_name_collision"]
    assert len(errors) == 2


def test_validate_rejects_duplicate_indicator_and_signal_names() -> None:
    spec = StrategySpec.template(strategy_id="derived_name_collision", hypothesis="derived columns must be unique")
    spec.signal.indicators = {"entry": IndicatorDef(type="SMA", params={"period": 3})}
    spec.signal.rules = {"entry": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_name_collision" for error in result.errors)


def test_validate_rejects_missing_required_signal_params() -> None:
    spec = StrategySpec.template(strategy_id="missing_signal_params", hypothesis="signal params are required")
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "close"}),
        "threshold": SignalRuleDef(type="Threshold", params={}),
        "compare": SignalRuleDef(type="Comparison", params={"left": "close"}),
    }

    result = validate(spec)

    assert result.status == "fail"
    errors = [error for error in result.errors if error["check"] == "signal_param_missing"]
    assert len(errors) == 3


def test_validate_rejects_formula_without_expr() -> None:
    spec = StrategySpec.template(strategy_id="missing_formula_expr", hypothesis="formula expr is required")
    spec.signal.rules = {"formula": SignalRuleDef(type="Formula", params={})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_param_missing" for error in result.errors)


def test_validate_rejects_formula_references_to_unknown_columns() -> None:
    spec = StrategySpec.template(strategy_id="bad_formula_expr", hypothesis="formula names must resolve")
    spec.signal.rules = {"formula": SignalRuleDef(type="Formula", params={"expr": "missing > 0"})}

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_column_missing" for error in result.errors)


def test_validate_rejects_invalid_signal_relationships() -> None:
    spec = StrategySpec.template(strategy_id="bad_relationship", hypothesis="relationships must be known operators")
    spec.signal.rules = {
        "threshold": SignalRuleDef(type="Threshold", params={"column": "close", "relationship": "roughly"}),
        "compare": SignalRuleDef(type="Comparison", params={"left": "close", "right": "open", "relationship": "near"}),
    }

    result = validate(spec)

    assert result.status == "fail"
    errors = [error for error in result.errors if error["check"] == "signal_relationship_invalid"]
    assert len(errors) == 2


def test_validate_rejects_composite_references_to_later_signals() -> None:
    spec = StrategySpec.template(strategy_id="composite_order", hypothesis="composites need computed input signals")
    spec.signal.rules = {
        "combo": SignalRuleDef(type="Composite", params={"signals": ["entry"]}),
        "entry": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_column_missing" for error in result.errors)


def test_validate_rejects_empty_or_invalid_composite_signals() -> None:
    spec = StrategySpec.template(strategy_id="bad_composite_signals", hypothesis="composites need explicit input signals")
    spec.signal.rules = {
        "entry": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "empty": SignalRuleDef(type="Composite", params={"signals": []}),
        "bad": SignalRuleDef(type="Composite", params={"signals": ["entry", 1]}),
    }

    result = validate(spec)

    assert result.status == "fail"
    errors = [error for error in result.errors if error["check"] == "signal_param_missing"]
    assert len(errors) == 2


def test_validate_rejects_invalid_composite_logic() -> None:
    spec = StrategySpec.template(strategy_id="bad_composite_logic", hypothesis="composite logic must be explicit")
    spec.signal.rules = {
        "entry": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "combo": SignalRuleDef(type="Composite", params={"signals": ["entry"], "logic": "adn"}),
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_logic_invalid" for error in result.errors)


def test_validate_rejects_multiple_terminal_signal_rules() -> None:
    spec = StrategySpec.template(strategy_id="ambiguous_terminal", hypothesis="multiple terminal signals are ambiguous")
    spec.signal.rules = {
        "above": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "trend": SignalRuleDef(type="Threshold", params={"column": "volume", "threshold": 1.0}),
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_terminal_ambiguous" for error in result.errors)


def test_validate_rejects_or_composite_mixing_event_and_level_signals() -> None:
    spec = StrategySpec.template(strategy_id="or_mixed_lifecycle", hypothesis="or composites must not mix lifecycles")
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
        "filter": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "entry": SignalRuleDef(type="Composite", params={"signals": ["cross", "filter"], "logic": "or"}),
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_lifecycle_ambiguous" for error in result.errors)


def test_signal_to_position_accepts_signal_rules() -> None:
    spec = StrategySpec.template(
        strategy_id="roc_timing_position",
        hypothesis="ROC timing maps categorical signals to positions",
    )
    spec.universe.symbols = ["CSI300"]
    spec.signal.indicators = {
        "roc_120": IndicatorDef(type="ROC", params={"column": "close", "period": 120})
    }
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={"column": "roc_120", "mode": "fixed", "bottom": -5.0, "top": 5.0},
        )
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "timing", "buy_weight": 1.0, "sell_weight": 0.0}
    spec.validation.train_period = ["2020-01-01", "2023-12-31"]
    spec.validation.test_period = ["2024-01-01", "2025-01-01"]
    spec.benchmark.symbols = ["CSI300"]
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001

    result = validate(spec)

    assert result.status == "pass"


def test_signal_to_position_requires_declared_signal_rule() -> None:
    spec = StrategySpec.template(
        strategy_id="roc_timing_bad_signal",
        hypothesis="SignalToPosition must consume a declared signal rule",
    )
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={"column": "close", "mode": "fixed", "bottom": -5.0, "top": 5.0},
        )
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "missing"}

    result = validate(spec)

    assert result.status == "fail"
    assert any(
        error["check"] == "optimizer_param_invalid"
        and "portfolio.params.signal must reference a signal rule" in error["message"]
        for error in result.errors
    )


def test_signal_to_position_rejects_non_categorical_signal_rule() -> None:
    spec = StrategySpec.template(
        strategy_id="signal_to_position_boolean",
        hypothesis="SignalToPosition must consume categorical signals",
    )
    spec.signal.rules = {
        "entry": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0})
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "entry"}

    result = validate(spec)

    assert result.status == "fail"
    assert any(
        error["check"] == "optimizer_param_invalid"
        and "BUY/SELL/HOLD" in error["message"]
        for error in result.errors
    )


def test_equal_weight_rejects_categorical_signal_rule() -> None:
    spec = StrategySpec.template(
        strategy_id="equal_weight_categorical",
        hypothesis="Categorical trading-intent labels require an explicit position mapper",
    )
    spec.signal.indicators = {
        "roc_120": IndicatorDef(type="ROC", params={"column": "close", "period": 120})
    }
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={"column": "roc_120", "mode": "fixed", "bottom": -5.0, "top": 5.0},
        )
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_categorical_portfolio_invalid" for error in result.errors)


def test_equal_weight_rejects_partial_trading_intent_output_domain(monkeypatch) -> None:
    class BuySellOnlySignal:
        name = "BuySellOnlyForValidationTest"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "SELL"], index=mktdata.index)

    monkeypatch.setitem(registry._SIGNAL_REGISTRY, BuySellOnlySignal.name, BuySellOnlySignal)
    spec = StrategySpec.template(
        strategy_id="equal_weight_partial_intent_domain",
        hypothesis="Any trading-intent label domain requires an explicit position mapper",
    )
    spec.signal.rules = {
        "timing": SignalRuleDef(type=BuySellOnlySignal.name, output_domain=["BUY", "SELL"])
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "signal_categorical_portfolio_invalid" for error in result.errors)


def test_signal_to_position_rejects_unsupported_output_domain_labels(monkeypatch) -> None:
    class BuyFlatSignal:
        name = "BuyFlatForValidationTest"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "FLAT"], index=mktdata.index)

    monkeypatch.setitem(registry._SIGNAL_REGISTRY, BuyFlatSignal.name, BuyFlatSignal)
    spec = StrategySpec.template(
        strategy_id="signal_to_position_unsupported_domain",
        hypothesis="SignalToPosition should reject non BUY/SELL/HOLD labels",
    )
    spec.signal.rules = {
        "timing": SignalRuleDef(type=BuyFlatSignal.name, output_domain=["BUY", "FLAT"])
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "timing"}

    result = validate(spec)

    assert result.status == "fail"
    assert any(
        error["check"] == "optimizer_param_invalid"
        and "unsupported labels" in error["message"]
        and "FLAT" in error["message"]
        for error in result.errors
    )


def test_signal_rule_output_domain_is_metadata_not_compute_params(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: categorical_metadata
research:
  hypothesis: custom categorical signal metadata should not be passed to compute
universe:
  type: static
  symbols: [AAA]
signal:
  signal_time: close_t
  rules:
    timing:
      type: Threshold
      output_domain: [BUY, SELL, HOLD]
      params:
        column: close
        threshold: 1.0
portfolio:
  type: SignalToPosition
  params:
    signal: timing
execution:
  trade_time: next_open
  fill_price_mode: next_open
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  train_period: [2020-01-01, 2022-12-31]
  test_period: [2023-01-01, 2023-12-31]
benchmark:
  symbols: [AAA]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    result = validate(spec)

    assert spec.signal.rules["timing"].output_domain == ["BUY", "SELL", "HOLD"]
    assert "output_domain" not in spec.signal.rules["timing"].params
    assert result.status == "fail"
    assert any(
        error["check"] == "signal_output_domain_mismatch"
        and "does not match declared output_domain" in error["message"]
        for error in result.errors
    )


def test_custom_categorical_signal_can_declare_output_domain(monkeypatch) -> None:
    class CustomTimingSignal:
        name = "CustomTimingForValidationTest"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "HOLD", "SELL"], index=mktdata.index)

    monkeypatch.setitem(registry._SIGNAL_REGISTRY, CustomTimingSignal.name, CustomTimingSignal)
    spec = StrategySpec.template(
        strategy_id="custom_categorical_signal",
        hypothesis="custom categorical signals can drive position latching",
    )
    spec.universe.symbols = ["AAA"]
    spec.signal.rules = {
        "timing": SignalRuleDef(type=CustomTimingSignal.name, output_domain=["BUY", "SELL", "HOLD"])
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "timing"}
    spec.validation.train_period = ["2020-01-01", "2022-12-31"]
    spec.validation.test_period = ["2023-01-01", "2023-12-31"]
    spec.benchmark.symbols = ["AAA"]
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001

    result = validate(spec)

    assert result.status == "pass"


def test_empty_signal_output_domain_is_not_part_of_canonical_hash_input() -> None:
    spec = StrategySpec.template(
        strategy_id="empty_output_domain_hash",
        hypothesis="default signal metadata should not change historical hashes",
    )
    spec.signal.rules = {
        "entry": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0})
    }

    canonical = _dataclass_to_canonical_dict(spec)

    assert "output_domain" not in canonical["signal"]["rules"]["entry"]


def test_empty_portfolio_rules_are_not_part_of_canonical_hash_input() -> None:
    spec = StrategySpec.template(
        strategy_id="empty_portfolio_rules_hash",
        hypothesis="default empty portfolio rules should not change historical hashes",
    )

    canonical = _dataclass_to_canonical_dict(spec)

    assert "rules" not in canonical["portfolio"]


def test_roc_timing_fixed_thresholds_must_not_overlap() -> None:
    spec = StrategySpec.template(
        strategy_id="roc_timing_bad_thresholds",
        hypothesis="ambiguous fixed thresholds fail validation",
    )
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={"column": "close", "mode": "fixed", "bottom": 5.0, "top": -5.0},
        )
    }

    result = validate(spec)

    assert result.status == "fail"
    assert any(
        error["check"] == "signal_threshold_invalid"
        and "bottom must be less than top" in error["message"]
        for error in result.errors
    )
