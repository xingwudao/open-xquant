from __future__ import annotations

from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec
from oxq.spec.validator import validate


def test_validate_rejects_reversed_test_period() -> None:
    spec = StrategySpec.template(strategy_id="bad_dates", hypothesis="date ranges must be ordered")
    spec.validation.test_period = ["2024-12-31", "2024-01-01"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "validation_period_order" for error in result.errors)


def test_validate_rejects_overlapping_train_and_test_periods() -> None:
    spec = StrategySpec.template(strategy_id="overlap_dates", hypothesis="OOS must not overlap train")
    spec.validation.train_period = ["2020-01-01", "2023-01-01"]
    spec.validation.test_period = ["2022-12-31", "2024-01-01"]

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "validation_period_order" for error in result.errors)


def test_validate_rejects_next_high_and_next_low_fill_modes() -> None:
    spec = StrategySpec.template(strategy_id="bad_fill", hypothesis="unsupported next extrema fills fail")
    spec.execution.fill_price_mode = "next_high"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "fill_price_mode_invalid" for error in result.errors)


def test_validate_rejects_trade_time_fill_price_mismatch() -> None:
    spec = StrategySpec.template(strategy_id="mismatch", hypothesis="execution declaration must match fill mode")
    spec.execution.trade_time = "next_open"
    spec.execution.fill_price_mode = "close"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "execution_timing_mismatch" for error in result.errors)


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
