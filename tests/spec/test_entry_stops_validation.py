"""Spec validation for EntryBarLowStopRule and TimeStopRule."""

import pytest

from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate


def _spec(rules: dict) -> StrategySpec:
    return StrategySpec.from_dict({
        "strategy_id": "entry_stops",
        "universe": {"type": "static", "symbols": ["AAA"]},
        "signal": {"indicators": {}, "rules": {"entry": {"type": "FormulaEvent", "params": {"expr": "close > open"}}}},
        "portfolio": {"type": "EqualWeight", "rules": rules},
    })


def _codes(spec: StrategySpec) -> set[str]:
    return {e["check"] for e in validate(spec).errors}


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "EntryBarLowStopRule", "params": {"buffer": 0.0}},
        {"type": "TimeStopRule", "params": {"max_bars": 3, "min_return": 0.0}},
    ],
)
def test_entry_stops_are_supported_and_count_as_exits(rule: dict) -> None:
    codes = _codes(_spec({"stop": rule}))
    assert not {"portfolio_rule_unsupported", "portfolio_rule_params_unsupported", "event_signal_exit_missing"} & codes


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "EntryBarLowStopRule", "params": {"buffer": 1.5}},
        {"type": "TimeStopRule", "params": {"max_bars": 0}},
        {"type": "TimeStopRule", "params": {"max_bars": 3, "lookback": 5}},
    ],
)
def test_entry_stops_reject_invalid_params(rule: dict) -> None:
    assert {"portfolio_rule_params_unsupported", "portfolio_rule_param_invalid"} & _codes(_spec({"stop": rule}))
