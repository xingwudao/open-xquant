"""Spec validation for FormulaExitRule and event signals without exits."""

from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate


def _spec(signal_type: str, portfolio_rules: dict) -> StrategySpec:
    return StrategySpec.from_dict({
        "strategy_id": "exit_check",
        "universe": {"type": "static", "symbols": ["AAA"]},
        "signal": {
            "indicators": {"fast": {"type": "SMA", "params": {"period": 2}}},
            "rules": {"entry": {"type": signal_type, "params": {"expr": "close > fast"}}},
        },
        "portfolio": {"type": "EqualWeight", "rules": portfolio_rules},
    })


def _codes(spec: StrategySpec) -> set[str]:
    return {e["check"] for e in validate(spec).errors}


def test_formula_exit_rule_is_supported_runtime_rule() -> None:
    spec = _spec("FormulaEvent", {"exit": {"type": "FormulaExitRule", "params": {"expr": "close < fast"}}})
    assert not {"portfolio_rule_unsupported", "portfolio_rule_params_unsupported", "portfolio_rule_expr_invalid"} & _codes(spec)


def test_formula_exit_rule_rejects_unknown_param() -> None:
    spec = _spec("FormulaEvent", {"exit": {"type": "FormulaExitRule", "params": {"expr": "close < fast", "x": 1}}})
    assert "portfolio_rule_params_unsupported" in _codes(spec)


def test_formula_exit_rule_rejects_unknown_column() -> None:
    spec = _spec("FormulaEvent", {"exit": {"type": "FormulaExitRule", "params": {"expr": "close < nope"}}})
    assert "portfolio_rule_expr_invalid" in _codes(spec)


def test_event_signal_without_any_exit_rule_is_fatal() -> None:
    assert "event_signal_exit_missing" in _codes(_spec("FormulaEvent", {}))


def test_level_signal_without_exit_rule_is_fine() -> None:
    assert "event_signal_exit_missing" not in _codes(_spec("Formula", {}))
