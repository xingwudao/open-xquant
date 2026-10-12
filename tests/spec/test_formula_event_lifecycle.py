"""FormulaEvent has an event (entry-trigger) lifecycle in compiler and validator."""

import pandas as pd

from oxq.spec import compiler, validator
from oxq.spec.schema import StrategySpec


def _spec(rules: dict) -> StrategySpec:
    return StrategySpec.from_dict({
        "strategy_id": "evt",
        "universe": {"type": "static", "symbols": ["AAA"]},
        "signal": {"indicators": {}, "rules": rules},
    })


def test_formula_event_is_event_lifecycle_in_compiler_and_validator() -> None:
    spec = _spec({"entry": {"type": "FormulaEvent", "params": {"expr": "close < open"}}})
    assert compiler._effective_signal_type(spec, "entry") in compiler._EVENT_SIGNAL_TYPES
    assert validator._effective_signal_type(spec, "entry") == "event"


def test_formula_stays_level_lifecycle() -> None:
    spec = _spec({"entry": {"type": "Formula", "params": {"expr": "close < open"}}})
    assert compiler._effective_signal_type(spec, "entry") not in compiler._EVENT_SIGNAL_TYPES
    assert validator._effective_signal_type(spec, "entry") == "level"


def test_event_signal_types_in_sync() -> None:
    assert validator._EVENT_SIGNAL_TYPES == compiler._EVENT_SIGNAL_TYPES


def _frame(flag: bool) -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=1)
    return {"AAA": pd.DataFrame({"entry": [flag]}, index=idx)}


def test_held_formula_event_position_survives_false_bar() -> None:
    opt = compiler._SignalFilteredEqualWeightOptimizer(
        signal_names=["entry"], signal_types={"entry": "FormulaEvent"},
    )
    assert opt.optimize(_frame(True), {}) == {"AAA": 1.0}
    opt.set_held_symbols(["AAA"])
    assert opt.optimize(_frame(False), {}) == {"AAA": 1.0}
    opt.reset_symbols(["AAA"])
    opt.set_held_symbols([])
    assert opt.optimize(_frame(False), {}) == {"CASH": 1.0}


def test_formula_event_expr_column_check_matches_formula() -> None:
    spec = _spec({"entry": {"type": "FormulaEvent", "params": {"expr": "missing_col < 0"}}})
    codes = {e["check"] for e in validator.validate(spec).errors}
    formula_spec = _spec({"entry": {"type": "Formula", "params": {"expr": "missing_col < 0"}}})
    formula_codes = {e["check"] for e in validator.validate(formula_spec).errors}
    # FormulaEvent additionally requires an exit rule; the expression checks must match.
    assert codes - {"event_signal_exit_missing"} == formula_codes
    assert formula_codes  # unknown column must be rejected
