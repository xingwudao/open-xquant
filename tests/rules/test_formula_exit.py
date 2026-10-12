"""Tests for FormulaExitRule."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Portfolio, Position, Rule
from oxq.rules.formula_exit import FormulaExitRule


def _portfolio(held: bool) -> Portfolio:
    positions = {"AAA": Position(symbol="AAA", shares=100, avg_cost=Decimal("10"))} if held else {}
    return Portfolio(cash=Decimal("0"), positions=positions)


def _row(white: float, yellow: float) -> pd.Series:
    return pd.Series({"close": 10.0, "white": white, "yellow": yellow}, name=pd.Timestamp("2024-01-02"))


def test_formula_exit_satisfies_rule_protocol() -> None:
    assert isinstance(FormulaExitRule(expr="white < yellow"), Rule)


def test_exits_held_position_when_expression_true() -> None:
    result = FormulaExitRule(expr="white < yellow").evaluate("AAA", _row(9.0, 10.0), _portfolio(True))
    assert result.target_positions == {"AAA": 0.0}
    assert "white < yellow" in result.reason


def test_no_exit_when_expression_false() -> None:
    result = FormulaExitRule(expr="white < yellow").evaluate("AAA", _row(11.0, 10.0), _portfolio(True))
    assert result.target_positions is None


def test_not_held_is_noop_even_if_expression_true() -> None:
    result = FormulaExitRule(expr="white < yellow").evaluate("AAA", _row(9.0, 10.0), _portfolio(False))
    assert result.target_positions is None


def test_nan_warmup_does_not_exit() -> None:
    result = FormulaExitRule(expr="white < yellow").evaluate("AAA", _row(np.nan, 10.0), _portfolio(True))
    assert result.target_positions is None


@pytest.mark.parametrize("expr", ["", "   ", None])
def test_rejects_empty_expression(expr: object) -> None:
    with pytest.raises(ValueError, match="FormulaExitRule requires a non-empty expr"):
        FormulaExitRule(expr=expr)
