"""Tests for ExitRule."""

import pandas as pd

from oxq.core.types import Portfolio, Position, Rule
from oxq.rules.exit import ExitRule


def test_exit_rule_satisfies_rule_protocol() -> None:
    assert isinstance(ExitRule(fast="sma_10", slow="sma_50"), Rule)


def test_exit_rule_sells_when_fast_below_slow() -> None:
    rule = ExitRule(fast="sma_10", slow="sma_50")
    row = pd.Series({"close": 140.0, "sma_10": 95.0, "sma_50": 100.0})
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=150.0)},
    )

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.side == "SELL"
    assert order.shares == 100


def test_exit_rule_no_sell_when_fast_above_slow() -> None:
    rule = ExitRule(fast="sma_10", slow="sma_50")
    row = pd.Series({"close": 160.0, "sma_10": 105.0, "sma_50": 100.0})
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=150.0)},
    )

    assert rule.evaluate("AAPL", row, portfolio) is None


def test_exit_rule_no_sell_without_position() -> None:
    rule = ExitRule(fast="sma_10", slow="sma_50")
    row = pd.Series({"close": 140.0, "sma_10": 95.0, "sma_50": 100.0})
    portfolio = Portfolio(cash=100_000.0)

    assert rule.evaluate("AAPL", row, portfolio) is None


def test_exit_rule_sells_full_position() -> None:
    rule = ExitRule(fast="sma_10", slow="sma_50")
    row = pd.Series({"close": 140.0, "sma_10": 95.0, "sma_50": 100.0})
    portfolio = Portfolio(
        cash=0.0,
        positions={"AAPL": Position(symbol="AAPL", shares=250, avg_cost=150.0)},
    )

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 250  # sells entire position
