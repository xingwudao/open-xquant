"""Tests for EntryRule."""

import pandas as pd

from oxq.core.types import Portfolio, Position, Rule
from oxq.rules.entry import EntryRule


def test_entry_rule_satisfies_rule_protocol() -> None:
    assert isinstance(EntryRule(signal="sig"), Rule)


def test_entry_rule_buys_on_signal() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=100)
    row = pd.Series({"close": 150.0, "sma_10_x_sma_50": True})
    portfolio = Portfolio(cash=100_000.0)

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.side == "BUY"
    assert order.shares == 100


def test_entry_rule_no_signal_no_order() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=100)
    row = pd.Series({"close": 150.0, "sma_10_x_sma_50": False})
    portfolio = Portfolio(cash=100_000.0)

    assert rule.evaluate("AAPL", row, portfolio) is None


def test_entry_rule_no_buy_if_already_holding() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=100)
    row = pd.Series({"close": 150.0, "sma_10_x_sma_50": True})
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=140.0)},
    )

    assert rule.evaluate("AAPL", row, portfolio) is None


def test_entry_rule_buys_different_symbol() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=50)
    row = pd.Series({"close": 300.0, "sma_10_x_sma_50": True})
    # Already holding AAPL, but evaluating MSFT
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=140.0)},
    )

    order = rule.evaluate("MSFT", row, portfolio)
    assert order is not None
    assert order.symbol == "MSFT"
    assert order.shares == 50
