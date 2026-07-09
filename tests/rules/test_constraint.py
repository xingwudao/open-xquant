"""Tests for pre-trade constraint rules."""

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, Position, RuleResult
from oxq.rules.constraint import BlacklistRule, CalendarRebalanceRule, MaxHoldingsRule, RebalanceFrequencyRule

# ---------------------------------------------------------------------------
# BlacklistRule
# ---------------------------------------------------------------------------


def test_blacklist_blocks_listed_symbol() -> None:
    rule = BlacklistRule(symbols={"AAPL", "TSLA"})
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("AAPL", row, portfolio)
    assert isinstance(result, RuleResult)
    assert result.weights == {"AAPL": 0.0}
    assert result.reason != ""


def test_blacklist_allows_unlisted_symbol() -> None:
    rule = BlacklistRule(symbols={"AAPL", "TSLA"})
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("GOOG", row, portfolio)
    assert result.weights is None


def test_blacklist_empty() -> None:
    rule = BlacklistRule(symbols=set())
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("AAPL", row, portfolio)
    assert result.weights is None


# ---------------------------------------------------------------------------
# MaxHoldingsRule
# ---------------------------------------------------------------------------


def test_max_holdings_blocks_when_at_limit() -> None:
    rule = MaxHoldingsRule(max_holdings=2)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100")),
            "GOOG": Position(symbol="GOOG", shares=50, avg_cost=Decimal("200")),
        },
    )
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("MSFT", row, portfolio)
    assert result.weights == {"MSFT": 0.0}
    assert result.reason != ""


def test_max_holdings_allows_existing_position() -> None:
    rule = MaxHoldingsRule(max_holdings=2)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100")),
            "GOOG": Position(symbol="GOOG", shares=50, avg_cost=Decimal("200")),
        },
    )
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("AAPL", row, portfolio)
    assert result.weights is None


def test_max_holdings_allows_below_limit() -> None:
    rule = MaxHoldingsRule(max_holdings=3)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100")),
        },
    )
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("MSFT", row, portfolio)
    assert result.weights is None


# ---------------------------------------------------------------------------
# RebalanceFrequencyRule
# ---------------------------------------------------------------------------


def test_rebalance_frequency_holds_within_interval() -> None:
    rule = RebalanceFrequencyRule(interval_days=5)
    portfolio = Portfolio(cash=Decimal("100000"))

    row1 = pd.Series({"close": 100.0}, name=pd.Timestamp("2024-01-02"))
    result1 = rule.evaluate("AAPL", row1, portfolio)
    assert result1.hold is False

    row2 = pd.Series({"close": 101.0}, name=pd.Timestamp("2024-01-03"))
    result2 = rule.evaluate("AAPL", row2, portfolio)
    assert result2.hold is True
    assert result2.reason != ""


def test_rebalance_frequency_allows_after_interval() -> None:
    rule = RebalanceFrequencyRule(interval_days=5)
    portfolio = Portfolio(cash=Decimal("100000"))

    # Need 5 trading days (bar evaluations), not 5 calendar days
    dates = pd.to_datetime([
        "2024-01-02",  # bar 0: first bar, allow (rebalance)
        "2024-01-03",  # bar 1
        "2024-01-04",  # bar 2
        "2024-01-05",  # bar 3
        "2024-01-08",  # bar 4 (skipped weekend)
        "2024-01-09",  # bar 5: allow (5 trading days since last rebalance)
    ])

    for d in dates[:-1]:
        rule.evaluate("AAPL", pd.Series({"close": 100.0}, name=d), portfolio)

    result = rule.evaluate(
        "AAPL", pd.Series({"close": 105.0}, name=dates[-1]), portfolio
    )
    assert result.hold is False


class TestRebalanceFrequencyRuleTradingDays:
    def test_counts_trading_days_not_calendar(self) -> None:
        """Interval should count trading days (step calls), not calendar days."""
        rule = RebalanceFrequencyRule(interval_days=3)
        portfolio = Portfolio(cash=Decimal("100000"))

        # Simulate: Mon, Tue, Wed, Thu, Fri, Mon (skip weekend)
        dates = pd.to_datetime([
            "2025-01-06",  # Mon - first bar, allow
            "2025-01-07",  # Tue - bar 1
            "2025-01-08",  # Wed - bar 2
            "2025-01-09",  # Thu - bar 3, allow (3 bars since last)
            "2025-01-10",  # Fri - bar 1
            "2025-01-13",  # Mon - bar 2 (3 calendar days but only 2 trading days)
        ])

        results = []
        for d in dates:
            row = pd.Series({"close": 100.0}, name=d)
            r = rule.evaluate("SYM", row, portfolio)
            results.append(r.hold)

        assert results[0] is False  # first bar: allow
        assert results[1] is True   # bar 1: hold
        assert results[2] is True   # bar 2: hold
        assert results[3] is False  # bar 3: allow (3 trading days)
        assert results[4] is True   # bar 1: hold
        assert results[5] is True   # Mon = only 2 trading days after Thu

    def test_multi_symbol_same_bar_counts_once(self) -> None:
        """Multiple symbols evaluated on same date should count as 1 trading day."""
        rule = RebalanceFrequencyRule(interval_days=2)
        portfolio = Portfolio(cash=Decimal("100000"))

        d1 = pd.Timestamp("2025-01-06")
        d2 = pd.Timestamp("2025-01-07")
        d3 = pd.Timestamp("2025-01-08")

        # Day 1: two symbols — first bar, allow
        r1a = rule.evaluate("A", pd.Series({"close": 1.0}, name=d1), portfolio)
        r1b = rule.evaluate("B", pd.Series({"close": 1.0}, name=d1), portfolio)
        assert r1a.hold is False
        assert r1b.hold is False

        # Day 2: two symbols — 1 trading day, hold
        r2a = rule.evaluate("A", pd.Series({"close": 1.0}, name=d2), portfolio)
        r2b = rule.evaluate("B", pd.Series({"close": 1.0}, name=d2), portfolio)
        assert r2a.hold is True
        assert r2b.hold is True

        # Day 3: two symbols — 2 trading days, allow
        r3a = rule.evaluate("A", pd.Series({"close": 1.0}, name=d3), portfolio)
        r3b = rule.evaluate("B", pd.Series({"close": 1.0}, name=d3), portfolio)
        assert r3a.hold is False
        assert r3b.hold is False


# ---------------------------------------------------------------------------
# CalendarRebalanceRule
# ---------------------------------------------------------------------------


def test_calendar_rebalance_allows_first_observed_bar_of_month() -> None:
    rule = CalendarRebalanceRule(schedule="month_start")
    portfolio = Portfolio(cash=Decimal("100000"))

    first = pd.Series({"close": 100.0}, name=pd.Timestamp("2024-01-02"))
    same_bar_other_symbol = pd.Series({"close": 100.0}, name=pd.Timestamp("2024-01-02"))
    next_day = pd.Series({"close": 101.0}, name=pd.Timestamp("2024-01-03"))
    next_month = pd.Series({"close": 102.0}, name=pd.Timestamp("2024-02-01"))

    assert rule.evaluate("AAA", first, portfolio).hold is False
    assert rule.evaluate("BBB", same_bar_other_symbol, portfolio).hold is False
    assert rule.evaluate("AAA", next_day, portfolio).hold is True
    assert rule.evaluate("AAA", next_month, portfolio).hold is False
