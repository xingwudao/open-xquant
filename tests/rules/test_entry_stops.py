"""Tests for EntryBarLowStopRule and TimeStopRule."""

from decimal import Decimal

import pandas as pd
import pytest

from oxq.core.types import Portfolio, Position, Rule
from oxq.rules.entry_stops import EntryBarLowStopRule, TimeStopRule

D = pd.Timestamp


def _held(avg_cost: str = "10") -> Portfolio:
    return Portfolio(cash=Decimal("0"), positions={"AAA": Position("AAA", 100, Decimal(avg_cost))})


def _flat() -> Portfolio:
    return Portfolio(cash=Decimal("1000"))


def _row(day: str, close: float, low: float) -> pd.Series:
    return pd.Series({"close": close, "low": low}, name=D(day))


# -- EntryBarLowStopRule ------------------------------------------------------


def test_entry_bar_low_stop_satisfies_rule_protocol() -> None:
    assert isinstance(EntryBarLowStopRule(), Rule)


def test_entry_bar_records_low_and_does_not_exit_same_bar() -> None:
    rule = EntryBarLowStopRule()
    assert rule.evaluate("AAA", _row("2024-01-02", 10.0, 9.5), _held()).target_positions is None


def test_exits_when_close_breaks_entry_bar_low() -> None:
    rule = EntryBarLowStopRule()
    rule.evaluate("AAA", _row("2024-01-02", 10.0, 9.5), _held())
    assert rule.evaluate("AAA", _row("2024-01-03", 9.6, 9.0), _held()).target_positions is None
    result = rule.evaluate("AAA", _row("2024-01-04", 9.4, 9.2), _held())
    assert result.target_positions == {"AAA": 0.0}


def test_buffer_lowers_stop_price() -> None:
    rule = EntryBarLowStopRule(buffer=0.02)  # stop = 9.5 * 0.98 = 9.31
    rule.evaluate("AAA", _row("2024-01-02", 10.0, 9.5), _held())
    assert rule.evaluate("AAA", _row("2024-01-03", 9.4, 9.3), _held()).target_positions is None
    assert rule.evaluate("AAA", _row("2024-01-04", 9.30, 9.2), _held()).target_positions == {"AAA": 0.0}


def test_second_call_on_same_bar_does_not_move_entry_low() -> None:
    rule = EntryBarLowStopRule()
    rule.evaluate("AAA", _row("2024-01-02", 10.0, 9.5), _held())
    rule.evaluate("AAA", _row("2024-01-02", 10.0, 9.5), _held())  # engine step 6
    assert rule.evaluate("AAA", _row("2024-01-03", 9.4, 9.0), _held()).target_positions == {"AAA": 0.0}


def test_state_resets_after_flat_observation() -> None:
    rule = EntryBarLowStopRule()
    rule.evaluate("AAA", _row("2024-01-02", 10.0, 9.5), _held())
    rule.evaluate("AAA", _row("2024-01-03", 9.0, 8.9), _flat())  # sold, seen flat
    rule.evaluate("AAA", _row("2024-01-04", 8.0, 7.5), _held())  # re-entry bar, new low 7.5
    assert rule.evaluate("AAA", _row("2024-01-05", 7.6, 7.4), _held()).target_positions is None


@pytest.mark.parametrize("buffer", [-0.01, 1.0, float("nan"), True])
def test_entry_bar_low_stop_rejects_bad_buffer(buffer: object) -> None:
    with pytest.raises(ValueError, match="buffer"):
        EntryBarLowStopRule(buffer=buffer)


# -- TimeStopRule -------------------------------------------------------------


def test_time_stop_satisfies_rule_protocol() -> None:
    assert isinstance(TimeStopRule(), Rule)


def test_time_stop_exits_after_max_bars_without_gain() -> None:
    rule = TimeStopRule(max_bars=3, min_return=0.0)
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    for day in days[:3]:
        assert rule.evaluate("AAA", _row(day, 10.0, 9.8), _held("10")).target_positions is None
    assert rule.evaluate("AAA", _row(days[3], 10.0, 9.8), _held("10")).target_positions == {"AAA": 0.0}


def test_time_stop_keeps_winner() -> None:
    rule = TimeStopRule(max_bars=3, min_return=0.0)
    for day in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        rule.evaluate("AAA", _row(day, 10.0, 9.8), _held("10"))
    assert rule.evaluate("AAA", _row("2024-01-05", 10.5, 10.1), _held("10")).target_positions is None


def test_time_stop_counts_each_date_once() -> None:
    rule = TimeStopRule(max_bars=3, min_return=0.0)
    for day in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        rule.evaluate("AAA", _row(day, 10.0, 9.8), _held("10"))
        rule.evaluate("AAA", _row(day, 10.0, 9.8), _held("10"))  # engine calls twice per bar
    assert rule.evaluate("AAA", _row("2024-01-04", 10.0, 9.8), _held("10")).target_positions is None


def test_time_stop_uses_recorded_entry_cost_not_live_avg_cost() -> None:
    rule = TimeStopRule(max_bars=1, min_return=0.0)
    rule.evaluate("AAA", _row("2024-01-02", 10.0, 9.8), _held("10"))
    # A rebalancing top-up raised avg_cost to 11; 10.5 is still a gain vs entry cost 10.
    assert rule.evaluate("AAA", _row("2024-01-03", 10.5, 10.2), _held("11")).target_positions is None


@pytest.mark.parametrize(("max_bars", "min_return"), [(0, 0.0), (2.0, 0.0), (True, 0.0), (3, float("inf")), (3, -1.0)])
def test_time_stop_rejects_bad_params(max_bars: object, min_return: object) -> None:
    with pytest.raises(ValueError, match="TimeStopRule"):
        TimeStopRule(max_bars=max_bars, min_return=min_return)
