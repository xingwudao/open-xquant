"""Tests for Crossover signal."""

import pandas as pd

from oxq.core.types import Signal
from oxq.signals.crossover import Crossover


def _make_mktdata(
    fast_vals: list[float], slow_vals: list[float],
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=len(fast_vals))
    df = pd.DataFrame(
        {"close": fast_vals, "sma_10": fast_vals, "sma_50": slow_vals},
        index=dates,
    )
    return {"AAPL": df}


def test_crossover_satisfies_signal_protocol() -> None:
    assert isinstance(Crossover(), Signal)


def test_crossover_detects_cross_up() -> None:
    # Day 0: fast(8) <= slow(10)
    # Day 1: fast(9) <= slow(10)
    # Day 2: fast(11) > slow(10) → cross up!
    # Day 3: fast(12) > slow(10) → no cross (already above)
    mktdata = _make_mktdata(
        fast_vals=[8, 9, 11, 12],
        slow_vals=[10, 10, 10, 10],
    )
    result = Crossover().compute(mktdata, fast="sma_10", slow="sma_50")
    series = result["AAPL"]
    assert not series.iloc[1]   # no cross yet
    assert series.iloc[2]       # cross up here
    assert not series.iloc[3]   # already above, not a new cross


def test_crossover_no_signal_when_always_above() -> None:
    mktdata = _make_mktdata(
        fast_vals=[15, 16, 17],
        slow_vals=[10, 10, 10],
    )
    result = Crossover().compute(mktdata, fast="sma_10", slow="sma_50")
    # First value is NaN due to shift; rest should be False
    assert not result["AAPL"].iloc[1]
    assert not result["AAPL"].iloc[2]


def test_crossover_multi_symbol() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"sma_10": [8, 9, 11], "sma_50": [10, 10, 10]}, index=dates,
        ),
        "MSFT": pd.DataFrame(
            {"sma_10": [12, 11, 9], "sma_50": [10, 10, 10]}, index=dates,
        ),
    }
    result = Crossover().compute(mktdata, fast="sma_10", slow="sma_50")
    assert "AAPL" in result
    assert "MSFT" in result
    # AAPL crosses up on day 2
    assert result["AAPL"].iloc[2]
    # MSFT never crosses up (goes from above to below)
    assert not result["MSFT"].iloc[1]
    assert not result["MSFT"].iloc[2]


def test_crossover_has_name() -> None:
    assert Crossover().name == "Crossover"
