"""Tests for Composite signal."""

import pandas as pd
import pytest

from oxq.signals.composite import Composite


def _make_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame(
        {
            "sig_a": [True, True, False, True],
            "sig_b": [True, False, False, True],
            "sig_c": [False, False, True, True],
        },
        index=idx,
    )


def test_composite_and():
    result = Composite().compute(_make_df(), signals=["sig_a", "sig_b"], logic="and")
    assert list(result) == [True, False, False, True]


def test_composite_or():
    result = Composite().compute(_make_df(), signals=["sig_a", "sig_b"], logic="or")
    assert list(result) == [True, True, False, True]


def test_composite_three_signals_and():
    result = Composite().compute(_make_df(), signals=["sig_a", "sig_b", "sig_c"], logic="and")
    assert list(result) == [False, False, False, True]


def test_composite_three_signals_or():
    result = Composite().compute(_make_df(), signals=["sig_a", "sig_b", "sig_c"], logic="or")
    assert list(result) == [True, True, True, True]


def test_composite_single_signal():
    result = Composite().compute(_make_df(), signals=["sig_a"], logic="and")
    assert list(result) == [True, True, False, True]


def test_composite_empty_signals():
    result = Composite().compute(_make_df(), signals=[], logic="and")
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_composite_rejects_unknown_logic():
    with pytest.raises(ValueError, match="logic"):
        Composite().compute(_make_df(), signals=["sig_a", "sig_b"], logic="adn")


def test_composite_per_symbol():
    idx = pd.bdate_range("2024-01-01", periods=2)
    df_a = pd.DataFrame({"x": [True, False], "y": [True, True]}, index=idx)
    df_b = pd.DataFrame({"x": [False, True], "y": [True, False]}, index=idx)
    result_a = Composite().compute(df_a, signals=["x", "y"], logic="and")
    result_b = Composite().compute(df_b, signals=["x", "y"], logic="and")
    assert list(result_a) == [True, False]
    assert list(result_b) == [False, False]


def test_composite_has_name():
    assert Composite().name == "Composite"
