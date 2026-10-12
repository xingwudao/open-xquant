"""Tests for FormulaEvent signal."""

import pandas as pd

from oxq.core.types import Signal
from oxq.signals.formula_event import FormulaEvent


def _df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame({"j": [5.0, -12.0, -15.0, 3.0], "v": [10, 5, 5, 10], "vma": [8, 8, 8, 8]}, index=idx)


def test_formula_event_satisfies_signal_protocol() -> None:
    assert isinstance(FormulaEvent(), Signal)


def test_formula_event_evaluates_expression_per_bar() -> None:
    result = FormulaEvent().compute(_df(), expr="j <= -10 and v < vma")
    assert list(result) == [False, True, True, False]
    assert result.dtype == bool
