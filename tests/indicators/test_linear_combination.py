"""Tests for LinearCombination indicator."""

import math

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Indicator
from oxq.indicators.linear_combination import LinearCombination


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]}, index=dates)


def test_linear_combination_satisfies_indicator_protocol() -> None:
    assert isinstance(LinearCombination(), Indicator)


def test_linear_combination_hand_calculated() -> None:
    # 3*a - 2*b + 1: row0 = 3 - 20 + 1 = -16; row1 = 6 - 40 + 1 = -33
    result = LinearCombination().compute(_frame(), columns=["a", "b"], weights=[3.0, -2.0], constant=1.0)
    assert result.iloc[:2].tolist() == pytest.approx([-16.0, -33.0])


def test_linear_combination_propagates_nan() -> None:
    result = LinearCombination().compute(_frame(), columns=["a", "b"], weights=[3.0, -2.0])
    assert math.isnan(result.iloc[2])


def test_linear_combination_mean_of_four_columns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=1)
    frame = pd.DataFrame({"m1": [4.0], "m2": [8.0], "m3": [12.0], "m4": [16.0]}, index=dates)
    result = LinearCombination().compute(frame, columns=["m1", "m2", "m3", "m4"], weights=[0.25] * 4)
    assert result.iloc[0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("columns", "weights"),
    [([], []), (["a"], [1.0, 2.0]), (["a", "a"], [1.0, 1.0]), (["a"], [float("nan")]), ("a", [1.0])],
)
def test_linear_combination_rejects_invalid_params(columns: object, weights: object) -> None:
    with pytest.raises(ValueError, match="LinearCombination"):
        LinearCombination().compute(_frame(), columns=columns, weights=weights)


def test_linear_combination_missing_column_raises_key_error() -> None:
    with pytest.raises(KeyError):
        LinearCombination().compute(_frame(), columns=["missing"], weights=[1.0])
