"""Tests for TdxSMA indicator (TongDaXin SMA(X, N, M))."""

import math

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Indicator
from oxq.indicators.tdx_sma import TdxSMA


def _frame(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.DataFrame({"close": values}, index=dates)


def test_tdx_sma_satisfies_indicator_protocol() -> None:
    assert isinstance(TdxSMA(), Indicator)


def test_tdx_sma_hand_calculated_recursion() -> None:
    # N=3, M=1: Y0 = 10; Y1 = (13 + 2*10)/3 = 11; Y2 = (16 + 2*11)/3 = 12.666...
    result = TdxSMA().compute(_frame([10.0, 13.0, 16.0]), column="close", n=3, m=1)
    assert result.tolist() == pytest.approx([10.0, 11.0, 38.0 / 3.0])


def test_tdx_sma_leading_nan_starts_at_first_valid_value() -> None:
    result = TdxSMA().compute(_frame([np.nan, np.nan, 10.0, 13.0]), column="close", n=3, m=1)
    assert math.isnan(result.iloc[0]) and math.isnan(result.iloc[1])
    assert result.iloc[2:].tolist() == pytest.approx([10.0, 11.0])


def test_tdx_sma_skips_interior_nan_in_recursion() -> None:
    # Interior NaN (e.g. RSV 0/0 on a flat 9-day window) must not reset the state.
    # Y after NaN uses the last valid Y: (19 + 2*12.666...)/3 = 14.777...
    result = TdxSMA().compute(_frame([10.0, 13.0, 16.0, np.nan, 19.0]), column="close", n=3, m=1)
    assert result.iloc[3] == pytest.approx(38.0 / 3.0)
    assert result.iloc[4] == pytest.approx((19.0 + 2.0 * 38.0 / 3.0) / 3.0)


@pytest.mark.parametrize(("n", "m"), [(0, 1), (3, 0), (3, 4), (3.0, 1), (True, 1)])
def test_tdx_sma_rejects_invalid_weights(n: object, m: object) -> None:
    with pytest.raises(ValueError, match="TdxSMA requires integers 0 < m <= n"):
        TdxSMA().compute(_frame([1.0, 2.0]), column="close", n=n, m=m)
