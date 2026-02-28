"""Tests for SMA indicator."""

import numpy as np
import pandas as pd

from oxq.core.types import Indicator
from oxq.indicators.sma import SMA


def _make_mktdata(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1000},
        index=dates,
    )


def test_sma_satisfies_indicator_protocol() -> None:
    assert isinstance(SMA(), Indicator)


def test_sma_basic() -> None:
    mktdata = _make_mktdata([10, 20, 30, 40, 50])
    result = SMA().compute(mktdata, period=3)
    assert len(result) == 5
    # First 2 values are NaN (period-1)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    # (10+20+30)/3 = 20, (20+30+40)/3 = 30, (30+40+50)/3 = 40
    assert result.iloc[2] == 20.0
    assert result.iloc[3] == 30.0
    assert result.iloc[4] == 40.0


def test_sma_default_column_is_close() -> None:
    mktdata = _make_mktdata([100, 200, 300])
    result = SMA().compute(mktdata, period=2)
    assert result.iloc[1] == 150.0  # (100+200)/2


def test_sma_custom_column() -> None:
    mktdata = _make_mktdata([10, 20, 30])
    mktdata["volume"] = [100, 200, 300]
    result = SMA().compute(mktdata, column="volume", period=2)
    assert result.iloc[1] == 150.0  # (100+200)/2


def test_sma_period_equals_length() -> None:
    mktdata = _make_mktdata([10, 20, 30])
    result = SMA().compute(mktdata, period=3)
    # Only last value is non-NaN
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 20.0  # (10+20+30)/3


def test_sma_has_name() -> None:
    assert SMA().name == "SMA"
