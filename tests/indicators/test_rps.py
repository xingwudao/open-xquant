import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Indicator
from oxq.indicators.rps import RPS


def test_rps_satisfies_indicator_protocol() -> None:
    assert isinstance(RPS(), Indicator)


def test_rps_computes_cross_sectional_percentile_rank() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4, tz="UTC")
    data = {
        "AAA": pd.DataFrame({"close": [100.0, 100.0, 121.0, 133.1]}, index=dates),
        "BBB": pd.DataFrame({"close": [100.0, 100.0, 90.0, 81.0]}, index=dates),
        "CCC": pd.DataFrame({"close": [100.0, 100.0, 110.0, 121.0]}, index=dates),
    }

    result = RPS().compute_cross_section(data, column="close", period=2, scale=100.0)

    assert set(result) == {"AAA", "BBB", "CCC"}
    assert np.isnan(result["AAA"].iloc[0])
    assert np.isnan(result["AAA"].iloc[1])
    assert result["AAA"].iloc[2] == pytest.approx(100.0)
    assert result["CCC"].iloc[2] == pytest.approx(200.0 / 3.0)
    assert result["BBB"].iloc[2] == pytest.approx(100.0 / 3.0)


def test_rps_returns_nan_when_not_enough_cross_sectional_members() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    data = {
        "AAA": pd.DataFrame({"close": [100.0, 100.0, 121.0]}, index=dates),
        "BBB": pd.DataFrame({"close": [100.0, 100.0, np.nan]}, index=dates),
    }

    result = RPS().compute_cross_section(data, column="close", period=2, min_symbols=2)

    assert np.isnan(result["AAA"].iloc[2])
    assert np.isnan(result["BBB"].iloc[2])
