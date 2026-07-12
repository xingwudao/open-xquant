import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Indicator
from oxq.indicators.rps import RPS


def test_rps_satisfies_indicator_protocol() -> None:
    assert isinstance(RPS(), Indicator)


def test_rps_single_symbol_compute_returns_numeric_nan_series() -> None:
    frame = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.bdate_range("2024-01-01", periods=2, tz="UTC"),
    )

    result = RPS().compute(frame)

    assert result.index.equals(frame.index)
    assert result.dtype == np.dtype("float64")
    assert result.isna().all()


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


@pytest.mark.parametrize("period", [0, -1])
def test_rps_rejects_non_positive_period(period: int) -> None:
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    data = {"AAA": pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)}

    with pytest.raises(ValueError, match="period must be a positive integer"):
        RPS().compute_cross_section(data, period=period)


@pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf])
def test_rps_rejects_non_finite_scale(scale: float) -> None:
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    data = {"AAA": pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)}

    with pytest.raises(ValueError, match="scale must be a positive finite real number"):
        RPS().compute_cross_section(data, scale=scale)
