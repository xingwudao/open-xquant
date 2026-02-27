import pandas as pd
import pytest


@pytest.fixture()
def sample_mktdata() -> dict[str, pd.DataFrame]:
    """Three symbols with different volume/close profiles."""
    dates = pd.date_range("2024-01-02", periods=3, freq="B", name="date")
    return {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 152.0, 155.0], "volume": [2_000_000, 2_500_000, 3_000_000]},
            index=dates,
        ),
        "PENNY": pd.DataFrame(
            {"close": [0.5, 0.4, 0.3], "volume": [500_000, 400_000, 300_000]},
            index=dates,
        ),
        "GOOG": pd.DataFrame(
            {"close": [140.0, 141.0, 142.0], "volume": [1_500_000, 1_600_000, 1_800_000]},
            index=dates,
        ),
    }
