from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a tmp dir with a sample AAPL.parquet file."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B", name="date")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "AAPL.parquet")
    return tmp_path
