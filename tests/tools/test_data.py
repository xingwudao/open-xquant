"""Tests for oxq.tools.data — migrated from tests/mcp_server/test_data_tools.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oxq.tools.data import inspect_symbol, list_symbols, load_symbols


@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
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
    df.to_parquet(tmp_path / "MSFT.parquet")
    return tmp_path


def test_list_symbols(sample_data_dir: Path) -> None:
    result = list_symbols(data_dir=str(sample_data_dir))
    assert set(result["symbols"]) == {"AAPL", "MSFT"}
    assert result["count"] == 2


def test_list_symbols_empty(tmp_path: Path) -> None:
    result = list_symbols(data_dir=str(tmp_path))
    assert result["symbols"] == []
    assert result["count"] == 0


def test_inspect_symbol(sample_data_dir: Path) -> None:
    result = inspect_symbol(symbol="AAPL", data_dir=str(sample_data_dir))
    assert result["symbol"] == "AAPL"
    assert result["rows"] == 5
    assert result["columns"] == ["open", "high", "low", "close", "volume"]
    assert "date_range" in result


def test_inspect_missing_symbol(tmp_path: Path) -> None:
    result = inspect_symbol(symbol="UNKNOWN", data_dir=str(tmp_path))
    assert result["error"] is not None
    assert "UNKNOWN" in result["error"]


def test_load_symbols_yfinance(tmp_path: Path) -> None:
    mock_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [104.0],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
    )
    with patch("oxq.data.loaders.yfinance", create=True) as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        result = load_symbols(
            symbols=["AAPL"],
            start="2024-01-01",
            end="2024-12-31",
            source="yfinance",
            data_dir=str(tmp_path),
        )

    assert "AAPL" in result["rows"]
    assert result["rows"]["AAPL"] == 1
