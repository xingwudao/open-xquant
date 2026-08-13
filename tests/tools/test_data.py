"""Tests for oxq.tools.data — migrated from test_data_tools.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def test_load_symbols_tushare_uses_environment_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "tool-secret")
    daily = pd.DataFrame(
        {
            "ts_code": ["600519.SH"],
            "trade_date": ["20240102"],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "vol": [10.0],
        }
    )
    factors = pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20240102"], "adj_factor": [2.0]})
    client = MagicMock()
    client.daily.return_value = daily
    client.adj_factor.return_value = factors
    tushare = SimpleNamespace(__version__="1.4.29", pro_api=MagicMock(return_value=client))
    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        result = load_symbols(
            symbols=["600519.SH"],
            start="20240102",
            end="20240102",
            source="tushare",
            data_dir=str(tmp_path),
        )
    assert result["rows"] == {"600519.SH": 1}
    tushare.pro_api.assert_called_once_with("tool-secret")


def test_load_symbols_unknown_source_lists_all_builtins() -> None:
    result = load_symbols([], "20240101", "20240102", source="unknown")
    assert result == {"error": "Unknown source 'unknown'. Use 'yfinance', 'akshare', or 'tushare'."}


def test_load_symbols_tushare_returns_partial_errors_without_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "tool-secret")
    daily = pd.DataFrame(
        {
            "ts_code": ["600519.SH"],
            "trade_date": ["20240102"],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "vol": [10.0],
        }
    )
    factors = pd.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20240102"], "adj_factor": [2.0]})
    client = MagicMock()
    client.daily.side_effect = [daily, RuntimeError("provider failure")]
    client.adj_factor.return_value = factors
    tushare = SimpleNamespace(__version__="1.4.29", pro_api=MagicMock(return_value=client))
    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        result = load_symbols(
            symbols=["600519.SH", "000001.SZ"],
            start="20240102",
            end="20240102",
            source="tushare",
            data_dir=str(tmp_path),
        )
    assert result["rows"] == {"600519.SH": 1}
    assert result["errors"] == {"000001.SZ": "Tushare request failed for '000001.SZ': provider failure"}
    assert "tool-secret" not in str(result)
