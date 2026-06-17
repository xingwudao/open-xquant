"""Tests for timezone discipline — Phase 1 of timezone-currency plan."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oxq.data.loaders import AkShareDownloader, YFinanceDownloader


def test_yfinance_preserves_timezone(tmp_path: Path) -> None:
    """YFinance data has timezone; _normalize_df must NOT strip it."""
    mock_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [104.0, 105.0],
            "Volume": [1000, 1100],
        },
        index=pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03"], name="Date",
        ).tz_localize("America/New_York"),
    )
    with patch("oxq.data.loaders.yfinance", create=True) as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        downloader = YFinanceDownloader()
        path = downloader.download("AAPL", "2024-01-02", "2024-01-03", dest_dir=tmp_path)

    result = pd.read_parquet(path)
    assert result.index.tz is not None, "YFinance timezone was stripped"


def test_akshare_marks_shanghai(tmp_path: Path) -> None:
    """AkShare data has no timezone; downloader must mark as Asia/Shanghai."""
    mock_df = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [1800.0, 1810.0],
            "最高": [1850.0, 1860.0],
            "最低": [1790.0, 1800.0],
            "收盘": [1840.0, 1850.0],
            "成交量": [50000, 51000],
        }
    )
    with patch("oxq.data.loaders.akshare", create=True) as mock_ak:
        mock_ak.stock_zh_a_hist.return_value = mock_df

        downloader = AkShareDownloader()
        path = downloader.download("600519", "20240102", "20240103", dest_dir=tmp_path)

    result = pd.read_parquet(path)
    assert result.index.tz is not None, "AkShare data should have timezone"
    assert str(result.index.tz) == "Asia/Shanghai"


def test_parquet_roundtrip_tz(tmp_path: Path) -> None:
    """Parquet files must preserve timezone through save/load cycle."""
    dates = pd.date_range("2024-01-02", periods=3, freq="B", name="date",
                          tz="America/New_York")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000, 1100, 1200],
        },
        index=dates,
    )
    path = tmp_path / "TEST.parquet"
    df.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert loaded.index.tz is not None
    assert str(loaded.index.tz) == "America/New_York"


def test_local_provider_legacy_parquet_gets_utc(
    tmp_path: Path, caplog,
) -> None:
    """Legacy parquet files without timezone get UTC + warning."""
    import logging

    from oxq.data.market import LocalMarketDataProvider

    # Create a naive (no tz) parquet file
    dates = pd.date_range("2024-01-02", periods=3, freq="B", name="date")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000, 1100, 1200],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "AAPL.parquet")

    provider = LocalMarketDataProvider(data_dir=tmp_path)
    with caplog.at_level(logging.WARNING):
        result = provider.get_bars("AAPL", "2024-01-01", "2024-12-31")

    assert result.index.tz is not None, "Legacy parquet should get UTC timezone"
    assert str(result.index.tz) == "UTC"
    assert "timezone" in caplog.text.lower()


def test_local_provider_rejects_unsafe_symbol_path(tmp_path: Path) -> None:
    from oxq.data.market import LocalMarketDataProvider

    provider = LocalMarketDataProvider(data_dir=tmp_path)

    with pytest.raises(ValueError, match="Unsafe symbol"):
        provider.get_bars("../outside", "2024-01-01", "2024-12-31")


def test_engine_rejects_naive_index() -> None:
    """Engine must reject data with naive (no timezone) DatetimeIndex."""
    from oxq.core.engine import Engine
    from oxq.core.strategy import Strategy
    from oxq.portfolio.optimizers import EqualWeightOptimizer
    from oxq.trade.sim_broker import SimBroker
    from oxq.universe.static import StaticUniverse

    # Create naive data (no timezone)
    dates = pd.bdate_range("2024-01-01", periods=10)
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": range(10),
            },
            index=dates,
        )
    }

    class FakeMarket:
        def get_bars(self, symbol, start, end):
            df = data[symbol]
            return df[(df.index >= start) & (df.index <= end)]

        def get_latest(self, symbol):
            return data[symbol].iloc[-1]

    strategy = Strategy(
        name="test",
        universe=StaticUniverse(("AAPL",)),
        signals={},
        portfolio=EqualWeightOptimizer(),
    )

    engine = Engine()
    # Engine should no longer silently default to Asia/Shanghai for naive data.
    # Instead it should raise an error.
    with pytest.raises(ValueError, match="timezone"):
        engine.run(
            strategy,
            market=FakeMarket(),
            broker=SimBroker(),
            start="2024-01-01",
            end="2024-01-15",
        )


def test_alpaca_preserves_utc_timezone() -> None:
    """Alpaca market data should preserve UTC timezone, not strip it."""
    from oxq.contrib.alpaca.market_data import _bar_to_df

    bars = [
        {"o": 100.0, "h": 105.0, "l": 99.0, "c": 104.0, "v": 1000,
         "t": "2024-01-02T05:00:00Z"},
        {"o": 101.0, "h": 106.0, "l": 100.0, "c": 105.0, "v": 1100,
         "t": "2024-01-03T05:00:00Z"},
    ]
    df = _bar_to_df(bars)
    assert df.index.tz is not None, "Alpaca timezone was stripped"
