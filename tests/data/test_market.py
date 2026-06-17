from pathlib import Path

import pandas as pd
import pytest

from oxq.core.errors import SymbolNotFoundError
from oxq.data.market import LocalMarketDataProvider
from oxq.data.providers import MarketDataProvider


def test_satisfies_protocol(sample_data_dir: Path) -> None:
    provider: MarketDataProvider = LocalMarketDataProvider(data_dir=sample_data_dir)
    assert isinstance(provider, MarketDataProvider)


def test_get_bars_full_range(sample_data_dir: Path) -> None:
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    df = provider.get_bars("AAPL", "2024-01-01", "2024-12-31")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert len(df) == 5


def test_get_bars_date_filter(sample_data_dir: Path) -> None:
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    df = provider.get_bars("AAPL", "2024-01-03", "2024-01-04")
    assert len(df) == 2


def test_get_bars_filters_non_sessions_when_calendar_is_set(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-06"], utc=True)
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "SPY.parquet")
    provider = LocalMarketDataProvider(data_dir=tmp_path, calendar="XNYS")

    bars = provider.get_bars("SPY", "2024-01-02", "2024-01-06")

    assert list(bars.index) == [dates[0]]


def test_get_bars_sorts_source_rows_by_index(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2024-01-03", "2024-01-02"], utc=True)
    df = pd.DataFrame(
        {
            "open": [2.0, 1.0],
            "high": [2.0, 1.0],
            "low": [2.0, 1.0],
            "close": [2.0, 1.0],
            "volume": [100, 100],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "SPY.parquet")
    provider = LocalMarketDataProvider(data_dir=tmp_path)

    bars = provider.get_bars("SPY", "2024-01-02", "2024-01-03")

    assert list(bars.index) == sorted(dates)
    assert list(bars["close"]) == [1.0, 2.0]


def test_get_bars_rejects_duplicate_index(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True)
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "SPY.parquet")
    provider = LocalMarketDataProvider(data_dir=tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        provider.get_bars("SPY", "2024-01-02", "2024-01-03")


def test_get_bars_out_of_range_returns_available(sample_data_dir: Path) -> None:
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    df = provider.get_bars("AAPL", "2020-01-01", "2020-12-31")
    assert len(df) == 0


def test_get_bars_symbol_not_found(sample_data_dir: Path) -> None:
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    with pytest.raises(SymbolNotFoundError, match="MSFT"):
        provider.get_bars("MSFT", "2024-01-01", "2024-12-31")


def test_get_latest(sample_data_dir: Path) -> None:
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    s = provider.get_latest("AAPL")
    assert isinstance(s, pd.Series)
    assert s["close"] == 108.0


def test_get_latest_symbol_not_found(sample_data_dir: Path) -> None:
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    with pytest.raises(SymbolNotFoundError):
        provider.get_latest("MSFT")


def test_uses_resolve_data_dir_default(monkeypatch, sample_data_dir: Path) -> None:
    monkeypatch.setattr(
        "oxq.data.market.resolve_data_dir",
        lambda dest_dir=None: sample_data_dir,
    )
    provider = LocalMarketDataProvider()
    df = provider.get_bars("AAPL", "2024-01-01", "2024-12-31")
    assert len(df) == 5
