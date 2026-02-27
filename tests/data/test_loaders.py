from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oxq.core.errors import DownloadError
from oxq.data.loaders import (
    AkShareDownloader,
    Downloader,
    YFinanceDownloader,
    resolve_data_dir,
)


def test_resolve_with_explicit_dir(tmp_path: Path) -> None:
    result = resolve_data_dir(tmp_path / "custom")
    assert result == tmp_path / "custom"


def test_resolve_with_env_var(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "env"))
    result = resolve_data_dir()
    assert result == tmp_path / "env" / "market"


def test_resolve_default(monkeypatch) -> None:
    monkeypatch.delenv("OXQ_DATA_DIR", raising=False)
    result = resolve_data_dir()
    assert result == Path.home() / ".oxq" / "data" / "market"


def test_explicit_dir_overrides_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "env"))
    result = resolve_data_dir(tmp_path / "explicit")
    assert result == tmp_path / "explicit"


def test_yfinance_downloader_satisfies_protocol() -> None:
    downloader: Downloader = YFinanceDownloader()
    assert isinstance(downloader, Downloader)


def test_yfinance_download_saves_parquet(tmp_path) -> None:
    mock_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [104.0, 105.0],
            "Volume": [1000, 1100],
        },
        index=pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03"], name="Date"
        ),
    )
    with patch("oxq.data.loaders.yfinance", create=True) as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        downloader = YFinanceDownloader()
        path = downloader.download("AAPL", "2024-01-02", "2024-01-03", dest_dir=tmp_path)

    assert path == tmp_path / "AAPL.parquet"
    assert path.exists()
    result = pd.read_parquet(path)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert len(result) == 2


def test_yfinance_download_many(tmp_path) -> None:
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

        downloader = YFinanceDownloader()
        paths = downloader.download_many(
            ["AAPL", "MSFT"], "2024-01-02", "2024-01-03", dest_dir=tmp_path
        )

    assert set(paths.keys()) == {"AAPL", "MSFT"}
    assert all(p.exists() for p in paths.values())


def test_yfinance_download_empty_raises(tmp_path) -> None:
    with patch("oxq.data.loaders.yfinance", create=True) as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        downloader = YFinanceDownloader()
        with pytest.raises(DownloadError, match="AAPL"):
            downloader.download("AAPL", "2024-01-02", "2024-01-03", dest_dir=tmp_path)


def test_akshare_downloader_satisfies_protocol() -> None:
    downloader: Downloader = AkShareDownloader()
    assert isinstance(downloader, Downloader)


def test_akshare_download_saves_parquet(tmp_path) -> None:
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

    assert path == tmp_path / "600519.parquet"
    assert path.exists()
    result = pd.read_parquet(path)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert len(result) == 2


def test_akshare_download_empty_raises(tmp_path) -> None:
    with patch("oxq.data.loaders.akshare", create=True) as mock_ak:
        mock_ak.stock_zh_a_hist.return_value = pd.DataFrame()

        downloader = AkShareDownloader()
        with pytest.raises(DownloadError, match="600519"):
            downloader.download("600519", "20240102", "20240103", dest_dir=tmp_path)
