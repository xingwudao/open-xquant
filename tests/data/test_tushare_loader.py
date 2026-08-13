from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from oxq.data.loaders import Downloader, TushareDownloader


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600519.SH", "600519.SH"],
            "trade_date": ["20240102", "20240103"],
            "open": [100.0, 110.0],
            "high": [105.0, 115.0],
            "low": [95.0, 105.0],
            "close": [102.0, 112.0],
            "vol": [10.25, 20.0],
        }
    )


def _factors_with_later_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600519.SH", "600519.SH", "600519.SH"],
            "trade_date": ["20240102", "20240103", "20240104"],
            "adj_factor": [1.0, 1.5, 2.0],
        }
    )


def _module(client: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(__version__="1.4.29", pro_api=MagicMock(return_value=client))


def test_tushare_downloader_satisfies_protocol() -> None:
    downloader: Downloader = TushareDownloader(token="secret")
    assert isinstance(downloader, Downloader)


def test_tushare_download_writes_qfq_share_volume_and_manifest(tmp_path: Path) -> None:
    client = MagicMock()
    client.daily.return_value = _daily().iloc[::-1].reset_index(drop=True)
    client.adj_factor.return_value = _factors_with_later_reference().iloc[::-1].reset_index(drop=True)
    tushare = _module(client)

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        path = TushareDownloader(token="secret").download(
            "600519.SH", "2024-01-02", "2024-01-04", dest_dir=tmp_path
        )

    result = pd.read_parquet(path)
    assert path == tmp_path / "600519.SH.parquet"
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert str(result.index.tz) == "Asia/Shanghai"
    assert result.index.is_monotonic_increasing
    assert result["open"].tolist() == [50.0, 82.5]
    assert result["close"].tolist() == [51.0, 84.0]
    assert result["volume"].tolist() == [1025, 2000]
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "tushare"
    assert manifest["extra"] == {
        "adjust": "qfq",
        "adjustment_reference_date": "20240104",
        "adjustment_reference_factor": 2.0,
        "source_volume_unit": "lot",
        "volume_unit": "share",
        "tushare_version": "1.4.29",
    }
    assert "secret" not in path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    client.daily.assert_called_once_with(
        ts_code="600519.SH", start_date="20240102", end_date="20240104"
    )
    client.adj_factor.assert_called_once_with(
        ts_code="600519.SH", start_date="20240102", end_date="20240104"
    )
