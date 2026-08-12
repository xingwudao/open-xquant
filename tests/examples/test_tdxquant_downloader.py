from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from examples.custom_data_sources.tdxquant_downloader import TdxQuantDownloader

from oxq.data.manifest import read_manifest
from oxq.data.providers import Downloader


class FakeResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def market_payload(
    symbol: str = "600519.SH",
    *,
    dates: list[str] | None = None,
) -> dict[str, object]:
    actual_dates = dates or ["20240103", "20240102"]
    return {
        "id": 1,
        "result": {
            "ErrorId": "0",
            "Value": {
                symbol: {
                    "ErrorId": "0",
                    "Date": actual_dates,
                    "Open": ["1810.00", "1800.00"],
                    "High": ["1860.00", "1850.00"],
                    "Low": ["1800.00", "1790.00"],
                    "Close": ["1850.00", "1840.00"],
                    "Volume": ["51000.00", "50000.00"],
                }
            },
            "has_more": False,
        },
    }


def test_downloader_satisfies_protocol() -> None:
    downloader: Downloader = TdxQuantDownloader()
    assert isinstance(downloader, Downloader)


def test_download_posts_expected_request_and_writes_standard_files(
    tmp_path: Path,
) -> None:
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(market_payload()),
    ) as mock_urlopen:
        path = TdxQuantDownloader().download(
            "600519.sh", "2024-01-02", "2024-01-03", tmp_path
        )

    request = mock_urlopen.call_args.args[0]
    body: dict[str, Any] = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://127.0.0.1:17709/"
    assert request.method == "POST"
    assert request.headers["Content-type"] == "application/json"
    assert mock_urlopen.call_args.kwargs == {"timeout": 10.0}
    assert body == {
        "id": 1,
        "method": "get_market_data",
        "params": {
            "field_list": ["Open", "High", "Low", "Close", "Volume"],
            "stock_list": ["600519.SH"],
            "start_time": "20240102",
            "end_time": "20240103",
            "count": -1,
            "dividend_type": "front",
            "period": "1d",
            "fill_data": False,
        },
    }

    assert path == tmp_path / "600519.SH.parquet"
    frame = pd.read_parquet(path)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.name == "date"
    assert str(frame.index.tz) == "Asia/Shanghai"
    assert frame.index.is_monotonic_increasing
    assert frame.index.strftime("%Y%m%d").tolist() == ["20240102", "20240103"]
    assert frame["volume"].dtype == "int64"
    assert frame.loc["2024-01-02", "close"] == 1840.0

    manifest = read_manifest(tmp_path / "600519.SH.manifest.json")
    assert manifest is not None
    assert manifest["provider"] == "tdxquant"
    assert manifest["symbol"] == "600519.SH"
    assert manifest["start"] == "2024-01-02"
    assert manifest["end"] == "2024-01-03"
    assert manifest["rows"] == 2
    assert manifest["extra"] == {
        "dividend_type": "front",
        "period": "1d",
        "transport": "tdxquant_http",
    }


def test_none_dividend_type_is_forwarded(tmp_path: Path) -> None:
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(market_payload()),
    ) as mock_urlopen:
        TdxQuantDownloader(dividend_type="none").download(
            "600519.SH", "2024-01-02", "2024-01-03", tmp_path
        )

    request = mock_urlopen.call_args.args[0]
    body: dict[str, Any] = json.loads(request.data.decode("utf-8"))
    params = body["params"]
    assert isinstance(params, dict)
    assert params["dividend_type"] == "none"
    manifest = read_manifest(tmp_path / "600519.SH.manifest.json")
    assert manifest is not None
    assert manifest["extra"]["dividend_type"] == "none"
