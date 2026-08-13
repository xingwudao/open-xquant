from __future__ import annotations

import importlib
import json
import socket
from decimal import Decimal
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import examples.custom_data_sources.tdxquant_downloader as tdxquant_downloader
import pandas as pd
import pytest
from examples.custom_data_sources.tdxquant_downloader import TdxQuantDownloader, main

from oxq.core.errors import DownloadError
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


class RawResponse:
    def __init__(self, raw: bytes, status: int = 200) -> None:
        self.raw = raw
        self.status = status

    def __enter__(self) -> RawResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.raw


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


def test_local_transport_disables_environment_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:8080")
    monkeypatch.delenv("no_proxy", raising=False)
    module = importlib.reload(tdxquant_downloader)
    opener = getattr(module.urlopen, "__self__", None)

    assert opener is not None
    assert all(
        handler.__class__.__name__ != "ProxyHandler" for handler in opener.handlers
    )
    redirect_handler = next(
        handler
        for handler in opener.handlers
        if handler.__class__.__name__ == "_NoRedirectHandler"
    )
    assert (
        redirect_handler.redirect_request(
            Request("http://127.0.0.1:17709/"),
            None,
            302,
            "Found",
            {},
            "http://127.0.0.1:17709/redirected",
        )
        is None
    )


def test_ohlc_values_accept_finite_decimal() -> None:
    assert tdxquant_downloader._ohlc_values([Decimal("1810.25")], "600519.SH") == [
        1810.25
    ]


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


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:17709/",
        "http://192.168.1.10:17709/",
        "http://tdx.example.com:17709/",
        "not-a-url",
    ],
)
def test_rejects_non_loopback_http_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="loopback HTTP"):
        TdxQuantDownloader(endpoint=endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:17709/",
        "http://localhost:17709/",
        "http://[::1]:17709/",
    ],
)
def test_accepts_loopback_http_endpoint(endpoint: str) -> None:
    assert TdxQuantDownloader(endpoint=endpoint).endpoint == endpoint


@pytest.mark.parametrize("dividend_type", ["back", "qfq", ""])
def test_rejects_unsupported_dividend_type(dividend_type: str) -> None:
    with pytest.raises(ValueError, match="front.*none"):
        TdxQuantDownloader(dividend_type=dividend_type)


@pytest.mark.parametrize(
    "timeout",
    [True, "10", object(), 10**1000, 0.0, -1.0, float("nan"), float("inf")],
)
def test_rejects_non_finite_or_non_positive_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="finite.*greater than zero"):
        TdxQuantDownloader(timeout=timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "symbol",
    ["600519", "SH600519", "600519.XSHG", "ABC.SH", "60051.SH"],
)
def test_rejects_invalid_symbol(symbol: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="six digits.*SH.*SZ.*BJ"):
        TdxQuantDownloader().download(
            symbol, "2024-01-02", "2024-01-03", tmp_path
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("20240102", "2024-01-03"),
        ("2024-1-2", "2024-01-03"),
        ("2024-02-30", "2024-03-01"),
        ("2024-01-03", "2024-01-02"),
    ],
)
def test_rejects_invalid_date_range(start: str, end: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="date"):
        TdxQuantDownloader().download("600519.SH", start, end, tmp_path)


def test_rejects_unlocalizable_date_before_request(tmp_path: Path) -> None:
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen"
    ) as mock_urlopen:
        with pytest.raises(ValueError, match="date"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-01", "9999-12-31", tmp_path
            )

    mock_urlopen.assert_not_called()


@pytest.mark.parametrize(
    ("side_effect", "message"),
    [
        (URLError("connection refused"), "start.*TdxQuant.*17709"),
        (
            socket.timeout("timed out"),  # noqa: UP041
            "timed out.*http://127.0.0.1:17709/.*10.0",
        ),  # noqa: UP041
        (
            HTTPError(
                "http://127.0.0.1:17709/", 503, "unavailable", {}, None
            ),
            "HTTP 503",
        ),
    ],
)
def test_transport_errors_are_wrapped(
    side_effect: Exception,
    message: str,
    tmp_path: Path,
) -> None:
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        side_effect=side_effect,
    ):
        with pytest.raises(DownloadError, match=message):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_truncated_http_response_is_wrapped_without_writes(tmp_path: Path) -> None:
    response = FakeResponse(market_payload())
    response.read = MagicMock(  # type: ignore[method-assign]
        side_effect=IncompleteRead(b"partial", 100)
    )

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=response,
    ):
        with pytest.raises(DownloadError, match="incomplete HTTP response"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )

    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_socket_read_error_is_wrapped_without_writes(tmp_path: Path) -> None:
    response = FakeResponse(market_payload())
    response.read = MagicMock(  # type: ignore[method-assign]
        side_effect=ConnectionResetError("peer reset")
    )

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=response,
    ):
        with pytest.raises(DownloadError, match="reading the HTTP response"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )

    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "reason",
    [TimeoutError("timed out"), socket.timeout("timed out")],  # noqa: UP041
)
def test_wrapped_timeout_errors_include_endpoint_and_duration(
    reason: BaseException,
    tmp_path: Path,
) -> None:
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        side_effect=URLError(reason),
    ):
        with pytest.raises(
            DownloadError,
            match="timed out.*http://127.0.0.1:17709/.*10.0",
        ):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_invalid_json_is_rejected_without_writes(tmp_path: Path) -> None:
    response = FakeResponse({})
    response.read = lambda: b"not-json"  # type: ignore[method-assign]
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=response,
    ):
        with pytest.raises(DownloadError, match="valid JSON"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not (tmp_path / "600519.SH.parquet").exists()


def test_oversized_json_integer_is_rejected_without_writes(tmp_path: Path) -> None:
    response = RawResponse(b'{"id":' + b"1" * 5000 + b"}")
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=response,
    ):
        with pytest.raises(DownloadError, match="valid JSON"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )

    assert not (tmp_path / "600519.SH.parquet").exists()


def test_raw_non_integral_large_json_volume_is_rejected_without_writes(
    tmp_path: Path,
) -> None:
    raw = json.dumps(market_payload()).replace(
        '"51000.00"', "9007199254740993.0"
    ).encode("utf-8")
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=RawResponse(raw),
    ):
        with pytest.raises(DownloadError, match="unsafe volume"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not (tmp_path / "600519.SH.parquet").exists()
    assert not (tmp_path / "600519.SH.manifest.json").exists()


def test_non_2xx_response_is_rejected_without_reading_body(tmp_path: Path) -> None:
    response = FakeResponse({}, status=503)
    response.read = MagicMock(side_effect=AssertionError("body must not be read"))
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=response,
    ):
        with pytest.raises(DownloadError, match="HTTP 503"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not (tmp_path / "600519.SH.parquet").exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"id": 2, "result": {}}, "request id"),
        ({"id": True, "result": {}}, "request id"),
        ({"id": 1}, "result"),
        (
            {"id": 1, "result": {"ErrorId": "100", "ErrorMsg": "failed"}},
            "TdxQuant error 100.*failed",
        ),
        (
            {
                "id": 1,
                "result": {
                    "ErrorId": "0",
                    "Value": {
                        "600519.SH": {"ErrorId": "101", "ErrorMsg": "missing"}
                    },
                },
            },
            "600519.SH.*101.*missing",
        ),
    ],
)
def test_response_errors_are_rejected_without_writes(
    payload: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match=message):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
    )
    assert not (tmp_path / "600519.SH.parquet").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("High", ["1860.00"], "uneven field lengths"),
        ("Open", ["bad", "1800.00"], "invalid dates or OHLCV"),
        ("Date", ["20241301", "20240102"], "invalid dates or OHLCV"),
        ("Date", ["NaT", "20240102"], "invalid dates or OHLCV"),
        ("Date", ["20240102", "20240102"], "duplicate dates"),
        ("Volume", ["1.5", "2.0"], "unsafe volume"),
        ("Close", ["NaN", "1840.00"], "non-finite OHLCV"),
    ],
)
def test_malformed_market_arrays_are_rejected(
    field: str,
    value: list[str],
    message: str,
    tmp_path: Path,
) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    bars[field] = value

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match=message):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not (tmp_path / "600519.SH.parquet").exists()
    assert not (tmp_path / "600519.SH.manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Open", "0"),
        ("Low", "1900.00"),
        ("Close", "1900.00"),
    ],
)
def test_non_positive_or_inconsistent_ohlc_is_rejected_without_writes(
    field: str,
    value: str,
    tmp_path: Path,
) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    items = bars[field]
    assert isinstance(items, list)
    bars[field] = [value, *items[1:]]

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match="positive and consistent OHLC"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )

    assert not (tmp_path / "600519.SH.parquet").exists()
    assert not (tmp_path / "600519.SH.manifest.json").exists()


def test_empty_market_arrays_are_rejected(tmp_path: Path) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    for field in ("Date", "Open", "High", "Low", "Close", "Volume"):
        bars[field] = []

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match="No data"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )


def test_non_array_market_field_is_rejected(tmp_path: Path) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    bars["Open"] = "not-an-array"

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match="valid Open array"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )


def test_parquet_write_failure_does_not_write_success_manifest(
    tmp_path: Path,
) -> None:
    with (
        patch(
            "examples.custom_data_sources.tdxquant_downloader.urlopen",
            return_value=FakeResponse(market_payload()),
        ),
        patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk full")),
        patch(
            "examples.custom_data_sources.tdxquant_downloader.write_manifest"
        ) as mock_manifest,
    ):
        with pytest.raises(OSError, match="disk full"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    mock_manifest.assert_not_called()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:7709/",
        "http://localhost:80/",
        "http://[::1]:17710/",
    ],
)
def test_rejects_loopback_endpoint_without_official_port(endpoint: str) -> None:
    with pytest.raises(ValueError, match="17709"):
        TdxQuantDownloader(endpoint=endpoint)


@pytest.mark.parametrize(
    ("volume", "accepted"),
    [
        ("-9223372036854775808", False),
        ("-1", False),
        ("0", True),
        ("9223372036854775807", True),
        ("-9223372036854775809", False),
        ("9223372036854775808", False),
    ],
)
def test_validates_volume_at_exact_int64_bounds(
    volume: str,
    accepted: bool,
    tmp_path: Path,
) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    bars["Volume"] = [volume, volume]

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        if accepted:
            path = TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
        else:
            with pytest.raises(DownloadError, match="unsafe volume"):
                TdxQuantDownloader().download(
                    "600519.SH", "2024-01-02", "2024-01-03", tmp_path
                )
            return

    assert pd.read_parquet(path)["volume"].tolist() == [int(volume), int(volume)]


def test_rejects_huge_decimal_volume_before_integer_materialization() -> None:
    with (
        patch.object(
            tdxquant_downloader,
            "int",
            side_effect=AssertionError("unsafe integer materialization"),
            create=True,
        ),
        pytest.raises(DownloadError, match="unsafe volume"),
    ):
        tdxquant_downloader._volume_values(["1e100000000"], "600519.SH")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Open", "bad", "invalid dates or OHLCV"),
        ("Close", "NaN", "non-finite OHLCV"),
        ("Volume", "1.5", "unsafe volume"),
    ],
)
def test_rejects_invalid_out_of_window_market_data(
    field: str,
    value: str,
    message: str,
    tmp_path: Path,
) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    bars["Date"] = ["20240104", *bars["Date"]]
    for market_field in ("Open", "High", "Low", "Close", "Volume"):
        items = bars[market_field]
        assert isinstance(items, list)
        bars[market_field] = [value if market_field == field else items[0], *items]

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match=message):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not (tmp_path / "600519.SH.parquet").exists()


@pytest.mark.parametrize(
    ("date", "value"),
    [
        ("20240102", True),
        ("20240104", False),
    ],
)
def test_rejects_boolean_ohlc_before_date_range_filtering(
    date: str,
    value: bool,
    tmp_path: Path,
) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    if date == "20240104":
        bars["Date"] = [date, *bars["Date"]]
        for field in ("Open", "High", "Low", "Close", "Volume"):
            items = bars[field]
            assert isinstance(items, list)
            bars[field] = [items[0], *items]
    items = bars["Open"]
    assert isinstance(items, list)
    bars["Open"] = [value, *items[1:]]

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match="invalid dates or OHLCV"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not (tmp_path / "600519.SH.parquet").exists()


def test_rejects_duplicate_dates_outside_requested_window(tmp_path: Path) -> None:
    payload = market_payload()
    result = payload["result"]
    assert isinstance(result, dict)
    values = result["Value"]
    assert isinstance(values, dict)
    bars = values["600519.SH"]
    assert isinstance(bars, dict)
    bars["Date"] = ["20240104", "20240104", *bars["Date"]]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        items = bars[field]
        assert isinstance(items, list)
        bars[field] = [items[0], items[0], *items]

    with patch(
        "examples.custom_data_sources.tdxquant_downloader.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(DownloadError, match="duplicate dates"):
            TdxQuantDownloader().download(
                "600519.SH", "2024-01-02", "2024-01-03", tmp_path
            )
    assert not (tmp_path / "600519.SH.parquet").exists()


def test_download_many_is_serial_and_preserves_requested_keys(tmp_path: Path) -> None:
    downloader = TdxQuantDownloader(dividend_type="none")
    with patch.object(downloader, "download") as mock_download:
        mock_download.side_effect = [
            tmp_path / "600519.SH.parquet",
            tmp_path / "000001.SZ.parquet",
        ]
        result = downloader.download_many(
            ["600519.SH", "000001.SZ"],
            "2024-01-02",
            "2024-01-03",
            tmp_path,
        )

    assert list(result) == ["600519.SH", "000001.SZ"]
    assert mock_download.call_args_list == [
        call("600519.SH", "2024-01-02", "2024-01-03", tmp_path),
        call("000001.SZ", "2024-01-02", "2024-01-03", tmp_path),
    ]


def test_main_downloads_one_symbol_and_prints_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "600519.SH.parquet"
    fake_downloader = MagicMock()
    fake_downloader.download.return_value = output
    with patch(
        "examples.custom_data_sources.tdxquant_downloader.TdxQuantDownloader",
        return_value=fake_downloader,
    ) as downloader_type:
        result = main(
            [
                "600519.SH",
                "2024-01-02",
                "2024-01-03",
                "--dest-dir",
                str(tmp_path),
                "--dividend-type",
                "none",
                "--timeout",
                "2.5",
            ]
        )

    assert result == 0
    downloader_type.assert_called_once_with(
        endpoint="http://127.0.0.1:17709/",
        dividend_type="none",
        timeout=2.5,
    )
    fake_downloader.download.assert_called_once_with(
        "600519.SH", "2024-01-02", "2024-01-03", tmp_path
    )
    assert capsys.readouterr().out.strip() == str(output)
