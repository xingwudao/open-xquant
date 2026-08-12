from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import examples.custom_data_sources.pytdx_downloader as module
import pytest
from examples.custom_data_sources.pytdx_downloader import PyTdxDownloader

from oxq.core.errors import DownloadError


@pytest.fixture(autouse=True)
def forbid_real_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not open a real socket")

    monkeypatch.setattr(socket, "socket", fail_socket)


@pytest.mark.parametrize(
    "host",
    ["", " host", "host ", "http://host", "host:7709", "user@host", "a/b"],
)
def test_rejects_invalid_host(host: str) -> None:
    with pytest.raises(ValueError, match="host"):
        PyTdxDownloader(host=host)


@pytest.mark.parametrize("port", [True, 0, 65536, -1, 7709.0])
def test_rejects_invalid_port(port: object) -> None:
    with pytest.raises(ValueError, match="port"):
        PyTdxDownloader(host="quote.example", port=port)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        PyTdxDownloader(host="quote.example", timeout=timeout)


def test_rejects_non_boolean_auto_adjust() -> None:
    with pytest.raises(ValueError, match="auto_adjust"):
        PyTdxDownloader(host="quote.example", auto_adjust=1)  # type: ignore[arg-type]


def test_normalizes_supported_symbols() -> None:
    assert module._normalize_symbol("510300.sh") == ("510300.SH", 1, "510300")
    assert module._normalize_symbol("159919.sz") == ("159919.SZ", 0, "159919")


@pytest.mark.parametrize("symbol", ["510300", "SH510300", "510300.BJ", "abc.SH"])
def test_rejects_unsupported_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match="six digits.*SH.*SZ"):
        module._normalize_symbol(symbol)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("20200501", "2026-01-01"),
        ("2020-5-1", "2026-01-01"),
        ("2020-02-30", "2026-01-01"),
        ("2026-01-02", "2026-01-01"),
    ],
)
def test_rejects_invalid_date_range(start: str, end: str) -> None:
    with pytest.raises(ValueError, match="date"):
        module._parse_date_range(start, end)


def test_missing_pytdx_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(DownloadError, match=r"uv run --with pytdx==1\.72"):
        module._load_pytdx()


def test_connected_api_uses_conservative_options_and_disconnects() -> None:
    api = MagicMock()
    api.connect.return_value = api
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with module._connected_api("quote.example", 7709, 4.5) as (opened, version):
            assert opened is api
            assert version == "1.72"

    api_class.assert_called_once_with(
        multithread=False,
        heartbeat=False,
        auto_retry=False,
        raise_exception=True,
    )
    api.connect.assert_called_once_with("quote.example", 7709, time_out=4.5)
    api.disconnect.assert_called_once_with()


def test_connected_api_rejects_false_connect_result() -> None:
    api = MagicMock()
    api.connect.return_value = False
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(DownloadError, match="Cannot connect"):
            with module._connected_api("quote.example", 7709, 5.0):
                pass


def test_disconnect_failure_is_a_download_error() -> None:
    api = MagicMock()
    api.connect.return_value = api
    api.disconnect.side_effect = RuntimeError("disconnect failed")
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(DownloadError, match="disconnect"):
            with module._connected_api("quote.example", 7709, 5.0):
                pass


def test_disconnect_failure_does_not_mask_body_error() -> None:
    api = MagicMock()
    api.connect.return_value = api
    api.disconnect.side_effect = RuntimeError("disconnect failed")
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(LookupError, match="body failed"):
            with module._connected_api("quote.example", 7709, 5.0):
                raise LookupError("body failed")
