from __future__ import annotations

import importlib
import math
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from importlib import metadata
from typing import Any

from oxq.core.errors import DownloadError

_SYMBOL_PATTERN = re.compile(r"^[0-9]{6}\.(SH|SZ)$")
_HOST_PATTERN = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)"
    r"(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*$"
)
_INSTALL_HINT = (
    "uv run --with pytdx==1.72 python "
    "examples/custom_data_sources/pytdx_downloader.py --help"
)


def _normalize_symbol(symbol: str) -> tuple[str, int, str]:
    normalized = symbol.upper()
    match = _SYMBOL_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            "symbol must contain six digits and a .SH or .SZ suffix"
        )
    suffix = match.group(1)
    return normalized, 1 if suffix == "SH" else 0, normalized[:6]


def _parse_date_range(start: str, end: str) -> tuple[date, date]:
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            "start and end must be valid YYYY-MM-DD dates"
        ) from exc
    if start != start_date.isoformat() or end != end_date.isoformat():
        raise ValueError("start and end must be canonical YYYY-MM-DD dates")
    if start_date > end_date:
        raise ValueError("start date must not be later than end date")
    return start_date, end_date


def _load_pytdx() -> tuple[type[Any], str]:
    try:
        hq = importlib.import_module("pytdx.hq")
        api_class = hq.TdxHq_API
        version = metadata.version("pytdx")
    except (
        ModuleNotFoundError,
        ImportError,
        AttributeError,
        metadata.PackageNotFoundError,
    ) as exc:
        raise DownloadError(
            f"pytdx==1.72 is required; run: {_INSTALL_HINT}"
        ) from exc
    return api_class, version


@contextmanager
def _connected_api(
    host: str,
    port: int,
    timeout: float,
) -> Iterator[tuple[Any, str]]:
    api_class, version = _load_pytdx()
    api = api_class(
        multithread=False,
        heartbeat=False,
        auto_retry=False,
        raise_exception=True,
    )
    try:
        result = api.connect(host, port, time_out=timeout)
    except Exception as exc:
        raise DownloadError(
            f"Cannot connect to TDX quote server {host}:{port}."
        ) from exc
    if result is False or result is None:
        raise DownloadError(
            f"Cannot connect to TDX quote server {host}:{port}."
        )
    try:
        yield api, version
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            api.disconnect()
        except Exception as exc:
            if not active_exception:
                raise DownloadError(
                    f"Cannot disconnect from TDX quote server {host}:{port}."
                ) from exc


class PyTdxDownloader:
    """Example downloader that connects directly to a TDX quote server."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 7709,
        auto_adjust: bool = True,
        timeout: float = 5.0,
    ) -> None:
        if host != host.strip() or _HOST_PATTERN.fullmatch(host) is None:
            raise ValueError(
                "host must be an IPv4 address or hostname without scheme or port"
            )
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise ValueError("port must be an integer from 1 to 65535")
        if not isinstance(auto_adjust, bool):
            raise ValueError("auto_adjust must be a boolean")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        self.host = host
        self.port = port
        self.auto_adjust = auto_adjust
        self.timeout = timeout
