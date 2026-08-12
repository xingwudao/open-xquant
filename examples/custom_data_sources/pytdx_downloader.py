from __future__ import annotations

import importlib
import math
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

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
_BAR_CATEGORY = 9
_PAGE_SIZE = 800
_MAX_PAGES = 128
_OHLC = ("open", "high", "low", "close")
_COLUMNS = (*_OHLC, "volume")


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


def _price(value: object, symbol: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (str, int, float, Decimal)
    ):
        raise DownloadError(f"TDX returned invalid OHLC for '{symbol}'.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DownloadError(f"TDX returned invalid OHLC for '{symbol}'.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise DownloadError(
            f"TDX returned invalid data; finite positive OHLC required for '{symbol}'."
        )
    return parsed


def _volume(value: object, symbol: str) -> int:
    message = f"TDX returned unsafe non-negative integer volume for '{symbol}'."
    if isinstance(value, bool) or not isinstance(
        value, (str, int, float, Decimal)
    ):
        raise DownloadError(message)
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DownloadError(message) from exc
    if (
        not parsed.is_finite()
        or parsed < 0
        or parsed != parsed.to_integral_value()
        or parsed > np.iinfo(np.int64).max
    ):
        raise DownloadError(message)
    return int(parsed)


def _parse_bar_page(payload: object, symbol: str) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise DownloadError(f"TDX bar response for '{symbol}' must be a list.")
    required = {"datetime", "open", "high", "low", "close", "vol"}
    rows: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise DownloadError(
                f"TDX returned a bar without required bar fields for '{symbol}'."
            )
        raw_datetime = item["datetime"]
        if not isinstance(raw_datetime, str):
            raise DownloadError(f"TDX returned an invalid bar date for '{symbol}'.")
        try:
            timestamp = pd.Timestamp(
                pd.to_datetime(raw_datetime, errors="raise")
            )
        except (TypeError, ValueError) as exc:
            raise DownloadError(
                f"TDX returned an invalid bar date for '{symbol}'."
            ) from exc
        values = {field: _price(item[field], symbol) for field in _OHLC}
        if not (
            values["low"] <= values["open"] <= values["high"]
            and values["low"] <= values["close"] <= values["high"]
        ):
            raise DownloadError(f"TDX returned inconsistent OHLC for '{symbol}'.")
        rows.append(
            {
                "date": timestamp.tz_localize(None).normalize(),
                **values,
                "volume": _volume(item["vol"], symbol),
            }
        )
    if not rows:
        return pd.DataFrame(
            {
                **{field: pd.Series(dtype="float64") for field in _OHLC},
                "volume": pd.Series(dtype="int64"),
            },
            index=pd.DatetimeIndex([], name="date", tz="Asia/Shanghai"),
        )
    frame = pd.DataFrame(rows).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date").tz_localize(
        "Asia/Shanghai"
    )
    frame[list(_OHLC)] = frame[list(_OHLC)].astype("float64")
    frame["volume"] = frame["volume"].astype("int64")
    return frame.loc[:, list(_COLUMNS)]


def _merge_bar_pages(
    pages: list[pd.DataFrame],
    symbol: str,
) -> pd.DataFrame:
    combined = pd.concat(pages)
    unique_rows: list[pd.DataFrame] = []
    for _, group in combined.groupby(level=0, sort=False):
        if len(group.drop_duplicates()) != 1:
            raise DownloadError(f"TDX returned conflicting bars for '{symbol}'.")
        unique_rows.append(group.iloc[[0]])
    result = pd.concat(unique_rows).sort_index()
    if result.index.has_duplicates:
        raise DownloadError(f"TDX returned duplicate bar dates for '{symbol}'.")
    return result.loc[:, list(_COLUMNS)]


def _fetch_raw_bars(
    api: Any,
    market: int,
    code: str,
    start_date: date,
    symbol: str,
) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    previous_fingerprint: tuple[tuple[object, ...], ...] | None = None
    for page_number in range(_MAX_PAGES):
        offset = page_number * _PAGE_SIZE
        try:
            payload = api.get_security_bars(
                _BAR_CATEGORY,
                market,
                code,
                offset,
                _PAGE_SIZE,
            )
        except Exception as exc:
            raise DownloadError(f"TDX bar request failed for '{symbol}'.") from exc
        if payload is None:
            raise DownloadError(f"TDX returned no bar response for '{symbol}'.")
        page = _parse_bar_page(payload, symbol)
        if page.empty:
            break
        fingerprint = tuple(
            page.reset_index().itertuples(index=False, name=None)
        )
        if fingerprint == previous_fingerprint:
            raise DownloadError(f"TDX repeated a bar page for '{symbol}'.")
        previous_fingerprint = fingerprint
        pages.append(page)
        if page.index.min().date() < start_date or len(page) < _PAGE_SIZE:
            break
    else:
        raise DownloadError(
            f"TDX bar pagination exceeded {_MAX_PAGES} pages for '{symbol}'."
        )
    if not pages:
        raise DownloadError(f"No data returned for '{symbol}'.")
    return _merge_bar_pages(pages, symbol)


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
