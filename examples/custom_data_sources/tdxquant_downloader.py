from __future__ import annotations

import argparse
import json
import math
import re
import socket
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from email.message import Message
from http.client import HTTPException
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np
import pandas as pd

from oxq.core.errors import DownloadError
from oxq.data.loaders import resolve_data_dir
from oxq.data.manifest import write_manifest

_FIELDS = ("Open", "High", "Low", "Close", "Volume")
_REQUEST_ID = 1
_SYMBOL_PATTERN = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> None:
        return None


_LOCAL_OPENER = build_opener(ProxyHandler({}), _NoRedirectHandler())
urlopen = _LOCAL_OPENER.open


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    try:
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in _LOOPBACK_HOSTS
            and parsed.port == 17709
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "endpoint must be a loopback HTTP URL on port 17709 using 127.0.0.1, "
            "localhost, or ::1"
        )


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.upper()
    if _SYMBOL_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "symbol must contain six digits and a .SH, .SZ, or .BJ suffix"
        )
    return normalized


def _parse_date_range(start: str, end: str) -> tuple[date, date]:
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("start and end must be valid YYYY-MM-DD dates") from exc
    if start != start_date.isoformat() or end != end_date.isoformat():
        raise ValueError("start and end must be canonical YYYY-MM-DD dates")
    if start_date > end_date:
        raise ValueError("start date must not be later than end date")
    return start_date, end_date


class TdxQuantDownloader:
    """Example downloader for the official local TdxQuant HTTP API."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:17709/",
        dividend_type: str = "front",
        timeout: float = 10.0,
    ) -> None:
        _validate_endpoint(endpoint)
        if dividend_type not in {"front", "none"}:
            raise ValueError("dividend_type must be 'front' or 'none'")
        timeout_message = "timeout must be finite and greater than zero"
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError(timeout_message)
        try:
            normalized_timeout = float(timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(timeout_message) from exc
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
            raise ValueError(timeout_message)
        self.endpoint = endpoint
        self.dividend_type = dividend_type
        self.timeout = normalized_timeout

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        normalized_symbol = _normalize_symbol(symbol)
        start_date, end_date = _parse_date_range(start, end)
        start_compact = start_date.strftime("%Y%m%d")
        end_compact = end_date.strftime("%Y%m%d")
        body: dict[str, object] = {
            "id": _REQUEST_ID,
            "method": "get_market_data",
            "params": {
                "field_list": list(_FIELDS),
                "stock_list": [normalized_symbol],
                "start_time": start_compact,
                "end_time": end_compact,
                "count": -1,
                "dividend_type": self.dividend_type,
                "period": "1d",
                "fill_data": False,
            },
        }
        payload = self._post_json(body)
        frame = _frame_from_payload(payload, normalized_symbol, start, end)

        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{normalized_symbol}.parquet"
        frame.to_parquet(path)
        write_manifest(
            parquet_path=path,
            symbol=normalized_symbol,
            provider="tdxquant",
            start=start,
            end=end,
            rows=len(frame),
            extra={
                "dividend_type": self.dividend_type,
                "period": "1d",
                "transport": "tdxquant_http",
            },
        )
        return path

    def _post_json(self, body: dict[str, object]) -> dict[str, object]:
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                if not 200 <= response.status < 300:
                    raise DownloadError(
                        f"TdxQuant returned HTTP {response.status}."
                    )
                raw = response.read()
        except HTTPError as exc:
            raise DownloadError(f"TdxQuant returned HTTP {exc.code}.") from exc
        except HTTPException as exc:
            raise DownloadError(
                "TdxQuant returned an incomplete HTTP response."
            ) from exc
        except (socket.timeout, TimeoutError) as exc:  # noqa: UP041
            raise DownloadError(
                f"TdxQuant request timed out for {self.endpoint} after "
                f"{self.timeout} seconds."
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise DownloadError(
                    f"TdxQuant request timed out for {self.endpoint} after "
                    f"{self.timeout} seconds."
                ) from exc
            raise DownloadError(
                "Cannot connect to TdxQuant; start a supported TdxQuant client "
                "and verify the local endpoint on port 17709."
            ) from exc
        except OSError as exc:
            raise DownloadError(
                "TdxQuant transport failed while reading the HTTP response."
            ) from exc

        try:
            decoded: object = json.loads(raw.decode("utf-8"), parse_float=Decimal)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DownloadError("TdxQuant did not return valid JSON.") from exc
        if not isinstance(decoded, dict):
            raise DownloadError("TdxQuant returned a non-object JSON response.")
        return cast(dict[str, object], decoded)

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return {
            symbol: self.download(symbol, start, end, dest_dir)
            for symbol in symbols
        }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DownloadError(f"TdxQuant response is missing a valid {label} object.")
    return cast(dict[str, object], value)


def _error_text(value: Mapping[str, object]) -> str:
    for key in ("ErrorMsg", "ErrorInfo", "Message"):
        message = value.get(key)
        if isinstance(message, str) and message:
            return message
    return "no error message"


def _bars_from_payload(
    payload: Mapping[str, object], symbol: str
) -> Mapping[str, object]:
    response_id = payload.get("id")
    if type(response_id) is not int or response_id != _REQUEST_ID:
        raise DownloadError("TdxQuant returned an unexpected request id.")
    result = _mapping(payload.get("result"), "result")
    top_error = str(result.get("ErrorId", "missing"))
    if top_error != "0":
        raise DownloadError(
            f"TdxQuant error {top_error}: {_error_text(result)}."
        )
    values = _mapping(result.get("Value"), "Value")
    bars = _mapping(values.get(symbol), f"Value[{symbol}]")
    symbol_error = str(bars.get("ErrorId", "missing"))
    if symbol_error != "0":
        raise DownloadError(
            f"TdxQuant error for {symbol}: {symbol_error}: {_error_text(bars)}."
        )
    return bars


def _volume_values(values: list[object], symbol: str) -> list[int]:
    limits = np.iinfo(np.int64)
    volume: list[int] = []
    for value in values:
        if isinstance(value, (bool, Decimal)):
            raise DownloadError(f"TdxQuant returned unsafe volume for '{symbol}'.")
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise DownloadError(
                f"TdxQuant returned unsafe volume for '{symbol}'."
            ) from exc
        if not parsed.is_finite() or parsed != parsed.to_integral_value():
            raise DownloadError(f"TdxQuant returned unsafe volume for '{symbol}'.")
        integer = int(parsed)
        if integer < limits.min or integer > limits.max:
            raise DownloadError(f"TdxQuant returned unsafe volume for '{symbol}'.")
        volume.append(integer)
    return volume


def _ohlc_values(values: list[object], symbol: str) -> list[float]:
    parsed_values: list[float] = []
    try:
        for value in values:
            if isinstance(value, bool):
                raise TypeError("boolean OHLC value")
            if not isinstance(value, (str, int, float, Decimal)):
                raise TypeError("unsupported OHLC value")
            parsed_values.append(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DownloadError(
            f"TdxQuant returned invalid dates or OHLCV values for '{symbol}'."
        ) from exc
    return parsed_values


def _frame_from_payload(
    payload: Mapping[str, object], symbol: str, start: str, end: str
) -> pd.DataFrame:
    bars = _bars_from_payload(payload, symbol)
    series: dict[str, list[object]] = {}
    for field in ("Date", *_FIELDS):
        items = bars.get(field)
        if not isinstance(items, list):
            raise DownloadError(
                f"TdxQuant response for '{symbol}' has no valid {field} array."
            )
        series[field] = cast(list[object], items)
    lengths = {len(items) for items in series.values()}
    if lengths == {0}:
        raise DownloadError(f"No data returned for '{symbol}' ({start} to {end}).")
    if len(lengths) != 1:
        raise DownloadError(f"TdxQuant returned uneven field lengths for '{symbol}'.")

    try:
        date_values = cast(list[str], series["Date"])
        index = pd.DatetimeIndex(
            pd.to_datetime(date_values, format="%Y%m%d", errors="raise"),
            name="date",
        ).tz_localize("Asia/Shanghai")
        if index.hasnans:
            raise ValueError("date array contains NaT")
        volume = _volume_values(series["Volume"], symbol)
        frame = pd.DataFrame(
            {
                **{
                    field.lower(): _ohlc_values(series[field], symbol)
                    for field in _FIELDS
                    if field != "Volume"
                },
                "volume": volume,
            },
            index=index,
        )
    except (TypeError, ValueError) as exc:
        raise DownloadError(
            f"TdxQuant returned invalid dates or OHLCV values for '{symbol}'."
        ) from exc

    if frame.index.has_duplicates:
        raise DownloadError(f"TdxQuant returned duplicate dates for '{symbol}'.")

    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(numeric).all():
        raise DownloadError(f"TdxQuant returned non-finite OHLCV data for '{symbol}'.")
    if (
        (frame[["open", "high", "low", "close"]] <= 0).to_numpy().any()
        or not (
            (frame["low"] <= frame["open"])
            & (frame["open"] <= frame["high"])
            & (frame["low"] <= frame["close"])
            & (frame["close"] <= frame["high"])
        ).all()
    ):
        raise DownloadError(
            f"TdxQuant requires positive and consistent OHLC for '{symbol}'."
        )

    frame = frame.sort_index()
    lower = pd.Timestamp(start, tz="Asia/Shanghai")
    upper = pd.Timestamp(end, tz="Asia/Shanghai")
    frame = frame.loc[(frame.index >= lower) & (frame.index <= upper)]
    if frame.empty:
        raise DownloadError(f"No data returned for '{symbol}' ({start} to {end}).")
    frame[["open", "high", "low", "close"]] = frame[
        ["open", "high", "low", "close"]
    ].astype("float64")
    frame["volume"] = frame["volume"].astype("int64")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download daily bars through the local TdxQuant HTTP API."
    )
    parser.add_argument("symbol", help="Six-digit symbol with .SH, .SZ, or .BJ")
    parser.add_argument("start", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("end", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--dest-dir", type=Path)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:17709/"
    )
    parser.add_argument(
        "--dividend-type", choices=("front", "none"), default="front"
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    downloader = TdxQuantDownloader(
        endpoint=args.endpoint,
        dividend_type=args.dividend_type,
        timeout=args.timeout,
    )
    path = downloader.download(
        args.symbol, args.start, args.end, args.dest_dir
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
