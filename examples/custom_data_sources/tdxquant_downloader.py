from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from oxq.core.errors import DownloadError
from oxq.data.loaders import resolve_data_dir
from oxq.data.manifest import write_manifest

_FIELDS = ("Open", "High", "Low", "Close", "Volume")
_REQUEST_ID = 1


class TdxQuantDownloader:
    """Example downloader for the official local TdxQuant HTTP API."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:17709/",
        dividend_type: str = "front",
        timeout: float = 10.0,
    ) -> None:
        self.endpoint = endpoint
        self.dividend_type = dividend_type
        self.timeout = timeout

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        normalized_symbol = symbol.upper()
        start_compact = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d")
        end_compact = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d")
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
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            decoded: object = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise DownloadError("TdxQuant returned a non-object JSON response.")
        payload = cast(dict[str, object], decoded)
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


def _frame_from_payload(
    payload: dict[str, object], symbol: str, start: str, end: str
) -> pd.DataFrame:
    result = cast(dict[str, object], payload["result"])
    values = cast(dict[str, object], result["Value"])
    bars = cast(dict[str, object], values[symbol])
    series = {field: cast(list[object], bars[field]) for field in ("Date", *_FIELDS)}
    lengths = {len(items) for items in series.values()}
    if lengths == {0}:
        raise DownloadError(f"No data returned for '{symbol}' ({start} to {end}).")
    if len(lengths) != 1:
        raise DownloadError(f"TdxQuant returned uneven field lengths for '{symbol}'.")

    index = pd.DatetimeIndex(
        pd.to_datetime(series["Date"], format="%Y%m%d", errors="raise"),
        name="date",
    ).tz_localize("Asia/Shanghai")
    frame = pd.DataFrame(
        {
            field.lower(): pd.to_numeric(series[field], errors="raise")
            for field in _FIELDS
        },
        index=index,
    )
    frame = frame.sort_index()
    lower = pd.Timestamp(start, tz="Asia/Shanghai")
    upper = pd.Timestamp(end, tz="Asia/Shanghai")
    frame = frame.loc[(frame.index >= lower) & (frame.index <= upper)]
    numeric = frame[list(field.lower() for field in _FIELDS)].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        raise DownloadError(f"TdxQuant returned non-finite OHLCV data for '{symbol}'.")
    volume = frame["volume"]
    if not np.equal(volume, np.floor(volume)).all():
        raise DownloadError(f"TdxQuant returned non-integral volume for '{symbol}'.")
    frame[["open", "high", "low", "close"]] = frame[
        ["open", "high", "low", "close"]
    ].astype("float64")
    frame["volume"] = volume.astype("int64")
    return frame
