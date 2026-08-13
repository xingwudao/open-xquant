from __future__ import annotations

import os
import re
from datetime import datetime
from importlib import import_module as _import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from oxq.core.errors import DownloadError
from oxq.data.manifest import write_manifest
from oxq.data.providers import Downloader

__all__ = [
    "AkShareDownloader",
    "Downloader",
    "TushareDownloader",
    "YFinanceDownloader",
    "resolve_data_dir",
]


_TUSHARE_SYMBOL_RE = re.compile(r"^[0-9]{6}\.(SH|SZ|BJ)$")
_TUSHARE_DATE_RE = re.compile(r"^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{8})$")
importlib = SimpleNamespace(import_module=_import_module)


def _normalize_tushare_date(value: str, *, field: str) -> str:
    if not _TUSHARE_DATE_RE.fullmatch(value):
        raise DownloadError(
            f"Invalid Tushare {field} date '{value}'. Use YYYY-MM-DD or YYYYMMDD."
        )
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise DownloadError(f"Invalid Tushare {field} date '{value}'. Use YYYY-MM-DD or YYYYMMDD.")


def resolve_data_dir(dest_dir: Path | None = None) -> Path:
    """Resolve data storage directory. Priority: parameter > OXQ_DATA_DIR > default."""
    if dest_dir is not None:
        return _expand_path(dest_dir)
    env = os.environ.get("OXQ_DATA_DIR")
    if env:
        return _expand_path(Path(env) / "market")
    return Path.home() / ".oxq" / "data" / "market"


def _expand_path(path: Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw API DataFrame to standard schema.

    Preserves timezone information on the index. If the source provides
    a tz-aware DatetimeIndex, it is kept as-is.
    """
    df = df.rename(columns=str.lower)
    df = df.rename_axis("date")
    cols = ["open", "high", "low", "close", "volume"]
    df = df[cols]
    df["volume"] = df["volume"].astype("int64")
    return df


def _normalize_tushare_frames(
    daily: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    symbol: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, str, float]:
    if daily.empty:
        raise DownloadError(f"No Tushare daily data returned for '{symbol}' ({start} to {end}).")
    if factors.empty:
        raise DownloadError(
            f"No Tushare adjustment factors returned for '{symbol}' ({start} to {end})."
        )

    daily_columns = {"trade_date", "open", "high", "low", "close", "vol"}
    factor_columns = {"trade_date", "adj_factor"}
    if not daily_columns.issubset(daily.columns):
        raise DownloadError("Tushare daily response is missing required fields.")
    if not factor_columns.issubset(factors.columns):
        raise DownloadError("Tushare adjustment response is missing required fields.")

    factor_dates = factors["trade_date"].astype(str)
    eligible_factors = factors.loc[factor_dates <= end].copy()
    if eligible_factors.empty:
        raise DownloadError(
            "No Tushare adjustment factors on or before the requested end date."
        )
    eligible_dates = eligible_factors["trade_date"].astype(str)
    reference_index = eligible_dates.idxmax()
    reference_date = eligible_dates.loc[reference_index]
    reference_factor = float(eligible_factors.loc[reference_index, "adj_factor"])
    if not np.isfinite(reference_factor) or reference_factor <= 0:
        raise DownloadError("Tushare adjustment reference factor must be positive and finite.")

    merged = daily.loc[:, ["trade_date", "open", "high", "low", "close", "vol"]].merge(
        eligible_factors.loc[:, ["trade_date", "adj_factor"]],
        on="trade_date",
        how="left",
        validate="one_to_one",
    )
    if merged["adj_factor"].isna().any():
        raise DownloadError("Tushare daily data is missing adjustment factors.")

    adjustment = merged["adj_factor"].astype(float) / reference_factor
    values = merged.loc[:, ["open", "high", "low", "close"]].astype(float)
    values = values.mul(adjustment, axis="index")

    scaled_volume = merged["vol"].astype(float).to_numpy() * 100
    rounded_volume = np.rint(scaled_volume)
    if not np.all(np.isfinite(scaled_volume)) or not np.all(
        np.abs(scaled_volume - rounded_volume) <= 1e-6
    ):
        raise DownloadError("Tushare volume must convert from lots to whole shares.")
    int64_limits = np.iinfo(np.int64)
    if not np.all(
        (rounded_volume >= int64_limits.min) & (rounded_volume <= int64_limits.max)
    ):
        raise DownloadError("Tushare volume must fit within int64 share limits.")

    frame = values.assign(volume=rounded_volume.astype("int64"))
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(merged["trade_date"], format="%Y%m%d")
    ).tz_localize("Asia/Shanghai")
    frame.index.name = "date"
    return frame.sort_index(), reference_date, reference_factor


class TushareDownloader:
    def __init__(self, token: str | None = None) -> None:
        self._explicit_token = token
        self._resolved_token: str | None = None
        self._module: Any = None
        self._client: Any = None

    def _get_module_and_client(self) -> tuple[Any, Any]:
        if self._client is not None:
            return self._module, self._client
        raw_token = (
            self._explicit_token
            if self._explicit_token is not None
            else os.environ.get("TUSHARE_TOKEN")
        )
        token = raw_token.strip() if raw_token is not None else ""
        if not token:
            raise DownloadError(
                "Tushare token is required; pass token or set TUSHARE_TOKEN."
            )
        module = importlib.import_module("tushare")
        client = module.pro_api(token)
        self._resolved_token = token
        self._module = module
        self._client = client
        return module, client

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        normalized_start = _normalize_tushare_date(start, field="start")
        normalized_end = _normalize_tushare_date(end, field="end")
        if not _TUSHARE_SYMBOL_RE.fullmatch(symbol):
            raise DownloadError(f"Invalid Tushare symbol '{symbol}'.")
        if normalized_start > normalized_end:
            raise DownloadError("Tushare start date must not be after end date.")
        module, client = self._get_module_and_client()
        daily = client.daily(
            ts_code=symbol, start_date=normalized_start, end_date=normalized_end
        )
        factors = client.adj_factor(
            ts_code=symbol, start_date=normalized_start, end_date=normalized_end
        )
        frame, reference_date, reference_factor = _normalize_tushare_frames(
            daily,
            factors,
            symbol=symbol,
            start=normalized_start,
            end=normalized_end,
        )
        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{symbol}.parquet"
        frame.to_parquet(path)
        write_manifest(
            parquet_path=path,
            symbol=symbol,
            provider="tushare",
            start=start,
            end=end,
            rows=len(frame),
            extra={
                "adjust": "qfq",
                "adjustment_reference_date": reference_date,
                "adjustment_reference_factor": reference_factor,
                "source_volume_unit": "lot",
                "volume_unit": "share",
                "tushare_version": str(getattr(module, "__version__", "unknown")),
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
        return {symbol: self.download(symbol, start, end, dest_dir) for symbol in symbols}


class YFinanceDownloader:
    """Download market data via yfinance. Covers US and global equities."""

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        yfinance = globals().get("yfinance") or importlib.import_module("yfinance")

        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        ticker = yfinance.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=self.auto_adjust)
        if df.empty:
            msg = f"No data returned for '{symbol}' ({start} to {end})."
            raise DownloadError(msg)

        df = _normalize_df(df)
        path = data_dir / f"{symbol}.parquet"
        df.to_parquet(path)
        write_manifest(
            parquet_path=path,
            symbol=symbol,
            provider="yfinance",
            start=start,
            end=end,
            rows=len(df),
            extra={"auto_adjust": self.auto_adjust},
        )
        return path

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return {s: self.download(s, start, end, dest_dir) for s in symbols}


class AkShareDownloader:
    """Download A-share market data via akshare."""

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        akshare = globals().get("akshare") or importlib.import_module("akshare")

        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        df = akshare.stock_zh_a_hist(
            symbol=symbol,
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df.empty:
            msg = f"No data returned for '{symbol}' ({start} to {end})."
            raise DownloadError(msg)

        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index = df.index.tz_localize("Asia/Shanghai")
        df = df[["open", "high", "low", "close", "volume"]]
        df["volume"] = df["volume"].astype("int64")

        path = data_dir / f"{symbol}.parquet"
        df.to_parquet(path)
        write_manifest(
            parquet_path=path,
            symbol=symbol,
            provider="akshare",
            start=start,
            end=end,
            rows=len(df),
            extra={"adjust": "qfq"},
        )
        return path

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return {s: self.download(s, start, end, dest_dir) for s in symbols}
