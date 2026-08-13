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
_TUSHARE_RESPONSE_DATE_RE = re.compile(r"^[0-9]{8}$")
_TUSHARE_DAILY_COLUMNS = {
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
}
_TUSHARE_FACTOR_COLUMNS = {"ts_code", "trade_date", "adj_factor"}
_TUSHARE_PRICE_COLUMNS = ["open", "high", "low", "close"]
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


def _valid_price_envelope(values: pd.DataFrame) -> bool:
    return bool(
        (
            values["high"].ge(values[["open", "close"]].max(axis="columns"))
            & values["low"].le(values[["open", "close"]].min(axis="columns"))
            & values["high"].ge(values["low"])
        ).all()
    )


def _validate_tushare_daily_frame(
    frame: object, *, symbol: str, start: str, end: str
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DownloadError("Tushare daily response must be a pandas DataFrame.")
    if frame.empty:
        raise DownloadError(
            f"No Tushare daily data returned for '{symbol}' ({start} to {end})."
        )
    missing = sorted(_TUSHARE_DAILY_COLUMNS.difference(frame.columns))
    if missing:
        raise DownloadError(
            f"Tushare daily response is missing required fields: {', '.join(missing)}."
        )
    if not frame["ts_code"].eq(symbol).fillna(False).all():
        raise DownloadError("Tushare daily response contains an unexpected ts_code.")

    validated = frame.copy()
    dates = validated["trade_date"].astype(str)
    if (
        not dates.str.fullmatch(_TUSHARE_RESPONSE_DATE_RE).all()
        or pd.to_datetime(dates, format="%Y%m%d", errors="coerce").isna().any()
    ):
        raise DownloadError("Tushare daily response contains an invalid trade_date.")
    if dates.duplicated().any():
        raise DownloadError("Tushare daily response contains duplicate trade_date values.")
    if not dates.between(start, end).all():
        raise DownloadError("Tushare daily response is outside the requested date range.")

    prices = validated.loc[:, _TUSHARE_PRICE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or not prices.gt(0).all().all():
        raise DownloadError("Tushare prices must be positive and finite.")
    if not _valid_price_envelope(prices):
        raise DownloadError("Tushare unadjusted price envelope is invalid.")

    volume = pd.to_numeric(validated["vol"], errors="coerce")
    if not np.isfinite(volume.to_numpy(dtype=float)).all():
        raise DownloadError("Tushare volume must be finite.")
    if volume.lt(0).any():
        raise DownloadError("Tushare volume must be non-negative.")

    validated["trade_date"] = dates
    validated[_TUSHARE_PRICE_COLUMNS] = prices
    validated["vol"] = volume
    return validated


def _validate_tushare_factor_frame(
    frame: object, *, symbol: str, start: str
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DownloadError("Tushare adjustment response must be a pandas DataFrame.")
    if frame.empty:
        raise DownloadError(
            f"No Tushare adjustment factors returned for '{symbol}'."
        )
    missing = sorted(_TUSHARE_FACTOR_COLUMNS.difference(frame.columns))
    if missing:
        raise DownloadError(
            "Tushare adjustment response is missing required fields: "
            f"{', '.join(missing)}."
        )
    if not frame["ts_code"].eq(symbol).fillna(False).all():
        raise DownloadError("Tushare adjustment response contains an unexpected ts_code.")

    validated = frame.copy()
    dates = validated["trade_date"].astype(str)
    if (
        not dates.str.fullmatch(_TUSHARE_RESPONSE_DATE_RE).all()
        or pd.to_datetime(dates, format="%Y%m%d", errors="coerce").isna().any()
    ):
        raise DownloadError("Tushare adjustment response contains an invalid trade_date.")
    if dates.duplicated().any():
        raise DownloadError(
            "Tushare adjustment response contains duplicate trade_date values."
        )
    if (dates < start).any():
        raise DownloadError("Tushare adjustment response precedes the requested date range.")

    adjustment_factors = pd.to_numeric(validated["adj_factor"], errors="coerce")
    if (
        not np.isfinite(adjustment_factors.to_numpy(dtype=float)).all()
        or adjustment_factors.le(0).any()
    ):
        raise DownloadError("Tushare adjustment factors must be positive and finite.")

    validated["trade_date"] = dates
    validated["adj_factor"] = adjustment_factors
    return validated.reset_index(drop=True)


def _normalize_tushare_frames(
    daily: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    symbol: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, str, float]:
    daily = _validate_tushare_daily_frame(
        daily, symbol=symbol, start=start, end=end
    )
    factors = _validate_tushare_factor_frame(factors, symbol=symbol, start=start)

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

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        adjustment = merged["adj_factor"] / reference_factor
        values = merged.loc[:, _TUSHARE_PRICE_COLUMNS].mul(
            adjustment, axis="index"
        )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise DownloadError(
            "Tushare qfq arithmetic overflow produced non-finite adjusted prices."
        )
    if not values.gt(0).all().all() or not _valid_price_envelope(values):
        raise DownloadError("Tushare adjusted price envelope is invalid.")

    scaled_volume = merged["vol"].to_numpy(dtype=float) * 100
    rounded_volume = np.rint(scaled_volume)
    if not np.all(np.isfinite(scaled_volume)):
        raise DownloadError("Tushare volume must fit within int64 share limits.")
    if not np.all(np.abs(scaled_volume - rounded_volume) <= 1e-6):
        raise DownloadError("Tushare volume must convert from lots to whole shares.")
    rounded_volume_ints = [int(value) for value in rounded_volume]
    int64_limits = np.iinfo(np.int64)
    if any(
        value < int64_limits.min or value > int64_limits.max
        for value in rounded_volume_ints
    ):
        raise DownloadError("Tushare volume must fit within int64 share limits.")

    frame = values.assign(volume=np.asarray(rounded_volume_ints, dtype="int64"))
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
        import_failed = False
        try:
            module = importlib.import_module("tushare")
        except ModuleNotFoundError:
            import_failed = True
        if import_failed:
            raise DownloadError(
                "Tushare is not installed; run `uv sync --extra tushare`."
            )
        client_error: str | None = None
        try:
            client = module.pro_api(token)
        except Exception as exc:
            client_error = str(exc).replace(token, "***")
        if client_error is not None:
            raise DownloadError(
                f"Tushare client initialization failed: {client_error}"
            )
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
        request_error: str | None = None
        try:
            daily = client.daily(
                ts_code=symbol, start_date=normalized_start, end_date=normalized_end
            )
            factors = client.adj_factor(
                ts_code=symbol, start_date=normalized_start, end_date=normalized_end
            )
        except Exception as exc:
            request_error = str(exc).replace(self._resolved_token or "", "***")
        if request_error is not None:
            raise DownloadError(
                f"Tushare request failed for '{symbol}': {request_error}"
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
