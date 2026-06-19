from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from oxq.core.errors import SymbolNotFoundError
from oxq.data.loaders import resolve_data_dir
from oxq.market_calendar import normalize_exchange_calendar

logger = logging.getLogger(__name__)


class LocalMarketDataProvider:
    """Read market data from local Parquet files. Implements MarketDataProvider Protocol."""

    def __init__(self, data_dir: Path | None = None, currency: str | None = None, calendar: str | None = None) -> None:
        self._data_dir = resolve_data_dir(data_dir)
        self._currency = currency
        self._calendar = normalize_exchange_calendar(calendar) if calendar else None

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self._symbol_path(symbol)
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        if hasattr(df.index, "tz") and df.index.tz is None:
            logger.warning(
                "Parquet file for '%s' has no timezone on index. "
                "Assuming UTC. Re-download data to fix.",
                symbol,
            )
            df.index = df.index.tz_localize("UTC")
        df = _normalize_time_index(df, symbol)
        if self._currency:
            df.attrs["currency"] = self._currency
        sliced = df.loc[start:end]  # type: ignore[misc]  # pandas string-based label slicing
        return self._filter_sessions(sliced, start, end)

    def get_latest(self, symbol: str) -> pd.Series:
        path = self._symbol_path(symbol)
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        df = _normalize_time_index(df, symbol)
        if self._currency:
            df.attrs["currency"] = self._currency
        return df.iloc[-1]

    def _filter_sessions(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        if not self._calendar or df.empty:
            return df
        import exchange_calendars as xcals

        cal = xcals.get_calendar(self._calendar)
        sessions = cal.sessions_in_range(pd.Timestamp(start).date(), pd.Timestamp(end).date())
        session_dates = {pd.Timestamp(session).date() for session in sessions}
        mask = [pd.Timestamp(index_value).date() in session_dates for index_value in df.index]
        return df.loc[mask]

    def _symbol_path(self, symbol: str) -> Path:
        if _unsafe_symbol(symbol):
            raise ValueError(f"Unsafe symbol path: {symbol}")
        data_path = self._data_dir.resolve()
        path = (data_path / f"{symbol}.parquet").resolve()
        if not path.is_relative_to(data_path):
            raise ValueError(f"Unsafe symbol path: {symbol}")
        return path


def _unsafe_symbol(symbol: str) -> bool:
    if not symbol or "/" in symbol or "\\" in symbol:
        return True
    path = Path(symbol)
    return path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)


def _normalize_time_index(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.index.has_duplicates:
        raise ValueError(f"Parquet file for '{symbol}' has duplicate index values")
    return df.sort_index()
