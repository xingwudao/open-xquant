"""EQuantMarketDataProvider — MarketDataProvider backed by eFactorCraft + edatatools.

Provides OHLCV data through the oxq ``MarketDataProvider`` protocol,
fetching from eFactorCraft's ``get_data`` or reading from a local
Parquet cache directory (compatible with LocalMarketDataProvider format).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from oxq.data.providers import MarketDataProvider

logger = logging.getLogger(__name__)

# Default cache directory: $OXQ_DATA_DIR/market or ~/.oxq/data/market
_DEFAULT_DATA_DIR = Path.home() / ".oxq" / "data" / "market"


class EQuantMarketDataProvider:
    """Market data provider that uses eFactorCraft + edatatools.

    Supports two modes:
    1. **Local cache**: reads Parquet files from ``data_dir`` (fast, deterministic).
    2. **Live fetch**: downloads data via ``efactorcraft.get_data`` and caches it.

    Parameters
    ----------
    data_dir : Path or str, optional
        Directory containing per-symbol ``{SYMBOL}.parquet`` files.
    source : str
        Data source for eFactorCraft: "yahoo", "akshare", "tushare", "baostock".
    timezone : str
        Timezone for the DateTimeIndex (e.g., "Asia/Shanghai" for A-shares,
        "US/Eastern" for US equities).
    currency : str
        ISO 4217 currency code (e.g., "CNY", "USD").
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        source: str = "yahoo",
        timezone: str = "US/Eastern",
        currency: str = "USD",
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._source = source
        self._timezone = timezone
        self._currency = currency

    # ------------------------------------------------------------------
    # MarketDataProvider protocol
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Return OHLCV bars for *symbol* between *start* and *end*.

        Prefers local Parquet cache; falls back to downloading via
        eFactorCraft and caching for future use.
        """
        parquet_path = self._data_dir / f"{symbol}.parquet"

        if parquet_path.exists():
            return self._read_cached(parquet_path, start, end)

        logger.info("Downloading %s via eFactorCraft (%s)...", symbol, self._source)
        return self._download_and_cache(symbol, start, end)

    def get_latest(self, symbol: str) -> pd.Series:
        """Return the most recent bar for *symbol*."""
        parquet_path = self._data_dir / f"{symbol}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No cached data for {symbol}. Call get_bars first."
            )
        df = pd.read_parquet(parquet_path)
        df = self._localize(df)
        return df.iloc[-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_cached(
        self, parquet_path: Path, start: str, end: str,
    ) -> pd.DataFrame:
        """Read and slice a cached Parquet file."""
        df = pd.read_parquet(parquet_path)
        df = self._localize(df)

        # Convert string dates to the local tz for slicing
        tz = df.index.tz
        start_ts = pd.Timestamp(start, tz=tz)
        end_ts = pd.Timestamp(end, tz=tz)
        df = df[start_ts:end_ts]

        df.attrs["currency"] = self._currency
        df.attrs["timezone"] = str(tz) if tz else self._timezone
        return df

    def _download_and_cache(
        self, symbol: str, start: str, end: str,
    ) -> pd.DataFrame:
        """Download via eFactorCraft, cache as Parquet, and return."""
        import efactorcraft

        # Build a mini universe DataFrame for eFactorCraft
        universe = pd.DataFrame({"code": [symbol], "name": [symbol]})
        panel = efactorcraft.get_data(
            universe, start_date=start, end_date=end, source=self._source,
            progress=False,
        )

        if panel.empty:
            raise ValueError(f"No data returned for {symbol} via {self._source}")

        # Convert to per-symbol Parquet format compatible with
        # LocalMarketDataProvider
        keep_cols = ["open", "high", "low", "close", "volume", "adjusted"]
        available = [c for c in keep_cols if c in panel.columns]
        df = panel[["date"] + available].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Localize timezone
        df = self._localize(df)

        # Cache
        self._data_dir.mkdir(parents=True, exist_ok=True)
        # Strip tz for Parquet storage (consistent with LocalMarketDataProvider)
        df_utc = df.copy()
        if df_utc.index.tz is not None:
            df_utc.index = df_utc.index.tz_convert("UTC")
        df_utc.to_parquet(self._data_dir / f"{symbol}.parquet")

        # Slice to requested range
        tz = df.index.tz
        start_ts = pd.Timestamp(start, tz=tz)
        end_ts = pd.Timestamp(end, tz=tz)
        df = df[start_ts:end_ts]

        df.attrs["currency"] = self._currency
        df.attrs["timezone"] = str(tz) if tz else self._timezone
        return df

    def _localize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a tz-aware DateTimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        if df.index.tz is None:
            df.index = df.index.tz_localize(self._timezone)
        return df
