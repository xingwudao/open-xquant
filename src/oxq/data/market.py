from __future__ import annotations

from pathlib import Path

import pandas as pd

from oxq.core.errors import SymbolNotFoundError
from oxq.data.loaders import resolve_data_dir


class LocalMarketDataProvider:
    """Read market data from local Parquet files. Implements MarketDataProvider Protocol."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = resolve_data_dir(data_dir)

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self._data_dir / f"{symbol}.parquet"
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        return df.loc[start:end]  # type: ignore[misc]  # pandas string-based label slicing

    def get_latest(self, symbol: str) -> pd.Series:
        path = self._data_dir / f"{symbol}.parquet"
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        return df.iloc[-1]
