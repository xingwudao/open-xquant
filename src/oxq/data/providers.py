from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    """Market data interface: the only entry point for strategy to access market data."""

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...

    def get_latest(self, symbol: str) -> pd.Series: ...
