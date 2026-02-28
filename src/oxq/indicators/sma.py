"""Simple Moving Average indicator."""

from __future__ import annotations

import pandas as pd


class SMA:
    """Simple Moving Average.

    Computes the rolling mean of a specified column over a given period.
    """

    name = "SMA"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        """Return the SMA series (first ``period - 1`` values will be NaN)."""
        return mktdata[column].rolling(period).mean()
