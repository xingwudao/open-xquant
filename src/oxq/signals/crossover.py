"""Crossover signal — detects when a fast line crosses above a slow line."""

from __future__ import annotations

import pandas as pd


class Crossover:
    """Detect upward crossover between two indicator columns.

    Produces ``True`` on bars where *fast* crosses above *slow*
    (i.e. previous bar fast <= slow, current bar fast > slow).
    """

    name = "Crossover"

    def compute(
        self,
        mktdata: dict[str, pd.DataFrame],
        fast: str = "",
        slow: str = "",
    ) -> dict[str, pd.Series]:
        """Return cross-up boolean series for every symbol in *mktdata*."""
        result: dict[str, pd.Series] = {}
        for symbol, df in mktdata.items():
            f = df[fast]
            s = df[slow]
            result[symbol] = (f.shift(1) <= s.shift(1)) & (f > s)
        return result
