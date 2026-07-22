"""SMA — Simple Moving Average (eQuant-backed)."""

from __future__ import annotations

import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class SMA:
    """Simple Moving Average.

    Computes the rolling mean of a specified column over a given period.
    """

    name = "SMA"
    formula = r"SMA_t = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        """Return the SMA series (first ``period - 1`` values will be NaN)."""
        import ettr
        panel = to_panel(mktdata)
        result = ettr.sma(panel, close_col=column, n=period, append=True)
        return from_panel(result, f"SMA_{period}", mktdata.index)
