"""ROC timing signal with BUY, SELL, HOLD output."""

from __future__ import annotations

import pandas as pd


class ROCTiming:
    """ROC reversal timing signal.

    Returns uppercase categorical trading-intent labels:
    `BUY` when ROC is below the bottom threshold, `SELL` when ROC is above
    the top threshold, and `HOLD` otherwise. In `rolling_quantile` mode,
    thresholds are computed from prior rows only so the current bar does not
    influence its own signal.
    """

    name = "ROCTiming"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "",
        mode: str = "fixed",
        bottom: float = -5.0,
        top: float = 5.0,
        q_window: int = 60,
        q_bottom: float = 0.05,
        q_top: float = 0.95,
    ) -> pd.Series:
        if column not in mktdata.columns:
            raise KeyError(column)

        values = mktdata[column]
        if mode == "fixed":
            if bottom >= top:
                raise ValueError("bottom must be less than top")
            bottom_threshold = pd.Series(bottom, index=mktdata.index)
            top_threshold = pd.Series(top, index=mktdata.index)
        elif mode == "rolling_quantile":
            if q_window <= 0:
                raise ValueError("q_window must be positive")
            if not 0 <= q_bottom <= 1 or not 0 <= q_top <= 1:
                raise ValueError("q_bottom and q_top must be in [0, 1]")
            if q_bottom >= q_top:
                raise ValueError("q_bottom must be less than q_top")
            shifted = values.shift(1)
            bottom_threshold = shifted.rolling(q_window, min_periods=q_window).quantile(q_bottom)
            top_threshold = shifted.rolling(q_window, min_periods=q_window).quantile(q_top)
        else:
            raise ValueError("mode must be 'fixed' or 'rolling_quantile'")

        result = pd.Series("HOLD", index=mktdata.index, dtype="object")
        separated = bottom_threshold < top_threshold
        result = result.mask(separated & (values <= bottom_threshold), "BUY")
        result = result.mask(separated & (values >= top_threshold), "SELL")
        return result
