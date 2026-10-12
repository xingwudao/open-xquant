"""TongDaXin-style SMA(X, N, M): exponential smoothing with alpha = M / N."""

from __future__ import annotations

import pandas as pd


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class TdxSMA:
    """TongDaXin SMA(X, N, M).

    Recursion: ``Y_t = (M * X_t + (N - M) * Y_{t-1}) / N`` seeded with the first
    valid input. Interior NaN inputs are skipped by the recursion and the
    output carries the last state forward, so a flat window that makes RSV
    undefined does not reset K/D.
    """

    name = "TdxSMA"
    formula = r"Y_t = \frac{M X_t + (N - M) Y_{t-1}}{N}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", n: int = 3, m: int = 1,
    ) -> pd.Series:
        """Return the TongDaXin SMA series of ``column``."""
        if not (_is_strict_int(n) and _is_strict_int(m) and 0 < m <= n):
            raise ValueError("TdxSMA requires integers 0 < m <= n")
        return mktdata[column].ewm(alpha=m / n, adjust=False, ignore_na=True).mean()
