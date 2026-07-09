"""Relative Price Strength cross-sectional percentile rank."""

from __future__ import annotations

import pandas as pd


class RPS:
    """Cross-sectional rank of N-day returns.

    The runtime computes this indicator across all symbols at the same bar.
    A single-symbol ``compute`` method is still provided so the class satisfies
    the Indicator protocol, but it returns NaN because RPS is not a time-series
    rank within one symbol.
    """

    name = "RPS"
    formula = r"RPS_{i,t,N}=rank_{cross}(P_{i,t}/P_{i,t-N}-1)"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        period: int = 60,
        scale: float = 100.0,
        min_symbols: int = 1,
    ) -> pd.Series:
        del column, period, scale, min_symbols
        return pd.Series(pd.NA, index=mktdata.index, dtype="float64")

    def compute_cross_section(
        self,
        mktdata: dict[str, pd.DataFrame],
        column: str = "close",
        period: int = 60,
        scale: float = 100.0,
        min_symbols: int = 1,
    ) -> dict[str, pd.Series]:
        """Return same-date percentile ranks of simple N-day returns."""
        returns = {
            symbol: frame[column].astype(float) / frame[column].astype(float).shift(period) - 1.0
            for symbol, frame in mktdata.items()
        }
        matrix = pd.DataFrame(returns)
        ranks = matrix.rank(axis=1, pct=True, method="average") * float(scale)
        if min_symbols > 1:
            counts = matrix.notna().sum(axis=1)
            ranks = ranks.where(counts >= int(min_symbols))
        return {
            symbol: ranks[symbol].reindex(frame.index)
            for symbol, frame in mktdata.items()
        }
