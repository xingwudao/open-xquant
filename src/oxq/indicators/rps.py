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
        _validate_rps_params(period=period, scale=scale, min_symbols=min_symbols)
        del column
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
        _validate_rps_params(period=period, scale=scale, min_symbols=min_symbols)
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


def _validate_rps_params(*, period: int, scale: float, min_symbols: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")
    if isinstance(min_symbols, bool) or not isinstance(min_symbols, int) or min_symbols <= 0:
        raise ValueError("min_symbols must be a positive integer")
    if float(scale) <= 0:
        raise ValueError("scale must be positive")
