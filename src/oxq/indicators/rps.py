"""RPS — Relative Price Strength (eQuant-backed via eclassic.rps).

Cross-sectional percentile rank of N-day returns across all symbols.
"""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class RPS:
    """Cross-sectional rank of N-day returns.

    The runtime computes this indicator across all symbols at the same bar.
    A single-symbol ``compute`` method returns NaN because RPS is not a
    time-series rank within one symbol.
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
        return pd.Series(float("nan"), index=mktdata.index, dtype="float64")

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
        import eclassic

        # Build a long-format panel from all per-symbol DataFrames
        panels = []
        for symbol, frame in mktdata.items():
            p = to_panel(frame, code=symbol)
            panels.append(p)
        panel = pd.concat(panels, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])

        # eclassic.rps computes cross-sectional ranks per date
        result = eclassic.rps(panel, close_col=column, n=period,
                              new_col="rps", append=True)
        col_name = f"rps_{period}"

        # Scale from [0,1] if eclassic returns raw percentile
        out: dict[str, pd.Series] = {}
        for symbol, frame in mktdata.items():
            symbol_mask = result["code"] == symbol
            if symbol_mask.any():
                symbol_data = result.loc[symbol_mask, ["date", col_name]]
                s = from_panel(symbol_data.rename(columns={col_name: col_name}),
                               col_name, frame.index, code=symbol)
                if abs(float(scale) - 1.0) > 1e-12:
                    s = s * float(scale) / 100.0 if abs(float(scale) - 100.0) < 1e-12 else s * float(scale)
            else:
                s = pd.Series(float("nan"), index=frame.index, dtype="float64")

            if min_symbols > 1:
                # Count non-NaN values per date
                val_counts = result[result["code"] != symbol].groupby("date")[col_name].count()
                val_counts = val_counts.reindex(pd.to_datetime(frame.index.get_level_values(0)
                    if isinstance(frame.index, pd.MultiIndex) else frame.index))
                s = s.where(val_counts.values >= int(min_symbols) if len(val_counts) == len(s)
                           else val_counts.reindex(s.index, fill_value=0) >= int(min_symbols))

            out[symbol] = s.reindex(frame.index)
        return out


def _validate_rps_params(*, period: int, scale: float, min_symbols: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")
    if isinstance(min_symbols, bool) or not isinstance(min_symbols, int) or min_symbols <= 0:
        raise ValueError("min_symbols must be a positive integer")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not math.isfinite(float(scale))
        or scale <= 0
    ):
        raise ValueError("scale must be a positive finite real number")
