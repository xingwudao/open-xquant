"""Backtest result and performance analytics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oxq.core.types import Fill, Portfolio


@dataclass
class BacktestResult:
    """Container for backtest output with basic performance metrics."""

    portfolio: Portfolio
    trades: list[Fill]
    equity_curve: list[tuple[object, float]]  # [(date, value), ...]
    mktdata: dict[str, pd.DataFrame] = field(repr=False)

    # -- Metrics --------------------------------------------------------------

    def total_return(self) -> float:
        """Total return as a fraction (e.g. 0.15 = 15%)."""
        if len(self.equity_curve) < 2:
            return 0.0
        first = self.equity_curve[0][1]
        last = self.equity_curve[-1][1]
        if first == 0.0:
            return 0.0
        return (last - first) / first

    def sharpe_ratio(self, trading_days: int = 252) -> float:
        """Annualized Sharpe ratio (assumes risk-free rate = 0)."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        returns = np.diff(values) / values[:-1]
        if len(returns) == 0 or np.std(returns) == 0.0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(trading_days))

    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.10 = -10%)."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return float(np.min(drawdown))
