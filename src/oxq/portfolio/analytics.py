"""Engine result and performance analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from oxq.core.types import BarSnapshot, Fill, Portfolio

if TYPE_CHECKING:
    from oxq.portfolio.orderbook import ManagedOrder


@dataclass
class RunResult:
    """Container for engine output with basic performance metrics."""

    portfolio: Portfolio
    trades: list[Fill]
    equity_curve: list[tuple[object, float]]  # [(date, value), ...]
    mktdata: dict[str, pd.DataFrame] = field(repr=False)
    benchmark_prices: dict[str, pd.Series] = field(default_factory=dict)
    snapshots: list[BarSnapshot] = field(default_factory=list)
    orders: list[ManagedOrder] = field(default_factory=list)

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

    def annualized_return(self, trading_days: int = 252) -> float:
        """Annualized return (CAGR): (V_final / V_initial) ^ (T / N) - 1."""
        if len(self.equity_curve) < 2:
            return 0.0
        first = self.equity_curve[0][1]
        last = self.equity_curve[-1][1]
        if first <= 0:
            return 0.0
        n = len(self.equity_curve) - 1
        if n <= 0:
            return 0.0
        return float((last / first) ** (trading_days / n) - 1)

    def annualized_volatility(self, trading_days: int = 252) -> float:
        """Annualized volatility: sigma_daily x sqrt(T)."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        log_returns = np.diff(np.log(values))
        if len(log_returns) == 0:
            return 0.0
        daily_vol = float(np.std(log_returns, ddof=1))
        return daily_vol * np.sqrt(trading_days)

    def calmar_ratio(self, trading_days: int = 252) -> float:
        """Calmar ratio: annualized_return / |MDD|."""
        ann_ret = self.annualized_return(trading_days)
        mdd = self.max_drawdown()
        if mdd == 0.0:
            return 0.0
        return float(ann_ret / abs(mdd))

    def sortino_ratio(
        self, risk_free: float = 0.0, trading_days: int = 252,
    ) -> float:
        """Sortino ratio: (annualized return - r_f) / downside deviation."""
        if len(self.equity_curve) < 2:
            return 0.0
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        log_returns = np.diff(np.log(values))
        if len(log_returns) == 0:
            return 0.0
        downside = log_returns[log_returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(trading_days))
        ann_ret = float(np.mean(log_returns) * trading_days)
        return float((ann_ret - risk_free) / downside_dev)

    # -- Series ----------------------------------------------------------------

    def daily_returns(self) -> pd.Series:
        """Daily simple returns as a Series with date index."""
        if len(self.equity_curve) < 2:
            return pd.Series(dtype=float)
        dates = [d for d, _ in self.equity_curve]
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        returns = np.diff(values) / values[:-1]
        return pd.Series(returns, index=dates[1:])

    def monthly_returns(self) -> pd.Series:
        """Monthly simple returns as a Series with month-period index.

        The first month's return is measured from the first data point.
        Subsequent months are measured from the previous month-end value.
        """
        if len(self.equity_curve) < 2:
            return pd.Series(dtype=float)
        dates = [d for d, _ in self.equity_curve]
        values = [v for _, v in self.equity_curve]
        s = pd.Series(values, index=pd.DatetimeIndex(dates))
        # Last value in each month
        month_end = s.groupby(s.index.to_period("M")).last()
        # Prepend first value as baseline so first month gets a return
        first_val = values[0]
        baseline = pd.Series([first_val], index=[s.index[0].to_period("M") - 1])
        month_end = pd.concat([baseline, month_end])
        return month_end.pct_change().dropna()

    def drawdown_series(self) -> pd.Series:
        """Daily drawdown from peak as a Series with date index."""
        if len(self.equity_curve) == 0:
            return pd.Series(dtype=float)
        dates = [d for d, _ in self.equity_curve]
        values = np.array([v for _, v in self.equity_curve], dtype=float)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return pd.Series(drawdown, index=dates)

    # -- Snapshot DataFrames ---------------------------------------------------

    def weights_df(self) -> pd.DataFrame:
        """Target weights (optimizer output) as a DataFrame, date x symbol."""
        if not self.snapshots:
            return pd.DataFrame()
        rows = {s.date: s.target_weights for s in self.snapshots}
        return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()

    def adj_weights_df(self) -> pd.DataFrame:
        """Adjusted weights (after rules) as a DataFrame, date x symbol."""
        if not self.snapshots:
            return pd.DataFrame()
        rows = {s.date: s.adjusted_weights for s in self.snapshots}
        return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()

    def positions_df(self) -> pd.DataFrame:
        """Position shares as a DataFrame, date x symbol."""
        if not self.snapshots:
            return pd.DataFrame()
        rows = {
            s.date: {sym: ps.shares for sym, ps in s.positions.items()}
            for s in self.snapshots
        }
        return pd.DataFrame.from_dict(rows, orient="index").fillna(0).sort_index()
