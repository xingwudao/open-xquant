"""ROC timing signal with BUY / SELL / HOLD states."""

from __future__ import annotations

import pandas as pd


class ROCTiming:
    """Emit tri-state timing signals from rate-of-change thresholds."""

    name = "ROCTiming"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        lookback: int = 120,
        threshold_mode: str = "fixed",
        buy_threshold: float = -0.10,
        sell_threshold: float = 0.20,
        q_window: int = 60,
        q_bottom: float = 0.05,
        q_top: float = 0.95,
        stop_loss_pct: float = 0.0,
    ) -> pd.Series:
        prices = mktdata[column]
        roc = prices.pct_change(lookback)
        signal = pd.Series(-1.0, index=mktdata.index, dtype=float)

        if threshold_mode == "fixed":
            buy = pd.Series(buy_threshold, index=mktdata.index, dtype=float)
            sell = pd.Series(sell_threshold, index=mktdata.index, dtype=float)
        elif threshold_mode == "rolling":
            min_periods = max(1, int(q_window * 0.8))
            buy = roc.rolling(q_window, min_periods=min_periods).quantile(q_bottom)
            sell = roc.rolling(q_window, min_periods=min_periods).quantile(q_top)
        else:
            raise ValueError("threshold_mode must be 'fixed' or 'rolling'")

        signal = signal.mask(roc < buy, 1.0)
        signal = signal.mask(roc > sell, 0.0)
        if stop_loss_pct > 0:
            peak = prices.rolling(max(lookback, 1), min_periods=1).max()
            drawdown = prices / peak - 1.0
            signal = signal.mask(drawdown <= -abs(stop_loss_pct), 0.0)
        signal = signal.mask(roc.isna())
        return signal
