"""Built-in technical indicators — eQuant-Py backed.

Wraps eTTR functions to satisfy the oxq ``Indicator`` Protocol
(``compute(self, mktdata: pd.DataFrame, **params) -> pd.Series``).

Multi-output eQuant indicators (MACD, Bollinger, etc.) are split into
independent oxq classes, each extracting a single output column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oxq.adapters.equant import to_panel, from_panel

# ── Trend ─────────────────────────────────────────────────────────────────


class EMA:
    """Exponential Moving Average (eQuant-backed)."""

    name = "EMA"
    formula = r"EMA_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA_{t-1}, \quad \alpha = \frac{2}{N+1}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.ema(panel, close_col=column, n=period, append=True)
        return from_panel(result, f"EMA_{period}", mktdata.index)


class WMA:
    """Weighted Moving Average (eQuant-backed)."""

    name = "WMA"
    formula = r"WMA_t = \frac{\sum_{i=0}^{N-1} (N-i) \cdot P_{t-i}}{\sum_{i=1}^{N} i}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.wma(panel, close_col=column, n=period, append=True)
        return from_panel(result, f"WMA_{period}", mktdata.index)


class DEMA:
    """Double Exponential Moving Average (eQuant-backed)."""

    name = "DEMA"
    formula = r"DEMA_t = 2 \cdot EMA_t - EMA(EMA_t)"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.dema(panel, close_col=column, n=period, append=True)
        return from_panel(result, f"DEMA_{period}", mktdata.index)


class TEMA:
    """Triple Exponential Moving Average.

    eTTR does not provide a standalone TEMA function, so we compute it from
    three stacked EMA calls on a single panel.
    """

    name = "TEMA"
    formula = r"TEMA_t = 3 \cdot EMA_t - 3 \cdot EMA_2_t + EMA_3_t"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        # EMA1: EMA of close
        panel = ettr.ema(panel, close_col=column, n=period, append=True,
                         new_col="_TEMA1")
        col1 = f"_TEMA1_{period}"
        # EMA2: EMA of EMA1
        panel = ettr.ema(panel, close_col=col1, n=period, append=True,
                         new_col="_TEMA2")
        col2 = f"_TEMA2_{period}"
        # EMA3: EMA of EMA2
        panel = ettr.ema(panel, close_col=col2, n=period, append=True,
                         new_col="_TEMA3")
        col3 = f"_TEMA3_{period}"

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        panel["_TEMA"] = 3.0 * panel[col1] - 3.0 * panel[col2] + panel[col3]
        return from_panel(panel, "_TEMA", mktdata.index)


# ── Momentum ──────────────────────────────────────────────────────────────


class RSI:
    """Relative Strength Index (eQuant-backed, Wilder smoothing default)."""

    name = "RSI"
    formula = r"RSI = 100 - \frac{100}{1 + \frac{AvgGain}{AvgLoss}}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 14,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.rsi(panel, close_col=column, n=period, wilder=True, append=True)
        return from_panel(result, f"RSI_{period}", mktdata.index)


class MACDLine:
    """MACD Line = EMA(fast) - EMA(slow) (eQuant-backed)."""

    name = "MACDLine"
    formula = r"MACD = EMA_{fast} - EMA_{slow}"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        fast_period: int = 12,
        slow_period: int = 26,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.macd(panel, close_col=column, n_fast=fast_period,
                           n_slow=slow_period, new_col="MACD", append=True)
        return from_panel(result, "MACD", mktdata.index)


class MACDSignal:
    """MACD Signal Line = EMA of MACD Line.

    Requires: register MACDLine before this indicator.
    """

    name = "MACDSignal"
    formula = r"Signal = EMA_9(MACD)"
    depends_on = ("macd",)

    def compute(
        self,
        mktdata: pd.DataFrame,
        macd_col: str = "macd",
        signal_period: int = 9,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        # Copy the macd column into a "close-like" column for ettr
        panel["_macd_for_signal"] = panel["close"]  # placeholder; we use macd_col
        col_name = panel.columns[panel.columns.get_loc("close")]
        # Use ettr.ema on the MACD line column
        result = ettr.ema(
            panel, close_col=macd_col if macd_col in panel.columns else col_name,
            n=signal_period, wilder=False, new_col="_MACDSig", append=True,
        )
        return from_panel(result, f"_MACDSig_{signal_period}", mktdata.index)


class MACDHistogram:
    """MACD Histogram = MACD Line - Signal Line.

    Requires: register MACDLine and MACDSignal first.
    """

    name = "MACDHistogram"
    formula = r"Histogram = MACD - Signal"
    depends_on = ("macd", "macd_signal")

    def compute(
        self,
        mktdata: pd.DataFrame,
        macd_col: str = "macd",
        signal_col: str = "macd_signal",
    ) -> pd.Series:
        return mktdata[macd_col] - mktdata[signal_col]


class ROC:
    """Rate of Change (eQuant-backed)."""

    name = "ROC"
    formula = r"ROC_t = \frac{P_t - P_{t-N}}{P_{t-N}} \times 100"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 10,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        # eTTR's ROC returns ROC_N; with type="continuous" it matches pct_change
        result = ettr.roc(panel, close_col=column, n=period, type="continuous",
                          append=True)
        return from_panel(result, f"ROC_{period}", mktdata.index)


class PPO:
    """Percentage Price Oscillator.

    eTTR provides ``po_`` which is (EMA_fast - EMA_slow) / EMA_slow * 100,
    matching the standard PPO formula.
    """

    name = "PPO"
    formula = r"PPO = \frac{EMA_{fast} - EMA_{slow}}{EMA_{slow}} \times 100"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        fast_period: int = 12,
        slow_period: int = 26,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.po_(panel, close_col=column, n_fast=fast_period,
                          n_slow=slow_period, append=True)
        return from_panel(result, "PO", mktdata.index)


class CCI:
    """Commodity Channel Index (eQuant-backed)."""

    name = "CCI"
    formula = r"CCI = \frac{TP - SMA(TP)}{0.015 \cdot MAD(TP)}, \quad TP = \frac{H+L+C}{3}"

    def compute(
        self, mktdata: pd.DataFrame, period: int = 20,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.cci(panel, n=period, append=True)
        return from_panel(result, f"CCI_{period}", mktdata.index)


# ── Volatility ────────────────────────────────────────────────────────────


class BollingerUpper:
    """Bollinger Band — upper band (eQuant-backed)."""

    name = "BollingerUpper"
    formula = r"Upper = SMA_N + k \cdot \sigma_N"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        period: int = 20,
        offset: float = 2.0,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.bollinger(panel, close_col=column, n=period, sd=offset,
                                append=True)
        return from_panel(result, "BB_upper", mktdata.index)


class BollingerLower:
    """Bollinger Band — lower band (eQuant-backed)."""

    name = "BollingerLower"
    formula = r"Lower = SMA_N - k \cdot \sigma_N"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        period: int = 20,
        offset: float = 2.0,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.bollinger(panel, close_col=column, n=period, sd=offset,
                                append=True)
        return from_panel(result, "BB_lower", mktdata.index)


class ATR:
    """Average True Range (eQuant-backed, Wilder smoothing)."""

    name = "ATR"
    formula = r"TR = \max(H-L, |H-C_{prev}|, |L-C_{prev}|), \quad ATR = Wilder(TR, N)"

    def compute(
        self, mktdata: pd.DataFrame, period: int = 14,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.atr(panel, n=period, wilder=True, append=True)
        return from_panel(result, f"ATR_{period}", mktdata.index)


# ── Volume ────────────────────────────────────────────────────────────────


class OBV:
    """On-Balance Volume (eQuant-backed)."""

    name = "OBV"
    formula = r"OBV_t = OBV_{t-1} + sign(\Delta C_t) \cdot V_t"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close",
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.obv(panel, close_col=column, append=True)
        return from_panel(result, "OBV", mktdata.index)


class VWAP:
    """Volume-Weighted Average Price (eQuant-backed).

    eTTR provides a cumulative VWAP. For a rolling window version,
    we fall back to the manual calculation.
    """

    name = "VWAP"
    formula = r"VWAP = \frac{\sum_{i} TP_i \cdot V_i}{\sum_{i} V_i}"

    def compute(
        self, mktdata: pd.DataFrame, period: int = 20,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        # eTTR provides cumulative VWAP; we also support a rolling version
        if period is not None and period > 0:
            # Roll our own rolling VWAP for per-symbol windowing
            tp = (mktdata["high"] + mktdata["low"] + mktdata["close"]) / 3.0
            tp_vol = tp * mktdata["volume"]
            return tp_vol.rolling(period).sum() / mktdata["volume"].rolling(period).sum()
        result = ettr.vwap(panel, append=True)
        return from_panel(result, "VWAP", mktdata.index)


class MFI:
    """Money Flow Index (eQuant-backed)."""

    name = "MFI"
    formula = r"MFI = 100 - \frac{100}{1 + \frac{PositiveFlow}{NegativeFlow}}"

    def compute(
        self, mktdata: pd.DataFrame, period: int = 14,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.mfi(panel, n=period, append=True)
        return from_panel(result, f"MFI_{period}", mktdata.index)


# ── Trend Strength ────────────────────────────────────────────────────────


class ADX:
    """Average Directional Index (eQuant-backed)."""

    name = "ADX"
    formula = r"ADX = Wilder\left(\frac{|+DI - (-DI)|}{+DI + (-DI)} \times 100, N\right)"

    def compute(
        self, mktdata: pd.DataFrame, period: int = 14,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.adx(panel, n=period, append=True)
        return from_panel(result, f"ADX_{period}", mktdata.index)


class AROON:
    """Aroon Oscillator = Aroon Up - Aroon Down (eQuant-backed)."""

    name = "AROON"
    formula = r"Aroon = \frac{bars\_since\_high}{N} \times 100 - \frac{bars\_since\_low}{N} \times 100"

    def compute(
        self, mktdata: pd.DataFrame, period: int = 25,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.aroon(panel, n=period, append=True)
        return from_panel(result, "Aroon_osc", mktdata.index)


# ── Stochastic ────────────────────────────────────────────────────────────


class StochK:
    """Stochastic %K (eQuant-backed — fast %K from the stoch function)."""

    name = "StochK"
    formula = r"\%K = \frac{C - L_N}{H_N - L_N} \times 100"

    def compute(
        self, mktdata: pd.DataFrame, period: int = 14,
    ) -> pd.Series:
        import ettr
        panel = to_panel(mktdata)
        result = ettr.stoch(panel, n_fast_k=period, append=True)
        return from_panel(result, "Stoch_fastK", mktdata.index)
