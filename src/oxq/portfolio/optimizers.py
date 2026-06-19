"""Portfolio optimizer implementations."""

from __future__ import annotations

import pandas as pd


class EqualWeightOptimizer:
    """Assigns equal weight to all symbols."""

    name: str = "EqualWeight"

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        if not signals:
            return {"CASH": 1.0}
        weight = 1.0 / len(signals)
        return {symbol: weight for symbol in signals}


class RiskParityOptimizer:
    """Weights inversely proportional to volatility."""

    name: str = "RiskParity"

    def __init__(self, volatility_col: str) -> None:
        self.volatility_col = volatility_col

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        inv_vols: dict[str, float] = {}
        for symbol, df in indicators.items():
            if self.volatility_col not in df.columns:
                continue
            vol = float(df[self.volatility_col].iloc[-1])
            if pd.isna(vol) or vol <= 0:
                continue
            inv_vols[symbol] = 1.0 / vol

        if not inv_vols:
            return {"CASH": 1.0}

        total = sum(inv_vols.values())
        return {symbol: iv / total for symbol, iv in inv_vols.items()}


class KellyOptimizer:
    """Kelly criterion-based position sizing."""

    name: str = "Kelly"

    def __init__(
        self,
        win_rate_col: str,
        avg_win_col: str,
        avg_loss_col: str,
        fraction: float = 1.0,
    ) -> None:
        self.win_rate_col = win_rate_col
        self.avg_win_col = avg_win_col
        self.avg_loss_col = avg_loss_col
        self.fraction = fraction

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}

        for symbol, df in indicators.items():
            required = {self.win_rate_col, self.avg_win_col, self.avg_loss_col}
            if not required.issubset(df.columns):
                continue
            win_rate = float(df[self.win_rate_col].iloc[-1])
            avg_win = float(df[self.avg_win_col].iloc[-1])
            avg_loss = float(df[self.avg_loss_col].iloc[-1])

            if avg_loss <= 0:
                continue

            payoff_ratio = avg_win / avg_loss
            kelly_pct = win_rate - (1 - win_rate) / payoff_ratio
            kelly_pct = max(kelly_pct, 0.0) * self.fraction

            if kelly_pct > 0:
                weights[symbol] = kelly_pct

        if not weights:
            return {"CASH": 1.0}

        total = sum(weights.values())
        if total > 1.0:
            weights = {s: w / total for s, w in weights.items()}
        else:
            weights["CASH"] = 1.0 - total

        return weights


class TopNRankingOptimizer:
    """Rank symbols by score, select top N, normalize to target weights."""

    name: str = "TopNRanking"

    def __init__(
        self,
        score_col: str,
        n: int = 5,
        filter_negative: bool = True,
        max_weight: float = 1.0,
    ) -> None:
        self.score_col = score_col
        self.n = n
        self.filter_negative = filter_negative
        self.max_weight = max_weight

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for symbol, df in indicators.items():
            if self.score_col not in df.columns:
                continue
            val = float(df[self.score_col].iloc[-1])
            if pd.isna(val):
                continue
            if self.filter_negative and val <= 0:
                continue
            scores[symbol] = val

        if not scores:
            return {"CASH": 1.0}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[: self.n]

        total = sum(v for _, v in top)
        if total <= 0:
            return {"CASH": 1.0}

        weights: dict[str, float] = {}
        cash = 0.0
        for s, v in top:
            w = v / total
            if w > self.max_weight:
                cash += w - self.max_weight
                w = self.max_weight
            weights[s] = w

        if cash > 0:
            weights["CASH"] = cash

        return weights if weights else {"CASH": 1.0}


class PctEquityOptimizer:
    """Allocates a fixed percentage of equity to each signaled symbol."""

    name: str = "PctEquity"

    def __init__(self, pct: float = 0.10) -> None:
        self.pct = pct

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        if not signals:
            return {"CASH": 1.0}

        total_pct = self.pct * len(signals)
        if total_pct > 1.0:
            weight = 1.0 / len(signals)
            return {symbol: weight for symbol in signals}

        weights: dict[str, float] = {symbol: self.pct for symbol in signals}
        weights["CASH"] = 1.0 - total_pct
        return weights


class SignalPositionOptimizer:
    """Map BUY / SELL / HOLD signals to target weights."""

    name: str = "SignalPosition"

    def __init__(self, signal_col: str, weight: float = 1.0) -> None:
        self.signal_col = signal_col
        self.weight = weight
        self._invested: dict[str, bool] = {}

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for symbol, df in indicators.items():
            if self.signal_col not in df.columns:
                continue
            value = float(df[self.signal_col].iloc[-1])
            if pd.isna(value):
                value = -1.0
            if value > 0:
                self._invested[symbol] = True
            elif value == 0:
                self._invested[symbol] = False
            if self._invested.get(symbol, False):
                weights[symbol] = self.weight

        if not weights:
            return {"CASH": 1.0}
        total = sum(weights.values())
        if total < 1.0:
            weights["CASH"] = 1.0 - total
        return weights
