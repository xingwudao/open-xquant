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


class SignalToPositionOptimizer:
    """Map BUY/SELL/HOLD signal values to target weights with HOLD preserving state."""

    name: str = "SignalToPosition"

    def __init__(
        self,
        signal: str,
        buy_weight: float = 1.0,
        sell_weight: float = 0.0,
        hold_behavior: str = "maintain",
    ) -> None:
        if not signal:
            raise ValueError("signal is required")
        if hold_behavior != "maintain":
            raise ValueError("hold_behavior must be 'maintain'")
        self.signal = signal
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.hold_behavior = hold_behavior
        self._weights: dict[str, float] = {}
        self.skip_rebalance = False
        self.held_symbols: set[str] = set()
        self.pending_reduction_symbols: set[str] = set()
        self._held_position_symbols: set[str] | None = None
        self._pending_buy_symbols: set[str] = set()

    def reset(self) -> None:
        """Clear latched target weights at the start of an independent run."""
        self._weights = {}
        self.skip_rebalance = False
        self.held_symbols = set()
        self.pending_reduction_symbols = set()
        self._held_position_symbols = None
        self._pending_buy_symbols = set()

    def set_held_symbols(self, symbols: list[str]) -> None:
        """Snapshot symbols with actual positions before signal mapping."""
        self._held_position_symbols = set(symbols)

    def set_pending_buy_symbols(self, symbols: list[str]) -> None:
        """Snapshot symbols with open BUY orders before signal mapping."""
        self._pending_buy_symbols = set(symbols)

    def clear_pending_reductions(self, symbols: list[str]) -> None:
        """Mark pending reduction targets as reached."""
        for symbol in symbols:
            self.pending_reduction_symbols.discard(symbol)

    def reset_symbols(self, symbols: list[str]) -> None:
        """Clear latched target weights for symbols that fully exited."""
        for symbol in symbols:
            self._weights.pop(symbol, None)
            self.pending_reduction_symbols.discard(symbol)
            self.held_symbols.discard(symbol)

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        saw_signal = False
        saw_trade_intent = False
        buy_symbols: set[str] = set()
        seen_symbols: set[str] = set()
        self.held_symbols = set()
        for symbol, frame in signals.items():
            if self.signal not in frame.columns or frame.empty:
                continue
            seen_symbols.add(symbol)
            saw_signal = True
            value = frame[self.signal].iloc[-1]
            action = "HOLD" if pd.isna(value) else str(value).upper()
            if action == "BUY":
                saw_trade_intent = True
                buy_symbols.add(symbol)
                buy_weight = float(self.buy_weight)
                self._weights[symbol] = buy_weight
                self.pending_reduction_symbols.discard(symbol)
            elif action == "SELL":
                saw_trade_intent = True
                has_latched_position = symbol in self._weights or symbol in self.pending_reduction_symbols
                has_actual_position = (
                    symbol in self._held_position_symbols
                    if self._held_position_symbols is not None
                    else has_latched_position
                )
                has_pending_buy = symbol in self._pending_buy_symbols
                if symbol in self._weights and self.sell_weight > 0 and has_actual_position:
                    sell_weight = float(self.sell_weight)
                    self._weights[symbol] = sell_weight
                    self.pending_reduction_symbols.add(symbol)
                elif has_latched_position and has_actual_position:
                    self._weights.pop(symbol, None)
                    self.pending_reduction_symbols.add(symbol)
                elif has_latched_position and has_pending_buy:
                    self._weights.pop(symbol, None)
                    self.pending_reduction_symbols.add(symbol)
                else:
                    self._weights.pop(symbol, None)
                    self.pending_reduction_symbols.discard(symbol)
            elif action == "HOLD":
                if symbol in self._weights:
                    self.held_symbols.add(symbol)
                continue
            else:
                raise ValueError("expected BUY, SELL, or HOLD")

        for symbol in self._weights:
            if symbol not in seen_symbols:
                self.held_symbols.add(symbol)

        self.skip_rebalance = saw_signal and not saw_trade_intent
        positive_weights = {symbol: weight for symbol, weight in self._weights.items() if weight > 0}
        total = sum(positive_weights.values())
        if total <= 0:
            self._weights = {}
            return {"CASH": 1.0}
        if total > 1.0:
            preserved = {symbol: weight for symbol, weight in positive_weights.items() if symbol not in buy_symbols}
            reserved = sum(preserved.values())
            if reserved > 1.0:
                result = {symbol: weight / reserved for symbol, weight in preserved.items()}
            elif buy_symbols:
                result = dict(preserved)
                buy_weights = {
                    symbol: weight for symbol, weight in positive_weights.items() if symbol in buy_symbols
                }
                buy_total = sum(buy_weights.values())
                capacity = max(0.0, 1.0 - reserved)
                if buy_total > 0 and capacity > 0:
                    result.update({symbol: weight / buy_total * capacity for symbol, weight in buy_weights.items()})
            else:
                result = {symbol: weight / total for symbol, weight in positive_weights.items()}
            self._weights = {symbol: weight for symbol, weight in result.items() if weight > 0}
            return result

        result = dict(positive_weights)
        if total < 1.0:
            result["CASH"] = 1.0 - total
        self._weights = {symbol: weight for symbol, weight in result.items() if symbol != "CASH" and weight > 0}
        return result
