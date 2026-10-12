"""Entry-aware exit rules that remember the bar on which a position opened.

The engine evaluates every rule twice per bar (pre-optimization and post-trade
monitoring). Both rules therefore key their state on the bar date
(``row.name``) and never on the number of calls.

With ``trade_time: next_open`` the first bar on which a rule sees the symbol
held is the fill bar (fills are applied before rules run), so that bar is the
"entry bar". State is dropped as soon as a bar shows the symbol flat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, RuleResult


@dataclass
class _EntryState:
    entry_date: object
    entry_low: float
    entry_cost: float
    last_date: object
    bars_held: int = 0


class _EntryTracker:
    """Track the first bar on which each symbol is observed as held."""

    def __init__(self) -> None:
        self._states: dict[str, _EntryState] = {}

    def observe(self, symbol: str, row: pd.Series, portfolio: Portfolio) -> _EntryState | None:
        position = portfolio.positions.get(symbol)
        if position is None:
            self._states.pop(symbol, None)
            return None
        state = self._states.get(symbol)
        if state is None:
            state = _EntryState(
                entry_date=row.name,
                entry_low=float(row["low"]),
                entry_cost=float(position.avg_cost),
                last_date=row.name,
            )
            self._states[symbol] = state
        elif row.name != state.last_date:
            state.bars_held += 1
            state.last_date = row.name
        return state


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class EntryBarLowStopRule:
    """Exit when the close breaks the low of the entry bar (minus ``buffer``)."""

    name = "EntryBarLowStopRule"

    def __init__(self, buffer: float = 0.0) -> None:
        if not _is_finite_number(buffer) or not 0.0 <= buffer < 1.0:
            raise ValueError("EntryBarLowStopRule buffer must be a finite number in [0, 1)")
        self.buffer = float(buffer)
        self._tracker = _EntryTracker()

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        state = self._tracker.observe(symbol, row, portfolio)
        if state is None or row.name == state.entry_date:
            return RuleResult()
        stop = state.entry_low * (1.0 - self.buffer)
        if float(row["close"]) < stop:
            return RuleResult(
                target_positions={symbol: 0.0},
                reason=f"close {row['close']} < entry-bar stop {stop:.4f} ({state.entry_date}), exit {symbol}",
            )
        return RuleResult()


class TimeStopRule:
    """Exit once held for ``max_bars`` bars while return vs entry cost <= ``min_return``.

    The check keeps applying after ``max_bars`` (a limit-down bar may block the
    first exit), so past that point it also acts as a break-even stop.
    """

    name = "TimeStopRule"

    def __init__(self, max_bars: int = 3, min_return: float = 0.0) -> None:
        if not (isinstance(max_bars, int) and not isinstance(max_bars, bool) and max_bars > 0):
            raise ValueError("TimeStopRule max_bars must be a positive integer")
        if not _is_finite_number(min_return) or min_return <= -1.0:
            raise ValueError("TimeStopRule min_return must be a finite number greater than -1")
        self.max_bars = max_bars
        self.min_return = float(min_return)
        self._tracker = _EntryTracker()

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        state = self._tracker.observe(symbol, row, portfolio)
        if state is None or state.bars_held < self.max_bars or state.entry_cost <= 0:
            return RuleResult()
        ret = float(row["close"]) / state.entry_cost - 1.0
        if ret <= self.min_return:
            return RuleResult(
                target_positions={symbol: 0.0},
                reason=f"held {state.bars_held} bars, return {ret:.4f} <= {self.min_return}, exit {symbol}",
            )
        return RuleResult()
