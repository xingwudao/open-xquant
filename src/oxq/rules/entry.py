"""Entry rule — generates BUY orders when a signal fires."""

from __future__ import annotations

import pandas as pd

from oxq.core.types import Order, Portfolio


class EntryRule:
    """Buy when the named signal column is True and no position is held."""

    name = "EntryRule"

    def __init__(self, signal: str, shares: int = 100) -> None:
        self.signal = signal
        self.shares = shares

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        if row.get(self.signal) and symbol not in portfolio.positions:
            return Order(symbol=symbol, side="BUY", shares=self.shares)
        return None
