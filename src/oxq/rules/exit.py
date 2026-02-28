"""Exit rule — generates SELL orders when fast MA drops below slow MA."""

from __future__ import annotations

import pandas as pd

from oxq.core.types import Order, Portfolio


class ExitRule:
    """Sell entire position when the fast indicator drops below the slow one."""

    name = "ExitRule"

    def __init__(self, fast: str, slow: str) -> None:
        self.fast = fast
        self.slow = slow

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        if symbol in portfolio.positions and row[self.fast] < row[self.slow]:
            pos = portfolio.positions[symbol]
            return Order(symbol=symbol, side="SELL", shares=pos.shares)
        return None
