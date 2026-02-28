"""Simulated broker for backtesting — implements OrderRouter + FillReceiver."""

from __future__ import annotations

import pandas as pd

from oxq.core.types import Fill, Order


class SimBroker:
    """Simulated broker that fills orders at the current bar's close price.

    Implements both :class:`OrderRouter` and :class:`FillReceiver` protocols.
    In backtest mode, orders are queued via :meth:`submit_order` and filled
    via :meth:`fill_pending_orders` each bar.
    """

    def __init__(self) -> None:
        self._pending: list[Order] = []
        self._fills: list[Fill] = []
        self._order_count: int = 0

    # -- OrderRouter ----------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Queue an order for execution. Returns an order id."""
        self._order_count += 1
        self._pending.append(order)
        return f"order_{self._order_count}"

    # -- FillReceiver ---------------------------------------------------------

    def get_fills(self) -> list[Fill]:
        """Return fills accumulated since the last call, then clear."""
        fills = list(self._fills)
        self._fills.clear()
        return fills

    # -- Engine hook ----------------------------------------------------------

    def fill_pending_orders(
        self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> None:
        """Fill all pending orders at each symbol's close price for *date*."""
        for order in self._pending:
            price: float = mktdata[order.symbol].loc[date, "close"]  # type: ignore[assignment]
            self._fills.append(
                Fill(order=order, filled_price=price, filled_at=str(date)),
            )
        self._pending.clear()
