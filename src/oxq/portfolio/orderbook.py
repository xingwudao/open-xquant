"""Order book — tracks orders through their lifecycle."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Literal

from oxq.core.types import Fill, Order


class ManagedOrder:
    """An order with lifecycle state, tracked by the OrderBook.

    Wraps an immutable Order with mutable state fields for
    tracking the order through its lifecycle.

    Attributes
    ----------
    order : Order
        The immutable order request.
    id : str
        Unique order identifier assigned by the OrderBook.
    status : str
        Current lifecycle state.
    created_at : str
        ISO date string when the order was created.
    filled_at : str or None
        ISO date string when the order was filled.
    filled_price : Decimal or None
        Actual fill price.
    filled_shares : int or None
        Number of shares actually filled.
    trail_high_water : Decimal or None
        High-water mark for trailing stop orders.
    """

    def __init__(self, order: Order, id: str, created_at: str) -> None:
        self.order = order
        self.id = id
        self.status: Literal[
            "open", "filled", "partial", "canceled", "expired", "rejected"
        ] = "open"
        self.created_at = created_at
        self.filled_at: str | None = None
        self.filled_price: Decimal | None = None
        self.filled_shares: int | None = None
        self.due_at: str | None = None
        self.status_reason: str = ""
        self.trail_high_water: Decimal | None = None


class OrderBook:
    """Order book tracking all orders and their lifecycle state.

    Supports market, limit, stop, stop_limit, and trailing_stop
    order types. Provides methods to add orders, query open orders,
    update status, and cancel orders.

    When adding a non-market order for the same symbol + side + order_type
    as an existing open non-market order, the old order is automatically
    canceled (deduplication). Market order replacement is broker-specific
    and must be handled before calling :meth:`add`.
    """

    def __init__(self) -> None:
        self._orders: list[ManagedOrder] = []
        self._counter: int = 0

    def add(self, order: Order, created_at: str) -> ManagedOrder:
        """Add an order to the book.

        Deduplicates non-market orders by canceling any existing open order
        with the same symbol + side + order_type.

        Parameters
        ----------
        order : Order
            The order to add.
        created_at : str
            ISO date string.

        Returns
        -------
        ManagedOrder
            The managed order with assigned ID.
        """
        for existing in self._orders:
            if (
                order.order_type != "market"
                and existing.status == "open"
                and existing.order.symbol == order.symbol
                and existing.order.side == order.side
                and existing.order.order_type == order.order_type
            ):
                existing.status = "canceled"
                existing.status_reason = "replaced"
        self._counter += 1
        managed = ManagedOrder(
            order=order, id=f"ord_{self._counter}", created_at=created_at,
        )
        self._orders.append(managed)
        return managed

    def get_open_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        """Return all open orders, optionally filtered by symbol.

        Parameters
        ----------
        symbol : str or None
            If provided, filter by symbol.

        Returns
        -------
        list[ManagedOrder]
            Open orders.
        """
        result = [m for m in self._orders if m.status == "open"]
        if symbol is not None:
            result = [m for m in result if m.order.symbol == symbol]
        return result

    def get_all_orders(self) -> list[ManagedOrder]:
        """Return all orders with their current lifecycle status."""
        return list(self._orders)

    def cancel_orders(
        self,
        symbol: str,
        side: str | None = None,
    ) -> list[ManagedOrder]:
        """Cancel all open orders for a symbol.

        Parameters
        ----------
        symbol : str
            The symbol whose orders to cancel.
        side : str or None
            If provided, only cancel orders with this side.

        Returns
        -------
        list[ManagedOrder]
            The canceled orders.
        """
        canceled = []
        for m in self._orders:
            if m.status != "open" or m.order.symbol != symbol:
                continue
            if side is not None and m.order.side != side:
                continue
            m.status = "canceled"
            m.status_reason = "canceled"
            canceled.append(m)
        return canceled

    def cap_pending_sells(self, symbol: str, max_shares: int) -> None:
        """Cap shares on open SELL orders to at most *max_shares*.

        If *max_shares* is 0 or negative the order is canceled instead.
        Order type, prices, and other fields are preserved.

        Parameters
        ----------
        symbol : str
            Symbol to cap.
        max_shares : int
            Maximum allowed sell shares (typically current position size).
        """
        for m in self._orders:
            if (
                m.status == "open"
                and m.order.symbol == symbol
                and m.order.side == "SELL"
                and m.order.order_type != "market"
            ):
                if max_shares <= 0:
                    m.status = "canceled"
                    m.status_reason = "position_closed"
                elif m.order.shares > max_shares:
                    m.order = replace(m.order, shares=max_shares)

    def fill(
        self,
        managed_order: ManagedOrder,
        price: Decimal,
        date: str,
        fee: Decimal = Decimal("0"),
    ) -> Fill:
        """Mark an order as filled and return the Fill.

        Parameters
        ----------
        managed_order : ManagedOrder
            The order to fill.
        price : Decimal
            Fill price.
        date : str
            Fill date.
        fee : Decimal
            Transaction fee.

        Returns
        -------
        Fill
            The completed fill.
        """
        managed_order.status = "filled"
        managed_order.filled_at = date
        managed_order.filled_price = price
        managed_order.filled_shares = managed_order.order.shares
        return Fill(
            order=managed_order.order,
            filled_price=price,
            filled_at=date,
            fee=fee,
        )
