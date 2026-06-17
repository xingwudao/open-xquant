"""LiveBroker — routes orders to a real brokerage via AlpacaClient."""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any

import pandas as pd

from oxq.contrib.alpaca.client import AlpacaClient
from oxq.core.types import Fill, Order
from oxq.portfolio.orderbook import ManagedOrder, OrderBook


class LiveBroker:
    """Broker Protocol implementation backed by the Alpaca Trading API.

    Parameters
    ----------
    api_key : str or None
        Alpaca API key. Falls back to ALPACA_API_KEY env var.
    secret_key : str or None
        Alpaca secret key. Falls back to ALPACA_SECRET_KEY env var.
    paper : bool
        If True (default), connect to Paper Trading environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        self._client = AlpacaClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self._order_book = OrderBook()
        self._fills: list[Fill] = []
        self._lock = threading.Lock()
        self._id_map: dict[str, ManagedOrder] = {}
        self._client.start_trade_stream(self._on_ws_message)

    # -- Account & Positions ---------------------------------------------------

    def get_account(self) -> dict[str, Any]:
        """Get account info (status, equity, buying power, etc.)."""
        return self._client.get_account()

    def get_positions_detail(self) -> list[dict[str, Any]]:
        """Get current positions from the brokerage."""
        return self._client.get_positions()

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get the status of a specific order by its Alpaca ID."""
        return self._client.get_order(order_id)

    # -- OrderRouter -----------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit an order to Alpaca and track it in the local order book.

        Parameters
        ----------
        order : Order
            The order to submit.

        Returns
        -------
        str
            The Alpaca order ID.
        """
        params = _order_to_alpaca(order)
        resp = self._client.submit_order(params)
        alpaca_id = resp["id"]
        managed = self._order_book.add(order, created_at="")
        managed.id = alpaca_id
        self._id_map[alpaca_id] = managed
        return alpaca_id

    # -- FillReceiver ----------------------------------------------------------

    def get_fills(self) -> list[Fill]:
        """Return and clear all pending fills.

        Returns
        -------
        list[Fill]
            Fills received since the last call.
        """
        with self._lock:
            fills = self._fills
            self._fills = []
            return fills

    # -- Broker lifecycle hooks ------------------------------------------------

    def on_bar_open(self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp) -> None:
        """No-op for live trading — orders are processed by the exchange."""

    def on_bar_close(self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp) -> None:
        """No-op for live trading — orders are processed by the exchange."""

    # -- Order management ------------------------------------------------------

    def get_open_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        """Return open orders, optionally filtered by symbol.

        Parameters
        ----------
        symbol : str or None
            If provided, filter by symbol.

        Returns
        -------
        list[ManagedOrder]
            Open orders.
        """
        return self._order_book.get_open_orders(symbol)

    def cancel_orders(self, symbol: str, side: str | None = None) -> list[ManagedOrder]:
        """Cancel open orders for a symbol via Alpaca, then update local book.

        Parameters
        ----------
        symbol : str
            Symbol whose orders to cancel.
        side : str or None
            If provided, only cancel orders with this side.

        Returns
        -------
        list[ManagedOrder]
            The canceled orders.
        """
        to_cancel = self._order_book.get_open_orders(symbol)
        if side:
            to_cancel = [m for m in to_cancel if m.order.side == side]
        for managed in to_cancel:
            self._client.cancel_order(managed.id)
        return self._order_book.cancel_orders(symbol, side)

    def cancel_market_orders(self, symbol: str, side: str | None = None, reason: str = "canceled") -> list[ManagedOrder]:
        """Cancel open market orders for a symbol via Alpaca."""
        canceled: list[ManagedOrder] = []
        for managed in self._order_book.get_open_orders(symbol):
            if managed.order.order_type != "market":
                continue
            if side is not None and managed.order.side != side:
                continue
            self._client.cancel_order(managed.id)
            managed.status = "canceled"
            managed.status_reason = reason
            canceled.append(managed)
        return canceled

    def cap_pending_sells(self, symbol: str, max_shares: int) -> None:
        """Cap pending sell orders to at most *max_shares*.

        Cancels oversized orders on Alpaca and re-submits with capped shares.

        Parameters
        ----------
        symbol : str
            Symbol to cap.
        max_shares : int
            Maximum allowed sell shares.
        """
        sells = [
            m for m in self._order_book.get_open_orders(symbol)
            if m.order.side == "SELL" and m.order.order_type != "market"
        ]
        for managed in sells:
            if max_shares <= 0 or managed.order.shares > max_shares:
                self._client.cancel_order(managed.id)
                del self._id_map[managed.id]
        self._order_book.cap_pending_sells(symbol, max_shares)
        for managed in self._order_book.get_open_orders(symbol):
            if (
                managed.order.side == "SELL"
                and managed.order.order_type != "market"
                and managed.id not in self._id_map
            ):
                params = _order_to_alpaca(managed.order)
                resp = self._client.submit_order(params)
                new_id = resp["id"]
                managed.id = new_id
                self._id_map[new_id] = managed

    # -- WebSocket callback ----------------------------------------------------

    def _on_ws_message(self, msg: dict[str, Any]) -> None:
        """Route incoming WebSocket messages to the appropriate handler."""
        data = msg.get("data", {})
        if data.get("event") == "fill":
            self._on_fill_event(data)

    def _on_fill_event(self, data: dict[str, Any]) -> None:
        """Process a fill event and record the Fill."""
        order_data = data.get("order", {})
        alpaca_id = order_data.get("id", "")
        managed = self._id_map.get(alpaca_id)
        if managed is None:
            return
        filled_price = Decimal(order_data.get("filled_avg_price", "0"))
        filled_at = order_data.get("filled_at", "")
        fill = self._order_book.fill(managed, filled_price, filled_at, fee=Decimal("0"))
        with self._lock:
            self._fills.append(fill)

    def close(self) -> None:
        """Stop the WebSocket stream and close the HTTP client."""
        self._client.stop_trade_stream()
        self._client.close()


def _order_to_alpaca(order: Order) -> dict[str, str]:
    """Convert an oxq Order to Alpaca order parameters.

    Parameters
    ----------
    order : Order
        The order to convert.

    Returns
    -------
    dict[str, str]
        Alpaca-compatible order parameters.
    """
    params: dict[str, str] = {
        "symbol": order.symbol,
        "side": order.side.lower(),
        "qty": str(order.shares),
        "type": order.order_type,
        "time_in_force": "day",
    }
    if order.limit_price is not None:
        params["limit_price"] = str(order.limit_price)
    if order.stop_price is not None:
        params["stop_price"] = str(order.stop_price)
    if order.trail_pct is not None:
        params["trail_percent"] = str(order.trail_pct * 100)
    return params
