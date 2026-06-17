"""Simulated broker — implements OrderRouter + FillReceiver for backtesting."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

import pandas as pd

from oxq.core.types import Fill, Order
from oxq.portfolio.orderbook import ManagedOrder, OrderBook
from oxq.trade.fees import FeeModel
from oxq.trade.slippage import SlippageModel


class FillPriceMode(Enum):
    """Fill price mode for market orders."""

    CLOSE = "close"
    MID = "mid"
    NEXT_OPEN = "next_open"
    NEXT_HIGH = "next_high"
    NEXT_LOW = "next_low"


class SimBroker:
    """Simulated broker with order book, fee and slippage models.

    Implements both :class:`OrderRouter` and :class:`FillReceiver` protocols.
    Market orders are queued and filled at bar close via
    :meth:`fill_market_orders`. Non-market orders (stop, limit,
    trailing_stop) are held in an internal OrderBook and processed
    each bar via :meth:`process_pending_orders`.

    Parameters
    ----------
    fee_model : FeeModel or None
        Fee calculation model. If None, no fees are charged.
    slippage_model : SlippageModel or None
        Slippage simulation model. If None, orders fill at raw price.

    Examples
    --------
    >>> broker = SimBroker()
    >>> broker = SimBroker(
    ...     fee_model=PercentageFee(),
    ...     slippage_model=PercentageSlippage(),
    ... )
    """

    def __init__(
        self,
        fee_model: FeeModel | None = None,
        slippage_model: SlippageModel | None = None,
        fill_price_mode: FillPriceMode = FillPriceMode.CLOSE,
    ) -> None:
        self._fee_model = fee_model
        self._slippage_model = slippage_model
        self._fill_price_mode = fill_price_mode
        self._order_book = OrderBook()
        self._pending_market: list[ManagedOrder] = []
        self._fills: list[Fill] = []
        self._current_date: pd.Timestamp | None = None

    # -- Broker lifecycle hooks -----------------------------------------------

    def on_bar_open(
        self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> None:
        """Process pending stop/limit/trailing_stop orders at bar open."""
        if self._fill_price_mode == FillPriceMode.NEXT_OPEN:
            self.fill_due_market_orders(mktdata, date)
        self.process_pending_orders(mktdata, date)

    def on_bar_close(
        self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> None:
        """Fill market orders at bar close."""
        self.fill_market_orders(mktdata, date)

    # -- OrderRouter ----------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit an order.

        Market orders are queued for end-of-bar fill.
        Non-market orders are added to the order book.

        Parameters
        ----------
        order : Order
            The order to submit.

        Returns
        -------
        str
            Order ID.
        """
        created_at = self._current_date.isoformat() if self._current_date is not None else ""
        managed = self._order_book.add(order, created_at=created_at)
        if order.order_type == "market":
            self._pending_market.append(managed)
        return managed.id

    def set_current_date(self, date: pd.Timestamp) -> None:
        """Set the engine bar date used to timestamp newly submitted orders."""
        self._current_date = pd.Timestamp(date)

    # -- Order Processing -----------------------------------------------------

    def process_pending_orders(
        self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> None:
        """Check all pending stop/limit/trailing_stop orders for trigger.

        Called by the Engine at the start of the Order stage.

        Parameters
        ----------
        mktdata : dict[str, pd.DataFrame]
            Market data keyed by symbol.
        date : pd.Timestamp
            Current bar date.
        """
        for managed in list(self._order_book.get_open_orders()):
            order = managed.order
            if order.order_type == "market":
                continue
            if order.symbol not in mktdata:
                continue
            if date not in mktdata[order.symbol].index:
                continue

            close = Decimal(str(float(mktdata[order.symbol].loc[date, "close"])))  # type: ignore[arg-type]
            if not close.is_finite():
                continue

            if order.order_type == "stop":
                triggered = self._check_stop(order, close)
                if triggered and order.stop_price is not None:
                    fill_price = self._apply_slippage(order, order.stop_price)
                    fee = self._calc_fee(order, fill_price)
                    fill = self._order_book.fill(managed, fill_price, date.isoformat(), fee)
                    self._fills.append(fill)

            elif order.order_type == "limit":
                triggered = self._check_limit(order, close)
                if triggered and order.limit_price is not None:
                    fill_price = order.limit_price
                    fee = self._calc_fee(order, fill_price)
                    fill = self._order_book.fill(managed, fill_price, date.isoformat(), fee)
                    self._fills.append(fill)

            elif order.order_type == "trailing_stop":
                if managed.trail_high_water is None:
                    managed.trail_high_water = close
                else:
                    managed.trail_high_water = max(managed.trail_high_water, close)
                hwm = managed.trail_high_water
                stop_level = hwm * (1 - Decimal(str(order.trail_pct)))
                if close <= stop_level:
                    fill_price = self._apply_slippage(order, stop_level)
                    fee = self._calc_fee(order, fill_price)
                    fill = self._order_book.fill(managed, fill_price, date.isoformat(), fee)
                    self._fills.append(fill)

    def fill_market_orders(
        self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> None:
        """Fill all pending market orders at the configured fill price.

        Parameters
        ----------
        mktdata : dict[str, pd.DataFrame]
            Market data keyed by symbol.
        date : pd.Timestamp
            Current bar date.
        """
        still_pending: list[ManagedOrder] = []
        for managed in self._pending_market:
            order = managed.order
            raw_price = self._get_fill_price(managed, mktdata, date)
            if raw_price is None:
                still_pending.append(managed)
                continue
            fill_price = self._apply_slippage(order, raw_price)
            fee = self._calc_fee(order, fill_price)
            fill = self._order_book.fill(managed, fill_price, date.isoformat(), fee)
            self._fills.append(fill)
        self._pending_market = still_pending

    def fill_due_market_orders(
        self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> None:
        """Fill market orders whose configured execution time has arrived."""
        self.fill_market_orders(mktdata, date)

    # -- FillReceiver ---------------------------------------------------------

    def get_fills(self) -> list[Fill]:
        """Return fills accumulated since the last call, then clear."""
        fills = list(self._fills)
        self._fills.clear()
        return fills

    # -- Query ----------------------------------------------------------------

    def get_open_orders(self, symbol: str | None = None) -> list[ManagedOrder]:
        """Return open orders from the order book.

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

    def cap_pending_sells(self, symbol: str, max_shares: int) -> None:
        """Cap pending SELL order shares to current position size.

        Prevents stop/limit/trailing_stop orders from selling more
        shares than currently held after a partial position reduction.

        Parameters
        ----------
        symbol : str
            Symbol to cap.
        max_shares : int
            Maximum sell shares (current position size).
        """
        self._order_book.cap_pending_sells(symbol, max_shares)

    def cancel_orders(
        self, symbol: str, side: str | None = None,
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
        return self._order_book.cancel_orders(symbol, side)

    # -- Backward Compatibility -----------------------------------------------

    def fill_pending_orders(
        self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> None:
        """Legacy method: process pending + fill market in one call.

        Parameters
        ----------
        mktdata : dict[str, pd.DataFrame]
            Market data keyed by symbol.
        date : pd.Timestamp
            Current bar date.
        """
        self.process_pending_orders(mktdata, date)
        self.fill_market_orders(mktdata, date)

    # -- Private helpers ------------------------------------------------------

    def _get_fill_price(
        self, managed: ManagedOrder, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp,
    ) -> Decimal | None:
        """Get fill price based on fill_price_mode."""
        symbol = managed.order.symbol
        df = mktdata[symbol]
        if date not in df.index:
            return None
        if self._fill_price_mode == FillPriceMode.CLOSE:
            price = Decimal(str(float(df.loc[date, "close"])))  # type: ignore[arg-type]
            return price if price.is_finite() else None

        if self._fill_price_mode == FillPriceMode.MID:
            open_price = Decimal(str(float(df.loc[date, "open"])))
            close_price = Decimal(str(float(df.loc[date, "close"])))
            if not open_price.is_finite() or not close_price.is_finite():
                return None
            return (open_price + close_price) / 2

        if managed.created_at:
            created_at = pd.Timestamp(managed.created_at)
            if pd.Timestamp(date) <= created_at:
                return None
            col = {
                FillPriceMode.NEXT_OPEN: "open",
                FillPriceMode.NEXT_HIGH: "high",
                FillPriceMode.NEXT_LOW: "low",
            }[self._fill_price_mode]
            price = Decimal(str(float(df.loc[date, col])))  # type: ignore[arg-type]
            return price if price.is_finite() else None

        # Legacy direct-broker use without an engine-created timestamp.
        idx = df.index.get_loc(date)
        if idx + 1 >= len(df):
            return None  # no next bar
        next_date = df.index[idx + 1]
        col = {
            FillPriceMode.NEXT_OPEN: "open",
            FillPriceMode.NEXT_HIGH: "high",
            FillPriceMode.NEXT_LOW: "low",
        }[self._fill_price_mode]
        price = Decimal(str(float(df.loc[next_date, col])))  # type: ignore[arg-type]
        return price if price.is_finite() else None

    def _apply_slippage(self, order: Order, price: Decimal) -> Decimal:
        if self._slippage_model:
            return self._slippage_model.adjust(order, price)
        return price

    def _calc_fee(self, order: Order, fill_price: Decimal) -> Decimal:
        if self._fee_model:
            return self._fee_model.calculate(order, fill_price)
        return Decimal("0")

    @staticmethod
    def _check_stop(order: Order, close: Decimal) -> bool:
        if order.stop_price is None:
            return False
        if order.side == "SELL":
            return close <= order.stop_price
        return close >= order.stop_price

    @staticmethod
    def _check_limit(order: Order, close: Decimal) -> bool:
        if order.limit_price is None:
            return False
        if order.side == "SELL":
            return close >= order.limit_price
        return close <= order.limit_price
