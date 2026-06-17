"""Tests for OrderBook."""

from decimal import Decimal

from oxq.core.types import Order
from oxq.portfolio.orderbook import ManagedOrder, OrderBook


def test_add_order_returns_managed_order() -> None:
    book = OrderBook()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    managed = book.add(order, created_at="2024-01-02")
    assert isinstance(managed, ManagedOrder)
    assert managed.order is order
    assert managed.status == "open"
    assert managed.id.startswith("ord_")


def test_get_open_orders_all() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(Order(symbol="MSFT", side="BUY", shares=50), "2024-01-02")
    assert len(book.get_open_orders()) == 2


def test_get_open_orders_by_symbol() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(Order(symbol="MSFT", side="BUY", shares=50), "2024-01-02")
    assert len(book.get_open_orders(symbol="AAPL")) == 1


def test_cancel_orders() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(
        Order(
            symbol="AAPL",
            side="SELL",
            shares=50,
            order_type="stop",
            stop_price=Decimal("140"),
        ),
        "2024-01-02",
    )
    canceled = book.cancel_orders("AAPL")
    assert len(canceled) == 2
    assert all(m.status == "canceled" for m in canceled)
    assert len(book.get_open_orders()) == 0


def test_fill_order() -> None:
    book = OrderBook()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    managed = book.add(order, "2024-01-02")
    fill = book.fill(managed, price=Decimal("150"), date="2024-01-02")
    assert fill.filled_price == Decimal("150")
    assert managed.status == "filled"
    assert len(book.get_open_orders()) == 0


def test_dedup_replaces_same_symbol_side_type() -> None:
    book = OrderBook()
    o1 = Order(
        symbol="AAPL",
        side="SELL",
        shares=100,
        order_type="stop",
        stop_price=Decimal("140"),
    )
    o2 = Order(
        symbol="AAPL",
        side="SELL",
        shares=100,
        order_type="stop",
        stop_price=Decimal("145"),
    )
    book.add(o1, "2024-01-02")
    book.add(o2, "2024-01-03")
    open_orders = book.get_open_orders(symbol="AAPL")
    assert len(open_orders) == 1
    assert open_orders[0].order.stop_price == Decimal("145")


def test_cancel_orders_with_side_filter() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(
        Order(
            symbol="AAPL",
            side="SELL",
            shares=50,
            order_type="stop",
            stop_price=Decimal("140"),
        ),
        "2024-01-02",
    )
    canceled = book.cancel_orders("AAPL", side="SELL")
    assert len(canceled) == 1
    assert canceled[0].order.side == "SELL"
    assert len(book.get_open_orders()) == 1
    assert book.get_open_orders()[0].order.side == "BUY"


def test_fill_with_fee() -> None:
    book = OrderBook()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    managed = book.add(order, "2024-01-02")
    fill = book.fill(managed, price=Decimal("150"), date="2024-01-02", fee=Decimal("15"))
    assert fill.fee == Decimal("15")


def test_market_orders_are_not_deduped_in_shared_orderbook() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(Order(symbol="AAPL", side="BUY", shares=200), "2024-01-02")
    orders = book.get_all_orders()
    assert len(book.get_open_orders()) == 2
    assert [order.status for order in orders] == ["open", "open"]


def test_filled_order_not_in_open_orders() -> None:
    book = OrderBook()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    managed = book.add(order, "2024-01-02")
    book.fill(managed, price=Decimal("150"), date="2024-01-02")
    assert len(book.get_open_orders()) == 0


# ---------------------------------------------------------------------------
# cap_pending_sells
# ---------------------------------------------------------------------------


def test_cap_pending_sells_reduces_shares() -> None:
    """Shares on a stop SELL should be capped to max_shares."""
    book = OrderBook()
    book.add(
        Order(symbol="AAPL", side="SELL", shares=500,
              order_type="stop", stop_price=Decimal("140")),
        "2024-01-02",
    )
    book.cap_pending_sells("AAPL", max_shares=300)
    open_orders = book.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order.shares == 300
    # Other fields preserved
    assert open_orders[0].order.stop_price == Decimal("140")
    assert open_orders[0].order.order_type == "stop"


def test_cap_pending_sells_cancels_when_zero() -> None:
    """When max_shares is 0, the order should be canceled."""
    book = OrderBook()
    book.add(
        Order(symbol="AAPL", side="SELL", shares=500,
              order_type="limit", limit_price=Decimal("180")),
        "2024-01-02",
    )
    book.cap_pending_sells("AAPL", max_shares=0)
    assert len(book.get_open_orders()) == 0


def test_cap_pending_sells_noop_when_within_limit() -> None:
    """If shares <= max_shares, the order should not be modified."""
    book = OrderBook()
    book.add(
        Order(symbol="AAPL", side="SELL", shares=300,
              order_type="stop", stop_price=Decimal("140")),
        "2024-01-02",
    )
    book.cap_pending_sells("AAPL", max_shares=500)
    open_orders = book.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order.shares == 300


def test_cap_pending_sells_ignores_market_orders() -> None:
    """Market orders should not be affected by cap."""
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="SELL", shares=500), "2024-01-02")
    book.cap_pending_sells("AAPL", max_shares=100)
    open_orders = book.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order.shares == 500


def test_cap_pending_sells_ignores_buy_orders() -> None:
    """BUY orders should not be affected by cap."""
    book = OrderBook()
    book.add(
        Order(symbol="AAPL", side="BUY", shares=500,
              order_type="stop", stop_price=Decimal("160")),
        "2024-01-02",
    )
    book.cap_pending_sells("AAPL", max_shares=100)
    open_orders = book.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order.shares == 500


def test_cap_pending_sells_ignores_other_symbols() -> None:
    """Only the specified symbol's orders should be capped."""
    book = OrderBook()
    book.add(
        Order(symbol="AAPL", side="SELL", shares=500,
              order_type="stop", stop_price=Decimal("140")),
        "2024-01-02",
    )
    book.add(
        Order(symbol="MSFT", side="SELL", shares=500,
              order_type="stop", stop_price=Decimal("280")),
        "2024-01-02",
    )
    book.cap_pending_sells("AAPL", max_shares=200)
    aapl = book.get_open_orders("AAPL")
    msft = book.get_open_orders("MSFT")
    assert aapl[0].order.shares == 200
    assert msft[0].order.shares == 500


def test_cap_pending_sells_multiple_order_types() -> None:
    """Both stop and limit SELL orders for the same symbol should be capped."""
    book = OrderBook()
    book.add(
        Order(symbol="AAPL", side="SELL", shares=500,
              order_type="stop", stop_price=Decimal("140")),
        "2024-01-02",
    )
    book.add(
        Order(symbol="AAPL", side="SELL", shares=500,
              order_type="limit", limit_price=Decimal("180")),
        "2024-01-02",
    )
    book.cap_pending_sells("AAPL", max_shares=300)
    open_orders = book.get_open_orders("AAPL")
    assert len(open_orders) == 2
    for o in open_orders:
        assert o.order.shares == 300


def test_cap_pending_sells_preserves_trailing_stop_hwm() -> None:
    """Capping shares should not lose the trailing stop high-water mark."""
    book = OrderBook()
    managed = book.add(
        Order(symbol="AAPL", side="SELL", shares=500,
              order_type="trailing_stop", trail_pct=0.05),
        "2024-01-02",
    )
    managed.trail_high_water = Decimal("200")
    book.cap_pending_sells("AAPL", max_shares=300)
    open_orders = book.get_open_orders("AAPL")
    assert open_orders[0].order.shares == 300
    assert open_orders[0].trail_high_water == Decimal("200")
