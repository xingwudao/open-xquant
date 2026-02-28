"""Tests for SimBroker."""

import pandas as pd

from oxq.backtest.broker import SimBroker
from oxq.core.types import FillReceiver, Order, OrderRouter


def test_sim_broker_satisfies_order_router_protocol() -> None:
    assert isinstance(SimBroker(), OrderRouter)


def test_sim_broker_satisfies_fill_receiver_protocol() -> None:
    assert isinstance(SimBroker(), FillReceiver)


def test_submit_order_returns_id() -> None:
    broker = SimBroker()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    order_id = broker.submit_order(order)
    assert order_id == "order_1"


def test_fill_pending_orders() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=2)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"close": [150.0, 155.0]}, index=dates,
        ),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0].filled_price == 150.0
    assert fills[0].order.symbol == "AAPL"


def test_get_fills_clears_after_read() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.fill_pending_orders(mktdata, dates[0])

    assert len(broker.get_fills()) == 1
    assert len(broker.get_fills()) == 0  # cleared


def test_multi_symbol_fill() -> None:
    broker = SimBroker()
    dates = pd.bdate_range("2024-01-01", periods=1)
    mktdata = {
        "AAPL": pd.DataFrame({"close": [150.0]}, index=dates),
        "MSFT": pd.DataFrame({"close": [300.0]}, index=dates),
    }
    broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
    broker.submit_order(Order(symbol="MSFT", side="BUY", shares=50))
    broker.fill_pending_orders(mktdata, dates[0])

    fills = broker.get_fills()
    assert len(fills) == 2
    prices = {f.order.symbol: f.filled_price for f in fills}
    assert prices["AAPL"] == 150.0
    assert prices["MSFT"] == 300.0
