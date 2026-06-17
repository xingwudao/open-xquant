"""Tests for LiveBroker — Broker Protocol implementation over Alpaca."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest

from oxq.core.types import Broker, Order
from oxq.trade.live_broker import LiveBroker


@pytest.fixture
def mock_client():
    """Return a LiveBroker with mocked AlpacaClient."""
    with patch("oxq.trade.live_broker.AlpacaClient") as mock_cls:
        instance = mock_cls.return_value
        instance.submit_order.return_value = {"id": "alpaca-001", "status": "accepted"}
        instance.start_trade_stream.return_value = None
        instance.stop_trade_stream.return_value = None
        broker = LiveBroker(api_key="k", secret_key="s")
        yield broker, instance


class TestSubmitOrder:
    def test_market_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        oid = broker.submit_order(order)
        assert oid == "alpaca-001"
        call_args = client.submit_order.call_args[0][0]
        assert call_args["symbol"] == "AAPL"
        assert call_args["side"] == "buy"
        assert call_args["qty"] == "100"
        assert call_args["type"] == "market"
        assert call_args["time_in_force"] == "day"

    def test_limit_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="GOOG", side="SELL", shares=50, order_type="limit", limit_price=Decimal("150.50"))
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "limit"
        assert call_args["limit_price"] == "150.50"

    def test_stop_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="TSLA", side="SELL", shares=10, order_type="stop", stop_price=Decimal("200.00"))
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "stop"
        assert call_args["stop_price"] == "200.00"

    def test_stop_limit_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(
            symbol="MSFT", side="BUY", shares=25, order_type="stop_limit",
            stop_price=Decimal("300.00"), limit_price=Decimal("305.00"),
        )
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "stop_limit"
        assert call_args["stop_price"] == "300.00"
        assert call_args["limit_price"] == "305.00"

    def test_trailing_stop_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AMZN", side="SELL", shares=15, order_type="trailing_stop", trail_pct=0.05)
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "trailing_stop"
        assert call_args["trail_percent"] == "5.0"

    def test_order_registered_in_orderbook(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        oid = broker.submit_order(order)
        open_orders = broker.get_open_orders("AAPL")
        assert len(open_orders) == 1
        assert open_orders[0].id == oid

    def test_market_orders_are_not_locally_replaced_without_remote_cancel(self, mock_client):
        broker, client = mock_client
        client.submit_order.side_effect = [
            {"id": "alpaca-001", "status": "accepted"},
            {"id": "alpaca-002", "status": "accepted"},
        ]

        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=100))
        broker.submit_order(Order(symbol="AAPL", side="BUY", shares=200))

        client.cancel_order.assert_not_called()
        open_orders = broker.get_open_orders("AAPL")
        assert [order.id for order in open_orders] == ["alpaca-001", "alpaca-002"]


class TestGetFills:
    def test_empty_fills(self, mock_client):
        broker, _ = mock_client
        assert broker.get_fills() == []

    def test_fill_from_websocket_callback(self, mock_client):
        broker, _ = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        broker.submit_order(order)
        broker._on_fill_event({
            "event": "fill",
            "order": {
                "id": "alpaca-001",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "100",
                "type": "market",
                "filled_avg_price": "150.25",
                "filled_at": "2026-03-11T10:00:00Z",
            },
        })
        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("150.25")
        assert fills[0].order.symbol == "AAPL"
        assert fills[0].filled_at == "2026-03-11T10:00:00Z"

    def test_get_fills_clears_queue(self, mock_client):
        broker, _ = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        broker.submit_order(order)
        broker._on_fill_event({
            "event": "fill",
            "order": {
                "id": "alpaca-001",
                "filled_avg_price": "150.25",
                "filled_at": "2026-03-11T10:00:00Z",
            },
        })
        broker.get_fills()
        assert broker.get_fills() == []


class TestOnWsMessage:
    def test_full_message_envelope_routes_to_fill(self, mock_client):
        """_on_ws_message unwraps {"data": {"event": "fill", ...}} correctly."""
        broker, _ = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        broker.submit_order(order)
        broker._on_ws_message({
            "data": {
                "event": "fill",
                "order": {
                    "id": "alpaca-001",
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": "100",
                    "type": "market",
                    "filled_avg_price": "152.00",
                    "filled_at": "2026-03-11T14:30:00Z",
                },
            },
        })
        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("152.00")
        assert fills[0].filled_at == "2026-03-11T14:30:00Z"

    def test_non_fill_event_ignored(self, mock_client):
        """_on_ws_message ignores events that are not 'fill'."""
        broker, _ = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        broker.submit_order(order)
        broker._on_ws_message({
            "data": {
                "event": "partial_fill",
                "order": {
                    "id": "alpaca-001",
                    "filled_avg_price": "150.00",
                    "filled_at": "2026-03-11T14:30:00Z",
                },
            },
        })
        assert broker.get_fills() == []


class TestAccountAndPositions:
    def test_get_account_delegates(self, mock_client):
        broker, client = mock_client
        client.get_account.return_value = {"status": "ACTIVE", "equity": "100000"}
        result = broker.get_account()
        assert result["status"] == "ACTIVE"
        client.get_account.assert_called_once()

    def test_get_positions_detail_delegates(self, mock_client):
        broker, client = mock_client
        client.get_positions.return_value = [{"symbol": "AAPL", "qty": "50"}]
        result = broker.get_positions_detail()
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        client.get_positions.assert_called_once()

    def test_get_order_status_delegates(self, mock_client):
        broker, client = mock_client
        client.get_order.return_value = {"id": "abc", "status": "filled"}
        result = broker.get_order_status("abc")
        assert result["status"] == "filled"
        client.get_order.assert_called_once_with("abc")


class TestProtocolCompliance:
    def test_satisfies_broker_protocol(self, mock_client):
        broker, _ = mock_client
        assert isinstance(broker, Broker)


class TestClose:
    def test_close_stops_trade_stream(self, mock_client):
        """close() delegates to client.stop_trade_stream() and client.close()."""
        broker, client = mock_client
        broker.close()
        client.stop_trade_stream.assert_called_once()
        client.close.assert_called_once()


class TestCancelOrders:
    def test_cancel_orders_calls_alpaca(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AAPL", side="SELL", shares=50, order_type="stop", stop_price=Decimal("140.00"))
        broker.submit_order(order)
        canceled = broker.cancel_orders("AAPL")
        assert len(canceled) == 1
        assert canceled[0].status == "canceled"
        client.cancel_order.assert_called_once_with("alpaca-001")

    def test_cancel_orders_with_side_filter(self, mock_client):
        broker, client = mock_client
        client.submit_order.side_effect = [
            {"id": "a1", "status": "accepted"},
            {"id": "a2", "status": "accepted"},
        ]
        broker.submit_order(Order(
            symbol="AAPL", side="BUY", shares=50, order_type="limit", limit_price=Decimal("140.00"),
        ))
        broker.submit_order(Order(
            symbol="AAPL", side="SELL", shares=50, order_type="stop", stop_price=Decimal("130.00"),
        ))
        canceled = broker.cancel_orders("AAPL", side="SELL")
        assert len(canceled) == 1
        assert canceled[0].order.side == "SELL"

    def test_cancel_market_orders_preserves_protective_sells(self, mock_client):
        broker, client = mock_client
        client.submit_order.side_effect = [
            {"id": "market-sell", "status": "accepted"},
            {"id": "stop-sell", "status": "accepted"},
        ]
        broker.submit_order(Order(symbol="AAPL", side="SELL", shares=50))
        broker.submit_order(Order(
            symbol="AAPL", side="SELL", shares=50, order_type="stop", stop_price=Decimal("130.00"),
        ))

        canceled = broker.cancel_market_orders("AAPL", side="SELL", reason="exit_sell_submitted")

        open_orders = broker.get_open_orders("AAPL")
        assert len(canceled) == 1
        assert canceled[0].id == "market-sell"
        assert canceled[0].status_reason == "exit_sell_submitted"
        assert len(open_orders) == 1
        assert open_orders[0].id == "stop-sell"
        assert open_orders[0].order.order_type == "stop"
        client.cancel_order.assert_called_once_with("market-sell")


class TestCapPendingSells:
    def test_cap_reduces_shares(self, mock_client):
        broker, client = mock_client
        client.submit_order.side_effect = [
            {"id": "a1", "status": "accepted"},
            {"id": "a2", "status": "accepted"},
        ]
        order = Order(symbol="AAPL", side="SELL", shares=100, order_type="stop", stop_price=Decimal("140.00"))
        broker.submit_order(order)
        broker.cap_pending_sells("AAPL", max_shares=50)
        open_orders = broker.get_open_orders("AAPL")
        assert len(open_orders) == 1
        assert open_orders[0].order.shares == 50

    def test_cap_zero_cancels(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AAPL", side="SELL", shares=100, order_type="stop", stop_price=Decimal("140.00"))
        broker.submit_order(order)
        broker.cap_pending_sells("AAPL", max_shares=0)
        open_orders = broker.get_open_orders("AAPL")
        assert len(open_orders) == 0


class TestLifecycleHooks:
    def test_on_bar_open_is_noop(self, mock_client):
        broker, client = mock_client
        broker.on_bar_open({}, pd.Timestamp("2026-03-11"))
        client.submit_order.assert_not_called()

    def test_on_bar_close_is_noop(self, mock_client):
        broker, client = mock_client
        broker.on_bar_close({}, pd.Timestamp("2026-03-11"))
        client.submit_order.assert_not_called()
