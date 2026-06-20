"""Tests for OrderGenerator — generates trade plans from target weights."""

from __future__ import annotations

from decimal import Decimal

from oxq.core.types import Order, Position
from oxq.trade.order_generator import generate_orders


class TestGenerateOrders:
    def test_basic_buy(self):
        """Target weight > 0, no existing position -> BUY."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 1
        assert result[0].order.symbol == "AAPL"
        assert result[0].order.side == "BUY"
        assert result[0].order.shares == 50
        assert result[0].target_shares == 50
        assert result[0].current_shares == 0
        assert result[0].target_weight == Decimal("0.5")
        assert result[0].estimated_amount == Decimal("5000")

    def test_basic_sell(self):
        """Target weight < current weight -> SELL delta."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.3")},
            positions={"AAPL": Position(symbol="AAPL", shares=50, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 1
        assert result[0].order.side == "SELL"
        assert result[0].order.shares == 20  # 50 - 30

    def test_no_change(self):
        """Target matches current -> no order."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={"AAPL": Position(symbol="AAPL", shares=50, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 0

    def test_clear_position(self):
        """Symbol in positions but not in target_weights -> full SELL."""
        result = generate_orders(
            target_weights={"GOOG": Decimal("0.5")},
            positions={"AAPL": Position(symbol="AAPL", shares=50, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100"), "GOOG": Decimal("200")},
            total_capital=Decimal("10000"),
        )
        sells = [r for r in result if r.order.side == "SELL"]
        assert len(sells) == 1
        assert sells[0].order.symbol == "AAPL"
        assert sells[0].order.shares == 50

    def test_rebalance_orders_sell_before_buy(self):
        result = generate_orders(
            target_weights={"AAA": Decimal("0.5")},
            positions={"ZZZ": Position(symbol="ZZZ", shares=100, avg_cost=Decimal("10"))},
            prices={"AAA": Decimal("10"), "ZZZ": Decimal("10")},
            total_capital=Decimal("1000"),
        )

        assert [item.order.side for item in result] == ["SELL", "BUY"]
        assert [item.order.symbol for item in result] == ["ZZZ", "AAA"]

    def test_lot_size(self):
        """lot_size=100 rounds down to nearest lot."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
            lot_size=100,
        )
        assert len(result) == 0  # 50 shares < 100 lot_size -> rounds to 0

    def test_lot_size_rounds_down(self):
        """lot_size=100, enough capital for 1 lot."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("1.0")},
            positions={},
            prices={"AAPL": Decimal("50")},
            total_capital=Decimal("10000"),
            lot_size=100,
        )
        assert result[0].order.shares == 200  # floor(200 / 100) * 100

    def test_multiple_symbols(self):
        """Multiple symbols with different weights."""
        result = generate_orders(
            target_weights={
                "AAPL": Decimal("0.4"),
                "GOOG": Decimal("0.3"),
            },
            positions={},
            prices={"AAPL": Decimal("100"), "GOOG": Decimal("150")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 2
        by_sym = {r.order.symbol: r for r in result}
        assert by_sym["AAPL"].order.shares == 40
        assert by_sym["GOOG"].order.shares == 20

    def test_market_order_type(self):
        """All generated orders use market order type."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert result[0].order.order_type == "market"

    def test_planned_order_context(self):
        """PlannedOrder includes current_weight for review."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.6")},
            positions={"AAPL": Position(symbol="AAPL", shares=30, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert result[0].current_weight == Decimal("0.3")
        assert result[0].current_shares == 30

    def test_pending_protective_orders_do_not_reduce_projected_position(self):
        """Protective non-market orders should not trigger compensating buys."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("1.0")},
            positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("10"))},
            prices={"AAPL": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[
                Order(
                    symbol="AAPL",
                    side="SELL",
                    shares=50,
                    order_type="stop",
                    stop_price=Decimal("9"),
                )
            ],
        )

        assert result == []

    def test_pending_market_buys_reserve_capital_for_new_orders(self):
        """Open market buys reserve cash while other targets are planned."""
        result = generate_orders(
            target_weights={"MSFT": Decimal("1.0")},
            positions={},
            prices={"AAPL": Decimal("10"), "MSFT": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[
                Order(
                    symbol="AAPL",
                    side="BUY",
                    shares=50,
                )
            ],
        )

        assert len(result) == 1
        assert result[0].order.symbol == "MSFT"
        assert result[0].order.shares == 50

    def test_pending_market_buys_reserve_estimated_execution_costs(self):
        result = generate_orders(
            target_weights={"MSFT": Decimal("1.0")},
            positions={},
            prices={"AAPL": Decimal("10"), "MSFT": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[Order(symbol="AAPL", side="BUY", shares=50)],
            buy_cost_estimator=lambda _symbol, price, shares: price * shares * Decimal("2"),
        )

        assert result == []

    def test_pending_market_buys_do_not_double_discount_other_target_weights(self):
        """Target weights stay anchored to total capital before budget capping."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5"), "MSFT": Decimal("0.5")},
            positions={},
            prices={"AAPL": Decimal("10"), "MSFT": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[
                Order(
                    symbol="AAPL",
                    side="BUY",
                    shares=50,
                )
            ],
        )

        assert len(result) == 1
        assert result[0].order.symbol == "MSFT"
        assert result[0].order.shares == 50

    def test_pending_market_buy_does_not_create_sell_without_position(self):
        """Unfilled buys are not sellable inventory when targets fall."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0")},
            positions={},
            prices={"AAPL": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[
                Order(
                    symbol="AAPL",
                    side="BUY",
                    shares=100,
                )
            ],
        )

        assert result == []

    def test_pending_market_buy_increases_sell_needed_when_target_drops(self):
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("10"))},
            prices={"AAPL": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[Order(symbol="AAPL", side="BUY", shares=100)],
        )

        assert len(result) == 1
        assert result[0].order.side == "SELL"
        assert result[0].order.shares == 100

    def test_pending_market_sell_does_not_create_duplicate_exit(self):
        result = generate_orders(
            target_weights={"AAPL": Decimal("0")},
            positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("10"))},
            prices={"AAPL": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[Order(symbol="AAPL", side="SELL", shares=100)],
        )

        assert result == []

    def test_buy_orders_are_capped_by_estimated_execution_costs(self):
        result = generate_orders(
            target_weights={"AAPL": Decimal("1.0")},
            positions={},
            prices={"AAPL": Decimal("10")},
            total_capital=Decimal("1000"),
            buy_cost_estimator=lambda _symbol, price, shares: (price * shares * Decimal("1.01")) + Decimal("1"),
        )

        assert len(result) == 1
        assert result[0].order.shares == 98

    def test_explicit_buying_power_uses_net_planned_sell_proceeds(self):
        result = generate_orders(
            target_weights={"CCC": Decimal("1.0")},
            positions={"AAA": Position(symbol="AAA", shares=100, avg_cost=Decimal("10"))},
            prices={"AAA": Decimal("10"), "CCC": Decimal("10")},
            total_capital=Decimal("1000"),
            buying_power=Decimal("0"),
            sell_proceeds_estimator=lambda _symbol, price, shares: price * shares - Decimal("10"),
        )

        assert [(item.order.symbol, item.order.side, item.order.shares) for item in result] == [
            ("AAA", "SELL", 100),
            ("CCC", "BUY", 99),
        ]

    def test_explicit_buying_power_credits_pending_sell_proceeds(self):
        result = generate_orders(
            target_weights={"CCC": Decimal("1.0")},
            positions={"AAA": Position(symbol="AAA", shares=100, avg_cost=Decimal("10"))},
            prices={"AAA": Decimal("10"), "CCC": Decimal("10")},
            total_capital=Decimal("1000"),
            pending_orders=[Order(symbol="AAA", side="SELL", shares=100)],
            buying_power=Decimal("0"),
            sell_proceeds_estimator=lambda _symbol, price, shares: price * shares - Decimal("10"),
        )

        assert [(item.order.symbol, item.order.side, item.order.shares) for item in result] == [
            ("CCC", "BUY", 99),
        ]
