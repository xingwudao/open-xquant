"""Tests for core data types."""

from decimal import Decimal

from oxq.core.types import Constraint, Fill, Order, Portfolio, Position, Rule, RuleResult, Signal


def test_order_is_frozen() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    assert order.symbol == "AAPL"
    assert order.side == "BUY"
    assert order.shares == 100
    assert order.order_type == "market"


def test_order_stop() -> None:
    order = Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("142.50"),
    )
    assert order.order_type == "stop"
    assert order.stop_price == Decimal("142.50")


def test_order_limit() -> None:
    order = Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="limit", limit_price=Decimal("185"),
    )
    assert order.order_type == "limit"
    assert order.limit_price == Decimal("185")


def test_order_trailing_stop() -> None:
    order = Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="trailing_stop", trail_pct=0.05,
    )
    assert order.order_type == "trailing_stop"
    assert order.trail_pct == 0.05


def test_fill_is_frozen() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    fill = Fill(order=order, filled_price=Decimal("150"), filled_at="2024-01-02")
    assert fill.filled_price == Decimal("150")
    assert fill.filled_at == "2024-01-02"
    assert fill.order is order
    assert fill.fee == Decimal("0")


def test_fill_with_fee() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    fill = Fill(
        order=order, filled_price=Decimal("150"),
        filled_at="2024-01-02", fee=Decimal("5"),
    )
    assert fill.fee == Decimal("5")


def test_position_is_frozen() -> None:
    pos = Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))
    assert pos.symbol == "AAPL"
    assert pos.shares == 100
    assert pos.avg_cost == Decimal("150")


def test_portfolio_total_value_cash_only() -> None:
    portfolio = Portfolio(cash=Decimal("100000"))
    assert portfolio.total_value({}) == Decimal("100000")


def test_portfolio_total_value_with_positions() -> None:
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150")),
            "MSFT": Position(symbol="MSFT", shares=50, avg_cost=Decimal("300")),
        },
    )
    prices = {"AAPL": Decimal("160"), "MSFT": Decimal("310")}
    # cash + 100*160 + 50*310 = 50000 + 16000 + 15500 = 81500
    assert portfolio.total_value(prices) == Decimal("81500")


def test_portfolio_total_value_missing_price() -> None:
    portfolio = Portfolio(
        cash=Decimal("10000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    # If price not in dict, position valued at 0
    assert portfolio.total_value({}) == Decimal("10000")


def test_rule_result_defaults():
    r = RuleResult()
    assert r.weights is None
    assert r.constraints is None
    assert r.target_positions is None
    assert r.hold is False
    assert r.reason == ""


def test_rule_result_pre_trade():
    r = RuleResult(
        weights={"AAPL": 0.0, "GOOG": 0.5},
        constraints={"AAPL": Constraint(max_shares=1000)},
        reason="blacklist AAPL",
    )
    assert r.weights["AAPL"] == 0.0
    assert r.constraints["AAPL"].max_shares == 1000


def test_rule_result_post_trade():
    r = RuleResult(
        target_positions={"AAPL": 0.0},
        reason="stop loss triggered",
    )
    assert r.target_positions["AAPL"] == 0.0


def test_constraint_defaults():
    c = Constraint()
    assert c.max_shares is None
    assert c.max_value is None


def test_rule_protocol_returns_rule_result():
    class MyRule:
        name = "test"
        def evaluate(self, symbol, row, portfolio, prices=None):
            return RuleResult(reason="test")
    assert isinstance(MyRule(), Rule)


def test_portfolio_optimizer_protocol():
    from oxq.core.types import PortfolioOptimizer

    class MyOptimizer:
        name = "test"

        def optimize(self, signals, indicators):
            return {"AAPL": 0.5, "CASH": 0.5}

    assert isinstance(MyOptimizer(), PortfolioOptimizer)


def test_signal_protocol_per_symbol():
    """Signal.compute takes a single DataFrame, returns a single Series."""
    class MySignal:
        name = "test"
        def compute(self, mktdata, **params):
            return mktdata["close"] > 100
    assert isinstance(MySignal(), Signal)


def test_strategy_new_shape_excludes_universe():
    from oxq.core.strategy import Strategy

    class FakeOptimizer:
        name = "fake"
        def optimize(self, signals, indicators):
            return {"CASH": 1.0}

    class FakeRule:
        name = "rule"
        def evaluate(self, symbol, row, portfolio, prices=None):
            return RuleResult(reason="test")

    s = Strategy(
        name="test",
        signals={},
        portfolio=FakeOptimizer(),
        rules=[FakeRule()],
    )
    assert s.name == "test"
    assert s.portfolio.name == "fake"
    assert len(s.rules) == 1
    assert "universe" not in s.__dataclass_fields__
    assert "_legacy_universe" not in s.__dataclass_fields__
    assert not hasattr(s, "entry_rules")
    assert not hasattr(s, "indicators")


def test_strategy_legacy_universe_adapter():
    from oxq.core.strategy import Strategy
    from oxq.universe.static import StaticUniverse

    class FakeOptimizer:
        name = "fake"
        def optimize(self, signals, indicators):
            return {"CASH": 1.0}

    universe = StaticUniverse(("AAPL",))
    s = Strategy(
        name="test",
        universe=universe,
        signals={},
        portfolio=FakeOptimizer(),
    )
    assert s.universe is universe


def test_strategy_legacy_positional_constructor_preserves_universe_shape():
    from oxq.core.strategy import Strategy
    from oxq.universe.static import StaticUniverse

    class FakeOptimizer:
        name = "fake"

        def optimize(self, signals, indicators):
            return {"CASH": 1.0}

    universe = StaticUniverse(("AAPL",))
    optimizer = FakeOptimizer()

    s = Strategy(
        "legacy",
        universe,
        {},
        optimizer,
        "old positional hypothesis",
        {"total_return": {"min": 0.05}},
        ["SPY"],
    )

    assert s.name == "legacy"
    assert s.universe is universe
    assert s.signals == {}
    assert s.portfolio is optimizer
    assert s.rules == []
    assert s.hypothesis == "old positional hypothesis"
    assert s.objectives == {"total_return": {"min": 0.05}}
    assert s.benchmarks == ["SPY"]
    assert "universe" not in s.__dataclass_fields__
