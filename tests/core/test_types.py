"""Tests for core data types."""

from oxq.core.types import Fill, Order, Portfolio, Position


def test_order_is_frozen() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    assert order.symbol == "AAPL"
    assert order.side == "BUY"
    assert order.shares == 100
    assert order.order_type == "market"


def test_fill_is_frozen() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    fill = Fill(order=order, filled_price=150.0, filled_at="2024-01-02")
    assert fill.filled_price == 150.0
    assert fill.filled_at == "2024-01-02"
    assert fill.order is order


def test_position_is_frozen() -> None:
    pos = Position(symbol="AAPL", shares=100, avg_cost=150.0)
    assert pos.symbol == "AAPL"
    assert pos.shares == 100
    assert pos.avg_cost == 150.0


def test_portfolio_total_value_cash_only() -> None:
    portfolio = Portfolio(cash=100_000.0)
    assert portfolio.total_value({}) == 100_000.0


def test_portfolio_total_value_with_positions() -> None:
    portfolio = Portfolio(
        cash=50_000.0,
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=150.0),
            "MSFT": Position(symbol="MSFT", shares=50, avg_cost=300.0),
        },
    )
    prices = {"AAPL": 160.0, "MSFT": 310.0}
    # cash + 100*160 + 50*310 = 50000 + 16000 + 15500 = 81500
    assert portfolio.total_value(prices) == 81_500.0


def test_portfolio_total_value_missing_price() -> None:
    portfolio = Portfolio(
        cash=10_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=150.0)},
    )
    # If price not in dict, position valued at 0
    assert portfolio.total_value({}) == 10_000.0
