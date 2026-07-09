"""Tests for FeeModel."""

from decimal import Decimal

from oxq.core.types import Order
from oxq.trade.fees import FeeModel, PercentageFee, SideAwarePercentageFee


def test_percentage_fee_satisfies_protocol() -> None:
    assert isinstance(PercentageFee(), FeeModel)


def test_percentage_fee_basic() -> None:
    fee_model = PercentageFee(rate=Decimal("0.001"), min_fee=Decimal("5"))
    order = Order(symbol="AAPL", side="BUY", shares=100)
    # 150 * 100 * 0.001 = 15
    fee = fee_model.calculate(order, Decimal("150"))
    assert fee == Decimal("15.0")


def test_percentage_fee_min_fee() -> None:
    fee_model = PercentageFee(rate=Decimal("0.001"), min_fee=Decimal("5"))
    order = Order(symbol="AAPL", side="BUY", shares=10)
    # 150 * 10 * 0.001 = 1.5, below min_fee -> 5
    fee = fee_model.calculate(order, Decimal("150"))
    assert fee == Decimal("5")


def test_percentage_fee_sell() -> None:
    fee_model = PercentageFee(rate=Decimal("0.001"), min_fee=Decimal("5"))
    order = Order(symbol="AAPL", side="SELL", shares=100)
    fee = fee_model.calculate(order, Decimal("160"))
    assert fee == Decimal("16.0")


def test_side_aware_percentage_fee_applies_sell_tax_only_to_sells() -> None:
    fee_model = SideAwarePercentageFee(
        buy_rate=Decimal("0.0003"),
        sell_rate=Decimal("0.0003"),
        sell_tax_rate=Decimal("0.001"),
        min_fee=Decimal("0"),
    )

    buy = Order(symbol="600519.SH", side="BUY", shares=100)
    sell = Order(symbol="600519.SH", side="SELL", shares=100)

    assert fee_model.calculate(buy, Decimal("100")) == Decimal("3.0000")
    assert fee_model.calculate(sell, Decimal("100")) == Decimal("13.0000")


def test_side_aware_percentage_fee_applies_sell_min_commission_before_tax() -> None:
    fee_model = SideAwarePercentageFee(
        buy_rate=Decimal("0.0001"),
        sell_rate=Decimal("0.0001"),
        sell_tax_rate=Decimal("0.001"),
        min_fee=Decimal("5"),
    )
    sell = Order(symbol="600519.SH", side="SELL", shares=100)

    assert fee_model.calculate(sell, Decimal("100")) == Decimal("15.0000")
