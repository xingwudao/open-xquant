"""Transaction fee models for simulated trading."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from oxq.core.types import Order


@runtime_checkable
class FeeModel(Protocol):
    """Protocol for transaction fee calculation.

    Implementations compute the total fee for a single order fill.
    The fee is recorded on the Fill object and deducted from
    portfolio cash by the engine.

    The fee should always be returned as a non-negative Decimal.
    """

    def calculate(self, order: Order, fill_price: Decimal) -> Decimal:
        """Calculate the transaction fee for a filled order.

        Parameters
        ----------
        order : Order
            The order being filled.
        fill_price : Decimal
            The execution price (after slippage adjustment).

        Returns
        -------
        Decimal
            Total fee amount (non-negative).
        """
        ...


class PercentageFee:
    """Fee calculated as a percentage of trade value, with a minimum.

    Computes fee as ``fill_price * shares * rate``, floored at
    ``min_fee``. This models typical brokerage commission structures.

    Parameters
    ----------
    rate : Decimal
        Fee rate as a decimal fraction of trade value.
        Default is Decimal("0.001") (0.1%).
    min_fee : Decimal
        Minimum fee per trade. Default is Decimal("5").

    Examples
    --------
    >>> broker = SimBroker(fee_model=PercentageFee())
    """

    def __init__(
        self,
        rate: Decimal = Decimal("0.001"),
        min_fee: Decimal = Decimal("5"),
    ) -> None:
        self.rate = rate
        self.min_fee = min_fee

    def calculate(self, order: Order, fill_price: Decimal) -> Decimal:
        """Calculate fee as percentage of trade value, floored at min_fee."""
        trade_value = fill_price * order.shares
        fee = trade_value * self.rate
        return max(fee, self.min_fee)


class SideAwarePercentageFee:
    """Fee model with buy/sell rates plus optional sell-side tax.

    This models markets such as A-shares where brokerage commission applies to
    both sides, while stamp tax applies only to sells.
    """

    def __init__(
        self,
        buy_rate: Decimal = Decimal("0.001"),
        sell_rate: Decimal = Decimal("0.001"),
        sell_tax_rate: Decimal = Decimal("0"),
        min_fee: Decimal = Decimal("0"),
    ) -> None:
        self.buy_rate = buy_rate
        self.sell_rate = sell_rate
        self.sell_tax_rate = sell_tax_rate
        self.min_fee = min_fee

    def calculate(self, order: Order, fill_price: Decimal) -> Decimal:
        """Calculate side-aware percentage fee plus sell-side tax."""
        trade_value = fill_price * order.shares
        if order.side == "SELL":
            commission = max(trade_value * self.sell_rate, self.min_fee)
            return commission + trade_value * self.sell_tax_rate
        return max(trade_value * self.buy_rate, self.min_fee)
