"""OrderGenerator — generates trade plans from target weights."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from oxq.core.types import Order, Position


@dataclass(frozen=True)
class PlannedOrder:
    """A trade order with context for human review.

    Attributes
    ----------
    order : Order
        The actual oxq Order object.
    current_shares : int
        Current position size.
    target_shares : int
        Desired position size after execution.
    current_weight : Decimal
        Current portfolio weight for this symbol.
    target_weight : Decimal
        Target portfolio weight.
    estimated_amount : Decimal
        Estimated trade value (shares * price).
    """

    order: Order
    current_shares: int
    target_shares: int
    current_weight: Decimal
    target_weight: Decimal
    estimated_amount: Decimal


def generate_orders(
    target_weights: dict[str, Decimal],
    positions: dict[str, Position],
    prices: dict[str, Decimal],
    total_capital: Decimal,
    lot_size: int = 1,
    currency: str = "CNY",
    pending_orders: list[Order] | None = None,
    buy_cost_estimator: Callable[[str, Decimal, int], Decimal] | None = None,
    sell_proceeds_estimator: Callable[[str, Decimal, int], Decimal] | None = None,
    buying_power: Decimal | None = None,
) -> list[PlannedOrder]:
    """Generate a trade plan from target weights.

    Computes the difference between target and current positions,
    producing BUY/SELL orders with context for human review.

    Parameters
    ----------
    target_weights : dict[str, Decimal]
        Target portfolio weights keyed by symbol. Symbols in positions
        but not in target_weights will be fully sold.
    positions : dict[str, Position]
        Current portfolio positions.
    prices : dict[str, Decimal]
        Current market prices per symbol.
    total_capital : Decimal
        Total portfolio value (cash + positions market value).
    lot_size : int
        Minimum trade unit. Default 1 (US stocks). Use 100 for A-shares.
    pending_orders : list[Order] or None
        Open orders already submitted but not filled.
    buy_cost_estimator : Callable or None
        Function returning total estimated cash needed for a BUY order.
    sell_proceeds_estimator : Callable or None
        Function returning estimated net cash credited by a SELL order.
    buying_power : Decimal or None
        Optional cash budget for new buys. Defaults to total capital when omitted.

    Returns
    -------
    list[PlannedOrder]
        Ordered list of planned trades with context.
    """
    planned: list[PlannedOrder] = []
    pending_delta: dict[str, int] = {}
    pending_sell_shares: dict[str, int] = {}
    pending_buy_notional = Decimal("0")
    pending_sell_proceeds = Decimal("0")
    estimate_cost = buy_cost_estimator or _default_buy_cost_estimator
    estimate_proceeds = sell_proceeds_estimator or _default_sell_proceeds_estimator
    for order in pending_orders or []:
        if order.order_type != "market":
            continue
        signed_shares = order.shares if order.side == "BUY" else -order.shares
        pending_delta[order.symbol] = pending_delta.get(order.symbol, 0) + signed_shares
        if order.side == "BUY" and order.symbol in prices:
            pending_buy_notional += estimate_cost(order.symbol, prices[order.symbol], order.shares)
        elif order.side == "SELL":
            pending_sell_shares[order.symbol] = pending_sell_shares.get(order.symbol, 0) + order.shares
            if order.symbol in prices:
                pending_sell_proceeds += max(
                    Decimal("0"),
                    estimate_proceeds(order.symbol, prices[order.symbol], order.shares),
                )
    reserved_capital = max(Decimal("0"), pending_buy_notional)
    buy_budget = total_capital if buying_power is None else buying_power
    if buying_power is None:
        remaining_buy_budget = max(Decimal("0"), buy_budget - reserved_capital)
    else:
        remaining_buy_budget = max(Decimal("0"), buy_budget - reserved_capital + pending_sell_proceeds)

    # All symbols: union of targets and current positions.
    # Pending-only symbols are already represented by open orders and should
    # not generate compensating trades unless a new target or position exists.
    all_symbols = set(target_weights.keys()) | set(positions.keys())

    for side_pass in ("SELL", "BUY"):
        for symbol in sorted(all_symbols):
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            target_weight = target_weights.get(symbol, Decimal("0"))
            current_shares = positions[symbol].shares if symbol in positions else 0
            projected_shares = current_shares + pending_delta.get(symbol, 0)
            current_value = price * current_shares
            current_weight = current_value / total_capital if total_capital > 0 else Decimal("0")

            # Compute target shares with lot_size rounding
            raw_target = total_capital * target_weight / price
            target_shares = int(raw_target / lot_size) * lot_size

            buy_delta = target_shares - projected_shares
            sellable_after_pending_sells = max(0, current_shares - pending_sell_shares.get(symbol, 0))
            sell_delta = min(max(0, projected_shares - target_shares), sellable_after_pending_sells)
            if side_pass == "SELL":
                if sell_delta <= 0:
                    continue
                side = "SELL"
                shares = min(sell_delta, current_shares)
                if shares <= 0:
                    continue
                planned_target_shares = current_shares - shares
                if buying_power is not None:
                    remaining_buy_budget += max(Decimal("0"), estimate_proceeds(symbol, price, shares))
            else:
                if buy_delta <= 0:
                    continue
                side = "BUY"
                shares = buy_delta
                affordable_shares = int((remaining_buy_budget / price) / lot_size) * lot_size
                shares = min(shares, affordable_shares)
                while shares > 0 and estimate_cost(symbol, price, shares) > remaining_buy_budget:
                    shares -= lot_size
                if shares <= 0:
                    continue
                remaining_buy_budget -= estimate_cost(symbol, price, shares)
                planned_target_shares = projected_shares + shares

            planned.append(
                PlannedOrder(
                    order=Order(symbol=symbol, side=side, shares=shares, currency=currency),
                    current_shares=current_shares,
                    target_shares=planned_target_shares,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    estimated_amount=price * shares,
                )
            )

    return sorted(planned, key=lambda item: 0 if item.order.side == "SELL" else 1)


def _default_buy_cost_estimator(_symbol: str, price: Decimal, shares: int) -> Decimal:
    return price * shares


def _default_sell_proceeds_estimator(_symbol: str, price: Decimal, shares: int) -> Decimal:
    return price * shares
