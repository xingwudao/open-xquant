"""Pre-trade constraint rules — blacklist, holdings limits, rebalance cadence."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

import pandas as pd

from oxq.core.types import Order, Portfolio, RuleResult


def select_max_holdings_items(
    items: list[tuple[str, float]],
    max_holdings: int,
    *,
    held_symbols: Iterable[str] = (),
    pending_buy_symbols: Iterable[str] = (),
    preserve_existing_holdings: bool = False,
) -> list[tuple[str, float]]:
    """Select capped positive targets with deterministic portfolio priority."""
    if not preserve_existing_holdings:
        return sorted(items, key=lambda item: (-item[1], item[0]))[:max_holdings]

    held = set(held_symbols)
    pending = set(pending_buy_symbols)
    target_symbols = {symbol for symbol, _weight in items}
    available_slots = max(0, max_holdings - len(pending - target_symbols))
    reserved_symbols = held | pending
    return sorted(
        items,
        key=lambda item: (item[0] not in reserved_symbols, -item[1], item[0]),
    )[:available_slots]


class BlacklistRule:
    """Blocks trading for blacklisted symbols by setting their weight to 0."""

    name = "BlacklistRule"

    def __init__(self, symbols: set[str]) -> None:
        self.symbols = symbols

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        if symbol in self.symbols:
            return RuleResult(
                weights={symbol: 0.0},
                reason=f"{symbol} is blacklisted",
            )
        return RuleResult()


class MaxHoldingsRule:
    """Blocks new positions when portfolio is at max holdings limit."""

    name = "MaxHoldingsRule"

    def __init__(self, max_holdings: int) -> None:
        self.max_holdings = max_holdings

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        if symbol in portfolio.positions:
            return RuleResult()
        if len(portfolio.positions) >= self.max_holdings:
            return RuleResult(
                weights={symbol: 0.0},
                reason=f"max holdings {self.max_holdings} reached, blocking {symbol}",
            )
        return RuleResult()

    def evaluate_batch(
        self,
        target_weights: dict[str, float],
        portfolio: Portfolio,
        pending_orders: list[Order] | None = None,
    ) -> RuleResult:
        """Return complete target weights capped across the order batch."""
        items = [
            (symbol, float(weight))
            for symbol, weight in target_weights.items()
            if symbol != "CASH" and float(weight) > 0.0
        ]
        pending_buy_symbols = {
            order.symbol
            for order in pending_orders or []
            if order.order_type == "market" and order.side == "BUY"
        }
        selected_items = select_max_holdings_items(
            items,
            self.max_holdings,
            held_symbols=portfolio.positions,
            pending_buy_symbols=pending_buy_symbols,
            preserve_existing_holdings=True,
        )
        selected = {symbol for symbol, _weight in selected_items}
        blocked = sorted(symbol for symbol, _weight in items if symbol not in selected)
        if not blocked:
            return RuleResult()
        constrained = dict(selected_items)
        invested = sum(constrained.values())
        constrained["CASH"] = max(float(target_weights.get("CASH", 0.0)), 1.0 - invested)
        return RuleResult(
            weights=constrained,
            reason=f"max holdings {self.max_holdings} reached",
        )


class RebalanceFrequencyRule:
    """Freezes trading within a rebalance interval.

    Counts trading days (bars processed), not calendar days.
    Allows trading on the first bar, then blocks until interval_days
    trading days have passed.
    """

    name = "RebalanceFrequencyRule"

    def __init__(self, interval_days: int = 5) -> None:
        self.interval_days = interval_days
        self._bars_since_rebalance: int | None = None
        self._last_evaluated_date: pd.Timestamp | None = None
        self._last_rebalance_date: pd.Timestamp | None = None

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        bar_date = row.name if hasattr(row, "name") else None
        if bar_date is None:
            return RuleResult()

        # First bar ever: allow trading
        if self._bars_since_rebalance is None:
            self._bars_since_rebalance = 0
            self._last_evaluated_date = bar_date
            self._last_rebalance_date = bar_date
            return RuleResult()

        # Count new trading days (avoid double-counting for multi-symbol bars)
        if bar_date != self._last_evaluated_date:
            self._bars_since_rebalance += 1
            self._last_evaluated_date = bar_date

        # Allow all symbols on a rebalance date
        if bar_date == self._last_rebalance_date:
            return RuleResult()

        if self._bars_since_rebalance >= self.interval_days:
            self._bars_since_rebalance = 0
            self._last_rebalance_date = bar_date
            return RuleResult()

        return RuleResult(
            hold=True,
            reason=(
                f"rebalance interval: {self._bars_since_rebalance} bars"
                f" < {self.interval_days} bars"
            ),
        )


class CalendarRebalanceRule:
    """Allow rebalancing on the first observed bar of a calendar period."""

    name = "CalendarRebalanceRule"

    def __init__(self, schedule: str = "month_start") -> None:
        if schedule not in {"week_start", "month_start"}:
            raise ValueError("schedule must be week_start or month_start")
        self.schedule = schedule
        self._last_evaluated_date: pd.Timestamp | None = None
        self._allowed_date: pd.Timestamp | None = None
        self._allowed_period: tuple[int, ...] | None = None

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        del symbol, portfolio, prices
        bar_date = row.name if hasattr(row, "name") else None
        if bar_date is None:
            return RuleResult()
        timestamp = pd.Timestamp(bar_date)
        period = self._period_key(timestamp)

        if timestamp != self._last_evaluated_date:
            self._last_evaluated_date = timestamp
            if period != self._allowed_period:
                self._allowed_period = period
                self._allowed_date = timestamp

        if timestamp == self._allowed_date:
            return RuleResult()

        return RuleResult(
            hold=True,
            reason=f"{self.schedule} rebalance only",
        )

    def _period_key(self, timestamp: pd.Timestamp) -> tuple[int, ...]:
        if self.schedule == "week_start":
            iso = timestamp.isocalendar()
            return int(iso.year), int(iso.week)
        return int(timestamp.year), int(timestamp.month)
