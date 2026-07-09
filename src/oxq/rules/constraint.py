"""Pre-trade constraint rules — blacklist, holdings limits, rebalance cadence."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, RuleResult


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
