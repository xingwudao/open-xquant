"""Backtest engine — executes the 4-phase pipeline."""

from __future__ import annotations

import pandas as pd

from oxq.backtest.analytics import BacktestResult
from oxq.backtest.broker import SimBroker
from oxq.core.strategy import Strategy
from oxq.core.types import Fill, Portfolio, Position
from oxq.data.providers import MarketDataProvider


class BacktestEngine:
    """Event-driven backtesting engine.

    Executes the Universe → Indicator → Signal → Rule pipeline.
    Supports partial execution via *run_through*.
    """

    def run(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker: SimBroker,
        start: str,
        end: str,
        initial_cash: float = 100_000.0,
        run_through: str | None = None,
    ) -> BacktestResult:
        """Run the full backtest pipeline.

        Parameters
        ----------
        strategy : Strategy
            The strategy definition.
        market : MarketDataProvider
            Data provider for loading bars.
        broker : SimBroker
            Simulated broker (OrderRouter + FillReceiver).
        start, end : str
            Date range for the backtest.
        initial_cash : float
            Starting cash.
        run_through : str | None
            Stop after this phase: ``"indicator"`` or ``"signal"``.
            ``None`` runs the full pipeline including rules.
        """
        portfolio = Portfolio(cash=initial_cash)

        # ── Phase 0: Universe ────────────────────────────────────────
        universe = strategy.universe.get_universe(as_of_date=end)

        mktdata: dict[str, pd.DataFrame] = {}
        for symbol in universe.symbols:
            mktdata[symbol] = market.get_bars(symbol, start, end).copy()

        # ── Phase 1: Indicator (vectorized, per symbol) ──────────────
        for symbol in universe.symbols:
            for ind_name, (indicator, params) in strategy.indicators.items():
                mktdata[symbol][ind_name] = indicator.compute(
                    mktdata[symbol], **params,
                )

        if run_through == "indicator":
            return BacktestResult(
                portfolio=portfolio, trades=[], equity_curve=[], mktdata=mktdata,
            )

        # ── Phase 2: Signal (vectorized, cross-sectional) ────────────
        for sig_name, (signal, params) in strategy.signals.items():
            results = signal.compute(mktdata, **params)
            for symbol, series in results.items():
                mktdata[symbol][sig_name] = series

        if run_through == "signal":
            return BacktestResult(
                portfolio=portfolio, trades=[], equity_curve=[], mktdata=mktdata,
            )

        # ── Phase 3: Rule (bar-by-bar state machine) ─────────────────
        dates = mktdata[universe.symbols[0]].index
        trades: list[Fill] = []
        equity_curve: list[tuple[object, float]] = []

        for date in dates:
            # Exit rules first (higher priority)
            for rule in strategy.exit_rules:
                for symbol in universe.symbols:
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        broker.submit_order(order)

            # Entry rules second (lower priority)
            for rule in strategy.entry_rules:
                for symbol in universe.symbols:
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        broker.submit_order(order)

            # Fill orders at this bar's close
            broker.fill_pending_orders(mktdata, date)

            # Process fills → update portfolio
            for fill in broker.get_fills():
                _apply_fill(portfolio, fill)
                trades.append(fill)

            # Record equity curve
            prices = {
                s: float(mktdata[s].loc[date, "close"])
                for s in universe.symbols
            }
            equity_curve.append((date, portfolio.total_value(prices)))

        return BacktestResult(
            portfolio=portfolio,
            trades=trades,
            equity_curve=equity_curve,
            mktdata=mktdata,
        )


def _apply_fill(portfolio: Portfolio, fill: Fill) -> None:
    """Update portfolio state based on a fill."""
    order = fill.order
    symbol = order.symbol
    cost = fill.filled_price * order.shares

    if order.side == "BUY":
        portfolio.cash -= cost
        if symbol in portfolio.positions:
            old = portfolio.positions[symbol]
            total_shares = old.shares + order.shares
            total_cost = old.avg_cost * old.shares + cost
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                avg_cost=total_cost / total_shares,
            )
        else:
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                shares=order.shares,
                avg_cost=fill.filled_price,
            )
    elif order.side == "SELL":
        portfolio.cash += cost
        if symbol in portfolio.positions:
            old = portfolio.positions[symbol]
            remaining = old.shares - order.shares
            if remaining <= 0:
                del portfolio.positions[symbol]
            else:
                portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    shares=remaining,
                    avg_cost=old.avg_cost,
                )
