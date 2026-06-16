"""Universal strategy execution engine.

Executes the new pipeline: Universe -> Indicator -> Signal -> Optimize -> Rules.
Provider-agnostic — the same engine serves backtest, paper trading,
and live trading.  The difference is which providers you plug in.
"""

from __future__ import annotations

import logging
import time as _time
from decimal import Decimal
from typing import Literal

import pandas as pd

from oxq.core.strategy import Strategy
from oxq.core.types import BarSnapshot, Broker, Fill, Order, Portfolio, Position, PositionSnapshot, Rule
from oxq.data.providers import MarketDataProvider
from oxq.observe.tracer import DefaultTracer
from oxq.portfolio.analytics import RunResult
from oxq.trade.order_generator import generate_orders

logger = logging.getLogger(__name__)


class Engine:
    """Universal strategy execution engine.

    Executes the pipeline: Universe -> Indicator -> Signal -> Optimize -> Rules.
    Provider-agnostic: the same engine serves backtest, paper trading,
    and live trading — the difference is which providers you plug in.

    Example — backtest mode::

        engine = Engine()
        result = engine.run(
            strategy,
            market=LocalMarketDataProvider(),
            broker=sim_broker,
            start="2023-01-01",
            end="2024-12-31",
        )

    Example — step-by-step mode (live trading)::

        engine = Engine()
        engine.setup(strategy=strategy, market=market, broker=broker,
                     start="2024-01-01", end="2024-12-31")
        for date in engine.dates:
            engine.step(date)
        result = engine.result
    """

    def setup(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker: Broker,
        start: str,
        end: str,
        initial_cash: float = 100_000.0,
        run_through: Literal["indicator", "signal"] | None = None,
        tracer: DefaultTracer | None = None,
        rules: list[Rule] | None = None,
        lot_size: int = 1,
        cash_annual_return: float = 0.0,
        data_start: str | None = None,
    ) -> None:
        """Initialize engine state and run vectorized phases.

        Parameters
        ----------
        strategy : Strategy
            The strategy definition.
        market : MarketDataProvider
            Data provider for loading bars.
        broker : Broker
            Unified broker interface (order routing, fills, lifecycle).
        start, end : str
            Date range for trading.
        initial_cash : float
            Starting cash.
        run_through : str | None
            Stop after this phase: ``"indicator"`` or ``"signal"``.
        rules : list[Rule] | None
            Optional list of Rule instances for pre/post-trade evaluation.
        lot_size : int
            Minimum trade unit (e.g. 100 for A-shares).
        cash_annual_return : float
            Annual cash interest rate.
        data_start : str | None
            Start date for loading market data (for indicator warmup).
            If None, uses ``start``.
        """
        self._strategy = strategy
        self._broker = broker
        self._tracer = tracer
        self._rules: list[Rule] = rules or []
        self._lot_size = lot_size
        self._cash_annual_return = cash_annual_return
        if tracer:
            tracer.on_run_start(
                strategy_name=strategy.name,
                config={"start": start, "end": end, "initial_cash": initial_cash},
            )

        # -- Phase 0: Universe ------------------------------------------------
        self._universe = strategy.universe.get_universe(as_of_date=end)

        self._mktdata: dict[str, pd.DataFrame] = {}
        for symbol in self._universe.symbols:
            load_start = data_start or start
            self._mktdata[symbol] = market.get_bars(symbol, load_start, end).copy()
            # Require tz-aware index — data providers must supply timezone
            if hasattr(self._mktdata[symbol].index, "tz") and self._mktdata[symbol].index.tz is None:
                msg = (
                    f"Market data for '{symbol}' has no timezone on index. "
                    f"Data providers must supply tz-aware DatetimeIndex."
                )
                raise ValueError(msg)
            if "timezone" not in self._mktdata[symbol].attrs:
                tz = getattr(self._mktdata[symbol].index, "tz", None)
                if tz is not None:
                    self._mktdata[symbol].attrs["timezone"] = str(tz)
            if "currency" not in self._mktdata[symbol].attrs:
                self._mktdata[symbol].attrs["currency"] = "CNY"

        # -- Currency validation: all symbols must share same currency (v1) --
        currencies = {
            sym: self._mktdata[sym].attrs["currency"]
            for sym in self._universe.symbols
        }
        unique_currencies = set(currencies.values())
        if len(unique_currencies) > 1:
            msg = (
                f"Mixed currencies detected: {currencies}. "
                f"v1 requires all symbols to share the same currency."
            )
            raise ValueError(msg)
        run_currency = next(iter(unique_currencies))

        self._portfolio = Portfolio(
            cash=Decimal(str(initial_cash)), currency=run_currency,
        )

        # Set _step_start with matching timezone from the loaded data
        first_sym = self._universe.symbols[0]
        data_tz = getattr(self._mktdata[first_sym].index, "tz", None)
        self._step_start = pd.Timestamp(start, tz=data_tz)

        # -- Benchmark data (recorded for post-run analysis) ----------
        self._benchmark_prices: dict[str, pd.Series] = {}
        for bench_symbol in strategy.benchmarks:
            bench_bars = market.get_bars(bench_symbol, start, end)
            self._benchmark_prices[bench_symbol] = bench_bars["close"].copy()

        # -- Phase 1: Indicator (collected from all modules) -------------------
        all_indicators: dict[str, tuple] = {}

        # From signals
        for _sig_name, (signal, _params) in strategy.signals.items():
            for ind_name, ind_spec in getattr(signal, "required_indicators", {}).items():
                if ind_name not in all_indicators:
                    all_indicators[ind_name] = ind_spec

        # From portfolio optimizer
        for ind_name, ind_spec in getattr(strategy.portfolio, "required_indicators", {}).items():
            if ind_name not in all_indicators:
                all_indicators[ind_name] = ind_spec

        # From universe
        for ind_name, ind_spec in getattr(strategy.universe, "required_indicators", {}).items():
            if ind_name not in all_indicators:
                all_indicators[ind_name] = ind_spec

        # From rules
        for rule in self._rules:
            for ind_name, ind_spec in getattr(rule, "required_indicators", {}).items():
                if ind_name not in all_indicators:
                    all_indicators[ind_name] = ind_spec

        for ind_name, (indicator, params) in all_indicators.items():
            t0 = _time.perf_counter()
            for symbol in self._universe.symbols:
                for dep_col in getattr(indicator, "depends_on", ()):
                    if dep_col not in self._mktdata[symbol].columns:
                        logger.warning(
                            "Indicator '%s' depends on column '%s' which does "
                            "not yet exist in mktdata. Ensure the producing "
                            "indicator is registered first.",
                            ind_name,
                            dep_col,
                        )
                self._mktdata[symbol][ind_name] = indicator.compute(
                    self._mktdata[symbol], **params,
                )
            elapsed = (_time.perf_counter() - t0) * 1000
            if tracer:
                sample = self._mktdata[self._universe.symbols[0]][ind_name]
                tracer.on_indicator(
                    name=ind_name, params=params,
                    output_summary={"rows": len(sample), "non_null": int(sample.notna().sum())},
                    duration_ms=elapsed,
                )

        if run_through == "indicator":
            return

        # -- Phase 2: Signal (vectorized, per-symbol) --------------------------
        for sig_name, (signal, params) in strategy.signals.items():
            t0 = _time.perf_counter()
            for symbol in self._universe.symbols:
                self._mktdata[symbol][sig_name] = signal.compute(
                    self._mktdata[symbol], **params,
                )
            elapsed = (_time.perf_counter() - t0) * 1000
            if tracer:
                sample = self._mktdata[self._universe.symbols[0]][sig_name]
                tracer.on_signal(
                    name=sig_name, inputs=params,
                    output_summary={"signal_count": int(sample.sum()) if hasattr(sample, 'sum') else 0},
                    duration_ms=elapsed,
                )

        # -- Phase 3 state init -----------------------------------------------
        self._trades: list[Fill] = []
        self._equity_curve: list[tuple[object, float]] = []
        self._last_known_price: dict[str, float] = {}
        self._snapshots: list[BarSnapshot] = []

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Union of all dates across symbols in mktdata."""
        symbols = self._universe.symbols
        result = self._mktdata[symbols[0]].index
        for sym in symbols[1:]:
            result = result.union(self._mktdata[sym].index)
        return result

    @property
    def result(self) -> RunResult:
        """Current result based on accumulated state."""
        return RunResult(
            portfolio=self._portfolio,
            trades=self._trades,
            equity_curve=self._equity_curve,
            mktdata=self._mktdata,
            benchmark_prices=self._benchmark_prices,
            snapshots=self._snapshots,
        )

    def step(self, date: pd.Timestamp) -> None:
        """Process a single bar through the new pipeline."""
        universe = self._universe
        strategy = self._strategy
        broker = self._broker
        portfolio = self._portfolio
        mktdata = self._mktdata

        # ── Step 1: Build bar_prices ──────────────────────────────────
        bar_prices: dict[str, Decimal] = {}
        for s in universe.symbols:
            if date in mktdata[s].index:
                bar_prices[s] = Decimal(
                    str(float(mktdata[s].loc[date, "close"])),
                )
            elif s in self._last_known_price:
                bar_prices[s] = Decimal(str(self._last_known_price[s]))
        portfolio.bar_prices = bar_prices

        # ── Step 2: Portfolio optimizer → target_weights ──────────────
        signals_data: dict[str, pd.DataFrame] = {}
        indicators_data: dict[str, pd.DataFrame] = {}
        for s in universe.symbols:
            if date in mktdata[s].index:
                sliced = mktdata[s].loc[:date]
                signals_data[s] = sliced
                indicators_data[s] = sliced

        target_weights = strategy.portfolio.optimize(signals_data, indicators_data)
        raw_target_weights = dict(target_weights)

        # ── Step 3: Pre-trade rules ───────────────────────────────────
        hold = False
        for rule in self._rules:
            for symbol in universe.symbols:
                if date not in mktdata[symbol].index:
                    continue
                row = mktdata[symbol].loc[date]
                result = rule.evaluate(symbol, row, portfolio, prices=bar_prices)
                if result.hold:
                    hold = True
                if result.weights is not None:
                    target_weights.update(result.weights)

        adjusted_weights = dict(target_weights)

        # ── Step 4: Trading algorithm (skip if hold) ──────────────────
        if not hold:
            total_capital = portfolio.total_value(bar_prices)
            planned = generate_orders(
                target_weights={
                    s: Decimal(str(w))
                    for s, w in target_weights.items()
                    if s != "CASH"
                },
                positions=portfolio.positions,
                prices=bar_prices,
                total_capital=total_capital,
                lot_size=self._lot_size,
                currency=portfolio.currency,
            )
            for p in planned:
                broker.submit_order(p.order)

        # ── Step 5: Broker executes ───────────────────────────────────
        broker.on_bar_open(mktdata, date)
        for fill in broker.get_fills():
            _apply_fill(portfolio, fill)
            self._trades.append(fill)

        broker.on_bar_close(mktdata, date)
        for fill in broker.get_fills():
            _apply_fill(portfolio, fill)
            self._trades.append(fill)

        # ── Step 6: Post-trade monitoring rules ───────────────────────
        exit_targets: dict[str, float] = {}
        for rule in self._rules:
            for symbol in list(portfolio.positions.keys()):
                if date not in mktdata[symbol].index:
                    continue
                row = mktdata[symbol].loc[date]
                result = rule.evaluate(symbol, row, portfolio, prices=bar_prices)
                if result.target_positions is not None:
                    for sym, target_ratio in result.target_positions.items():
                        if sym in exit_targets:
                            exit_targets[sym] = min(exit_targets[sym], target_ratio)
                        else:
                            exit_targets[sym] = target_ratio

        # ── Step 7: Execute exits ─────────────────────────────────────
        if exit_targets:
            for sym, target_ratio in exit_targets.items():
                if sym not in portfolio.positions:
                    continue
                pos = portfolio.positions[sym]
                target_shares = int(pos.shares * target_ratio)
                sell_shares = pos.shares - target_shares
                if sell_shares > 0:
                    broker.submit_order(Order(symbol=sym, side="SELL", shares=sell_shares, currency=portfolio.currency))

            # ── Step 8: Broker executes exit orders ───────────────────
            broker.on_bar_close(mktdata, date)
            fully_exited: list[str] = []
            for fill in broker.get_fills():
                _apply_fill(portfolio, fill)
                self._trades.append(fill)
                if fill.order.side == "SELL" and fill.order.symbol not in portfolio.positions:
                    fully_exited.append(fill.order.symbol)

            reset_symbols = getattr(strategy.portfolio, "reset_symbols", None)
            if fully_exited and callable(reset_symbols):
                reset_symbols(list(dict.fromkeys(fully_exited)))

        # ── Cash interest ──────────────────────────────────────────
        if self._cash_annual_return > 0:
            daily_rate = (1 + self._cash_annual_return) ** (1 / 252) - 1
            interest = portfolio.cash * Decimal(str(daily_rate))
            portfolio.cash += interest

        # ── Step 9: Record equity curve ───────────────────────────────
        prices: dict[str, Decimal] = {}
        for s in universe.symbols:
            if date in mktdata[s].index:
                close = Decimal(str(float(mktdata[s].loc[date, "close"])))
                if not close.is_finite():
                    if s in self._last_known_price:
                        prices[s] = Decimal(str(self._last_known_price[s]))
                    continue
                self._last_known_price[s] = float(close)
                prices[s] = close
            elif s in self._last_known_price:
                prices[s] = Decimal(str(self._last_known_price[s]))
        self._equity_curve.append((date, float(portfolio.total_value(prices))))

        # ── Step 10: Record bar snapshot ──────────────────────────────
        pos_snapshot = {
            sym: PositionSnapshot(shares=pos.shares, avg_cost=float(pos.avg_cost))
            for sym, pos in portfolio.positions.items()
        }
        tv = float(portfolio.total_value(prices))
        self._snapshots.append(
            BarSnapshot(
                date=date,
                target_weights=raw_target_weights,
                adjusted_weights=adjusted_weights,
                positions=pos_snapshot,
                cash=float(portfolio.cash),
                total_value=tv,
            )
        )

    def run(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker: Broker,
        start: str,
        end: str,
        initial_cash: float = 100_000.0,
        run_through: Literal["indicator", "signal"] | None = None,
        tracer: DefaultTracer | None = None,
        rules: list[Rule] | None = None,
        lot_size: int = 1,
        cash_annual_return: float = 0.0,
        data_start: str | None = None,
    ) -> RunResult:
        """Run the strategy pipeline.

        Parameters
        ----------
        strategy : Strategy
            The strategy definition.
        market : MarketDataProvider
            Data provider for loading bars.
        broker : Broker
            Unified broker interface (order routing, fills, lifecycle).
        start, end : str
            Date range for trading.
        initial_cash : float
            Starting cash.
        run_through : str | None
            Stop after this phase: ``"indicator"`` or ``"signal"``.
            ``None`` runs the full pipeline including rules.
        rules : list[Rule] | None
            Optional list of Rule instances.
        lot_size : int
            Minimum trade unit (e.g. 100 for A-shares).
        cash_annual_return : float
            Annual cash interest rate.
        data_start : str | None
            Start date for loading market data (for indicator warmup).
            If None, uses ``start``.
        """
        self.setup(
            strategy=strategy, market=market, broker=broker,
            start=start, end=end, initial_cash=initial_cash,
            run_through=run_through, tracer=tracer, rules=rules,
            lot_size=lot_size, cash_annual_return=cash_annual_return,
            data_start=data_start,
        )

        if run_through == "indicator":
            return RunResult(
                portfolio=self._portfolio, trades=[], equity_curve=[],
                mktdata=self._mktdata,
                benchmark_prices=self._benchmark_prices,
                snapshots=[],
            )

        if run_through == "signal":
            return RunResult(
                portfolio=self._portfolio, trades=[], equity_curve=[],
                mktdata=self._mktdata,
                benchmark_prices=self._benchmark_prices,
                snapshots=[],
            )

        for date in self.dates:
            if date < self._step_start:
                continue
            self.step(date)

        if self._tracer:
            for rule in self._rules:
                self._tracer.on_rule(
                    name=rule.name, rule_type="rule",
                    output_summary={"total_trades": len(self._trades)},
                    duration_ms=0.0,
                )
            self._tracer.on_run_end("ok")

        return self.result


def _apply_fill(portfolio: Portfolio, fill: Fill) -> None:
    """Update portfolio state based on a fill."""
    order = fill.order
    symbol = order.symbol
    cost = fill.filled_price * order.shares

    if order.side == "BUY":
        portfolio.cash -= cost + fill.fee
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
        portfolio.cash += cost - fill.fee
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
