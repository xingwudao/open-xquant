"""Universal strategy execution engine.

Executes the new pipeline: Universe -> Indicator -> Signal -> Optimize -> Rules.
Provider-agnostic — the same engine serves backtest, paper trading,
and live trading.  The difference is which providers you plug in.
"""

from __future__ import annotations

import copy
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
from oxq.universe.base import UniverseProvider

logger = logging.getLogger(__name__)


def _clone_rules(rules: list[Rule] | None) -> list[Rule]:
    return copy.deepcopy(list(rules or []))


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
        universe: UniverseProvider | None = None,
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
        universe : UniverseProvider | None
            Tradable universe for this run. New SDK code should pass this
            explicitly so the same Strategy can be reused across universes.
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
        self._rules: list[Rule] = _clone_rules(rules if rules is not None else getattr(strategy, "rules", []))
        self._lot_size = lot_size
        self._cash_annual_return = cash_annual_return
        reset_optimizer = getattr(strategy.portfolio, "reset", None)
        if callable(reset_optimizer):
            reset_optimizer()
        if tracer:
            tracer.on_run_start(
                strategy_name=strategy.name,
                config={"start": start, "end": end, "initial_cash": initial_cash},
            )

        # -- Phase 0: Universe ------------------------------------------------
        run_universe = universe if universe is not None else getattr(strategy, "_legacy_universe", None)
        if run_universe is None:
            msg = "Engine.run requires a universe; pass universe=... for this run"
            raise ValueError(msg)
        self._universe = run_universe.get_universe(as_of_date=end)

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

        # From the run universe
        for ind_name, ind_spec in getattr(run_universe, "required_indicators", {}).items():
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
                    output_summary=_signal_output_summary(sample),
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
        get_all_orders = getattr(self._broker, "get_all_orders", None)
        orders = list(get_all_orders()) if callable(get_all_orders) else []
        return RunResult(
            portfolio=self._portfolio,
            trades=self._trades,
            equity_curve=self._equity_curve,
            mktdata=self._mktdata,
            benchmark_prices=self._benchmark_prices,
            snapshots=self._snapshots,
            orders=orders,
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

        set_current_date = getattr(broker, "set_current_date", None)
        if callable(set_current_date):
            set_current_date(date)

        # Fill previously submitted next-bar market orders before optimizing
        # for the new bar, so target generation sees the actual portfolio.
        fill_due_market_orders = getattr(broker, "fill_due_market_orders", None)
        if callable(fill_due_market_orders):
            _sync_broker_cash(broker, portfolio)
            fill_due_market_orders(mktdata, date)
        _apply_fills(portfolio, broker.get_fills(), self._trades, strategy.portfolio)

        # ── Step 2: Portfolio optimizer → target_weights ──────────────
        signals_data: dict[str, pd.DataFrame] = {}
        indicators_data: dict[str, pd.DataFrame] = {}
        for s in universe.symbols:
            if date in mktdata[s].index:
                sliced = mktdata[s].loc[:date]
                signals_data[s] = sliced
                indicators_data[s] = sliced

        set_held_symbols = getattr(strategy.portfolio, "set_held_symbols", None)
        if callable(set_held_symbols):
            set_held_symbols(list(portfolio.positions.keys()))
        set_pending_buy_symbols = getattr(strategy.portfolio, "set_pending_buy_symbols", None)
        if callable(set_pending_buy_symbols):
            pending_buy_symbols = [
                managed.order.symbol
                for managed in broker.get_open_orders()
                if managed.order.order_type == "market" and managed.order.side == "BUY"
            ]
            set_pending_buy_symbols(pending_buy_symbols)
        target_weights = strategy.portfolio.optimize(signals_data, indicators_data)
        optimizer_hold = bool(getattr(strategy.portfolio, "skip_rebalance", False))
        raw_target_weights = dict(target_weights)

        # ── Step 3: Pre-trade rules ───────────────────────────────────
        rule_hold = False
        rule_weight_override = False
        rule_weight_overrides: dict[str, float] = {}
        rule_reasons: dict[str, list[str]] = {}

        def record_rule_reason(symbol: str, reason: str) -> None:
            if not reason:
                return
            rule_reasons.setdefault(symbol, []).append(reason)

        for rule in self._rules:
            for symbol in universe.symbols:
                if date not in mktdata[symbol].index:
                    continue
                row = mktdata[symbol].loc[date]
                result = rule.evaluate(symbol, row, portfolio, prices=bar_prices)
                if result.hold:
                    rule_hold = True
                    record_rule_reason("__all__", result.reason)
                if result.weights is not None:
                    rule_weight_override = True
                    rule_weight_overrides.update(result.weights)
                    target_weights.update(result.weights)
                    for target_symbol in result.weights:
                        record_rule_reason(target_symbol, result.reason)
                if result.constraints is not None:
                    for constrained_symbol in result.constraints:
                        record_rule_reason(constrained_symbol, result.reason)

        def current_portfolio_weights() -> dict[str, float]:
            total_value = portfolio.total_value(bar_prices)
            if total_value <= 0:
                return {}

            weights: dict[str, float] = {}
            for symbol, position in portfolio.positions.items():
                price = bar_prices.get(symbol)
                if price is None:
                    continue
                value = Decimal(position.shares) * price
                if value != 0:
                    weights[symbol] = float(value / total_value)
            if portfolio.cash != 0:
                weights["CASH"] = float(portfolio.cash / total_value)
            return weights

        def sync_pending_reductions() -> bool:
            pending_reduction_symbols = set(getattr(strategy.portfolio, "pending_reduction_symbols", set()))
            if not pending_reduction_symbols:
                return True

            total_value = portfolio.total_value(bar_prices)
            pending_buy_symbols = {
                managed.order.symbol
                for managed in broker.get_open_orders()
                if managed.order.order_type == "market" and managed.order.side == "BUY"
            }
            pending_sell_symbols = {
                managed.order.symbol
                for managed in broker.get_open_orders()
                if managed.order.order_type == "market" and managed.order.side == "SELL"
            }
            reached_symbols: list[str] = []
            all_reached = True
            for symbol in pending_reduction_symbols:
                position = portfolio.positions.get(symbol)
                price = bar_prices.get(symbol)
                if symbol in pending_buy_symbols or symbol in pending_sell_symbols:
                    all_reached = False
                    continue
                if position is None or position.shares <= 0 or price is None or total_value <= 0:
                    reached_symbols.append(symbol)
                    continue
                current_weight = float((Decimal(position.shares) * price) / total_value)
                target_weight = float(target_weights.get(symbol, 0.0))
                if current_weight > target_weight + 1e-9:
                    all_reached = False
                else:
                    reached_symbols.append(symbol)

            if reached_symbols:
                clear_pending_reductions = getattr(strategy.portfolio, "clear_pending_reductions", None)
                if callable(clear_pending_reductions):
                    clear_pending_reductions(reached_symbols)
            return all_reached

        pending_reductions_reached = sync_pending_reductions()

        def optimizer_hold_target_reached() -> bool:
            if not pending_reductions_reached:
                return False
            for symbol, weight in target_weights.items():
                if symbol == "CASH" or weight <= 0:
                    continue
                position = portfolio.positions.get(symbol)
                if position is None or position.shares <= 0:
                    return False
            return True

        hold = rule_hold or (
            optimizer_hold
            and not rule_weight_override
            and optimizer_hold_target_reached()
        )
        if optimizer_hold and not rule_weight_override and hold:
            record_rule_reason("__all__", "signal_hold")

        if hold:
            adjusted_weights = (
                dict(self._snapshots[-1].adjusted_weights)
                if self._snapshots
                else current_portfolio_weights()
            )
        elif optimizer_hold and rule_weight_override:
            adjusted_weights = current_portfolio_weights()
            adjusted_weights.update(rule_weight_overrides)
            invested_weight = sum(
                weight for symbol, weight in adjusted_weights.items()
                if symbol != "CASH" and weight > 0
            )
            adjusted_weights["CASH"] = max(0.0, 1.0 - invested_weight)
        else:
            adjusted_weights = dict(target_weights)

        # ── Step 4: Trading algorithm (skip if hold) ──────────────────
        if not hold:
            total_capital = portfolio.total_value(bar_prices)
            estimate_market_buy_cost = getattr(broker, "estimate_market_buy_cost", None)
            estimate_market_sell_proceeds = getattr(broker, "estimate_market_sell_proceeds", None)
            buy_cost_estimator = None
            if callable(estimate_market_buy_cost):
                def buy_cost_estimator(symbol: str, price: Decimal, shares: int) -> Decimal:
                    return estimate_market_buy_cost(symbol, price, shares, portfolio.currency)
            sell_proceeds_estimator = None
            if callable(estimate_market_sell_proceeds):
                def sell_proceeds_estimator(symbol: str, price: Decimal, shares: int) -> Decimal:
                    return estimate_market_sell_proceeds(symbol, price, shares, portfolio.currency)

            order_weights = adjusted_weights
            order_positions = portfolio.positions
            buying_power = None
            if optimizer_hold and rule_weight_override:
                overridden_symbols = set(rule_weight_overrides)
                order_weights = {
                    symbol: weight for symbol, weight in adjusted_weights.items()
                    if symbol in overridden_symbols
                }
                order_positions = {
                    symbol: position for symbol, position in portfolio.positions.items()
                    if symbol in overridden_symbols
                }
                buying_power = portfolio.cash
            else:
                held_symbols = set(getattr(strategy.portfolio, "held_symbols", set()))
                pending_reduction_symbols = set(getattr(strategy.portfolio, "pending_reduction_symbols", set()))
                frozen_symbols = {
                    symbol for symbol in held_symbols - pending_reduction_symbols
                    if symbol in portfolio.positions
                }
                if frozen_symbols:
                    order_weights = {
                        symbol: weight for symbol, weight in adjusted_weights.items()
                        if symbol not in frozen_symbols
                    }
                    order_positions = {
                        symbol: position for symbol, position in portfolio.positions.items()
                        if symbol not in frozen_symbols
                    }
                    buying_power = portfolio.cash

            planned = generate_orders(
                target_weights={
                    s: Decimal(str(w))
                    for s, w in order_weights.items()
                    if s != "CASH"
                },
                positions=order_positions,
                prices=bar_prices,
                total_capital=total_capital,
                lot_size=self._lot_size,
                currency=portfolio.currency,
                pending_orders=[managed.order for managed in broker.get_open_orders() if managed.order.order_type == "market"],
                buy_cost_estimator=buy_cost_estimator,
                sell_proceeds_estimator=sell_proceeds_estimator,
                buying_power=buying_power,
            )
            for p in planned:
                broker.submit_order(p.order)

        # ── Step 5: Broker executes ───────────────────────────────────
        _sync_broker_cash(broker, portfolio)
        broker.on_bar_open(mktdata, date)
        _apply_fills(portfolio, broker.get_fills(), self._trades, strategy.portfolio)

        _sync_broker_cash(broker, portfolio)
        broker.on_bar_close(mktdata, date)
        _apply_fills(portfolio, broker.get_fills(), self._trades, strategy.portfolio)

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
                        record_rule_reason(sym, result.reason)
                        if sym in exit_targets:
                            exit_targets[sym] = min(exit_targets[sym], target_ratio)
                        else:
                            exit_targets[sym] = target_ratio

        # ── Step 7: Execute exits ─────────────────────────────────────
        if exit_targets:
            held_weights = current_portfolio_weights()
            adjusted_weights = {}
            cash_weight = float(held_weights.get("CASH", 0.0))
            for sym, target_ratio in exit_targets.items():
                base_weight = float(held_weights.get(sym, 0.0))
                retained_weight = float(base_weight) * float(target_ratio)
                adjusted_weights[sym] = retained_weight
                cash_weight += float(base_weight) - retained_weight
            for sym, held_weight in held_weights.items():
                if sym == "CASH" or sym in exit_targets:
                    continue
                adjusted_weights[sym] = float(held_weight)
            if cash_weight:
                adjusted_weights["CASH"] = cash_weight
            for sym, target_ratio in exit_targets.items():
                if sym not in portfolio.positions:
                    continue
                pos = portfolio.positions[sym]
                target_shares = int(pos.shares * target_ratio)
                sell_shares = pos.shares - target_shares
                if sell_shares > 0:
                    cancel_market_orders = getattr(broker, "cancel_market_orders", None)
                    if callable(cancel_market_orders):
                        cancel_market_orders(sym, "BUY", reason="exit_sell_submitted")
                        cancel_market_orders(sym, "SELL", reason="exit_sell_submitted")
                    else:
                        for managed in broker.get_open_orders(sym):
                            if managed.order.order_type != "market":
                                continue
                            if managed.order.side not in {"BUY", "SELL"}:
                                continue
                            managed.status = "canceled"
                            managed.status_reason = "exit_sell_submitted"
                    broker.submit_order(Order(symbol=sym, side="SELL", shares=sell_shares, currency=portfolio.currency))

            # ── Step 8: Broker executes exit orders ───────────────────
            _sync_broker_cash(broker, portfolio)
            broker.on_bar_close(mktdata, date)
            _apply_fills(portfolio, broker.get_fills(), self._trades, strategy.portfolio)

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
                rule_reasons={symbol: "; ".join(reasons) for symbol, reasons in rule_reasons.items()},
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
        universe: UniverseProvider | None = None,
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
        universe : UniverseProvider | None
            Tradable universe for this run. New SDK code should pass this
            explicitly so the same Strategy can be reused across universes.
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
            universe=universe,
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


def _signal_output_summary(sample: pd.Series) -> dict[str, object]:
    non_null = sample.dropna()
    summary: dict[str, object] = {
        "rows": len(sample),
        "non_null": int(non_null.size),
    }
    if non_null.empty:
        summary["signal_count"] = 0
        return summary

    if pd.api.types.is_bool_dtype(non_null):
        summary["signal_count"] = int(non_null.astype(bool).sum())
        return summary
    if pd.api.types.is_numeric_dtype(non_null):
        summary["signal_count"] = int((non_null != 0).sum())
        return summary

    labels = non_null.astype(str).str.upper()
    counts = {label: int(count) for label, count in labels.value_counts().sort_index().items()}
    summary["value_counts"] = counts
    no_event_labels = {"", "0", "FALSE", "HOLD", "NONE", "NAN"}
    summary["signal_count"] = int(
        sum(count for label, count in counts.items() if label not in no_event_labels)
    )
    return summary


def _apply_fill(portfolio: Portfolio, fill: Fill) -> bool:
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
        if symbol not in portfolio.positions:
            logger.warning("Rejecting SELL fill for %s with no position", symbol)
            return False
        old = portfolio.positions[symbol]
        if order.shares > old.shares:
            logger.warning(
                "Rejecting SELL fill for %s: shares=%s exceeds position=%s",
                symbol,
                order.shares,
                old.shares,
            )
            return False
        portfolio.cash += cost - fill.fee
        remaining = old.shares - order.shares
        if remaining <= 0:
            del portfolio.positions[symbol]
        else:
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                shares=remaining,
                avg_cost=old.avg_cost,
            )
    return True


def _sync_broker_cash(broker: Broker, portfolio: Portfolio) -> None:
    set_available_cash = getattr(broker, "set_available_cash", None)
    if callable(set_available_cash):
        set_available_cash(portfolio.cash)


def _apply_fills(portfolio: Portfolio, fills: list[Fill], trades: list[Fill], optimizer: object) -> None:
    """Apply fills and notify stateful optimizers when positions fully exit."""
    fully_exited: list[str] = []
    for fill in fills:
        if not _apply_fill(portfolio, fill):
            continue
        trades.append(fill)
        if fill.order.side == "SELL" and fill.order.symbol not in portfolio.positions:
            fully_exited.append(fill.order.symbol)

    reset_symbols = getattr(optimizer, "reset_symbols", None)
    if fully_exited and callable(reset_symbols):
        reset_symbols(list(dict.fromkeys(fully_exited)))
