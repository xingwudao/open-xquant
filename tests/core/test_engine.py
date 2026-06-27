"""Tests for Engine — new pipeline integration."""

from decimal import Decimal

import pandas as pd

from oxq.core.engine import Engine, _apply_fill
from oxq.core.strategy import Strategy
from oxq.core.types import Fill, Order, Portfolio, Position, RuleResult
from oxq.indicators.sma import SMA
from oxq.portfolio.optimizers import EqualWeightOptimizer, SignalToPositionOptimizer
from oxq.signals.crossover import Crossover
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.universe.static import StaticUniverse


class FakeMarketDataProvider:
    """In-memory market data provider for testing."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC"))]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_trending_data() -> dict[str, pd.DataFrame]:
    """Create data: downtrend -> uptrend -> downtrend to trigger crossover signals.

    Structure (120 bars):
    - Bars 0-49:  downtrend 200 -> 102 (SMA10 < SMA50 at bar 49)
    - Bars 50-89: uptrend 102 -> 182 (SMA10 crosses above SMA50 -> golden cross)
    - Bars 90-119: downtrend 182 -> 122 (SMA10 crosses below SMA50 -> death cross)
    """
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 -> 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 -> 180
    for i in range(30):
        closes.append(180 - i * 2)       # 180 -> 122

    return {
        "AAPL": pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        ),
    }


def _make_strategy() -> Strategy:
    crossover = Crossover()
    crossover.required_indicators = {  # type: ignore[attr-defined]
        "sma_10": (SMA(), {"period": 10}),
        "sma_50": (SMA(), {"period": 50}),
    }
    return Strategy(
        name="test_sma_crossover",
        universe=StaticUniverse(("AAPL",)),
        signals={
            "sma_10_x_sma_50": (crossover, {"fast": "sma_10", "slow": "sma_50"}),
        },
        portfolio=EqualWeightOptimizer(),
    )


def test_engine_full_pipeline() -> None:
    """Basic run with EqualWeightOptimizer, verify equity curve and signal columns."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = Engine()
    sim_broker = SimBroker()

    result = engine.run(
        strategy, market=market, broker=sim_broker,
        start="2024-01-01", end="2024-12-31",
    )

    # Equity curve should have one entry per bar
    assert len(result.equity_curve) == 120
    assert len(result.snapshots) == 120
    # mktdata should have indicator and signal columns
    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_50" in df.columns
    assert "sma_10_x_sma_50" in df.columns


def test_engine_run_accepts_universe_separate_from_strategy() -> None:
    data = _make_trending_data()
    data["MSFT"] = data["AAPL"].copy()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    strategy._legacy_universe = None

    result = Engine().run(
        strategy,
        market=market,
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
        universe=StaticUniverse(("MSFT",)),
    )

    assert set(result.mktdata) == {"MSFT"}


def test_engine_run_uses_strategy_rules_by_default() -> None:
    data = _make_trending_data()

    class RecordingRule:
        name = "RecordingRule"

        def __init__(self) -> None:
            self.calls: list[tuple[str, pd.Timestamp]] = []

        def evaluate(self, symbol, row, portfolio, prices=None):
            self.calls.append((symbol, row.name))
            return RuleResult()

    strategy = _make_strategy()
    strategy.rules = [RecordingRule()]
    engine = Engine()

    result = engine.run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
    )

    assert len(result.equity_curve) == 120
    assert strategy.rules[0].calls == []
    assert engine._rules[0] is not strategy.rules[0]
    assert engine._rules[0].calls
    assert engine._rules[0].calls[0][0] == "AAPL"


def test_engine_with_stop_loss_rule() -> None:
    """Post-trade rule triggers sells when stop loss condition met."""
    from oxq.rules.exit import ExitRule

    crossover = Crossover()
    crossover.required_indicators = {  # type: ignore[attr-defined]
        "sma_10": (SMA(), {"period": 10}),
        "sma_50": (SMA(), {"period": 50}),
    }

    strategy = Strategy(
        name="stop_loss_test",
        universe=StaticUniverse(("AAPL",)),
        signals={
            "sma_10_x_sma_50": (crossover, {"fast": "sma_10", "slow": "sma_50"}),
        },
        portfolio=EqualWeightOptimizer(),
    )

    # ExitRule acts as post-trade monitoring: sell when fast < slow
    exit_rule = ExitRule(fast="sma_10", slow="sma_50")

    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    broker = SimBroker()

    result = Engine().run(
        strategy, market=market, broker=broker,
        start="2024-01-01", end="2024-12-31",
        rules=[exit_rule],
    )

    # The exit rule should trigger sells when sma_10 < sma_50
    sell_trades = [t for t in result.trades if t.order.side == "SELL"]
    assert len(sell_trades) > 0
    assert len(result.equity_curve) == 120


def test_engine_run_through_signal() -> None:
    """Verify early stop at signal phase."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = Engine()
    sim_broker = SimBroker()

    result = engine.run(
        strategy, market=market, broker=sim_broker,
        start="2024-01-01", end="2024-12-31",
        run_through="signal",
    )

    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_10_x_sma_50" in df.columns
    assert len(result.trades) == 0
    assert len(result.equity_curve) == 0


def test_engine_step_matches_run() -> None:
    """step() called for each date should produce same result as run()."""
    data = _make_trending_data()
    strategy = _make_strategy()

    # Run via run()
    engine1 = Engine()
    result1 = engine1.run(
        strategy, market=FakeMarketDataProvider(data), broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
    )

    # Run via setup() + step()
    engine2 = Engine()
    engine2.setup(
        strategy=strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
    )
    for date in engine2.dates:
        engine2.step(date)
    result2 = engine2.result

    # Equity curves should match
    assert len(result1.equity_curve) == len(result2.equity_curve)
    for (d1, v1), (d2, v2) in zip(result1.equity_curve, result2.equity_curve):
        assert d1 == d2
        assert abs(v1 - v2) < 0.01

    assert len(result1.snapshots) == len(result2.snapshots)


def test_apply_fill_buy() -> None:
    """Unit test for _apply_fill with BUY order."""
    portfolio = Portfolio(cash=Decimal("100000"))
    fill = Fill(
        order=Order(symbol="AAPL", side="BUY", shares=100),
        filled_price=Decimal("150"),
        filled_at="2024-01-02",
    )
    _apply_fill(portfolio, fill)

    assert portfolio.cash == Decimal("85000")  # 100000 - 100*150
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].shares == 100
    assert portfolio.positions["AAPL"].avg_cost == Decimal("150")


def test_apply_fill_sell() -> None:
    """Unit test for _apply_fill with SELL order."""
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    fill = Fill(
        order=Order(symbol="AAPL", side="SELL", shares=100),
        filled_price=Decimal("160"),
        filled_at="2024-03-01",
    )
    _apply_fill(portfolio, fill)

    assert portfolio.cash == Decimal("66000")  # 50000 + 100*160
    assert "AAPL" not in portfolio.positions


def test_apply_fill_rejects_oversell() -> None:
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    fill = Fill(
        order=Order(symbol="AAPL", side="SELL", shares=101),
        filled_price=Decimal("160"),
        filled_at="2024-03-01",
    )

    assert _apply_fill(portfolio, fill) is False
    assert portfolio.cash == Decimal("50000")
    assert portfolio.positions["AAPL"].shares == 100


def test_apply_fill_rejects_sell_without_position() -> None:
    portfolio = Portfolio(cash=Decimal("50000"))
    fill = Fill(
        order=Order(symbol="AAPL", side="SELL", shares=100),
        filled_price=Decimal("160"),
        filled_at="2024-03-01",
    )

    assert _apply_fill(portfolio, fill) is False
    assert portfolio.cash == Decimal("50000")
    assert "AAPL" not in portfolio.positions


def test_next_open_exit_replaces_pending_market_sell() -> None:
    class CashOptimizer:
        def optimize(self, signals, indicators):  # noqa: ANN001, ANN201
            return {"CASH": 1.0}

    class AlwaysExitRule:
        name = "AlwaysExitRule"

        def evaluate(self, symbol, row, portfolio, prices=None):  # noqa: ANN001, ANN201
            if symbol in portfolio.positions:
                return RuleResult(target_positions={symbol: 0.0})
            return RuleResult()

    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [100.0, 110.0],
                "high": [101.0, 111.0],
                "low": [99.0, 109.0],
                "close": [100.0, 110.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }
    strategy = Strategy(
        name="pending_sell_exit",
        universe=StaticUniverse(("AAPL",)),
        signals={},
        portfolio=CashOptimizer(),
    )
    broker = SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS")
    engine = Engine()
    engine.setup(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=broker,
        start="2024-01-02",
        end="2024-01-03",
        initial_cash=0,
        rules=[AlwaysExitRule()],
    )
    engine._portfolio = Portfolio(  # noqa: SLF001
        cash=Decimal("0"),
        currency="CNY",
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100"))},
    )

    engine.step(dates[0])
    sell_orders = [order for order in broker.get_all_orders() if order.order.side == "SELL"]
    assert [order.status for order in sell_orders] == ["canceled", "open"]

    engine.step(dates[1])

    sell_trades = [trade for trade in engine.result.trades if trade.order.side == "SELL"]
    filled_sell_orders = [
        order
        for order in broker.get_all_orders()
        if order.order.side == "SELL" and order.status == "filled"
    ]
    assert len(sell_trades) == 1
    assert len(filled_sell_orders) == 1
    assert engine.result.portfolio.cash == Decimal("11000")
    assert "AAPL" not in engine.result.portfolio.positions


def test_engine_sets_dataframe_attrs() -> None:
    """Engine should set timezone and currency attrs on DataFrames."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()

    result = Engine().run(
        strategy, market=market, broker=SimBroker(),
        rules=[], start="2024-01-01", end="2024-12-31",
    )

    df = result.mktdata["AAPL"]
    assert "timezone" in df.attrs
    assert df.attrs["timezone"] == "UTC"
    assert "currency" in df.attrs
    assert df.attrs["currency"] == "CNY"


def test_engine_collects_indicators_from_all_modules() -> None:
    """Engine collects required_indicators from signals, portfolio, universe, and rules."""
    from oxq.rules.exit import ExitRule

    crossover = Crossover()
    crossover.required_indicators = {  # type: ignore[attr-defined]
        "sma_10": (SMA(), {"period": 10}),
    }

    optimizer = EqualWeightOptimizer()
    optimizer.required_indicators = {  # type: ignore[attr-defined]
        "sma_50": (SMA(), {"period": 50}),
    }

    universe = StaticUniverse(("AAPL",))
    object.__setattr__(universe, "required_indicators", {  # type: ignore[attr-defined]
        "sma_20": (SMA(), {"period": 20}),
    })

    exit_rule = ExitRule(fast="sma_10", slow="sma_50")
    exit_rule.required_indicators = {  # type: ignore[attr-defined]
        "sma_30": (SMA(), {"period": 30}),
    }

    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    strategy = Strategy(
        name="multi_source_indicators",
        universe=universe,
        signals={"cross": (crossover, {"fast": "sma_10", "slow": "sma_50"})},
        portfolio=optimizer,
    )

    result = Engine().run(
        strategy, market=market, broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
        rules=[exit_rule],
    )

    df = result.mktdata["AAPL"]
    # sma_10 from signal, sma_50 from portfolio, sma_20 from universe, sma_30 from rule
    assert "sma_10" in df.columns
    assert "sma_50" in df.columns
    assert "sma_20" in df.columns
    assert "sma_30" in df.columns


def test_engine_benchmarks() -> None:
    """Verify benchmark prices recorded."""
    dates = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    bench_closes = [100.0, 102.0, 104.0, 106.0, 108.0]
    aapl_closes = [50.0, 51.0, 52.0, 53.0, 54.0]

    data = {
        "AAPL": pd.DataFrame(
            {
                "open": aapl_closes,
                "high": aapl_closes,
                "low": aapl_closes,
                "close": aapl_closes,
                "volume": [1_000_000] * 5,
            },
            index=dates,
        ),
        "BENCH": pd.DataFrame(
            {
                "open": bench_closes,
                "high": bench_closes,
                "low": bench_closes,
                "close": bench_closes,
                "volume": [500_000] * 5,
            },
            index=dates,
        ),
    }

    strategy = Strategy(
        name="bench_test",
        universe=StaticUniverse(("AAPL",)),
        signals={},
        portfolio=EqualWeightOptimizer(),
        benchmarks=["BENCH"],
    )

    broker = SimBroker()
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=broker,
        start="2024-01-01",
        end="2024-12-31",
    )

    assert "BENCH" in result.benchmark_prices
    bench_series = result.benchmark_prices["BENCH"]
    assert isinstance(bench_series, pd.Series)
    assert list(bench_series.values) == bench_closes


class AlwaysBuyOptimizer:
    """Portfolio optimizer that always allocates 100% to first symbol."""

    def optimize(
        self, signals: dict[str, pd.DataFrame], indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        symbols = list(signals.keys())
        if symbols:
            return {symbols[0]: 1.0}
        return {"CASH": 1.0}


class NeverBuyOptimizer:
    """Portfolio optimizer that always holds cash."""

    def optimize(
        self, signals: dict[str, pd.DataFrame], indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        return {"CASH": 1.0}


class AlwaysExitRule:
    """Rule that exits any open position."""

    name = "AlwaysExit"

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        if symbol in portfolio.positions:
            return RuleResult(target_positions={symbol: 0.0})
        return RuleResult()


class ResetAwareAlwaysBuyOptimizer(AlwaysBuyOptimizer):
    """Optimizer that records exit reset notifications."""

    def __init__(self) -> None:
        self.reset_symbols_seen: list[str] = []

    def reset_symbols(self, symbols: list[str]) -> None:
        self.reset_symbols_seen.extend(symbols)


class CloseSellBroker(SimBroker):
    """Broker that emits a full SELL from the normal on_bar_close path."""

    def __init__(self) -> None:
        super().__init__()
        self._buy_shares = 0
        self._close_count = 0

    def submit_order(self, order: Order) -> str:
        if order.side == "BUY":
            self._buy_shares = order.shares
        return super().submit_order(order)

    def on_bar_close(self, mktdata: dict[str, pd.DataFrame], date: pd.Timestamp) -> None:
        self._close_count += 1
        if self._close_count == 1:
            super().on_bar_close(mktdata, date)
            return
        if self._buy_shares:
            self._fills.append(
                Fill(
                    order=Order(symbol="AAA", side="SELL", shares=self._buy_shares),
                    filled_price=Decimal("10"),
                    filled_at=date.isoformat(),
                )
            )
            self._buy_shares = 0


class DroppingBroker(SimBroker):
    """Broker that rejects submitted orders without leaving open orders."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted_orders: list[Order] = []

    def submit_order(self, order: Order) -> str:
        self.submitted_orders.append(order)
        return f"dropped-{len(self.submitted_orders)}"


class DroppingSellBroker(SimBroker):
    """Broker that fills buys but drops sells without leaving open orders."""

    def __init__(self) -> None:
        super().__init__()
        self.dropped_sells: list[Order] = []

    def submit_order(self, order: Order) -> str:
        if order.side == "SELL":
            self.dropped_sells.append(order)
            return f"dropped-sell-{len(self.dropped_sells)}"
        return super().submit_order(order)


class HoldingSellBroker(SimBroker):
    """Broker that leaves submitted market sells open for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.held_sells: list[Order] = []

    def submit_order(self, order: Order) -> str:
        order_id = super().submit_order(order)
        if order.side == "SELL":
            self.held_sells.append(order)
            self._pending_market = [
                managed
                for managed in self._pending_market
                if managed.order is not order
            ]
        return order_id


def test_engine_notifies_optimizer_after_full_exit() -> None:
    """Exit fills should notify optimizers that keep per-symbol entry state."""
    dates = pd.bdate_range("2024-01-01", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [10.0, 10.0],
                "low": [10.0, 10.0],
                "close": [10.0, 10.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }
    optimizer = ResetAwareAlwaysBuyOptimizer()
    strategy = Strategy(
        name="reset_after_exit",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=optimizer,
    )

    Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-01-02",
        rules=[AlwaysExitRule()],
    )

    assert optimizer.reset_symbols_seen == ["AAA", "AAA"]


def test_snapshot_adjusted_weights_include_exit_targets() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [10.0, 10.0],
                "low": [10.0, 10.0],
                "close": [10.0, 10.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }
    strategy = Strategy(
        name="snapshot_exit_target",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-01-02",
        rules=[AlwaysExitRule()],
    )

    assert result.snapshots[0].target_weights["AAA"] == 1.0
    assert result.snapshots[0].adjusted_weights["AAA"] == 0.0


def test_snapshot_adjusted_weights_partial_exit_uses_held_exposure() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [10.0, 10.0],
                "low": [10.0, 10.0],
                "close": [10.0, 10.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }

    class BuyThenCashOptimizer:
        def __init__(self) -> None:
            self.calls = 0

        def optimize(
            self,
            signals: dict[str, pd.DataFrame],
            indicators: dict[str, pd.DataFrame],
        ) -> dict[str, float]:
            self.calls += 1
            if self.calls == 1:
                return {"AAA": 1.0}
            return {"CASH": 1.0}

    class HalfExitRule:
        name = "HalfExit"

        def evaluate(
            self,
            symbol: str,
            row: pd.Series,
            portfolio: Portfolio,
            prices: dict[str, Decimal] | None = None,
        ) -> RuleResult:
            if symbol in portfolio.positions:
                return RuleResult(target_positions={symbol: 0.5})
            return RuleResult()

    strategy = Strategy(
        name="snapshot_partial_exit_target",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=BuyThenCashOptimizer(),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS"),
        start="2024-01-02",
        end="2024-01-03",
        rules=[HalfExitRule()],
    )

    assert result.snapshots[1].target_weights == {"CASH": 1.0}
    assert result.snapshots[1].adjusted_weights == {"AAA": 0.5, "CASH": 0.5}


def test_signal_to_position_hold_does_not_rebalance_existing_position() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 50.0],
                "high": [100.0, 50.0],
                "low": [100.0, 50.0],
                "close": [100.0, 50.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }

    class BuyThenHoldSignal:
        name = "BuyThenHold"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "HOLD"], index=mktdata.index)

    strategy = Strategy(
        name="signal_hold_no_rebalance",
        universe=StaticUniverse(("AAA",)),
        signals={"timing": (BuyThenHoldSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-03",
        initial_cash=950.0,
    )

    assert len(result.trades) == 1
    assert result.trades[0].order.side == "BUY"
    assert result.trades[0].order.shares == 9
    assert result.snapshots[1].positions["AAA"].shares == 9
    assert result.snapshots[1].rule_reasons == {"__all__": "signal_hold"}


def test_signal_to_position_hold_retries_missing_latched_position() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }

    class BuyThenHoldSignal:
        name = "BuyThenHold"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "HOLD"], index=mktdata.index)

    broker = DroppingBroker()
    strategy = Strategy(
        name="signal_hold_retries_missing_position",
        universe=StaticUniverse(("AAA",)),
        signals={"timing": (BuyThenHoldSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=broker,
        start="2024-01-02",
        end="2024-01-03",
    )

    assert [order.side for order in broker.submitted_orders] == ["BUY", "BUY"]
    assert result.trades == []
    assert result.snapshots[1].positions == {}


def test_signal_to_position_hold_allows_rule_weight_override() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }

    class BuyThenHoldSignal:
        name = "BuyThenHold"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "HOLD"], index=mktdata.index)

    class BlockSecondBarRule:
        name = "BlockSecondBar"

        def evaluate(
            self,
            symbol: str,
            row: pd.Series,
            portfolio: Portfolio,
            prices: dict[str, Decimal] | None = None,
        ) -> RuleResult:
            if row.name == dates[1]:
                return RuleResult(weights={symbol: 0.0}, reason="blocked")
            return RuleResult()

    strategy = Strategy(
        name="signal_hold_rule_override",
        universe=StaticUniverse(("AAA",)),
        signals={"timing": (BuyThenHoldSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-03",
        rules=[BlockSecondBarRule()],
    )

    assert [trade.order.side for trade in result.trades] == ["BUY", "SELL"]
    assert "AAA" not in result.snapshots[1].positions
    assert result.snapshots[1].rule_reasons == {"AAA": "blocked"}


def test_signal_to_position_hold_rule_override_preserves_other_symbols() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 50.0],
                "high": [100.0, 50.0],
                "low": [100.0, 50.0],
                "close": [100.0, 50.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }

    class BuyThenHoldSignal:
        name = "BuyThenHold"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "HOLD"], index=mktdata.index)

    class BlockAAASecondBarRule:
        name = "BlockAAASecondBar"

        def evaluate(
            self,
            symbol: str,
            row: pd.Series,
            portfolio: Portfolio,
            prices: dict[str, Decimal] | None = None,
        ) -> RuleResult:
            if symbol == "AAA" and row.name == dates[1]:
                return RuleResult(weights={symbol: 0.0}, reason="blocked")
            return RuleResult()

    strategy = Strategy(
        name="signal_hold_rule_override_preserves_others",
        universe=StaticUniverse(("AAA", "BBB")),
        signals={"timing": (BuyThenHoldSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-03",
        rules=[BlockAAASecondBarRule()],
    )

    aaa_sides = [trade.order.side for trade in result.trades if trade.order.symbol == "AAA"]
    bbb_trades = [trade for trade in result.trades if trade.order.symbol == "BBB"]

    assert aaa_sides == ["BUY", "SELL"]
    assert len(bbb_trades) == 1
    assert bbb_trades[0].order.side == "BUY"
    assert result.snapshots[1].positions["BBB"].shares == result.snapshots[0].positions["BBB"].shares


def test_signal_to_position_hold_rule_override_buy_uses_cash_budget() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
                "timing_input": ["BUY", "HOLD"],
            },
            index=dates,
        ),
        "CCC": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
                "timing_input": ["HOLD", "HOLD"],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    class AddCCCSecondBarRule:
        name = "AddCCCSecondBar"

        def evaluate(
            self,
            symbol: str,
            row: pd.Series,
            portfolio: Portfolio,
            prices: dict[str, Decimal] | None = None,
        ) -> RuleResult:
            if symbol == "CCC" and row.name == dates[1]:
                return RuleResult(weights={symbol: 0.5}, reason="add_ccc")
            return RuleResult()

    strategy = Strategy(
        name="signal_hold_rule_override_buy_cash_budget",
        universe=StaticUniverse(("AAA", "CCC")),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-03",
        rules=[AddCCCSecondBarRule()],
    )

    ccc_orders = [managed.order for managed in result.orders if managed.order.symbol == "CCC"]

    assert ccc_orders == []
    assert result.snapshots[1].rule_reasons == {"CCC": "add_ccc"}


def test_signal_to_position_mixed_hold_does_not_rebalance_held_symbol() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 50.0],
                "high": [100.0, 50.0],
                "low": [100.0, 50.0],
                "close": [100.0, 50.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            if mktdata["close"].iloc[0] == 100 and mktdata["close"].iloc[-1] == 100:
                return pd.Series(["BUY", "SELL"], index=mktdata.index)
            return pd.Series(["BUY", "HOLD"], index=mktdata.index)

    strategy = Strategy(
        name="signal_mixed_hold_no_rebalance",
        universe=StaticUniverse(("AAA", "BBB")),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-03",
    )

    aaa_sides = [trade.order.side for trade in result.trades if trade.order.symbol == "AAA"]
    bbb_trades = [trade for trade in result.trades if trade.order.symbol == "BBB"]

    assert aaa_sides == ["BUY", "SELL"]
    assert len(bbb_trades) == 1
    assert bbb_trades[0].order.side == "BUY"
    assert result.snapshots[1].positions["BBB"].shares == result.snapshots[0].positions["BBB"].shares


def test_signal_to_position_hold_retries_unfilled_exit() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
                "volume": [1_000_000, 1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }

    class BuySellHoldSignal:
        name = "BuySellHold"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return pd.Series(["BUY", "SELL", "HOLD"], index=mktdata.index)

    broker = DroppingSellBroker()
    strategy = Strategy(
        name="signal_hold_retries_unfilled_exit",
        universe=StaticUniverse(("AAA",)),
        signals={"timing": (BuySellHoldSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=broker,
        start="2024-01-02",
        end="2024-01-04",
    )

    assert [trade.order.side for trade in result.trades] == ["BUY"]
    assert [order.side for order in broker.dropped_sells] == ["SELL", "SELL"]
    assert result.snapshots[2].positions["AAA"].shares == result.snapshots[0].positions["AAA"].shares


def test_signal_to_position_mixed_hold_buy_uses_cash_budget_for_frozen_holdings() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 150.0, 150.0],
                "high": [100.0, 150.0, 150.0],
                "low": [100.0, 150.0, 150.0],
                "close": [100.0, 150.0, 150.0],
                "volume": [1_000_000, 1_000_000, 1_000_000],
                "timing_input": ["BUY", "HOLD", "HOLD"],
            },
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
                "volume": [1_000_000, 1_000_000, 1_000_000],
                "timing_input": ["HOLD", "BUY", "HOLD"],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    strategy = Strategy(
        name="signal_mixed_hold_buy_cash_budget",
        universe=StaticUniverse(("AAA", "BBB")),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing", buy_weight=0.5),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-04",
    )

    bbb_buys = [trade.order for trade in result.trades if trade.order.symbol == "BBB"]

    assert len(bbb_buys) == 1
    assert bbb_buys[0].side == "BUY"
    assert bbb_buys[0].shares == 500


def test_signal_to_position_missing_bar_latched_position_is_frozen() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1_000_000],
                "timing_input": ["BUY"],
            },
            index=dates[:1],
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 200.0],
                "high": [100.0, 200.0],
                "low": [100.0, 200.0],
                "close": [100.0, 200.0],
                "volume": [1_000_000, 1_000_000],
                "timing_input": ["BUY", "SELL"],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    strategy = Strategy(
        name="signal_missing_bar_latched_position_frozen",
        universe=StaticUniverse(("AAA", "BBB")),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing", buy_weight=0.5),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-03",
    )

    aaa_buy_orders = [
        managed.order for managed in result.orders
        if managed.order.symbol == "AAA" and managed.order.side == "BUY"
    ]

    assert len(aaa_buy_orders) == 1


def test_signal_to_position_mixed_sell_buy_uses_unfrozen_sell_proceeds() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
                "timing_input": ["BUY", "SELL"],
            },
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
                "timing_input": ["BUY", "HOLD"],
            },
            index=dates,
        ),
        "CCC": pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1_000_000, 1_000_000],
                "timing_input": ["HOLD", "BUY"],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    strategy = Strategy(
        name="signal_mixed_sell_buy_uses_sell_proceeds",
        universe=StaticUniverse(("AAA", "BBB", "CCC")),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing", buy_weight=0.5),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-03",
    )

    day_two_trades = [
        trade.order
        for trade in result.trades
        if trade.filled_at == dates[1].isoformat()
    ]

    assert [(order.symbol, order.side, order.shares) for order in day_two_trades] == [
        ("AAA", "SELL", 500),
        ("CCC", "BUY", 500),
    ]


def test_signal_to_position_completed_partial_sell_freezes_later_hold_symbol() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 200.0],
                "high": [100.0, 100.0, 100.0, 200.0],
                "low": [100.0, 100.0, 100.0, 200.0],
                "close": [100.0, 100.0, 100.0, 200.0],
                "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
                "timing_input": ["BUY", "SELL", "HOLD", "HOLD"],
            },
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0, 100.0],
                "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
                "timing_input": ["BUY", "HOLD", "HOLD", "BUY"],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    strategy = Strategy(
        name="signal_partial_sell_completed_hold_freezes",
        universe=StaticUniverse(("AAA", "BBB")),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing", buy_weight=0.5, sell_weight=0.25),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-05",
    )

    aaa_sides = [trade.order.side for trade in result.trades if trade.order.symbol == "AAA"]

    assert aaa_sides == ["BUY", "SELL"]


def test_signal_to_position_pending_partial_sell_keeps_reduction_state() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0, 10.0],
                "high": [100.0, 100.0, 10.0],
                "low": [100.0, 100.0, 10.0],
                "close": [100.0, 100.0, 10.0],
                "volume": [1_000_000, 1_000_000, 1_000_000],
                "timing_input": ["BUY", "SELL", "HOLD"],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    optimizer = SignalToPositionOptimizer(signal="timing", buy_weight=0.5, sell_weight=0.25)
    strategy = Strategy(
        name="signal_pending_partial_sell_keeps_reduction",
        universe=StaticUniverse(("AAA",)),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=optimizer,
    )
    broker = HoldingSellBroker()

    Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=broker,
        start="2024-01-02",
        end="2024-01-04",
    )

    assert [order.side for order in broker.held_sells] == ["SELL"]
    assert optimizer.pending_reduction_symbols == {"AAA"}


def test_signal_to_position_pending_next_close_buy_latches_later_sell() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0, 100.0],
                "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
                "timing_input": ["BUY", "SELL", "HOLD", "HOLD"],
            },
            index=dates,
        ),
    }

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    strategy = Strategy(
        name="signal_pending_next_close_buy_latches_sell",
        universe=StaticUniverse(("AAA",)),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=SignalToPositionOptimizer(signal="timing"),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(fill_price_mode=FillPriceMode.NEXT_CLOSE, market_calendar="XNYS"),
        start="2024-01-02",
        end="2024-01-05",
    )

    assert [(trade.order.side, trade.filled_at) for trade in result.trades] == [
        ("BUY", dates[1].isoformat()),
        ("SELL", dates[3].isoformat()),
    ]
    assert result.portfolio.positions == {}


def test_signal_to_position_optimizer_resets_between_engine_runs() -> None:
    dates = pd.bdate_range("2024-01-02", periods=1, tz="UTC")

    def frame(label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1_000_000],
                "timing_input": [label],
            },
            index=dates,
        )

    class ColumnSignal:
        name = "ColumnSignal"

        def compute(self, mktdata: pd.DataFrame) -> pd.Series:
            return mktdata["timing_input"]

    optimizer = SignalToPositionOptimizer(signal="timing")
    strategy = Strategy(
        name="signal_hold_reset_between_runs",
        universe=StaticUniverse(("AAA",)),
        signals={"timing": (ColumnSignal(), {})},
        portfolio=optimizer,
    )

    first = Engine().run(
        strategy,
        market=FakeMarketDataProvider({"AAA": frame("BUY")}),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-02",
    )
    second = Engine().run(
        strategy,
        market=FakeMarketDataProvider({"AAA": frame("HOLD")}),
        broker=SimBroker(),
        start="2024-01-02",
        end="2024-01-02",
    )

    assert [trade.order.side for trade in first.trades] == ["BUY"]
    assert second.trades == []
    assert second.snapshots[0].target_weights == {"CASH": 1.0}


def test_engine_notifies_optimizer_after_normal_broker_full_exit() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [10.0, 10.0],
                "low": [10.0, 10.0],
                "close": [10.0, 10.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }
    optimizer = ResetAwareAlwaysBuyOptimizer()
    strategy = Strategy(
        name="normal_exit_reset",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=optimizer,
    )

    Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=CloseSellBroker(),
        start="2024-01-01",
        end="2024-01-02",
    )

    assert optimizer.reset_symbols_seen == ["AAA"]


def test_engine_next_open_fills_on_next_bar_before_optimization() -> None:
    """NEXT_OPEN market orders should fill on the next bar, before new targets."""
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 12.0, 14.0],
                "high": [10.0, 12.0, 14.0],
                "low": [10.0, 12.0, 14.0],
                "close": [12.0, 12.0, 14.0],
                "volume": [1_000_000, 1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }
    strategy = Strategy(
        name="next_open_causal",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS"),
        start="2024-01-01",
        end="2024-01-03",
    )

    buy_trades = [t for t in result.trades if t.order.side == "BUY"]
    assert len(buy_trades) == 1
    assert buy_trades[0].filled_price == Decimal("12")
    assert buy_trades[0].filled_at == dates[1].isoformat()


def test_engine_rejects_next_open_buy_when_gap_up_exceeds_cash() -> None:
    """A next-open gap up should not create negative cash."""
    dates = pd.bdate_range("2024-01-01", periods=2, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 20.0],
                "high": [10.0, 20.0],
                "low": [10.0, 20.0],
                "close": [10.0, 20.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }
    strategy = Strategy(
        name="next_open_gap",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS"),
        start="2024-01-01",
        end="2024-01-02",
        initial_cash=100.0,
    )

    assert result.trades == []
    assert result.portfolio.positions == {}
    assert result.portfolio.cash == Decimal("100.0")
    rejected_orders = [order for order in result.orders if order.status == "rejected"]
    assert len(rejected_orders) == 1
    assert rejected_orders[0].status_reason == "insufficient_cash"
    assert rejected_orders[0].order.symbol == "AAA"


def test_engine_cancels_pending_next_open_buy_before_exit_sell() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    data = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 5.0, 5.0],
                "high": [10.0, 5.0, 5.0],
                "low": [10.0, 5.0, 5.0],
                "close": [10.0, 5.0, 5.0],
                "volume": [1_000_000, 1_000_000, 1_000_000],
            },
            index=dates,
        ),
    }
    strategy = Strategy(
        name="cancel_pending_buy_before_exit",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )

    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN, market_calendar="XNYS"),
        start="2024-01-01",
        end="2024-01-03",
        initial_cash=1000.0,
        rules=[AlwaysExitRule()],
    )

    canceled_buys = [order for order in result.orders if order.order.side == "BUY" and order.status == "canceled"]
    assert result.portfolio.positions == {}
    assert canceled_buys
    assert all(order.status_reason == "exit_sell_submitted" for order in canceled_buys)


def test_engine_lot_size() -> None:
    """lot_size=100 rounds trade shares to multiples of 100."""
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    closes = [1.0, 1.0, 1.0]

    data = {
        "AAA": pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1_000_000] * 3,
            },
            index=dates,
        ),
    }

    strategy = Strategy(
        name="lot_size_test",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )

    broker = SimBroker()
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=broker,
        start="2024-01-01",
        end="2024-12-31",
        initial_cash=250.0,
        lot_size=100,
    )

    # 250 / 1.0 = 250, rounded down to 200 with lot_size=100
    buy_trades = [t for t in result.trades if t.order.side == "BUY"]
    assert len(buy_trades) > 0
    for t in buy_trades:
        assert t.order.shares % 100 == 0, f"shares {t.order.shares} not multiple of 100"
    # First buy should be 200 shares
    assert buy_trades[0].order.shares == 200


def test_engine_cash_annual_return() -> None:
    """cash_annual_return accrues interest on idle cash."""
    dates = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    closes = [100.0] * 5

    data = {
        "AAA": pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1_000_000] * 5,
            },
            index=dates,
        ),
    }

    strategy = Strategy(
        name="cash_return_test",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=NeverBuyOptimizer(),
    )

    initial = 1_000_000.0
    annual_rate = 0.025
    broker = SimBroker()
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=broker,
        start="2024-01-01",
        end="2024-12-31",
        initial_cash=initial,
        cash_annual_return=annual_rate,
    )

    daily_rate = (1 + annual_rate) ** (1 / 252) - 1
    expected = initial * (1 + daily_rate) ** 5
    final_equity = result.equity_curve[-1][1]
    assert final_equity > initial, "Cash interest should increase equity"
    assert abs(final_equity - expected) < 0.01, (
        f"Expected ~{expected:.2f}, got {final_equity:.2f}"
    )
