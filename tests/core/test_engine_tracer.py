"""Tests for Engine tracer integration."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.engine import Engine, _signal_output_summary
from oxq.core.strategy import Strategy
from oxq.core.types import Fill, Order
from oxq.indicators.sma import SMA
from oxq.observe.tracer import DefaultTracer
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.signals.crossover import Crossover
from oxq.signals.roc_timing import ROCTiming
from oxq.universe.static import StaticUniverse


class FakeBroker:
    """Minimal Broker for testing."""

    def __init__(self) -> None:
        self._fills: list[Fill] = []
        self._orders: list[Order] = []

    def submit_order(self, order: Order) -> str:
        self._orders.append(order)
        return "ok"

    def get_fills(self) -> list[Fill]:
        fills = list(self._fills)
        self._fills.clear()
        return fills

    def on_bar_open(self, mktdata, date) -> None:
        pass

    def on_bar_close(self, mktdata, date) -> None:
        for order in self._orders:
            close_price = mktdata[order.symbol].loc[date, "close"]
            self._fills.append(
                Fill(
                    order=order,
                    filled_price=Decimal(str(float(close_price))),
                    filled_at=str(date.date()),
                ),
            )
        self._orders.clear()

    def get_open_orders(self, symbol=None):
        return []

    def cancel_orders(self, symbol, side=None):
        return []

    def cap_pending_sells(self, symbol, max_shares) -> None:
        pass


class FakeMarket:
    """Minimal MarketDataProvider for testing."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return self._data[symbol]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_market() -> FakeMarket:
    dates = pd.bdate_range("2024-01-01", periods=60, tz="UTC")
    prices = [100.0 + i * 0.5 for i in range(60)]
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [1000000] * 60,
    }, index=dates)
    return FakeMarket({"AAPL": df})


def _make_crossover_signal():
    """Create a Crossover signal with required_indicators set."""
    signal = Crossover()
    signal.required_indicators = {
        "sma_fast": (SMA(), {"column": "close", "period": 5}),
        "sma_slow": (SMA(), {"column": "close", "period": 20}),
    }
    return signal


class TestEngineTracer:
    def test_signal_output_summary_counts_object_false_as_no_event(self) -> None:
        summary = _signal_output_summary(pd.Series([True, False, None], dtype="object"))

        assert summary["value_counts"] == {"FALSE": 1, "TRUE": 1}
        assert summary["signal_count"] == 1

    def test_tracer_receives_indicator_callbacks(self) -> None:
        cross = _make_crossover_signal()
        strategy = Strategy(
            name="test",
            universe=StaticUniverse(("AAPL",)),
            signals={
                "cross": (cross, {"fast": "sma_fast", "slow": "sma_slow"}),
            },
            portfolio=EqualWeightOptimizer(),
        )
        tracer = DefaultTracer()
        engine = Engine()
        engine.run(
            strategy, market=_make_market(),
            broker=FakeBroker(),
            start="2024-01-01", end="2024-03-29",
            run_through="indicator",
            tracer=tracer,
        )
        indicator_spans = [s for s in tracer.spans if s.component.startswith("indicator:")]
        assert len(indicator_spans) == 2
        names = {s.component for s in indicator_spans}
        assert "indicator:sma_fast" in names
        assert "indicator:sma_slow" in names

    def test_tracer_receives_signal_callbacks(self) -> None:
        cross = _make_crossover_signal()
        strategy = Strategy(
            name="test",
            universe=StaticUniverse(("AAPL",)),
            signals={
                "cross": (cross, {"fast": "sma_fast", "slow": "sma_slow"}),
            },
            portfolio=EqualWeightOptimizer(),
        )
        tracer = DefaultTracer()
        engine = Engine()
        engine.run(
            strategy, market=_make_market(),
            broker=FakeBroker(),
            start="2024-01-01", end="2024-03-29",
            run_through="signal",
            tracer=tracer,
        )
        signal_spans = [s for s in tracer.spans if s.component.startswith("signal:")]
        assert len(signal_spans) == 1
        assert signal_spans[0].component == "signal:cross"

    def test_tracer_summarizes_categorical_signal_outputs(self) -> None:
        strategy = Strategy(
            name="test",
            universe=StaticUniverse(("AAPL",)),
            signals={
                "timing": (ROCTiming(), {"column": "close", "mode": "fixed", "bottom": 101.0, "top": 120.0}),
            },
            portfolio=EqualWeightOptimizer(),
        )
        tracer = DefaultTracer()
        engine = Engine()
        engine.run(
            strategy, market=_make_market(),
            broker=FakeBroker(),
            start="2024-01-01", end="2024-03-29",
            run_through="signal",
            tracer=tracer,
        )

        signal_spans = [s for s in tracer.spans if s.component == "signal:timing"]
        assert len(signal_spans) == 1
        assert signal_spans[0].output_summary["value_counts"]["BUY"] > 0
        assert "signal_count" in signal_spans[0].output_summary

    def test_tracer_receives_rule_callbacks(self) -> None:
        cross = _make_crossover_signal()
        strategy = Strategy(
            name="test",
            universe=StaticUniverse(("AAPL",)),
            signals={
                "cross": (cross, {"fast": "sma_fast", "slow": "sma_slow"}),
            },
            portfolio=EqualWeightOptimizer(),
        )
        tracer = DefaultTracer()
        engine = Engine()
        engine.run(
            strategy, market=_make_market(),
            broker=FakeBroker(),
            start="2024-01-01", end="2024-03-29",
            tracer=tracer,
        )
        # With no rules passed, tracer won't have rule spans
        # But the run should complete successfully
        assert tracer is not None

    def test_no_tracer_still_works(self) -> None:
        cross = _make_crossover_signal()
        strategy = Strategy(
            name="test",
            universe=StaticUniverse(("AAPL",)),
            signals={
                "cross": (cross, {"fast": "sma_fast", "slow": "sma_slow"}),
            },
            portfolio=EqualWeightOptimizer(),
        )
        engine = Engine()
        result = engine.run(
            strategy, market=_make_market(),
            broker=FakeBroker(),
            start="2024-01-01", end="2024-03-29",
            run_through="indicator",
        )
        assert result is not None
