"""Tests for Engine bar snapshot recording."""

import pandas as pd

from oxq.core.engine import Engine
from oxq.core.strategy import Strategy
from oxq.core.types import BarSnapshot
from oxq.trade.sim_broker import SimBroker
from oxq.universe.static import StaticUniverse


class FakeMarketDataProvider:
    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC"))]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_simple_data(n: int = 5) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    closes = [100.0 + i for i in range(n)]
    return {
        "AAA": pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        ),
    }


def _make_two_symbol_data(n: int = 5) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    return {
        symbol: pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        )
        for symbol, closes in {
            "AAA": [100.0 + i for i in range(n)],
            "BBB": [50.0 + i for i in range(n)],
        }.items()
    }


class AlwaysBuyOptimizer:
    name = "always_buy"

    def optimize(self, signals, indicators):
        symbols = list(signals.keys())
        if symbols:
            return {symbols[0]: 1.0}
        return {"CASH": 1.0}


def test_run_records_snapshots() -> None:
    """run() should produce one BarSnapshot per bar."""
    data = _make_simple_data(5)
    strategy = Strategy(
        name="snap_test",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
    )

    assert len(result.snapshots) == 5
    for snap in result.snapshots:
        assert isinstance(snap, BarSnapshot)


def test_snapshot_target_weights_before_rules() -> None:
    """target_weights should reflect raw optimizer output."""
    data = _make_simple_data(3)
    strategy = Strategy(
        name="snap_weights",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
    )

    # AlwaysBuyOptimizer returns {"AAA": 1.0}
    assert result.snapshots[0].target_weights == {"AAA": 1.0}


def test_snapshot_positions_after_execution() -> None:
    """positions should reflect holdings after fills."""
    data = _make_simple_data(3)
    strategy = Strategy(
        name="snap_pos",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
        initial_cash=10000.0,
    )

    # After first bar, should have bought AAA
    snap = result.snapshots[0]
    assert "AAA" in snap.positions
    assert snap.positions["AAA"].shares > 0
    assert snap.positions["AAA"].avg_cost > 0
    assert snap.cash >= 0
    assert snap.total_value > 0


def test_step_mode_records_snapshots() -> None:
    """setup() + step() should produce same snapshots as run()."""
    data = _make_simple_data(5)
    strategy = Strategy(
        name="snap_step",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )

    engine = Engine()
    engine.setup(
        strategy=strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
    )
    for date in engine.dates:
        engine.step(date)

    result = engine.result
    assert len(result.snapshots) == 5
    for snap in result.snapshots:
        assert isinstance(snap, BarSnapshot)


def test_snapshot_adjusted_weights_with_hold_rule() -> None:
    """A first-bar hold should record current portfolio weights."""
    from oxq.core.types import RuleResult

    class AlwaysHoldRule:
        name = "always_hold"

        def evaluate(self, symbol, row, portfolio, prices=None):
            return RuleResult(hold=True, reason="test hold")

    data = _make_simple_data(3)
    strategy = Strategy(
        name="snap_hold",
        universe=StaticUniverse(("AAA",)),
        signals={},
        portfolio=AlwaysBuyOptimizer(),
    )
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
        rules=[AlwaysHoldRule()],
    )

    # target_weights should be optimizer output
    assert result.snapshots[0].target_weights == {"AAA": 1.0}
    # adjusted_weights reflects the executable frozen state.
    assert result.snapshots[0].adjusted_weights == {"CASH": 1.0}
    # No positions since hold prevented trading
    assert result.snapshots[0].positions == {}


def test_snapshot_adjusted_weights_freeze_when_hold_rule_fires() -> None:
    """Hold rules should freeze adjusted weights when signals change."""
    from oxq.core.types import RuleResult

    class FlipOptimizer:
        name = "flip"

        def __init__(self) -> None:
            self.calls = 0

        def optimize(self, signals, indicators):
            self.calls += 1
            if self.calls == 1:
                return {"AAA": 1.0}
            return {"BBB": 1.0}

    class HoldAfterFirstBarRule:
        name = "hold_after_first_bar"

        def __init__(self) -> None:
            self.dates: set[pd.Timestamp] = set()

        def evaluate(self, symbol, row, portfolio, prices=None):
            if row.name in self.dates:
                return RuleResult()
            self.dates.add(row.name)
            if len(self.dates) == 1:
                return RuleResult()
            return RuleResult(hold=True, reason="rebalance skipped")

    data = _make_two_symbol_data(3)
    strategy = Strategy(
        name="snap_hold_freeze",
        universe=StaticUniverse(("AAA", "BBB")),
        signals={},
        portfolio=FlipOptimizer(),
    )
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        broker=SimBroker(),
        start="2024-01-01",
        end="2024-12-31",
        rules=[HoldAfterFirstBarRule()],
    )

    assert result.snapshots[0].adjusted_weights == {"AAA": 1.0}
    assert result.snapshots[1].target_weights == {"BBB": 1.0}
    assert result.snapshots[1].adjusted_weights == {"AAA": 1.0}
    assert result.snapshots[1].rule_reasons == {"__all__": "rebalance skipped"}
