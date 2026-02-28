"""Tests for BacktestEngine — full pipeline integration."""

import pandas as pd

from oxq.backtest.broker import SimBroker
from oxq.backtest.engine import BacktestEngine
from oxq.core.strategy import Strategy
from oxq.core.types import Portfolio
from oxq.indicators.sma import SMA
from oxq.rules.entry import EntryRule
from oxq.rules.exit import ExitRule
from oxq.signals.crossover import Crossover
from oxq.universe.static import StaticUniverse


class FakeMarketDataProvider:
    """In-memory market data provider for testing."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= start) & (df.index <= end)]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_trending_data() -> dict[str, pd.DataFrame]:
    """Create data: downtrend → uptrend → downtrend to trigger crossover signals.

    Structure (120 bars):
    - Bars 0-49:  downtrend 200 → 102 (SMA10 < SMA50 at bar 49)
    - Bars 50-89: uptrend 102 → 182 (SMA10 crosses above SMA50 → golden cross)
    - Bars 90-119: downtrend 182 → 122 (SMA10 crosses below SMA50 → death cross)
    """
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 → 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 → 180
    for i in range(30):
        closes.append(180 - i * 2)       # 180 → 122

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
    return Strategy(
        name="test_sma_crossover",
        hypothesis="SMA10 crossing above SMA50 predicts positive returns",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_10": (SMA(), {"period": 10}),
            "sma_50": (SMA(), {"period": 50}),
        },
        signals={
            "sma_10_x_sma_50": (Crossover(), {"fast": "sma_10", "slow": "sma_50"}),
        },
        entry_rules=[EntryRule(signal="sma_10_x_sma_50", shares=100)],
        exit_rules=[ExitRule(fast="sma_10", slow="sma_50")],
    )


def test_engine_full_pipeline() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = BacktestEngine()

    result = engine.run(
        strategy, market=market, broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
    )

    # Should have at least some trades
    assert len(result.trades) > 0
    # All trades should be for AAPL
    assert all(t.order.symbol == "AAPL" for t in result.trades)
    # Equity curve should have one entry per bar
    assert len(result.equity_curve) == 120
    # mktdata should have indicator and signal columns
    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_50" in df.columns
    assert "sma_10_x_sma_50" in df.columns


def test_engine_run_through_indicator() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = BacktestEngine()

    result = engine.run(
        strategy, market=market, broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
        run_through="indicator",
    )

    # Indicators computed, but no signals or trades
    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_50" in df.columns
    assert "sma_10_x_sma_50" not in df.columns
    assert len(result.trades) == 0
    assert len(result.equity_curve) == 0


def test_engine_run_through_signal() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = BacktestEngine()

    result = engine.run(
        strategy, market=market, broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
        run_through="signal",
    )

    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_10_x_sma_50" in df.columns
    assert len(result.trades) == 0


def test_engine_portfolio_cash_changes() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = BacktestEngine()

    result = engine.run(
        strategy, market=market, broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
        initial_cash=100_000.0,
    )

    # If any trades happened, cash should differ from initial
    if len(result.trades) > 0:
        # Either we still hold a position, or cash changed from fills
        has_position = len(result.portfolio.positions) > 0
        cash_changed = result.portfolio.cash != 100_000.0
        assert has_position or cash_changed


def test_engine_metrics() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = BacktestEngine()

    result = engine.run(
        strategy, market=market, broker=SimBroker(),
        start="2024-01-01", end="2024-12-31",
    )

    # Metrics should return numbers without errors
    tr = result.total_return()
    sr = result.sharpe_ratio()
    mdd = result.max_drawdown()
    assert isinstance(tr, float)
    assert isinstance(sr, float)
    assert isinstance(mdd, float)
    assert mdd <= 0.0  # drawdown is always <= 0


def test_apply_fill_buy() -> None:
    from oxq.backtest.engine import _apply_fill
    from oxq.core.types import Fill, Order

    portfolio = Portfolio(cash=100_000.0)
    fill = Fill(
        order=Order(symbol="AAPL", side="BUY", shares=100),
        filled_price=150.0,
        filled_at="2024-01-02",
    )
    _apply_fill(portfolio, fill)

    assert portfolio.cash == 85_000.0  # 100000 - 100*150
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].shares == 100
    assert portfolio.positions["AAPL"].avg_cost == 150.0


def test_apply_fill_sell() -> None:
    from oxq.backtest.engine import _apply_fill
    from oxq.core.types import Fill, Order, Position

    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=150.0)},
    )
    fill = Fill(
        order=Order(symbol="AAPL", side="SELL", shares=100),
        filled_price=160.0,
        filled_at="2024-03-01",
    )
    _apply_fill(portfolio, fill)

    assert portfolio.cash == 66_000.0  # 50000 + 100*160
    assert "AAPL" not in portfolio.positions
