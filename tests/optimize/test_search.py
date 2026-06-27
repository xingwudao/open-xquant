"""Tests for GridSearch — parameter optimization and helpers."""

from decimal import Decimal

import pandas as pd
import pytest

from oxq.core.strategy import Strategy
from oxq.core.types import Portfolio, RuleResult
from oxq.indicators.sma import SMA
from oxq.optimize.paramset import ParameterSet
from oxq.optimize.search import (
    METRIC_DIRECTIONS,
    GridSearch,
    SearchResult,
    TrialResult,
    _apply_params,
    _extract_metric,
    _resolve_direction,
)
from oxq.portfolio.analytics import RunResult
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.signals.crossover import Crossover
from oxq.trade.sim_broker import SimBroker
from oxq.universe.static import StaticUniverse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMarketDataProvider:
    """In-memory market data provider for testing."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC"))]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_result(values: list[float]) -> RunResult:
    """Build a RunResult from a sequence of portfolio values."""
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    equity_curve = [(d, v) for d, v in zip(dates, values)]
    return RunResult(
        portfolio=Portfolio(cash=Decimal(str(values[-1])) if values else Decimal("0")),
        trades=[],
        equity_curve=equity_curve,
        mktdata={},
    )


def _make_trending_data(n: int = 120) -> dict[str, pd.DataFrame]:
    """Trending price data: down → up → down for SMA crossover signals."""
    dates = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 → 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 → 180
    for i in range(n - 90):
        closes.append(180 - i * 2)       # 180 → ...
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


def _make_crossover_signal(fast: int = 10, slow: int = 50):
    """Create a Crossover signal with required_indicators."""
    signal = Crossover()
    signal.required_indicators = {
        "sma_fast": (SMA(), {"period": fast}),
        "sma_slow": (SMA(), {"period": slow}),
    }
    return signal


def _make_strategy(fast: int = 10, slow: int = 50) -> Strategy:
    return Strategy(
        name="test_sma",
        hypothesis="SMA crossover",
        universe=StaticUniverse(("AAPL",)),
        signals={
            "sma_cross": (_make_crossover_signal(fast, slow), {"fast": "sma_fast", "slow": "sma_slow"}),
        },
        portfolio=EqualWeightOptimizer(),
    )


# ---------------------------------------------------------------------------
# _resolve_direction
# ---------------------------------------------------------------------------


def test_resolve_direction_explicit() -> None:
    assert _resolve_direction("sharpe_ratio", "maximize") == "maximize"
    assert _resolve_direction("sharpe_ratio", "minimize") == "minimize"


def test_resolve_direction_from_registry() -> None:
    assert _resolve_direction("sharpe_ratio", None) == "maximize"
    assert _resolve_direction("annualized_volatility", None) == "minimize"
    assert _resolve_direction("max_drawdown", None) == "maximize"


def test_resolve_direction_unknown_defaults_maximize() -> None:
    assert _resolve_direction("unknown_metric", None) == "maximize"


def test_resolve_direction_callable_defaults_maximize() -> None:
    assert _resolve_direction(lambda r: 0.0, None) == "maximize"


def test_resolve_direction_invalid_raises() -> None:
    with pytest.raises(ValueError, match="metric_direction"):
        _resolve_direction("sharpe_ratio", "invalid")


# ---------------------------------------------------------------------------
# _extract_metric
# ---------------------------------------------------------------------------


def test_extract_metric_by_name() -> None:
    result = _make_result([100, 110, 115])
    val = _extract_metric(result, "total_return")
    assert val == pytest.approx(0.15, rel=1e-4)


def test_extract_metric_by_callable() -> None:
    result = _make_result([100, 110, 115])
    val = _extract_metric(result, lambda r: r.total_return() * 2)
    assert val == pytest.approx(0.30, rel=1e-4)


def test_extract_metric_unknown_raises() -> None:
    result = _make_result([100, 110])
    with pytest.raises(ValueError, match="Unknown metric"):
        _extract_metric(result, "nonexistent_metric")


# ---------------------------------------------------------------------------
# _apply_params
# ---------------------------------------------------------------------------


def test_apply_params_modifies_signal() -> None:
    strategy = _make_strategy()
    params = {"sma_cross": {"fast": "sma_new", "slow": "sma_slow"}}
    new = _apply_params(strategy, params)
    assert new.signals["sma_cross"][1]["fast"] == "sma_new"


def test_apply_params_preserves_name_and_universe() -> None:
    strategy = _make_strategy()
    new = _apply_params(strategy, {})
    assert new.name == strategy.name
    assert new.universe is strategy.universe


def test_apply_params_unmatched_names_are_ignored() -> None:
    """Params for names not in the strategy are silently ignored."""
    strategy = _make_strategy()
    params = {"NonexistentComponent": {"threshold": 0.99}}
    new = _apply_params(strategy, params)
    # Strategy unchanged
    assert new.name == strategy.name


# ---------------------------------------------------------------------------
# SearchResult — edge cases
# ---------------------------------------------------------------------------


def test_search_result_top_n_larger_than_results() -> None:
    """top_n(n) with n > len returns all results."""
    trials = [_make_trial(0.5), _make_trial(0.8)]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    top5 = sr.top_n(5)
    assert len(top5) == 2
    assert top5[0].metric_value == 0.8


def test_search_result_to_dataframe_with_signal_params() -> None:
    """to_dataframe flattens signal params as component.param columns."""
    trials = [
        TrialResult(
            params={
                "sma_cross": {"fast": "sma_10"},
            },
            metric_value=0.5,
            run_result=_make_result([100, 105, 110]),
        ),
    ]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    df = sr.to_dataframe()
    assert "sma_cross.fast" in df.columns
    assert df["sma_cross.fast"].iloc[0] == "sma_10"


# ---------------------------------------------------------------------------
# METRIC_DIRECTIONS registry
# ---------------------------------------------------------------------------


def test_metric_directions_has_standard_metrics() -> None:
    assert "sharpe_ratio" in METRIC_DIRECTIONS
    assert "total_return" in METRIC_DIRECTIONS
    assert "max_drawdown" in METRIC_DIRECTIONS
    assert "annualized_volatility" in METRIC_DIRECTIONS


def test_max_drawdown_direction_is_maximize() -> None:
    """max_drawdown is negative; maximize = least bad."""
    assert METRIC_DIRECTIONS["max_drawdown"] == "maximize"


# ---------------------------------------------------------------------------
# TrialResult / SearchResult
# ---------------------------------------------------------------------------


def _make_trial(metric_value: float, values: list[float] | None = None) -> TrialResult:
    if values is None:
        values = [100, 100 + metric_value * 100]
    return TrialResult(
        params={"sma": {"period": int(abs(metric_value) * 100)}},
        metric_value=metric_value,
        run_result=_make_result(values),
    )


def test_search_result_best_maximize() -> None:
    trials = [_make_trial(0.5), _make_trial(1.2), _make_trial(0.8)]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    assert sr.best.metric_value == 1.2


def test_search_result_best_minimize() -> None:
    trials = [_make_trial(0.5), _make_trial(0.1), _make_trial(0.8)]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="annualized_volatility",
        metric_direction="minimize",
    )
    assert sr.best.metric_value == 0.1


def test_search_result_top_n() -> None:
    trials = [_make_trial(v) for v in [0.1, 0.5, 0.3, 0.9, 0.7]]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    top3 = sr.top_n(3)
    assert len(top3) == 3
    assert [t.metric_value for t in top3] == [0.9, 0.7, 0.5]


def test_search_result_to_dataframe() -> None:
    trials = [
        TrialResult(
            params={"sma_fast": {"period": 10}, "sma_slow": {"period": 50}},
            metric_value=0.5,
            run_result=_make_result([100, 105, 110]),
        ),
    ]
    sr = SearchResult(
        all_results=trials,
        paramset=ParameterSet("test"),
        metric="sharpe_ratio",
        metric_direction="maximize",
    )
    df = sr.to_dataframe()
    assert len(df) == 1
    assert "sma_fast.period" in df.columns
    assert "sma_slow.period" in df.columns
    assert "metric_value" in df.columns
    assert "total_return" in df.columns
    assert "sharpe_ratio" in df.columns
    assert df["sma_fast.period"].iloc[0] == 10


# ---------------------------------------------------------------------------
# GridSearch.run — integration
# ---------------------------------------------------------------------------


def test_grid_search_runs_all_combos() -> None:
    """GridSearch evaluates every valid parameter combination."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("test")
    ps.add("sma_cross", "fast", values=["sma_fast"])

    strategy = _make_strategy()
    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric="sharpe_ratio",
    )

    assert len(result.all_results) == 1
    assert result.metric == "sharpe_ratio"
    assert result.metric_direction == "maximize"


def test_grid_search_uses_strategy_rules_when_no_explicit_rules() -> None:
    class AlwaysTrueSignal:
        name = "AlwaysTrue"

        def compute(self, mktdata, **params):
            return pd.Series(True, index=mktdata.index)

    class HoldAllRule:
        name = "HoldAllRule"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, symbol, row, portfolio, prices=None):
            self.calls += 1
            return RuleResult(hold=True, reason="hold all")

    data = _make_trending_data(n=120)
    market = FakeMarketDataProvider(data)
    strategy = Strategy(
        name="rule_strategy",
        universe=StaticUniverse(("AAPL",)),
        signals={"always": (AlwaysTrueSignal(), {})},
        portfolio=EqualWeightOptimizer(),
        rules=[HoldAllRule()],
    )

    result = GridSearch(ParameterSet("test")).run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-01-31",
        metric="total_return",
    )

    assert len(result.all_results) == 1
    assert result.all_results[0].run_result.trades == []
    assert strategy.rules[0].calls == 0


def test_grid_search_custom_metric() -> None:
    """GridSearch with a callable metric works."""
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("test")
    ps.add("sma_cross", "fast", values=["sma_fast"])

    strategy = _make_strategy()
    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric=lambda r: r.total_return() + r.sharpe_ratio(),
    )

    assert len(result.all_results) == 1
    assert result.metric == "<custom>"


def test_grid_search_minimize_direction() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    ps = ParameterSet("test")
    ps.add("sma_cross", "fast", values=["sma_fast"])

    strategy = _make_strategy()
    gs = GridSearch(ps)
    result = gs.run(
        strategy=strategy,
        market=market,
        broker_factory=SimBroker,
        start="2024-01-01",
        end="2024-12-31",
        metric="annualized_volatility",
    )

    assert result.metric_direction == "minimize"
