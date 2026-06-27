"""Tests for strategy tools."""

from __future__ import annotations

import pickle

import pytest

from oxq.core.registry import list_indicators, list_rules
from oxq.core.strategy import Strategy
from oxq.tools import session
from oxq.tools.strategy import (
    indicator_describe,
    indicator_list,
    strategy_add_rule,
    strategy_add_signal,
    strategy_create,
    strategy_inspect,
)


@pytest.fixture(autouse=True)
def _reset_session():
    """Reset session state before each test."""
    session.clear()


# ---------------------------------------------------------------------------
# strategy_create
# ---------------------------------------------------------------------------


def test_strategy_create() -> None:
    result = strategy_create(
        name="test",
        hypothesis="SMA crossover predicts returns",
        objectives={"total_return": {"min": 0.05}},
        benchmarks=["SPY"],
    )
    assert result["name"] == "test"
    assert result["hypothesis"] == "SMA crossover predicts returns"
    assert result["objectives"] == {"total_return": {"min": 0.05}}
    assert result["benchmarks"] == ["SPY"]
    assert "test" in session._strategies
    assert getattr(session._strategies["test"], "_legacy_universe", None) is None


def test_strategy_create_missing_hypothesis() -> None:
    result = strategy_create(name="bad", hypothesis="", objectives={"r": {"min": 0.0}})
    assert "error" in result


def test_strategy_create_missing_objectives() -> None:
    result = strategy_create(name="bad", hypothesis="test hypothesis")
    assert "error" in result


# ---------------------------------------------------------------------------
# strategy_add_signal (with indicators)
# ---------------------------------------------------------------------------


def test_strategy_add_signal() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_signal(
        strategy="s1",
        name="cross",
        type="Crossover",
        params={"fast": "sma_10", "slow": "sma_50"},
        indicators={
            "sma_10": {"type": "SMA", "params": {"column": "close", "period": 10}},
            "sma_50": {"type": "SMA", "params": {"column": "close", "period": 50}},
        },
    )
    assert result["signal"] == "cross"
    assert result["type"] == "Crossover"
    assert "sma_10" in result["indicators"]
    strat = session._strategies["s1"]
    assert "cross" in strat.signals
    signal = strat.signals["cross"][0]
    assert "sma_10" in signal.required_indicators
    assert "sma_50" in signal.required_indicators


def test_strategy_add_signal_without_indicators() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_signal(
        strategy="s1",
        name="cross",
        type="Crossover",
        params={"fast": "sma_10", "slow": "sma_50"},
    )
    assert result["signal"] == "cross"
    assert result["indicators"] == []


def test_strategy_add_signal_unknown_indicator_type() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_signal(
        strategy="s1",
        name="cross",
        type="Crossover",
        params={"fast": "x", "slow": "y"},
        indicators={"x": {"type": "FooBar", "params": {}}},
    )
    assert "error" in result
    assert "FooBar" in result["error"]


def test_strategy_add_signal_unknown_type() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_signal(strategy="s1", name="x", type="MACD")
    assert "error" in result


# ---------------------------------------------------------------------------
# strategy_add_rule
# ---------------------------------------------------------------------------


def test_strategy_add_rule_exit() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_rule(
        strategy="s1",
        name="sell_on_cross",
        type="ExitRule",
        params={"fast": "sma_10", "slow": "sma_50"},
    )
    assert result["type"] == "ExitRule"
    strat = session._strategies["s1"]
    pending = getattr(strat, "_pending_rules", [])
    assert len(pending) == 1
    assert pending[0].fast == "sma_10"


def test_strategy_add_rule_unknown_type() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_add_rule(strategy="s1", name="x", type="StopLoss")
    assert "error" in result


# ---------------------------------------------------------------------------
# strategy_inspect
# ---------------------------------------------------------------------------


def test_strategy_inspect() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_signal(
        strategy="s1", name="cross", type="Crossover",
        params={"fast": "sma_10", "slow": "sma_50"},
        indicators={
            "sma_10": {"type": "SMA", "params": {"column": "close", "period": 10}},
            "sma_50": {"type": "SMA", "params": {"column": "close", "period": 50}},
        },
    )
    strategy_add_rule(
        strategy="s1", name="sell", type="ExitRule",
        params={"fast": "sma_10", "slow": "sma_50"},
    )

    result = strategy_inspect("s1")
    assert result["name"] == "s1"
    assert result["universe_binding"] == "none"
    assert "cross" in result["signals"]
    assert "indicators" in result["signals"]["cross"]
    assert "sma_10" in result["signals"]["cross"]["indicators"]
    assert len(result["rules"]) == 1


def test_strategy_inspect_not_found() -> None:
    result = strategy_inspect("missing")
    assert "error" in result


def test_strategy_set_universe_sets_run_default_not_strategy_body() -> None:
    from oxq.tools.strategy import strategy_set_universe

    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    result = strategy_set_universe(strategy="s1", type="static", symbols=["AAPL", "MSFT"])

    assert "error" not in result
    assert getattr(session._strategies["s1"], "_legacy_universe", None) is None
    assert tuple(session._strategy_universes["s1"].symbols) == ("AAPL", "MSFT")
    inspect = strategy_inspect("s1")
    assert inspect["universe"] == ["AAPL", "MSFT"]
    assert inspect["universe_binding"] == "run_default"


def test_strategy_create_clears_existing_run_default_universe() -> None:
    from oxq.tools.strategy import strategy_set_universe

    strategy_create(name="s1", hypothesis="old", objectives={"r": {"min": 0.0}})
    strategy_set_universe(strategy="s1", type="static", symbols=["AAPL"])

    result = strategy_create(name="s1", hypothesis="new", objectives={"r": {"min": 0.0}})

    assert "error" not in result
    assert "s1" not in session._strategy_universes
    inspect = strategy_inspect("s1")
    assert inspect["universe_binding"] == "none"


def test_session_load_migrates_legacy_strategy_universe(tmp_path, monkeypatch) -> None:
    from oxq.portfolio.optimizers import EqualWeightOptimizer
    from oxq.universe.static import StaticUniverse

    universe = StaticUniverse(("AAPL", "MSFT"))
    strategy = Strategy(name="legacy", signals={}, portfolio=EqualWeightOptimizer())
    strategy.__dict__["universe"] = universe
    strategy.__dict__.pop("_legacy_universe", None)

    session_file = tmp_path / "legacy_session.pkl"
    with open(session_file, "wb") as f:
        pickle.dump({"strategies": {"legacy": strategy}}, f)

    monkeypatch.setattr(session, "_SESSION_FILE", session_file)
    session._load()

    assert "legacy" in session._strategies
    assert tuple(session._strategy_universes["legacy"].symbols) == ("AAPL", "MSFT")


def test_session_load_migrates_legacy_pending_rules(tmp_path, monkeypatch) -> None:
    from oxq.portfolio.optimizers import EqualWeightOptimizer
    from oxq.rules.constraint import RebalanceFrequencyRule

    strategy = Strategy(name="legacy", signals={}, portfolio=EqualWeightOptimizer())
    strategy.__dict__["_pending_rules"] = [RebalanceFrequencyRule(interval_days=5)]
    strategy.__dict__.pop("rules", None)

    session_file = tmp_path / "legacy_rules_session.pkl"
    with open(session_file, "wb") as f:
        pickle.dump({"strategies": {"legacy": strategy}}, f)

    monkeypatch.setattr(session, "_SESSION_FILE", session_file)
    session._load()

    assert len(session._strategies["legacy"].rules) == 1
    assert session._strategies["legacy"].rules[0].name == "RebalanceFrequencyRule"
    assert session._strategies["legacy"].rules[0].interval_days == 5


# ---------------------------------------------------------------------------
# indicator_describe
# ---------------------------------------------------------------------------


def test_indicator_describe_rsi() -> None:
    result = indicator_describe(type="RSI")
    assert result["name"] == "RSI"
    assert "frac" in result["formula"]
    assert result["params"]["period"] == "14"
    assert result["params"]["column"] == "close"
    assert result["depends_on"] == []


def test_indicator_describe_macd_signal_depends_on() -> None:
    result = indicator_describe(type="MACDSignal")
    assert result["name"] == "MACDSignal"
    assert "macd" in result["depends_on"]


def test_indicator_describe_unknown() -> None:
    result = indicator_describe(type="FooBar")
    assert "error" in result
    assert "FooBar" in result["error"]


def test_indicator_describe_all_have_formula() -> None:
    """Every indicator returned by describe should have a non-empty formula."""
    for name in list_indicators():
        result = indicator_describe(type=name)
        assert "error" not in result, f"{name} returned error"
        assert len(result["formula"]) > 0, f"{name} has empty formula"


# ---------------------------------------------------------------------------
# indicator_list
# ---------------------------------------------------------------------------


def test_indicator_list_count() -> None:
    result = indicator_list()
    # >= rather than == because plan 027 made the registry mutable at runtime;
    # other tests in the same process may have registered mocks before this runs.
    assert len(result["indicators"]) >= 47


def test_indicator_list_structure() -> None:
    result = indicator_list()
    for item in result["indicators"]:
        assert "name" in item
        assert "formula" in item
        assert "description" in item
        assert len(item["formula"]) > 0, f"{item['name']} has empty formula"


def test_indicator_list_sorted() -> None:
    result = indicator_list()
    names = [i["name"] for i in result["indicators"]]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# strategy_add_rule — order / risk rules
# ---------------------------------------------------------------------------


def test_strategy_add_rule_stop_loss() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="sl",
        type="StopLossRule",
        params={"threshold": 0.05},
    )
    strat = session._strategies["s1"]
    pending = getattr(strat, "_pending_rules", [])
    assert len(pending) == 1
    assert pending[0].threshold == 0.05


def test_strategy_add_rule_take_profit() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="tp",
        type="TakeProfitRule",
        params={"threshold": 0.15},
    )
    strat = session._strategies["s1"]
    pending = getattr(strat, "_pending_rules", [])
    assert len(pending) == 1


def test_strategy_add_rule_trailing_stop() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="ts",
        type="TrailingStopRule",
        params={"trail_pct": 0.05},
    )
    strat = session._strategies["s1"]
    pending = getattr(strat, "_pending_rules", [])
    assert len(pending) == 1


def test_strategy_add_rule_max_drawdown_risk() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="mdd",
        type="MaxDrawdownRisk",
        params={"max_drawdown": 0.15},
    )
    strat = session._strategies["s1"]
    pending = getattr(strat, "_pending_rules", [])
    assert len(pending) == 1


def test_strategy_add_rule_daily_loss_limit_risk() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="dll",
        type="DailyLossLimitRisk",
        params={"max_daily_loss": 0.03},
    )
    strat = session._strategies["s1"]
    pending = getattr(strat, "_pending_rules", [])
    assert len(pending) == 1


# ---------------------------------------------------------------------------
# strategy_inspect — rules field
# ---------------------------------------------------------------------------


def test_strategy_inspect_shows_rules() -> None:
    strategy_create(name="s1", hypothesis="h", objectives={"r": {"min": 0.0}})
    strategy_add_rule(
        strategy="s1",
        name="sl",
        type="StopLossRule",
        params={"threshold": 0.05},
    )
    strategy_add_rule(
        strategy="s1",
        name="mdd",
        type="MaxDrawdownRisk",
        params={"max_drawdown": 0.15},
    )
    result = strategy_inspect("s1")
    assert len(result["rules"]) == 2


# ---------------------------------------------------------------------------
# RULE_TYPES registry completeness
# ---------------------------------------------------------------------------


def test_rule_types_registry_completeness() -> None:
    expected = {
        "BlacklistRule",
        "DailyLossLimitRisk",
        "ExitRule",
        "MaxDrawdownRisk",
        "MaxHoldingsRule",
        "RebalanceFrequencyRule",
        "StopLossRule",
        "TakeProfitRule",
        "TrailingStopRule",
    }
    assert set(list_rules().keys()) == expected
