"""Tests for engine tools."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

import oxq.tools.engine as engine_tools
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.tools import session
from oxq.tools.engine import engine_results, engine_run, engine_trade_list
from oxq.tools.strategy import (
    strategy_add_signal,
    strategy_create,
)
from oxq.trade.sim_broker import FillPriceMode


@pytest.fixture(autouse=True)
def _reset_session():
    """Reset session state before each test."""
    session.clear()


@pytest.fixture()
def sample_data_dir(tmp_path):
    """Create 120-bar trending data as AAPL.parquet."""
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 -> 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 -> 180
    for i in range(30):
        closes.append(180 - i * 2)       # 180 -> 122

    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "AAPL.parquet")
    return tmp_path


def _build_strategy(
    name: str = "sma_cross",
    objectives: dict | None = None,
) -> None:
    """Build a complete SMA crossover strategy via tools.

    Uses the new API: signals with required_indicators and EqualWeightOptimizer.
    """
    if objectives is None:
        objectives = {
            "total_return": {"min": -0.5},
            "max_drawdown": {"max": 0.0},
        }
    strategy_create(
        name=name,
        hypothesis="SMA10 crossing above SMA50 predicts positive returns",
        objectives=objectives,
    )
    # Add signal with its required indicators
    strategy_add_signal(
        strategy=name, name="cross_up", type="Crossover",
        params={"fast": "sma_10", "slow": "sma_50"},
        indicators={
            "sma_10": {"type": "SMA", "params": {"column": "close", "period": 10}},
            "sma_50": {"type": "SMA", "params": {"column": "close", "period": 50}},
        },
    )


def _build_full_strategy() -> None:
    _build_strategy()


# ---------------------------------------------------------------------------
# engine_run
# ---------------------------------------------------------------------------


def test_engine_run(sample_data_dir) -> None:
    _build_full_strategy()
    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    assert "run_id" in result
    assert "error" not in result
    assert result["equity_curve_length"] == 120


def test_engine_run_symbols_override_does_not_mutate_strategy(sample_data_dir) -> None:
    _build_full_strategy()
    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        data_dir=str(sample_data_dir),
    )

    assert "error" not in result
    assert getattr(session._strategies["sma_cross"], "_legacy_universe", None) is None
    assert "sma_cross" not in session._strategy_universes


def test_engine_run_missing_strategy() -> None:
    result = engine_run(
        strategy="nonexistent", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
    )
    assert "error" in result


def test_engine_run_rejects_unsupported_fill_price_mode(sample_data_dir) -> None:
    _build_full_strategy()

    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        data_dir=str(sample_data_dir),
        fill_price_mode="next_high",
    )

    assert "Unsupported fill_price_mode" in result["error"]


def test_engine_run_rejects_next_open_without_calendar(sample_data_dir) -> None:
    _build_full_strategy()

    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        data_dir=str(sample_data_dir),
        fill_price_mode="next_open",
        market_calendar=None,
    )

    assert "market_calendar is required" in result["error"]


def test_engine_run_passes_market_calendar_to_provider_and_broker(monkeypatch, sample_data_dir) -> None:
    captured: dict[str, object] = {}

    class FakeMarketProvider:
        def __init__(self, data_dir, currency=None, calendar=None) -> None:
            captured["provider_calendar"] = calendar

    class FakeBroker:
        def __init__(self, **kwargs) -> None:
            captured["broker_calendar"] = kwargs.get("market_calendar")

    def fake_run(self, *args, **kwargs) -> RunResult:
        return RunResult(
            portfolio=Portfolio(cash=Decimal("100000")),
            trades=[],
            equity_curve=[],
            mktdata={},
        )

    monkeypatch.setattr(engine_tools, "LocalMarketDataProvider", FakeMarketProvider)
    monkeypatch.setattr(engine_tools, "SimBroker", FakeBroker)
    monkeypatch.setattr(engine_tools.Engine, "run", fake_run)
    _build_full_strategy()

    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        data_dir=str(sample_data_dir),
        fill_price_mode="next_open",
        market_calendar="XHKG",
    )

    assert "error" not in result
    assert captured == {"provider_calendar": "XHKG", "broker_calendar": "XHKG"}


@pytest.mark.parametrize(
    ("fill_price_mode", "expected"),
    [
        ("next_close", FillPriceMode.NEXT_CLOSE),
        ("next_mid", FillPriceMode.NEXT_MID),
        ("next_avg", FillPriceMode.NEXT_AVG),
        ("next_hl2", FillPriceMode.NEXT_HL2),
    ],
)
def test_engine_run_accepts_next_session_close_modes(monkeypatch, sample_data_dir, fill_price_mode, expected) -> None:
    captured: dict[str, object] = {}

    class FakeMarketProvider:
        def __init__(self, data_dir, currency=None, calendar=None) -> None:
            captured["provider_calendar"] = calendar

    class FakeBroker:
        def __init__(self, **kwargs) -> None:
            captured["fill_price_mode"] = kwargs.get("fill_price_mode")
            captured["broker_calendar"] = kwargs.get("market_calendar")

    def fake_run(self, *args, **kwargs) -> RunResult:
        return RunResult(
            portfolio=Portfolio(cash=Decimal("100000")),
            trades=[],
            equity_curve=[],
            mktdata={},
        )

    monkeypatch.setattr(engine_tools, "LocalMarketDataProvider", FakeMarketProvider)
    monkeypatch.setattr(engine_tools, "SimBroker", FakeBroker)
    monkeypatch.setattr(engine_tools.Engine, "run", fake_run)
    _build_full_strategy()

    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        data_dir=str(sample_data_dir),
        fill_price_mode=fill_price_mode,
        market_calendar="XNYS",
    )

    assert "error" not in result
    assert captured == {"provider_calendar": "XNYS", "fill_price_mode": expected, "broker_calendar": "XNYS"}


def test_engine_run_through_indicator(sample_data_dir) -> None:
    _build_full_strategy()
    result = engine_run(
        strategy="sma_cross",
        symbols=["AAPL"],
        start="2024-01-01",
        end="2024-12-31",
        run_through="indicator",
        data_dir=str(sample_data_dir),
    )
    assert result["total_trades"] == 0
    assert result["equity_curve_length"] == 0


# ---------------------------------------------------------------------------
# engine_results
# ---------------------------------------------------------------------------


def test_engine_results(sample_data_dir) -> None:
    _build_full_strategy()
    run = engine_run(
        strategy="sma_cross", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    run_id = run["run_id"]

    result = engine_results(run_id)
    assert "metrics" in result
    assert "total_return" in result["metrics"]
    assert "sharpe_ratio" in result["metrics"]
    assert "max_drawdown" in result["metrics"]
    assert isinstance(result["metrics"]["total_return"], float)


def test_engine_results_objectives_check(sample_data_dir) -> None:
    _build_full_strategy()
    run = engine_run(
        strategy="sma_cross", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    run_id = run["run_id"]

    result = engine_results(run_id)
    assert len(result["objectives_check"]) > 0
    for check in result["objectives_check"]:
        assert "metric" in check
        assert "actual" in check
        assert "pass" in check


def test_engine_results_not_found() -> None:
    result = engine_results("nonexistent_123")
    assert "error" in result


def test_engine_results_drawdown_pass_when_better(sample_data_dir) -> None:
    """Drawdown -8% with max -99% should pass (abs(actual) < abs(max))."""
    _build_strategy("dd_pass", objectives={"max_drawdown": {"max": -0.99}})
    run = engine_run(
        strategy="dd_pass", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    result = engine_results(run["run_id"])
    dd_check = next(c for c in result["objectives_check"] if c["metric"] == "max_drawdown")
    assert dd_check["pass"] is True


def test_engine_results_drawdown_fail_when_worse(sample_data_dir) -> None:
    """Drawdown exceeding tight limit should fail."""
    _build_strategy("dd_fail", objectives={"max_drawdown": {"max": -0.001}})
    run = engine_run(
        strategy="dd_fail", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    result = engine_results(run["run_id"])
    dd_check = next(c for c in result["objectives_check"] if c["metric"] == "max_drawdown")
    assert dd_check["pass"] is False


def test_engine_results_drawdown_exact_boundary(sample_data_dir) -> None:
    """Drawdown exactly at boundary should pass (abs(actual) == abs(max))."""
    _build_strategy("dd_exact", objectives={"max_drawdown": {"max": -0.99}})
    run = engine_run(
        strategy="dd_exact", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    actual_dd = engine_results(run["run_id"])["metrics"]["max_drawdown"]

    # Rebuild with exact drawdown as boundary
    session.clear()
    _build_strategy("dd_exact", objectives={"max_drawdown": {"max": actual_dd}})
    run2 = engine_run(
        strategy="dd_exact", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    dd_check = next(
        c for c in engine_results(run2["run_id"])["objectives_check"]
        if c["metric"] == "max_drawdown"
    )
    assert dd_check["pass"] is True


def test_engine_results_volatility_uses_normal_comparison(sample_data_dir) -> None:
    """Non-drawdown metrics like volatility use normal <= comparison."""
    _build_strategy("vol_check", objectives={"annualized_volatility": {"max": 0.001}})
    run = engine_run(
        strategy="vol_check", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    result = engine_results(run["run_id"])
    vol_check = next(c for c in result["objectives_check"] if c["metric"] == "annualized_volatility")
    assert vol_check["pass"] is False


def test_engine_results_mixed_objectives(sample_data_dir) -> None:
    """Multiple objectives: drawdown uses abs comparison, others use normal."""
    _build_strategy("mixed", objectives={
        "total_return": {"min": -10.0},
        "max_drawdown": {"max": -0.99},
    })
    run = engine_run(
        strategy="mixed", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    checks = {
        c["metric"]: c["pass"]
        for c in engine_results(run["run_id"])["objectives_check"]
    }
    assert checks["total_return"] is True
    assert checks["max_drawdown"] is True


# ---------------------------------------------------------------------------
# engine_trade_list
# ---------------------------------------------------------------------------


def test_engine_trade_list(sample_data_dir) -> None:
    _build_full_strategy()
    run = engine_run(
        strategy="sma_cross", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    run_id = run["run_id"]

    result = engine_trade_list(run_id)
    assert result["total_trades"] >= 0
    if result["total_trades"] > 0:
        trade = result["trades"][0]
        assert "symbol" in trade
        assert "side" in trade
        assert "shares" in trade
        assert "price" in trade
        assert "date" in trade
        assert trade["symbol"] == "AAPL"


def test_engine_trade_list_includes_order_type_and_fee(sample_data_dir) -> None:
    _build_full_strategy()
    run = engine_run(
        strategy="sma_cross", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
    )
    result = engine_trade_list(run["run_id"])
    if result["total_trades"] > 0:
        trade = result["trades"][0]
        assert "order_type" in trade
        assert "fee" in trade
        assert trade["order_type"] == "market"
        assert trade["fee"] == 0.0


def test_engine_trade_list_not_found() -> None:
    result = engine_trade_list("nonexistent_123")
    assert "error" in result


# ---------------------------------------------------------------------------
# engine_run with fee/slippage
# ---------------------------------------------------------------------------


def test_engine_run_with_fee(sample_data_dir) -> None:
    _build_strategy("fee_test")
    run = engine_run(
        strategy="fee_test", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
        fee_rate=0.001,
        fee_min=5.0,
    )
    assert "error" not in run


def test_engine_run_with_slippage(sample_data_dir) -> None:
    _build_strategy("with_slip")
    run_with = engine_run(
        strategy="with_slip", symbols=["AAPL"],
        start="2024-01-01", end="2024-12-31",
        data_dir=str(sample_data_dir),
        slippage_rate=0.01,
    )
    assert "error" not in run_with
