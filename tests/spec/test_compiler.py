from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from oxq.audit.reproducibility import audit_reproducibility
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _build_optimizer, _write_artifacts, compile_strategy
from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec


def test_artifact_spec_hash_matches_serialized_spec(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="hash_test", hypothesis="hash artifacts are reproducible")
    spec.execution.initial_cash = 100_000

    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"


def test_crossover_latch_can_be_reset_after_exit() -> None:
    spec = StrategySpec.template(strategy_id="cross_reset", hypothesis="crossover exits clear active entry state")
    spec.signal.indicators = {
        "fast": IndicatorDef(type="SMA", params={"period": 2}),
        "slow": IndicatorDef(type="SMA", params={"period": 3}),
    }
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
    }

    optimizer = _build_optimizer(spec)
    entry_bar = pd.DataFrame({"cross": [True]})
    inactive_bar = pd.DataFrame({"cross": [False]})

    assert optimizer.optimize({"SPY": entry_bar}, {"SPY": entry_bar}) == {"SPY": 1.0}

    optimizer.reset_symbols(["SPY"])

    assert optimizer.optimize({"SPY": inactive_bar}, {"SPY": inactive_bar}) == {"CASH": 1.0}


def test_compile_strategy_rejects_unsupported_universe_type() -> None:
    spec = StrategySpec.template(strategy_id="unsupported_universe", hypothesis="unsupported universes fail clearly")
    spec.universe.type = "filter"

    with pytest.raises(ValueError, match="Unsupported universe.type 'filter'"):
        compile_strategy(spec)
