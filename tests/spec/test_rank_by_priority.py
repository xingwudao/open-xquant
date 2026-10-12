"""EqualWeight rank_by: priority for new entries when capped by max_holdings."""

import numpy as np
import pandas as pd

from oxq.spec import compiler
from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate


def _frames(js: dict[str, float]) -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=1)
    return {s: pd.DataFrame({"entry": [True], "j": [j]}, index=idx) for s, j in js.items()}


def test_signal_filtered_equal_weight_exposes_priority_by_rank_column() -> None:
    opt = compiler._SignalFilteredEqualWeightOptimizer(
        signal_names=["entry"], signal_types={"entry": "FormulaEvent"}, rank_by="j", rank_ascending=True,
    )
    frames = _frames({"A": -5.0, "B": -20.0, "C": np.nan})
    opt.optimize(frames, frames)
    assert opt.last_priority == {"B": 0, "A": 1, "C": 2}


def test_constrained_optimizer_uses_priority_for_max_holdings() -> None:
    spec = StrategySpec.from_dict({
        "strategy_id": "rank",
        "universe": {"type": "static", "symbols": ["A", "B", "C"]},
        "signal": {
            "indicators": {"j": {"type": "SMA", "params": {"period": 1}}},
            "rules": {"entry": {"type": "FormulaEvent", "params": {"expr": "close > 0"}}},
        },
        "portfolio": {
            "type": "EqualWeight",
            "params": {"rank_by": "j", "rank_ascending": True},
            "rules": {
                "cap": {"type": "MaxHoldingsRule", "params": {"max_holdings": 1}},
                "stop": {"type": "StopLossRule", "params": {"threshold": 0.1}},
            },
        },
    })
    optimizer = compiler._build_optimizer(spec)
    frames = _frames({"A": -5.0, "B": -20.0, "C": 3.0})
    weights = optimizer.optimize(frames, frames)
    assert [s for s, w in weights.items() if s != "CASH" and w > 0] == ["B"]


def test_rank_by_must_reference_known_column() -> None:
    spec = StrategySpec.from_dict({
        "strategy_id": "rank_bad",
        "universe": {"type": "static", "symbols": ["A"]},
        "signal": {"indicators": {}, "rules": {"entry": {"type": "Formula", "params": {"expr": "close > 0"}}}},
        "portfolio": {"type": "EqualWeight", "params": {"rank_by": "nope"}},
    })
    # _require_optimizer_column reports unknown columns as optimizer_column_missing.
    assert "optimizer_column_missing" in {e["check"] for e in validate(spec).errors}
