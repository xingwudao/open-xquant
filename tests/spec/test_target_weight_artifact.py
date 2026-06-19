import json
from decimal import Decimal

import pandas as pd

from oxq.core.types import BarSnapshot, Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _write_artifacts
from oxq.spec.schema import StrategySpec


def test_write_artifacts_includes_target_weights_csv(tmp_path):
    spec = StrategySpec.template(strategy_id="target_weight_artifact", hypothesis="target weights")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000.0")),
        trades=[],
        equity_curve=[],
        mktdata={},
        benchmark_prices={},
        snapshots=[
            BarSnapshot(
                date=pd.Timestamp("2024-01-02", tz="UTC"),
                target_weights={"510300.SS": 1.0},
                adjusted_weights={"510300.SS": 1.0},
                positions={},
                cash=0.0,
                total_value=100000.0,
            )
        ],
        orders=[],
    )

    _write_artifacts(spec, result, tmp_path, engine=None)

    target_weights = pd.read_csv(tmp_path / "target_weights.csv")
    assert target_weights.to_dict("records") == [
        {
            "date": "2024-01-02T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 1.0,
            "adjusted_weight": 1.0,
        }
    ]
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert "target_weights.csv" in hashes
