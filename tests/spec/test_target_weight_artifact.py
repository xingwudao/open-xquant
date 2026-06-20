import json
from decimal import Decimal

import pandas as pd

from oxq.audit.reproducibility import audit_reproducibility
from oxq.core.types import BarSnapshot, Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _write_artifacts, compile_run
from oxq.spec.schema import SignalRuleDef, StrategySpec


def test_write_artifacts_includes_target_weights_csv(tmp_path):
    spec = StrategySpec.template(strategy_id="target_weight_artifact", hypothesis="target weights")
    spec.universe.symbols = ["510300.SS"]
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
            "is_rebalance": True,
            "reason": "rebalance",
        }
    ]
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert "target_weights.csv" in hashes


def test_roc_timing_fixed_threshold_target_weights_match_baseline(tmp_path):
    spec, data_dir = _write_roc_timing_spec(
        tmp_path,
        strategy_id="prod_csi300_reversal_timing_v1",
        closes=[100.0, 80.0, 95.0, 130.0, 129.0],
        params={
            "lookback": 1,
            "threshold_mode": "fixed",
            "buy_threshold": -0.10,
            "sell_threshold": 0.20,
            "stop_loss_pct": 0,
        },
    )

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    target_weights = _symbol_target_weights(run_dir)
    assert target_weights == [
        {
            "date": "2024-01-02T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-03T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 1.0,
            "adjusted_weight": 1.0,
            "is_rebalance": True,
            "reason": "rebalance",
        },
        {
            "date": "2024-01-04T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 1.0,
            "adjusted_weight": 1.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-05T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": True,
            "reason": "rebalance",
        },
        {
            "date": "2024-01-08T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": False,
            "reason": "hold",
        },
    ]


def test_roc_timing_rolling_quantile_target_weights_match_baseline(tmp_path):
    spec, data_dir = _write_roc_timing_spec(
        tmp_path,
        strategy_id="prod_csi300_reversal_timing",
        closes=[60.0, 60.0, 60.0, 60.0, 70.0, 60.0, 60.0, 70.0],
        params={
            "lookback": 1,
            "threshold_mode": "rolling",
            "q_window": 3,
            "q_bottom": 0.25,
            "q_top": 0.75,
            "stop_loss_pct": 0,
        },
    )

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    target_weights = _symbol_target_weights(run_dir)
    assert target_weights == [
        {
            "date": "2024-01-02T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-03T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-04T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-05T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-08T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-09T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 1.0,
            "adjusted_weight": 1.0,
            "is_rebalance": True,
            "reason": "rebalance",
        },
        {
            "date": "2024-01-10T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 1.0,
            "adjusted_weight": 1.0,
            "is_rebalance": False,
            "reason": "hold",
        },
        {
            "date": "2024-01-11T00:00:00+00:00",
            "symbol": "510300.SS",
            "target_weight": 0.0,
            "adjusted_weight": 0.0,
            "is_rebalance": True,
            "reason": "rebalance",
        },
    ]


def test_reproducibility_audit_fails_when_target_weights_artifact_is_tampered(tmp_path):
    spec = StrategySpec.template(strategy_id="target_weights_audit", hypothesis="target weight hashes are audited")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000.0")),
        trades=[],
        equity_curve=[(pd.Timestamp("2024-01-02", tz="UTC"), 100000.0)],
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
    (tmp_path / "target_weights.csv").write_text(
        "date,symbol,target_weight,adjusted_weight,is_rebalance,reason\n"
        "2024-01-02T00:00:00+00:00,510300.SS,0.0,0.0,true,rebalance\n",
        encoding="utf-8",
    )

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "target_weights_hash" and check["status"] == "fail" for check in audit["checks"])


def _write_roc_timing_spec(tmp_path, strategy_id: str, closes: list[float], params: dict):
    spec = StrategySpec.template(strategy_id=strategy_id, hypothesis="CSI300 reversal timing")
    spec.market.calendar = "XNYS"
    spec.universe.symbols = ["510300.SS"]
    spec.benchmark.symbols = []
    dates = pd.bdate_range("2024-01-02", periods=len(closes), tz="UTC")
    spec.data.min_start_date = dates[0].date().isoformat()
    spec.signal.rules = {
        "roc_timing": SignalRuleDef(type="ROCTiming", params=params),
    }
    spec.portfolio.type = "SignalPosition"
    spec.portfolio.params = {"signal_col": "roc_timing", "weight": 1.0}
    spec.validation.train_period = []
    spec.validation.test_period = [dates[0].date().isoformat(), dates[-1].date().isoformat()]
    spec.validation.required_oos = False
    spec.execution.lot_size = 1
    spec.execution.lot_size_config.default = 1

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=dates,
    ).to_parquet(data_dir / "510300.SS.parquet")
    return spec, data_dir


def _symbol_target_weights(run_dir):
    target_weights = pd.read_csv(run_dir / "target_weights.csv")
    target_weights = target_weights[target_weights["symbol"] == "510300.SS"]
    records = target_weights.to_dict("records")
    for record in records:
        record["target_weight"] = float(record["target_weight"])
        record["adjusted_weight"] = float(record["adjusted_weight"])
        record["is_rebalance"] = bool(record["is_rebalance"])
    return records
