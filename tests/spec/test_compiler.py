from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import oxq.audit.reproducibility as reproducibility
import oxq.spec.compiler as compiler
from oxq.audit.reproducibility import _hash_json_file, audit_reproducibility
from oxq.core.engine import Engine
from oxq.core.types import Fill, Order, Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.portfolio.orderbook import ManagedOrder
from oxq.spec.compiler import (
    _build_metrics,
    _build_optimizer,
    _write_artifacts,
    compile_plan,
    compile_run,
    compile_strategy,
    compile_universe,
)
from oxq.spec.schema import IndicatorDef, PortfolioRuleDef, SignalRuleDef, StrategySpec


@pytest.fixture(autouse=True)
def _isolate_parent_run_digest(tmp_path):
    digest = tmp_path.parent / "run_digests.jsonl"
    lock = tmp_path.parent / "run_digests.jsonl.lock"
    digest.unlink(missing_ok=True)
    lock.unlink(missing_ok=True)
    yield
    digest.unlink(missing_ok=True)
    lock.unlink(missing_ok=True)


def test_spec_package_exports_compile_universe() -> None:
    from oxq.spec import compile_universe as exported_compile_universe

    assert exported_compile_universe is compile_universe


def test_artifact_spec_hash_matches_serialized_spec(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="hash_test", hypothesis="hash artifacts are reproducible")
    spec.execution.initial_cash = 100_000

    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")

    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"


def test_write_artifacts_persists_benchmark_curve(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="benchmark_artifact", hypothesis="benchmark prices should be reusable")
    spec.benchmark.symbols = ["SPY"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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
        benchmark_prices={"SPY": pd.Series([100.0, 110.0, 105.0], index=dates)},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    benchmark = pd.read_csv(tmp_path / "benchmark_curve.csv")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert benchmark.to_dict(orient="list") == {
        "date": [str(dates[0]), str(dates[1]), str(dates[2])],
        "value": [100.0, 110.0, 105.0],
    }
    assert "benchmark_curve.csv" in hashes


def test_write_artifacts_uses_later_benchmark_when_first_is_unusable(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="benchmark_fallback", hypothesis="usable benchmarks should be persisted")
    spec.benchmark.symbols = ["SPY", "QQQ"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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
        benchmark_prices={
            "SPY": pd.Series([None, None, None], index=dates),
            "QQQ": pd.Series([200.0, 220.0, 210.0], index=dates),
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    benchmark = pd.read_csv(tmp_path / "benchmark_curve.csv")
    assert benchmark.to_dict(orient="list") == {
        "date": [str(dates[0]), str(dates[1]), str(dates[2])],
        "value": [200.0, 220.0, 210.0],
    }


def test_reproducibility_audit_validates_benchmark_curve_hash(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="benchmark_hash", hypothesis="benchmark artifact hashes should be audited")
    spec.benchmark.symbols = ["SPY"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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
        benchmark_prices={"SPY": pd.Series([100.0, 110.0, 105.0], index=dates)},
    )

    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "benchmark_curve.csv").write_text("date,value\n2024-01-02 00:00:00+00:00,1.0\n", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "benchmark_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_allows_self_contained_run_without_parent_digest(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="self_contained", hypothesis="run directories are portable")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")

    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    (tmp_path.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"
    assert not any(check["id"] == "run_digest" for check in audit["checks"])


def test_run_digest_append_creates_lock_file(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="digest_lock", hypothesis="shared digest writes are locked")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_artifacts(spec, result, run_dir, Engine())

    assert (tmp_path / "run_digests.jsonl.lock").exists()
    assert (tmp_path / "run_digests.jsonl").read_text(encoding="utf-8").endswith("\n")


def test_reproducibility_audit_fails_when_parent_digest_mismatches(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="digest_mismatch", hypothesis="external digest detects artifact hash tampering")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")

    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    (tmp_path.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": tmp_path.name, "artifact_hashes": "sha256:badbadbadbadbadb"}) + "\n",
        encoding="utf-8",
    )

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    digest_check = next(check for check in audit["checks"] if check["id"] == "run_digest")
    assert digest_check["status"] == "fail"
    assert digest_check["severity"] == "fatal"


def test_reproducibility_audit_fails_when_positions_artifact_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="positions_hash", hypothesis="positions artifacts are audited")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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
    (tmp_path / "positions.csv").write_text("symbol,shares,avg_cost\nSPY,999,1.0\n", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "positions_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_parent_digest_is_corrupt(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="digest_corrupt", hypothesis="external digest corruption is fatal")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
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
    (tmp_path.parent / "run_digests.jsonl").write_text("{not-json\n", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    digest_check = next(check for check in audit["checks"] if check["id"] == "run_digest")
    assert audit["status"] == "fail"
    assert digest_check["severity"] == "fatal"


def test_artifacts_persist_full_order_lifecycle_and_hash(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="order_lifecycle", hypothesis="orders are auditable even without fills")
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    rejected = ManagedOrder(
        Order(symbol="SPY", side="BUY", shares=10, order_type="market"),
        id="ord_1",
        created_at=dates[0].isoformat(),
    )
    rejected.status = "rejected"
    rejected.due_at = dates[1].isoformat()
    expired = ManagedOrder(
        Order(symbol="QQQ", side="BUY", shares=5, order_type="market"),
        id="ord_2",
        created_at=dates[0].isoformat(),
    )
    expired.status = "expired"
    expired.due_at = dates[1].isoformat()
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100000.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1]},
                index=dates,
            )
        },
        orders=[rejected, expired],
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    orders = pd.read_csv(tmp_path / "orders.csv")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert set(orders["status"]) == {"rejected", "expired"}
    assert set(["id", "created_at", "due_at", "status"]).issubset(orders.columns)
    assert "orders.csv" in hashes


def test_write_artifacts_persists_target_weights(tmp_path) -> None:
    import json

    import pandas as pd

    from oxq.core.engine import Engine
    from oxq.core.types import BarSnapshot, Portfolio
    from oxq.portfolio.analytics import RunResult
    from oxq.spec.compiler import _write_artifacts
    from oxq.spec.schema import StrategySpec

    spec = StrategySpec.template(
        strategy_id="target_weight_artifact",
        hypothesis="target weights should be stable artifacts",
    )
    spec.universe.symbols = ["SPY"]
    spec.validation.train_period = ["2024-01-01", "2024-01-03"]
    spec.validation.test_period = ["2024-01-04", "2024-01-05"]

    result = RunResult(
        portfolio=Portfolio(cash=Decimal("0")),
        trades=[],
        equity_curve=[
            (pd.Timestamp("2024-01-02", tz="UTC"), 100000.0),
            (pd.Timestamp("2024-01-03", tz="UTC"), 101000.0),
        ],
        mktdata={},
        snapshots=[
            BarSnapshot(
                date=pd.Timestamp("2024-01-02", tz="UTC"),
                target_weights={"SPY": 1.0},
                adjusted_weights={"SPY": 1.0},
                positions={},
                cash=0.0,
                total_value=100000.0,
            ),
            BarSnapshot(
                date=pd.Timestamp("2024-01-03", tz="UTC"),
                target_weights={"CASH": 1.0},
                adjusted_weights={"CASH": 1.0},
                positions={},
                cash=101000.0,
                total_value=101000.0,
            ),
        ],
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    rows = pd.read_csv(tmp_path / "target_weights.csv")
    assert list(rows.columns) == [
        "date",
        "symbol",
        "raw_target_weight",
        "adjusted_target_weight",
        "reason",
    ]
    assert rows.to_dict("records") == [
        {
            "date": "2024-01-02 00:00:00+00:00",
            "symbol": "SPY",
            "raw_target_weight": 1.0,
            "adjusted_target_weight": 1.0,
            "reason": "target_changed",
        },
        {
            "date": "2024-01-03 00:00:00+00:00",
            "symbol": "CASH",
            "raw_target_weight": 1.0,
            "adjusted_target_weight": 1.0,
            "reason": "target_changed",
        },
        {
            "date": "2024-01-03 00:00:00+00:00",
            "symbol": "SPY",
            "raw_target_weight": 0.0,
            "adjusted_target_weight": 0.0,
            "reason": "target_changed",
        },
    ]

    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["schema_version"] == 5
    assert "target_weights.csv" in hashes


def test_write_artifacts_persists_compiled_plan(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="compiled_plan", hypothesis="compiled plans should be auditable")
    spec.universe.symbols = ["SPY"]
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"
    spec.signal.indicators = {
        "roc_1": IndicatorDef(type="ROC", params={"column": "close", "period": 1})
    }
    spec.signal.rules = {
        "positive": SignalRuleDef(
            type="Threshold",
            params={"column": "roc_1", "threshold": 0, "relationship": "gt"},
        )
    }
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-04"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    source_df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [100, 100, 100],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": source_df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    plan = json.loads((run_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    strategy_py = (run_dir / "strategy.py").read_text(encoding="utf-8")
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert plan["schema_version"] == 1
    assert plan["compilation_mode"] == "direct_runtime"
    assert plan["spec_hash"] == spec.compute_hash()
    assert plan["signals"]["indicators"]["roc_1"]["type"] == "ROC"
    assert plan["signals"]["rules"]["positive"]["effective_type"] == "Threshold"
    assert plan["signals"]["terminal_signals"] == ["positive"]
    assert plan["portfolio"]["runtime_type"] == "SignalFilteredEqualWeight"
    assert plan["execution"]["fill_price_mode"] == "next_open"
    assert plan["data"]["data_dir"] == str(data_dir)
    assert plan["data"]["effective_data_dir"] == str(data_dir)
    assert plan["data"]["spec_data_dir"] == ""
    assert "STRATEGY_SPEC =" in strategy_py
    assert "'fill_price_mode': ''" in strategy_py
    assert "COMPILED_PLAN =" in strategy_py
    assert "STRATEGY_FLOW =" in strategy_py
    assert "def define_universe() -> dict:" in strategy_py
    assert "def define_indicators() -> dict:" in strategy_py
    assert "def define_signals() -> dict:" in strategy_py
    assert "def define_portfolio() -> dict:" in strategy_py
    assert "def define_rules() -> list[dict]:" in strategy_py
    assert "def simulate_trading_flow() -> list[dict]:" in strategy_py
    assert "def build_strategy():" in strategy_py
    assert "Audit data appendix" in strategy_py
    assert strategy_py.index("def define_universe() -> dict:") < strategy_py.index("STRATEGY_SPEC =")
    assert strategy_py.index("def simulate_trading_flow() -> list[dict]:") < strategy_py.index("COMPILED_PLAN =")
    module_spec = importlib.util.spec_from_file_location("generated_strategy_review", run_dir / "strategy.py")
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    description = module.describe()
    assert [step["phase"] for step in description["strategy_flow"]] == [
        "universe",
        "indicator",
        "signal",
        "portfolio",
        "rule",
        "trade_simulation",
    ]
    assert description["universe"]["review_note"].startswith("This run evaluates")
    assert description["indicators"]["roc_1"]["type"] == "ROC"
    assert hashes["schema_version"] == 5
    assert "compiled_plan.json" in hashes
    assert "strategy.py" in hashes
    assert audit_reproducibility(run_dir)["status"] == "pass"


def test_strategy_py_serializes_date_values_as_literals(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="compiled_plan_dates", hypothesis="date values should be literal-safe")
    spec.research.created_at = date(2024, 1, 2)  # type: ignore[assignment]
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1, 1],
                },
                index=dates,
            )
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")

    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    strategy_py = (run_dir / "strategy.py").read_text(encoding="utf-8")
    assert "datetime.date" not in strategy_py
    assert "'created_at': '2024-01-02'" in strategy_py
    assert audit_reproducibility(run_dir)["status"] == "pass"


def test_compiled_plan_records_normalized_metric_assumptions(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="compiled_plan_metrics", hypothesis="compiled plans match metric runtime")
    spec.metrics.profile = "xquant_production"
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1, 1],
                },
                index=dates,
            )
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_artifacts(spec, result, run_dir, Engine())

    plan = json.loads((run_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert plan["metrics"]["profile"] == "xquant_production"
    assert plan["metrics"]["return_type"] == "log"
    assert plan["metrics"]["risk_free_rate"] == 0.02
    assert plan["metrics"]["return_type"] == metrics["metric_assumptions"]["return_type"]
    assert plan["metrics"]["risk_free_rate"] == metrics["metric_assumptions"]["risk_free_rate"]


def test_compiled_plan_hash_is_stable_across_run_dirs(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="compiled_plan_hash", hypothesis="compiled plan hashes exclude run metadata")
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

    first = tmp_path / "run_1"
    second = tmp_path / "run_2"
    first.mkdir()
    second.mkdir()
    _write_artifacts(spec, result, first, Engine())
    _write_artifacts(spec, result, second, Engine())

    first_hashes = json.loads((first / "artifact_hashes.json").read_text(encoding="utf-8"))
    second_hashes = json.loads((second / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert first_hashes["compiled_plan.json"] == second_hashes["compiled_plan.json"]
    assert first_hashes["strategy.py"] == second_hashes["strategy.py"]


def test_write_artifacts_preserves_target_weight_rule_reasons(tmp_path) -> None:
    from oxq.core.types import BarSnapshot

    spec = StrategySpec.template(
        strategy_id="target_weight_reasons",
        hypothesis="target weight artifacts should preserve rule reasons",
    )
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100000.0)],
        mktdata={},
        snapshots=[
            BarSnapshot(
                date=dates[0],
                target_weights={"SPY": 1.0},
                adjusted_weights={"SPY": 0.0},
                positions={},
                cash=100000.0,
                total_value=100000.0,
                rule_reasons={"SPY": "SPY is blacklisted"},
            ),
            BarSnapshot(
                date=dates[1],
                target_weights={"SPY": 1.0},
                adjusted_weights={"SPY": 1.0},
                positions={},
                cash=100000.0,
                total_value=100000.0,
                rule_reasons={"__all__": "rebalance interval: 1 bars < 5 bars"},
            ),
        ],
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    rows = pd.read_csv(tmp_path / "target_weights.csv")
    assert rows["reason"].to_list() == [
        "SPY is blacklisted",
        "rebalance interval: 1 bars < 5 bars",
    ]


def test_missing_ratio_ignores_derived_indicator_nans(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_ratio", hypothesis="derived warmup nans are not raw data missing")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
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
                    "sma_50": [None, None, 1.0],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_treats_non_midnight_daily_rows_as_sessions(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="non_midnight_rows", hypothesis="daily row timestamps map by session date")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    dates = pd.to_datetime(["2024-01-02 21:00", "2024-01-03 21:00"], utc=True)
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0
    assert manifest["data_fingerprints"]["SPY"]["row_count"] == 2


@pytest.mark.parametrize("calendar", ["XNYS", "ARCX", "XSHG", "XSHE"])
def test_exchange_calendar_sessions_accepts_supported_calendar_names(calendar: str) -> None:
    sessions = compiler._exchange_calendar_sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"), calendar)

    assert sessions is not None
    assert len(sessions) > 0


def test_data_fingerprint_covers_non_midnight_daily_row_values() -> None:
    expected_index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
    dates = pd.to_datetime(["2024-01-02 21:00", "2024-01-03 21:00"], utc=True)
    base = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    tampered = base.copy()
    tampered.loc[dates[1], "close"] = 2.0

    first = compiler._fingerprint_dataframe(
        compiler._reindex_for_fingerprint(base, expected_index),
        ["open", "high", "low", "close", "volume"],
    )
    second = compiler._fingerprint_dataframe(
        compiler._reindex_for_fingerprint(tampered, expected_index),
        ["open", "high", "low", "close", "volume"],
    )

    assert first["content_hash"] != second["content_hash"]


def test_data_fingerprint_preserves_non_midnight_source_index() -> None:
    expected_index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
    early_dates = pd.to_datetime(["2024-01-02 21:00", "2024-01-03 21:00"], utc=True)
    late_dates = pd.to_datetime(["2024-01-02 22:00", "2024-01-03 22:00"], utc=True)
    base = pd.DataFrame({"close": [1.0, 1.0]}, index=early_dates)
    shifted = pd.DataFrame({"close": [1.0, 1.0]}, index=late_dates)

    first = compiler._fingerprint_dataframe(compiler._reindex_for_fingerprint(base, expected_index), ["close"])
    second = compiler._fingerprint_dataframe(compiler._reindex_for_fingerprint(shifted, expected_index), ["close"])

    assert first["content_hash"] != second["content_hash"]
    assert first["start"] == early_dates[0].isoformat()


def test_missing_ratio_counts_absent_required_columns(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_required", hypothesis="absent required columns are missing data")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
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
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0


def test_missing_ratio_counts_empty_symbol_as_fully_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="empty_symbol", hypothesis="empty market data is missing data")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[],
        mktdata={
            "SPY": pd.DataFrame(columns=spec.data.required_columns),
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 1.0


def test_missing_ratio_counts_sparse_symbol_calendar(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="sparse_calendar", hypothesis="missing symbol days are data gaps")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
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
            ),
            "QQQ": pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1, 1],
                },
                index=[dates[0], dates[2]],
            ),
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0
    assert manifest["symbol_ranges"]["QQQ"]["start"] == "2024-01-01"
    assert manifest["symbol_ranges"]["QQQ"]["end"] == "2024-01-03"


def test_missing_ratio_counts_full_market_missing_business_day(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="full_gap", hypothesis="full-market gaps count as missing data")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[
            (dates[0], 100000.0),
            (pd.Timestamp("2024-01-02", tz="UTC"), 100000.0),
            (dates[1], 100001.0),
        ],
        mktdata={"SPY": frame, "QQQ": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0


def test_missing_ratio_does_not_count_xnys_holiday_as_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="holiday_gap", hypothesis="exchange holidays are not missing data")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-07-03", "2024-07-05"]
    dates = pd.to_datetime(["2024-07-03", "2024-07-05"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[
            (dates[0], 100000.0),
            (pd.Timestamp("2024-07-04", tz="UTC"), 100000.0),
            (dates[1], 100001.0),
        ],
        mktdata={"SPY": frame, "QQQ": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_does_not_count_xnys_special_closure(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="special_closure", hypothesis="special exchange closures are not missing data")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2018-12-04", "2018-12-06"]
    dates = pd.to_datetime(["2018-12-04", "2018-12-06"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_requires_exchange_calendar_when_sessions_are_unavailable(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="calendar_required", hypothesis="incomplete fallback calendars are unsafe")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2018-12-04", "2018-12-06"]
    dates = pd.to_datetime(["2018-12-04", "2018-12-06"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": frame},
    )
    monkeypatch.setattr(compiler, "_exchange_calendar_sessions", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="exchange_calendars is required"):
        _write_artifacts(spec, result, tmp_path, Engine())


def test_missing_ratio_does_not_count_xnys_good_friday_as_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="good_friday_gap", hypothesis="exchange-only holidays are not missing")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-03-28", "2024-04-01"]
    dates = pd.to_datetime(["2024-03-28", "2024-04-01"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[
            (dates[0], 100000.0),
            (pd.Timestamp("2024-03-29", tz="UTC"), 100000.0),
            (dates[1], 100001.0),
        ],
        mktdata={"SPY": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_counts_missing_requested_start_boundary(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="start_boundary_gap", hypothesis="requested boundary gaps count")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-05"]
    dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(date, 100000.0 + idx) for idx, date in enumerate(dates)],
        mktdata={"SPY": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0


def test_reproducibility_audit_fails_when_source_data_changes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="source_data_hash", hypothesis="source data changes break reproducibility")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    tampered = df.copy()
    tampered.loc[dates[1], "close"] = 2.0
    tampered.to_parquet(data_dir / "SPY.parquet")

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_fingerprint" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_normalizes_naive_source_data_index(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="naive_source_hash", hypothesis="provider localizes naive source index")
    naive_dates = pd.bdate_range("2024-01-02", periods=3)
    aware_dates = naive_dates.tz_localize("UTC")
    source_df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=naive_dates,
    )
    loaded_df = source_df.copy()
    loaded_df.index = aware_dates
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(aware_dates[0], 100000.0), (aware_dates[1], 100001.0), (aware_dates[2], 100003.0)],
        mktdata={"SPY": loaded_df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"
    assert any(check["id"] == "data_fingerprint" and check["status"] == "pass" for check in audit["checks"])


def test_reproducibility_audit_detects_non_midnight_source_index_change(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="source_time_hash", hypothesis="source timestamps are part of fingerprints")
    loaded_dates = pd.to_datetime(["2024-01-02 21:00", "2024-01-03 21:00"], utc=True)
    source_dates = pd.to_datetime(["2024-01-02 22:00", "2024-01-03 22:00"], utc=True)
    loaded_df = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=loaded_dates,
    )
    source_df = loaded_df.copy()
    source_df.index = source_dates
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(loaded_dates[0], 100000.0), (loaded_dates[1], 100001.0)],
        mktdata={"SPY": loaded_df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_fingerprint" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_filters_source_data_to_manifest_calendar(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="calendar_source_hash", hypothesis="audit uses runtime calendar filter")
    dates = pd.to_datetime(["2024-01-02", "2024-01-06"], utc=True)
    source_df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    loaded_df = source_df.iloc[[0]]
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0)],
        mktdata={"SPY": loaded_df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"
    assert any(check["id"] == "data_fingerprint" and check["status"] == "pass" for check in audit["checks"])


def test_reproducibility_calendar_alignment_accepts_aware_sessions(monkeypatch) -> None:
    class _AwareCalendar:
        @staticmethod
        def sessions_in_range(start, end):
            return pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")

    import exchange_calendars as xcals

    monkeypatch.setattr(xcals, "get_calendar", lambda _name: _AwareCalendar())
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=index)

    aligned = reproducibility._align_to_calendar_sessions(df, "XNYS", "2024-01-02", "2024-01-03")

    assert aligned.index.equals(index)


def test_reproducibility_calendar_alignment_maps_non_midnight_daily_rows(monkeypatch) -> None:
    class _Calendar:
        @staticmethod
        def sessions_in_range(start, end):
            return pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")

    import exchange_calendars as xcals

    monkeypatch.setattr(xcals, "get_calendar", lambda _name: _Calendar())
    dates = pd.to_datetime(["2024-01-02 21:00", "2024-01-03 21:00"], utc=True)
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=dates)

    aligned = reproducibility._align_to_calendar_sessions(df, "XNYS", "2024-01-02", "2024-01-03")

    assert aligned["close"].tolist() == [1.0, 2.0]


def test_reproducibility_audit_warns_when_source_data_dir_unavailable(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_data_dir", hypothesis="source fingerprints need source data")
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

    check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert check["status"] == "fail"
    assert check["severity"] == "fatal"
    assert audit["status"] == "fail"


def test_compile_run_rejects_strategy_id_path_traversal(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="../outside", hypothesis="strategy ids cannot be paths")

    with pytest.raises(ValueError, match="strategy_id"):
        compile_run(spec, out_dir=tmp_path)


def test_compile_run_rejects_invalid_specs_before_execution(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="invalid_costs", hypothesis="compile_run enforces validation")
    spec.cost.fee_rate = 0.0
    spec.cost.slippage_rate = 0.0

    with pytest.raises(ValueError, match="Spec validation failed"):
        compile_run(spec, out_dir=tmp_path)


@pytest.mark.parametrize(
    ("order_timing", "price_type", "expected_fill_price_mode"),
    [
        ("next_session_open", "open", compiler.FillPriceMode.NEXT_OPEN),
        ("next_session_close", "close", compiler.FillPriceMode.NEXT_CLOSE),
        ("next_session_mid", "mid", compiler.FillPriceMode.NEXT_MID),
        ("next_session_avg", "avg", compiler.FillPriceMode.NEXT_AVG),
        ("next_session_hl2", "hl2", compiler.FillPriceMode.NEXT_HL2),
    ],
)
def test_compile_run_explicit_next_session_modes_instantiate_matching_broker(
    tmp_path,
    monkeypatch,
    order_timing,
    price_type,
    expected_fill_price_mode,
) -> None:
    spec = StrategySpec.template(strategy_id=f"explicit_{price_type}", hypothesis="explicit execution reaches broker")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = order_timing
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = price_type
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    class BrokerProbe:
        def __init__(self, *args, **kwargs) -> None:
            assert kwargs["fill_price_mode"] == expected_fill_price_mode
            raise RuntimeError("broker probe complete")

    monkeypatch.setattr(compiler, "SimBroker", BrokerProbe)

    with pytest.raises(RuntimeError, match="broker probe complete"):
        compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")


def test_compile_run_passes_cash_annual_return_to_engine(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="cash_return", hypothesis="cash return reaches runtime")
    spec.execution.cash_annual_return = 0.025
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    captured: dict = {}

    class EngineProbe:
        def run(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(compiler, "Engine", EngineProbe)
    monkeypatch.setattr(compiler, "_write_artifacts", lambda *args, **kwargs: None)

    compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    assert captured["cash_annual_return"] == 0.025


def test_compile_run_uses_lot_size_config_default(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="lot_size_config", hypothesis="lot config controls runtime lot size")
    spec.execution.lot_size = 1
    spec.execution.lot_size_config.default = 100
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    captured: dict = {}

    class EngineProbe:
        def run(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(compiler, "Engine", EngineProbe)
    monkeypatch.setattr(compiler, "_write_artifacts", lambda *args, **kwargs: None)

    compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    assert captured["lot_size"] == 100


def test_compile_run_writes_execution_assumptions_artifact(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="execution_assumptions", hypothesis="execution assumptions are auditable")
    spec.market.calendar = "XSHE"
    spec.execution.fill_price_mode = ""
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"
    spec.execution.cash_annual_return = 0.025
    spec.execution.lot_size = 1
    spec.execution.lot_size_config.default = 100
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    assumptions = json.loads((run_dir / "execution_assumptions.json").read_text(encoding="utf-8"))
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert assumptions == {
        "schema_version": 1,
        "calendar": "XSHE",
        "runtime_calendar": "XSHG",
        "order_timing": "next_session_open",
        "price_bar": "next_session",
        "price_type": "open",
        "fill_price_mode": "next_open",
        "compatibility_source": "explicit_fields",
        "cash_annual_return": 0.025,
        "lot_size": 1,
        "lot_size_config": {
            "default": 100,
            "by_symbol": {},
        },
        "rebalance": {
            "frequency": "daily",
            "interval_days": 1,
            "source": "execution.rebalance.interval_days",
        },
    }
    assert "execution_assumptions.json" in hashes
    assert audit_reproducibility(run_dir)["status"] == "pass"


def test_compile_run_writes_rebalance_rule_reasons_to_target_weights(tmp_path) -> None:
    spec = StrategySpec.template(
        strategy_id="rebalance_reasons",
        hypothesis="rebalance rule reasons should be auditable",
    )
    spec.universe.symbols = ["SPY"]
    spec.universe.point_in_time = True
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-05"]
    spec.validation.required_oos = False
    spec.execution.rebalance.interval_days = 3

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
    pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    rows = pd.read_csv(run_dir / "target_weights.csv")
    assert rows["reason"].str.contains("rebalance interval").any()


def test_compile_run_preserves_portfolio_rebalance_rule_in_runtime_artifacts(tmp_path) -> None:
    spec = StrategySpec.template(
        strategy_id="portfolio_rebalance_runtime",
        hypothesis="portfolio rebalance rule should compile into runtime semantics",
    )
    spec.universe.symbols = ["SPY"]
    spec.universe.point_in_time = True
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-05"]
    spec.validation.required_oos = False
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(
        type="RebalanceFrequencyRule",
        params={"interval_days": 3},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
    pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    compiled_plan = json.loads((run_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    run_spec = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml")
    rows = pd.read_csv(run_dir / "target_weights.csv")

    assert run_spec.portfolio.rules["rebalance"].params["interval_days"] == 3
    assert compiled_plan["execution"]["rebalance"] == {
        "frequency": "daily",
        "interval_days": 3,
        "source": "portfolio.rules.rebalance",
    }
    assert compiled_plan["runtime_rules"][0]["type"] == "RebalanceFrequencyRule"
    assert compiled_plan["runtime_rules"][0]["params"]["interval_days"] == 3
    assert compiled_plan["runtime_rules"][0]["source"] == "portfolio.rules.rebalance"
    assert rows["reason"].str.contains("rebalance interval").any()


def test_compile_plan_rejects_unsupported_rebalance_rule_params() -> None:
    spec = StrategySpec.template(
        strategy_id="rebalance_extra_params_compile",
        hypothesis="compile plan must not drop unsupported rebalance params",
    )
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(
        type="RebalanceFrequencyRule",
        params={"interval_days": 3, "calendar": "XNYS"},
    )

    with pytest.raises(ValueError, match="unsupported keys"):
        compile_plan(spec)


def test_compile_plan_rejects_unsupported_portfolio_rules() -> None:
    spec = StrategySpec.template(
        strategy_id="compile_plan_bad_rule",
        hypothesis="compile previews must not drop unsupported portfolio rules",
    )
    spec.portfolio.rules["stop_loss"] = PortfolioRuleDef(
        type="StopLossRule",
        params={"threshold": 0.05},
    )

    with pytest.raises(ValueError, match="portfolio_rule_unsupported"):
        compile_plan(spec)


def test_compile_run_records_resolved_default_data_dir(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="resolved_data_dir", hypothesis="default data dir is auditable")
    market_dir = tmp_path / "oxq_data" / "market"
    market_dir.mkdir(parents=True)
    dates = pd.bdate_range("2022-01-03", periods=3, tz="UTC")
    pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    ).to_parquet(market_dir / "SPY.parquet")
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "oxq_data"))

    _, run_dir = compile_run(spec, out_dir=tmp_path / "runs")

    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert env["data_dir"] == str(market_dir)
    assert audit_reproducibility(run_dir)["status"] == "pass"


def test_compile_run_records_absolute_data_dir_for_relative_input(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="relative_data_dir", hypothesis="artifact data_dir is cwd-independent")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.bdate_range("2022-01-03", periods=3, tz="UTC")
    pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")
    monkeypatch.chdir(tmp_path)

    _, run_dir = compile_run(spec, data_dir="data", out_dir=tmp_path / "runs")

    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert env["data_dir"] == str(data_dir.resolve())


def test_compile_run_uses_spec_market_currency(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="usd_currency", hypothesis="spec currency controls portfolio currency")
    spec.market.currency = "USD"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.bdate_range("2022-01-03", periods=3, tz="UTC")
    pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")

    result, _ = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    assert result.portfolio.currency == "USD"
    assert {fill.order.currency for fill in result.trades} <= {"USD"}


def test_compile_run_accepts_xshe_calendar_alias_and_preserves_manifest_calendar(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="xshe_calendar", hypothesis="XSHE aliases for runtime calendar resolution")
    spec.market.calendar = "XSHE"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["calendar"] == "XSHE"


def test_compile_run_expires_pending_buy_for_latched_sparse_symbol_missing_next_open(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="sparse_latch", hypothesis="latched sparse symbols do not duplicate buys")
    spec.universe.symbols = ["AAA", "BBB"]
    spec.benchmark.symbols = []
    spec.data.min_start_date = "2024-01-01"
    spec.data.required_columns = ["open", "high", "low", "close", "volume", "fast", "slow"]
    spec.signal.rules = {"entry": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"})}
    spec.validation.train_period = ["2024-01-01", "2024-01-01"]
    spec.validation.test_period = ["2024-01-02", "2024-01-05"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    aaa_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"], utc=True)
    pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
            "volume": [100, 100, 100, 100],
            "fast": [0.0, 2.0, 2.0, 2.0],
            "slow": [1.0, 1.0, 1.0, 1.0],
        },
        index=aaa_dates,
    ).to_parquet(data_dir / "AAA.parquet")
    bbb_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], utc=True)
    pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [100, 100, 100, 100],
            "fast": [0.0, 0.0, 0.0, 0.0],
            "slow": [1.0, 1.0, 1.0, 1.0],
        },
        index=bbb_dates,
    ).to_parquet(data_dir / "BBB.parquet")

    result, _ = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    buy_fills = [fill for fill in result.trades if fill.order.symbol == "AAA" and fill.order.side == "BUY"]
    assert len(buy_fills) == 0


def test_compile_run_rejects_multiple_crossover_rules(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="multi_cross", hypothesis="each crossover has an exit")
    spec.signal.indicators = {
        "fast_a": IndicatorDef(type="SMA", params={"period": 2}),
        "slow_a": IndicatorDef(type="SMA", params={"period": 3}),
        "fast_b": IndicatorDef(type="SMA", params={"period": 4}),
        "slow_b": IndicatorDef(type="SMA", params={"period": 5}),
    }
    spec.signal.rules = {
        "cross_a": SignalRuleDef(type="Crossover", params={"fast": "fast_a", "slow": "slow_a"}),
        "cross_b": SignalRuleDef(type="Crossover", params={"fast": "fast_b", "slow": "slow_b"}),
    }

    with pytest.raises(ValueError, match="Multiple Crossover"):
        compile_run(spec, out_dir=tmp_path)


def test_metrics_artifact_hash_excludes_run_id(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="metrics_hash", hypothesis="run metadata is not metrics fingerprint")
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

    first = tmp_path / "run_1"
    second = tmp_path / "run_2"
    first.mkdir()
    second.mkdir()
    _write_artifacts(spec, result, first, Engine())
    _write_artifacts(spec, result, second, Engine())

    first_hashes = json.loads((first / "artifact_hashes.json").read_text(encoding="utf-8"))
    second_hashes = json.loads((second / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert first_hashes["metrics.json"] == second_hashes["metrics.json"]


def test_environment_artifact_hash_excludes_run_timestamp(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="environment_hash", hypothesis="volatile timestamps are not core hashes")
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

    first = tmp_path / "run_1"
    second = tmp_path / "run_2"
    first.mkdir()
    second.mkdir()
    _write_artifacts(spec, result, first, Engine())
    _write_artifacts(spec, result, second, Engine())

    first_hashes = json.loads((first / "artifact_hashes.json").read_text(encoding="utf-8"))
    second_hashes = json.loads((second / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert first_hashes["environment.json"] == second_hashes["environment.json"]


def test_metrics_json_sanitizes_non_finite_values(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="finite_metrics", hypothesis="metrics json must be standard json")
    dates = pd.bdate_range("2024-01-01", periods=1, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    metrics_text = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in metrics_text
    assert "Infinity" not in metrics_text
    json.loads(metrics_text)


def test_default_metrics_profile_preserves_existing_values() -> None:
    spec = StrategySpec.template(strategy_id="default_metrics_profile", hypothesis="default profile keeps current formulas")
    dates = pd.bdate_range("2024-01-01", periods=6, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("105000")),
        trades=[],
        equity_curve=[(date, value) for date, value in zip(dates, [100000.0, 102000.0, 99000.0, 103000.0, 97000.0, 105000.0])],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["metrics_profile"] == "open_xquant_default"
    assert metrics["metric_assumptions"] == {
        "return_type": "simple",
        "risk_free_rate": 0.0,
        "annualization_days": 252,
        "calmar_denominator": "max_drawdown",
        "evaluation_window": "full",
    }
    assert metrics["annualized_return"] == pytest.approx(result.annualized_return())
    assert metrics["annualized_volatility"] == pytest.approx(result.annualized_volatility())
    assert metrics["sharpe_ratio"] == pytest.approx(result.sharpe_ratio())
    assert metrics["calmar_ratio"] == pytest.approx(result.calmar_ratio())


def test_metrics_json_records_profile_assumptions() -> None:
    spec = StrategySpec.template(strategy_id="xquant_metrics_profile", hypothesis="metrics profile should affect artifacts")
    spec.metrics.profile = "xquant_production"
    spec.metrics.risk_free_rate = 0.02
    spec.metrics.return_type = "log"
    spec.metrics.annualization_days = 252
    spec.metrics.calmar_denominator = "max_drawdown"
    spec.metrics.evaluation_window = "full"
    dates = pd.bdate_range("2024-01-01", periods=6, tz="UTC")
    values = np.array([100000.0, 102000.0, 101000.0, 106000.0, 104000.0, 109000.0])
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("109000")),
        trades=[],
        equity_curve=[(date, value) for date, value in zip(dates, values)],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    log_returns = np.diff(np.log(values))
    assert metrics["metrics_profile"] == "xquant_production"
    assert metrics["metric_assumptions"] == {
        "return_type": "log",
        "risk_free_rate": 0.02,
        "annualization_days": 252,
        "calmar_denominator": "max_drawdown",
        "evaluation_window": "full",
    }
    assert metrics["annualized_return"] == pytest.approx(float(np.mean(log_returns) * 252))
    assert metrics["sharpe_ratio"] == pytest.approx(
        float((np.mean(log_returns) - 0.02 / 252) / np.std(log_returns) * np.sqrt(252))
    )


def test_metrics_evaluation_window_oos_uses_oos_top_level_values() -> None:
    spec = StrategySpec.template(strategy_id="oos_metric_window", hypothesis="top-level metrics can use oos window")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-03", "2024-01-05"]
    spec.metrics.evaluation_window = "oos"
    dates = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("150000")),
        trades=[
            Fill(
                order=Order(symbol="AAA", side="BUY", shares=1),
                filled_price=Decimal("1"),
                filled_at=dates[1].isoformat(),
                fee=Decimal("2"),
            ),
            Fill(
                order=Order(symbol="AAA", side="SELL", shares=1),
                filled_price=Decimal("1"),
                filled_at=dates[3].isoformat(),
                fee=Decimal("3"),
            ),
        ],
        equity_curve=[
            (dates[0], 100000.0),
            (dates[1], 110000.0),
            (dates[2], 121000.0),
            (dates[3], 133100.0),
            (dates[4], 146410.0),
        ],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["total_return"] == pytest.approx(metrics["oos_total_return"])
    assert metrics["annualized_return"] == pytest.approx(metrics["oos_annualized_return"])
    assert metrics["sharpe_ratio"] == pytest.approx(metrics["oos_sharpe_ratio"])
    assert metrics["is_total_return"] == pytest.approx(0.1)
    assert metrics["oos_total_return"] == pytest.approx(0.331)
    assert metrics["trade_count"] == 1
    assert metrics["cost_paid"] == pytest.approx(3.0)


def test_is_metrics_include_train_period_end_date_with_intraday_timestamps() -> None:
    spec = StrategySpec.template(strategy_id="is_metric_end", hypothesis="is metrics should include train end date")
    spec.validation.train_period = ["2024-01-01", "2024-01-03"]
    spec.validation.test_period = ["2024-01-04", "2024-01-05"]
    dates = [pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=16) for day in pd.bdate_range("2024-01-01", periods=5)]
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("133100")),
        trades=[],
        equity_curve=[
            (dates[0], 100000.0),
            (dates[1], 110000.0),
            (dates[2], 121000.0),
            (dates[3], 133100.0),
            (dates[4], 133100.0),
        ],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["is_total_return"] == pytest.approx(0.21)


def test_oos_metrics_respect_test_period_end() -> None:
    spec = StrategySpec.template(strategy_id="oos_metric_end", hypothesis="oos metrics should stop at test end")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-03", "2024-01-05"]
    dates = pd.bdate_range("2024-01-01", periods=6, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("300000")),
        trades=[
            Fill(
                order=Order(symbol="AAA", side="BUY", shares=1),
                filled_price=Decimal("1"),
                filled_at=dates[2].isoformat(),
            ),
            Fill(
                order=Order(symbol="AAA", side="SELL", shares=1),
                filled_price=Decimal("1"),
                filled_at=dates[5].isoformat(),
            ),
        ],
        equity_curve=[
            (dates[0], 100000.0),
            (dates[1], 100000.0),
            (dates[2], 110000.0),
            (dates[3], 121000.0),
            (dates[4], 133100.0),
            (dates[5], 300000.0),
        ],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["oos_total_return"] == pytest.approx(0.331)
    assert metrics["oos_trade_count"] == 1


def test_metrics_evaluation_window_oos_unavailable_does_not_use_full_window() -> None:
    spec = StrategySpec.template(strategy_id="oos_metric_missing", hypothesis="oos metrics should not fall back")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-10", "2024-01-12"]
    spec.metrics.evaluation_window = "oos"
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("150000")),
        trades=[
            Fill(
                order=Order(symbol="AAA", side="BUY", shares=1),
                filled_price=Decimal("1"),
                filled_at=dates[1].isoformat(),
                fee=Decimal("2"),
            )
        ],
        equity_curve=[(date, value) for date, value in zip(dates, [100000.0, 125000.0, 150000.0])],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["metric_assumptions"]["evaluation_window"] == "oos"
    assert metrics["metric_diagnostics"] == ["evaluation_window=oos unavailable: OOS equity curve has fewer than 2 points"]
    assert metrics["total_return"] is None
    assert metrics["annualized_return"] is None
    assert metrics["sharpe_ratio"] is None
    assert metrics["trade_count"] == 0
    assert metrics["cost_paid"] == pytest.approx(0.0)


def test_metrics_evaluation_window_oos_unavailable_for_short_full_run() -> None:
    spec = StrategySpec.template(strategy_id="oos_metric_short", hypothesis="short runs should not fall back")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-10", "2024-01-12"]
    spec.metrics.evaluation_window = "oos"
    dates = pd.bdate_range("2024-01-01", periods=1, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0)],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["metric_assumptions"]["evaluation_window"] == "oos"
    assert metrics["metric_diagnostics"] == ["evaluation_window=oos unavailable: OOS equity curve has fewer than 2 points"]
    assert metrics["total_return"] is None
    assert metrics["annualized_return"] is None
    assert metrics["sharpe_ratio"] is None


def test_oos_metrics_include_test_start_baseline() -> None:
    spec = StrategySpec.template(strategy_id="oos_baseline", hypothesis="oos metrics include first test-day move")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-03", "2024-01-05"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("80000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 80000.0), (dates[2], 80000.0)],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["oos_total_return"] == pytest.approx(-0.2)
    assert metrics["oos_max_drawdown"] == pytest.approx(-0.2)


def test_oos_metrics_handle_zero_baseline() -> None:
    spec = StrategySpec.template(strategy_id="oos_zero", hypothesis="zero baseline cannot produce returns")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-03", "2024-01-05"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("0")),
        trades=[],
        equity_curve=[(dates[0], 0.0), (dates[1], 0.0), (dates[2], 0.0)],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["oos_total_return"] is None
    assert metrics["oos_sharpe_ratio"] is None
    assert metrics["oos_max_drawdown"] is None


def test_reproducibility_audit_fails_when_artifact_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="tamper_test", hypothesis="artifact tampering fails audit")
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
    (tmp_path / "metrics.json").write_text('{"strategy_id": "tampered"}\n', encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "metrics_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_data_manifest_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="manifest_tamper", hypothesis="manifest tampering fails audit")
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
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["price_adjustment"] = "raw"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_environment_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="environment_tamper", hypothesis="environment tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    env["data_dir"] = str(tmp_path / "other_data")
    (tmp_path / "environment.json").write_text(json.dumps(env), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "environment_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_raw_spec_file_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="raw_spec_tamper", hypothesis="raw spec file tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    (tmp_path / "strategy_spec.yaml").write_text(
        (tmp_path / "strategy_spec.yaml").read_text(encoding="utf-8") + "\nunknown_field: changed\n",
        encoding="utf-8",
    )

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "strategy_spec_file_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_handles_corrupt_metrics_json(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="corrupt_metrics", hypothesis="corrupt metrics fail audit")
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
    (tmp_path / "metrics.json").write_text("{not-json", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "metrics_hash" and check["status"] == "fail" for check in audit["checks"])


def test_data_manifest_uses_test_period_start_when_train_period_absent(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="test_only", hypothesis="test-only specs record actual range")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
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

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["start"] == "2024-01-01"
    assert manifest["end"] == "2024-01-03"


def test_data_manifest_start_includes_min_start_date(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="warmup_start", hypothesis="manifest includes warmup data range")
    spec.data.min_start_date = "2023-12-01"
    dates = pd.bdate_range("2023-12-01", periods=3, tz="UTC")
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

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["start"] == "2023-12-01"


def test_reproducibility_audit_handles_corrupt_artifact_hashes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="corrupt_hashes", hypothesis="corrupt hash manifest fails cleanly")
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
    (tmp_path / "artifact_hashes.json").write_text("{not-json", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_handles_non_dict_artifact_hashes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="list_hashes", hypothesis="hash manifest type fails cleanly")
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
    (tmp_path / "artifact_hashes.json").write_text("[]", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_handles_bad_artifact_hash_schema_version(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_hash_schema", hypothesis="bad hash schema fails cleanly")
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
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["schema_version"] = "bad"
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_on_empty_artifact_hashes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="empty_hashes", hypothesis="empty hash manifest fails")
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
    (tmp_path / "artifact_hashes.json").write_text("{}\n", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_artifact_hash_key_is_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_hash_key", hypothesis="hash manifest keys are required")
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
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes.pop("metrics.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_rejects_hash_manifest_schema_downgrade(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="hash_downgrade", hypothesis="new artifacts cannot downgrade hash schema")
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
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes.pop("schema_version")
    hashes.pop("environment.json")
    hashes.pop("strategy_spec.yaml")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_rejects_fingerprint_symbol_path_traversal(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="unsafe_fingerprint_symbol", hypothesis="fingerprint symbols are paths")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"]["../outside"] = manifest["data_fingerprints"].pop("SPY")
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert "../outside" in fingerprint_check["message"]


def test_reproducibility_audit_rejects_malformed_v1_data_fingerprints(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_fingerprint_schema", hypothesis="fingerprints must be an object")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"] = ["SPY"]
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert fingerprint_check["severity"] == "fatal"


def test_reproducibility_audit_rejects_non_object_fingerprint_entries(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_fingerprint_entry", hypothesis="fingerprint entries must be objects")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"]["SPY"] = "bad"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert fingerprint_check["severity"] == "fatal"
    assert "fingerprint must be an object" in fingerprint_check["message"]


def test_reproducibility_audit_rejects_invalid_data_manifest_schema_version(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_manifest_schema", hypothesis="manifest schema version must parse")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "bad"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_rejects_non_object_environment_json(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_env_type", hypothesis="environment json must be an object")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": pd.DataFrame(
            {"open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1]},
            index=dates,
        )},
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "environment.json").write_text("[]", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "environment" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_rejects_non_object_data_manifest_json(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_manifest_type", hypothesis="manifest json must be an object")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": pd.DataFrame(
            {"open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1]},
            index=dates,
        )},
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "data_manifest.json").write_text("[]", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_rejects_invalid_manifest_symbols(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_manifest_symbols", hypothesis="manifest symbols must be a list")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": pd.DataFrame(
            {"open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1]},
            index=dates,
        )},
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["symbols"] = 1
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_requires_fingerprints_for_all_manifest_symbols(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="fingerprint_coverage", hypothesis="all symbols need source fingerprints")
    spec.universe.symbols = ["SPY", "QQQ"]
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": frame, "QQQ": frame},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame.to_parquet(data_dir / "SPY.parquet")
    frame.to_parquet(data_dir / "QQQ.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"].pop("QQQ")
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert "QQQ" in fingerprint_check["message"]


def test_reproducibility_audit_allows_legacy_artifacts_without_source_fingerprints(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="legacy_artifact", hypothesis="old artifacts use legacy audit")
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
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest.pop("schema_version", None)
    manifest.pop("data_fingerprints", None)
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    current_hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes = {
        "data_manifest.json": "sha256:" + hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()[:16],
        "equity_curve.csv": current_hashes["equity_curve.csv"],
        "trades.csv": current_hashes["trades.csv"],
        "metrics.json": current_hashes["metrics.json"],
    }
    (tmp_path / "execution_assumptions.json").unlink(missing_ok=True)
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (tmp_path.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": tmp_path.name, "artifact_hashes": _hash_json_file(tmp_path / "artifact_hashes.json")}) + "\n",
        encoding="utf-8",
    )

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert fingerprint_check["severity"] == "warning"


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


def test_signal_filtered_equal_weight_ignores_negative_numeric_signals() -> None:
    spec = StrategySpec.template(strategy_id="negative_signal", hypothesis="negative signals should not go long")
    spec.signal.rules = {
        "threshold": SignalRuleDef(type="Threshold", params={"column": "score", "threshold": 0.0}),
    }

    optimizer = _build_optimizer(spec)
    signal_bar = pd.DataFrame({"threshold": [-1]})

    assert optimizer.optimize({"SPY": signal_bar}, {"SPY": signal_bar}) == {"CASH": 1.0}


def test_signal_filtered_equal_weight_uses_only_terminal_composite_signal() -> None:
    spec = StrategySpec.template(strategy_id="composite_terminal", hypothesis="composite is the terminal entry signal")
    spec.signal.rules = {
        "above": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "combo": SignalRuleDef(type="Composite", params={"signals": ["above", "trend"], "logic": "and"}),
    }

    optimizer = _build_optimizer(spec)
    signal_bar = pd.DataFrame({"above": [True], "trend": [False], "combo": [False]})

    assert optimizer.optimize({"SPY": signal_bar}, {"SPY": signal_bar}) == {"CASH": 1.0}


def test_terminal_composite_with_crossover_latches_until_exit_reset() -> None:
    spec = StrategySpec.template(strategy_id="composite_cross_latch", hypothesis="event composite holds until exit")
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
        "filter": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "entry": SignalRuleDef(type="Composite", params={"signals": ["cross", "filter"], "logic": "and"}),
    }

    optimizer = _build_optimizer(spec)
    entry_bar = pd.DataFrame({"cross": [True], "filter": [True], "entry": [True]})
    inactive_bar = pd.DataFrame({"cross": [False], "filter": [True], "entry": [False]})

    assert optimizer.optimize({"SPY": entry_bar}, {"SPY": entry_bar}) == {"SPY": 1.0}
    optimizer.set_held_symbols(["SPY"])
    assert optimizer.optimize({"SPY": inactive_bar}, {"SPY": inactive_bar}) == {"SPY": 1.0}

    optimizer.reset_symbols(["SPY"])

    assert optimizer.optimize({"SPY": inactive_bar}, {"SPY": inactive_bar}) == {"CASH": 1.0}


def test_compile_strategy_rejects_or_composite_mixing_event_and_level_signals() -> None:
    spec = StrategySpec.template(strategy_id="or_mixed_lifecycle", hypothesis="or composites must not mix lifecycles")
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
        "filter": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "entry": SignalRuleDef(type="Composite", params={"signals": ["cross", "filter"], "logic": "or"}),
    }

    with pytest.raises(ValueError, match="cannot mix event and level signals"):
        compile_strategy(spec)


def test_compile_strategy_rejects_multiple_terminal_signal_rules() -> None:
    spec = StrategySpec.template(strategy_id="multi_terminal", hypothesis="ambiguous signal rules must not imply or")
    spec.signal.rules = {
        "above": SignalRuleDef(type="Threshold", params={"column": "close", "threshold": 1.0}),
        "trend": SignalRuleDef(type="Threshold", params={"column": "volume", "threshold": 1.0}),
    }

    with pytest.raises(ValueError, match="Exactly one terminal signal rule"):
        compile_strategy(spec)


def test_signal_filtered_equal_weight_keeps_latched_event_symbol_without_current_bar() -> None:
    spec = StrategySpec.template(strategy_id="sparse_latch", hypothesis="latched event holdings survive sparse calendars")
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
    }

    optimizer = _build_optimizer(spec)
    entry_bar = pd.DataFrame({"cross": [True]})

    assert optimizer.optimize({"SPY": entry_bar}, {"SPY": entry_bar}) == {"SPY": 1.0}
    optimizer.set_held_symbols(["SPY"])
    assert optimizer.optimize({}, {}) == {"SPY": 1.0}


def test_compile_universe_rejects_unsupported_universe_type() -> None:
    spec = StrategySpec.template(strategy_id="unsupported_universe", hypothesis="unsupported universes fail clearly")
    spec.universe.type = "filter"

    with pytest.raises(ValueError, match="Unsupported universe.type 'filter'"):
        compile_universe(spec)


def test_compile_strategy_excludes_universe_from_strategy_body() -> None:
    spec = StrategySpec.template(strategy_id="strategy_without_universe", hypothesis="strategy logic is reusable")
    spec.universe.symbols = ["SPY", "QQQ"]

    strategy = compile_strategy(spec)
    universe = compile_universe(spec)

    assert getattr(strategy, "_legacy_universe", None) is None
    assert tuple(universe.symbols) == ("SPY", "QQQ")


def test_compile_strategy_populates_runtime_rules_from_portfolio_rules() -> None:
    spec = StrategySpec.template(
        strategy_id="strategy_rules",
        hypothesis="strategy runtime rules should be reusable with the strategy body",
    )
    spec.portfolio.rules["rebalance"] = PortfolioRuleDef(
        type="RebalanceFrequencyRule",
        params={"interval_days": 3},
    )

    strategy = compile_strategy(spec)

    assert len(strategy.rules) == 1
    assert strategy.rules[0].name == "RebalanceFrequencyRule"
    assert strategy.rules[0].interval_days == 3


def test_roc_timing_signal_to_position_compiles_and_writes_target_weights(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 102, 103, 104, 105, 106, 107, 108],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100, 90, 89, 94, 110, 111, 105, 104],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-02", periods=8, freq="B", tz="UTC"),
    )
    frame.to_parquet(data_dir / "CSI300.parquet")

    spec = StrategySpec.template(
        strategy_id="roc_timing_position",
        hypothesis="ROC timing target positions are auditable",
    )
    spec.universe.symbols = ["CSI300"]
    spec.signal.indicators = {
        "roc_1": IndicatorDef(type="ROC", params={"column": "close", "period": 1})
    }
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={"column": "roc_1", "mode": "fixed", "bottom": -5.0, "top": 10.0},
        )
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "timing"}
    spec.validation.train_period = ["2024-01-02", "2024-01-05"]
    spec.validation.test_period = ["2024-01-08", "2024-01-11"]
    spec.benchmark.symbols = ["CSI300"]
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    weights = pd.read_csv(run_dir / "target_weights.csv")
    csi = weights[weights["symbol"] == "CSI300"]
    assert csi["adjusted_target_weight"].max() == 1.0
    assert "target_changed" in set(weights["reason"])


def _adjusted_weight_sequence(weights: pd.DataFrame, symbol: str) -> list[float]:
    dates = weights["date"].drop_duplicates().tolist()
    by_date = (
        weights[weights["symbol"] == symbol]
        .set_index("date")["adjusted_target_weight"]
        .to_dict()
    )
    return [float(by_date.get(date, 0.0)) for date in dates]


def test_roc_timing_fixed_threshold_target_weight_sequence(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "open": [100, 90, 89, 94, 110, 111, 105, 104],
            "high": [101, 91, 90, 95, 111, 112, 106, 105],
            "low": [99, 89, 88, 93, 109, 110, 104, 103],
            "close": [100, 90, 89, 94, 110, 111, 105, 104],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-02", periods=8, freq="B", tz="UTC"),
    )
    frame.to_parquet(data_dir / "CSI300.parquet")

    spec = StrategySpec.template(
        strategy_id="roc_timing_fixed_acceptance",
        hypothesis="fixed ROC timing target weights match expected state transitions",
    )
    spec.universe.symbols = ["CSI300"]
    spec.signal.indicators = {
        "roc_1": IndicatorDef(type="ROC", params={"column": "close", "period": 1})
    }
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={"column": "roc_1", "mode": "fixed", "bottom": -5.0, "top": 10.0},
        )
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "timing"}
    spec.validation.train_period = ["2024-01-02", "2024-01-05"]
    spec.validation.test_period = ["2024-01-08", "2024-01-11"]
    spec.benchmark.symbols = ["CSI300"]
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    weights = pd.read_csv(run_dir / "target_weights.csv")
    assert _adjusted_weight_sequence(weights, "CSI300") == [
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]

    trades = pd.read_csv(run_dir / "trades.csv")
    assert trades["side"].tolist() == ["BUY", "SELL", "BUY"]


def test_roc_timing_rolling_quantile_target_weight_sequence(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "open": [5, 4, 3, 1, 2, 6, 7, 6],
            "high": [5, 4, 3, 1, 2, 6, 7, 6],
            "low": [5, 4, 3, 1, 2, 6, 7, 6],
            "close": [5, 4, 3, 1, 2, 6, 7, 6],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-02", periods=8, freq="B", tz="UTC"),
    )
    frame.to_parquet(data_dir / "CSI300.parquet")

    spec = StrategySpec.template(
        strategy_id="roc_timing_rolling_acceptance",
        hypothesis="rolling ROC timing uses prior-window thresholds for target weights",
    )
    spec.universe.symbols = ["CSI300"]
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={
                "column": "close",
                "mode": "rolling_quantile",
                "q_window": 3,
                "q_bottom": 0.0,
                "q_top": 1.0,
            },
        )
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "timing"}
    spec.validation.train_period = ["2024-01-02", "2024-01-05"]
    spec.validation.test_period = ["2024-01-08", "2024-01-11"]
    spec.benchmark.symbols = ["CSI300"]
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    weights = pd.read_csv(run_dir / "target_weights.csv")
    assert _adjusted_weight_sequence(weights, "CSI300") == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
