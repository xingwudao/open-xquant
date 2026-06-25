from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd

from oxq.audit.reproducibility import _hash_json_file, audit_reproducibility
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _build_strategy_py_artifact, _hash_file, _write_artifacts
from oxq.spec.schema import StrategySpec


def test_reproducibility_audit_accepts_xshe_calendar_alias(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="xshe_audit", hypothesis="audit resolves aliased exchange calendars")
    spec.market.calendar = "XSHE"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    source_df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": source_df},
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


def test_reproducibility_audit_requires_execution_assumptions_for_schema_v2_hashes(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    (run_dir / "execution_assumptions.json").unlink()
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes.pop("execution_assumptions.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(
        check["id"] == "artifact_hashes"
        and check["status"] == "fail"
        and "execution_assumptions.json" in check["message"]
        for check in audit["checks"]
    )


def test_reproducibility_audit_allows_schema_v1_without_execution_assumptions(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    (run_dir / "execution_assumptions.json").unlink()
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["schema_version"] = 1
    hashes.pop("execution_assumptions.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"


def test_reproducibility_audit_allows_legacy_schema_zero_without_execution_assumptions(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    (run_dir / "execution_assumptions.json").unlink()
    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    manifest.pop("schema_version")
    manifest.pop("data_fingerprints")
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    current_hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes = {
        "schema_version": 0,
        "data_manifest.json": _hash_json_file(run_dir / "data_manifest.json"),
        "equity_curve.csv": current_hashes["equity_curve.csv"],
        "trades.csv": current_hashes["trades.csv"],
        "metrics.json": current_hashes["metrics.json"],
    }
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"


def test_reproducibility_audit_validates_target_weight_hash(tmp_path) -> None:
    import json

    import pandas as pd

    from oxq.audit import audit_reproducibility
    from oxq.spec.compiler import compile_run
    from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 102, 104, 103, 106, 108],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC"),
    )
    frame.to_parquet(data_dir / "SPY.parquet")

    spec = StrategySpec.template(
        strategy_id="target_weight_hash",
        hypothesis="target weight artifact tampering fails audit",
    )
    spec.universe.symbols = ["SPY"]
    spec.signal.indicators = {
        "roc_1": IndicatorDef(type="ROC", params={"column": "close", "period": 1})
    }
    spec.signal.rules = {
        "positive": SignalRuleDef(
            type="Threshold",
            params={"column": "roc_1", "threshold": 0, "relationship": "gt"},
        )
    }
    spec.validation.train_period = ["2024-01-02", "2024-01-04"]
    spec.validation.test_period = ["2024-01-05", "2024-01-09"]
    spec.benchmark.symbols = ["SPY"]
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")
    rows = pd.read_csv(run_dir / "target_weights.csv")
    rows.loc[0, "adjusted_target_weight"] = 0.25
    rows.to_csv(run_dir / "target_weights.csv", index=False)

    audit = audit_reproducibility(run_dir)
    assert audit["status"] == "fail"
    assert any(check["id"] == "target_weights_hash" for check in audit["checks"])

    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert hashes["schema_version"] == 5


def test_reproducibility_audit_validates_compiled_plan_hash(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    plan = json.loads((run_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    plan["compilation_mode"] = "tampered"
    (run_dir / "compiled_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(check["id"] == "compiled_plan_hash" for check in audit["checks"])


def test_reproducibility_audit_validates_attached_provenance_hashes(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    catalog_hash = "sha256:" + "4" * 64
    recipe_catalog_hash = "sha256:" + "5" * 64
    spec_audit = {
        "schema_version": 1,
        "status": "pass",
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog_hash,
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "portfolio.type",
                "spec_value": "EqualWeight",
                "status": "confirmed",
                "evidence": [],
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    (run_dir / "spec_audit.json").write_text(json.dumps(spec_audit, indent=2) + "\n", encoding="utf-8")
    (run_dir / "conversation_hash.txt").write_text(spec_audit["conversation_hash"] + "\n", encoding="utf-8")
    (run_dir / "component_catalog_hash.txt").write_text(catalog_hash + "\n", encoding="utf-8")
    (run_dir / "recipe_catalog_hash.txt").write_text(recipe_catalog_hash + "\n", encoding="utf-8")
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes.update(
        {
            "spec_audit.json": _hash_json_file(run_dir / "spec_audit.json"),
            "conversation_hash.txt": _hash_file(run_dir / "conversation_hash.txt"),
            "component_catalog_hash.txt": _hash_file(run_dir / "component_catalog_hash.txt"),
            "recipe_catalog_hash.txt": _hash_file(run_dir / "recipe_catalog_hash.txt"),
        }
    )
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": run_dir.name, "artifact_hashes": _hash_json_file(run_dir / "artifact_hashes.json")})
        + "\n",
        encoding="utf-8",
    )

    spec_audit["status"] = "block"
    (run_dir / "spec_audit.json").write_text(json.dumps(spec_audit, indent=2) + "\n", encoding="utf-8")

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(check["id"] == "spec_audit_hash" for check in audit["checks"])


def test_reproducibility_audit_rejects_compiled_plan_spec_hash_conflict(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    plan_path = run_dir / "compiled_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["spec_hash"] = "sha256:0000000000000000"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan_hash = _hash_json_file(plan_path)
    spec = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml")
    strategy_py_path = run_dir / "strategy.py"
    strategy_py_path.write_text(
        _build_strategy_py_artifact(spec, plan, spec_hash, plan_hash),
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["compiled_plan.json"] = plan_hash
    hashes["strategy.py"] = _hash_file(strategy_py_path)
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    consistency = next(check for check in audit["checks"] if check["id"] == "strategy_py_consistency")
    assert "compiled_plan.json spec_hash mismatch" in consistency["message"]


def test_reproducibility_audit_rejects_strategy_py_spec_conflict(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    strategy_py_path = run_dir / "strategy.py"
    strategy_py = strategy_py_path.read_text(encoding="utf-8")
    strategy_py_path.write_text(
        strategy_py.replace("'strategy_id': 'audit_execution_assumptions'", "'strategy_id': 'tampered'"),
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["strategy.py"] = _hash_file(strategy_py_path)
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    consistency = next(check for check in audit["checks"] if check["id"] == "strategy_py_consistency")
    assert "STRATEGY_SPEC conflicts" in consistency["message"]


def test_reproducibility_audit_allows_schema_4_without_strategy_py(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    (run_dir / "strategy.py").unlink()
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["schema_version"] = 4
    hashes.pop("strategy.py")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"


def test_reproducibility_audit_allows_schema_3_without_compiled_plan(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    (run_dir / "strategy.py").unlink()
    (run_dir / "compiled_plan.json").unlink()
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["schema_version"] = 3
    hashes.pop("strategy.py")
    hashes.pop("compiled_plan.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"


def test_reproducibility_audit_checks_target_weights_when_schema_2_manifest(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    hashes_path = run_dir / "artifact_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["schema_version"] = 2
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    target_weights = pd.read_csv(run_dir / "target_weights.csv")
    target_weights.loc[0, "adjusted_target_weight"] = 0.25
    target_weights.to_csv(run_dir / "target_weights.csv", index=False)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(check["id"] == "target_weights_hash" for check in audit["checks"])


def _write_minimal_run(tmp_path):
    spec = StrategySpec.template(strategy_id="audit_execution_assumptions", hypothesis="audit execution assumptions")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    source_df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": source_df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))
    return run_dir
