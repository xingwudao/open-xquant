from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd

from oxq.audit.reproducibility import _hash_json_file, audit_reproducibility
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _write_artifacts
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
