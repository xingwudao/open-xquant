from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from oxq.audit.reproducibility import _hash_json_file, audit_reproducibility
from oxq.core.component_manifest import compute_component_bundle_hash
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
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
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


def test_reproducibility_audit_requires_complete_provenance_bundle(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    spec_audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
        "spec_hash": (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip(),
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "4" * 64,
        "recipe_matches": [],
        "field_audits": [],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    (run_dir / "spec_audit.json").write_text(json.dumps(spec_audit), encoding="utf-8")
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["spec_audit.json"] = _hash_json_file(run_dir / "spec_audit.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    artifact_hash_check = next(check for check in audit["checks"] if check["id"] == "artifact_hashes")
    assert "conversation_hash.txt" in artifact_hash_check["message"]
    assert "component_catalog_hash.txt" in artifact_hash_check["message"]
    assert "recipe_catalog_hash.txt" in artifact_hash_check["message"]


def test_reproducibility_audit_rejects_unsafe_unknown_artifact_paths(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["../other_run/metrics.json"] = "sha256:" + "0" * 16
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    artifact_hash_check = next(
        check
        for check in audit["checks"]
        if check["id"] == "artifact_hashes" and "unsafe artifact paths" in check["message"]
    )
    assert "../other_run/metrics.json" in artifact_hash_check["message"]


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


def test_reproducibility_audit_hashes_provenance_json_canonically(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    (run_dir / "runtime_audit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "runtime_semantics_pass": True,
                "spec_hash": (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip(),
                "spec_audit_hash": "sha256:" + "1" * 16,
                "compiled_plan_hash": _hash_json_file(run_dir / "compiled_plan.json"),
                "compiled_plan_path": "compiled_plan.json",
                "material_field_audits": [],
                "blocking_findings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "component_manifests.json").write_text(
        json.dumps([], indent=2),
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["runtime_audit.json"] = _hash_json_file(run_dir / "runtime_audit.json")
    hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)
    runtime_payload = json.loads((run_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    component_payload = json.loads((run_dir / "component_manifests.json").read_text(encoding="utf-8"))
    (run_dir / "runtime_audit.json").write_text(json.dumps(runtime_payload, sort_keys=True), encoding="utf-8")
    (run_dir / "component_manifests.json").write_text(json.dumps(component_payload, sort_keys=True), encoding="utf-8")

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"
    assert any(check["id"] == "runtime_audit_hash" and check["status"] == "pass" for check in audit["checks"])
    assert any(check["id"] == "component_manifests_hash" and check["status"] == "pass" for check in audit["checks"])


def test_reproducibility_audit_verifies_archived_component_source(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    archive_base = run_dir / "component_extensions" / "00_custom_components"
    source_dir = archive_base / "custom_components" / "oxq_components" / "indicators"
    tests_dir = archive_base / "custom_components" / "tests"
    source_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (archive_base / "custom_components" / "oxq_components" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    source = source_dir / "archived_indicator.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "class ArchivedIndicator:",
                "    name = 'ArchivedIndicator'",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_file = tests_dir / "test_archived_indicator.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    manifest = archive_base / "workspace_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "ArchivedIndicator",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "oxq_components.indicators.archived_indicator",
                        "class": "ArchivedIndicator",
                        "protocol": "Indicator",
                        "tests": ["custom_components/tests/test_archived_indicator.py"],
                        "source_path": "oxq_components/indicators/archived_indicator.py",
                        "source_hash": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                        "test_hash": "sha256:" + hashlib.sha256(test_file.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bundle_hash = compute_component_bundle_hash(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = bundle_hash
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "component_manifests.json").write_text(
        json.dumps(
            [
                {
                    "manifest_path": "/deleted/workspace_manifest.json",
                    "archived_manifest_path": "component_extensions/00_custom_components/workspace_manifest.json",
                    "archived_extension_root": "component_extensions/00_custom_components/custom_components",
                    "bundle_hash": bundle_hash,
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    source.write_text(source.read_text(encoding="utf-8").replace("1.0", "2.0"), encoding="utf-8")

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(check["id"] == "component_bundle_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_prefers_legacy_run_manifest_over_mutable_manifest_path(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    run_manifest, run_bundle_hash = _write_component_manifest_bundle(run_dir, value="1.0")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_manifest, _workspace_bundle_hash = _write_component_manifest_bundle(workspace, value="2.0")
    (run_dir / "component_manifests.json").write_text(
        json.dumps(
            [
                {
                    "manifest_path": str(workspace_manifest),
                    "bundle_hash": run_bundle_hash,
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["component_manifest.json"] = _hash_json_file(run_manifest)
    hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"
    assert any(check["id"] == "component_bundle_hash" and check["status"] == "pass" for check in audit["checks"])


def test_reproducibility_audit_rejects_missing_archived_component_manifest(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_manifest, bundle_hash = _write_component_manifest_bundle(workspace, value="1.0")
    (run_dir / "component_manifests.json").write_text(
        json.dumps(
            [
                {
                    "manifest_path": str(workspace_manifest),
                    "archived_manifest_path": "component_extensions/00_custom_components/component_manifest.json",
                    "archived_extension_root": "component_extensions/00_custom_components/custom_components",
                    "bundle_hash": bundle_hash,
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(
        check["id"] == "component_bundle_hash" and "archived component manifest not found" in check["message"]
        for check in audit["checks"]
    )


def test_reproducibility_audit_rejects_tampered_archived_manifest_bundle_hash(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    manifest, bundle_hash = _write_component_manifest_bundle(run_dir, value="1.0")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = "sha256:tampered"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "component_manifests.json").write_text(
        json.dumps(
            [
                {
                    "manifest_path": "/deleted/component_manifest.json",
                    "bundle_hash": bundle_hash,
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["component_manifest.json"] = _hash_json_file(manifest)
    hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(
        check["id"] == "component_bundle_hash" and "manifest hash mismatch" in check["message"]
        for check in audit["checks"]
    )


def test_reproducibility_audit_rejects_symlinked_archived_component_manifest(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    external_manifest = tmp_path / "external_component_manifest.json"
    external_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "sha256:external",
                "components": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "archived_component_manifest.json").symlink_to(external_manifest)
    (run_dir / "component_manifests.json").write_text(
        json.dumps(
            [
                {
                    "manifest_path": "/deleted/component_manifest.json",
                    "archived_manifest_path": "archived_component_manifest.json",
                    "bundle_hash": "sha256:external",
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(
        check["id"] == "component_bundle_hash" and "must not be a symlink" in check["message"]
        for check in audit["checks"]
    )


def test_reproducibility_audit_rejects_symlinked_legacy_component_manifest(tmp_path) -> None:
    run_dir = _write_minimal_run(tmp_path)
    external_manifest = tmp_path / "external_component_manifest.json"
    external_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "sha256:external",
                "components": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "component_manifest.json").symlink_to(external_manifest)
    (run_dir / "component_manifests.json").write_text(
        json.dumps(
            [
                {
                    "manifest_path": "/deleted/component_manifest.json",
                    "bundle_hash": "sha256:external",
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["component_manifest.json"] = _hash_json_file(run_dir / "component_manifest.json")
    hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    _write_current_run_digest(run_dir)

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(
        check["id"] == "component_bundle_hash" and "archived component manifest must not be a symlink" in check["message"]
        for check in audit["checks"]
    )


def _write_component_manifest_bundle(base: Path, *, value: str) -> tuple[Path, str]:
    root = base / "custom_components"
    source_dir = root / "oxq_components" / "indicators"
    tests_dir = root / "tests"
    source_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (root / "oxq_components" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    source = source_dir / "archived_indicator.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "class ArchivedIndicator:",
                "    name = 'ArchivedIndicator'",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                f"        return pd.Series({value}, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_file = tests_dir / "test_archived_indicator.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    manifest = base / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "ArchivedIndicator",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "oxq_components.indicators.archived_indicator",
                        "class": "ArchivedIndicator",
                        "protocol": "Indicator",
                        "tests": ["custom_components/tests/test_archived_indicator.py"],
                        "source_path": "oxq_components/indicators/archived_indicator.py",
                        "source_hash": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                        "test_hash": "sha256:" + hashlib.sha256(test_file.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bundle_hash = compute_component_bundle_hash(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = bundle_hash
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest, bundle_hash


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


def _write_current_run_digest(run_dir) -> None:
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": run_dir.name, "artifact_hashes": _hash_json_file(run_dir / "artifact_hashes.json")})
        + "\n",
        encoding="utf-8",
    )
