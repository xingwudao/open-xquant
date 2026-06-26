import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.spec.compiler import _hash_json_file, compile_plan
from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec


def _write_spec_and_data(tmp_path, *, evaluation_window: str = "full"):
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
        strategy_id="json_backtest",
        hypothesis="json backtest output supports agents",
    )
    spec.universe.symbols = ["SPY"]
    spec.universe.point_in_time = True
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
    spec.metrics.evaluation_window = evaluation_window
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    return spec_path, data_dir


def _write_spec_audit(path: Path, spec_hash: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "pass",
                "spec_provenance_pass": True,
                "spec_hash": spec_hash,
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                "recipe_matches": [],
                "field_audits": [],
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_runtime_audit(
    path: Path,
    spec_hash: str,
    *,
    runtime_semantics_pass: bool = True,
    spec_path: Path | None = None,
    spec_audit_path: Path | None = None,
    effective_data_dir: str | None = None,
) -> None:
    spec_audit_hash = "sha256:" + "4" * 16
    compiled_plan_hash = "sha256:" + "5" * 16
    if spec_audit_path is not None:
        spec_audit_hash = _hash_json_file(spec_audit_path)
    if spec_path is not None:
        spec = StrategySpec.from_yaml(spec_path)
        compiled_plan_hash = _canonical_json_hash(compile_plan(spec, effective_data_dir=effective_data_dir))
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "runtime_semantics_pass": runtime_semantics_pass,
                "spec_hash": spec_hash,
                "spec_audit_hash": spec_audit_hash,
                "compiled_plan_hash": compiled_plan_hash,
                "compiled_plan_path": "compile_preview/compiled_plan.json",
                "material_field_audits": [],
                "blocking_findings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _canonical_json_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def test_backtest_run_json_outputs_artifact_paths(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["run_id"]
    assert payload["run_dir"]
    assert payload["metrics"]["trade_count"] >= 0
    assert payload["metrics"] == json.loads(Path(payload["artifacts"]["metrics_json"]).read_text(encoding="utf-8"))
    assert payload["warnings"] == []
    assert payload["errors"] == []
    assert set(payload["artifacts"]) >= {
        "strategy_spec_yaml",
        "environment_json",
        "data_manifest_json",
        "execution_assumptions_json",
        "compiled_plan_json",
        "strategy_py",
        "equity_curve_csv",
        "trades_csv",
        "positions_csv",
        "orders_csv",
        "target_weights_csv",
        "benchmark_curve_csv",
        "metrics_json",
        "artifact_hashes_json",
        "run_log_jsonl",
    }
    assert payload["artifacts"]["target_weights_csv"].endswith("target_weights.csv")
    assert payload["artifacts"]["compiled_plan_json"].endswith("compiled_plan.json")
    assert payload["artifacts"]["strategy_py"].endswith("strategy.py")
    assert payload["artifacts"]["benchmark_curve_csv"].endswith("benchmark_curve.csv")
    assert payload["artifacts"]["artifact_hashes_json"].endswith("artifact_hashes.json")


def test_backtest_run_records_component_manifest_artifacts(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest = _write_component_manifest(tmp_path)
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--component-manifest",
            str(manifest),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    result_payload = json.loads(result.output)
    run_dir = Path(result_payload["run_dir"])
    assert result_payload["artifacts"]["component_manifest_json"].endswith("component_manifest.json")
    assert result_payload["artifacts"]["component_manifests_json"].endswith("component_manifests.json")
    assert result_payload["artifacts"]["component_bundle_hash_txt"].endswith("component_bundle_hash.txt")
    assert (run_dir / "component_bundle_hash.txt").read_text(encoding="utf-8").strip() == digest
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "component_manifest.json" in hashes
    assert "component_manifests.json" in hashes
    assert "component_bundle_hash.txt" in hashes


def test_backtest_run_json_reports_validation_failure(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    raw = spec_path.read_text(encoding="utf-8")
    spec_path.write_text(raw.replace("fee_rate: 0.001", "fee_rate: -0.001"), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["run_id"] == ""
    assert payload["run_dir"] == ""
    assert payload["errors"]
    assert payload["artifacts"] == {}


def _write_component_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "custom_components"
    source_dir = root / "oxq_components" / "indicators"
    tests_dir = root / "tests"
    source_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (root / "oxq_components" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    source = source_dir / "workspace_backtest_indicator.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "class WorkspaceBacktestIndicator:",
                "    name = 'WorkspaceBacktestIndicator'",
                "    def compute(self, mktdata: pd.DataFrame, value: float = 1.0) -> pd.Series:",
                "        return pd.Series(float(value), index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_file = tests_dir / "test_workspace_backtest_indicator.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    manifest = tmp_path / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "WorkspaceBacktestIndicator",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "oxq_components.indicators.workspace_backtest_indicator",
                        "class": "WorkspaceBacktestIndicator",
                        "protocol": "Indicator",
                        "tests": ["custom_components/tests/test_workspace_backtest_indicator.py"],
                        "source_path": "oxq_components/indicators/workspace_backtest_indicator.py",
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
    return manifest


def test_backtest_run_json_reports_missing_spec_file(tmp_path) -> None:
    runner = CliRunner()
    missing_spec = tmp_path / "missing.yaml"

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(missing_spec),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["run_id"] == ""
    assert payload["run_dir"] == ""
    assert payload["artifacts"] == {}
    assert payload["metrics"] == {}
    assert payload["errors"][0]["check"] == "spec_file_missing"
    assert str(missing_spec) in payload["errors"][0]["message"]


def test_backtest_run_json_reports_runtime_failure(tmp_path) -> None:
    spec_path, _data_dir = _write_spec_and_data(tmp_path)
    missing_data_dir = tmp_path / "missing_data"
    missing_data_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(missing_data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["run_id"] == ""
    assert payload["run_dir"] == ""
    assert payload["artifacts"] == {}
    assert payload["metrics"] == {}
    assert payload["errors"][0]["check"] == "runtime_error"


def test_backtest_run_json_rejects_missing_runtime_audit_for_formal_spec_audit(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    _write_spec_audit(audit_path, spec_hash)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(audit_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["run_id"] == ""
    assert payload["run_dir"] == ""
    assert payload["artifacts"] == {}
    assert payload["errors"][0]["check"] == "runtime_audit_missing"
    assert "runtime_audit.json is required" in payload["errors"][0]["message"]


def test_backtest_run_json_rejects_runtime_audit_without_spec_audit(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    runtime_audit_path = tmp_path / "runtime_audit.json"
    _write_runtime_audit(runtime_audit_path, spec_hash)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["errors"][0]["check"] == "spec_audit_missing"
    assert "spec_audit.json is required" in payload["errors"][0]["message"]


def test_backtest_run_json_auto_rejects_sibling_failed_runtime_audit(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    _write_spec_audit(tmp_path / "spec_audit.json", spec_hash)
    _write_runtime_audit(tmp_path / "runtime_audit.json", spec_hash, runtime_semantics_pass=False)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["errors"][0]["check"] == "runtime_audit_failed"
    assert "runtime_semantics_pass" in payload["errors"][0]["message"]


def test_backtest_run_json_accepts_passing_pre_run_audits(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    _write_spec_audit(audit_path, spec_hash)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir),
    )
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_backtest_run_json_hashes_runtime_audit_with_resolved_data_dir(monkeypatch, tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    _write_spec_audit(audit_path, spec_hash)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir.resolve()),
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            "strategy_spec.yaml",
            "--spec-audit",
            "spec_audit.json",
            "--runtime-audit",
            "runtime_audit.json",
            "--data-dir",
            "data",
            "--out",
            "runs",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = Path(json.loads(result.output)["run_dir"])
    compiled_plan = json.loads((run_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    assert compiled_plan["data"]["data_dir"] == str(data_dir.resolve())


def test_backtest_run_json_rejects_stale_runtime_audit_hashes(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    _write_spec_audit(audit_path, spec_hash)
    _write_runtime_audit(runtime_audit_path, spec_hash)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["errors"][0]["check"] == "runtime_audit_failed"
    assert "spec_audit_hash mismatch" in payload["errors"][0]["message"]


def test_backtest_run_json_uses_artifact_metrics_for_oos_window(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path, evaluation_window="oos")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    artifact_metrics = json.loads(Path(payload["artifacts"]["metrics_json"]).read_text(encoding="utf-8"))
    assert payload["metrics"] == artifact_metrics
    assert payload["metrics"]["metric_assumptions"]["evaluation_window"] == "oos"
