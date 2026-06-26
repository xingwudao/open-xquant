import hashlib
import json
import shutil
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
    component_bundle_hashes: list[str] | None = None,
) -> None:
    spec_audit_hash = "sha256:" + "4" * 16
    compiled_plan_hash = "sha256:" + "5" * 16
    if spec_audit_path is not None:
        spec_audit_hash = _hash_json_file(spec_audit_path)
    if spec_path is not None:
        spec = StrategySpec.from_yaml(spec_path)
        compiled_plan_hash = _canonical_json_hash(compile_plan(spec, effective_data_dir=effective_data_dir))
    payload = {
        "schema_version": 1,
        "status": "pass",
        "runtime_semantics_pass": runtime_semantics_pass,
        "spec_hash": spec_hash,
        "spec_audit_hash": spec_audit_hash,
        "compiled_plan_hash": compiled_plan_hash,
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": [],
        "blocking_findings": [],
    }
    if component_bundle_hashes is not None:
        payload["component_bundle_hashes"] = component_bundle_hashes
    path.write_text(
        json.dumps(payload, indent=2),
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
    from oxq.core.component_manifest import load_component_manifests_from_run

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
    assert any(
        run_dir.glob(
            "component_extensions/*/custom_components/oxq_components/indicators/workspace_backtest_indicator.py"
        )
    )
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "component_manifest.json" in hashes
    assert "component_manifests.json" in hashes
    assert "component_bundle_hash.txt" in hashes
    manifest.unlink()
    shutil.rmtree(tmp_path / "custom_components")
    loaded = load_component_manifests_from_run(run_dir)
    assert loaded[0]["bundle_hash"] == digest


def test_backtest_run_archives_external_manifest_test_files(tmp_path) -> None:
    from oxq.core.component_manifest import load_component_manifests_from_run

    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest = _write_component_manifest(tmp_path)
    external_tests = tmp_path / "tests"
    external_tests.mkdir()
    external_test = external_tests / "test_workspace_backtest_indicator.py"
    external_test.write_text("def test_external_placeholder():\n    assert True\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["components"][0]["tests"] = ["tests/test_workspace_backtest_indicator.py"]
    payload["components"][0]["test_hash"] = "sha256:" + hashlib.sha256(external_test.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
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
    run_dir = Path(json.loads(result.output)["run_dir"])
    archived_external_test = next(run_dir.glob("component_extensions/*/tests/test_workspace_backtest_indicator.py"))
    assert archived_external_test.read_text(encoding="utf-8") == external_test.read_text(encoding="utf-8")
    manifest.unlink()
    shutil.rmtree(tmp_path / "custom_components")
    shutil.rmtree(external_tests)
    loaded = load_component_manifests_from_run(run_dir)
    assert loaded[0]["bundle_hash"] == digest


def test_backtest_run_rejects_symlinked_external_manifest_test_file_before_import(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest = _write_component_manifest(tmp_path)
    real_tests = tmp_path / "real_tests"
    real_tests.mkdir()
    real_test = real_tests / "test_workspace_backtest_indicator.py"
    real_test.write_text("def test_real_placeholder():\n    assert True\n", encoding="utf-8")
    linked_tests = tmp_path / "tests"
    linked_tests.mkdir()
    linked_test = linked_tests / "test_workspace_backtest_indicator.py"
    linked_test.symlink_to(real_test)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["components"][0]["tests"] = ["tests/test_workspace_backtest_indicator.py"]
    payload["components"][0]["test_hash"] = "sha256:" + hashlib.sha256(real_test.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
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

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "component_archive_failed"
    assert "symlinked external test files" in payload["errors"][0]["message"]


def test_backtest_run_archives_multiple_component_manifests(tmp_path) -> None:
    from oxq.core.component_manifest import load_component_manifests_from_run

    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest_a = _write_component_manifest(tmp_path)
    manifest_b = _write_named_component_manifest(
        tmp_path,
        root_name="more_components",
        manifest_name="workspace_manifest.json",
        class_name="SecondWorkspaceBacktestIndicator",
    )
    digests = []
    for manifest in (manifest_a, manifest_b):
        digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
            "component_bundle_hash"
        ]
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["bundle_hash"] = digest
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        digests.append(digest)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--component-manifest",
            str(manifest_a),
            "--component-manifest",
            str(manifest_b),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = Path(json.loads(result.output)["run_dir"])
    summary = json.loads((run_dir / "component_manifests.json").read_text(encoding="utf-8"))
    assert len(summary) == 2
    archived_paths = [item["archived_manifest_path"] for item in summary]
    assert any(path.endswith("/component_manifest.json") for path in archived_paths)
    assert any(path.endswith("/workspace_manifest.json") for path in archived_paths)
    assert all((run_dir / path).exists() for path in archived_paths)

    manifest_a.unlink()
    manifest_b.unlink()
    shutil.rmtree(tmp_path / "custom_components")
    shutil.rmtree(tmp_path / "more_components")

    loaded = load_component_manifests_from_run(run_dir)
    assert [item["bundle_hash"] for item in loaded] == digests


def test_backtest_run_rejects_archiving_extension_into_itself(tmp_path) -> None:
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
            str(tmp_path / "custom_components" / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "archive would be nested inside the source extension" in result.output
    assert not (tmp_path / "custom_components" / "runs").exists()


def test_backtest_run_rejects_component_extension_symlinks_before_running(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest = _write_component_manifest(tmp_path)
    external = tmp_path / "external_files"
    external.mkdir()
    (external / "secret.txt").write_text("not part of the component bundle\n", encoding="utf-8")
    (tmp_path / "custom_components" / "linked_external").symlink_to(external, target_is_directory=True)
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    out_dir = tmp_path / "runs"

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
            str(out_dir),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "refuses symlinks" in result.output
    assert not out_dir.exists()


def test_backtest_run_json_requires_runtime_audit_component_bundle_hashes(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest = _write_component_manifest(tmp_path)
    marker = tmp_path / "component_imported.txt"
    source = tmp_path / "custom_components" / "oxq_components" / "indicators" / "workspace_backtest_indicator.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "import pandas as pd",
            "\n".join(
                [
                    "import pandas as pd",
                    "from pathlib import Path",
                    f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')",
                ]
            ),
        ),
        encoding="utf-8",
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["components"][0]["source_hash"] = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    bundle_hash = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["bundle_hash"] = bundle_hash
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
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
            "--component-manifest",
            str(manifest),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "runtime_audit_failed"
    assert "component_bundle_hashes" in payload["errors"][0]["message"]
    assert not marker.exists()

    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir),
        component_bundle_hashes=[bundle_hash],
    )
    ok = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--component-manifest",
            str(manifest),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_ok"),
            "--json",
        ],
    )

    assert ok.exit_code == 0, ok.output
    assert marker.read_text(encoding="utf-8") == "imported"


def test_run_component_manifest_loader_rejects_recorded_bundle_hash_mismatch(tmp_path) -> None:
    from oxq.core.component_manifest import load_component_manifests_from_run

    manifest = _write_component_manifest(tmp_path)
    bundle_hash = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["bundle_hash"] = bundle_hash
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "component_manifests.json").write_text(
        json.dumps([{"manifest_path": str(manifest), "bundle_hash": "sha256:" + "0" * 64}]),
        encoding="utf-8",
    )

    try:
        load_component_manifests_from_run(run_dir)
    except ValueError as exc:
        assert "recorded component bundle hash mismatch" in str(exc)
    else:
        raise AssertionError("expected recorded component bundle hash mismatch")


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
    return _write_named_component_manifest(
        tmp_path,
        root_name="custom_components",
        manifest_name="component_manifest.json",
        class_name="WorkspaceBacktestIndicator",
    )


def _write_named_component_manifest(tmp_path: Path, *, root_name: str, manifest_name: str, class_name: str) -> Path:
    root = tmp_path / root_name
    source_dir = root / "oxq_components" / "indicators"
    tests_dir = root / "tests"
    source_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (root / "oxq_components" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    module_file = "".join(["_" + ch.lower() if ch.isupper() else ch for ch in class_name]).lstrip("_")
    source = source_dir / f"{module_file}.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                f"class {class_name}:",
                f"    name = '{class_name}'",
                "    def compute(self, mktdata: pd.DataFrame, value: float = 1.0) -> pd.Series:",
                "        return pd.Series(float(value), index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_file = tests_dir / f"test_{module_file}.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    manifest = tmp_path / manifest_name
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": root_name,
                "extension_root": root_name,
                "bundle_hash": "",
                "components": [
                    {
                        "name": class_name,
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": f"oxq_components.indicators.{module_file}",
                        "class": class_name,
                        "protocol": "Indicator",
                        "tests": [f"{root_name}/tests/test_{module_file}.py"],
                        "source_path": f"oxq_components/indicators/{module_file}.py",
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
