import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from click import ClickException
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.spec.compiler import _append_run_digest, _hash_file, _hash_json_file, compile_plan
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


def _refresh_run_digest_for_test(run_dir: Path) -> None:
    _append_run_digest(run_dir, _hash_json_file(run_dir / "artifact_hashes.json"))


def _write_spec_audit(
    path: Path,
    spec_hash: str,
    catalog_hash: str | None = None,
    *,
    spec_path: Path | None = None,
) -> None:
    if catalog_hash is None:
        catalog_path = path.with_name("component_catalog.json")
        catalog_hash = _write_component_catalog(catalog_path)
    if spec_path is None:
        sibling_spec_path = path.with_name("strategy_spec.yaml")
        spec_path = sibling_spec_path if sibling_spec_path.exists() else None
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "pass",
                "spec_provenance_pass": True,
                "spec_hash": spec_hash,
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": catalog_hash,
                "recipe_matches": [],
                "field_audits": _confirmed_field_audits(spec_path) if spec_path is not None else [],
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


def _confirmed_field_audits(spec_path: Path) -> list[dict]:
    spec = StrategySpec.from_yaml(spec_path)
    return [
        {
            "field_path": field_path,
            "spec_value": value,
            "status": "confirmed",
            "evidence": [f"user confirmed {field_path} = {json.dumps(value, sort_keys=True, default=str)}"],
            "blocking": False,
        }
        for field_path, value in _flatten_effective_fields(spec.to_effective_dict())
    ]


def _flatten_effective_fields(value: object, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        if not value and prefix:
            return [(prefix, {})]
        fields: list[tuple[str, object]] = []
        for key in sorted(value):
            child_path = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(_flatten_effective_fields(value[key], child_path))
        return fields
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return [(prefix, value)]
        fields = []
        for index, item in enumerate(value):
            fields.extend(_flatten_effective_fields(item, f"{prefix}[{index}]"))
        return fields
    return [(prefix, value)]


def _write_component_catalog(path: Path, component_manifests: list[dict] | None = None) -> str:
    from oxq.core.component_catalog import build_component_catalog, component_catalog_json

    catalog = build_component_catalog(component_manifests or [])
    path.write_text(component_catalog_json(catalog), encoding="utf-8")
    return str(catalog["catalog_hash"])


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
            "--allow-unaudited",
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


def test_backtest_run_json_requires_audits_by_default(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)

    result = CliRunner().invoke(
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
    assert payload["errors"][0]["check"] == "spec_audit_missing"
    assert "formal backtest requires audited gate artifacts" in payload["errors"][0]["message"]


def test_backtest_run_allow_unaudited_ignores_stale_sibling_audits(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    (tmp_path / "spec_audit.json").write_text(
        json.dumps({"schema_version": 3, "status": "pass", "spec_hash": "sha256:stale"}),
        encoding="utf-8",
    )
    (tmp_path / "runtime_audit.json").write_text("{not-json", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--allow-unaudited",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"


def test_backtest_run_records_component_manifest_artifacts(tmp_path) -> None:
    from oxq.core.component_manifest import load_component_manifest, load_component_manifests_from_run, scoped_component_registries
    from oxq.core.registry import list_indicators

    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest = _write_component_manifest(tmp_path)
    spec = StrategySpec.from_yaml(spec_path)
    spec.signal.indicators = {
        "roc_1": IndicatorDef(type="WorkspaceBacktestIndicator", params={"value": 1.0})
    }
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
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
            "--allow-unaudited",
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
    with scoped_component_registries():
        legacy = load_component_manifest(run_dir / "component_manifest.json")
    assert legacy["bundle_hash"] == digest
    with scoped_component_registries():
        loaded = load_component_manifests_from_run(run_dir)
    assert loaded[0]["bundle_hash"] == digest
    module_spec = importlib.util.spec_from_file_location("generated_strategy_artifact", run_dir / "strategy.py")
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    strategy = module.build_strategy()
    assert strategy.name == "json_backtest"
    assert "WorkspaceBacktestIndicator" not in list_indicators()


def test_run_component_manifest_loader_prefers_archived_legacy_manifest(tmp_path) -> None:
    from oxq.core.component_manifest import load_component_manifests_from_run, scoped_component_registries

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
            "--allow-unaudited",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = Path(json.loads(result.output)["run_dir"])
    summary = json.loads((run_dir / "component_manifests.json").read_text(encoding="utf-8"))
    assert (run_dir / "custom_components").is_dir()
    summary[0].pop("archived_manifest_path", None)
    summary[0].pop("archived_extension_root", None)
    (run_dir / "component_manifests.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    mutable_payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutable_payload["bundle_hash"] = "sha256:mutable-workspace"
    manifest.write_text(json.dumps(mutable_payload, indent=2, sort_keys=True), encoding="utf-8")

    with scoped_component_registries():
        loaded = load_component_manifests_from_run(run_dir)

    assert loaded[0]["bundle_hash"] == digest


def test_backtest_run_archives_external_manifest_test_files(tmp_path) -> None:
    from oxq.core.component_manifest import load_component_manifests_from_run, scoped_component_registries

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
            "--allow-unaudited",
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
    with scoped_component_registries():
        loaded = load_component_manifests_from_run(run_dir)
    assert loaded[0]["bundle_hash"] == digest


def test_backtest_run_rejects_symlinked_external_manifest_test_file_before_import(tmp_path) -> None:
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

    result = CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"])

    assert result.exit_code == 1
    assert "must not traverse symlinks" in str(result.exception)


def test_component_extension_external_tests_reject_symlinked_parent(tmp_path) -> None:
    from oxq.cli.main import _component_extension_external_test_files

    manifest = _write_component_manifest(tmp_path)
    real_tests = tmp_path / "real_tests"
    real_tests.mkdir()
    (real_tests / "test_workspace_backtest_indicator.py").write_text(
        "def test_real_placeholder():\n    assert True\n",
        encoding="utf-8",
    )
    linked_tests = tmp_path / "tests"
    linked_tests.symlink_to(real_tests, target_is_directory=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["components"][0]["tests"] = ["tests/test_workspace_backtest_indicator.py"]

    with pytest.raises(ClickException, match="symlinked external test files"):
        _component_extension_external_test_files(payload, manifest, tmp_path / "custom_components")


def test_backtest_run_archives_multiple_component_manifests(tmp_path) -> None:
    from oxq.core.component_manifest import load_component_manifests_from_run, scoped_component_registries

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
            "--allow-unaudited",
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

    with scoped_component_registries():
        loaded = load_component_manifests_from_run(run_dir)
    assert [item["bundle_hash"] for item in loaded] == digests


def test_run_component_manifest_loader_rejects_missing_recorded_archive(tmp_path) -> None:
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
            "--allow-unaudited",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = Path(json.loads(result.output)["run_dir"])
    summary = json.loads((run_dir / "component_manifests.json").read_text(encoding="utf-8"))
    archived_manifest = run_dir / summary[0]["archived_manifest_path"]
    archived_manifest.unlink()

    with pytest.raises(ValueError, match="archived component manifest not found"):
        load_component_manifests_from_run(run_dir)


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
            "--allow-unaudited",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "archive would be nested inside the source extension" in result.output
    assert not (tmp_path / "custom_components" / "runs").exists()


def test_backtest_run_rejects_component_extension_symlinks_before_running(tmp_path) -> None:
    manifest = _write_component_manifest(tmp_path)
    external = tmp_path / "external_files"
    external.mkdir()
    (external / "secret.txt").write_text("not part of the component bundle\n", encoding="utf-8")
    (tmp_path / "custom_components" / "linked_external").symlink_to(external, target_is_directory=True)
    out_dir = tmp_path / "runs"

    result = CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"])

    assert result.exit_code == 1
    assert "must not be a symlink" in str(result.exception)
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
    manifest_for_catalog = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_for_catalog["_manifest_path"] = str(manifest.resolve())
    catalog_hash = _write_component_catalog(tmp_path / "component_catalog.json", [manifest_for_catalog])
    _write_spec_audit(audit_path, spec_hash, catalog_hash=catalog_hash)
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
            "--allow-unaudited",
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


def test_backtest_run_rejects_component_catalog_bundle_mismatch_before_import(tmp_path) -> None:
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
        component_bundle_hashes=[bundle_hash],
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
    assert payload["errors"][0]["check"] == "component_catalog_failed"
    assert "component bundle hash mismatch" in payload["errors"][0]["message"]
    assert not marker.exists()


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
            "--allow-unaudited",
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


def _write_hashed_component_manifest(tmp_path: Path) -> tuple[Path, str]:
    manifest = _write_component_manifest(tmp_path)
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest, digest


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
            "--allow-unaudited",
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
            "--allow-unaudited",
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


def test_backtest_run_json_rejects_spec_audit_missing_effective_field_confirmations(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    _write_spec_audit(audit_path, spec_hash, spec_path=None)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["field_audits"] = []
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "spec_audit_failed"
    assert "missing confirmed audit row for effective spec field" in response["errors"][0]["message"]


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


def test_backtest_run_allow_unaudited_rejects_explicit_audits_without_catalog(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec = StrategySpec.from_yaml(spec_path)
    spec_hash = _canonical_json_hash(spec.to_dict())
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    _write_spec_audit(audit_path, spec_hash)
    (tmp_path / "component_catalog.json").unlink()
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
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--spec-audit",
            str(audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--allow-unaudited",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "component_catalog_missing"
    assert not (tmp_path / "runs").exists()


def test_backtest_run_json_wraps_bad_component_catalog(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec = StrategySpec.from_yaml(spec_path)
    spec_hash = spec.compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    catalog_path = tmp_path / "component_catalog.json"
    _write_spec_audit(audit_path, spec_hash)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir),
    )
    catalog_path.write_text("{not-json", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--spec-audit",
            str(audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--component-catalog",
            str(catalog_path),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["errors"][0]["check"] == "component_catalog_failed"
    assert "not valid JSON" in payload["errors"][0]["message"]


def test_backtest_run_preserves_explicit_default_fill_price_mode_hash(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw["execution"] = {
        **raw.get("execution", {}),
        "order_timing": "next_session_open",
        "price_bar": "next_session",
        "price_type": "open",
        "fill_price_mode": "next_open",
    }
    spec_path.write_text(yaml.dump(raw, sort_keys=False), encoding="utf-8")
    spec = StrategySpec.from_yaml(spec_path)
    spec_hash = spec.compute_hash()
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
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--spec-audit",
            str(audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = Path(json.loads(result.output)["run_dir"])
    assert (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip() == spec_hash
    run_spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    assert run_spec["execution"]["fill_price_mode"] == "next_open"


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
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    run_dir = Path(payload["run_dir"])
    assert (run_dir / "spec_audit.json").exists()
    assert (run_dir / "runtime_audit.json").exists()
    artifact_hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "spec_audit.json" in artifact_hashes
    assert "runtime_audit.json" in artifact_hashes


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
            "--allow-unaudited",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    artifact_metrics = json.loads(Path(payload["artifacts"]["metrics_json"]).read_text(encoding="utf-8"))
    assert payload["metrics"] == artifact_metrics
    assert payload["metrics"]["metric_assumptions"]["evaluation_window"] == "oos"


def test_backtest_compare_runs_blocks_different_cost_assumptions(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output

    spec = StrategySpec.from_yaml(spec_path)
    spec.cost.slippage_rate = 0.005
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert second.exit_code == 0, second.output

    compare = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            json.loads(second.output)["run_dir"],
            "--json",
        ],
    )

    assert compare.exit_code == 1
    payload = json.loads(compare.output)
    assert payload["comparable"] is False
    assert any(item["field"] == "cost" for item in payload["differences"])


def test_backtest_compare_runs_rejects_incomplete_run_dirs(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    result = CliRunner().invoke(main, ["backtest", "compare-runs", str(left), str(right), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["comparable"] is False
    assert payload["errors"][0]["check"] == "run_artifacts_missing"


def test_backtest_compare_runs_rejects_corrupt_required_json_artifacts(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    (second_run / "compiled_plan.json").write_text("{not-json", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["comparable"] is False
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"


def test_backtest_compare_runs_rejects_stale_artifact_hashes(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    compiled_plan_path = second_run / "compiled_plan.json"
    compiled_plan = json.loads(compiled_plan_path.read_text(encoding="utf-8"))
    compiled_plan["runtime_rules"] = [{"type": "changed-after-run"}]
    compiled_plan_path.write_text(json.dumps(compiled_plan, indent=2), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["comparable"] is False
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "artifact hash mismatch for compiled_plan.json" in payload["errors"][0]["message"]


def test_backtest_compare_runs_rejects_stale_provenance_hashes(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    spec_audit_path = second_run / "spec_audit.json"
    spec_audit_path.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    artifact_hashes_path = second_run / "artifact_hashes.json"
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    artifact_hashes["spec_audit.json"] = _hash_json_file(spec_audit_path)
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2), encoding="utf-8")
    _refresh_run_digest_for_test(second_run)
    spec_audit_path.write_text(json.dumps({"status": "fail"}), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "artifact hash mismatch for spec_audit.json" in payload["errors"][0]["message"]


def test_backtest_compare_runs_rejects_recomputed_artifact_hashes_with_stale_run_digest(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    compiled_plan_path = second_run / "compiled_plan.json"
    compiled_plan = json.loads(compiled_plan_path.read_text(encoding="utf-8"))
    compiled_plan["runtime_rules"] = [{"type": "recomputed-after-run"}]
    compiled_plan_path.write_text(json.dumps(compiled_plan, indent=2), encoding="utf-8")
    artifact_hashes_path = second_run / "artifact_hashes.json"
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    artifact_hashes["compiled_plan.json"] = _hash_json_file(compiled_plan_path)
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "run digest mismatch for artifact_hashes.json" in payload["errors"][0]["message"]


def test_backtest_compare_runs_rejects_present_provenance_without_artifact_hash(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    (second_run / "spec_audit.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "missing required hash for comparison artifact: spec_audit.json" in payload["errors"][0]["message"]


def test_backtest_compare_runs_rejects_stale_spec_hash_text(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    (second_run / "spec_hash.txt").write_text("sha256:stale-spec-hash\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "spec_hash.txt mismatch" in payload["errors"][0]["message"]


def test_backtest_compare_runs_rejects_corrupt_component_manifest_artifacts(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    manifest_path = second_run / "component_manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")
    artifact_hashes_path = second_run / "artifact_hashes.json"
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    artifact_hashes["component_manifest.json"] = _hash_file(manifest_path)
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2), encoding="utf-8")
    _refresh_run_digest_for_test(second_run)

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["comparable"] is False
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "component_manifest.json is not valid JSON" in payload["errors"][0]["message"]


def test_backtest_compare_runs_recomputes_archived_component_bundle_hashes(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest, _digest = _write_hashed_component_manifest(tmp_path)
    spec = StrategySpec.from_yaml(spec_path)
    spec.signal.indicators = {
        "roc_1": IndicatorDef(type="WorkspaceBacktestIndicator", params={"value": 1.0})
    }
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    runner = CliRunner()
    first = runner.invoke(
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
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
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
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    archived_source = next(
        second_run.glob(
            "component_extensions/*/custom_components/oxq_components/indicators/workspace_backtest_indicator.py"
        )
    )
    archived_source.write_text(archived_source.read_text(encoding="utf-8") + "\n# changed after run\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "component bundle" in payload["errors"][0]["message"]


def test_backtest_compare_runs_blocks_runtime_hash_differences(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    second_run = Path(json.loads(second.output)["run_dir"])
    compiled_plan_path = second_run / "compiled_plan.json"
    compiled_plan = json.loads(compiled_plan_path.read_text(encoding="utf-8"))
    compiled_plan["runtime_rules"] = [{"type": "valid-runtime-change"}]
    compiled_plan_path.write_text(json.dumps(compiled_plan, indent=2), encoding="utf-8")
    artifact_hashes_path = second_run / "artifact_hashes.json"
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    artifact_hashes["compiled_plan.json"] = _hash_json_file(compiled_plan_path)
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2), encoding="utf-8")
    _refresh_run_digest_for_test(second_run)

    compare = runner.invoke(
        main,
        [
            "backtest",
            "compare-runs",
            json.loads(first.output)["run_dir"],
            str(second_run),
            "--json",
        ],
    )

    assert compare.exit_code == 1
    payload = json.loads(compare.output)
    assert payload["comparable"] is False
    assert any(item["field"] == "compiled_plan_hash" for item in payload["differences"])


def test_backtest_compare_runs_blocks_audit_hash_differences(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_run = Path(json.loads(first.output)["run_dir"])
    second_run = Path(json.loads(second.output)["run_dir"])
    first_spec_audit_path = first_run / "spec_audit.json"
    first_runtime_audit_path = first_run / "runtime_audit.json"
    second_spec_audit_path = second_run / "spec_audit.json"
    second_runtime_audit_path = second_run / "runtime_audit.json"
    first_spec_audit_path.write_text(json.dumps({"status": "pass", "run": "a"}), encoding="utf-8")
    first_runtime_audit_path.write_text(json.dumps({"status": "pass", "run": "a"}), encoding="utf-8")
    second_spec_audit_path.write_text(json.dumps({"status": "pass", "run": "b"}), encoding="utf-8")
    second_runtime_audit_path.write_text(json.dumps({"status": "pass", "run": "b"}), encoding="utf-8")
    first_hashes_path = first_run / "artifact_hashes.json"
    second_hashes_path = second_run / "artifact_hashes.json"
    first_hashes = json.loads(first_hashes_path.read_text(encoding="utf-8"))
    second_hashes = json.loads(second_hashes_path.read_text(encoding="utf-8"))
    first_hashes["spec_audit.json"] = _hash_json_file(first_spec_audit_path)
    first_hashes["runtime_audit.json"] = _hash_json_file(first_runtime_audit_path)
    second_hashes["spec_audit.json"] = _hash_json_file(second_spec_audit_path)
    second_hashes["runtime_audit.json"] = _hash_json_file(second_runtime_audit_path)
    first_hashes_path.write_text(json.dumps(first_hashes, indent=2), encoding="utf-8")
    second_hashes_path.write_text(json.dumps(second_hashes, indent=2), encoding="utf-8")
    _refresh_run_digest_for_test(first_run)
    _refresh_run_digest_for_test(second_run)

    compare = runner.invoke(main, ["backtest", "compare-runs", str(first_run), str(second_run), "--json"])

    assert compare.exit_code == 1
    payload = json.loads(compare.output)
    assert payload["comparable"] is False
    fields = {item["field"] for item in payload["differences"]}
    assert "spec_audit_hash" in fields
    assert "runtime_audit_hash" in fields


def test_backtest_compare_runs_blocks_component_catalog_hash_differences(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_a"),
            "--allow-unaudited",
            "--json",
        ],
    )
    second = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs_b"),
            "--allow-unaudited",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_run = Path(json.loads(first.output)["run_dir"])
    second_run = Path(json.loads(second.output)["run_dir"])
    first_catalog_hash = first_run / "component_catalog_hash.txt"
    second_catalog_hash = second_run / "component_catalog_hash.txt"
    first_catalog_hash.write_text("sha256:catalog-a\n", encoding="utf-8")
    second_catalog_hash.write_text("sha256:catalog-b\n", encoding="utf-8")
    first_hashes_path = first_run / "artifact_hashes.json"
    second_hashes_path = second_run / "artifact_hashes.json"
    first_hashes = json.loads(first_hashes_path.read_text(encoding="utf-8"))
    second_hashes = json.loads(second_hashes_path.read_text(encoding="utf-8"))
    first_hashes["component_catalog_hash.txt"] = _hash_file(first_catalog_hash)
    second_hashes["component_catalog_hash.txt"] = _hash_file(second_catalog_hash)
    first_hashes_path.write_text(json.dumps(first_hashes, indent=2), encoding="utf-8")
    second_hashes_path.write_text(json.dumps(second_hashes, indent=2), encoding="utf-8")
    _refresh_run_digest_for_test(first_run)
    _refresh_run_digest_for_test(second_run)

    compare = runner.invoke(main, ["backtest", "compare-runs", str(first_run), str(second_run), "--json"])

    assert compare.exit_code == 1
    payload = json.loads(compare.output)
    assert payload["comparable"] is False
    assert any(item["field"] == "component_catalog_hash" for item in payload["differences"])
