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

from oxq.cli.main import _replace_run_digest_entry, main
from oxq.cli.research import VERSION_PHASE_DIRS
from oxq.run_digests import publish_run_artifacts
from oxq.spec.compiler import _hash_file, _hash_json_file, compile_plan
from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec


def _spec_audit_context() -> dict[str, object]:
    return {
        "strategy_idea_brief": "versions/v001/01_brainstorm/strategy_idea_brief.json",
        "strategy_idea_audit": "versions/v001/02_idea_audit/strategy_idea_audit.json",
        "strategy_idea_brief_hash": "sha256:" + "5" * 16,
        "strategy_idea_audit_hash": "sha256:" + "6" * 16,
        "unsupported_mappings": [],
    }


def _write_confirmation_event(
    path: Path,
    *,
    event_reference: str,
    artifact_path: str,
    artifact_hash: str,
    spec_audit_hash: str = "sha256:" + "8" * 16,
) -> dict[str, object]:
    event = {
        "timestamp": "2026-07-07T08:00:00Z",
        "phase": "spec_confirmation",
        "field_scope": "full_spec_table",
        "decision": "confirmed",
        "event_id": "spec-confirmation-1",
        "user_text": "确认",
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "spec_audit_path": "spec_audit.json",
        "spec_audit_hash": spec_audit_hash,
    }
    line = json.dumps(event, sort_keys=True, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")
    return {
        "path": event_reference,
        "event_id": event["event_id"],
        "decision": event["decision"],
        "line_number": 1,
        "event_hash": f"sha256:{hashlib.sha256(line.encode('utf-8')).hexdigest()}",
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "spec_audit_path": "spec_audit.json",
        "spec_audit_hash": spec_audit_hash,
    }


def _pre_confirmation_spec_audit_hash(payload: dict) -> str:
    candidate = json.loads(json.dumps(payload, default=str))
    candidate.pop("confirmation_event", None)
    candidate["status"] = "block"
    candidate["user_confirmation_status"] = "pending"
    canonical = json.dumps(candidate, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


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
    _replace_run_digest_entry(run_dir, _hash_json_file(run_dir / "artifact_hashes.json"))


def _publish_comparison_provenance_for_test(
    run_dir: Path,
    *,
    spec_audit: dict[str, object],
    runtime_audit: dict[str, object],
    component_catalog_hash: str,
) -> None:
    publish_run_artifacts(
        run_dir,
        {
            "spec_audit.json": json.dumps(spec_audit).encode(),
            "runtime_audit.json": json.dumps(runtime_audit).encode(),
            "conversation_hash.txt": ("sha256:" + "c" * 64 + "\n").encode(),
            "component_catalog_hash.txt": (component_catalog_hash + "\n").encode(),
            "recipe_catalog_hash.txt": ("sha256:" + "e" * 64 + "\n").encode(),
        },
        canonical_json={"spec_audit.json", "runtime_audit.json"},
    )


def _write_spec_audit(
    path: Path,
    spec_hash: str,
    catalog_hash: str | None = None,
    *,
    spec_path: Path | None = None,
    confirmation_event_path: Path | None = None,
    confirmation_event_reference: str = "conversations/confirmations.jsonl",
) -> None:
    if catalog_hash is None:
        catalog_path = path.with_name("component_catalog.json")
        catalog_hash = _write_component_catalog(catalog_path)
    if spec_path is None:
        sibling_spec_path = path.with_name("strategy_spec.yaml")
        spec_path = sibling_spec_path if sibling_spec_path.exists() else None
    confirmation_table = path.with_name("spec_confirmation_table.md")
    if spec_path is not None:
        table_text = _spec_confirmation_table_text(spec_path)
    else:
        table_text = "| Field | Confirmed Value |\n| --- | --- |\n| spec_hash | confirmed |\n"
    confirmation_table.write_text(table_text, encoding="utf-8")
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": str(confirmation_table),
            "hash": _hash_file(confirmation_table),
            "hash_type": "sha256",
        },
        "spec_provenance_pass": True,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog_hash,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": _confirmed_field_audits(spec_path) if spec_path is not None else [],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    audit["confirmation_event"] = _write_confirmation_event(
        confirmation_event_path or path.parent / "conversations" / "confirmations.jsonl",
        event_reference=confirmation_event_reference,
        artifact_path=str(confirmation_table),
        artifact_hash=_hash_file(confirmation_table),
        spec_audit_hash=_pre_confirmation_spec_audit_hash(audit),
    )
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def _spec_confirmation_table_text(spec_path: Path) -> str:
    spec = StrategySpec.from_yaml(spec_path)
    rows = [
        "| Section | Field path | Spec value | Source | Audit status | Impact |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for field_path, value in _flatten_effective_fields(spec.to_effective_dict()):
        section = field_path.split(".", 1)[0]
        rows.append(
            "| "
            + " | ".join(
                [
                    section,
                    field_path,
                    json.dumps(value, sort_keys=True, default=str),
                    "User confirmed full SPEC table",
                    "confirmed",
                    "material",
                ]
            )
            + " |"
        )
    return "\n".join(rows) + "\n"


def _confirmed_field_audits(spec_path: Path) -> list[dict]:
    spec = StrategySpec.from_yaml(spec_path)
    return [
        {
            "field_path": field_path,
            "spec_value": value,
            "status": "confirmed",
            "material_category": _material_category_for_field_path(field_path),
            "evidence": [f"user confirmed {field_path} = {json.dumps(value, sort_keys=True, default=str)}"],
            "blocking": False,
        }
        for field_path, value in _flatten_effective_fields(spec.to_effective_dict())
    ]


def _material_category_for_field_path(field_path: str) -> str:
    if field_path.startswith(("signal.", "rules.")):
        return "strategy_logic"
    if field_path.startswith("portfolio."):
        return "portfolio_construction"
    if field_path.startswith("execution."):
        return "execution_assumption"
    if field_path.startswith("cost."):
        return "cost_assumption"
    if field_path.startswith(("data.", "universe.", "market.", "benchmark.")):
        return "data_assumption"
    if field_path.startswith("validation."):
        return "validation_assumption"
    if field_path.startswith(("metrics.", "decision_policy.")):
        return "metric_assumption"
    if field_path.startswith(("robustness.", "risk.")):
        return "risk_assumption"
    if field_path == "required_oxq_version":
        return "system_provenance"
    return "backtest_assumption"


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
    strategy_source_path: Path | None = None,
) -> None:
    spec_audit_hash = "sha256:" + "4" * 16
    compiled_plan_hash = "sha256:" + "5" * 16
    material_field_audits: list[dict[str, object]] = []
    if spec_audit_path is not None:
        spec_audit_hash = _hash_json_file(spec_audit_path)
    if spec_path is not None:
        spec = StrategySpec.from_yaml(spec_path)
        compiled_plan = compile_plan(spec, effective_data_dir=effective_data_dir)
        compiled_plan_hash = _canonical_json_hash(compiled_plan)
        effective_spec = spec.to_effective_dict()
        material_field_audits = [
            {
                "field_path": field_path,
                "spec_value": effective_spec[field_path],
                "runtime_path": runtime_path,
                "runtime_value": compiled_plan[runtime_path],
                "status": "preserved",
                "evidence": ["test fixture"],
                "blocking": False,
            }
            for field_path, runtime_path in (
                ("required_oxq_version", "open_xquant_version"),
                ("market", "market"),
                ("universe", "universe"),
                ("data", "data"),
                ("signal", "signals"),
                ("portfolio", "portfolio"),
                ("execution", "execution"),
                ("cost", "cost"),
                ("benchmark", "benchmark"),
                ("validation", "validation"),
                ("metrics", "metrics"),
            )
        ]
    strategy_source_path = strategy_source_path or path.with_name("strategy.py")
    strategy_source_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_source_path.write_text(
        "# Generated strategy source used by the presentation gate.\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": runtime_semantics_pass,
        # Kept in tests to prove a worker assertion cannot replace coordinator evidence.
        "strategy_source_printed": True,
        "strategy_source_path": str(strategy_source_path),
        "strategy_source_hash": "sha256:" + hashlib.sha256(strategy_source_path.read_bytes()).hexdigest(),
        "spec_hash": spec_hash,
        "spec_audit_hash": spec_audit_hash,
        "compiled_plan_hash": compiled_plan_hash,
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": material_field_audits,
        "blocking_findings": [],
    }
    payload["component_bundle_hashes"] = component_bundle_hashes or []
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def test_backtest_run_rejects_required_version_mismatch_before_component_import_or_output_mutation(
    monkeypatch,
    tmp_path,
) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec = StrategySpec.from_yaml(spec_path)
    spec.required_oxq_version = "999.0.0"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    manifest = _write_component_manifest(tmp_path)
    bundle_hash = json.loads(
        CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output
    )["component_bundle_hash"]
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["bundle_hash"] = bundle_hash
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    spec_audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    catalog_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    catalog_manifest["_manifest_path"] = str(manifest.resolve())
    catalog_hash = _write_component_catalog(tmp_path / "component_catalog.json", [catalog_manifest])
    _write_spec_audit(spec_audit_path, spec_hash, catalog_hash=catalog_hash, spec_path=spec_path)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        effective_data_dir=str(data_dir),
        component_bundle_hashes=[bundle_hash],
    )
    out_dir = tmp_path / "runs"
    _write_backtest_authorization(
        tmp_path / "backtest_authorization.json",
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        runtime_audit_path=runtime_audit_path,
        component_manifest_paths=[manifest],
        data_dir=data_dir,
        run_out=out_dir,
    )
    imported = False

    def import_sentinel(_manifest_paths):
        nonlocal imported
        imported = True
        raise AssertionError("component import ran before required-version gate")

    monkeypatch.setattr("oxq.cli.main._load_component_manifests", import_sentinel)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(spec_audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
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
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "runtime_audit_failed"
    assert "required_oxq_version" in response["errors"][0]["message"]
    assert "open_xquant_version" in response["errors"][0]["message"]
    assert imported is False
    assert not out_dir.exists()


def test_backtest_run_rejects_material_runtime_mismatch_before_component_import_or_output_mutation(
    monkeypatch,
    tmp_path,
) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest, bundle_hash = _write_hashed_component_manifest(tmp_path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["_manifest_path"] = str(manifest.resolve())
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    spec_audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    catalog_hash = _write_component_catalog(
        tmp_path / "component_catalog.json",
        [manifest_payload],
    )
    _write_spec_audit(
        spec_audit_path,
        spec_hash,
        catalog_hash=catalog_hash,
        spec_path=spec_path,
    )
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        effective_data_dir=str(data_dir),
        component_bundle_hashes=[bundle_hash],
    )
    runtime_payload = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    execution_row = next(
        row
        for row in runtime_payload["material_field_audits"]
        if row["field_path"] == "execution"
    )
    execution_row["runtime_value"]["initial_cash"] += 1
    runtime_audit_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    out_dir = tmp_path / "runs"
    _write_backtest_authorization(
        tmp_path / "backtest_authorization.json",
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        runtime_audit_path=runtime_audit_path,
        component_manifest_paths=[manifest],
        data_dir=data_dir,
        run_out=out_dir,
    )
    imported = False

    def import_sentinel(_manifest_paths):
        nonlocal imported
        imported = True
        raise AssertionError("component import ran before full runtime audit gate")

    monkeypatch.setattr("oxq.cli.main._load_component_manifests", import_sentinel)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(spec_audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
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
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "runtime_audit_failed"
    assert "material_field_audits" in response["errors"][0]["message"]
    assert imported is False
    assert not out_dir.exists()


def _write_backtest_authorization(
    path: Path,
    *,
    spec_path: Path,
    spec_audit_path: Path,
    runtime_audit_path: Path,
    component_catalog_path: Path | None = None,
    component_manifest_paths: list[Path] | None = None,
    data_dir: Path | str = "data",
    run_out: Path | str,
    include_source_presentation: bool = True,
    source_presentation_options: dict[str, object] | None = None,
) -> None:
    if component_catalog_path is None:
        component_catalog_path = spec_audit_path.with_name("component_catalog.json")
    payload = {
        "status": "authorized",
        "strategy_spec": str(spec_path),
        "spec_audit": str(spec_audit_path),
        "runtime_audit": str(runtime_audit_path),
        "component_catalog": str(component_catalog_path),
        "component_manifests": [str(path) for path in (component_manifest_paths or [])],
        "data_dir": str(data_dir),
        "run_out": str(run_out),
        "spec_hash": StrategySpec.from_yaml(spec_path).compute_hash(),
        "spec_audit_hash": _hash_json_file(spec_audit_path),
        "runtime_audit_hash": _hash_json_file(runtime_audit_path),
    }
    if include_source_presentation:
        payload["strategy_source_presentation"] = _write_strategy_source_presentation(
            runtime_audit_path,
            run_out=run_out,
            **(source_presentation_options or {}),
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_strategy_source_presentation(
    runtime_audit_path: Path,
    *,
    run_out: Path | str,
    event_path: Path | None = None,
    event_reference: str = "conversations/test-conversation/runtime-source-presentations.jsonl",
    version_id: str | None = None,
    active_run: str | None = None,
) -> dict[str, object]:
    runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    event_path = event_path or runtime_audit_path.parent / event_reference
    event = {
        "schema_version": 1,
        "timestamp": "2026-07-12T08:00:00Z",
        "phase": "runtime_source_presentation",
        "presentation": "complete_strategy_source",
        "presented_by_role": "coordinator",
        "event_id": "strategy-source-presentation-1",
        "strategy_source_path": runtime_audit["strategy_source_path"],
        "strategy_source_hash": runtime_audit["strategy_source_hash"],
        "runtime_audit_path": str(runtime_audit_path),
        "runtime_audit_hash": _hash_json_file(runtime_audit_path),
        "compiled_plan_hash": runtime_audit["compiled_plan_hash"],
        "version_id": version_id,
        "active_run": active_run,
        "run_out": str(run_out),
    }
    line = json.dumps(event, sort_keys=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text(line + "\n", encoding="utf-8")
    return {
        "path": event_reference,
        "line_number": 1,
        "event_hash": "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest(),
        **{key: value for key, value in event.items() if key not in {"schema_version", "timestamp"}},
    }


def _write_formal_gate_inputs(
    tmp_path: Path,
    *,
    include_source_presentation: bool = True,
) -> tuple[Path, Path, Path, Path, Path]:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    spec_audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    catalog_hash = _write_component_catalog(tmp_path / "component_catalog.json")
    _write_spec_audit(spec_audit_path, spec_hash, catalog_hash=catalog_hash, spec_path=spec_path)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        effective_data_dir=str(data_dir),
    )
    out_dir = tmp_path / "runs"
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        runtime_audit_path=runtime_audit_path,
        data_dir=data_dir,
        run_out=out_dir,
        include_source_presentation=include_source_presentation,
    )
    return spec_path, data_dir, spec_audit_path, runtime_audit_path, out_dir


def _invoke_formal_gate(
    spec_path: Path,
    data_dir: Path,
    spec_audit_path: Path,
    runtime_audit_path: Path,
    out_dir: Path,
):
    return CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(spec_audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(out_dir),
            "--json",
        ],
    )


def _write_governed_formal_workspace(
    workspace: Path,
    *,
    active_version: str = "v001",
    custom_layout: bool = False,
    components_dir: str = "components",
) -> dict[str, Path]:
    versions_root = "research_versions" if custom_layout else "versions"
    phase_names = {
        "01_brainstorm": "artifacts/idea" if custom_layout else "01_brainstorm",
        "02_idea_audit": "artifacts/idea_audit" if custom_layout else "02_idea_audit",
        "03_component_authoring": "artifacts/components" if custom_layout else "03_component_authoring",
        "04_spec_build": "artifacts/spec" if custom_layout else "04_spec_build",
        "05_data_inspection": "artifacts/data" if custom_layout else "05_data_inspection",
        "06_spec_audit": "artifacts/spec_audit" if custom_layout else "06_spec_audit",
        "07_compile_preview": "artifacts/compile" if custom_layout else "07_compile_preview",
        "08_runtime_audit": "artifacts/runtime_audit" if custom_layout else "08_runtime_audit",
        "09_backtests": "artifacts/backtests" if custom_layout else "09_backtests",
        "10_reports": "artifacts/reports" if custom_layout else "10_reports",
    }
    phases = {
        phase: workspace / f"{versions_root}/{active_version}/{phase_name}"
        for phase, phase_name in phase_names.items()
    }
    for path in (*phases.values(), workspace / ".open-xquant", workspace / "conversations/demo"):
        path.mkdir(parents=True, exist_ok=True)
    (workspace / ".open-xquant/workspace.yaml").write_text(
        "schema_version: 1\n"
        "workflow:\n  layout: version_governed\n"
        "paths:\n"
        f"  versions_dir: {versions_root}\n"
        "  conversations_dir: conversations\n"
        f"  components_dir: {components_dir}\n",
        encoding="utf-8",
    )
    workspace_paths = {
        "versions_dir": versions_root,
        "conversations_dir": "conversations",
        "components_dir": components_dir,
        "governance_dir": "governance",
        "runs_dir": "runs",
        "final_dir": "final",
        "comparisons_dir": "comparisons",
        "current_manifest": "current.json",
        "lineage_manifest": "lineage.json",
        "workflow_manifest": "workflow_manifest.json",
        "experiment_registry": "experiments.jsonl",
        "comparison_registry": "comparisons/comparisons.jsonl",
    }
    (workspace / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout": "version_governed",
                "strategy_family_id": "formal-gate",
                "paths": workspace_paths,
            }
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "formal-gate",
                "active_version": active_version,
                "active_phase": "09_backtests",
                "active_run": "",
            }
        ),
        encoding="utf-8",
    )
    (workspace / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "formal-gate",
                "versions": [
                    {
                        "version_id": active_version,
                        "parent_version_id": "",
                        "created_reason": "initial_strategy_version",
                        "status": "active",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "version_id": active_version,
        "strategy_family_id": "formal-gate",
        "parent_version_id": "",
        "created_reason": "initial_strategy_version",
        "status": "active",
        "active_phase": "09_backtests",
        "source_conversation": "",
        "phase_paths": {
            phase: path.relative_to(workspace).as_posix()
            for phase, path in phases.items()
        },
    }
    (workspace / f"{versions_root}/{active_version}/version_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (workspace / f"{versions_root}/{active_version}/phase_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version_id": active_version,
                "current_phase": "09_backtests",
                "status": "active",
                "completed_phases": list(VERSION_PHASE_DIRS[:8]),
                "blocked_phase": "",
            }
        ),
        encoding="utf-8",
    )
    return phases


def _write_governed_path_gate_inputs(
    workspace: Path,
    phases: dict[str, Path],
) -> tuple[Path, Path, Path, Path, Path]:
    spec_path, data_dir = _write_spec_and_data(phases["04_spec_build"])
    table_path = phases["06_spec_audit"] / "spec_confirmation_table.md"
    table_path.write_text("formal table\n", encoding="utf-8")
    spec_audit_path = phases["06_spec_audit"] / "spec_audit.json"
    spec_audit_path.write_text(
        json.dumps({"spec_confirmation_table": {"path": str(table_path)}}),
        encoding="utf-8",
    )
    compiled_plan_path = phases["07_compile_preview"] / "compiled_plan.json"
    compiled_plan_path.write_text("{}\n", encoding="utf-8")
    runtime_audit_path = phases["08_runtime_audit"] / "runtime_audit.json"
    runtime_audit_path.write_text(
        json.dumps({"compiled_plan_path": str(compiled_plan_path)}),
        encoding="utf-8",
    )
    (phases["04_spec_build"] / "component_catalog.json").write_text("{}\n", encoding="utf-8")
    (phases["08_runtime_audit"] / "backtest_authorization.json").write_text("{}\n", encoding="utf-8")
    return spec_path, data_dir, spec_audit_path, runtime_audit_path, phases["09_backtests"]


def _governance_manifest_paths(workspace: Path) -> dict[str, Path]:
    return {
        "current.json": workspace / "current.json",
        "lineage.json": workspace / "lineage.json",
        "workflow_manifest.json": workspace / "workflow_manifest.json",
        "version_manifest.json": workspace / "versions/v001/version_manifest.json",
        "phase_state.json": workspace / "versions/v001/phase_state.json",
    }


def _invoke_governed_path_gate(
    monkeypatch,
    workspace: Path,
    phases: dict[str, Path],
    inputs: tuple[Path, Path, Path, Path, Path],
    *,
    component_manifest: Path | None = None,
):
    monkeypatch.chdir(workspace)
    spec_path, data_dir, spec_audit_path, runtime_audit_path, out_dir = inputs
    args = [
        "backtest",
        "run",
        str(spec_path),
        "--spec-audit",
        str(spec_audit_path),
        "--runtime-audit",
        str(runtime_audit_path),
        "--component-catalog",
        str(phases["04_spec_build"] / "component_catalog.json"),
        "--data-dir",
        str(data_dir),
        "--out",
        str(out_dir),
        "--json",
    ]
    if component_manifest is not None:
        args[3:3] = ["--component-manifest", str(component_manifest)]
    return CliRunner().invoke(main, args)


def _invoke_governed_external_gate(
    monkeypatch,
    workspace: Path,
    inputs: tuple[Path, Path, Path, Path, Path],
    *,
    component_manifest: Path | None = None,
    component_catalog: Path | None = None,
):
    monkeypatch.chdir(workspace)
    spec_path, data_dir, spec_audit_path, runtime_audit_path, _ = inputs
    active_version = json.loads((workspace / "current.json").read_text(encoding="utf-8"))["active_version"]
    governed_out = workspace / f"versions/{active_version}/09_backtests"
    args = [
        "backtest",
        "run",
        str(spec_path),
        "--spec-audit",
        str(spec_audit_path),
        "--runtime-audit",
        str(runtime_audit_path),
        "--component-catalog",
        str(component_catalog or spec_audit_path.with_name("component_catalog.json")),
        "--data-dir",
        str(data_dir),
        "--out",
        str(governed_out),
        "--json",
    ]
    if component_manifest is not None:
        args[3:3] = ["--component-manifest", str(component_manifest)]
    return CliRunner().invoke(main, args)


@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    [
        ("current.json", "schema_version", None),
        ("current.json", "strategy_family_id", ""),
        ("current.json", "active_phase", "11_unknown"),
        ("current.json", "active_run", 1),
        ("version_manifest.json", "schema_version", True),
        ("version_manifest.json", "strategy_family_id", "other-family"),
        ("version_manifest.json", "parent_version_id", 1),
        ("version_manifest.json", "created_reason", ""),
        ("version_manifest.json", "status", "superseded"),
        ("version_manifest.json", "active_phase", "08_runtime_audit"),
        ("version_manifest.json", "source_conversation", 1),
        ("phase_state.json", "schema_version", 2),
        ("phase_state.json", "version_id", "v999"),
        ("phase_state.json", "current_phase", "08_runtime_audit"),
        ("phase_state.json", "status", 1),
        ("phase_state.json", "completed_phases", "01_brainstorm"),
        ("phase_state.json", "blocked_phase", None),
    ],
)
def test_governed_formal_backtest_rejects_incomplete_governance_schema(
    monkeypatch,
    tmp_path,
    artifact: str,
    field: str,
    value: object,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    version_dir = workspace / "versions/v001"
    artifact_path = workspace / artifact if artifact == "current.json" else version_dir / artifact
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    response = json.loads(result.output)
    assert result.exit_code == 1
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert artifact in response["errors"][0]["message"]
    assert not any(phases["09_backtests"].iterdir())


@pytest.mark.parametrize("artifact", list(_governance_manifest_paths(Path("workspace"))))
@pytest.mark.parametrize("corruption", ["symlink", "missing", "corrupt"])
def test_governed_formal_backtest_requires_canonical_governance_manifest(
    monkeypatch,
    tmp_path,
    artifact: str,
    corruption: str,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    artifact_path = _governance_manifest_paths(workspace)[artifact]
    if corruption == "symlink":
        external = tmp_path / f"external-{artifact}"
        external.write_bytes(artifact_path.read_bytes())
        artifact_path.unlink()
        artifact_path.symlink_to(external)
    elif corruption == "missing":
        artifact_path.unlink()
    else:
        artifact_path.write_text("{not-json", encoding="utf-8")

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert artifact in response["errors"][0]["message"]
    assert not any(phases["09_backtests"].iterdir())


@pytest.mark.parametrize(
    ("config_key", "filename"),
    [
        ("current_manifest", "current.json"),
        ("lineage_manifest", "lineage.json"),
        ("workflow_manifest", "workflow_manifest.json"),
    ],
)
def test_governed_formal_backtest_rejects_noncanonical_root_manifest_lexeme(
    monkeypatch,
    tmp_path,
    config_key: str,
    filename: str,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    config_path = workspace / ".open-xquant/workspace.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paths"][config_key] = f"./{filename}"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert config_key in response["errors"][0]["message"]
    assert not any(phases["09_backtests"].iterdir())


@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    [
        ("workflow_manifest.json", "strategy_family_id", "other-family"),
        ("lineage.json", "strategy_family_id", "other-family"),
        ("lineage.json", "active_version", "v999"),
        ("lineage.json", "active_status", "superseded"),
    ],
)
def test_governed_formal_backtest_rejects_cross_manifest_identity_mismatch(
    monkeypatch,
    tmp_path,
    artifact: str,
    field: str,
    value: str,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    artifact_path = _governance_manifest_paths(workspace)[artifact]
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if field == "active_version":
        payload["versions"][0]["version_id"] = value
    elif field == "active_status":
        payload["versions"][0]["status"] = value
    else:
        payload[field] = value
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert artifact in response["errors"][0]["message"]
    assert not any(phases["09_backtests"].iterdir())


@pytest.mark.parametrize("phase", VERSION_PHASE_DIRS)
@pytest.mark.parametrize("corruption", ["missing", "escape"])
def test_governed_formal_backtest_requires_every_safe_contained_phase_path(
    monkeypatch,
    tmp_path,
    phase: str,
    corruption: str,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    manifest_path = workspace / "versions/v001/version_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "missing":
        manifest["phase_paths"].pop(phase)
    else:
        manifest["phase_paths"][phase] = f"../escaped/{phase}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    response = json.loads(result.output)
    assert result.exit_code == 1
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert f"phase_paths.{phase}" in response["errors"][0]["message"]
    assert not any(phases["09_backtests"].iterdir())


def test_governed_formal_backtest_validates_governance_before_component_payload_read(
    monkeypatch,
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    current_path = workspace / "current.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current.pop("schema_version")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    payload_read = False

    def payload_read_sentinel(_manifest_paths):
        nonlocal payload_read
        payload_read = True
        raise AssertionError("component payloads must not be read before governance")

    monkeypatch.setattr("oxq.cli.main._read_component_manifest_payloads", payload_read_sentinel)

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    response = json.loads(result.output)
    assert result.exit_code == 1
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert payload_read is False
    assert not any(phases["09_backtests"].iterdir())


def test_governed_formal_backtest_rejects_external_gate_artifacts_before_import_or_output(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    _write_governed_formal_workspace(workspace)
    external = tmp_path / "external"
    external.mkdir()
    inputs = _write_formal_gate_inputs(external)
    manifest, _ = _write_hashed_component_manifest(external)
    imported = False

    def import_sentinel(_manifest_paths):
        nonlocal imported
        imported = True
        raise AssertionError("component import ran before governed artifact binding")

    monkeypatch.setattr("oxq.cli.main._load_component_manifests", import_sentinel)
    result = _invoke_governed_external_gate(
        monkeypatch,
        workspace,
        inputs,
        component_manifest=manifest,
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "formal_gate_path_failed"
    assert "active version v001 phase_paths.04_spec_build/strategy_spec.yaml" in payload["errors"][0]["message"]
    assert imported is False
    assert not inputs[-1].exists()


def test_governed_formal_backtest_rejects_symlink_escaped_authoritative_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    external = tmp_path / "external"
    external.mkdir()
    inputs = _write_formal_gate_inputs(external)
    linked_spec = phases["04_spec_build"] / "strategy_spec.yaml"
    linked_spec.symlink_to(inputs[0])
    escaped_inputs = (linked_spec, *inputs[1:])

    result = _invoke_governed_external_gate(monkeypatch, workspace, escaped_inputs)

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "formal_gate_path_failed"
    assert "must stay within active version phase_paths.04_spec_build" in payload["errors"][0]["message"]
    assert not inputs[-1].exists()


def test_governed_formal_backtest_rejects_cross_version_spec_audit_and_catalog(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace, active_version="v001")
    stale_spec_dir = workspace / "versions/v999/04_spec_build"
    stale_audit_dir = workspace / "versions/v999/06_spec_audit"
    stale_spec_dir.mkdir(parents=True)
    stale_audit_dir.mkdir(parents=True)
    spec_path, data_dir = _write_spec_and_data(stale_spec_dir)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    component_catalog_path = stale_spec_dir / "component_catalog.json"
    catalog_hash = _write_component_catalog(component_catalog_path)
    spec_audit_path = stale_audit_dir / "spec_audit.json"
    _write_spec_audit(
        spec_audit_path,
        spec_hash,
        catalog_hash=catalog_hash,
        spec_path=spec_path,
        confirmation_event_path=workspace / "conversations/demo/confirmations.jsonl",
        confirmation_event_reference="conversations/demo/confirmations.jsonl",
    )
    runtime_audit_path = phases["08_runtime_audit"] / "runtime_audit.json"
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        effective_data_dir=str(data_dir),
        strategy_source_path=phases["07_compile_preview"] / "strategy.py",
    )
    _write_backtest_authorization(
        phases["08_runtime_audit"] / "backtest_authorization.json",
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        runtime_audit_path=runtime_audit_path,
        component_catalog_path=component_catalog_path,
        data_dir=data_dir,
        run_out=phases["09_backtests"],
        source_presentation_options={
            "event_path": workspace / "conversations/demo/runtime-source-presentations.jsonl",
            "event_reference": "conversations/demo/runtime-source-presentations.jsonl",
            "version_id": "v001",
            "active_run": None,
        },
    )
    inputs = (spec_path, data_dir, spec_audit_path, runtime_audit_path, phases["09_backtests"])

    result = _invoke_governed_external_gate(
        monkeypatch,
        workspace,
        inputs,
        component_catalog=component_catalog_path,
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "formal_gate_path_failed"
    assert "active version v001" in payload["errors"][0]["message"]
    assert "phase_paths.04_spec_build/strategy_spec.yaml" in payload["errors"][0]["message"]
    assert not any(inputs[-1].iterdir())


@pytest.mark.parametrize("nested_reference", ["confirmation_table", "compiled_plan"])
def test_governed_formal_backtest_rejects_nested_reference_outside_active_phase_before_output(
    tmp_path,
    monkeypatch,
    nested_reference: str,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace, custom_layout=True)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    outside = tmp_path / f"outside-{nested_reference}.json"
    outside.write_text("{}\n", encoding="utf-8")
    if nested_reference == "confirmation_table":
        payload = json.loads(inputs[2].read_text(encoding="utf-8"))
        payload["spec_confirmation_table"]["path"] = str(outside)
        inputs[2].write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = json.loads(inputs[3].read_text(encoding="utf-8"))
        payload["compiled_plan_path"] = str(outside)
        inputs[3].write_text(json.dumps(payload), encoding="utf-8")

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert nested_reference.replace("_", " ") in response["errors"][0]["message"]
    assert not any(inputs[-1].iterdir())


@pytest.mark.parametrize("via_symlink", [False, True])
def test_governed_formal_backtest_rejects_component_manifest_outside_configured_root_before_import(
    tmp_path,
    monkeypatch,
    via_symlink: bool,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(
        workspace,
        custom_layout=True,
        components_dir="workspace_extensions",
    )
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    external = tmp_path / "external"
    external.mkdir()
    external_manifest, _digest = _write_hashed_component_manifest(external)
    manifest = external_manifest
    if via_symlink:
        components_root = workspace / "workspace_extensions"
        components_root.mkdir()
        manifest = components_root / "escaped_manifest.json"
        manifest.symlink_to(external_manifest)
    imported = False

    def import_sentinel(_manifest_paths):
        nonlocal imported
        imported = True
        raise AssertionError("component import ran before configured-root validation")

    monkeypatch.setattr("oxq.cli.main._load_component_manifests", import_sentinel)
    result = _invoke_governed_path_gate(
        monkeypatch,
        workspace,
        phases,
        inputs,
        component_manifest=manifest,
    )

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "formal_gate_path_failed"
    assert "paths.components_dir" in response["errors"][0]["message"]
    assert imported is False
    assert not any(inputs[-1].iterdir())


def test_governed_formal_path_gate_accepts_custom_roots_and_phases_before_spec_validation(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(
        workspace,
        custom_layout=True,
        components_dir="workspace_extensions",
    )
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    components_root = workspace / "workspace_extensions"
    components_root.mkdir()
    manifest, _digest = _write_hashed_component_manifest(components_root)
    reached_spec_gate = False

    def stop_at_spec_gate(*_args, **_kwargs):
        nonlocal reached_spec_gate
        reached_spec_gate = True
        raise ClickException("stop after governed path validation")

    monkeypatch.setattr("oxq.cli.main._require_pre_backtest_spec_audit", stop_at_spec_gate)
    result = _invoke_governed_path_gate(
        monkeypatch,
        workspace,
        phases,
        inputs,
        component_manifest=manifest,
    )

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "spec_audit_failed"
    assert reached_spec_gate is True
    assert not any(inputs[-1].iterdir())


def test_formal_backtest_requests_manifest_bound_spec_provenance_validation(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace, custom_layout=True)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    captured: dict[str, object] = {}

    def provenance_sentinel(_path, **kwargs):
        captured.update(kwargs)
        return {
            "status": "fail",
            "errors": [{"path": "spec_mapping_contract_hash", "message": "stale mapping"}],
        }

    monkeypatch.setattr("oxq.spec.audit_schema.validate_spec_audit_file", provenance_sentinel)
    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "spec_audit_failed"
    assert captured["require_formal_provenance"] is True
    assert not any(inputs[-1].iterdir())


def test_backtest_run_blocks_worker_boolean_without_source_presentation_before_output(tmp_path) -> None:
    inputs = _write_formal_gate_inputs(tmp_path, include_source_presentation=False)

    result = _invoke_formal_gate(*inputs)

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "backtest_authorization_failed"
    assert "strategy_source_presentation" in payload["errors"][0]["message"]
    assert not inputs[-1].exists()


def test_backtest_run_blocks_mismatched_source_presentation_hash_before_output(tmp_path) -> None:
    inputs = _write_formal_gate_inputs(tmp_path)
    authorization_path = inputs[3].with_name("backtest_authorization.json")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["strategy_source_presentation"]["strategy_source_hash"] = "sha256:" + "f" * 64
    authorization_path.write_text(json.dumps(authorization, indent=2), encoding="utf-8")

    result = _invoke_formal_gate(*inputs)

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "backtest_authorization_failed"
    assert "strategy_source_hash" in payload["errors"][0]["message"]
    assert not inputs[-1].exists()


def test_backtest_run_accepts_recorded_source_presentation(tmp_path) -> None:
    inputs = _write_formal_gate_inputs(tmp_path)

    result = _invoke_formal_gate(*inputs)

    assert result.exit_code == 0, result.output
    assert Path(json.loads(result.output)["run_dir"]).is_dir()


def test_missing_source_presentation_blocks_before_component_import_or_output(
    tmp_path,
    monkeypatch,
) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    manifest, bundle_hash = _write_hashed_component_manifest(tmp_path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["_manifest_path"] = str(manifest.resolve())
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    spec_audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    catalog_hash = _write_component_catalog(tmp_path / "component_catalog.json", [manifest_payload])
    _write_spec_audit(spec_audit_path, spec_hash, catalog_hash=catalog_hash, spec_path=spec_path)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        effective_data_dir=str(data_dir),
        component_bundle_hashes=[bundle_hash],
    )
    out_dir = tmp_path / "runs"
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=spec_audit_path,
        runtime_audit_path=runtime_audit_path,
        component_manifest_paths=[manifest],
        data_dir=data_dir,
        run_out=out_dir,
        include_source_presentation=False,
    )
    imported = False

    def import_sentinel(_manifest_paths):
        nonlocal imported
        imported = True
        raise AssertionError("component import ran before source-presentation gate")

    monkeypatch.setattr("oxq.cli.main._load_component_manifests", import_sentinel)
    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(spec_audit_path),
            "--runtime-audit",
            str(runtime_audit_path),
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
    assert "strategy_source_presentation" in result.output
    assert imported is False
    assert not out_dir.exists()


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
        json.dumps({"schema_version": 4, "status": "pass", "spec_hash": "sha256:stale"}),
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
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=audit_path,
        runtime_audit_path=runtime_audit_path,
        component_manifest_paths=[manifest],
        data_dir=data_dir,
        run_out=tmp_path / "runs_ok",
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


def test_backtest_run_json_rejects_extra_runtime_audit_component_bundle_hashes(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    _write_spec_audit(audit_path, spec_hash, spec_path=spec_path)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir),
        component_bundle_hashes=["sha256:" + "9" * 16],
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
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "runtime_audit_failed"
    assert "component_bundle_hashes mismatch" in payload["errors"][0]["message"]


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
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=audit_path,
        runtime_audit_path=runtime_audit_path,
        data_dir=data_dir,
        run_out=tmp_path / "runs",
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


def test_backtest_run_json_rejects_tampered_confirmation_event(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    run_out = tmp_path / "runs"
    _write_spec_audit(audit_path, spec_hash, spec_path=spec_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    event_path = audit_path.parent / str(audit["confirmation_event"]["path"])
    event_path.write_text(
        event_path.read_text(encoding="utf-8").replace('"field_scope": "full_spec_table"', '"field_scope": "partial"'),
        encoding="utf-8",
    )
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir),
    )
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=audit_path,
        runtime_audit_path=runtime_audit_path,
        data_dir=data_dir,
        run_out=run_out,
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
            str(run_out),
            "--json",
        ],
    )

    assert result.exit_code == 1
    response = json.loads(result.output)
    assert response["errors"][0]["check"] == "spec_audit_failed"
    assert "confirmation_event" in response["errors"][0]["message"]


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
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=audit_path,
        runtime_audit_path=runtime_audit_path,
        data_dir=data_dir,
        run_out=tmp_path / "runs",
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


def test_backtest_run_json_rejects_missing_backtest_authorization(tmp_path) -> None:
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

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "backtest_authorization_missing"


def test_backtest_run_rejects_missing_authorization_before_component_import(monkeypatch, tmp_path) -> None:
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
    imported = False

    def import_sentinel(_manifest_paths):
        nonlocal imported
        imported = True
        raise AssertionError("component loader ran before authorization")

    monkeypatch.setattr("oxq.cli.main._load_component_manifests", import_sentinel)

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
    assert payload["errors"][0]["check"] == "backtest_authorization_missing"
    assert imported is False


def test_backtest_run_json_accepts_passing_pre_run_audits(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
    audit_path = tmp_path / "spec_audit.json"
    runtime_audit_path = tmp_path / "runtime_audit.json"
    run_out = tmp_path / "runs"
    _write_spec_audit(audit_path, spec_hash)
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir),
    )
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=audit_path,
        runtime_audit_path=runtime_audit_path,
        data_dir=data_dir,
        run_out=run_out,
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
            str(run_out),
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

    spec_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    authorization = json.loads(
        runtime_audit_path.with_name("backtest_authorization.json").read_text(encoding="utf-8")
    )
    assert "hash_type" not in spec_audit["confirmation_event"]
    assert "strategy_idea_brief_hash_type" not in spec_audit
    assert "strategy_idea_audit_hash_type" not in spec_audit
    assert "hash_type" not in runtime_audit
    assert "hash_type" not in authorization

    policy_boundary = "Require `hash_type` only on structured references whose schema defines that field"
    for contract_path in (
        Path("agent/skills/audit-artifact-lineage/SKILL.md"),
        Path("agent/skills/govern-research-workspace/SKILL.md"),
        Path("agent/roles/oxq-lineage-auditor-worker.md"),
        Path("docs/strategy-workflow-artifact-governance.md"),
    ):
        assert policy_boundary in contract_path.read_text(encoding="utf-8"), contract_path


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
    _write_backtest_authorization(
        runtime_audit_path.with_name("backtest_authorization.json"),
        spec_path=spec_path,
        spec_audit_path=audit_path,
        runtime_audit_path=runtime_audit_path,
        data_dir=data_dir,
        run_out=tmp_path / "runs",
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
    _write_runtime_audit(
        runtime_audit_path,
        spec_hash,
        spec_path=spec_path,
        spec_audit_path=audit_path,
        effective_data_dir=str(data_dir),
    )
    runtime_payload = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    runtime_payload["spec_audit_hash"] = "sha256:" + "4" * 16
    runtime_audit_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")

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


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "malformed"])
def test_backtest_compare_runs_requires_exactly_one_valid_requested_run_digest(
    tmp_path,
    corruption: str,
) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()

    def run(out: Path):
        result = runner.invoke(
            main,
            [
                "backtest",
                "run",
                str(spec_path),
                "--data-dir",
                str(data_dir),
                "--out",
                str(out),
                "--allow-unaudited",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        return Path(json.loads(result.output)["run_dir"])

    left_run = run(tmp_path / "runs_a")
    right_run = run(tmp_path / "runs_b")
    digest_path = right_run.parent / "run_digests.jsonl"
    original_line = digest_path.read_text(encoding="utf-8").strip()
    if corruption == "missing":
        digest_path.write_text(
            json.dumps(
                {
                    "run_id": "other-run",
                    "artifact_hashes": "sha256:" + "0" * 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    elif corruption == "duplicate":
        digest_path.write_text(f"{original_line}\n{original_line}\n", encoding="utf-8")
    else:
        digest_path.write_text(
            json.dumps({"run_id": right_run.name, "artifact_hashes": 7}) + "\n",
            encoding="utf-8",
        )

    result = runner.invoke(
        main,
        ["backtest", "compare-runs", str(left_run), str(right_run), "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["errors"][0]["check"] == "run_artifacts_invalid"
    assert "run_digests.jsonl" in payload["errors"][0]["message"]


def test_compile_robustness_refresh_then_compare_preserves_single_run_digest(tmp_path) -> None:
    from oxq.robustness.runner import run_robustness

    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()

    def compile_run(out: Path) -> Path:
        result = runner.invoke(
            main,
            [
                "backtest",
                "run",
                str(spec_path),
                "--data-dir",
                str(data_dir),
                "--out",
                str(out),
                "--allow-unaudited",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        return Path(json.loads(result.output)["run_dir"])

    left_run = compile_run(tmp_path / "runs_a")
    right_run = compile_run(tmp_path / "runs_b")

    robustness = run_robustness(right_run)
    compare = runner.invoke(
        main,
        ["backtest", "compare-runs", str(left_run), str(right_run), "--json"],
    )

    matching_digests = [
        json.loads(line)
        for line in (right_run.parent / "run_digests.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("run_id") == right_run.name
    ]
    assert robustness["status"] != "error"
    assert len(matching_digests) == 1
    assert compare.exit_code == 0, compare.output


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
    _publish_comparison_provenance_for_test(
        second_run,
        spec_audit={"status": "pass"},
        runtime_audit={"status": "pass"},
        component_catalog_hash="sha256:" + "d" * 64,
    )
    spec_audit_path = second_run / "spec_audit.json"
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
    component_manifest, _ = _write_hashed_component_manifest(tmp_path)
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
            "--component-manifest",
            str(component_manifest),
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
    _publish_comparison_provenance_for_test(
        first_run,
        spec_audit={"status": "pass", "run": "a"},
        runtime_audit={"status": "pass", "run": "a"},
        component_catalog_hash="sha256:" + "a" * 64,
    )
    _publish_comparison_provenance_for_test(
        second_run,
        spec_audit={"status": "pass", "run": "b"},
        runtime_audit={"status": "pass", "run": "b"},
        component_catalog_hash="sha256:" + "a" * 64,
    )

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
    _publish_comparison_provenance_for_test(
        first_run,
        spec_audit={"status": "pass"},
        runtime_audit={"status": "pass"},
        component_catalog_hash="sha256:" + "a" * 64,
    )
    _publish_comparison_provenance_for_test(
        second_run,
        spec_audit={"status": "pass"},
        runtime_audit={"status": "pass"},
        component_catalog_hash="sha256:" + "b" * 64,
    )

    compare = runner.invoke(main, ["backtest", "compare-runs", str(first_run), str(second_run), "--json"])

    assert compare.exit_code == 1
    payload = json.loads(compare.output)
    assert payload["comparable"] is False
    assert any(item["field"] == "component_catalog_hash" for item in payload["differences"])


@pytest.mark.parametrize("versions_dir", [None, "", 7])
def test_formal_backtest_rejects_malformed_configured_versions_dir_as_invalid_governed(
    monkeypatch,
    tmp_path,
    versions_dir: object,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    config_path = workspace / ".open-xquant/workspace.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.pop("workflow")
    config["paths"]["versions_dir"] = versions_dir
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    before = {
        path.relative_to(workspace).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(workspace.rglob("*"))
    }

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    assert result.exit_code == 1
    assert "workspace paths.versions_dir must be a non-empty string" in result.output
    assert {
        path.relative_to(workspace).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(workspace.rglob("*"))
    } == before


def test_formal_backtest_rejects_workspace_root_drift_without_mutation(
    monkeypatch,
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    phases = _write_governed_formal_workspace(workspace)
    inputs = _write_governed_path_gate_inputs(workspace, phases)
    config_path = workspace / ".open-xquant/workspace.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paths"]["versions_dir"] = "research_versions"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    before = {
        path.relative_to(workspace).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(workspace.rglob("*"))
    }

    result = _invoke_governed_path_gate(monkeypatch, workspace, phases, inputs)

    assert result.exit_code == 1
    assert "workflow_manifest.json.paths.versions_dir" in result.output
    assert "explicit migration" in result.output
    assert {
        path.relative_to(workspace).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(workspace.rglob("*"))
    } == before
