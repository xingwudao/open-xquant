from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from oxq.cli.research import _WORKSPACE_PATH_DEFAULTS, VERSION_PHASE_DIRS
from oxq.core.component_catalog import _catalog_hash
from oxq.spec.audit_schema import validate_spec_audit, validate_spec_audit_file
from oxq.spec.mapping_contract import (
    validate_mapping_contract_file,
    validate_mapping_contract_for_builder_pass_file,
)
from oxq.spec.runtime_audit_schema import (
    validate_runtime_audit,
    validate_runtime_audit_file,
    validate_strategy_source_presentation,
)
from oxq.spec.schema import StrategySpec


def _payload(field_audits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": {
            "path": "conversations/demo/confirmations.jsonl",
            "event_id": "spec-confirmation-1",
            "decision": "confirmed",
            "line_number": 1,
            "event_hash": "sha256:" + "7" * 16,
            "artifact_path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "artifact_hash": "sha256:" + "4" * 16,
            "spec_audit_path": "versions/v001/06_spec_audit/spec_audit.json",
            "spec_audit_hash": "sha256:" + "8" * 16,
        },
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        "strategy_idea_brief": "versions/v001/01_brainstorm/strategy_idea_brief.json",
        "strategy_idea_audit": "versions/v001/02_idea_audit/strategy_idea_audit.json",
        "strategy_idea_brief_hash": "sha256:" + "5" * 16,
        "strategy_idea_audit_hash": "sha256:" + "6" * 16,
        "spec_mapping_contract": "versions/v001/04_spec_build/spec_mapping_contract.json",
        "spec_mapping_contract_hash": "sha256:" + "9" * 16,
        "spec_mapping_contract_status": "pass",
        "recipe_matches": [],
        "field_audits": field_audits,
        "component_audits": [],
        "unsupported_mappings": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }


@pytest.mark.parametrize(
    ("validator", "expected_path"),
    [
        (validate_spec_audit_file, "$"),
        (validate_runtime_audit_file, "$"),
        (validate_mapping_contract_file, "$"),
        (validate_mapping_contract_for_builder_pass_file, "$"),
    ],
)
def test_file_validators_normalize_invalid_utf8(tmp_path, validator, expected_path: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(b"\xff\xfe")

    result = validator(path)

    assert result["status"] == "fail"
    assert result["errors"][0]["path"] == expected_path


def test_spec_audit_component_catalog_normalizes_invalid_utf8(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    catalog_path = tmp_path / "component_catalog.json"
    audit_path.write_text("{}", encoding="utf-8")
    catalog_path.write_bytes(b"\xff\xfe")

    result = validate_spec_audit_file(audit_path, component_catalog_path=catalog_path)

    assert result["status"] == "fail"
    assert result["errors"][0]["path"] == "component_catalog"


def _confirmed(path: str, value: Any, evidence: str | None = None) -> dict[str, Any]:
    return {
        "field_path": path,
        "spec_value": value,
        "status": "confirmed",
        "material_category": "execution_assumption",
        "evidence": [evidence or f"User: {path} = {json.dumps(value, sort_keys=True, default=str)}"],
        "blocking": False,
    }


def _write_confirmation_event_line(
    path: Path,
    *,
    artifact_path: str,
    artifact_hash: str,
    event_artifact_hash: str | None = None,
    spec_audit_path: str = "spec_audit.json",
    spec_audit_hash: str = "sha256:" + "8" * 16,
    decision: str = "confirmed",
) -> dict[str, Any]:
    event = {
        "timestamp": "2026-07-07T08:00:00Z",
        "phase": "spec_confirmation",
        "field_scope": "full_spec_table",
        "decision": decision,
        "event_id": "spec-confirmation-1",
        "user_text": "确认",
        "artifact_path": artifact_path,
        "artifact_hash": event_artifact_hash or artifact_hash,
        "spec_audit_path": spec_audit_path,
        "spec_audit_hash": spec_audit_hash,
    }
    line = json.dumps(event, sort_keys=True, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")
    path_parts = path.parts
    event_reference = str(path)
    if "conversations" in path_parts:
        conversations_index = path_parts.index("conversations")
        event_reference = Path(*path_parts[conversations_index:]).as_posix()
    return {
        "path": event_reference,
        "event_id": event["event_id"],
        "decision": decision,
        "line_number": 1,
        "event_hash": f"sha256:{hashlib.sha256(line.encode('utf-8')).hexdigest()}",
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "spec_audit_path": spec_audit_path,
        "spec_audit_hash": spec_audit_hash,
    }


def _pre_confirmation_spec_audit_hash(payload: dict[str, Any]) -> str:
    candidate = json.loads(json.dumps(payload))
    candidate.pop("confirmation_event", None)
    candidate["status"] = "block"
    candidate["user_confirmation_status"] = "pending"
    canonical = json.dumps(candidate, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _canonical_json_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _write_governed_provenance_fixture(tmp_path: Path) -> tuple[Path, StrategySpec, dict[str, Path]]:
    workspace = tmp_path / "workspace"
    version_dir = workspace / "research_versions/v003"
    phases = {
        "01_brainstorm": version_dir / "custom/idea",
        "02_idea_audit": version_dir / "custom/idea_audit",
        "03_component_authoring": version_dir / "custom/components",
        "04_spec_build": version_dir / "custom/spec",
        "05_data_inspection": version_dir / "custom/data",
        "06_spec_audit": version_dir / "custom/spec_audit",
        "07_compile_preview": version_dir / "custom/compile",
        "08_runtime_audit": version_dir / "custom/runtime_audit",
        "09_backtests": version_dir / "custom/backtests",
        "10_reports": version_dir / "custom/reports",
    }
    for phase_path in phases.values():
        phase_path.mkdir(parents=True)
    (workspace / ".open-xquant").mkdir()
    (workspace / ".open-xquant/workspace.yaml").write_text(
        "schema_version: 1\n"
        "workflow:\n  layout: version_governed\n"
        "paths:\n  versions_dir: research_versions\n",
        encoding="utf-8",
    )
    (workspace / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout": "version_governed",
                "strategy_family_id": "provenance",
                "paths": {
                    **_WORKSPACE_PATH_DEFAULTS,
                    "versions_dir": "research_versions",
                },
            }
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "provenance",
                "active_version": "v003",
                "active_phase": "06_spec_audit",
                "active_run": "",
            }
        ),
        encoding="utf-8",
    )
    lineage_path = workspace / "lineage.json"
    lineage_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "provenance",
                "versions": [
                    {
                        "version_id": "v003",
                        "parent_version_id": "",
                        "created_reason": "initial_strategy_version",
                        "status": "active",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version_id": "v003",
                "strategy_family_id": "provenance",
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "active",
                "active_phase": "06_spec_audit",
                "source_conversation": "",
                "phase_paths": {
                    phase: path.relative_to(workspace).as_posix()
                    for phase, path in phases.items()
                },
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "phase_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version_id": "v003",
                "current_phase": "06_spec_audit",
                "status": "active",
                "completed_phases": list(VERSION_PHASE_DIRS[:5]),
                "blocked_phase": "",
            }
        ),
        encoding="utf-8",
    )
    conversation_hash = "sha256:" + "2" * 16
    idea_brief_path = phases["01_brainstorm"] / "strategy_idea_brief.json"
    idea_brief_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_name": "provenance",
                "conversation_hash": conversation_hash,
            }
        ),
        encoding="utf-8",
    )
    idea_audit_path = phases["02_idea_audit"] / "strategy_idea_audit.json"
    idea_audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "idea_workflow_pass": True,
                "strategy_idea_brief": idea_brief_path.relative_to(workspace).as_posix(),
                "strategy_idea_brief_hash": _canonical_json_hash(idea_brief_path),
                "conversation_hash": conversation_hash,
                "next_required_phase": "build",
            }
        ),
        encoding="utf-8",
    )
    mapping_path = phases["04_spec_build"] / "spec_mapping_contract.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_format": "strategy_idea_brief",
                "source_fields": ["strategy_name"],
                "field_mappings": [
                    {
                        "source_field": "strategy_name",
                        "target_field": "strategy_id",
                        "semantic": "strategy",
                        "status": "mapped",
                        "confirmation_required": False,
                        "blocking": False,
                        "reason": "The confirmed strategy name supplies the strategy id.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec = StrategySpec.template(strategy_id="provenance", hypothesis="formal provenance binding")
    audit_path = phases["06_spec_audit"] / "spec_audit.json"
    payload = _payload([])
    payload.update(
        {
            "status": "block",
            "audit_conclusion": "blocked",
            "user_confirmation_status": "pending",
            "spec_confirmation_table": None,
            "confirmation_event": None,
            "spec_provenance_pass": False,
            "spec_hash": spec.compute_hash(),
            "conversation_hash": conversation_hash,
            "strategy_idea_brief": idea_brief_path.relative_to(workspace).as_posix(),
            "strategy_idea_audit": idea_audit_path.relative_to(workspace).as_posix(),
            "strategy_idea_brief_hash": _canonical_json_hash(idea_brief_path),
            "strategy_idea_audit_hash": _canonical_json_hash(idea_audit_path),
            "spec_mapping_contract": mapping_path.relative_to(workspace).as_posix(),
            "spec_mapping_contract_hash": _canonical_json_hash(mapping_path),
            "spec_mapping_contract_status": "pass",
            "blocking_findings": [{"message": "fixture remains pre-confirmation"}],
        }
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    return audit_path, spec, {
        "workspace": workspace,
        "workflow_manifest": workspace / "workflow_manifest.json",
        "current": workspace / "current.json",
        "lineage": lineage_path,
        "version_manifest": version_dir / "version_manifest.json",
        "phase_state": version_dir / "phase_state.json",
        "idea_brief": idea_brief_path,
        "idea_audit": idea_audit_path,
        "mapping": mapping_path,
    }


def test_formal_spec_audit_accepts_active_manifest_owned_provenance(tmp_path) -> None:
    audit_path, spec, _paths = _write_governed_provenance_fixture(tmp_path)

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_formal_spec_audit_rejects_top_level_audit_symlink_outside_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    audit_path, spec, paths = _write_governed_provenance_fixture(tmp_path)
    workspace = paths["workspace"]
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    for field, artifact in (
        ("strategy_idea_brief", "idea_brief"),
        ("strategy_idea_audit", "idea_audit"),
        ("spec_mapping_contract", "mapping"),
    ):
        payload[field] = str(paths[artifact])
    idea_audit = json.loads(paths["idea_audit"].read_text(encoding="utf-8"))
    idea_audit["strategy_idea_brief"] = str(paths["idea_brief"])
    paths["idea_audit"].write_text(json.dumps(idea_audit), encoding="utf-8")
    payload["strategy_idea_audit_hash"] = _canonical_json_hash(paths["idea_audit"])
    outside_audit = tmp_path / "outside-spec_audit.json"
    outside_audit.write_text(json.dumps(payload), encoding="utf-8")
    audit_path = workspace / "spec_audit.json"
    audit_path.symlink_to(outside_audit)
    monkeypatch.chdir(workspace)

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_audit.json" for error in result["errors"])


@pytest.mark.parametrize(
    ("parent_version_id", "created_reason", "expected_status"),
    [
        (None, "initial_strategy_version", "pass"),
        ("", "initial_strategy_version", "pass"),
        ("v002", "parameter_change", "pass"),
        (None, "parameter_change", "fail"),
        ("", "parameter_change", "fail"),
        ("v002", "initial_strategy_version", "fail"),
    ],
)
def test_formal_governance_enforces_parent_reason_bidirectional_invariant(
    tmp_path,
    parent_version_id: str | None,
    created_reason: str,
    expected_status: str,
) -> None:
    audit_path, spec, paths = _write_governed_provenance_fixture(tmp_path)
    version_manifest = json.loads(paths["version_manifest"].read_text(encoding="utf-8"))
    version_manifest["parent_version_id"] = parent_version_id
    version_manifest["created_reason"] = created_reason
    paths["version_manifest"].write_text(json.dumps(version_manifest), encoding="utf-8")
    lineage = json.loads(paths["lineage"].read_text(encoding="utf-8"))
    lineage["versions"][0]["parent_version_id"] = parent_version_id
    lineage["versions"][0]["created_reason"] = created_reason
    paths["lineage"].write_text(json.dumps(lineage), encoding="utf-8")

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == expected_status
    if expected_status == "fail":
        assert any("initial_strategy_version" in error["message"] for error in result["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strategy_idea_brief", None),
        ("strategy_idea_brief", "research_versions/v999/strategy_idea_brief.json"),
        ("strategy_idea_brief_hash", None),
        ("strategy_idea_brief_hash", "sha256:" + "f" * 16),
        ("conversation_hash", None),
        ("conversation_hash", "sha256:" + "e" * 16),
    ],
)
def test_formal_spec_audit_rejects_internal_idea_audit_identity(
    tmp_path,
    field: str,
    value: str | None,
) -> None:
    audit_path, spec, paths = _write_governed_provenance_fixture(tmp_path)
    idea_audit = json.loads(paths["idea_audit"].read_text(encoding="utf-8"))
    if value is None:
        idea_audit.pop(field)
    else:
        idea_audit[field] = value
    paths["idea_audit"].write_text(json.dumps(idea_audit), encoding="utf-8")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["strategy_idea_audit_hash"] = _canonical_json_hash(paths["idea_audit"])
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == "fail"
    assert any(
        error["path"] == f"strategy_idea_audit.{field}"
        for error in result["errors"]
    )


@pytest.mark.parametrize(
    ("artifact", "error_path"),
    [
        ("idea_brief", "strategy_idea_brief"),
        ("idea_audit", "strategy_idea_audit"),
        ("mapping", "spec_mapping_contract"),
    ],
)
def test_formal_spec_audit_rejects_symlinked_manifest_owned_leaf(
    tmp_path,
    artifact: str,
    error_path: str,
) -> None:
    audit_path, spec, paths = _write_governed_provenance_fixture(tmp_path)
    canonical_path = paths[artifact]
    external_path = tmp_path / f"external-{canonical_path.name}"
    external_path.write_bytes(canonical_path.read_bytes())
    canonical_path.unlink()
    canonical_path.symlink_to(external_path)

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == error_path for error in result["errors"])


@pytest.mark.parametrize("field", ["strategy_idea_brief", "strategy_idea_audit"])
def test_formal_spec_audit_rejects_cross_version_idea_artifact(tmp_path, field: str) -> None:
    audit_path, spec, paths = _write_governed_provenance_fixture(tmp_path)
    stale_path = paths["workspace"] / f"research_versions/v999/{field}.json"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload[field] = stale_path.relative_to(paths["workspace"]).as_posix()
    payload[f"{field}_hash"] = _canonical_json_hash(stale_path)
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == field for error in result["errors"])


def test_formal_spec_audit_rejects_invented_idea_hash_and_blocked_idea_audit(tmp_path) -> None:
    audit_path, spec, paths = _write_governed_provenance_fixture(tmp_path)
    idea_audit = json.loads(paths["idea_audit"].read_text(encoding="utf-8"))
    idea_audit["status"] = "block"
    paths["idea_audit"].write_text(json.dumps(idea_audit), encoding="utf-8")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["strategy_idea_brief_hash"] = "sha256:" + "f" * 16
    payload["strategy_idea_audit_hash"] = _canonical_json_hash(paths["idea_audit"])
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == "fail"
    error_paths = {error["path"] for error in result["errors"]}
    assert "strategy_idea_brief_hash" in error_paths
    assert "strategy_idea_audit.status" in error_paths


@pytest.mark.parametrize(
    ("recorded_hash", "recorded_status", "mapping_status", "expected_paths"),
    [
        ("invented", "pass", "mapped", {"spec_mapping_contract_hash"}),
        ("canonical", "block", "mapped", {"spec_mapping_contract_status"}),
        (
            "stale",
            "pass",
            "blocked",
            {"spec_mapping_contract_hash", "spec_mapping_contract_status", "spec_mapping_contract.builder_pass"},
        ),
    ],
)
def test_formal_spec_audit_binds_current_mapping_hash_and_status(
    tmp_path,
    recorded_hash: str,
    recorded_status: str,
    mapping_status: str,
    expected_paths: set[str],
) -> None:
    audit_path, spec, paths = _write_governed_provenance_fixture(tmp_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if mapping_status == "blocked":
        mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
        mapping["field_mappings"][0].update(
            {
                "target_field": "",
                "status": "blocked",
                "blocking": True,
                "reason": "The current mapping is blocked.",
            }
        )
        paths["mapping"].write_text(json.dumps(mapping), encoding="utf-8")
    if recorded_hash == "canonical":
        payload["spec_mapping_contract_hash"] = _canonical_json_hash(paths["mapping"])
    elif recorded_hash == "invented":
        payload["spec_mapping_contract_hash"] = "sha256:" + "e" * 16
    payload["spec_mapping_contract_status"] = recorded_status
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(
        audit_path,
        spec=spec,
        require_formal_provenance=True,
    )

    assert result["status"] == "fail"
    error_paths = {error["path"] for error in result["errors"]}
    assert expected_paths <= error_paths


def _spec_confirmation_table(spec: dict[str, Any]) -> str:
    rows = [
        "| Section | Field path | Spec value | Source | Audit status | Impact |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for field_path, value in _flatten_effective_fields(spec):
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


def _write_bound_spec_audit_file(tmp_path: Path, spec: dict[str, Any], table_text: str) -> Path:
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text(table_text, encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    return audit_path


def _write_workspace_bound_spec_audit(
    workspace: Path,
    *,
    conversations_dir: str | None,
    event_path: Path,
    event_reference: str,
) -> Path:
    config_dir = workspace / ".open-xquant"
    config_dir.mkdir(parents=True)
    paths = "" if conversations_dir is None else f"paths:\n  conversations_dir: {conversations_dir}\n"
    (config_dir / "workspace.yaml").write_text(f"schema_version: 1\n{paths}", encoding="utf-8")
    configured_conversations_dir = conversations_dir or "conversations"
    (workspace / configured_conversations_dir).mkdir(parents=True, exist_ok=True)

    spec = {"execution": {"initial_cash": 100000}}
    audit_path = workspace / "spec_audit.json"
    table_path = workspace / "spec_confirmation_table.md"
    table_path.write_text(_spec_confirmation_table(spec), encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["spec_confirmation_table"] = {
        "path": table_path.name,
        "hash": table_hash,
        "hash_type": "sha256",
    }
    event_path.parent.mkdir(parents=True, exist_ok=True)
    payload["confirmation_event"] = _write_confirmation_event_line(
        event_path,
        artifact_path=table_path.name,
        artifact_hash=table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    payload["confirmation_event"]["path"] = event_reference
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    return audit_path


def _runtime_payload(
    effective_spec: dict[str, Any],
    compiled_plan: dict[str, Any],
) -> dict[str, Any]:
    runtime_paths = {
        "required_oxq_version": "open_xquant_version",
        "market": "market",
        "universe": "universe",
        "data": "data",
        "signal": "signals",
        "portfolio": "portfolio",
        "execution": "execution",
        "cost": "cost",
        "benchmark": "benchmark",
        "validation": "validation",
        "metrics": "metrics",
    }
    return {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": True,
        "strategy_source_path": "versions/v001/07_compile_preview/strategy.py",
        "strategy_source_hash": "sha256:" + "4" * 64,
        "spec_hash": "sha256:" + "1" * 16,
        "spec_audit_hash": "sha256:" + "2" * 16,
        "compiled_plan_hash": "sha256:" + "3" * 16,
        "compiled_plan_path": "versions/v001/07_compile_preview/compiled_plan.json",
        "material_field_audits": [
            {
                "field_path": field_path,
                "spec_value": json.loads(json.dumps(effective_spec[field_path])),
                "runtime_path": runtime_path,
                "runtime_value": json.loads(json.dumps(compiled_plan[runtime_path])),
                "status": "preserved",
                "evidence": ["Compiled plan preserves the material field."],
                "blocking": False,
            }
            for field_path, runtime_path in runtime_paths.items()
        ],
        "blocking_findings": [],
    }


def test_runtime_audit_rejects_legacy_worker_asserted_source_presentation() -> None:
    effective_spec, compiled_plan = _runtime_material_values()
    payload = _runtime_payload(effective_spec, compiled_plan)
    payload["schema_version"] = 1
    payload.pop("strategy_source_path")
    payload.pop("strategy_source_hash")
    payload["strategy_source_printed"] = True

    result = validate_runtime_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "schema_version" for error in result["errors"])
    assert any(error["path"] == "strategy_source_path" for error in result["errors"])
    assert any(error["path"] == "strategy_source_hash" for error in result["errors"])


def _write_version_bound_source_presentation(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    version_dir = tmp_path / "research_versions/v001"
    compile_dir = version_dir / "07_compile_preview"
    runtime_dir = version_dir / "08_runtime_audit"
    run_out = version_dir / "09_backtests"
    conversations_dir = tmp_path / "evidence/dialogues/demo"
    for path in (compile_dir, runtime_dir, run_out, conversations_dir, tmp_path / ".open-xquant"):
        path.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".open-xquant/workspace.yaml").write_text(
        "schema_version: 1\n"
        "workflow:\n  layout: version_governed\n"
        "paths:\n"
        "  versions_dir: research_versions\n"
        "  conversations_dir: evidence/dialogues\n",
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"active_version": "v001", "active_run": ""}),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "phase_paths": {
                    "07_compile_preview": "research_versions/v001/07_compile_preview",
                    "08_runtime_audit": "research_versions/v001/08_runtime_audit",
                    "09_backtests": "research_versions/v001/09_backtests",
                },
            }
        ),
        encoding="utf-8",
    )
    source_path = compile_dir / "strategy.py"
    source_path.write_text("# exact generated strategy\n", encoding="utf-8")
    source_hash = f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    runtime_audit_path = runtime_dir / "runtime_audit.json"
    runtime_audit = {
        "strategy_source_path": "research_versions/v001/07_compile_preview/strategy.py",
        "strategy_source_hash": source_hash,
        "compiled_plan_hash": "sha256:" + "3" * 16,
    }
    runtime_audit_path.write_text(json.dumps(runtime_audit), encoding="utf-8")
    runtime_hash = f"sha256:{hashlib.sha256(json.dumps(runtime_audit, sort_keys=True).encode()).hexdigest()[:16]}"
    event = {
        "schema_version": 1,
        "timestamp": "2026-07-12T08:00:00Z",
        "phase": "runtime_source_presentation",
        "presentation": "complete_strategy_source",
        "presented_by_role": "coordinator",
        "event_id": "source-presentation-v001",
        "strategy_source_path": "research_versions/v001/07_compile_preview/strategy.py",
        "strategy_source_hash": source_hash,
        "runtime_audit_path": "research_versions/v001/08_runtime_audit/runtime_audit.json",
        "runtime_audit_hash": runtime_hash,
        "compiled_plan_hash": runtime_audit["compiled_plan_hash"],
        "version_id": "v001",
        "active_run": None,
        "run_out": "research_versions/v001/09_backtests",
    }
    line = json.dumps(event, sort_keys=True)
    event_path = conversations_dir / "runtime-source-presentations.jsonl"
    event_path.write_text(line + "\n", encoding="utf-8")
    authorization = {
        "strategy_source_presentation": {
            "path": "evidence/dialogues/demo/runtime-source-presentations.jsonl",
            "line_number": 1,
            "event_hash": f"sha256:{hashlib.sha256(line.encode()).hexdigest()}",
            **{key: value for key, value in event.items() if key not in {"schema_version", "timestamp"}},
        }
    }
    authorization_path = runtime_dir / "backtest_authorization.json"
    return authorization, authorization_path, runtime_audit_path, run_out


def test_source_presentation_accepts_active_version_manifest_bindings(tmp_path) -> None:
    authorization, authorization_path, runtime_audit_path, run_out = _write_version_bound_source_presentation(tmp_path)

    result = validate_strategy_source_presentation(
        authorization,
        authorization_path=authorization_path,
        runtime_audit_path=runtime_audit_path,
        run_out=run_out,
    )

    assert result == {"status": "pass", "errors": []}


def test_source_presentation_rejects_mismatched_active_version(tmp_path) -> None:
    authorization, authorization_path, runtime_audit_path, run_out = _write_version_bound_source_presentation(tmp_path)
    authorization["strategy_source_presentation"]["version_id"] = "v002"

    result = validate_strategy_source_presentation(
        authorization,
        authorization_path=authorization_path,
        runtime_audit_path=runtime_audit_path,
        run_out=run_out,
    )

    assert result["status"] == "fail"
    assert "version_id mismatch" in result["errors"][0]["message"]


def test_source_presentation_rejects_event_without_event_id(tmp_path) -> None:
    authorization, authorization_path, runtime_audit_path, run_out = _write_version_bound_source_presentation(tmp_path)
    reference = authorization["strategy_source_presentation"]
    event_path = tmp_path / reference["path"]
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event.pop("event_id")
    line = json.dumps(event, sort_keys=True)
    event_path.write_text(line + "\n", encoding="utf-8")
    reference.pop("event_id")
    reference["event_hash"] = f"sha256:{hashlib.sha256(line.encode()).hexdigest()}"

    result = validate_strategy_source_presentation(
        authorization,
        authorization_path=authorization_path,
        runtime_audit_path=runtime_audit_path,
        run_out=run_out,
    )

    assert result["status"] == "fail"
    assert "event_id" in result["errors"][0]["message"]


def _rewrite_source_presentation_event(
    authorization: dict[str, Any],
    workspace: Path,
    event: dict[str, Any],
) -> None:
    reference = authorization["strategy_source_presentation"]
    line = json.dumps(event, sort_keys=True)
    (workspace / reference["path"]).write_text(line + "\n", encoding="utf-8")
    reference["event_hash"] = f"sha256:{hashlib.sha256(line.encode()).hexdigest()}"


def test_source_presentation_rejects_root_level_conversation_event(tmp_path) -> None:
    authorization, authorization_path, runtime_audit_path, run_out = _write_version_bound_source_presentation(tmp_path)
    reference = authorization["strategy_source_presentation"]
    nested_event = tmp_path / reference["path"]
    root_event = tmp_path / "evidence/dialogues/runtime-source-presentations.jsonl"
    root_event.write_text(nested_event.read_text(encoding="utf-8"), encoding="utf-8")
    reference["path"] = "evidence/dialogues/runtime-source-presentations.jsonl"

    result = validate_strategy_source_presentation(
        authorization,
        authorization_path=authorization_path,
        runtime_audit_path=runtime_audit_path,
        run_out=run_out,
    )

    assert result["status"] == "fail"
    assert "<conversation_id>/runtime-source-presentations.jsonl" in result["errors"][0]["message"]


@pytest.mark.parametrize("timestamp", [None, "2026-13-40T25:61:61Z"])
def test_source_presentation_requires_valid_utc_timestamp(tmp_path, timestamp: str | None) -> None:
    authorization, authorization_path, runtime_audit_path, run_out = _write_version_bound_source_presentation(tmp_path)
    reference = authorization["strategy_source_presentation"]
    event_path = tmp_path / reference["path"]
    event = json.loads(event_path.read_text(encoding="utf-8"))
    if timestamp is None:
        event.pop("timestamp")
    else:
        event["timestamp"] = timestamp
    _rewrite_source_presentation_event(authorization, tmp_path, event)

    result = validate_strategy_source_presentation(
        authorization,
        authorization_path=authorization_path,
        runtime_audit_path=runtime_audit_path,
        run_out=run_out,
    )

    assert result["status"] == "fail"
    assert "timestamp" in result["errors"][0]["message"]


def test_source_presentation_requires_documented_identity_fields(tmp_path) -> None:
    authorization, authorization_path, runtime_audit_path, run_out = _write_version_bound_source_presentation(tmp_path)
    reference = authorization["strategy_source_presentation"]
    event_path = tmp_path / reference["path"]
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event.pop("active_run")
    reference.pop("active_run")
    _rewrite_source_presentation_event(authorization, tmp_path, event)

    result = validate_strategy_source_presentation(
        authorization,
        authorization_path=authorization_path,
        runtime_audit_path=runtime_audit_path,
        run_out=run_out,
    )

    assert result["status"] == "fail"
    assert "active_run" in result["errors"][0]["message"]


class _EffectiveSpec:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.required_oxq_version = payload.get("required_oxq_version")

    def to_effective_dict(self) -> dict[str, Any]:
        return self._payload


def _flatten_effective_fields(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        fields: list[tuple[str, Any]] = []
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


def test_strict_confirmed_coverage_accepts_direct_user_evidence() -> None:
    spec = {"execution": {"initial_cash": 100000}}
    payload = _payload(
        [
            _confirmed(
                "execution.initial_cash",
                100000,
                "User: run the strategy with initial cash 100000.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_pass_audit_rejects_non_empty_blocking_findings() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["blocking_findings"] = [
        {
            "message": "Previously blocked question was resolved by user confirmation.",
            "resolution": "confirmed",
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "blocking_findings" for error in result["errors"])


def test_pass_audit_requires_confirmation_event_reference() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload.pop("confirmation_event")

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event" for error in result["errors"])


def test_confirmed_user_confirmation_requires_pass_status() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["status"] = "block"

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "status" and "must be pass" in error["message"] for error in result["errors"])


def test_state_machine_rejects_block_fail_pending_state() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["status"] = "block"
    payload["audit_conclusion"] = "fail"
    payload["user_confirmation_status"] = "pending"
    payload["spec_provenance_pass"] = False
    payload["spec_confirmation_table"] = None
    payload["blocking_findings"] = [{"message": "invalid state triple"}]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "state" for error in result["errors"])


def test_pass_audit_rejects_blocking_field_audit_row() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["field_audits"][0]["blocking"] = True

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_audits[0].blocking" for error in result["errors"])


def test_pass_audit_rejects_unresolved_field_audit_status() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["field_audits"][0]["status"] = "contradiction"

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_audits[0].status" for error in result["errors"])


def test_pass_audit_rejects_blocking_component_audit_row() -> None:
    payload = _payload([])
    payload["component_audits"] = [
        {
            "component_path": "portfolio.type",
            "component_type": "PortfolioOptimizer",
            "status": "catalog",
            "evidence": ["component found"],
            "blocking": True,
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "component_audits[0].blocking" for error in result["errors"])


def test_pass_audit_rejects_missing_component_audit_row() -> None:
    payload = _payload([])
    payload["component_audits"] = [
        {
            "component_path": "portfolio.type",
            "component_type": "PortfolioOptimizer",
            "status": "missing",
            "evidence": ["component not found"],
            "blocking": False,
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "component_audits[0].status" for error in result["errors"])


def test_pass_audit_rejects_non_canonical_component_audit_row() -> None:
    payload = _payload([])
    payload["component_audits"] = [
        {
            "component_path": "portfolio.type",
            "component_type": "PortfolioOptimizer",
            "status": "non_canonical",
            "evidence": ["component exists but is not canonical for this mapping"],
            "blocking": False,
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "component_audits[0].status" for error in result["errors"])


def test_pass_audit_rejects_unresolved_exception_lists() -> None:
    payload = _payload([_confirmed("execution.price_type", "open")])
    payload["agent_added_fields"] = [
        {
            "field_path": "execution.price_type",
            "message": "Agent expanded next_session_open.",
            "resolution": "confirmed",
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "agent_added_fields" for error in result["errors"])


def test_strict_confirmed_coverage_rejects_stale_default_duplicate_row() -> None:
    spec = {"execution": {"initial_cash": 100000}}
    payload = _payload(
        [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000,
                "status": "default",
                "material_category": "execution_assumption",
                "evidence": ["Default checklist row before user confirmation."],
                "blocking": False,
            },
            _confirmed("execution.initial_cash", 100000),
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("conflicting non-confirmed audit rows" in error["message"] for error in result["errors"])


def test_strict_confirmed_coverage_rejects_framework_default_evidence() -> None:
    spec = {"market": {"region": "us"}}
    payload = _payload(
        [
            _confirmed(
                "market.region",
                "us",
                "Framework default; calendar XSHG is configured explicitly.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("evidence denies user confirmation" in error["message"] for error in result["errors"])


def test_strict_confirmed_coverage_rejects_legacy_brand_default_evidence() -> None:
    spec = {"market": {"region": "us"}}
    legacy_default = "Open" + "XQuant default; configured explicitly."
    payload = _payload(
        [
            _confirmed(
                "market.region",
                "us",
                legacy_default,
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("evidence denies user confirmation" in error["message"] for error in result["errors"])


def test_strict_confirmed_coverage_rejects_effective_default_coverage_evidence() -> None:
    spec = {"cost": {"buy_fee_rate": None}}
    payload = _payload(
        [
            _confirmed(
                "cost.buy_fee_rate",
                None,
                "Effective StrategySpec default value. Inherits from fee_rate when buy_fee_rate is absent from YAML. "
                "Documented for full SPEC coverage.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("evidence denies user confirmation" in error["message"] for error in result["errors"])


def test_strict_confirmed_coverage_accepts_user_confirmed_default_checklist() -> None:
    spec = {"metrics": {"risk_free_rate": 0.0}}
    payload = _payload(
        [
            _confirmed(
                "metrics.risk_free_rate",
                0.0,
                "User confirmed the Default Confirmation Checklist metrics group.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_spec_hash_must_match_strategy_spec_compute_hash_when_spec_is_supplied() -> None:
    spec = StrategySpec.from_dict(
        {
            "schema_version": "0.1",
            "strategy_id": "hash_gate_test",
            "name": "Hash Gate Test",
            "universe": {"symbols": ["SPY"]},
            "benchmark": {"symbols": ["SPY"]},
            "execution": {"lot_size_config": {"default": 1}},
            "cost": {"fee_rate": 0.001, "slippage_rate": 0.001},
            "validation": {"test_period": ["2022-01-01", "2023-01-01"], "required_oos": False},
        }
    )
    payload = _payload([])
    payload["spec_hash"] = "sha256:" + "f" * 16

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_hash" and "strategy spec hash" in error["message"] for error in result["errors"])

    payload["spec_hash"] = spec.compute_hash()

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_catalog_hash_must_match_component_catalog_payload_when_catalog_is_supplied() -> None:
    catalog = {"components": [], "recipes": []}
    catalog["catalog_hash"] = _catalog_hash(catalog)
    payload = _payload([])
    payload["catalog_hash"] = "sha256:" + "9" * 16

    result = validate_spec_audit(payload, component_catalog=catalog)

    assert result["status"] == "fail"
    assert any(error["path"] == "catalog_hash" and "component catalog hash" in error["message"] for error in result["errors"])

    payload["catalog_hash"] = catalog["catalog_hash"]

    result = validate_spec_audit(payload, component_catalog=catalog)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_pass_audit_rejects_blocking_unsupported_mappings() -> None:
    payload = _payload([])
    payload["unsupported_mappings"] = [
        {
            "source_field": "portfolio.constraints.min_position_value",
            "requested_semantic": "minimum notional position size",
            "reason": "current strategy_spec.yaml parses the field but cannot execute it",
            "disposition": "blocked",
            "blocking": True,
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "unsupported_mappings[0].blocking" for error in result["errors"])


def test_pass_audit_accepts_explicit_empty_unsupported_mappings() -> None:
    payload = _payload([])
    payload["unsupported_mappings"] = []

    result = validate_spec_audit(payload)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_blocked_audit_may_omit_spec_confirmation_table() -> None:
    payload = _payload([])
    payload["status"] = "block"
    payload["audit_conclusion"] = "blocked"
    payload["user_confirmation_status"] = "pending"
    payload["spec_provenance_pass"] = False
    payload["blocking_findings"] = [{"message": "execution.initial_cash mistranslated"}]
    payload.pop("spec_confirmation_table")

    result = validate_spec_audit(payload)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_blocked_audit_rejects_non_null_spec_confirmation_table() -> None:
    payload = _payload([])
    payload["status"] = "block"
    payload["audit_conclusion"] = "blocked"
    payload["user_confirmation_status"] = "pending"
    payload["spec_provenance_pass"] = False
    payload["blocking_findings"] = [{"message": "execution.initial_cash mistranslated"}]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_confirmation_table" for error in result["errors"])


def test_all_pass_pending_audit_requires_spec_confirmation_table() -> None:
    payload = _payload([])
    payload["status"] = "block"
    payload["audit_conclusion"] = "all_pass"
    payload["user_confirmation_status"] = "pending"
    payload.pop("spec_confirmation_table")

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_confirmation_table" for error in result["errors"])


def test_all_pass_pending_audit_rejects_blocking_unsupported_mapping() -> None:
    payload = _payload([])
    payload["status"] = "block"
    payload["audit_conclusion"] = "all_pass"
    payload["user_confirmation_status"] = "pending"
    payload["unsupported_mappings"] = [
        {
            "source_field": "portfolio.cross_sectional_winsorization",
            "requested_semantic": "clip cross-sectional scores before rank",
            "reason": "No executable target exists.",
            "disposition": "blocked",
            "blocking": True,
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "unsupported_mappings" for error in result["errors"])


def test_spec_audit_file_rejects_non_object_json_without_crashing(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    audit_path.write_text("[]", encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "$" for error in result["errors"])


def test_strict_spec_audit_file_rejects_missing_confirmation_table(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": "sha256:" + "1" * 16,
        "hash_type": "sha256",
    }
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_confirmation_table.path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_stale_confirmation_table_hash(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text("| Field | Value |\n| --- | --- |\n", encoding="utf-8")
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": "sha256:" + "1" * 16,
        "hash_type": "sha256",
    }
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_confirmation_table.hash" for error in result["errors"])


def test_strict_spec_audit_file_accepts_bound_confirmation_event(tmp_path) -> None:
    spec = {"execution": {"initial_cash": 100000}}
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text(_spec_confirmation_table(spec), encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_strict_spec_audit_file_accepts_event_inside_configured_conversations_root(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    event_path = tmp_path / "evidence/dialogues/demo/confirmations.jsonl"
    audit_path = _write_workspace_bound_spec_audit(
        tmp_path,
        conversations_dir="evidence/dialogues",
        event_path=event_path,
        event_reference="evidence/dialogues/demo/confirmations.jsonl",
    )

    result = validate_spec_audit_file(
        audit_path,
        spec={"execution": {"initial_cash": 100000}},
        verify_confirmation_table=True,
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_strict_spec_audit_file_uses_default_conversations_root_when_setting_absent(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    audit_path = _write_workspace_bound_spec_audit(
        tmp_path,
        conversations_dir=None,
        event_path=tmp_path / "conversations/demo/confirmations.jsonl",
        event_reference="conversations/demo/confirmations.jsonl",
    )

    result = validate_spec_audit_file(
        audit_path,
        spec={"execution": {"initial_cash": 100000}},
        verify_confirmation_table=True,
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_strict_spec_audit_file_rejects_event_outside_configured_conversations_root(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    audit_path = _write_workspace_bound_spec_audit(
        tmp_path,
        conversations_dir="evidence/dialogues",
        event_path=tmp_path / "confirmations.jsonl",
        event_reference="confirmations.jsonl",
    )

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.path" for error in result["errors"])


@pytest.mark.parametrize("configured_root", ["/tmp/dialogues", "../dialogues"])
def test_strict_spec_audit_file_rejects_unsafe_configured_conversations_root(
    tmp_path,
    monkeypatch,
    configured_root: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    audit_path = _write_workspace_bound_spec_audit(
        tmp_path,
        conversations_dir="evidence/dialogues",
        event_path=tmp_path / "evidence/dialogues/demo/confirmations.jsonl",
        event_reference="evidence/dialogues/demo/confirmations.jsonl",
    )
    (tmp_path / ".open-xquant/workspace.yaml").write_text(
        f"schema_version: 1\npaths:\n  conversations_dir: {configured_root}\n",
        encoding="utf-8",
    )

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_confirmation_event_path_traversal(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    audit_path = _write_workspace_bound_spec_audit(
        tmp_path,
        conversations_dir="evidence/dialogues",
        event_path=tmp_path / "confirmations.jsonl",
        event_reference="evidence/dialogues/../../confirmations.jsonl",
    )

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_absolute_confirmation_event_path(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    event_path = tmp_path / "evidence/dialogues/demo/confirmations.jsonl"
    audit_path = _write_workspace_bound_spec_audit(
        tmp_path,
        conversations_dir="evidence/dialogues",
        event_path=event_path,
        event_reference=str(event_path),
    )

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_confirmation_event_symlink_escape(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    event_path = outside / "demo/confirmations.jsonl"
    audit_path = _write_workspace_bound_spec_audit(
        tmp_path,
        conversations_dir="evidence/dialogues",
        event_path=event_path,
        event_reference="evidence/dialogues/linked/demo/confirmations.jsonl",
    )
    link = tmp_path / "evidence/dialogues/linked"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_rejected_confirmation_event_with_valid_hashes(tmp_path) -> None:
    spec = {"execution": {"initial_cash": 100000}}
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text(_spec_confirmation_table(spec), encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
        decision="rejected",
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.decision" for error in result["errors"])


def test_strict_spec_audit_file_accepts_faithful_archived_run_copy(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = {"execution": {"initial_cash": 100000}}
    audit_dir = tmp_path / "versions/v001/06_spec_audit"
    run_dir = tmp_path / "versions/v001/09_backtests/run1"
    event_dir = tmp_path / "conversations/demo"
    audit_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    event_dir.mkdir(parents=True)

    table_path = audit_dir / "spec_confirmation_table.md"
    table_path.write_text(_spec_confirmation_table(spec), encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["spec_confirmation_table"] = {
        "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        event_dir / "confirmations.jsonl",
        artifact_path="versions/v001/06_spec_audit/spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_path="versions/v001/06_spec_audit/spec_audit.json",
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    original_audit_path = audit_dir / "spec_audit.json"
    original_audit_path.write_text(json.dumps(payload), encoding="utf-8")
    archived_audit_path = run_dir / "spec_audit.json"
    archived_audit_path.write_text(original_audit_path.read_text(encoding="utf-8"), encoding="utf-8")

    original_result = validate_spec_audit_file(original_audit_path, spec=spec, verify_confirmation_table=True)
    archived_result = validate_spec_audit_file(archived_audit_path, spec=spec, verify_confirmation_table=True)

    assert original_result["status"] == "pass"
    assert archived_result["status"] == "pass"
    assert archived_result["errors"] == []


def test_strict_spec_audit_file_rejects_header_only_confirmation_table(tmp_path) -> None:
    spec = {"execution": {"initial_cash": 100000}}
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text("| Field | Confirmed Value |\n| --- | --- |\n", encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_confirmation_table.content" for error in result["errors"])


def test_strict_spec_audit_file_rejects_duplicate_confirmation_table_field(tmp_path) -> None:
    spec = {"execution": {"initial_cash": 100000}}
    table_text = _spec_confirmation_table(spec).replace(
        "| execution | execution.initial_cash | 100000 | User confirmed full SPEC table | confirmed | material |\n",
        "| execution | execution.initial_cash | 1 | User confirmed full SPEC table | confirmed | material |\n"
        "| execution | execution.initial_cash | 100000 | User confirmed full SPEC table | confirmed | material |\n",
    )
    audit_path = _write_bound_spec_audit_file(tmp_path, spec, table_text)

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any("duplicate effective StrategySpec field row" in error["message"] for error in result["errors"])


def test_strict_spec_audit_file_rejects_unknown_confirmation_table_field(tmp_path) -> None:
    spec = {"execution": {"initial_cash": 100000}}
    table_text = _spec_confirmation_table(spec) + (
        "| portfolio | portfolio.initial_cash | 100000 | User confirmed full SPEC table | confirmed | material |\n"
    )
    audit_path = _write_bound_spec_audit_file(tmp_path, spec, table_text)

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any("unknown field path" in error["message"] for error in result["errors"])


def test_strict_spec_audit_file_rejects_unconfirmed_confirmation_table_row(tmp_path) -> None:
    spec = {"execution": {"initial_cash": 100000}}
    table_text = _spec_confirmation_table(spec).replace(
        "| execution | execution.initial_cash | 100000 | User confirmed full SPEC table | confirmed | material |",
        "| execution | execution.initial_cash | 100000 | User confirmed full SPEC table | unconfirmed | material |",
    )
    audit_path = _write_bound_spec_audit_file(tmp_path, spec, table_text)

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"].endswith(".audit_status") for error in result["errors"])


def test_strict_spec_audit_file_accepts_escaped_pipe_in_confirmation_table_value(tmp_path) -> None:
    spec = {"metadata": {"hypothesis": "alpha | beta"}}
    table_text = "\n".join(
        [
            "| Section | Field path | Spec value | Source | Audit status | Impact |",
            "| --- | --- | --- | --- | --- | --- |",
            '| metadata | metadata.hypothesis | "alpha \\| beta" | User confirmed full SPEC table | confirmed | material |',
            "",
        ]
    )
    audit_path = _write_bound_spec_audit_file(tmp_path, spec, table_text)

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_strict_spec_audit_file_rejects_non_utf8_confirmation_table(tmp_path) -> None:
    spec = {"execution": {"initial_cash": 100000}}
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_bytes(b"\xff\xfe\x00")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, spec=spec, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any("UTF-8 Markdown" in error["message"] for error in result["errors"])


def test_strict_spec_audit_file_rejects_directory_confirmation_table_without_crashing(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.mkdir()
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": table_path.name,
        "hash": "sha256:" + "1" * 16,
        "hash_type": "sha256",
    }
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_confirmation_table.path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_non_utf8_confirmation_event_without_crashing(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text("| Field | Value |\n| --- | --- |\n", encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    event_path = tmp_path / "conversations/demo/confirmations.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(b"\xff\xfe\x00")
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": table_path.name,
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = {
        **payload["confirmation_event"],
        "path": "conversations/demo/confirmations.jsonl",
        "artifact_path": table_path.name,
        "artifact_hash": table_hash,
    }
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_confirmation_event_line_mismatch(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text("| Field | Confirmed Value |\n| --- | --- |\n", encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        event_artifact_hash="sha256:" + "a" * 16,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.artifact_hash" for error in result["errors"])


def test_strict_spec_audit_file_rejects_reused_confirmation_event_from_other_audit(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text("| Field | Confirmed Value |\n| --- | --- |\n", encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_path="other/spec_audit.json",
        spec_audit_hash=_pre_confirmation_spec_audit_hash(payload),
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.spec_audit_path" for error in result["errors"])


def test_strict_spec_audit_file_rejects_fake_pre_confirmation_audit_hash(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    table_path = tmp_path / "spec_confirmation_table.md"
    table_path.write_text("| Field | Confirmed Value |\n| --- | --- |\n", encoding="utf-8")
    table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    payload = _payload([])
    payload["spec_confirmation_table"] = {
        "path": "spec_confirmation_table.md",
        "hash": table_hash,
        "hash_type": "sha256",
    }
    payload["confirmation_event"] = _write_confirmation_event_line(
        tmp_path / "conversations/demo/confirmations.jsonl",
        artifact_path="spec_confirmation_table.md",
        artifact_hash=table_hash,
        spec_audit_hash="sha256:" + "a" * 16,
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_spec_audit_file(audit_path, verify_confirmation_table=True)

    assert result["status"] == "fail"
    assert any(error["path"] == "confirmation_event.spec_audit_hash" for error in result["errors"])


def test_spec_audit_requires_idea_and_mapping_contract_fields() -> None:
    payload = _payload([])
    for field in (
        "strategy_idea_brief",
        "strategy_idea_audit",
        "strategy_idea_brief_hash",
        "strategy_idea_audit_hash",
        "unsupported_mappings",
    ):
        payload.pop(field)

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    paths = {error["path"] for error in result["errors"]}
    assert "strategy_idea_brief" in paths
    assert "strategy_idea_audit" in paths
    assert "strategy_idea_brief_hash" in paths
    assert "strategy_idea_audit_hash" in paths
    assert "unsupported_mappings" in paths


def test_field_audits_require_material_category() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["field_audits"][0].pop("material_category")

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_audits[0].material_category" for error in result["errors"])


def test_spec_supplied_rejects_yaml_only_field_audit_path() -> None:
    spec = {"execution": {"initial_cash": 100000.0}}
    payload = _payload(
        [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000.0,
                "status": "contradiction",
                "material_category": "execution_assumption",
                "evidence": ["Effective value differs from user-confirmed source value."],
                "blocking": True,
            },
            {
                "field_path": "portfolio.initial_cash",
                "spec_value": 1000000,
                "status": "contradiction",
                "material_category": "execution_assumption",
                "evidence": ["YAML-only misplaced source path; not effective StrategySpec semantics."],
                "blocking": True,
            },
        ]
    )
    payload["status"] = "block"
    payload["audit_conclusion"] = "blocked"
    payload["user_confirmation_status"] = "rejected"
    payload["spec_provenance_pass"] = False
    payload["spec_confirmation_table"] = None
    payload["blocking_findings"] = [{"message": "initial_cash is mapped to a non-operative YAML path"}]

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_audits[portfolio.initial_cash]" for error in result["errors"])
    assert any("effective StrategySpec field" in error["message"] for error in result["errors"])


def test_yaml_only_source_path_belongs_in_contradiction_not_field_audits() -> None:
    spec = {"execution": {"initial_cash": 100000.0}}
    payload = _payload(
        [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000.0,
                "status": "contradiction",
                "material_category": "execution_assumption",
                "evidence": ["User confirmed 1000000, but effective value remains default 100000.0."],
                "blocking": True,
            }
        ]
    )
    payload["status"] = "block"
    payload["audit_conclusion"] = "blocked"
    payload["user_confirmation_status"] = "rejected"
    payload["spec_provenance_pass"] = False
    payload["spec_confirmation_table"] = None
    payload["contradictions"] = [
        {
            "message": "initial_cash was placed under a non-operative YAML path",
            "effective_field_path": "execution.initial_cash",
            "source_yaml_path": "portfolio.initial_cash",
            "expected_value": 1000000,
            "effective_value": 100000.0,
            "builder_required_fix": "move the value to execution.initial_cash and remove portfolio.initial_cash",
        }
    ]
    payload["blocking_findings"] = [{"message": "return to build for initial_cash mapping fix"}]

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_strict_confirmed_coverage_flattens_structured_lists_like_spec_fields() -> None:
    spec = {
        "signal": {
            "rules": {
                "foo": {
                    "params": {
                        "conditions": [
                            {"column": "ret_20", "threshold": 0},
                            {"column": "vol_20", "threshold": 0.2},
                        ]
                    }
                }
            }
        }
    }
    payload = _payload(
        [
            _confirmed("signal.rules.foo.params.conditions[0].column", "ret_20"),
            _confirmed("signal.rules.foo.params.conditions[0].threshold", 0),
            _confirmed("signal.rules.foo.params.conditions[1].column", "vol_20"),
            _confirmed("signal.rules.foo.params.conditions[1].threshold", 0.2),
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "pass"
    assert result["errors"] == []


def _runtime_material_values(number: int | float = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    effective_spec = {
        field: {"nested": {"value": number}}
        for field in (
            "market",
            "universe",
            "data",
            "signal",
            "portfolio",
            "execution",
            "cost",
            "benchmark",
            "validation",
            "metrics",
        )
    }
    effective_spec["required_oxq_version"] = "0.1.0"
    compiled_plan = {
        runtime_path: json.loads(json.dumps(effective_spec[field_path]))
        for field_path, runtime_path in (
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
    }
    compiled_plan["open_xquant_version"] = "0.1.0"
    return effective_spec, compiled_plan


def test_strict_runtime_audit_rejects_required_version_mismatch_even_with_forged_matching_row() -> None:
    effective_spec, compiled_plan = _runtime_material_values()
    effective_spec["required_oxq_version"] = "999.0.0"
    payload = _runtime_payload(effective_spec, compiled_plan)
    version_row = next(
        row for row in payload["material_field_audits"] if row["field_path"] == "required_oxq_version"
    )
    version_row["spec_value"] = "0.1.0"
    version_row["runtime_value"] = "0.1.0"

    result = validate_runtime_audit(
        payload,
        spec=_EffectiveSpec(effective_spec),
        compiled_plan=compiled_plan,
        require_material_coverage=True,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == "required_oxq_version" for error in result["errors"])
    assert any(error["path"].endswith(".spec_value") for error in result["errors"])


@pytest.mark.parametrize(
    ("required_version", "compiled_version", "error_path"),
    [
        ("", "0.1.0", "required_oxq_version"),
        (None, "0.1.0", "required_oxq_version"),
        ("0.1.0", "", "open_xquant_version"),
        ("0.1.0", None, "open_xquant_version"),
    ],
)
def test_strict_runtime_audit_requires_non_empty_version_strings(
    required_version: object,
    compiled_version: object,
    error_path: str,
) -> None:
    effective_spec, compiled_plan = _runtime_material_values()
    effective_spec["required_oxq_version"] = required_version
    compiled_plan["open_xquant_version"] = compiled_version
    payload = _runtime_payload(effective_spec, compiled_plan)

    result = validate_runtime_audit(
        payload,
        spec=_EffectiveSpec(effective_spec),
        compiled_plan=compiled_plan,
        require_material_coverage=True,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == error_path for error in result["errors"])


@pytest.mark.parametrize(
    ("missing_field", "error_path"),
    [
        ("required_oxq_version", "required_oxq_version"),
        ("open_xquant_version", "open_xquant_version"),
    ],
)
def test_strict_runtime_audit_rejects_missing_actual_version_fields(
    missing_field: str,
    error_path: str,
) -> None:
    effective_spec, compiled_plan = _runtime_material_values()
    payload = _runtime_payload(effective_spec, compiled_plan)
    if missing_field == "required_oxq_version":
        effective_spec.pop(missing_field)
    else:
        compiled_plan.pop(missing_field)

    result = validate_runtime_audit(
        payload,
        spec=_EffectiveSpec(effective_spec),
        compiled_plan=compiled_plan,
        require_material_coverage=True,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == error_path for error in result["errors"])


@pytest.mark.parametrize("number", [1, 1.0])
@pytest.mark.parametrize("value_field", ["spec_value", "runtime_value"])
def test_strict_runtime_audit_rejects_bool_for_numeric_material_value(
    number: int | float,
    value_field: str,
) -> None:
    effective_spec, compiled_plan = _runtime_material_values(number)
    payload = _runtime_payload(effective_spec, compiled_plan)
    market_row = next(row for row in payload["material_field_audits"] if row["field_path"] == "market")
    market_row[value_field]["nested"]["value"] = True

    result = validate_runtime_audit(
        payload,
        spec=_EffectiveSpec(effective_spec),
        compiled_plan=compiled_plan,
        require_material_coverage=True,
    )

    assert result["status"] == "fail"
    assert any(error["path"].endswith(f".{value_field}") for error in result["errors"])


def test_strict_runtime_audit_preserves_json_number_equality() -> None:
    effective_spec, compiled_plan = _runtime_material_values(1)
    payload = _runtime_payload(effective_spec, compiled_plan)
    market_row = next(row for row in payload["material_field_audits"] if row["field_path"] == "market")
    market_row["spec_value"]["nested"]["value"] = 1.0
    market_row["runtime_value"]["nested"]["value"] = 1.0

    result = validate_runtime_audit(
        payload,
        spec=_EffectiveSpec(effective_spec),
        compiled_plan=compiled_plan,
        require_material_coverage=True,
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_strict_runtime_audit_accepts_arbitrarily_large_json_integer() -> None:
    effective_spec, compiled_plan = _runtime_material_values(10**1000)
    payload = _runtime_payload(effective_spec, compiled_plan)

    result = validate_runtime_audit(
        payload,
        spec=_EffectiveSpec(effective_spec),
        compiled_plan=compiled_plan,
        require_material_coverage=True,
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("value_field", ["spec_value", "runtime_value"])
def test_runtime_audit_rejects_nested_non_finite_material_value(
    non_finite: float,
    value_field: str,
) -> None:
    effective_spec, compiled_plan = _runtime_material_values()
    payload = _runtime_payload(effective_spec, compiled_plan)
    market_index, market_row = next(
        (index, row)
        for index, row in enumerate(payload["material_field_audits"])
        if row["field_path"] == "market"
    )
    market_row[value_field] = {"nested": [0, {"value": non_finite}]}

    result = validate_runtime_audit(payload)

    assert result["status"] == "fail"
    assert any(
        error["path"].startswith(f"material_field_audits[{market_index}].{value_field}")
        and "finite JSON" in error["message"]
        for error in result["errors"]
    )


def test_strict_runtime_audit_validation_does_not_mutate_inputs() -> None:
    effective_spec, compiled_plan = _runtime_material_values()
    payload = _runtime_payload(effective_spec, compiled_plan)
    original_payload = json.loads(json.dumps(payload))
    original_compiled_plan = json.loads(json.dumps(compiled_plan))

    result = validate_runtime_audit(
        payload,
        spec=_EffectiveSpec(effective_spec),
        compiled_plan=compiled_plan,
        require_material_coverage=True,
    )

    assert result["status"] == "pass"
    assert payload == original_payload
    assert compiled_plan == original_compiled_plan
