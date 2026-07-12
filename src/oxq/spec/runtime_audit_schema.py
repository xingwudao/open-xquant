"""Deterministic schema checks for Agent-authored runtime_audit.json."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from oxq.spec.schema import StrategySpec

RUNTIME_AUDIT_SCHEMA_VERSION = 2
VERSION_RUNTIME_PATH = ("required_oxq_version", "open_xquant_version")

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "runtime_semantics_pass",
    "strategy_source_path",
    "strategy_source_hash",
    "spec_hash",
    "spec_audit_hash",
    "compiled_plan_hash",
    "compiled_plan_path",
    "material_field_audits",
    "blocking_findings",
}

_ALLOWED_STATUS = {"pass", "block", "fail"}
_ALLOWED_FIELD_STATUS = {"preserved", "missing", "mismatch", "not_applicable"}
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{16,64}$")
_FULL_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MATERIAL_RUNTIME_PATHS = (
    VERSION_RUNTIME_PATH,
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


def validate_runtime_audit_file(
    path: str | Path,
    *,
    spec: StrategySpec | None = None,
    compiled_plan: dict[str, Any] | None = None,
    require_material_coverage: bool = False,
    require_version_coverage: bool = False,
) -> dict[str, Any]:
    """Validate a runtime_audit.json file and return deterministic findings."""
    audit_path = Path(path)
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except (OSError, UnicodeError) as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    return validate_runtime_audit(
        payload,
        spec=spec,
        compiled_plan=compiled_plan,
        require_material_coverage=require_material_coverage,
        require_version_coverage=require_version_coverage,
    )


def validate_runtime_audit(
    payload: Any,
    *,
    spec: StrategySpec | None = None,
    compiled_plan: dict[str, Any] | None = None,
    require_material_coverage: bool = False,
    require_version_coverage: bool = False,
) -> dict[str, Any]:
    """Validate a parsed runtime audit payload."""
    errors: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return _result("fail", [{"path": "$", "message": "runtime_audit must be a JSON object"}])

    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS.difference(payload))
    for field in missing:
        errors.append({"path": field, "message": "missing required field"})

    status = payload.get("status")
    if not isinstance(status, str) or status not in _ALLOWED_STATUS:
        errors.append({"path": "status", "message": f"must be one of {sorted(_ALLOWED_STATUS)}"})

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version != RUNTIME_AUDIT_SCHEMA_VERSION:
        errors.append({"path": "schema_version", "message": f"must be {RUNTIME_AUDIT_SCHEMA_VERSION}"})

    if "runtime_semantics_pass" in payload and not isinstance(payload["runtime_semantics_pass"], bool):
        errors.append({"path": "runtime_semantics_pass", "message": "must be a boolean"})
    if status == "pass" and payload.get("runtime_semantics_pass") is not True:
        errors.append({"path": "runtime_semantics_pass", "message": "must be true when status is pass"})
    strategy_source_path = payload.get("strategy_source_path")
    if not isinstance(strategy_source_path, str) or not strategy_source_path:
        errors.append({"path": "strategy_source_path", "message": "must be a non-empty string"})
    strategy_source_hash = payload.get("strategy_source_hash")
    if not isinstance(strategy_source_hash, str) or not _FULL_HASH_RE.fullmatch(strategy_source_hash):
        errors.append({"path": "strategy_source_hash", "message": "must be a full sha256:<64 hex> hash"})

    for field in ("spec_hash", "spec_audit_hash", "compiled_plan_hash"):
        value = payload.get(field)
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            errors.append({"path": field, "message": "must be a sha256:<hex> hash"})

    if "component_bundle_hashes" in payload:
        hashes = payload["component_bundle_hashes"]
        if not isinstance(hashes, list):
            errors.append({"path": "component_bundle_hashes", "message": "must be a list"})
        else:
            for index, value in enumerate(hashes):
                if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
                    errors.append({"path": f"component_bundle_hashes[{index}]", "message": "must be a sha256:<hex> hash"})

    compiled_plan_path = payload.get("compiled_plan_path")
    if not isinstance(compiled_plan_path, str) or not compiled_plan_path:
        errors.append({"path": "compiled_plan_path", "message": "must be a non-empty string"})

    for field in ("material_field_audits", "blocking_findings"):
        if field in payload and not isinstance(payload[field], list):
            errors.append({"path": field, "message": "must be a list"})

    field_rows = payload.get("material_field_audits", [])
    if isinstance(field_rows, list):
        seen_field_paths: dict[str, int] = {}
        for index, item in enumerate(field_rows):
            if not isinstance(item, dict):
                errors.append({"path": f"material_field_audits[{index}]", "message": "must be an object"})
                continue
            field_path = _require_str(item, f"material_field_audits[{index}]", "field_path", errors)
            if field_path:
                if field_path in seen_field_paths:
                    errors.append(
                        {
                            "path": f"material_field_audits[{index}].field_path",
                            "message": (
                                "duplicate material field path; first seen at "
                                f"material_field_audits[{seen_field_paths[field_path]}]"
                            ),
                        }
                    )
                else:
                    seen_field_paths[field_path] = index
            if "spec_value" not in item:
                errors.append({"path": f"material_field_audits[{index}].spec_value", "message": "missing required field"})
            else:
                _validate_json_material_value(
                    item["spec_value"],
                    f"material_field_audits[{index}].spec_value",
                    errors,
                )
            _require_str(item, f"material_field_audits[{index}]", "runtime_path", errors)
            if "runtime_value" not in item:
                errors.append(
                    {"path": f"material_field_audits[{index}].runtime_value", "message": "missing required field"}
                )
            else:
                _validate_json_material_value(
                    item["runtime_value"],
                    f"material_field_audits[{index}].runtime_value",
                    errors,
                )
            _require_enum(item, f"material_field_audits[{index}]", "status", _ALLOWED_FIELD_STATUS, errors)
            if "evidence" not in item or not isinstance(item["evidence"], list):
                errors.append({"path": f"material_field_audits[{index}].evidence", "message": "must be a list"})
            if "blocking" in item and not isinstance(item["blocking"], bool):
                errors.append({"path": f"material_field_audits[{index}].blocking", "message": "must be a boolean"})
            if status == "pass" and item.get("blocking") is True:
                errors.append(
                    {
                        "path": f"material_field_audits[{index}].blocking",
                        "message": "blocking material field row cannot pass runtime audit",
                    }
                )
            if status == "pass" and item.get("status") in {"missing", "mismatch"}:
                errors.append(
                    {
                        "path": f"material_field_audits[{index}].status",
                        "message": "unresolved material field row cannot pass runtime audit",
                    }
                )

        if require_material_coverage:
            _validate_material_coverage(payload, field_rows, spec, compiled_plan, errors)
        elif require_version_coverage:
            _validate_material_coverage(
                payload,
                field_rows,
                spec,
                compiled_plan,
                errors,
                material_paths=(VERSION_RUNTIME_PATH,),
            )

    blocking_findings = payload.get("blocking_findings", [])
    if isinstance(blocking_findings, list):
        for index, item in enumerate(blocking_findings):
            if not isinstance(item, dict):
                errors.append({"path": f"blocking_findings[{index}]", "message": "must be an object"})
            elif "message" not in item or not isinstance(item["message"], str):
                errors.append({"path": f"blocking_findings[{index}].message", "message": "must be a string"})
        if status == "pass" and blocking_findings:
            errors.append({"path": "blocking_findings", "message": "must be empty when status is pass"})

    return _result("fail" if errors else "pass", errors)


def _validate_material_coverage(
    payload: dict[str, Any],
    field_rows: list[Any],
    spec: StrategySpec | None,
    compiled_plan: dict[str, Any] | None,
    errors: list[dict[str, str]],
    *,
    material_paths: tuple[tuple[str, str], ...] = MATERIAL_RUNTIME_PATHS,
) -> None:
    if spec is None:
        errors.append({"path": "material_field_audits", "message": "strict coverage requires a StrategySpec"})
        return
    if not isinstance(compiled_plan, dict):
        errors.append({"path": "material_field_audits", "message": "strict coverage requires a compiled plan"})
        return

    spec_payload = spec.to_effective_dict()
    required_version = _normalize_version(spec_payload.get("required_oxq_version"))
    compiled_version = _normalize_version(compiled_plan.get("open_xquant_version"))
    if required_version is None:
        errors.append({"path": "required_oxq_version", "message": "must be a non-empty string"})
    if compiled_version is None:
        errors.append({"path": "open_xquant_version", "message": "must be a non-empty string"})
    if required_version is not None and compiled_version is not None and required_version != compiled_version:
        errors.append(
            {
                "path": "required_oxq_version",
                "message": (
                    "must exactly match compiled_plan.open_xquant_version after normalization: "
                    f"required_oxq_version={required_version!r}, open_xquant_version={compiled_version!r}"
                ),
            }
        )

    rows_by_field: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(field_rows):
        if isinstance(row, dict) and isinstance(row.get("field_path"), str):
            rows_by_field.setdefault(row["field_path"], []).append((index, row))

    for field_path, runtime_path in material_paths:
        matches = rows_by_field.get(field_path, [])
        if not matches:
            errors.append(
                {
                    "path": "material_field_audits",
                    "message": f"missing material field audit row for {field_path}",
                }
            )
            continue
        if len(matches) != 1:
            continue
        index, row = matches[0]
        prefix = f"material_field_audits[{index}]"
        if not _json_values_equal(row.get("spec_value"), spec_payload.get(field_path)):
            errors.append({"path": f"{prefix}.spec_value", "message": f"does not match SPEC field {field_path}"})
        if row.get("runtime_path") != runtime_path:
            errors.append({"path": f"{prefix}.runtime_path", "message": f"must be {runtime_path}"})
        if not _json_values_equal(row.get("runtime_value"), compiled_plan.get(runtime_path)):
            errors.append(
                {"path": f"{prefix}.runtime_value", "message": f"does not match compiled runtime field {runtime_path}"}
            )
        if payload.get("status") == "pass" and row.get("status") != "preserved":
            errors.append({"path": f"{prefix}.status", "message": "formal pass requires preserved material fields"})
        if payload.get("status") == "pass" and row.get("blocking") is not False:
            errors.append({"path": f"{prefix}.blocking", "message": "formal pass requires blocking=false"})


def _normalize_version(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _validate_json_material_value(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append({"path": path, "message": "must contain only finite JSON numeric values"})
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_material_value(item, f"{path}[{index}]", errors)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append({"path": path, "message": "must contain only JSON object keys and values"})
                continue
            _validate_json_material_value(item, f"{path}.{key}", errors)
        return
    errors.append({"path": path, "message": "must contain only finite JSON values"})


def _json_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return False
        left_is_finite = not isinstance(left, float) or math.isfinite(left)
        right_is_finite = not isinstance(right, float) or math.isfinite(right)
        return left_is_finite and right_is_finite and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_values_equal(left_item, right_item) for left_item, right_item in zip(left, right))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_json_values_equal(left[key], right[key]) for key in left)
        )
    return False


def _require_str(item: dict[str, Any], prefix: str, field: str, errors: list[dict[str, str]]) -> str:
    if field not in item or not isinstance(item[field], str):
        errors.append({"path": f"{prefix}.{field}", "message": "must be a string"})
        return ""
    return item[field]


def _require_enum(
    item: dict[str, Any], prefix: str, field: str, allowed: set[str], errors: list[dict[str, str]]
) -> None:
    value = item.get(field)
    if not isinstance(value, str) or value not in allowed:
        errors.append({"path": f"{prefix}.{field}", "message": f"must be one of {sorted(allowed)}"})


def _result(status: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    return {"status": status, "errors": errors}


def validate_strategy_source_presentation(
    authorization: Any,
    *,
    authorization_path: str | Path,
    runtime_audit_path: str | Path,
    run_out: str | Path,
) -> dict[str, Any]:
    """Validate coordinator-owned durable evidence for the exact generated source."""
    try:
        _require_strategy_source_presentation(
            authorization,
            authorization_path=Path(authorization_path),
            runtime_audit_path=Path(runtime_audit_path),
            run_out=Path(run_out),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        return _result("fail", [{"path": "strategy_source_presentation", "message": str(exc)}])
    return _result("pass", [])


def _require_strategy_source_presentation(
    authorization: Any,
    *,
    authorization_path: Path,
    runtime_audit_path: Path,
    run_out: Path,
) -> None:
    if not isinstance(authorization, dict):
        raise ValueError("backtest authorization must be an object")
    reference = authorization.get("strategy_source_presentation")
    if not isinstance(reference, dict):
        raise ValueError("strategy_source_presentation must be a durable event reference")

    runtime_audit = _read_json_object(runtime_audit_path, "runtime_audit.json")
    source_path_raw = _required_text(runtime_audit, "strategy_source_path", "runtime audit")
    source_hash = _required_full_hash(runtime_audit, "strategy_source_hash", "runtime audit")
    compiled_plan_hash = _required_hash(runtime_audit, "compiled_plan_hash", "runtime audit")
    runtime_audit_hash = _canonical_json_hash(runtime_audit)

    workspace_root, workspace = _runtime_workspace(runtime_audit_path)
    version_id, active_run, phase_paths = _active_runtime_context(workspace_root, workspace)
    source_path = _resolve_artifact_reference(
        source_path_raw,
        workspace_root=workspace_root,
        artifact_parent=runtime_audit_path.parent,
    )
    expected_runtime_audit = runtime_audit_path.resolve(strict=False)
    expected_run_out = _resolve_artifact_reference(
        str(run_out),
        workspace_root=workspace_root,
        artifact_parent=authorization_path.parent,
    )

    if phase_paths is not None:
        expected_source = phase_paths["07_compile_preview"] / "strategy.py"
        expected_runtime_audit = phase_paths["08_runtime_audit"] / "runtime_audit.json"
        expected_phase_run_out = phase_paths["09_backtests"]
        if source_path != expected_source:
            raise ValueError("strategy_source_path must be the active version phase_paths.07_compile_preview/strategy.py")
        if runtime_audit_path.resolve(strict=False) != expected_runtime_audit:
            raise ValueError("runtime_audit_path must be the active version phase_paths.08_runtime_audit/runtime_audit.json")
        if expected_run_out != expected_phase_run_out:
            raise ValueError("run_out must be exactly the active version phase_paths.09_backtests directory")

    if not source_path.is_file():
        raise ValueError(f"strategy_source_path file not found: {source_path_raw}")
    actual_source_hash = f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    if source_hash != actual_source_hash:
        raise ValueError(
            f"runtime audit strategy_source_hash mismatch: audit={source_hash}, actual={actual_source_hash}"
        )

    expected_fields: dict[str, Any] = {
        "strategy_source_path": source_path,
        "strategy_source_hash": source_hash,
        "runtime_audit_path": expected_runtime_audit,
        "runtime_audit_hash": runtime_audit_hash,
        "compiled_plan_hash": compiled_plan_hash,
        "version_id": version_id,
        "active_run": active_run,
        "run_out": expected_run_out,
    }
    for field in ("strategy_source_hash", "runtime_audit_hash", "compiled_plan_hash"):
        _required_hash(reference, field, "strategy_source_presentation")
    _required_text(reference, "event_id", "strategy_source_presentation")
    for field in ("strategy_source_path", "runtime_audit_path", "run_out"):
        raw_value = _required_text(reference, field, "strategy_source_presentation")
        actual_path = _resolve_artifact_reference(
            raw_value,
            workspace_root=workspace_root,
            artifact_parent=authorization_path.parent,
        )
        if actual_path != expected_fields[field]:
            raise ValueError(f"strategy_source_presentation {field} mismatch")
    for field in ("strategy_source_hash", "runtime_audit_hash", "compiled_plan_hash", "version_id", "active_run"):
        if reference.get(field) != expected_fields[field]:
            raise ValueError(f"strategy_source_presentation {field} mismatch")

    event_path = _presentation_event_path(
        _required_text(reference, "path", "strategy_source_presentation"),
        workspace_root,
        workspace,
    )
    line_number = reference.get("line_number")
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number <= 0:
        raise ValueError("strategy_source_presentation line_number must be a positive integer")
    event_hash = _required_full_hash(reference, "event_hash", "strategy_source_presentation")
    lines = event_path.read_text(encoding="utf-8").splitlines()
    if line_number > len(lines):
        raise ValueError(f"strategy_source_presentation line {line_number} not found")
    line = lines[line_number - 1]
    actual_event_hash = f"sha256:{hashlib.sha256(line.encode('utf-8')).hexdigest()}"
    if event_hash != actual_event_hash:
        raise ValueError(
            f"strategy_source_presentation event_hash mismatch: recorded={event_hash}, actual={actual_event_hash}"
        )
    event = json.loads(line)
    if not isinstance(event, dict):
        raise ValueError("strategy source presentation event must be an object")
    if event.get("schema_version") != 1:
        raise ValueError("strategy source presentation event schema_version must be 1")
    _required_utc_timestamp(event, "timestamp", "strategy source presentation event")
    if event.get("phase") != "runtime_source_presentation":
        raise ValueError("strategy source presentation event phase must be runtime_source_presentation")
    if event.get("presentation") != "complete_strategy_source":
        raise ValueError("strategy source presentation event must record complete_strategy_source")
    if event.get("presented_by_role") != "coordinator":
        raise ValueError("strategy source presentation event must be owned by the coordinator")

    matching_fields = (
        "event_id",
        "phase",
        "presentation",
        "presented_by_role",
        "strategy_source_hash",
        "runtime_audit_hash",
        "compiled_plan_hash",
        "version_id",
        "active_run",
    )
    for field in matching_fields:
        if field not in reference:
            raise ValueError(f"strategy_source_presentation {field} is required")
        if field not in event:
            raise ValueError(f"strategy source presentation event {field} is required")
    for field in matching_fields:
        if event.get(field) != reference.get(field):
            raise ValueError(f"strategy_source_presentation {field} must match the recorded event")
    for field in ("strategy_source_path", "runtime_audit_path", "run_out"):
        event_path_value = _required_text(event, field, "strategy source presentation event")
        resolved_event_path = _resolve_artifact_reference(
            event_path_value,
            workspace_root=workspace_root,
            artifact_parent=authorization_path.parent,
        )
        if resolved_event_path != expected_fields[field]:
            raise ValueError(f"strategy source presentation event {field} mismatch")


def _runtime_workspace(runtime_audit_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = runtime_audit_path.resolve(strict=False)
    for parent in resolved.parents:
        config_path = parent / ".open-xquant" / "workspace.yaml"
        if config_path.is_file():
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(".open-xquant/workspace.yaml must contain an object")
            return parent, payload
    return resolved.parent, {}


def _active_runtime_context(
    workspace_root: Path,
    workspace: dict[str, Any],
) -> tuple[str | None, str | None, dict[str, Path] | None]:
    workflow = workspace.get("workflow")
    paths = workspace.get("paths")
    version_governed = (
        isinstance(workflow, dict) and workflow.get("layout") == "version_governed"
    ) or (isinstance(paths, dict) and bool(paths.get("versions_dir")))
    if not version_governed:
        return None, None, None
    if paths is not None and not isinstance(paths, dict):
        raise ValueError("workspace paths must contain an object")
    paths = paths if isinstance(paths, dict) else {}
    current = _read_json_object(workspace_root / "current.json", "current.json")
    version_id = _required_text(current, "active_version", "current.json")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version_id):
        raise ValueError("current.json active_version is unsafe")
    active_run_value = current.get("active_run")
    if active_run_value == "":
        active_run_value = None
    elif active_run_value is not None and not isinstance(active_run_value, str):
        raise ValueError("current.json active_run must be a non-empty string when present")

    versions_raw = paths.get("versions_dir", "versions")
    versions_relative = _safe_relative_path(versions_raw, "workspace paths.versions_dir")
    versions_root = (workspace_root / versions_relative).resolve(strict=False)
    _require_contained(versions_root, workspace_root.resolve(), "workspace paths.versions_dir")
    version_dir = (versions_root / version_id).resolve(strict=False)
    _require_contained(version_dir, versions_root, "active version directory")
    manifest = _read_json_object(version_dir / "version_manifest.json", "version_manifest.json")
    if manifest.get("version_id") != version_id:
        raise ValueError("version_manifest.json version_id must match current.json active_version")
    raw_phase_paths = manifest.get("phase_paths")
    if not isinstance(raw_phase_paths, dict):
        raise ValueError("version_manifest.json phase_paths must be an object")
    resolved_phases: dict[str, Path] = {}
    for phase in ("07_compile_preview", "08_runtime_audit", "09_backtests"):
        raw_path = raw_phase_paths.get(phase)
        relative = _safe_relative_path(raw_path, f"phase_paths.{phase}")
        resolved_phase = (workspace_root / relative).resolve(strict=False)
        _require_contained(resolved_phase, version_dir, f"phase_paths.{phase}")
        resolved_phases[phase] = resolved_phase
    return version_id, active_run_value, resolved_phases


def _presentation_event_path(raw_path: str, workspace_root: Path, workspace: dict[str, Any]) -> Path:
    event_relative = _safe_relative_path(raw_path, "strategy_source_presentation.path")
    paths = workspace.get("paths")
    if paths is not None and not isinstance(paths, dict):
        raise ValueError("workspace paths must contain an object")
    conversations_raw = paths.get("conversations_dir", "conversations") if isinstance(paths, dict) else "conversations"
    conversations_relative = _safe_relative_path(conversations_raw, "workspace paths.conversations_dir")
    conversations_root = (workspace_root / conversations_relative).resolve(strict=False)
    _require_contained(conversations_root, workspace_root.resolve(), "workspace paths.conversations_dir")
    event_path = (workspace_root / event_relative).resolve(strict=False)
    _require_contained(event_path, conversations_root, "strategy_source_presentation.path")
    event_subpath = event_path.relative_to(conversations_root)
    if len(event_subpath.parts) != 2 or event_subpath.name != "runtime-source-presentations.jsonl":
        raise ValueError(
            "strategy_source_presentation.path must be exactly "
            "<paths.conversations_dir>/<conversation_id>/runtime-source-presentations.jsonl"
        )
    if not event_path.is_file():
        raise ValueError(f"strategy_source_presentation event file not found: {raw_path}")
    return event_path


def _resolve_artifact_reference(raw_path: str, *, workspace_root: Path, artifact_parent: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve(strict=False)
    workspace_candidate = workspace_root / path
    if workspace_candidate.exists():
        return workspace_candidate.resolve(strict=False)
    return (artifact_parent / path).resolve(strict=False)


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty safe relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path


def _require_contained(path: Path, parent: Path, field: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{field} must stay within {parent}") from exc


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _required_text(payload: dict[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} {field} must be a non-empty string")
    return value


def _required_utc_timestamp(payload: dict[str, Any], field: str, label: str) -> str:
    value = _required_text(payload, field, label)
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value):
        raise ValueError(f"{label} {field} must be a UTC timestamp in YYYY-MM-DDTHH:MM:SSZ format")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(
            f"{label} {field} must be a valid UTC timestamp in YYYY-MM-DDTHH:MM:SSZ format"
        ) from exc
    return value


def _required_hash(payload: dict[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{label} {field} must be a sha256:<hex> hash")
    return value


def _required_full_hash(payload: dict[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not _FULL_HASH_RE.fullmatch(value):
        raise ValueError(f"{label} {field} must be a full sha256:<64 hex> hash")
    return value


def _canonical_json_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
