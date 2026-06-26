"""Deterministic schema checks for Agent-authored runtime_audit.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RUNTIME_AUDIT_SCHEMA_VERSION = 1

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "runtime_semantics_pass",
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


def validate_runtime_audit_file(path: str | Path) -> dict[str, Any]:
    """Validate a runtime_audit.json file and return deterministic findings."""
    audit_path = Path(path)
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except OSError as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    return validate_runtime_audit(payload)


def validate_runtime_audit(payload: Any) -> dict[str, Any]:
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
        for index, item in enumerate(field_rows):
            if not isinstance(item, dict):
                errors.append({"path": f"material_field_audits[{index}]", "message": "must be an object"})
                continue
            _require_str(item, f"material_field_audits[{index}]", "field_path", errors)
            if "spec_value" not in item:
                errors.append({"path": f"material_field_audits[{index}].spec_value", "message": "missing required field"})
            _require_str(item, f"material_field_audits[{index}]", "runtime_path", errors)
            if "runtime_value" not in item:
                errors.append(
                    {"path": f"material_field_audits[{index}].runtime_value", "message": "missing required field"}
                )
            _require_enum(item, f"material_field_audits[{index}]", "status", _ALLOWED_FIELD_STATUS, errors)
            if "evidence" not in item or not isinstance(item["evidence"], list):
                errors.append({"path": f"material_field_audits[{index}].evidence", "message": "must be a list"})
            if "blocking" in item and not isinstance(item["blocking"], bool):
                errors.append({"path": f"material_field_audits[{index}].blocking", "message": "must be a boolean"})

    blocking_findings = payload.get("blocking_findings", [])
    if isinstance(blocking_findings, list):
        for index, item in enumerate(blocking_findings):
            if not isinstance(item, dict):
                errors.append({"path": f"blocking_findings[{index}]", "message": "must be an object"})
            elif "message" not in item or not isinstance(item["message"], str):
                errors.append({"path": f"blocking_findings[{index}].message", "message": "must be a string"})

    return _result("fail" if errors else "pass", errors)


def _require_str(item: dict[str, Any], prefix: str, field: str, errors: list[dict[str, str]]) -> None:
    if field not in item or not isinstance(item[field], str):
        errors.append({"path": f"{prefix}.{field}", "message": "must be a string"})


def _require_enum(
    item: dict[str, Any], prefix: str, field: str, allowed: set[str], errors: list[dict[str, str]]
) -> None:
    value = item.get(field)
    if not isinstance(value, str) or value not in allowed:
        errors.append({"path": f"{prefix}.{field}", "message": f"must be one of {sorted(allowed)}"})


def _result(status: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    return {"status": status, "errors": errors}
