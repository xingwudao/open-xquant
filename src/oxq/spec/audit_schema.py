"""Deterministic schema checks for Agent-authored spec_audit.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SPEC_AUDIT_SCHEMA_VERSION = 2

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "spec_provenance_pass",
    "runtime_semantics_pass",
    "spec_hash",
    "conversation_hash",
    "catalog_hash",
    "recipe_matches",
    "field_audits",
    "component_audits",
    "missing_user_requirements",
    "agent_added_fields",
    "contradictions",
    "blocking_findings",
}

_ALLOWED_STATUS = {"pass", "block", "fail"}
_ALLOWED_RECIPE_STATUS = {"used", "available_but_not_used", "not_applicable"}
_ALLOWED_FIELD_STATUS = {"confirmed", "default", "unconfirmed", "contradiction", "agent_added"}
_ALLOWED_COMPONENT_STATUS = {"catalog", "recipe", "missing", "non_canonical"}
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{16,64}$")
_NEGATIVE_CONFIRMATION_RE = re.compile(
    r"(未指定|没有指定|未确认|没有确认|未明确|用户未|用户没有|not specified|not confirmed|unconfirmed|"
    r"did not specify|did not confirm|not explicitly specified|not explicitly confirmed|"
    r"agent\s+(?:chose|added|inferred|split)|agent将|agent自行)",
    re.IGNORECASE,
)
_POSITIVE_CONFIRMATION_RE = re.compile(
    r"(用户(?:已)?确认|用户接受|明确确认|确认了|user confirmed|explicitly confirmed|confirmed in turn|"
    r"accepted by user|user accepted|approved by user)",
    re.IGNORECASE,
)
_LATER_CONFIRMATION_CONTEXT_RE = re.compile(
    r"(后来|随后|之后|后续|第[^，。；;\s]*轮|第[^，。；;\s]*次|later|then|afterward|afterwards|"
    r"subsequently|in turn\s*\d+|turn\s*\d+)",
    re.IGNORECASE,
)
_HISTORICAL_NEGATIVE_PREFIX_RE = re.compile(
    r"(起初|最初|原先|一开始|此前|之前|先前|曾经|initially|originally|previously|earlier|before)\W*$",
    re.IGNORECASE,
)


def validate_spec_audit_file(path: str | Path) -> dict[str, Any]:
    """Validate a spec_audit.json file and return deterministic findings."""
    audit_path = Path(path)
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except OSError as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    return validate_spec_audit(payload)


def validate_spec_audit(payload: Any) -> dict[str, Any]:
    """Validate a parsed spec audit payload."""
    errors: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return _result("fail", [{"path": "$", "message": "spec_audit must be a JSON object"}])

    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS.difference(payload))
    for field in missing:
        errors.append({"path": field, "message": "missing required field"})

    status = payload.get("status")
    if not isinstance(status, str) or status not in _ALLOWED_STATUS:
        errors.append({"path": "status", "message": f"must be one of {sorted(_ALLOWED_STATUS)}"})

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version != SPEC_AUDIT_SCHEMA_VERSION:
        errors.append({"path": "schema_version", "message": f"must be {SPEC_AUDIT_SCHEMA_VERSION}"})

    for field in ("spec_provenance_pass", "runtime_semantics_pass"):
        if field in payload and not isinstance(payload[field], bool):
            errors.append({"path": field, "message": "must be a boolean"})
    if status == "pass":
        for field in ("spec_provenance_pass", "runtime_semantics_pass"):
            if field in payload and payload.get(field) is not True:
                errors.append({"path": field, "message": "must be true when status is pass"})

    for field in ("spec_hash", "conversation_hash", "catalog_hash"):
        value = payload.get(field)
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            errors.append({"path": field, "message": "must be a sha256:<hex> hash"})

    for field in (
        "recipe_matches",
        "field_audits",
        "component_audits",
        "missing_user_requirements",
        "agent_added_fields",
        "contradictions",
        "blocking_findings",
    ):
        if field in payload and not isinstance(payload[field], list):
            errors.append({"path": field, "message": "must be a list"})

    for index, item in enumerate(payload.get("recipe_matches", []) if isinstance(payload.get("recipe_matches"), list) else []):
        if not isinstance(item, dict):
            errors.append({"path": f"recipe_matches[{index}]", "message": "must be an object"})
            continue
        _require_str(item, f"recipe_matches[{index}]", "recipe", errors)
        _require_enum(item, f"recipe_matches[{index}]", "status", _ALLOWED_RECIPE_STATUS, errors)
        if "evidence" not in item or not isinstance(item["evidence"], list):
            errors.append({"path": f"recipe_matches[{index}].evidence", "message": "must be a list"})
        if "canonical" not in item or not isinstance(item["canonical"], bool):
            errors.append({"path": f"recipe_matches[{index}].canonical", "message": "must be a boolean"})

    for index, item in enumerate(payload.get("field_audits", []) if isinstance(payload.get("field_audits"), list) else []):
        if not isinstance(item, dict):
            errors.append({"path": f"field_audits[{index}]", "message": "must be an object"})
            continue
        _require_str(item, f"field_audits[{index}]", "field_path", errors)
        _require_enum(item, f"field_audits[{index}]", "status", _ALLOWED_FIELD_STATUS, errors)
        if "spec_value" not in item:
            errors.append({"path": f"field_audits[{index}].spec_value", "message": "missing required field"})
        if "evidence" not in item or not isinstance(item["evidence"], list):
            errors.append({"path": f"field_audits[{index}].evidence", "message": "must be a list"})
        elif item.get("status") == "confirmed" and _evidence_denies_confirmation(item["evidence"]):
            errors.append(
                {
                    "path": f"field_audits[{index}].status",
                    "message": "confirmed is inconsistent with evidence that says the user did not specify or confirm the field",
                }
            )
        if "blocking" in item and not isinstance(item["blocking"], bool):
            errors.append({"path": f"field_audits[{index}].blocking", "message": "must be a boolean"})

    for index, item in enumerate(payload.get("component_audits", []) if isinstance(payload.get("component_audits"), list) else []):
        if not isinstance(item, dict):
            errors.append({"path": f"component_audits[{index}]", "message": "must be an object"})
            continue
        _require_str(item, f"component_audits[{index}]", "component_path", errors)
        _require_str(item, f"component_audits[{index}]", "component_type", errors)
        _require_enum(item, f"component_audits[{index}]", "status", _ALLOWED_COMPONENT_STATUS, errors)
        if "evidence" in item and not isinstance(item["evidence"], list):
            errors.append({"path": f"component_audits[{index}].evidence", "message": "must be a list"})
        if "blocking" in item and not isinstance(item["blocking"], bool):
            errors.append({"path": f"component_audits[{index}].blocking", "message": "must be a boolean"})

    for field in ("missing_user_requirements", "agent_added_fields", "contradictions", "blocking_findings"):
        for index, item in enumerate(payload.get(field, []) if isinstance(payload.get(field), list) else []):
            if not isinstance(item, dict):
                errors.append({"path": f"{field}[{index}]", "message": "must be an object"})
            elif "message" not in item or not isinstance(item["message"], str):
                errors.append({"path": f"{field}[{index}].message", "message": "must be a string"})

    return _result("fail" if errors else "pass", errors)


def _require_str(item: dict[str, Any], prefix: str, field: str, errors: list[dict[str, str]]) -> None:
    if field not in item or not isinstance(item[field], str):
        errors.append({"path": f"{prefix}.{field}", "message": "must be a string"})


def _evidence_denies_confirmation(evidence: list[Any]) -> bool:
    has_negative = False
    unresolved_negative = False
    for entry in evidence:
        if not isinstance(entry, str):
            continue
        negative_match = _NEGATIVE_CONFIRMATION_RE.search(entry)
        if negative_match:
            has_negative = True
            before_negative = entry[: negative_match.start()]
            after_negative = entry[negative_match.end() :]
            if _POSITIVE_CONFIRMATION_RE.search(after_negative) and _LATER_CONFIRMATION_CONTEXT_RE.search(after_negative):
                unresolved_negative = False
            elif (
                _POSITIVE_CONFIRMATION_RE.search(before_negative)
                and _LATER_CONFIRMATION_CONTEXT_RE.search(before_negative)
                and _HISTORICAL_NEGATIVE_PREFIX_RE.search(before_negative)
            ):
                unresolved_negative = False
            else:
                unresolved_negative = True
            continue
        if (
            unresolved_negative
            and _POSITIVE_CONFIRMATION_RE.search(entry)
            and _LATER_CONFIRMATION_CONTEXT_RE.search(entry)
        ):
            unresolved_negative = False
    return has_negative and unresolved_negative


def _require_enum(
    item: dict[str, Any],
    prefix: str,
    field: str,
    allowed: set[str],
    errors: list[dict[str, str]],
) -> None:
    value = item.get(field)
    if not isinstance(value, str) or value not in allowed:
        errors.append({"path": f"{prefix}.{field}", "message": f"must be one of {sorted(allowed)}"})


def _result(status: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "status": status,
        "schema_version": SPEC_AUDIT_SCHEMA_VERSION,
        "errors": errors,
    }
