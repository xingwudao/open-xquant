"""Deterministic schema checks for Agent-authored spec_audit.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SPEC_AUDIT_SCHEMA_VERSION = 3

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "spec_provenance_pass",
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


def validate_spec_audit_file(
    path: str | Path,
    *,
    spec_path: str | Path | None = None,
    spec: Any | None = None,
    require_confirmed_coverage: bool = False,
) -> dict[str, Any]:
    """Validate a spec_audit.json file and return deterministic findings."""
    audit_path = Path(path)
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except OSError as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    if spec is None and spec_path is not None:
        try:
            from oxq.spec.schema import StrategySpec

            spec = StrategySpec.from_yaml(spec_path)
        except Exception as exc:
            return _result("fail", [{"path": "spec", "message": f"invalid strategy spec: {exc}"}])
    return validate_spec_audit(payload, spec=spec, require_confirmed_coverage=require_confirmed_coverage)


def validate_spec_audit(
    payload: Any,
    *,
    spec: Any | None = None,
    require_confirmed_coverage: bool = False,
) -> dict[str, Any]:
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

    if "spec_provenance_pass" in payload and not isinstance(payload["spec_provenance_pass"], bool):
        errors.append({"path": "spec_provenance_pass", "message": "must be a boolean"})
    if status == "pass":
        if "spec_provenance_pass" in payload and payload.get("spec_provenance_pass") is not True:
            errors.append({"path": "spec_provenance_pass", "message": "must be true when status is pass"})

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

    if require_confirmed_coverage:
        if spec is None:
            errors.append(
                {
                    "path": "spec",
                    "message": "strict confirmed coverage requires a strategy spec",
                }
            )
        else:
            errors.extend(_validate_confirmed_effective_field_coverage(payload, spec))

    return _result("fail" if errors else "pass", errors)


def _validate_confirmed_effective_field_coverage(payload: dict[str, Any], spec: Any) -> list[dict[str, str]]:
    """Require every effective StrategySpec field to have a confirmed audit row."""
    errors: list[dict[str, str]] = []
    try:
        effective_spec = spec.to_effective_dict()
    except AttributeError:
        effective_spec = spec
    effective_fields = dict(_flatten_effective_fields(effective_spec))
    field_rows = payload.get("field_audits")
    if not isinstance(field_rows, list):
        return errors

    rows_by_path: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, item in enumerate(field_rows):
        if not isinstance(item, dict) or not isinstance(item.get("field_path"), str):
            continue
        rows_by_path.setdefault(item["field_path"], []).append((index, item))

    for field_path, expected_value in effective_fields.items():
        rows = rows_by_path.get(field_path, [])
        if not rows:
            errors.append(
                {
                    "path": f"field_audits[{field_path}]",
                    "message": "missing confirmed audit row for effective spec field",
                }
            )
            continue
        confirmed_rows = [(index, row) for index, row in rows if row.get("status") == "confirmed"]
        if not confirmed_rows:
            statuses = sorted({str(row.get("status")) for _, row in rows})
            errors.append(
                {
                    "path": f"field_audits[{field_path}].status",
                    "message": f"effective spec field must be confirmed before formal backtest; got {statuses}",
                }
            )
            continue
        non_confirmed_statuses = sorted({str(row.get("status")) for _, row in rows if row.get("status") != "confirmed"})
        if non_confirmed_statuses:
            errors.append(
                {
                    "path": f"field_audits[{field_path}].status",
                    "message": "effective spec field has conflicting non-confirmed audit rows; "
                    f"got {non_confirmed_statuses}",
                }
            )
        for confirmed_index, confirmed_row in confirmed_rows:
            _validate_confirmed_effective_field_row(errors, field_path, expected_value, confirmed_index, confirmed_row)

    return errors


def _validate_confirmed_effective_field_row(
    errors: list[dict[str, str]],
    field_path: str,
    expected_value: Any,
    index: int,
    confirmed_row: dict[str, Any],
) -> None:
    if confirmed_row.get("blocking") is True:
        errors.append(
            {
                "path": f"field_audits[{index}].blocking",
                "message": "confirmed effective spec field must not be blocking",
            }
        )
    if not _json_equivalent(confirmed_row.get("spec_value"), expected_value):
        errors.append(
            {
                "path": f"field_audits[{index}].spec_value",
                "message": "confirmed audit value does not match effective spec value",
            }
        )
    evidence = confirmed_row.get("evidence")
    if not isinstance(evidence, list) or not any(isinstance(item, str) and item.strip() for item in evidence):
        errors.append(
            {
                "path": f"field_audits[{index}].evidence",
                "message": "confirmed effective spec field requires non-empty user confirmation evidence",
            }
        )
    elif _evidence_denies_confirmation(evidence):
        errors.append(
            {
                "path": f"field_audits[{index}].evidence",
                "message": "confirmed effective spec field evidence denies user confirmation",
            }
        )


def _flatten_effective_fields(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value and prefix:
            return [(prefix, {})]
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


def _json_equivalent(left: Any, right: Any) -> bool:
    return _canonical_json_value(left) == _canonical_json_value(right)


def _canonical_json_value(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except TypeError:
        return str(value)


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
