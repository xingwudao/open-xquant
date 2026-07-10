"""Deterministic schema checks for Agent-authored spec_audit.json."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

SPEC_AUDIT_SCHEMA_VERSION = 4

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "audit_conclusion",
    "user_confirmation_status",
    "spec_provenance_pass",
    "spec_hash",
    "conversation_hash",
    "catalog_hash",
    "strategy_idea_brief",
    "strategy_idea_audit",
    "strategy_idea_brief_hash",
    "strategy_idea_audit_hash",
    "recipe_matches",
    "field_audits",
    "component_audits",
    "unsupported_mappings",
    "missing_user_requirements",
    "agent_added_fields",
    "contradictions",
    "blocking_findings",
}

_ALLOWED_STATUS = {"pass", "block", "fail"}
_ALLOWED_AUDIT_CONCLUSION = {"all_pass", "blocked", "fail"}
_ALLOWED_CONFIRMATION_STATUS = {"pending", "confirmed", "rejected"}
_ALLOWED_RECIPE_STATUS = {"used", "available_but_not_used", "not_applicable"}
_ALLOWED_FIELD_STATUS = {"confirmed", "default", "unconfirmed", "contradiction", "agent_added"}
_ALLOWED_STATE_TRIPLES = {
    ("pass", "all_pass", "confirmed"),
    ("block", "all_pass", "pending"),
    ("block", "blocked", "pending"),
    ("block", "blocked", "rejected"),
    ("fail", "fail", "pending"),
    ("fail", "fail", "rejected"),
}
_ALLOWED_MATERIAL_CATEGORY = {
    "strategy_logic",
    "portfolio_construction",
    "execution_assumption",
    "backtest_assumption",
    "data_assumption",
    "cost_assumption",
    "validation_assumption",
    "risk_assumption",
    "metric_assumption",
    "system_provenance",
}
_ALLOWED_COMPONENT_STATUS = {"catalog", "recipe", "missing", "non_canonical"}
_ALLOWED_UNSUPPORTED_MAPPING_DISPOSITION = {"blocked", "deferred_framework", "excluded_non_material", "not_applicable"}
_REQUIRED_CONFIRMATION_TABLE_COLUMNS = ("section", "field path", "spec value", "source", "audit status", "impact")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{16,64}$")
_NEGATIVE_CONFIRMATION_RE = re.compile(
    r"(未指定|没有指定|未确认|没有确认|未明确|用户未|用户没有|not specified|not confirmed|unconfirmed|"
    r"did not specify|did not confirm|not explicitly specified|not explicitly confirmed|"
    r"agent\s+(?:chose|added|inferred|split)|agent将|agent自行|"
    r"framework default|runtime default|parser default|template default|openxquant default|"
    r"effective\s+strategyspec\s+default|documented\s+for\s+full\s+spec\s+coverage|absent\s+from\s+yaml|"
    r"框架默认|运行时默认|解析器默认|模板默认|系统默认)",
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
    component_catalog_path: str | Path | None = None,
    component_catalog: Any | None = None,
    require_confirmed_coverage: bool = False,
    verify_confirmation_table: bool = False,
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
    if component_catalog is None and component_catalog_path is not None:
        catalog_path = Path(component_catalog_path)
        try:
            component_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _result("fail", [{"path": "component_catalog", "message": f"invalid JSON: {exc}"}])
        except OSError as exc:
            return _result("fail", [{"path": "component_catalog", "message": str(exc)}])
    result = validate_spec_audit(
        payload,
        spec=spec,
        component_catalog=component_catalog,
        require_confirmed_coverage=require_confirmed_coverage,
    )
    if not isinstance(payload, dict):
        return result
    requires_table = _requires_spec_confirmation_table(
        status=payload.get("status"),
        audit_conclusion=payload.get("audit_conclusion"),
        confirmation_status=payload.get("user_confirmation_status"),
    )
    requires_event = _requires_confirmation_event(
        status=payload.get("status"),
        confirmation_status=payload.get("user_confirmation_status"),
    )
    if verify_confirmation_table or requires_table or requires_event:
        errors = list(result["errors"])
        errors.extend(_validate_spec_confirmation_table_artifact(payload.get("spec_confirmation_table"), audit_path, spec=spec))
        errors.extend(
            _validate_confirmation_event_artifact(
                payload.get("confirmation_event"),
                audit_path,
                payload.get("spec_confirmation_table"),
                payload,
            )
        )
        return _result("fail" if errors else "pass", errors)
    return result


def validate_spec_audit(
    payload: Any,
    *,
    spec: Any | None = None,
    component_catalog: Any | None = None,
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

    audit_conclusion = payload.get("audit_conclusion")
    if not isinstance(audit_conclusion, str) or audit_conclusion not in _ALLOWED_AUDIT_CONCLUSION:
        errors.append(
            {
                "path": "audit_conclusion",
                "message": f"must be one of {sorted(_ALLOWED_AUDIT_CONCLUSION)}",
            }
        )

    confirmation_status = payload.get("user_confirmation_status")
    if not isinstance(confirmation_status, str) or confirmation_status not in _ALLOWED_CONFIRMATION_STATUS:
        errors.append(
            {
                "path": "user_confirmation_status",
                "message": f"must be one of {sorted(_ALLOWED_CONFIRMATION_STATUS)}",
            }
        )

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version != SPEC_AUDIT_SCHEMA_VERSION:
        errors.append({"path": "schema_version", "message": f"must be {SPEC_AUDIT_SCHEMA_VERSION}"})

    if "spec_provenance_pass" in payload and not isinstance(payload["spec_provenance_pass"], bool):
        errors.append({"path": "spec_provenance_pass", "message": "must be a boolean"})
    if status == "pass":
        if "spec_provenance_pass" in payload and payload.get("spec_provenance_pass") is not True:
            errors.append({"path": "spec_provenance_pass", "message": "must be true when status is pass"})
        if audit_conclusion != "all_pass":
            errors.append({"path": "audit_conclusion", "message": "must be all_pass when status is pass"})
        if confirmation_status != "confirmed":
            errors.append({"path": "user_confirmation_status", "message": "must be confirmed when status is pass"})
    if confirmation_status == "confirmed" and status != "pass":
        errors.append({"path": "status", "message": "must be pass when user_confirmation_status is confirmed"})
    if confirmation_status == "confirmed" and audit_conclusion != "all_pass":
        errors.append({"path": "audit_conclusion", "message": "must be all_pass when user_confirmation_status is confirmed"})
    if status == "fail" and audit_conclusion != "fail":
        errors.append({"path": "audit_conclusion", "message": "must be fail when status is fail"})
    if audit_conclusion == "blocked" and status != "block":
        errors.append({"path": "status", "message": "must be block when audit_conclusion is blocked"})
    if status == "block" and audit_conclusion == "all_pass" and confirmation_status != "pending":
        errors.append(
            {
                "path": "user_confirmation_status",
                "message": "must be pending for an all_pass audit awaiting user confirmation",
            }
        )
    if (
        isinstance(status, str)
        and status in _ALLOWED_STATUS
        and isinstance(audit_conclusion, str)
        and audit_conclusion in _ALLOWED_AUDIT_CONCLUSION
        and isinstance(confirmation_status, str)
        and confirmation_status in _ALLOWED_CONFIRMATION_STATUS
        and (status, audit_conclusion, confirmation_status) not in _ALLOWED_STATE_TRIPLES
    ):
        errors.append(
            {
                "path": "state",
                "message": "must be one of pass/all_pass/confirmed, block/all_pass/pending, "
                "block/blocked/pending, block/blocked/rejected, fail/fail/pending, or fail/fail/rejected",
            }
        )

    _validate_spec_confirmation_table(
        payload.get("spec_confirmation_table"),
        audit_conclusion=audit_conclusion,
        require_table=_requires_spec_confirmation_table(
            status=status,
            audit_conclusion=audit_conclusion,
            confirmation_status=confirmation_status,
        ),
        errors=errors,
    )
    _validate_confirmation_event(
        payload.get("confirmation_event"),
        require_event=_requires_confirmation_event(status=status, confirmation_status=confirmation_status),
        errors=errors,
    )

    for field in (
        "spec_hash",
        "conversation_hash",
        "catalog_hash",
        "strategy_idea_brief_hash",
        "strategy_idea_audit_hash",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            errors.append({"path": field, "message": "must be a sha256:<hex> hash"})

    for field in ("strategy_idea_brief", "strategy_idea_audit"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append({"path": field, "message": "must be a non-empty string"})

    if spec is not None and callable(getattr(spec, "compute_hash", None)):
        expected_spec_hash = spec.compute_hash()
        if isinstance(payload.get("spec_hash"), str) and payload.get("spec_hash") != expected_spec_hash:
            errors.append(
                {
                    "path": "spec_hash",
                    "message": f"must match strategy spec hash {expected_spec_hash}",
                }
            )

    if component_catalog is not None:
        errors.extend(_validate_catalog_hash(payload, component_catalog))

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
    all_pass_gate = status == "pass" or audit_conclusion == "all_pass"
    if all_pass_gate:
        for field in (
            "missing_user_requirements",
            "agent_added_fields",
            "contradictions",
            "blocking_findings",
            "unsupported_mappings",
        ):
            value = payload.get(field)
            if isinstance(value, list) and value:
                errors.append({"path": field, "message": "must be empty when audit is all_pass or status is pass"})
    if "unsupported_mappings" in payload and not isinstance(payload["unsupported_mappings"], list):
        errors.append({"path": "unsupported_mappings", "message": "must be a list"})

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
        _require_enum(item, f"field_audits[{index}]", "material_category", _ALLOWED_MATERIAL_CATEGORY, errors)
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
        if all_pass_gate and item.get("blocking") is True:
            errors.append(
                {
                    "path": f"field_audits[{index}].blocking",
                    "message": "blocking field audit row cannot pass formal spec audit",
                }
            )
        if all_pass_gate and item.get("status") in {"unconfirmed", "contradiction", "agent_added"}:
            errors.append(
                {
                    "path": f"field_audits[{index}].status",
                    "message": "unresolved field audit row cannot pass formal spec audit",
                }
            )

    if spec is not None:
        errors.extend(_validate_effective_field_audit_paths(payload, spec))

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
        if all_pass_gate and item.get("blocking") is True:
            errors.append(
                {
                    "path": f"component_audits[{index}].blocking",
                    "message": "blocking component audit row cannot pass formal spec audit",
                }
            )
        if all_pass_gate and item.get("status") in {"missing", "non_canonical"}:
            errors.append(
                {
                    "path": f"component_audits[{index}].status",
                    "message": "unresolved component audit row cannot pass formal spec audit",
                }
            )

    for index, item in enumerate(payload.get("unsupported_mappings", []) if isinstance(payload.get("unsupported_mappings"), list) else []):
        if not isinstance(item, dict):
            errors.append({"path": f"unsupported_mappings[{index}]", "message": "must be an object"})
            continue
        _require_str(item, f"unsupported_mappings[{index}]", "source_field", errors)
        _require_str(item, f"unsupported_mappings[{index}]", "requested_semantic", errors)
        _require_str(item, f"unsupported_mappings[{index}]", "reason", errors)
        _require_enum(
            item,
            f"unsupported_mappings[{index}]",
            "disposition",
            _ALLOWED_UNSUPPORTED_MAPPING_DISPOSITION,
            errors,
        )
        if "blocking" not in item or not isinstance(item["blocking"], bool):
            errors.append({"path": f"unsupported_mappings[{index}].blocking", "message": "must be a boolean"})
        elif all_pass_gate and item["blocking"] is True:
            errors.append(
                {
                    "path": f"unsupported_mappings[{index}].blocking",
                    "message": "blocking unsupported mapping cannot pass formal spec audit",
                }
            )
        if all_pass_gate and item.get("disposition") in {"blocked", "deferred_framework"}:
            errors.append(
                {
                    "path": f"unsupported_mappings[{index}].disposition",
                    "message": "unresolved unsupported mapping cannot pass formal spec audit",
                }
            )

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


def _validate_effective_field_audit_paths(payload: dict[str, Any], spec: Any) -> list[dict[str, str]]:
    """Reject field audit rows that do not point at effective StrategySpec fields."""
    errors: list[dict[str, str]] = []
    try:
        effective_spec = spec.to_effective_dict()
    except AttributeError:
        effective_spec = spec
    effective_fields = {field_path for field_path, _ in _flatten_effective_fields(effective_spec)}
    field_rows = payload.get("field_audits")
    if not isinstance(field_rows, list):
        return errors

    for item in field_rows:
        if not isinstance(item, dict) or not isinstance(item.get("field_path"), str):
            continue
        field_path = item["field_path"]
        if field_path not in effective_fields:
            errors.append(
                {
                    "path": f"field_audits[{field_path}]",
                    "message": "field audit path must be an effective StrategySpec field; "
                    "move YAML-only or misplaced source paths to evidence, contradictions.source_yaml_path, "
                    "and builder_required_fix instead",
                }
            )
    return errors


def _requires_spec_confirmation_table(
    *,
    status: Any,
    audit_conclusion: Any,
    confirmation_status: Any,
) -> bool:
    return status == "pass" or audit_conclusion == "all_pass" or confirmation_status == "confirmed"


def _requires_confirmation_event(*, status: Any, confirmation_status: Any) -> bool:
    return status == "pass" or confirmation_status == "confirmed"


def _validate_spec_confirmation_table(
    value: Any,
    *,
    audit_conclusion: Any,
    require_table: bool,
    errors: list[dict[str, str]],
) -> None:
    if value is None:
        if require_table:
            errors.append(
                {
                    "path": "spec_confirmation_table",
                    "message": "must be present when audit is all_pass, pending user confirmation, or confirmed",
                }
            )
        return
    if audit_conclusion == "blocked":
        errors.append(
            {
                "path": "spec_confirmation_table",
                "message": "must be null or omitted when audit_conclusion is blocked",
            }
        )
        return
    if not isinstance(value, dict):
        errors.append({"path": "spec_confirmation_table", "message": "must be an object or null"})
        return
    path = value.get("path")
    if not isinstance(path, str) or not path:
        errors.append({"path": "spec_confirmation_table.path", "message": "must be a non-empty string"})
    digest = value.get("hash")
    if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
        errors.append({"path": "spec_confirmation_table.hash", "message": "must be a sha256:<hex> hash"})
    hash_type = value.get("hash_type", "sha256")
    if hash_type != "sha256":
        errors.append({"path": "spec_confirmation_table.hash_type", "message": "must be sha256"})


def _validate_confirmation_event(value: Any, *, require_event: bool, errors: list[dict[str, str]]) -> None:
    if value is None:
        if require_event:
            errors.append(
                {
                    "path": "confirmation_event",
                    "message": "must be present when user_confirmation_status is confirmed",
                }
            )
        return
    if not isinstance(value, dict):
        errors.append({"path": "confirmation_event", "message": "must be an object or null"})
        return
    for field in ("path", "event_id", "artifact_path", "spec_audit_path"):
        item = value.get(field)
        if not isinstance(item, str) or not item:
            errors.append({"path": f"confirmation_event.{field}", "message": "must be a non-empty string"})
    line_number = value.get("line_number")
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number <= 0:
        errors.append({"path": "confirmation_event.line_number", "message": "must be a positive integer"})
    for field in ("event_hash", "artifact_hash", "spec_audit_hash"):
        digest = value.get(field)
        if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
            errors.append({"path": f"confirmation_event.{field}", "message": "must be a sha256:<hex> hash"})


def _validate_spec_confirmation_table_artifact(value: Any, audit_path: Path, *, spec: Any | None = None) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not isinstance(value, dict):
        return errors
    raw_path = value.get("path")
    recorded_hash = value.get("hash")
    if not isinstance(raw_path, str) or not raw_path:
        return errors
    if not isinstance(recorded_hash, str) or not _HASH_RE.fullmatch(recorded_hash):
        return errors
    table_path = _resolve_audit_artifact_path(raw_path, audit_path)
    if not table_path.exists():
        errors.append({"path": "spec_confirmation_table.path", "message": f"file not found: {raw_path}"})
        return errors
    table_bytes = table_path.read_bytes()
    full_hash = f"sha256:{hashlib.sha256(table_bytes).hexdigest()}"
    short_hash = f"sha256:{hashlib.sha256(table_bytes).hexdigest()[:16]}"
    if recorded_hash not in {short_hash, full_hash}:
        errors.append(
            {
                "path": "spec_confirmation_table.hash",
                "message": f"hash mismatch: recorded={recorded_hash}, actual={short_hash}",
            }
        )
    if spec is not None:
        try:
            table_text = table_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(
                {
                    "path": "spec_confirmation_table.content",
                    "message": f"must be UTF-8 Markdown: {exc}",
                }
            )
        else:
            errors.extend(_validate_spec_confirmation_table_content(table_text, spec))
    return errors


def _validate_spec_confirmation_table_content(table_text: str, spec: Any) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    try:
        effective_spec = spec.to_effective_dict()
    except AttributeError:
        effective_spec = spec
    expected_fields = dict(_flatten_effective_fields(effective_spec))
    table_rows = _parse_markdown_table(table_text)
    if table_rows is None:
        return [
            {
                "path": "spec_confirmation_table.content",
                "message": "must be a Markdown table with Section, Field path, Spec value, Source, "
                "Audit status, and Impact columns",
            }
        ]
    field_rows: dict[str, str] = {}
    for index, row in enumerate(table_rows):
        field_path = row.get("field path", "").strip()
        if not field_path:
            errors.append({"path": f"spec_confirmation_table.content[{index}].field_path", "message": "must be non-empty"})
            continue
        if field_path in field_rows:
            errors.append(
                {
                    "path": f"spec_confirmation_table.content[{field_path}]",
                    "message": "duplicate effective StrategySpec field row",
                }
            )
            continue
        if field_path not in expected_fields:
            errors.append(
                {
                    "path": f"spec_confirmation_table.content[{field_path}]",
                    "message": "unknown field path not present in effective StrategySpec",
                }
            )
            continue
        section = row.get("section", "").strip()
        expected_section = field_path.split(".", 1)[0]
        if section != expected_section:
            errors.append(
                {
                    "path": f"spec_confirmation_table.content[{field_path}].section",
                    "message": f"must be {expected_section}",
                }
            )
        audit_status = row.get("audit status", "").strip().lower()
        if audit_status != "confirmed":
            errors.append(
                {
                    "path": f"spec_confirmation_table.content[{field_path}].audit_status",
                    "message": "must be confirmed",
                }
            )
        for column in ("source", "impact"):
            if not row.get(column, "").strip():
                errors.append(
                    {
                        "path": f"spec_confirmation_table.content[{field_path}].{column.replace(' ', '_')}",
                        "message": "must be non-empty",
                    }
                )
        field_rows[field_path] = row.get("spec value", "")
    if not field_rows:
        return [
            {
                "path": "spec_confirmation_table.content",
                "message": "must include one row for every effective StrategySpec field",
            }
        ]
    for field_path, expected_value in expected_fields.items():
        if field_path not in field_rows:
            errors.append(
                {
                    "path": "spec_confirmation_table.content",
                    "message": f"missing effective StrategySpec field {field_path}",
                }
            )
            continue
        if not _markdown_value_matches(field_rows[field_path], expected_value):
            errors.append(
                {
                    "path": "spec_confirmation_table.content",
                    "message": f"value for {field_path} does not match effective StrategySpec value",
                }
            )
    return errors


def _parse_markdown_table(table_text: str) -> list[dict[str, str]] | None:
    rows: list[list[str]] = []
    for raw_line in table_text.splitlines():
        line = raw_line.strip()
        cells = _split_markdown_table_row(line)
        if cells is None:
            continue
        if cells and all(cell and set(cell).issubset({"-", ":"}) for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return None
    headers = [_normalize_markdown_header(cell) for cell in rows[0]]
    if len(headers) != len(set(headers)):
        return None
    if any(column not in headers for column in _REQUIRED_CONFIRMATION_TABLE_COLUMNS):
        return None
    parsed: list[dict[str, str]] = []
    for cells in rows[1:]:
        if len(cells) != len(headers):
            return None
        parsed.append({headers[index]: cells[index].strip() for index in range(len(headers))})
    return parsed


def _split_markdown_table_row(line: str) -> list[str] | None:
    if not line.startswith("|") or not line.endswith("|"):
        return None
    cells: list[str] = []
    current: list[str] = []
    index = 1
    end = len(line) - 1
    while index < end:
        char = line[index]
        if char == "\\" and index + 1 < end and line[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return cells


def _normalize_markdown_header(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _markdown_value_matches(value_text: str, expected_value: Any) -> bool:
    stripped = value_text.strip().strip("`").strip()
    if _json_equivalent(stripped, expected_value):
        return True
    variants = {
        json.dumps(expected_value, sort_keys=True, default=str),
        json.dumps(expected_value, sort_keys=True, default=str, ensure_ascii=False),
        str(expected_value),
    }
    if expected_value is None:
        variants.update({"null", "None"})
    if isinstance(expected_value, bool):
        variants.update({str(expected_value).lower(), str(expected_value)})
    if stripped in variants:
        return True
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return _json_equivalent(parsed, expected_value)


def _validate_confirmation_event_artifact(
    value: Any,
    audit_path: Path,
    spec_confirmation_table: Any,
    audit_payload: Any,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not isinstance(value, dict):
        return errors
    raw_path = value.get("path")
    line_number = value.get("line_number")
    event_hash = value.get("event_hash")
    if not isinstance(raw_path, str) or not raw_path:
        return errors
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number <= 0:
        return errors
    if not isinstance(event_hash, str) or not _HASH_RE.fullmatch(event_hash):
        return errors
    event_path = _resolve_audit_artifact_path(raw_path, audit_path)
    if not event_path.exists():
        errors.append({"path": "confirmation_event.path", "message": f"file not found: {raw_path}"})
        return errors
    lines = event_path.read_text(encoding="utf-8").splitlines()
    if line_number > len(lines):
        errors.append(
            {
                "path": "confirmation_event.line_number",
                "message": f"line {line_number} not found in {raw_path}",
            }
        )
        return errors
    line = lines[line_number - 1]
    full_hash = f"sha256:{hashlib.sha256(line.encode('utf-8')).hexdigest()}"
    short_hash = f"sha256:{hashlib.sha256(line.encode('utf-8')).hexdigest()[:16]}"
    if event_hash not in {short_hash, full_hash}:
        errors.append(
            {
                "path": "confirmation_event.event_hash",
                "message": f"hash mismatch: recorded={event_hash}, actual={short_hash}",
            }
        )
        return errors
    try:
        event_payload = json.loads(line)
    except json.JSONDecodeError as exc:
        errors.append({"path": "confirmation_event.line", "message": f"invalid JSON: {exc}"})
        return errors
    if not isinstance(event_payload, dict):
        errors.append({"path": "confirmation_event.line", "message": "must be a JSON object"})
        return errors
    if isinstance(spec_confirmation_table, dict):
        table_path = spec_confirmation_table.get("path")
        table_hash = spec_confirmation_table.get("hash")
        if value.get("artifact_path") != table_path:
            errors.append(
                {
                    "path": "confirmation_event.artifact_path",
                    "message": "must match spec_confirmation_table.path",
                }
            )
        if value.get("artifact_hash") != table_hash:
            errors.append(
                {
                    "path": "confirmation_event.artifact_hash",
                    "message": "must match spec_confirmation_table.hash",
                }
            )
    spec_audit_ref = value.get("spec_audit_path")
    if isinstance(spec_audit_ref, str) and spec_audit_ref:
        referenced_audit_path = _resolve_audit_artifact_path(spec_audit_ref, audit_path)
        references_current_audit = referenced_audit_path.resolve() == audit_path.resolve()
        is_archived_copy = False
        if not references_current_audit:
            is_archived_copy = _is_faithful_archived_audit_copy(referenced_audit_path, audit_payload)
        if not references_current_audit and not is_archived_copy:
            errors.append(
                {
                    "path": "confirmation_event.spec_audit_path",
                    "message": "must reference the current spec_audit.json or an identical archived copy",
                }
            )
    expected_pre_hashes = _pre_confirmation_spec_audit_hashes(audit_payload)
    if expected_pre_hashes and value.get("spec_audit_hash") not in expected_pre_hashes:
        errors.append(
            {
                "path": "confirmation_event.spec_audit_hash",
                "message": "must match the pre-confirmation spec_audit.json hash",
            }
        )
    for field in ("event_id", "artifact_path", "artifact_hash", "spec_audit_path", "spec_audit_hash"):
        if event_payload.get(field) != value.get(field):
            errors.append(
                {
                    "path": f"confirmation_event.{field}",
                    "message": f"must match confirmations.jsonl {field}",
                }
            )
    if event_payload.get("phase") != "spec_confirmation":
        errors.append({"path": "confirmation_event.phase", "message": "must be spec_confirmation"})
    if event_payload.get("field_scope") != "full_spec_table":
        errors.append({"path": "confirmation_event.field_scope", "message": "must be full_spec_table"})
    return errors


def _pre_confirmation_spec_audit_hashes(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    if payload.get("status") != "pass" and payload.get("user_confirmation_status") != "confirmed":
        return set()
    candidate = deepcopy(payload)
    candidate.pop("confirmation_event", None)
    candidate["status"] = "block"
    candidate["user_confirmation_status"] = "pending"
    candidates = [candidate]
    if "next_required_phase" in candidate:
        pending_candidate = deepcopy(candidate)
        pending_candidate["next_required_phase"] = "user_spec_confirmation"
        candidates.append(pending_candidate)
    hashes: set[str] = set()
    for item in candidates:
        canonical = json.dumps(item, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        hashes.add(f"sha256:{digest[:16]}")
        hashes.add(f"sha256:{digest}")
    return hashes


def _resolve_audit_artifact_path(raw_path: str, audit_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if (audit_path.parent / path).exists():
        return audit_path.parent / path
    return Path.cwd() / path


def _is_faithful_archived_audit_copy(referenced_audit_path: Path, audit_payload: dict[str, Any]) -> bool:
    try:
        referenced_payload = json.loads(referenced_audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return referenced_payload == audit_payload


def _validate_catalog_hash(payload: dict[str, Any], component_catalog: Any) -> list[dict[str, str]]:
    if not isinstance(component_catalog, dict):
        return [{"path": "component_catalog", "message": "must be a JSON object"}]
    errors: list[dict[str, str]] = []
    catalog_hash = component_catalog.get("catalog_hash")
    if not isinstance(catalog_hash, str) or not _HASH_RE.fullmatch(catalog_hash):
        errors.append({"path": "component_catalog.catalog_hash", "message": "must be a sha256:<hex> hash"})
    else:
        try:
            from oxq.core.component_catalog import _catalog_hash

            computed_hash = _catalog_hash(component_catalog)
        except Exception as exc:
            errors.append({"path": "component_catalog.catalog_hash", "message": f"could not compute catalog hash: {exc}"})
        else:
            if computed_hash != catalog_hash:
                errors.append(
                    {
                        "path": "component_catalog.catalog_hash",
                        "message": f"must match computed component catalog hash {computed_hash}",
                    }
                )
    if isinstance(payload.get("catalog_hash"), str) and isinstance(catalog_hash, str):
        if payload.get("catalog_hash") != catalog_hash:
            errors.append(
                {
                    "path": "catalog_hash",
                    "message": f"must match component catalog hash {catalog_hash}",
                }
            )
    return errors


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
