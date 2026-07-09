"""Validation contract for external source-to-SPEC mapping reports."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from oxq.spec.schema import StrategySpec

MAPPING_CONTRACT_SCHEMA_VERSION = 1

_ALLOWED_SEMANTICS = {"strategy", "run", "report", "studio", "metadata", "unsupported"}
_ALLOWED_STATUS = {"mapped", "needs_user_confirmation", "unsupported", "excluded_non_material", "blocked"}
_DYNAMIC_STRATEGY_TARGET_PREFIXES = (
    "signal.indicators.",
    "signal.rules.",
    "portfolio.params.",
    "portfolio.rules.",
    "execution.lot_size_config.by_symbol.",
    "robustness.parameter_perturbation.",
    "decision_policy.reject_if.",
    "decision_policy.promote_if.",
)


def validate_mapping_contract_file(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except OSError as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    return validate_mapping_contract(payload)


def validate_mapping_contract_for_builder_pass_file(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except OSError as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    return validate_mapping_contract_for_builder_pass(payload)


def validate_mapping_contract(payload: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return _result("fail", [{"path": "$", "message": "mapping contract must be a JSON object"}])
    if payload.get("schema_version") != MAPPING_CONTRACT_SCHEMA_VERSION:
        errors.append({"path": "schema_version", "message": f"must be {MAPPING_CONTRACT_SCHEMA_VERSION}"})
    _require_str(payload, "source_format", "source_format", errors)
    mappings = payload.get("field_mappings")
    if not isinstance(mappings, list):
        errors.append({"path": "field_mappings", "message": "must be a list"})
        mappings = []
    seen_source_fields: dict[str, int] = {}
    for index, item in enumerate(mappings):
        path = f"field_mappings[{index}]"
        if not isinstance(item, dict):
            errors.append({"path": path, "message": "must be an object"})
            continue
        source_field = _require_str(item, path, "source_field", errors)
        if source_field:
            if source_field in seen_source_fields:
                errors.append(
                    {
                        "path": f"{path}.source_field",
                        "message": f"duplicate source_field; first seen at field_mappings[{seen_source_fields[source_field]}]",
                    }
                )
            else:
                seen_source_fields[source_field] = index
        target_field = item.get("target_field")
        if target_field is not None and not isinstance(target_field, str):
            errors.append({"path": f"{path}.target_field", "message": "must be a string"})
        semantic = _require_enum(item, path, "semantic", _ALLOWED_SEMANTICS, errors)
        status = _require_enum(item, path, "status", _ALLOWED_STATUS, errors)
        if "confirmation_required" not in item or not isinstance(item["confirmation_required"], bool):
            errors.append({"path": f"{path}.confirmation_required", "message": "must be a boolean"})
        if "blocking" not in item or not isinstance(item["blocking"], bool):
            errors.append({"path": f"{path}.blocking", "message": "must be a boolean"})
        _require_str(item, path, "reason", errors)
        if status in {"mapped", "needs_user_confirmation"} and not item.get("target_field"):
            errors.append({"path": f"{path}.target_field", "message": "mapped fields require a target_field"})
        if (
            semantic == "strategy"
            and status in {"mapped", "needs_user_confirmation"}
            and isinstance(target_field, str)
            and target_field
            and not _is_effective_strategy_target_field(target_field)
        ):
            errors.append(
                {
                    "path": f"{path}.target_field",
                    "message": "strategy target_field must be an effective StrategySpec field path",
                }
            )
        if semantic == "strategy" and status == "excluded_non_material":
            errors.append({"path": f"{path}.status", "message": "strategy semantics must be mapped, blocked, or unsupported"})
        if semantic == "strategy" and status == "unsupported" and item.get("blocking") is not True:
            errors.append(
                {
                    "path": f"{path}.blocking",
                    "message": "unsupported strategy semantics require blocking=true",
                }
            )
        if semantic == "strategy" and status == "needs_user_confirmation" and item.get("blocking") is not True:
            errors.append(
                {
                    "path": f"{path}.blocking",
                    "message": "strategy semantics needing user confirmation require blocking=true",
                }
            )
        if status == "mapped" and item.get("blocking") is True:
            errors.append({"path": f"{path}.blocking", "message": "mapped fields cannot be blocking"})
        if status == "blocked" and item.get("blocking") is not True:
            errors.append({"path": f"{path}.blocking", "message": "blocked mappings require blocking=true"})
        if status == "needs_user_confirmation" and item.get("confirmation_required") is not True:
            errors.append(
                {
                    "path": f"{path}.confirmation_required",
                    "message": "needs_user_confirmation requires confirmation_required=true",
                }
            )
        if status in {"unsupported", "blocked"} and not item.get("reason"):
            errors.append({"path": f"{path}.reason", "message": "unsupported or blocked fields require a reason"})
    return _result("fail" if errors else "pass", errors)


def validate_mapping_contract_for_builder_pass(payload: Any) -> dict[str, Any]:
    result = validate_mapping_contract(payload)
    errors = list(result["errors"])
    if not isinstance(payload, dict):
        return _result("fail", errors)
    mappings = payload.get("field_mappings")
    if not isinstance(mappings, list):
        return _result("fail", errors)
    for index, item in enumerate(mappings):
        if not isinstance(item, dict):
            continue
        if item.get("semantic") != "strategy":
            continue
        status = item.get("status")
        if status in {"blocked", "unsupported", "needs_user_confirmation"} or item.get("blocking") is True:
            errors.append(
                {
                    "path": f"field_mappings[{index}].status",
                    "message": "builder pass requires strategy mappings to be mapped and non-blocking",
                }
            )
    return _result("fail" if errors else "pass", errors)


def _require_str(payload: dict[str, Any], path: str, field: str, errors: list[dict[str, str]]) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append({"path": f"{path}.{field}" if path != field else field, "message": "must be a non-empty string"})
        return ""
    return value


def _require_enum(
    payload: dict[str, Any],
    path: str,
    field: str,
    allowed: set[str],
    errors: list[dict[str, str]],
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or value not in allowed:
        errors.append({"path": f"{path}.{field}", "message": f"must be one of {sorted(allowed)}"})
        return ""
    return value


def _is_effective_strategy_target_field(field_path: str) -> bool:
    if field_path in _effective_strategy_field_paths():
        return True
    return any(field_path.startswith(prefix) for prefix in _DYNAMIC_STRATEGY_TARGET_PREFIXES)


@lru_cache(maxsize=1)
def _effective_strategy_field_paths() -> frozenset[str]:
    return frozenset(path for path, _value in _flatten_effective_fields(StrategySpec.template().to_effective_dict()))


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


def _result(status: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    return {"status": status, "errors": errors}
