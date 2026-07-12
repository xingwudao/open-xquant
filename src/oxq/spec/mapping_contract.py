"""Validation contract for external source-to-SPEC mapping reports."""

from __future__ import annotations

import json
from collections.abc import Collection
from functools import lru_cache
from pathlib import Path
from typing import Any

from oxq.spec.schema import StrategySpec

MAPPING_CONTRACT_SCHEMA_VERSION = 1

_ALLOWED_SEMANTICS = {"strategy", "run", "report", "studio", "metadata", "unsupported"}
_ALLOWED_STATUS = {"mapped", "needs_user_confirmation", "unsupported", "excluded_non_material", "blocked"}
_IDEA_BRIEF_INVENTORY_EXCLUSIONS = {"conversation_hash", "schema_version"}
_DYNAMIC_STRATEGY_TARGET_PREFIXES = (
    "portfolio.params.",
    "execution.lot_size_config.by_symbol.",
    "robustness.parameter_perturbation.",
    "decision_policy.reject_if.",
    "decision_policy.promote_if.",
)


def validate_mapping_contract_file(
    path: str | Path,
    *,
    spec: StrategySpec | None = None,
    effective_field_paths: Collection[str] | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except (OSError, UnicodeError) as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    return validate_mapping_contract(payload, spec=spec, effective_field_paths=effective_field_paths)


def validate_mapping_contract_for_builder_pass_file(
    path: str | Path,
    *,
    spec: StrategySpec | None = None,
    effective_field_paths: Collection[str] | None = None,
    idea_brief_path: str | Path | None = None,
) -> dict[str, Any]:
    mapping_path = Path(path)
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("fail", [{"path": "$", "message": f"invalid JSON: {exc}"}])
    except (OSError, UnicodeError) as exc:
        return _result("fail", [{"path": "$", "message": str(exc)}])
    uses_idea_brief = isinstance(payload, dict) and payload.get("source_format") == "strategy_idea_brief"
    if not uses_idea_brief:
        idea_brief_path = None
    if idea_brief_path is None and uses_idea_brief:
        idea_brief_path, discovery_error = _discover_manifest_owned_idea_brief(mapping_path)
        if discovery_error is not None:
            return _result("fail", [{"path": "idea_brief", "message": discovery_error}])
    idea_brief: Any | None = None
    if idea_brief_path is not None:
        try:
            idea_brief = json.loads(Path(idea_brief_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _result("fail", [{"path": "idea_brief", "message": f"invalid JSON: {exc}"}])
        except (OSError, UnicodeError) as exc:
            return _result("fail", [{"path": "idea_brief", "message": str(exc)}])
    return validate_mapping_contract_for_builder_pass(
        payload,
        spec=spec,
        effective_field_paths=effective_field_paths,
        idea_brief=idea_brief,
    )


def validate_mapping_contract(
    payload: Any,
    *,
    spec: StrategySpec | None = None,
    effective_field_paths: Collection[str] | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return _result("fail", [{"path": "$", "message": "mapping contract must be a JSON object"}])
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != MAPPING_CONTRACT_SCHEMA_VERSION:
        errors.append({"path": "schema_version", "message": f"must be {MAPPING_CONTRACT_SCHEMA_VERSION}"})
    _require_str(payload, "source_format", "source_format", errors)
    source_fields = payload.get("source_fields")
    if source_fields is not None:
        if not isinstance(source_fields, list):
            errors.append({"path": "source_fields", "message": "must be a list"})
        else:
            seen_inventory: dict[str, int] = {}
            for index, source_field in enumerate(source_fields):
                if not isinstance(source_field, str) or not source_field.strip():
                    errors.append({"path": f"source_fields[{index}]", "message": "must be a non-empty string"})
                    continue
                if source_field in seen_inventory:
                    errors.append(
                        {
                            "path": f"source_fields[{index}]",
                            "message": (
                                "duplicate source field; first seen at "
                                f"source_fields[{seen_inventory[source_field]}]"
                            ),
                        }
                    )
                else:
                    seen_inventory[source_field] = index
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
            and not _is_effective_strategy_target_field(
                target_field,
                spec=spec,
                effective_field_paths=effective_field_paths,
            )
        ):
            errors.append(
                {
                    "path": f"{path}.target_field",
                    "message": "strategy target_field must be an effective StrategySpec field path",
                }
            )
        if semantic == "strategy" and status == "excluded_non_material":
            errors.append({"path": f"{path}.status", "message": "strategy semantics must be mapped, blocked, or unsupported"})
        if semantic == "unsupported" and status not in {"unsupported", "blocked"}:
            errors.append(
                {
                    "path": f"{path}.status",
                    "message": "unsupported semantics must use unsupported or blocked status",
                }
            )
        if semantic == "unsupported" and item.get("blocking") is not True:
            errors.append(
                {
                    "path": f"{path}.blocking",
                    "message": "unsupported semantics require blocking=true",
                }
            )
        if status == "unsupported" and item.get("blocking") is not True:
            errors.append(
                {
                    "path": f"{path}.blocking",
                    "message": "unsupported mappings require blocking=true",
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
        if status != "needs_user_confirmation" and item.get("confirmation_required") is True:
            errors.append(
                {
                    "path": f"{path}.confirmation_required",
                    "message": "confirmation_required=true requires needs_user_confirmation status",
                }
            )
        if status in {"unsupported", "blocked"} and not item.get("reason"):
            errors.append({"path": f"{path}.reason", "message": "unsupported or blocked fields require a reason"})
    return _result("fail" if errors else "pass", errors)


def validate_mapping_contract_for_builder_pass(
    payload: Any,
    *,
    spec: StrategySpec | None = None,
    effective_field_paths: Collection[str] | None = None,
    idea_brief: Any | None = None,
) -> dict[str, Any]:
    result = validate_mapping_contract(payload, spec=spec, effective_field_paths=effective_field_paths)
    errors = list(result["errors"])
    if spec is None and not effective_field_paths:
        errors.append(
            {
                "path": "$",
                "message": (
                    "builder pass requires a concrete effective leaf inventory "
                    "from spec or effective_field_paths"
                ),
            }
        )
    if not isinstance(payload, dict):
        return _result("fail", errors)
    if payload.get("source_format") == "strategy_idea_brief" and idea_brief is None:
        errors.append(
            {
                "path": "idea_brief",
                "message": "builder pass requires the canonical strategy idea brief inventory",
            }
        )
    mappings = payload.get("field_mappings")
    if not isinstance(mappings, list):
        return _result("fail", errors)
    source_fields = payload.get("source_fields")
    if not isinstance(source_fields, list) or not source_fields:
        errors.append({"path": "source_fields", "message": "builder pass requires a non-empty source inventory"})
    else:
        inventory_positions = {
            source_field: index
            for index, source_field in enumerate(source_fields)
            if isinstance(source_field, str) and source_field.strip()
        }
        mapping_positions: dict[str, list[int]] = {}
        for index, item in enumerate(mappings):
            if isinstance(item, dict) and isinstance(item.get("source_field"), str):
                mapping_positions.setdefault(item["source_field"], []).append(index)
        for source_field, inventory_index in inventory_positions.items():
            if source_field not in mapping_positions:
                errors.append(
                    {
                        "path": f"source_fields[{inventory_index}]",
                        "message": "missing field_mappings row for declared source field",
                    }
                )
        for source_field, indexes in mapping_positions.items():
            if source_field not in inventory_positions:
                for index in indexes:
                    errors.append(
                        {
                            "path": f"field_mappings[{index}].source_field",
                            "message": "source field is not declared in source_fields inventory",
                        }
                    )
        if idea_brief is not None:
            if not isinstance(idea_brief, dict):
                errors.append({"path": "idea_brief", "message": "canonical strategy idea brief must be an object"})
            else:
                expected_source_fields = _flatten_idea_brief_fields(idea_brief)
                inventory = set(inventory_positions)
                for source_field, inventory_index in inventory_positions.items():
                    if source_field not in expected_source_fields:
                        errors.append(
                            {
                                "path": f"source_fields[{inventory_index}]",
                                "message": "source field is not present in the canonical strategy idea brief",
                            }
                        )
                missing_source_fields = sorted(expected_source_fields - inventory)
                if missing_source_fields:
                    errors.append(
                        {
                            "path": "source_fields",
                            "message": (
                                "source inventory is missing canonical strategy idea brief fields: "
                                + ", ".join(missing_source_fields)
                            ),
                        }
                    )
    for index, item in enumerate(mappings):
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if item.get("semantic") == "unsupported":
            errors.append(
                {
                    "path": f"field_mappings[{index}].semantic",
                    "message": "unsupported semantics cannot pass the builder gate",
                }
            )
        if item.get("confirmation_required") is True:
            errors.append(
                {
                    "path": f"field_mappings[{index}].confirmation_required",
                    "message": "builder pass requires confirmation_required=false",
                }
            )
        if status in {"blocked", "unsupported", "needs_user_confirmation"} or item.get("blocking") is True:
            errors.append(
                {
                    "path": f"field_mappings[{index}].status",
                    "message": "builder pass requires mappings to be mapped or excluded_non_material and non-blocking",
                }
            )
    return _result("fail" if errors else "pass", errors)


def _flatten_idea_brief_fields(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        if not value and prefix:
            return {prefix}
        fields: set[str] = set()
        for key in sorted(value):
            if not prefix and key in _IDEA_BRIEF_INVENTORY_EXCLUSIONS:
                continue
            child_path = f"{prefix}.{key}" if prefix else str(key)
            fields.update(_flatten_idea_brief_fields(value[key], child_path))
        return fields
    if isinstance(value, list):
        if not value or all(not isinstance(item, (dict, list)) for item in value):
            return {prefix} if prefix else set()
        fields = set()
        for index, item in enumerate(value):
            fields.update(_flatten_idea_brief_fields(item, f"{prefix}[{index}]"))
        return fields
    return {prefix} if prefix else set()


def _discover_manifest_owned_idea_brief(
    mapping_path: Path,
) -> tuple[Path | None, str | None]:
    resolved_mapping = mapping_path.resolve(strict=False)
    for workspace_root in resolved_mapping.parents:
        if not (workspace_root / ".open-xquant" / "workspace.yaml").is_file():
            continue
        from oxq.spec.audit_schema import _active_governed_provenance_paths

        phase_paths, errors = _active_governed_provenance_paths(workspace_root)
        if errors:
            first = errors[0]
            return None, f"{first['path']}: {first['message']}"
        if phase_paths is None:
            return None, "workspace does not define manifest-owned strategy idea phases"
        expected_mapping = (phase_paths["04_spec_build"] / "spec_mapping_contract.json").resolve(strict=False)
        if resolved_mapping != expected_mapping:
            return None, "mapping contract is not the active manifest-owned spec_mapping_contract.json"
        return phase_paths["01_brainstorm"] / "strategy_idea_brief.json", None
    return None, None


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


def _is_effective_strategy_target_field(
    field_path: str,
    *,
    spec: StrategySpec | None,
    effective_field_paths: Collection[str] | None,
) -> bool:
    if not field_path or any(not part for part in field_path.split(".")):
        return False
    if effective_field_paths is not None:
        return field_path in effective_field_paths
    if spec is not None:
        effective_leaf_paths = {
            path
            for path, value in _flatten_effective_fields(spec.to_effective_dict())
            if not isinstance(value, dict)
        }
        return field_path in effective_leaf_paths
    if field_path in _template_effective_strategy_field_paths():
        return True
    structured_dynamic = _is_structured_dynamic_target_field(field_path)
    if structured_dynamic is not None:
        return structured_dynamic
    return any(field_path.startswith(prefix) for prefix in _DYNAMIC_STRATEGY_TARGET_PREFIXES)


def _is_structured_dynamic_target_field(field_path: str) -> bool | None:
    structures = (
        ("signal.indicators.", {"type", "lag_bars"}),
        ("signal.rules.", {"type", "output_domain"}),
        ("portfolio.rules.", {"type"}),
    )
    for prefix, direct_fields in structures:
        if not field_path.startswith(prefix):
            continue
        parts = field_path[len(prefix):].split(".")
        if len(parts) < 2 or any(not part for part in parts):
            return False
        structural_path = parts[1:]
        if len(structural_path) == 1 and structural_path[0] in direct_fields:
            return True
        return structural_path[0] == "params"
    return None


@lru_cache(maxsize=1)
def _template_effective_strategy_field_paths() -> frozenset[str]:
    return frozenset(_collect_effective_field_paths(StrategySpec.template().to_effective_dict()))


def _collect_effective_field_paths(value: Any, prefix: str = "") -> set[str]:
    fields = {prefix} if prefix else set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            fields.update(_collect_effective_field_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.update(_collect_effective_field_paths(child, f"{prefix}[{index}]"))
    return fields


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
