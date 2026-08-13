"""Schema-backed quant operator manifest loading."""

from __future__ import annotations

import copy
import hashlib
import json
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from oxq.operators._schema import load_contract_schema
from oxq.operators._version import is_semantic_version
from oxq.operators.errors import InvalidManifestError, InvalidParameterError
from oxq.operators.types import OperatorAvailability, OperatorCausality, OperatorLifecycle, OperatorScope


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def manifest_digest(payload: Mapping[str, Any]) -> str:
    digest_payload = {key: value for key, value in payload.items() if key != "manifest_digest"}
    return "sha256:" + hashlib.sha256(_canonical_json(digest_payload).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class OperatorManifest:
    schema_version: int
    operator_id: str
    operator_version: str
    semantic_name: str
    distribution: str
    module: str
    callable: str
    execution_scope: OperatorScope
    lifecycle: OperatorLifecycle
    causality: OperatorCausality
    availability: OperatorAvailability
    raw: Mapping[str, Any]
    digest: str

    def validate_parameters(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        declarations = self.raw["parameters"]
        assert isinstance(declarations, Mapping)
        unknown = sorted(set(supplied) - set(declarations))
        if unknown:
            raise InvalidParameterError(
                f"unknown parameters: {', '.join(unknown)}",
                operator_id=self.operator_id,
                details={"parameters": unknown},
            )
        resolved: dict[str, Any] = {}
        for name, declaration_value in declarations.items():
            assert isinstance(declaration_value, Mapping)
            declaration = declaration_value
            if name in supplied:
                value = _thaw(supplied[name])
            elif "default" in declaration:
                value = _thaw(declaration["default"])
            elif declaration["required"]:
                raise InvalidParameterError(f"required parameter is missing: {name}", operator_id=self.operator_id)
            else:
                continue
            _validate_parameter_value(name, value, declaration, self.operator_id)
            resolved[name] = _thaw(value)
        return resolved


def load_operator_manifest(source: str | Path | Mapping[str, Any]) -> OperatorManifest:
    payload = _read_payload(source)
    schema = load_contract_schema("operator-manifest-v1.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        if list(error.absolute_path) == ["operator_version"]:
            raise InvalidManifestError(
                "operator_version must be semantic versioning",
                operator_id=_optional_operator_id(payload),
            )
        path = ".".join(str(part) for part in error.absolute_path)
        location = path or "manifest"
        raise InvalidManifestError(f"{location}: {error.message}", operator_id=_optional_operator_id(payload))
    if not is_semantic_version(payload["operator_version"]):
        raise InvalidManifestError("operator_version must be semantic versioning", operator_id=payload["operator_id"])
    if payload["lifecycle"] == "fit_transform" and "fitted_state" not in payload:
        raise InvalidManifestError("fit_transform manifest requires fitted_state", operator_id=payload["operator_id"])
    _validate_manifest_semantics(payload)
    actual_digest = manifest_digest(payload)
    declared_digest = payload.get("manifest_digest")
    if declared_digest is not None and declared_digest != actual_digest:
        raise InvalidManifestError(
            f"manifest_digest mismatch: declared={declared_digest}, actual={actual_digest}",
            operator_id=payload["operator_id"],
        )
    availability = payload["availability"]
    return OperatorManifest(
        schema_version=payload["schema_version"],
        operator_id=payload["operator_id"],
        operator_version=payload["operator_version"],
        semantic_name=payload["semantic_name"],
        distribution=payload["distribution"],
        module=payload["module"],
        callable=payload["callable"],
        execution_scope=OperatorScope(payload["execution_scope"]),
        lifecycle=OperatorLifecycle(payload["lifecycle"]),
        causality=OperatorCausality(payload["causality"]),
        availability=OperatorAvailability(availability["value"]),
        raw=_freeze(payload),
        digest=actual_digest,
    )


def _read_payload(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    path = Path(source)
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise InvalidManifestError(f"operator manifest is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidManifestError(f"operator manifest must contain an object: {path}")
    return payload


def _optional_operator_id(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("operator_id")
    return value if isinstance(value, str) else None


def _validate_parameter_value(name: str, value: Any, declaration: Mapping[str, Any], operator_id: str) -> None:
    expected = declaration["type"]
    valid = {
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, Mapping),
    }[expected]
    if not valid:
        raise InvalidParameterError(f"parameter {name} must have type {expected}", operator_id=operator_id)
    if "enum" in declaration and value not in declaration["enum"]:
        raise InvalidParameterError(f"parameter {name} must be one of {declaration['enum']}", operator_id=operator_id)
    if "minimum" in declaration and value < declaration["minimum"]:
        raise InvalidParameterError(f"parameter {name} is below minimum {declaration['minimum']}", operator_id=operator_id)
    if "maximum" in declaration and value > declaration["maximum"]:
        raise InvalidParameterError(f"parameter {name} exceeds maximum {declaration['maximum']}", operator_id=operator_id)


def _validate_manifest_semantics(payload: dict[str, Any]) -> None:
    operator_id = payload["operator_id"]
    inputs = payload["inputs"]
    required = set(inputs["required_columns"])
    optional = set(inputs["optional_columns"])
    overlap = sorted(required & optional)
    if overlap:
        raise InvalidManifestError(
            f"inputs required_columns and optional_columns overlap: {', '.join(overlap)}",
            operator_id=operator_id,
        )
    missing_dtypes = sorted((required | optional) - set(inputs["dtypes"]))
    if missing_dtypes:
        raise InvalidManifestError(
            f"inputs dtypes missing declarations: {', '.join(missing_dtypes)}",
            operator_id=operator_id,
        )
    parameters = payload["parameters"]
    for name, declaration in parameters.items():
        if declaration["type"] not in {"integer", "number"} and (
            "minimum" in declaration or "maximum" in declaration
        ):
            raise InvalidManifestError(
                f"parameter {name} bounds require a numeric parameter type",
                operator_id=operator_id,
            )
        if declaration["required"] and "default" in declaration:
            raise InvalidManifestError(
                f"required parameter {name} must not declare a default",
                operator_id=operator_id,
            )
        if "default" in declaration:
            try:
                _validate_parameter_value(name, declaration["default"], declaration, operator_id)
            except InvalidParameterError as exc:
                raise InvalidManifestError(
                    f"parameter {name} default is invalid: {exc}",
                    operator_id=operator_id,
                ) from exc
    outputs = payload["outputs"]
    warmup = outputs["warmup"]
    if warmup["kind"] == "parameter":
        warmup_name = warmup["parameter"]
        if warmup_name not in parameters:
            raise InvalidManifestError(
                f"outputs warmup references unknown parameter: {warmup_name}",
                operator_id=operator_id,
            )
        warmup_parameter = parameters[warmup_name]
        if warmup_parameter["type"] != "integer":
            raise InvalidManifestError(
                f"outputs warmup parameter must be an integer: {warmup_name}",
                operator_id=operator_id,
            )
        if not warmup_parameter["required"] and "default" not in warmup_parameter:
            raise InvalidManifestError(
                f"outputs warmup parameter must be required or declare a default: {warmup_name}",
                operator_id=operator_id,
            )
        if not warmup_parameter["affects_warmup"]:
            raise InvalidManifestError(
                f"outputs warmup parameter must set affects_warmup=true: {warmup_name}",
                operator_id=operator_id,
            )
    for field in outputs["fields"]:
        try:
            references = {
                name
                for _, name, _, _ in string.Formatter().parse(field["name_template"])
                if name is not None
            }
        except ValueError as exc:
            raise InvalidManifestError(
                f"output field template is invalid: {field['name_template']}",
                operator_id=operator_id,
            ) from exc
        unknown = sorted(references - set(parameters))
        if unknown:
            raise InvalidManifestError(
                f"output field template references unknown parameters: {', '.join(unknown)}",
                operator_id=operator_id,
            )
        unresolved = sorted(
            name
            for name in references
            if not parameters[name]["required"] and "default" not in parameters[name]
        )
        if unresolved:
            raise InvalidManifestError(
                "output field template parameters must be required or declare a default: "
                + ", ".join(unresolved),
                operator_id=operator_id,
            )
        inconsistent = sorted(name for name in references if not parameters[name]["affects_output_fields"])
        if inconsistent:
            raise InvalidManifestError(
                "output field template parameters must set affects_output_fields=true: "
                + ", ".join(inconsistent),
                operator_id=operator_id,
            )
