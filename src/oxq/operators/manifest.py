"""Schema-backed quant operator manifest loading."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from pandas.api.types import is_float_dtype, is_integer_dtype, pandas_dtype

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
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _find_recursive_container(value: Any, path: str = "manifest") -> str | None:
    stack: list[tuple[Any, str, bool]] = [(value, path, False)]
    active: set[int] = set()
    while stack:
        current, current_path, exiting = stack.pop()
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        identity = id(current)
        if exiting:
            active.remove(identity)
            continue
        if identity in active:
            return current_path
        active.add(identity)
        stack.append((current, current_path, True))
        if isinstance(current, Mapping):
            children = [(item, f"{current_path}.{key}") for key, item in current.items()]
        else:
            children = [(item, f"{current_path}[{index}]") for index, item in enumerate(current)]
        for child, child_path in reversed(children):
            stack.append((child, child_path, False))
    return None


def _reject_recursive_containers(value: Any) -> None:
    recursive_path = _find_recursive_container(value)
    if recursive_path is not None:
        raise InvalidManifestError(f"manifest contains a recursive or cyclic container: {recursive_path}")


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
                value = supplied[name]
                if isinstance(supplied, MappingProxyType):
                    if _find_recursive_container(value, f"parameter {name}") is not None:
                        raise InvalidParameterError(
                            f"parameter {name} must form a finite JSON tree",
                            operator_id=self.operator_id,
                        )
                    value = _thaw(value)
            elif "default" in declaration:
                value = _thaw(declaration["default"])
            elif declaration["required"]:
                raise InvalidParameterError(f"required parameter is missing: {name}", operator_id=self.operator_id)
            else:
                continue
            _validate_parameter_value(name, value, declaration, self.operator_id)
            resolved[name] = _thaw(value)
        _validate_resolved_output_names(self.raw["outputs"], resolved, self.operator_id)
        return resolved


def load_operator_manifest(source: str | Path | Mapping[str, Any]) -> OperatorManifest:
    payload = _read_payload(source)
    non_string_key_path = _find_non_string_mapping_key(payload)
    if non_string_key_path is not None:
        raise InvalidManifestError(
            f"manifest mapping keys must be strings: {non_string_key_path}",
            operator_id=_optional_operator_id(payload),
        )
    nonfinite_path = _find_nonfinite_number(payload)
    if nonfinite_path is not None:
        raise InvalidManifestError(
            f"manifest numeric declaration must be finite: {nonfinite_path}",
            operator_id=_optional_operator_id(payload),
        )
    non_json_value = _find_non_json_value(payload)
    if non_json_value is not None:
        path, type_name = non_json_value
        raise InvalidManifestError(
            f"manifest values must form a JSON tree: {path} contains {type_name}",
            operator_id=_optional_operator_id(payload),
        )
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
        _reject_recursive_containers(source)
        payload = _thaw(source)
        assert isinstance(payload, dict)
        return payload
    path = Path(source)
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            parsed = json.loads(raw, object_pairs_hook=_JsonObjectPairs)
            payload = _materialize_json(parsed, path=path)
        else:
            payload = _load_yaml(raw, path=path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise InvalidManifestError(f"operator manifest is invalid: {path}: {exc}") from exc
    _reject_recursive_containers(payload)
    if not isinstance(payload, dict):
        raise InvalidManifestError(f"operator manifest must contain an object: {path}")
    return payload


class _JsonObjectPairs(list[tuple[str, Any]]):
    """Distinguish JSON objects from arrays until duplicate keys are checked."""


def _materialize_json(value: Any, *, path: Path, location: str = "manifest") -> Any:
    if isinstance(value, _JsonObjectPairs):
        result: dict[str, Any] = {}
        for key, item in value:
            if key in result:
                raise InvalidManifestError(f"operator manifest is invalid: {path}: {location}: duplicate mapping key: {key}")
            result[key] = _materialize_json(item, path=path, location=f"{location}.{key}")
        return result
    if isinstance(value, list):
        return [_materialize_json(item, path=path, location=f"{location}[{index}]") for index, item in enumerate(value)]
    return value


def _load_yaml(raw: str, *, path: Path) -> Any:
    loader = yaml.SafeLoader(raw)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _reject_duplicate_yaml_keys(node, loader=loader, path=path)
        return loader.construct_document(node)
    finally:
        loader.dispose()


def _reject_duplicate_yaml_keys(
    node: Any,
    *,
    loader: Any,
    path: Path,
    location: str = "manifest",
    active: set[int] | None = None,
) -> None:
    active = set() if active is None else active
    identity = id(node)
    if identity in active:
        return
    active.add(identity)
    try:
        if isinstance(node, yaml.MappingNode):
            seen: set[Any] = set()
            for key_node, value_node in node.value:
                key = key_node.value if key_node.tag == "tag:yaml.org,2002:merge" else loader.construct_object(key_node, deep=True)
                try:
                    duplicate = key in seen
                    seen.add(key)
                except TypeError:
                    duplicate = False
                if duplicate:
                    raise InvalidManifestError(f"operator manifest is invalid: {path}: {location}: duplicate mapping key: {key}")
                child_location = f"{location}.{key}" if isinstance(key, str) else location
                _reject_duplicate_yaml_keys(
                    value_node,
                    loader=loader,
                    path=path,
                    location=child_location,
                    active=active,
                )
        elif isinstance(node, yaml.SequenceNode):
            for index, item in enumerate(node.value):
                _reject_duplicate_yaml_keys(
                    item,
                    loader=loader,
                    path=path,
                    location=f"{location}[{index}]",
                    active=active,
                )
    finally:
        active.remove(identity)


def _optional_operator_id(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("operator_id")
    return value if isinstance(value, str) else None


def _find_non_string_mapping_key(value: Any, path: str = "manifest") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path}[{key!r}]"
            found = _find_non_string_mapping_key(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_non_string_mapping_key(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _find_nonfinite_number(value: Any, path: str = "manifest") -> str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return path
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _find_nonfinite_number(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_nonfinite_number(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _find_non_json_value(value: Any, path: str = "manifest") -> tuple[str, str] | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _find_non_json_value(item, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_non_json_value(item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    return path, type(value).__name__


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
    if expected in {"array", "object"}:
        _validate_parameter_json_tree(name, value, operator_id)
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidParameterError(f"parameter {name} must be finite", operator_id=operator_id)
    if "enum" in declaration and value not in declaration["enum"]:
        raise InvalidParameterError(f"parameter {name} must be one of {declaration['enum']}", operator_id=operator_id)
    if "minimum" in declaration and value < declaration["minimum"]:
        raise InvalidParameterError(f"parameter {name} is below minimum {declaration['minimum']}", operator_id=operator_id)
    if "maximum" in declaration and value > declaration["maximum"]:
        raise InvalidParameterError(f"parameter {name} exceeds maximum {declaration['maximum']}", operator_id=operator_id)


def _validate_parameter_json_tree(name: str, value: Any, operator_id: str) -> None:
    path = f"parameter {name}"
    if _find_recursive_container(value, path) is not None:
        raise InvalidParameterError(f"parameter {name} must form a finite JSON tree", operator_id=operator_id)
    non_string_key_path = _find_non_string_mapping_key(value, path)
    if non_string_key_path is not None:
        raise InvalidParameterError(
            f"parameter {name} JSON object must use string keys: {non_string_key_path}",
            operator_id=operator_id,
        )
    nonfinite_path = _find_nonfinite_number(value, path)
    if nonfinite_path is not None:
        raise InvalidParameterError(
            f"parameter {name} JSON numbers must be finite: {nonfinite_path}",
            operator_id=operator_id,
        )
    non_json_value = _find_non_json_value(value, path)
    if non_json_value is not None:
        invalid_path, type_name = non_json_value
        raise InvalidParameterError(
            f"parameter {name} must form a JSON tree: {invalid_path} contains {type_name}",
            operator_id=operator_id,
        )


def _validate_resolved_output_names(
    outputs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    operator_id: str,
) -> None:
    resolved_names: set[str] = set()
    for field in outputs["fields"]:
        template = field["name_template"]
        try:
            resolved_name = template.format(**parameters)
        except (AttributeError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise InvalidParameterError(
                f"output field template cannot format resolved parameters: {template}",
                operator_id=operator_id,
                details={"name_template": template},
            ) from exc
        if not resolved_name:
            raise InvalidParameterError(
                "output field name must not be empty",
                operator_id=operator_id,
                details={"name_template": template},
            )
        if resolved_name in {"date", "code"}:
            raise InvalidParameterError(
                f"output field resolves to reserved QuantPanel key: {resolved_name}",
                operator_id=operator_id,
                details={"name_template": template, "resolved_name": resolved_name},
            )
        if resolved_name in resolved_names:
            raise InvalidParameterError(
                f"duplicate output field name: {resolved_name}",
                operator_id=operator_id,
                details={"name_template": template, "resolved_name": resolved_name},
            )
        resolved_names.add(resolved_name)


def _validate_manifest_semantics(payload: dict[str, Any]) -> None:
    operator_id = payload["operator_id"]
    inputs = payload["inputs"]
    if payload["execution_scope"] == "cross_section" and inputs["min_history"] > 1:
        raise InvalidManifestError(
            "cross_section execution_scope requires inputs min_history=1",
            operator_id=operator_id,
        )
    required = set(inputs["required_columns"])
    optional = set(inputs["optional_columns"])
    reserved_optional = sorted(optional & {"date", "code"})
    if reserved_optional:
        raise InvalidManifestError(
            "inputs optional_columns must not contain reserved QuantPanel keys date or code: " + ", ".join(reserved_optional),
            operator_id=operator_id,
        )
    overlap = sorted(required & optional)
    if overlap:
        raise InvalidManifestError(
            f"inputs required_columns and optional_columns overlap: {', '.join(overlap)}",
            operator_id=operator_id,
        )
    expected_dtypes = required | optional
    declared_dtypes = set(inputs["dtypes"])
    missing_dtypes = sorted(expected_dtypes - declared_dtypes)
    unexpected_dtypes = sorted(declared_dtypes - expected_dtypes)
    if missing_dtypes or unexpected_dtypes:
        discrepancies = []
        if missing_dtypes:
            discrepancies.append(f"missing declarations: {', '.join(missing_dtypes)}")
        if unexpected_dtypes:
            discrepancies.append(f"unexpected declarations: {', '.join(unexpected_dtypes)}")
        raise InvalidManifestError(
            "inputs dtypes keys must exactly match required_columns and optional_columns; " + "; ".join(discrepancies),
            operator_id=operator_id,
        )
    parameters = payload["parameters"]
    for name, declaration in parameters.items():
        if declaration["type"] not in {"integer", "number"} and ("minimum" in declaration or "maximum" in declaration):
            raise InvalidManifestError(
                f"parameter {name} bounds require a numeric parameter type",
                operator_id=operator_id,
            )
        if "minimum" in declaration and "maximum" in declaration and declaration["minimum"] > declaration["maximum"]:
            raise InvalidManifestError(
                f"parameter {name} minimum must not exceed maximum",
                operator_id=operator_id,
            )
        if (
            declaration["type"] == "integer"
            and "minimum" in declaration
            and "maximum" in declaration
            and math.ceil(declaration["minimum"]) > math.floor(declaration["maximum"])
        ):
            raise InvalidManifestError(
                f"parameter {name} integer domain is empty",
                operator_id=operator_id,
            )
        if "enum" in declaration:
            enum_declaration = {key: value for key, value in declaration.items() if key != "enum"}
            for index, member in enumerate(declaration["enum"]):
                try:
                    _validate_parameter_value(name, member, enum_declaration, operator_id)
                except InvalidParameterError as exc:
                    raise InvalidManifestError(
                        f"parameter {name} enum member {index} is invalid: {exc}",
                        operator_id=operator_id,
                    ) from exc
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
    has_multiple_fields = len(outputs["fields"]) > 1
    if outputs["multiple"] is not has_multiple_fields:
        raise InvalidManifestError(
            "outputs multiple must be true if and only if fields contains more than one field",
            operator_id=operator_id,
        )
    warmup = outputs["warmup"]
    if warmup["kind"] == "fixed":
        rows = warmup["rows"]
        if not isinstance(rows, int) or isinstance(rows, bool):
            raise InvalidManifestError(
                "outputs warmup fixed rows must be an integer",
                operator_id=operator_id,
            )
    else:
        offset = warmup.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise InvalidManifestError(
                "outputs warmup parameter offset must be an integer",
                operator_id=operator_id,
            )
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
        if "default" in warmup_parameter and warmup_parameter["default"] + offset < 0:
            raise InvalidManifestError(
                f"outputs warmup parameter default plus offset must be non-negative: {warmup_name}",
                operator_id=operator_id,
            )
        if "enum" in warmup_parameter:
            unsafe_values = [value for value in warmup_parameter["enum"] if value + offset < 0]
            if unsafe_values:
                raise InvalidManifestError(
                    f"outputs warmup parameter enum plus offset must always be non-negative: {warmup_name}",
                    operator_id=operator_id,
                )
        elif "minimum" not in warmup_parameter:
            raise InvalidManifestError(
                f"outputs warmup parameter domain must declare a minimum to remain non-negative: {warmup_name}",
                operator_id=operator_id,
            )
        elif math.ceil(warmup_parameter["minimum"]) + offset < 0:
            raise InvalidManifestError(
                f"outputs warmup parameter minimum plus offset must be non-negative: {warmup_name}",
                operator_id=operator_id,
            )
    resolved_output_names: set[str] = set()
    for field in outputs["fields"]:
        has_bounds = "minimum" in field or "maximum" in field
        if has_bounds and not _is_ordered_numeric_dtype(field["dtype"]):
            raise InvalidManifestError(
                "output field bounds require a numeric dtype",
                operator_id=operator_id,
            )
        if "minimum" in field and "maximum" in field and field["minimum"] > field["maximum"]:
            raise InvalidManifestError(
                "output field minimum must not exceed maximum",
                operator_id=operator_id,
            )
        try:
            references = _template_references(field["name_template"])
        except ValueError as exc:
            raise InvalidManifestError(
                f"output field template is invalid: {field['name_template']}: {exc}",
                operator_id=operator_id,
            ) from exc
        unknown = sorted(references - set(parameters))
        if unknown:
            raise InvalidManifestError(
                f"output field template references unknown parameters: {', '.join(unknown)}",
                operator_id=operator_id,
            )
        composite = sorted(name for name in references if parameters[name]["type"] in {"array", "object"})
        if composite:
            raise InvalidManifestError(
                "output field template parameters must be scalar: " + ", ".join(composite),
                operator_id=operator_id,
            )
        unresolved = sorted(name for name in references if not parameters[name]["required"] and "default" not in parameters[name])
        if unresolved:
            raise InvalidManifestError(
                "output field template parameters must be required or declare a default: " + ", ".join(unresolved),
                operator_id=operator_id,
            )
        inconsistent = sorted(name for name in references if not parameters[name]["affects_output_fields"])
        if inconsistent:
            raise InvalidManifestError(
                "output field template parameters must set affects_output_fields=true: " + ", ".join(inconsistent),
                operator_id=operator_id,
            )
        try:
            _validate_template_parameter_types(field["name_template"], references, parameters)
        except (AttributeError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise InvalidManifestError(
                f"output field template format is incompatible with declared parameter types: {field['name_template']}",
                operator_id=operator_id,
            ) from exc
        try:
            _validate_template_parameter_domains(field["name_template"], parameters)
        except ValueError as exc:
            raise InvalidManifestError(
                f"output field template format is incompatible with declared parameter domain: {field['name_template']}",
                operator_id=operator_id,
            ) from exc
        if not references or all("default" in parameters[name] for name in references):
            defaults = {name: parameters[name]["default"] for name in references}
            try:
                resolved_name = field["name_template"].format(**defaults)
            except (AttributeError, KeyError, IndexError, ValueError, TypeError) as exc:
                raise InvalidManifestError(
                    f"output field template cannot format declared defaults: {field['name_template']}",
                    operator_id=operator_id,
                ) from exc
            if not resolved_name:
                raise InvalidManifestError(
                    "output field name must not be empty",
                    operator_id=operator_id,
                )
            if resolved_name in {"date", "code"}:
                raise InvalidManifestError(
                    f"output field resolves to reserved QuantPanel key: {resolved_name}",
                    operator_id=operator_id,
                )
            if resolved_name in resolved_output_names:
                raise InvalidManifestError(
                    f"duplicate output field name: {resolved_name}",
                    operator_id=operator_id,
                )
            resolved_output_names.add(resolved_name)


def _template_references(template: str) -> set[str]:
    references: set[str] = set()
    for _, field_name, format_spec, _ in string.Formatter().parse(template):
        if field_name is not None:
            if "." in field_name or "[" in field_name:
                raise ValueError("template fields must reference parameters directly")
            references.add(field_name)
        if format_spec:
            references.update(_template_references(format_spec))
    return references


def _validate_template_parameter_types(
    template: str,
    references: set[str],
    parameters: Mapping[str, Mapping[str, Any]],
) -> None:
    representative_values = {
        "integer": 0,
        "number": 0.0,
        "string": "",
        "boolean": False,
    }
    samples = {name: representative_values[parameters[name]["type"]] for name in references}
    template.format(**samples)


def _validate_template_parameter_domains(
    template: str,
    parameters: Mapping[str, Mapping[str, Any]],
) -> None:
    for _, field_name, format_spec, _ in string.Formatter().parse(template):
        if field_name is None:
            continue
        if format_spec and format_spec.endswith("c") and parameters[field_name]["type"] == "integer":
            _validate_code_point_parameter_domain(parameters[field_name])
        if format_spec:
            _validate_template_parameter_domains(format_spec, parameters)


def _validate_code_point_parameter_domain(declaration: Mapping[str, Any]) -> None:
    minimum_code_point = 0
    maximum_code_point = 0x10FFFF
    if "enum" in declaration:
        values = declaration["enum"]
        if all(minimum_code_point <= value <= maximum_code_point for value in values):
            return
        raise ValueError("integer enum contains a value outside the Unicode code point range")
    if "minimum" not in declaration or "maximum" not in declaration:
        raise ValueError("integer code point format requires a bounded parameter domain")
    effective_minimum = math.ceil(declaration["minimum"])
    effective_maximum = math.floor(declaration["maximum"])
    if effective_minimum > effective_maximum or effective_minimum < minimum_code_point or effective_maximum > maximum_code_point:
        raise ValueError("integer bounds are not contained in the Unicode code point range")


def _is_ordered_numeric_dtype(dtype: str) -> bool:
    try:
        parsed = pandas_dtype(dtype)
    except TypeError:
        return False
    return bool(is_integer_dtype(parsed) or is_float_dtype(parsed))
