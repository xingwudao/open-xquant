"""Normative cross-field checks for Quant Operator Contract v1.

Run the published Draft 2020-12 schema before these semantic checks.  This
module is deliberately standalone and has no dependency on ``oxq``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any


class ContractValidationError(ValueError):
    """Raised when a v1 cross-field semantic rule is violated."""


def _sha256_path_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the contract SHA-256 identifier for exact raw file bytes."""

    return f"sha256:{_sha256_path_hex(Path(path))}"


def _source_file_path(path: str) -> PurePosixPath:
    parts = path.split("/")
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ContractValidationError(
            f"source file must be a normalized relative POSIX path: {path!r}"
        )
    return candidate


def sha256_source_tree(
    root: str | os.PathLike[str], source_files: list[str]
) -> str:
    """Hash an explicit source file set using the v1 source-tree profile."""

    if not source_files:
        raise ContractValidationError("source file list must not be empty")
    if len(source_files) != len(set(source_files)):
        raise ContractValidationError("duplicate source file")

    root_path = Path(root).resolve(strict=True)
    digest = hashlib.sha256()
    for relative in sorted(source_files):
        posix_path = _source_file_path(relative)
        file_path = root_path.joinpath(*posix_path.parts).resolve(strict=True)
        if not file_path.is_relative_to(root_path) or not file_path.is_file():
            raise ContractValidationError(
                f"source file escapes root or is not a file: {relative!r}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_path_hex(file_path).encode("ascii"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _is_declared_dtype(value: object, dtype: str) -> bool:
    if dtype == "boolean":
        return isinstance(value, bool)
    if dtype == "int64":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(2**63) <= value < 2**63
        )
    if dtype == "float64":
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, int) and not isinstance(value, bool):
            try:
                return math.isfinite(value)
            except OverflowError:
                return False
        return False
    if dtype == "string":
        return isinstance(value, str)
    if dtype == "date":
        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if dtype == "datetime":
        if not isinstance(value, str) or "T" not in value:
            return False
        try:
            datetime.fromisoformat(value)
        except ValueError:
            return False
        return True
    return False


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True

    value_type = type(value)
    if (
        value_type.__module__.startswith("pandas.")
        and value_type.__name__ in {"NAType", "NaTType"}
    ):
        return True

    return type(value) is float and math.isnan(value)


def validate_quant_panel(panel: Mapping[str, Any]) -> None:
    """Validate QuantPanel relationships that JSON Schema cannot express."""

    columns = panel["columns"]
    declared_dtypes: dict[str, str] = {}
    required_columns: set[str] = set()
    for column in columns:
        name = column["name"]
        if name in declared_dtypes:
            raise ContractValidationError(f"duplicate QuantPanel column: {name!r}")
        declared_dtypes[name] = column["dtype"]
        if column["required"]:
            required_columns.add(name)

    allowed_fields = {"date", "code", *declared_dtypes}
    seen_keys: set[tuple[str, str]] = set()
    for record_index, record in enumerate(panel["records"]):
        undeclared_fields = set(record).difference(allowed_fields)
        if undeclared_fields:
            field = sorted(undeclared_fields)[0]
            raise ContractValidationError(
                f"undeclared QuantPanel field: {field!r} in record {record_index}"
            )

        missing_required = required_columns.difference(record)
        if missing_required:
            field = sorted(missing_required)[0]
            raise ContractValidationError(
                f"missing required QuantPanel column: {field!r} in record {record_index}"
            )

        for field, dtype in declared_dtypes.items():
            if (
                field in record
                and not _is_missing_value(record[field])
                and not _is_declared_dtype(record[field], dtype)
            ):
                raise ContractValidationError(
                    f"invalid QuantPanel value for {field!r}: expected {dtype}"
                )

        key = (record["date"], record["code"])
        if key in seen_keys:
            raise ContractValidationError(f"duplicate QuantPanel key: {key!r}")
        seen_keys.add(key)


def _validate_input_columns(input_contract: Mapping[str, Any]) -> None:
    required = input_contract["required_columns"]
    optional = input_contract["optional_columns"]
    for columns in (required, optional):
        if len(columns) != len(set(columns)):
            raise ContractValidationError("duplicate input column")
    overlap = set(required).intersection(optional)
    if overlap:
        raise ContractValidationError(
            f"required and optional input columns overlap: {sorted(overlap)!r}"
        )

    executable_sort_keys = {"date", "code", *required}
    for sort_key in input_contract.get("required_sort_order", []):
        if sort_key not in executable_sort_keys:
            raise ContractValidationError(
                f"invalid required sort key: {sort_key!r}; expected 'date', "
                "'code', or a required input column"
            )


_NUMERIC_CONSTRAINTS = {
    "minimum",
    "maximum",
    "exclusive_minimum",
    "exclusive_maximum",
}
_STRING_CONSTRAINTS = {"min_length", "max_length", "pattern"}
_ARRAY_CONSTRAINTS = {"min_items", "max_items"}


def _is_parameter_type(value: object, parameter_type: str) -> bool:
    if parameter_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if parameter_type == "boolean":
        return isinstance(value, bool)
    if parameter_type == "string":
        return isinstance(value, str)
    if parameter_type == "array":
        return isinstance(value, list)
    if parameter_type == "object":
        return isinstance(value, dict)
    return False


def _validate_constraint_applicability(
    parameter_type: str, constraints: Mapping[str, Any]
) -> None:
    applicable = {"enum"}
    if parameter_type in {"integer", "number"}:
        applicable.update(_NUMERIC_CONSTRAINTS)
    elif parameter_type == "string":
        applicable.update(_STRING_CONSTRAINTS)
    elif parameter_type == "array":
        applicable.update(_ARRAY_CONSTRAINTS)
    invalid = set(constraints).difference(applicable)
    if invalid:
        name = sorted(invalid)[0]
        raise ContractValidationError(
            f"constraint {name!r} is not valid for parameter type {parameter_type!r}"
        )


def _validate_constraint_coherence(constraints: Mapping[str, Any]) -> None:
    if (
        "min_length" in constraints
        and "max_length" in constraints
        and constraints["min_length"] > constraints["max_length"]
    ):
        raise ContractValidationError("conflicting constraints: min_length > max_length")
    if (
        "min_items" in constraints
        and "max_items" in constraints
        and constraints["min_items"] > constraints["max_items"]
    ):
        raise ContractValidationError("conflicting constraints: min_items > max_items")

    lower_bounds = [
        (constraints[name], name == "exclusive_minimum")
        for name in ("minimum", "exclusive_minimum")
        if name in constraints
    ]
    upper_bounds = [
        (constraints[name], name == "exclusive_maximum")
        for name in ("maximum", "exclusive_maximum")
        if name in constraints
    ]
    if lower_bounds and upper_bounds:
        lower_value = max(value for value, _ in lower_bounds)
        upper_value = min(value for value, _ in upper_bounds)
        lower_exclusive = any(
            value == lower_value and exclusive for value, exclusive in lower_bounds
        )
        upper_exclusive = any(
            value == upper_value and exclusive for value, exclusive in upper_bounds
        )
        if lower_value > upper_value or (
            lower_value == upper_value and (lower_exclusive or upper_exclusive)
        ):
            raise ContractValidationError("conflicting constraints: empty numeric range")

    if "pattern" in constraints:
        try:
            re.compile(constraints["pattern"])
        except re.error as error:
            raise ContractValidationError(
                f"invalid pattern constraint: {error}"
            ) from error


def _validate_parameter_value(
    name: str,
    value: object,
    definition: Mapping[str, Any],
    *,
    source: str,
) -> None:
    parameter_type = definition["type"]
    if not _is_parameter_type(value, parameter_type):
        raise ContractValidationError(
            f"invalid value for {source} parameter {name!r}: expected {parameter_type}"
        )

    constraints = definition["constraints"]
    label = f"{source} parameter {name!r}"
    if "enum" in constraints and value not in constraints["enum"]:
        raise ContractValidationError(f"{label} violates enum")
    if "minimum" in constraints and value < constraints["minimum"]:
        raise ContractValidationError(f"{label} violates minimum")
    if "maximum" in constraints and value > constraints["maximum"]:
        raise ContractValidationError(f"{label} violates maximum")
    if "exclusive_minimum" in constraints and value <= constraints["exclusive_minimum"]:
        raise ContractValidationError(f"{label} violates exclusive_minimum")
    if "exclusive_maximum" in constraints and value >= constraints["exclusive_maximum"]:
        raise ContractValidationError(f"{label} violates exclusive_maximum")
    if "min_length" in constraints and len(value) < constraints["min_length"]:
        raise ContractValidationError(f"{label} violates min_length")
    if "max_length" in constraints and len(value) > constraints["max_length"]:
        raise ContractValidationError(f"{label} violates max_length")
    if "pattern" in constraints and re.search(constraints["pattern"], value) is None:
        raise ContractValidationError(f"{label} violates pattern")
    if "min_items" in constraints and len(value) < constraints["min_items"]:
        raise ContractValidationError(f"{label} violates min_items")
    if "max_items" in constraints and len(value) > constraints["max_items"]:
        raise ContractValidationError(f"{label} violates max_items")


def _validate_parameter_definitions(parameters: Mapping[str, Any]) -> None:
    for name, definition in parameters.items():
        parameter_type = definition["type"]
        constraints = definition["constraints"]
        _validate_constraint_applicability(parameter_type, constraints)
        if "enum" in constraints:
            for enum_value in constraints["enum"]:
                if not _is_parameter_type(enum_value, parameter_type):
                    raise ContractValidationError(
                        f"enum value for parameter {name!r} does not match "
                        f"parameter type {parameter_type!r}"
                    )
        _validate_constraint_coherence(constraints)
        _validate_parameter_value(
            name, definition["default"], definition, source="default for"
        )


def _validate_seed_parameter(manifest: Mapping[str, Any]) -> None:
    determinism = manifest["determinism"]
    seed_parameter = determinism.get("seed_parameter")
    if not determinism["random_seed_required"]:
        if seed_parameter is not None:
            raise ContractValidationError(
                "seed_parameter is forbidden when random_seed_required is false"
            )
        return
    if seed_parameter is None:
        raise ContractValidationError(
            "seed_parameter is required when random_seed_required is true"
        )
    definition = manifest["parameters"].get(seed_parameter)
    if definition is None:
        raise ContractValidationError(f"unknown seed parameter: {seed_parameter!r}")
    if definition["type"] != "integer":
        raise ContractValidationError("seed parameter must have type 'integer'")


def validate_operator_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate OperatorManifest relationships that JSON Schema cannot express."""

    _validate_input_columns(manifest["input"])
    _validate_parameter_definitions(manifest["parameters"])
    _validate_seed_parameter(manifest)


_CONTRACT_SURFACE_ARTIFACTS = {
    "quant_panel_schema",
    "operator_manifest_schema",
    "operator_binding_schema",
    "reference_validator",
}


def _binding_mismatch(field: str) -> ContractValidationError:
    return ContractValidationError(f"operator binding mismatch: {field}")


def _binding_file_digest(
    path: str | os.PathLike[str],
    *,
    field: str,
) -> str:
    try:
        return sha256_file(path)
    except (OSError, ValueError) as error:
        raise _binding_mismatch(field) from error


def validate_operator_binding(
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: str | os.PathLike[str],
    source_root: str | os.PathLike[str],
    implementation_artifact_path: str | os.PathLike[str],
    contract_surface_paths: Mapping[str, str | os.PathLike[str]],
) -> None:
    """Validate one binding against exact manifest, source, and artifact bytes."""

    validate_operator_manifest(manifest)

    implementation = manifest["implementation"]
    manifest_fields = {
        "operator_id": manifest["operator_id"],
        "operator_version": manifest["operator_version"],
        "distribution": manifest["distribution"],
        "distribution_version": implementation["package_version"],
        "source_commit": implementation["source_commit"],
        "source_tree_digest": implementation["source_tree_digest"],
        "implementation_digest": implementation["implementation_digest"],
    }
    for field, expected in manifest_fields.items():
        if binding[field] != expected:
            raise _binding_mismatch(field)

    operator_manifest_pin = binding["contract_surface"][
        "operator_manifest_schema"
    ]
    if (
        binding["schema_release"] != operator_manifest_pin["release"]
        or binding["schema_digest"] != operator_manifest_pin["digest"]
    ):
        raise _binding_mismatch("legacy operator manifest schema pin")

    if binding["manifest_digest"] != _binding_file_digest(
        manifest_path,
        field="manifest_digest",
    ):
        raise _binding_mismatch("manifest_digest")

    try:
        actual_source_tree_digest = sha256_source_tree(
            source_root,
            implementation["source_files"],
        )
    except (OSError, ValueError) as error:
        raise _binding_mismatch("source_tree_digest") from error
    if binding["source_tree_digest"] != actual_source_tree_digest:
        raise _binding_mismatch("source_tree_digest")

    if binding["implementation_digest"] != _binding_file_digest(
        implementation_artifact_path,
        field="implementation_digest",
    ):
        raise _binding_mismatch("implementation_digest")

    if set(contract_surface_paths) != _CONTRACT_SURFACE_ARTIFACTS:
        raise _binding_mismatch("contract_surface artifact paths")

    surface_release = binding["surface_release"]
    for artifact in sorted(_CONTRACT_SURFACE_ARTIFACTS):
        pin = binding["contract_surface"][artifact]
        if pin["release"] != surface_release:
            raise _binding_mismatch(f"contract_surface.{artifact}.release")
        digest_field = f"contract_surface.{artifact}.digest"
        if pin["digest"] != _binding_file_digest(
            contract_surface_paths[artifact],
            field=digest_field,
        ):
            raise _binding_mismatch(digest_field)

    operator_manifest_schema_path = Path(
        contract_surface_paths["operator_manifest_schema"]
    )
    try:
        schema_id = json.loads(
            operator_manifest_schema_path.read_text(encoding="utf-8")
        )["$id"]
    except (KeyError, OSError, UnicodeError, ValueError) as error:
        raise _binding_mismatch("legacy operator manifest schema pin") from error
    if binding["schema_id"] != schema_id:
        raise _binding_mismatch("legacy operator manifest schema pin")


def validate_operator_request_parameters(
    manifest: Mapping[str, Any], parameters: Mapping[str, Any]
) -> None:
    """Validate request parameters against one already schema-valid manifest."""

    validate_operator_manifest(manifest)
    definitions = manifest["parameters"]
    unknown = set(parameters).difference(definitions)
    if unknown:
        name = sorted(unknown)[0]
        raise ContractValidationError(f"unknown request parameter: {name!r}")

    missing = {
        name
        for name, definition in definitions.items()
        if definition["required"] and name not in parameters
    }
    if missing:
        name = sorted(missing)[0]
        raise ContractValidationError(f"missing required request parameter: {name!r}")

    for name, value in parameters.items():
        _validate_parameter_value(
            name, value, definitions[name], source="request"
        )
