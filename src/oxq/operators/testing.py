"""Reusable contract checks for provider-compatible operator callables."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from itertools import combinations
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from oxq.operators._version import is_semantic_version
from oxq.operators.errors import (
    CausalityViolationError,
    ContractViolationError,
    InsufficientCrossSectionError,
    InsufficientHistoryError,
    MissingColumnError,
    OperatorError,
)
from oxq.operators.manifest import OperatorManifest
from oxq.operators.panel import QuantPanelAdapter
from oxq.operators.types import (
    OperatorAvailability,
    OperatorCausality,
    OperatorContext,
    OperatorLifecycle,
    OperatorRequest,
    OperatorResult,
    OperatorScope,
)

OperatorCallable = Callable[[OperatorRequest], OperatorResult]
_MAX_CERTIFIED_OPTIONAL_COLUMNS = 8
_TRADING_AVAILABILITY_ORDER = {
    OperatorAvailability.PRE_OPEN: 0,
    OperatorAvailability.OPEN: 1,
    OperatorAvailability.INTRADAY: 2,
    OperatorAvailability.CLOSE: 3,
    OperatorAvailability.AFTER_CLOSE: 4,
}


@dataclass(frozen=True, slots=True)
class ContractReport:
    operator_id: str
    operator_version: str
    distribution: str
    distribution_version: str
    manifest_digest: str
    implementation_digest: str
    parameters: Mapping[str, Any]
    parameters_digest: str
    input_dtypes: Mapping[str, str]
    input_dtypes_digest: str
    context: OperatorContext
    context_digest: str
    passed: bool
    checks: tuple[str, ...]


def verify_operator_contract(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    *,
    expected_distribution_version: str,
    expected_implementation_digest: str,
) -> ContractReport:
    """Run provider-neutral checks for a stateless operator entry point.

    The checker intentionally accepts a callable supplied by the caller. It
    never imports `manifest.module` or resolves `manifest.callable` itself.
    """

    if manifest.lifecycle is not OperatorLifecycle.STATELESS:
        raise ContractViolationError(
            "stateless contract checker requires lifecycle=stateless",
            operator_id=manifest.operator_id,
        )
    if request.operator_id != manifest.operator_id:
        raise ContractViolationError(
            "request operator_id does not match manifest",
            operator_id=manifest.operator_id,
        )
    if not isinstance(expected_distribution_version, str) or not is_semantic_version(expected_distribution_version):
        raise ContractViolationError(
            "expected_distribution_version must be semantic versioning",
            operator_id=manifest.operator_id,
            details={"distribution_version": expected_distribution_version},
        )
    causality_parameters = sorted(name for name, declaration in manifest.raw["parameters"].items() if declaration["affects_causality"])
    if causality_parameters:
        raise ContractViolationError(
            "contract checker cannot certify parameters that affect causality",
            operator_id=manifest.operator_id,
            details={"parameters": causality_parameters},
        )
    availability_parameters = sorted(
        name for name, declaration in manifest.raw["parameters"].items() if declaration["affects_availability"]
    )
    if availability_parameters:
        raise ContractViolationError(
            "contract checker cannot certify parameters that affect availability",
            operator_id=manifest.operator_id,
            details={"parameters": availability_parameters},
        )
    output_field_parameters = sorted(
        name for name, declaration in manifest.raw["parameters"].items() if declaration["affects_output_fields"]
    )
    if output_field_parameters:
        raise ContractViolationError(
            "contract checker cannot certify parameters that affect output fields",
            operator_id=manifest.operator_id,
            details={"parameters": output_field_parameters},
        )
    warmup_parameters = sorted(name for name, declaration in manifest.raw["parameters"].items() if declaration["affects_warmup"])
    if warmup_parameters:
        raise ContractViolationError(
            "contract checker cannot certify parameters that affect warmup",
            operator_id=manifest.operator_id,
            details={"parameters": warmup_parameters},
        )
    nan_policy = manifest.raw["outputs"]["nan_policy"]
    if nan_policy in {"propagate", "declared_missing"}:
        raise ContractViolationError(
            f"contract checker cannot certify nan_policy={nan_policy}",
            operator_id=manifest.operator_id,
            details={"nan_policy": nan_policy},
        )
    missing_value_policy = manifest.raw["inputs"]["missing_value_policy"]["kind"]
    if missing_value_policy != "require_complete":
        raise ContractViolationError(
            f"contract checker cannot certify input missing_value_policy={missing_value_policy}",
            operator_id=manifest.operator_id,
            details={"missing_value_policy": missing_value_policy},
        )
    parameters = manifest.validate_parameters(request.parameters)
    parameters_digest = _resolved_parameters_digest(parameters, manifest.operator_id)
    normalized = _copy_request(request, parameters=parameters)
    context_digest = _operator_context_digest(normalized.context, manifest.operator_id)
    determinism = manifest.raw.get("determinism", {})
    if determinism.get("bitwise", True):
        object_fields = [
            field["name_template"].format(**parameters) for field in manifest.raw["outputs"]["fields"] if field["dtype"] == "object"
        ]
        if object_fields:
            raise ContractViolationError(
                "contract checker cannot certify object output fields with bitwise determinism",
                operator_id=manifest.operator_id,
                details={"fields": object_fields},
            )
    _validate_availability(manifest, normalized)
    input_spec = manifest.raw["inputs"]
    try:
        QuantPanelAdapter.validate_panel(
            normalized.input_panel,
            normalized.context,
            require_canonical_order=input_spec["requires_sorted"],
        )
    except OperatorError as exc:
        raise _enrich_operator_error(exc, manifest.operator_id) from exc
    required_columns = set(manifest.raw["inputs"]["required_columns"])
    missing_columns = sorted(required_columns - set(normalized.input_panel.columns))
    if missing_columns:
        raise ContractViolationError(
            f"input is missing declared columns: {', '.join(missing_columns)}",
            operator_id=manifest.operator_id,
        )
    optional_columns = tuple(input_spec["optional_columns"])
    if len(optional_columns) > _MAX_CERTIFIED_OPTIONAL_COLUMNS:
        raise ContractViolationError(
            f"contract checker supports at most {_MAX_CERTIFIED_OPTIONAL_COLUMNS} optional input columns",
            operator_id=manifest.operator_id,
            details={"actual": len(optional_columns), "maximum": _MAX_CERTIFIED_OPTIONAL_COLUMNS},
        )
    missing_optional_columns = [column for column in optional_columns if column not in normalized.input_panel]
    if missing_optional_columns:
        raise ContractViolationError(
            "contract fixture must contain every declared optional input column",
            operator_id=manifest.operator_id,
            details={"columns": missing_optional_columns},
        )
    asset_count = normalized.input_panel["code"].nunique()
    if asset_count < input_spec["min_assets"]:
        raise InsufficientCrossSectionError(
            f"input has {asset_count} assets; minimum is {input_spec['min_assets']}",
            operator_id=manifest.operator_id,
        )
    if manifest.execution_scope is OperatorScope.CROSS_SECTION:
        assets_per_date = normalized.input_panel.groupby("date", sort=False, observed=True)["code"].nunique()
        if assets_per_date.lt(input_spec["min_assets"]).any():
            raise InsufficientCrossSectionError(
                f"input requires minimum {input_spec['min_assets']} at every date",
                operator_id=manifest.operator_id,
                details={
                    "min_assets": input_spec["min_assets"],
                    "assets_by_date": {str(date): int(count) for date, count in assets_per_date.items()},
                },
            )
    if input_spec["requires_complete_cross_section"]:
        assets_per_date = normalized.input_panel.groupby("date", sort=False, observed=True)["code"].nunique()
        if not assets_per_date.eq(asset_count).all():
            raise InsufficientCrossSectionError(
                "input does not contain a complete cross section at every date",
                operator_id=manifest.operator_id,
                details={"assets_by_date": {str(date): int(count) for date, count in assets_per_date.items()}},
            )
    min_history = input_spec["min_history"]
    history = normalized.input_panel.groupby("code", sort=False, observed=True).size()
    if not history.empty and int(history.min()) < min_history:
        raise InsufficientHistoryError(
            f"input history is shorter than minimum {min_history}",
            operator_id=manifest.operator_id,
            details={"rows_by_code": {str(code): int(rows) for code, rows in history.items()}},
        )
    declared_columns = required_columns | set(optional_columns)
    present_declared_columns = sorted(declared_columns & set(normalized.input_panel.columns))
    undeclared_columns = [column for column in normalized.input_panel.columns if column not in declared_columns | {"date", "code"}]
    if input_spec["missing_value_policy"]["kind"] == "require_complete":
        incomplete_columns = [column for column in present_declared_columns if normalized.input_panel[column].isna().any()]
        if incomplete_columns:
            raise ContractViolationError(
                "input violates missing_value_policy=require_complete",
                operator_id=manifest.operator_id,
                details={"columns": incomplete_columns},
            )
    for column in present_declared_columns:
        actual_dtype = str(normalized.input_panel[column].dtype)
        allowed_dtypes = input_spec["dtypes"][column]
        if actual_dtype not in allowed_dtypes:
            raise ContractViolationError(
                f"input column {column} dtype {actual_dtype} is not declared",
                operator_id=manifest.operator_id,
                details={"allowed": list(allowed_dtypes)},
            )
    _validate_object_input_values(manifest, normalized.input_panel, present_declared_columns)
    input_dtypes = MappingProxyType({column: str(normalized.input_panel[column].dtype) for column in present_declared_columns})
    input_dtypes_digest = _canonical_mapping_digest(
        input_dtypes,
        manifest.operator_id,
        label="input dtypes",
    )
    if not input_spec["requires_sorted"] and len(normalized.input_panel) < 2:
        raise ContractViolationError(
            "unordered input probe requires at least two rows",
            operator_id=manifest.operator_id,
            details={"required_rows": 2, "available_rows": len(normalized.input_panel)},
        )
    if manifest.execution_scope is OperatorScope.CROSS_SECTION:
        available_dates = int(normalized.input_panel["date"].nunique())
        if available_dates < 2:
            raise ContractViolationError(
                "cross_section scope probe requires at least two unique dates",
                operator_id=manifest.operator_id,
                details={"required_dates": 2, "available_dates": available_dates},
            )
    if manifest.execution_scope is OperatorScope.TIME_SERIES:
        available_assets = int(normalized.input_panel["code"].nunique())
        required_assets = input_spec["min_assets"] + 1
        if available_assets < required_assets:
            raise ContractViolationError(
                "time_series scope probe requires enough assets to exclude one symbol",
                operator_id=manifest.operator_id,
                details={"required_assets": required_assets, "available_assets": available_assets},
            )
    checks = ["request"]

    result = _invoke_operator(manifest, operator, normalized)
    checks.append("input_immutability")

    _validate_result(manifest, normalized, result, expected_implementation_digest)
    if manifest.raw["outputs"]["alignment"] == "explicit_keyed_output" and result.data.empty:
        raise ContractViolationError(
            "baseline explicit_keyed_output must contain at least one output row",
            operator_id=manifest.operator_id,
            details={"alignment": "explicit_keyed_output"},
        )
    checks.extend(("output_contract", "provenance"))

    _verify_required_column_failures(manifest, operator, normalized, required_columns)
    checks.append("required_inputs")

    declared_panel = normalized.input_panel.drop(columns=undeclared_columns)
    for present_count in range(len(optional_columns)):
        for present_subset in combinations(optional_columns, present_count):
            omitted_columns = tuple(column for column in optional_columns if column not in present_subset)
            panel_without_optional = declared_panel.drop(columns=list(omitted_columns))
            optional_request = _copy_request(normalized, panel=panel_without_optional)
            failure_message, failure_details = _optional_probe_failure(omitted_columns)
            optional_result = _invoke_operator(
                manifest,
                operator,
                optional_request,
                failure_message=failure_message,
                failure_details=failure_details,
            )
            _validate_result(manifest, optional_request, optional_result, expected_implementation_digest)
            optional_data = _deepcopy_frame(optional_result.data)
            optional_metadata = _snapshot_metadata(optional_result.metadata)
            optional_repeated = _invoke_operator(
                manifest,
                operator,
                _copy_request(optional_request),
                failure_message=failure_message,
                failure_details=failure_details,
            )
            _validate_result(manifest, optional_request, optional_repeated, expected_implementation_digest)
            omission_label = _optional_omission_label(omitted_columns)
            _require_deterministic_data(
                optional_data,
                optional_repeated.data,
                manifest,
                f"operator without optional input {omission_label} must be deterministic",
            )
            if optional_result.diagnostics != optional_repeated.diagnostics:
                raise ContractViolationError(
                    f"operator diagnostics without optional input {omission_label} must be deterministic",
                    operator_id=manifest.operator_id,
                )
            if optional_result.provenance != optional_repeated.provenance:
                raise ContractViolationError(
                    f"operator provenance without optional input {omission_label} must be deterministic",
                    operator_id=manifest.operator_id,
                )
            if not _metadata_equal(optional_metadata, optional_repeated.metadata, manifest):
                raise ContractViolationError(
                    f"operator metadata without optional input {omission_label} must be deterministic",
                    operator_id=manifest.operator_id,
                )
            _verify_behavioral_probes(
                manifest,
                operator,
                optional_request,
                optional_data,
                expected_implementation_digest,
            )
    if optional_columns:
        checks.append("optional_inputs")

    first_data = _deepcopy_frame(result.data)
    if undeclared_columns:
        declared_request = _copy_request(normalized, panel=declared_panel)
        declared_result = _invoke_operator(
            manifest,
            operator,
            declared_request,
            failure_message="operator failed without undeclared input columns",
            failure_details={"columns": undeclared_columns},
        )
        _validate_result(manifest, declared_request, declared_result, expected_implementation_digest)
        _require_exact_data(
            _canonical(first_data),
            _canonical(declared_result.data),
            manifest,
            "operator output must not depend on undeclared input columns",
        )
    first_metadata = _snapshot_metadata(result.metadata)
    repeated = _invoke_operator(manifest, operator, _copy_request(normalized))
    _validate_result(manifest, normalized, repeated, expected_implementation_digest)
    _require_deterministic_data(first_data, repeated.data, manifest, "operator output must be deterministic")
    if result.diagnostics != repeated.diagnostics:
        raise ContractViolationError(
            "operator diagnostics must be deterministic",
            operator_id=manifest.operator_id,
        )
    if result.provenance != repeated.provenance:
        raise ContractViolationError(
            "operator provenance must be deterministic",
            operator_id=manifest.operator_id,
        )
    if not _metadata_equal(first_metadata, repeated.metadata, manifest):
        raise ContractViolationError(
            "operator metadata must be deterministic",
            operator_id=manifest.operator_id,
        )
    checks.append("determinism")

    _verify_behavioral_probes(manifest, operator, normalized, first_data, expected_implementation_digest)
    checks.extend(_behavioral_check_names(manifest))

    return ContractReport(
        operator_id=manifest.operator_id,
        operator_version=manifest.operator_version,
        distribution=manifest.distribution,
        distribution_version=expected_distribution_version,
        manifest_digest=manifest.digest,
        implementation_digest=expected_implementation_digest,
        parameters=normalized.parameters,
        parameters_digest=parameters_digest,
        input_dtypes=input_dtypes,
        input_dtypes_digest=input_dtypes_digest,
        context=normalized.context,
        context_digest=context_digest,
        passed=True,
        checks=tuple(checks),
    )


def _validate_result(
    manifest: OperatorManifest,
    request: OperatorRequest,
    result: OperatorResult,
    expected_implementation_digest: str,
) -> None:
    if not isinstance(result, OperatorResult):
        raise ContractViolationError("operator must return OperatorResult", operator_id=manifest.operator_id)
    if result.provenance.operator_id != manifest.operator_id:
        raise ContractViolationError("result provenance operator_id mismatch", operator_id=manifest.operator_id)
    if result.provenance.operator_version != manifest.operator_version:
        raise ContractViolationError("result provenance operator_version mismatch", operator_id=manifest.operator_id)
    if result.provenance.implementation_digest != expected_implementation_digest:
        raise ContractViolationError(
            "result provenance implementation_digest mismatch",
            operator_id=manifest.operator_id,
            details={
                "expected": expected_implementation_digest,
                "actual": result.provenance.implementation_digest,
            },
        )
    if result.diagnostics.input_rows != len(request.input_panel):
        raise ContractViolationError("diagnostics.input_rows mismatch", operator_id=manifest.operator_id)
    if result.diagnostics.output_rows != len(result.data):
        raise ContractViolationError("diagnostics.output_rows mismatch", operator_id=manifest.operator_id)
    if isinstance(result.data, pd.DataFrame):
        invalid_columns = [
            {"position": position, "type": type(column).__name__}
            for position, column in enumerate(result.data.columns)
            if not isinstance(column, str)
        ]
        if invalid_columns:
            raise ContractViolationError(
                "operator output column labels must be strings",
                operator_id=manifest.operator_id,
                details={"columns": invalid_columns},
            )
    outputs = manifest.raw["outputs"]
    alignment = outputs["alignment"]
    try:
        QuantPanelAdapter.validate_output(request.input_panel, result.data, request.context, alignment=alignment)
    except OperatorError as exc:
        raise _enrich_operator_error(exc, manifest.operator_id) from exc
    try:
        resolved_fields = [field["name_template"].format(**request.parameters) for field in outputs["fields"]]
    except (AttributeError, KeyError, IndexError, ValueError, TypeError) as exc:
        raise ContractViolationError(
            "operator output field template could not be resolved",
            operator_id=manifest.operator_id,
        ) from exc
    empty_fields = [name for name in resolved_fields if not name]
    if empty_fields:
        raise ContractViolationError(
            "resolved operator output field names must be non-empty",
            operator_id=manifest.operator_id,
            details={"fields": empty_fields},
        )
    if len(set(resolved_fields)) != len(resolved_fields):
        raise ContractViolationError(
            "operator manifest contains duplicate resolved output fields",
            operator_id=manifest.operator_id,
            details={"fields": resolved_fields},
        )
    reserved_fields = sorted(set(resolved_fields) & {"date", "code"})
    if reserved_fields:
        raise ContractViolationError(
            "operator output fields contain a reserved QuantPanel key",
            operator_id=manifest.operator_id,
            details={"fields": reserved_fields},
        )
    expected_fields = set(resolved_fields)
    actual_fields = set(result.data.columns) - {"date", "code"}
    if actual_fields != expected_fields:
        raise ContractViolationError(
            "operator output must contain exactly the declared fields",
            operator_id=manifest.operator_id,
            details={"expected": sorted(expected_fields), "actual": sorted(actual_fields)},
        )
    declarations = dict(zip(resolved_fields, outputs["fields"], strict=True))
    for name, declaration in declarations.items():
        actual_dtype = str(result.data[name].dtype)
        if actual_dtype != declaration["dtype"]:
            raise ContractViolationError(
                f"output column {name} dtype {actual_dtype} does not match declared dtype {declaration['dtype']}",
                operator_id=manifest.operator_id,
            )
        non_null = result.data[name].dropna()
        if pd.api.types.is_numeric_dtype(result.data[name].dtype) and not np.isfinite(non_null.to_numpy()).all():
            raise ContractViolationError(
                f"output column {name} must contain only finite numeric values",
                operator_id=manifest.operator_id,
            )
        try:
            below_minimum = "minimum" in declaration and non_null.lt(declaration["minimum"]).any()
            above_maximum = "maximum" in declaration and non_null.gt(declaration["maximum"]).any()
        except TypeError as exc:
            raise ContractViolationError(
                f"output column {name} cannot be compared with its declared bounds",
                operator_id=manifest.operator_id,
            ) from exc
        if below_minimum:
            raise ContractViolationError(
                f"output column {name} contains values below declared minimum {declaration['minimum']}",
                operator_id=manifest.operator_id,
            )
        if above_maximum:
            raise ContractViolationError(
                f"output column {name} contains values above declared maximum {declaration['maximum']}",
                operator_id=manifest.operator_id,
            )
    warmup_rows = _resolve_warmup_rows(manifest, request)
    warmup_mask = _output_warmup_mask(request.input_panel, result.data, warmup_rows)
    expected_warmup_rows = _expected_warmup_rows(request.input_panel, warmup_rows)
    if result.diagnostics.warmup_rows != expected_warmup_rows:
        raise ContractViolationError(
            "diagnostics.warmup_rows mismatch",
            operator_id=manifest.operator_id,
            details={"expected": expected_warmup_rows, "actual": result.diagnostics.warmup_rows},
        )
    expected_dropped_rows = len(request.input_panel) - len(result.data)
    if result.diagnostics.dropped_rows != expected_dropped_rows:
        raise ContractViolationError(
            "diagnostics.dropped_rows mismatch",
            operator_id=manifest.operator_id,
            details={"expected": expected_dropped_rows, "actual": result.diagnostics.dropped_rows},
        )
    _validate_nan_policy(manifest, result, tuple(declarations), warmup_mask)


def _validate_availability(manifest: OperatorManifest, request: OperatorRequest) -> None:
    available = manifest.availability
    evaluation = OperatorAvailability(request.context.evaluation_time)
    if available is OperatorAvailability.PUBLICATION_TIME or evaluation is OperatorAvailability.PUBLICATION_TIME:
        valid = available is evaluation
    else:
        valid = _TRADING_AVAILABILITY_ORDER[evaluation] >= _TRADING_AVAILABILITY_ORDER[available]
    if not valid:
        raise CausalityViolationError(
            f"operator output is not available at evaluation_time={evaluation.value}",
            operator_id=manifest.operator_id,
            details={"availability": available.value, "evaluation_time": evaluation.value},
        )


def _enrich_operator_error(error: OperatorError, operator_id: str) -> OperatorError:
    return type(error)(
        str(error),
        operator_id=operator_id,
        details=error.details,
        retryable=error.retryable,
    )


def _validate_nan_policy(
    manifest: OperatorManifest,
    result: OperatorResult,
    output_fields: tuple[str, ...],
    warmup_mask: pd.Series,
) -> None:
    outputs = manifest.raw["outputs"]
    nan_policy = outputs["nan_policy"]
    missing = result.data[list(output_fields)].isna()
    if nan_policy == "none" and missing.any().any():
        raise ContractViolationError(
            "operator output violates nan_policy=none",
            operator_id=manifest.operator_id,
        )
    if nan_policy != "warmup_only" or not missing.any().any():
        return
    if missing.reset_index(drop=True).loc[~warmup_mask].any().any():
        raise ContractViolationError(
            "operator output contains NaN values outside declared warmup",
            operator_id=manifest.operator_id,
        )


def _resolve_warmup_rows(manifest: OperatorManifest, request: OperatorRequest) -> int:
    warmup = manifest.raw["outputs"]["warmup"]
    if warmup["kind"] == "fixed":
        value = warmup["rows"]
    else:
        value = request.parameters[warmup["parameter"]] + warmup.get("offset", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractViolationError(
            "declared warmup must resolve to a non-negative integer",
            operator_id=manifest.operator_id,
        )
    return value


def _output_warmup_mask(
    input_panel: pd.DataFrame,
    output_panel: pd.DataFrame,
    warmup_rows: int,
) -> pd.Series:
    ordered = input_panel[["date", "code"]].sort_values(["code", "date"], kind="stable").copy()
    positions = ordered.groupby("code", sort=False, observed=True).cumcount()
    positions.index = pd.MultiIndex.from_frame(ordered[["date", "code"]])
    output_keys = pd.MultiIndex.from_frame(output_panel[["date", "code"]])
    output_positions = positions.reindex(output_keys)
    return pd.Series(output_positions.to_numpy() < warmup_rows, dtype="bool")


def _expected_warmup_rows(input_panel: pd.DataFrame, warmup_rows: int) -> int:
    history = input_panel.groupby("code", sort=False, observed=True).size()
    return int(history.clip(upper=warmup_rows).sum())


def _resolved_parameters_digest(parameters: Mapping[str, Any], operator_id: str) -> str:
    return _canonical_mapping_digest(parameters, operator_id, label="resolved parameters")


def _canonical_mapping_digest(
    value: Mapping[str, Any],
    operator_id: str,
    *,
    label: str,
) -> str:
    try:
        canonical = json.dumps(
            _materialize_canonical_json(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractViolationError(
            f"{label} must be canonically serializable",
            operator_id=operator_id,
        ) from exc
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _materialize_canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _materialize_canonical_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_materialize_canonical_json(item) for item in value]
    return value


def _operator_context_digest(context: OperatorContext, operator_id: str) -> str:
    payload = {field.name: getattr(context, field.name) for field in fields(context)}
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractViolationError(
            "operator context must be canonically serializable",
            operator_id=operator_id,
        ) from exc
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _verify_required_column_failures(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    required_columns: set[str],
) -> None:
    for column in sorted(required_columns):
        probe_request = _copy_request(request, panel=request.input_panel.drop(columns=[column]))
        try:
            operator(probe_request)
        except MissingColumnError as exc:
            if exc.details.get("column") == column:
                continue
            raise ContractViolationError(
                f"missing required input column {column} must identify that column in MissingColumnError details",
                operator_id=manifest.operator_id,
                details={"column": column},
            ) from exc
        except Exception as exc:
            raise ContractViolationError(
                f"missing required input column {column} must raise MissingColumnError",
                operator_id=manifest.operator_id,
                details={"column": column},
            ) from exc
        raise ContractViolationError(
            f"missing required input column {column} must raise MissingColumnError",
            operator_id=manifest.operator_id,
            details={"column": column},
        )


def _optional_probe_failure(omitted_columns: tuple[str, ...]) -> tuple[str, Mapping[str, Any]]:
    if len(omitted_columns) == 1:
        column = omitted_columns[0]
        return f"operator failed without optional input column {column}", {"column": column}
    columns = list(omitted_columns)
    return f"operator failed without optional input columns {', '.join(columns)}", {"columns": columns}


def _optional_omission_label(omitted_columns: tuple[str, ...]) -> str:
    if len(omitted_columns) == 1:
        return f"column {omitted_columns[0]}"
    return f"columns {', '.join(omitted_columns)}"


def _invoke_operator(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    *,
    failure_message: str = "operator invocation failed",
    failure_details: Mapping[str, Any] | None = None,
) -> OperatorResult:
    snapshot = _deepcopy_frame(request.input_panel)
    try:
        result = operator(request)
    except OperatorError:
        raise
    except Exception as exc:
        raise ContractViolationError(
            failure_message,
            operator_id=manifest.operator_id,
            details=failure_details,
        ) from exc
    try:
        _assert_frame_bitwise_equal(request.input_panel, snapshot)
    except AssertionError as exc:
        raise ContractViolationError(
            "operator mutated input_panel",
            operator_id=manifest.operator_id,
        ) from exc
    return result


def _metadata_equal(left: Any, right: Any, manifest: OperatorManifest) -> bool:
    determinism = manifest.raw.get("determinism", {})
    bitwise = determinism.get("bitwise", True)
    absolute_tolerance = determinism.get("absolute_tolerance", 0.0)
    relative_tolerance = determinism.get("relative_tolerance", 0.0)
    return _structured_value_equal(
        left,
        right,
        bitwise=bitwise,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def _structured_value_equal(
    left: Any,
    right: Any,
    *,
    bitwise: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(
            _structured_value_equal(
                left[key],
                right[key],
                bitwise=bitwise,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _structured_value_equal(
                a,
                b,
                bitwise=bitwise,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if bitwise:
            try:
                _assert_array_representation_equal(left, right)
            except AssertionError:
                return False
            return True
        try:
            np.testing.assert_allclose(
                left,
                right,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                equal_nan=True,
            )
        except (AssertionError, TypeError):
            return _structured_value_equal(
                left.tolist(),
                right.tolist(),
                bitwise=False,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        return True
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            if bitwise:
                _assert_frame_bitwise_equal(left, right)
            else:
                pd.testing.assert_frame_equal(
                    left,
                    right,
                    check_exact=False,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
        except AssertionError:
            return False
        return True
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        try:
            if bitwise:
                _assert_series_bitwise_equal(left, right)
            else:
                pd.testing.assert_series_equal(
                    left,
                    right,
                    check_exact=False,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
        except AssertionError:
            return False
        return True
    if _both_scalar_missing(left, right):
        if bitwise and _is_representation_scalar(left) and _is_representation_scalar(right):
            return _scalar_representation(left) == _scalar_representation(right)
        return True
    if bitwise and _is_representation_scalar(left) and _is_representation_scalar(right):
        return _scalar_representation(left) == _scalar_representation(right)
    if not bitwise and _is_numeric_scalar(left) and _is_numeric_scalar(right):
        try:
            return bool(
                np.isclose(
                    left,
                    right,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                    equal_nan=True,
                )
            )
        except TypeError:
            return False
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, bool):
        return equal
    if hasattr(equal, "all"):
        try:
            return bool(equal.all())
        except (TypeError, ValueError):
            return False
    return False


def _snapshot_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _snapshot_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_metadata(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_metadata(item) for item in value]
    return copy.deepcopy(value)


def _both_scalar_missing(left: Any, right: Any) -> bool:
    if not pd.api.types.is_scalar(left) or not pd.api.types.is_scalar(right):
        return False
    try:
        return bool(pd.isna(left)) and bool(pd.isna(right))
    except (TypeError, ValueError):
        return False


def _deepcopy_frame(panel: pd.DataFrame) -> pd.DataFrame:
    snapshot = panel.copy(deep=True)
    for column in panel.select_dtypes(include="object").columns:
        snapshot[column] = panel[column].map(copy.deepcopy)
    return snapshot


def _validate_object_input_values(
    manifest: OperatorManifest,
    panel: pd.DataFrame,
    declared_columns: list[str],
) -> None:
    unsupported_columns = sorted(
        column
        for column in declared_columns
        if pd.api.types.is_object_dtype(panel[column].dtype) and any(not _is_certifiable_object_value(value) for value in panel[column])
    )
    if unsupported_columns:
        raise ContractViolationError(
            "contract checker cannot certify object input values representation-wise",
            operator_id=manifest.operator_id,
            details={"columns": unsupported_columns},
        )


def _is_certifiable_object_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_is_certifiable_object_value(key) and _is_certifiable_object_value(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_certifiable_object_value(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return all(_is_certifiable_object_value(item) for item in value.flat)
        return True
    return value is None or value is pd.NA or isinstance(value, (bool, int, float, str, bytes, np.generic))


def _behavioral_check_names(manifest: OperatorManifest) -> tuple[str, ...]:
    checks: list[str] = []
    if manifest.causality is OperatorCausality.PAST_ONLY and manifest.execution_scope in {
        OperatorScope.TIME_SERIES,
        OperatorScope.PANEL,
        OperatorScope.RESEARCH_ONLY,
    }:
        checks.append("causality")
    if not manifest.raw["inputs"]["requires_sorted"]:
        checks.append("unordered_input")
    if manifest.execution_scope is OperatorScope.TIME_SERIES:
        checks.append("batch_consistency")
    if manifest.execution_scope is OperatorScope.CROSS_SECTION:
        checks.append("scope_consistency")
    return tuple(checks)


def _verify_behavioral_probes(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    baseline: pd.DataFrame,
    expected_implementation_digest: str,
) -> None:
    if manifest.causality is OperatorCausality.PAST_ONLY and manifest.execution_scope in {
        OperatorScope.TIME_SERIES,
        OperatorScope.PANEL,
        OperatorScope.RESEARCH_ONLY,
    }:
        _verify_past_only_causality(
            manifest,
            operator,
            request,
            baseline,
            expected_implementation_digest,
        )

    if not manifest.raw["inputs"]["requires_sorted"]:
        permuted_panels = [request.input_panel.iloc[::-1].reset_index(drop=True)]
        if len(request.input_panel) >= 3:
            permuted_panels.append(
                pd.concat(
                    (request.input_panel.iloc[1:], request.input_panel.iloc[:1]),
                    ignore_index=True,
                )
            )
        for permuted_panel in permuted_panels:
            permuted_request = _copy_request(request, panel=permuted_panel)
            permuted_result = _invoke_operator(manifest, operator, permuted_request)
            _validate_result(manifest, permuted_request, permuted_result, expected_implementation_digest)
            _require_deterministic_data(
                _canonical(baseline),
                _canonical(permuted_result.data),
                manifest,
                "operator output must not depend on incidental input order",
            )

    if manifest.execution_scope is OperatorScope.TIME_SERIES:
        for excluded_code in request.input_panel["code"].drop_duplicates():
            isolated_panel = request.input_panel.loc[request.input_panel["code"] != excluded_code].reset_index(drop=True)
            isolated_request = _copy_request(request, panel=isolated_panel)
            isolated_result = _invoke_operator(manifest, operator, isolated_request)
            _validate_result(manifest, isolated_request, isolated_result, expected_implementation_digest)
            expected = baseline.loc[baseline["code"] != excluded_code]
            _require_deterministic_data(
                _canonical(expected),
                _canonical(isolated_result.data),
                manifest,
                "time_series output must not depend on other symbols",
            )

    if manifest.execution_scope is OperatorScope.CROSS_SECTION:
        _verify_cross_section_scope(
            manifest,
            operator,
            request,
            baseline,
            expected_implementation_digest,
        )


def _verify_past_only_causality(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    baseline: pd.DataFrame,
    expected_implementation_digest: str,
) -> None:
    dates = request.input_panel["date"].drop_duplicates().sort_values()
    input_spec = manifest.raw["inputs"]
    probe_executed = False
    for cutoff in dates.iloc[:-1]:
        prefix = request.input_panel.loc[request.input_panel["date"] <= cutoff].reset_index(drop=True)
        if not _is_valid_past_only_prefix(manifest, prefix):
            continue
        probe_executed = True
        prefix_request = _copy_request(request, panel=prefix)
        prefix_result = _invoke_operator(manifest, operator, prefix_request)
        _validate_result(manifest, prefix_request, prefix_result, expected_implementation_digest)
        expected = baseline.loc[baseline["date"] <= cutoff]
        _require_deterministic_data(
            _canonical(expected),
            _canonical(prefix_result.data),
            manifest,
            "past_only output changed when future history was truncated",
        )
    if not probe_executed:
        raise ContractViolationError(
            "past_only causality probe requires at least one valid proper prefix",
            operator_id=manifest.operator_id,
            details={"min_history": input_spec["min_history"], "available_dates": len(dates)},
        )


def _is_valid_past_only_prefix(manifest: OperatorManifest, prefix: pd.DataFrame) -> bool:
    input_spec = manifest.raw["inputs"]
    history = prefix.groupby("code", sort=False, observed=True).size()
    if history.empty or int(history.min()) < input_spec["min_history"]:
        return False
    asset_count = int(prefix["code"].nunique())
    if asset_count < input_spec["min_assets"]:
        return False
    if manifest.execution_scope is OperatorScope.CROSS_SECTION:
        assets_per_date = prefix.groupby("date", sort=False, observed=True)["code"].nunique()
        if assets_per_date.lt(input_spec["min_assets"]).any():
            return False
    return True


def _verify_cross_section_scope(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    baseline: pd.DataFrame,
    expected_implementation_digest: str,
) -> None:
    pieces: list[pd.DataFrame] = []
    for _, date_panel in request.input_panel.groupby("date", sort=True, observed=True):
        date_request = _copy_request(request, panel=date_panel.reset_index(drop=True))
        date_result = _invoke_operator(manifest, operator, date_request)
        _validate_result(manifest, date_request, date_result, expected_implementation_digest)
        pieces.append(date_result.data)
    per_date = pd.concat(pieces, ignore_index=True) if pieces else baseline.iloc[0:0].copy()
    _require_deterministic_data(
        _canonical(baseline),
        _canonical(per_date),
        manifest,
        "cross_section output must match combined per-date output",
    )


def _copy_request(
    request: OperatorRequest,
    *,
    parameters: dict[str, object] | None = None,
    panel: pd.DataFrame | None = None,
) -> OperatorRequest:
    return OperatorRequest(
        operator_id=request.operator_id,
        parameters=request.parameters if parameters is None else parameters,
        input_panel=_deepcopy_frame(request.input_panel if panel is None else panel),
        context=request.context,
    )


def _canonical(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["date", "code"], kind="stable", ignore_index=True)


def _require_deterministic_data(
    left: pd.DataFrame,
    right: pd.DataFrame,
    manifest: OperatorManifest,
    message: str,
) -> None:
    determinism = manifest.raw.get("determinism", {})
    bitwise = determinism.get("bitwise", True)
    absolute_tolerance = determinism.get("absolute_tolerance", 0.0)
    relative_tolerance = determinism.get("relative_tolerance", 0.0)
    try:
        if bitwise:
            _assert_frame_bitwise_equal(left, right)
        else:
            pd.testing.assert_frame_equal(
                left,
                right,
                check_exact=False,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
    except AssertionError as exc:
        raise ContractViolationError(message, operator_id=manifest.operator_id) from exc


def _assert_frame_bitwise_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    _assert_array_representation_equal(left.index.to_numpy(), right.index.to_numpy())
    _assert_array_representation_equal(left.columns.to_numpy(), right.columns.to_numpy())
    for position in range(len(left.columns)):
        left_values = left.iloc[:, position].to_numpy(copy=False)
        right_values = right.iloc[:, position].to_numpy(copy=False)
        _assert_array_representation_equal(left_values, right_values)


def _assert_series_bitwise_equal(left: pd.Series, right: pd.Series) -> None:
    pd.testing.assert_series_equal(left, right, check_exact=True)
    _assert_array_representation_equal(left.index.to_numpy(), right.index.to_numpy())
    _assert_array_representation_equal(left.to_numpy(copy=False), right.to_numpy(copy=False))


def _assert_array_representation_equal(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError("array representation differs")
    if left.dtype.hasobject:
        for left_item, right_item in zip(left.flat, right.flat, strict=True):
            _assert_object_value_representation_equal(left_item, right_item)
        return
    left_bytes = np.ascontiguousarray(left).view(np.uint8)
    right_bytes = np.ascontiguousarray(right).view(np.uint8)
    np.testing.assert_array_equal(left_bytes, right_bytes)


def _assert_object_value_representation_equal(left: Any, right: Any) -> None:
    if left is pd.NA or right is pd.NA:
        if left is not right:
            raise AssertionError("missing value representation differs")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_items = list(left.items())
        right_items = list(right.items())
        if len(left_items) != len(right_items):
            raise AssertionError("mapping representation differs")
        for (left_key, left_value), (right_key, right_value) in zip(left_items, right_items, strict=True):
            _assert_object_value_representation_equal(left_key, right_key)
            _assert_object_value_representation_equal(left_value, right_value)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("sequence representation differs")
        for left_item, right_item in zip(left, right, strict=True):
            _assert_object_value_representation_equal(left_item, right_item)
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        _assert_array_representation_equal(left, right)
        return
    if not _is_certifiable_object_value(left) or not _is_certifiable_object_value(right):
        raise AssertionError("object value cannot be compared representation-wise")
    if _is_representation_scalar(left) or _is_representation_scalar(right):
        if not (_is_representation_scalar(left) and _is_representation_scalar(right)):
            raise AssertionError("scalar representation differs")
        if _scalar_representation(left) != _scalar_representation(right):
            raise AssertionError("scalar representation differs")
        return
    if type(left) is not type(right) or left != right:
        raise AssertionError("object value representation differs")


def _is_representation_scalar(value: Any) -> bool:
    return isinstance(value, (float, np.generic)) and not isinstance(value, (bool, np.bool_))


def _scalar_representation(value: Any) -> tuple[str, bytes]:
    if isinstance(value, np.generic):
        array = np.asarray(value)
        return array.dtype.str, array.tobytes()
    return "python-float64", struct.pack(">d", value)


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))


def _require_exact_data(
    left: pd.DataFrame,
    right: pd.DataFrame,
    manifest: OperatorManifest,
    message: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError as exc:
        raise ContractViolationError(message, operator_id=manifest.operator_id) from exc
