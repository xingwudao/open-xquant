"""Reusable contract checks for provider-compatible operator callables."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from oxq.operators.errors import (
    CausalityViolationError,
    ContractViolationError,
    InsufficientCrossSectionError,
    InsufficientHistoryError,
)
from oxq.operators.manifest import OperatorManifest
from oxq.operators.panel import QuantPanelAdapter
from oxq.operators.types import (
    OperatorAvailability,
    OperatorCausality,
    OperatorLifecycle,
    OperatorRequest,
    OperatorResult,
    OperatorScope,
)

OperatorCallable = Callable[[OperatorRequest], OperatorResult]
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
    passed: bool
    checks: tuple[str, ...]


def verify_operator_contract(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
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
    parameters = manifest.validate_parameters(request.parameters)
    normalized = _copy_request(request, parameters=parameters)
    _validate_availability(manifest, normalized)
    input_spec = manifest.raw["inputs"]
    QuantPanelAdapter.validate_panel(
        normalized.input_panel,
        normalized.context,
        require_canonical_order=input_spec["requires_sorted"],
    )
    required_columns = set(manifest.raw["inputs"]["required_columns"])
    missing_columns = sorted(required_columns - set(normalized.input_panel.columns))
    if missing_columns:
        raise ContractViolationError(
            f"input is missing declared columns: {', '.join(missing_columns)}",
            operator_id=manifest.operator_id,
        )
    asset_count = normalized.input_panel["code"].nunique()
    if asset_count < input_spec["min_assets"]:
        raise InsufficientCrossSectionError(
            f"input has {asset_count} assets; minimum is {input_spec['min_assets']}",
            operator_id=manifest.operator_id,
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
    declared_columns = required_columns | set(input_spec["optional_columns"])
    present_declared_columns = sorted(declared_columns & set(normalized.input_panel.columns))
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
    checks = ["request"]

    result = _invoke_operator(manifest, operator, normalized)
    checks.append("input_immutability")

    _validate_result(manifest, normalized, result)
    checks.extend(("output_contract", "provenance"))

    first_data = _deepcopy_frame(result.data)
    first_metadata = copy.deepcopy(dict(result.metadata))
    repeated = _invoke_operator(manifest, operator, _copy_request(normalized))
    _validate_result(manifest, normalized, repeated)
    _require_equal_data(first_data, repeated.data, manifest, "operator output must be deterministic")
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
    if not _metadata_equal(first_metadata, repeated.metadata):
        raise ContractViolationError(
            "operator metadata must be deterministic",
            operator_id=manifest.operator_id,
        )
    checks.append("determinism")

    if manifest.causality is OperatorCausality.PAST_ONLY and manifest.execution_scope is OperatorScope.TIME_SERIES:
        _verify_past_only_causality(manifest, operator, normalized, first_data)
        checks.append("causality")

    if not input_spec["requires_sorted"]:
        shuffled_panel = normalized.input_panel.sample(frac=1, random_state=731).reset_index(drop=True)
        shuffled_request = _copy_request(normalized, panel=shuffled_panel)
        shuffled_result = _invoke_operator(manifest, operator, shuffled_request)
        _validate_result(manifest, shuffled_request, shuffled_result)
        _require_equal_data(
            _canonical(first_data),
            _canonical(shuffled_result.data),
            manifest,
            "operator output must not depend on incidental input order",
        )
        checks.append("unordered_input")

    if manifest.execution_scope is OperatorScope.TIME_SERIES:
        pieces: list[pd.DataFrame] = []
        for code in normalized.input_panel["code"].drop_duplicates():
            symbol_panel = normalized.input_panel.loc[normalized.input_panel["code"] == code].reset_index(drop=True)
            symbol_request = _copy_request(normalized, panel=symbol_panel)
            symbol_result = _invoke_operator(manifest, operator, symbol_request)
            _validate_result(manifest, symbol_request, symbol_result)
            pieces.append(symbol_result.data)
        per_symbol = pd.concat(pieces, ignore_index=True) if pieces else result.data.iloc[0:0].copy()
        _require_equal_data(
            _canonical(first_data),
            _canonical(per_symbol),
            manifest,
            "batch output must match per-symbol output",
        )
        checks.append("batch_consistency")

    if manifest.execution_scope is OperatorScope.CROSS_SECTION:
        _verify_cross_section_scope(manifest, operator, normalized, first_data)
        checks.append("scope_consistency")

    return ContractReport(operator_id=manifest.operator_id, passed=True, checks=tuple(checks))


def _validate_result(
    manifest: OperatorManifest,
    request: OperatorRequest,
    result: OperatorResult,
) -> None:
    if not isinstance(result, OperatorResult):
        raise ContractViolationError("operator must return OperatorResult", operator_id=manifest.operator_id)
    if result.provenance.operator_id != manifest.operator_id:
        raise ContractViolationError("result provenance operator_id mismatch", operator_id=manifest.operator_id)
    if result.provenance.operator_version != manifest.operator_version:
        raise ContractViolationError("result provenance operator_version mismatch", operator_id=manifest.operator_id)
    if result.diagnostics.input_rows != len(request.input_panel):
        raise ContractViolationError("diagnostics.input_rows mismatch", operator_id=manifest.operator_id)
    if result.diagnostics.output_rows != len(result.data):
        raise ContractViolationError("diagnostics.output_rows mismatch", operator_id=manifest.operator_id)
    outputs = manifest.raw["outputs"]
    alignment = outputs["alignment"]
    QuantPanelAdapter.validate_output(request.input_panel, result.data, request.context, alignment=alignment)
    try:
        resolved_fields = [
            field["name_template"].format(**request.parameters)
            for field in outputs["fields"]
        ]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ContractViolationError(
            "operator output field template could not be resolved",
            operator_id=manifest.operator_id,
        ) from exc
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


def _invoke_operator(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
) -> OperatorResult:
    snapshot = _deepcopy_frame(request.input_panel)
    result = operator(request)
    try:
        pd.testing.assert_frame_equal(request.input_panel, snapshot)
    except AssertionError as exc:
        raise ContractViolationError(
            "operator mutated input_panel",
            operator_id=manifest.operator_id,
        ) from exc
    return result


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_metadata_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_metadata_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(left, right)
        except AssertionError:
            return False
        return True
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        try:
            pd.testing.assert_series_equal(left, right)
        except AssertionError:
            return False
        return True
    if _both_scalar_missing(left, right):
        return True
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


def _verify_past_only_causality(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    baseline: pd.DataFrame,
) -> None:
    dates = request.input_panel["date"].drop_duplicates().sort_values()
    input_spec = manifest.raw["inputs"]
    for cutoff in dates.iloc[:-1]:
        prefix = request.input_panel.loc[request.input_panel["date"] <= cutoff].reset_index(drop=True)
        history = prefix.groupby("code", sort=False, observed=True).size()
        if history.empty or int(history.min()) < input_spec["min_history"]:
            continue
        prefix_request = _copy_request(request, panel=prefix)
        prefix_result = _invoke_operator(manifest, operator, prefix_request)
        _validate_result(manifest, prefix_request, prefix_result)
        expected = baseline.loc[baseline["date"] <= cutoff]
        _require_equal_data(
            _canonical(expected),
            _canonical(prefix_result.data),
            manifest,
            "past_only output changed when future history was truncated",
        )


def _verify_cross_section_scope(
    manifest: OperatorManifest,
    operator: OperatorCallable,
    request: OperatorRequest,
    baseline: pd.DataFrame,
) -> None:
    pieces: list[pd.DataFrame] = []
    for _, date_panel in request.input_panel.groupby("date", sort=True, observed=True):
        date_request = _copy_request(request, panel=date_panel.reset_index(drop=True))
        date_result = _invoke_operator(manifest, operator, date_request)
        _validate_result(manifest, date_request, date_result)
        pieces.append(date_result.data)
    per_date = pd.concat(pieces, ignore_index=True) if pieces else baseline.iloc[0:0].copy()
    _require_equal_data(
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


def _require_equal_data(
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
        pd.testing.assert_frame_equal(
            left,
            right,
            check_exact=bitwise,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    except AssertionError as exc:
        raise ContractViolationError(message, operator_id=manifest.operator_id) from exc
