"""Reusable contract checks for provider-compatible operator callables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from oxq.operators.errors import (
    ContractViolationError,
    InsufficientCrossSectionError,
    InsufficientHistoryError,
)
from oxq.operators.manifest import OperatorManifest
from oxq.operators.panel import QuantPanelAdapter
from oxq.operators.types import OperatorLifecycle, OperatorRequest, OperatorResult, OperatorScope

OperatorCallable = Callable[[OperatorRequest], OperatorResult]


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
    QuantPanelAdapter.validate_panel(normalized.input_panel, normalized.context)
    required_columns = set(manifest.raw["inputs"]["required_columns"])
    missing_columns = sorted(required_columns - set(normalized.input_panel.columns))
    if missing_columns:
        raise ContractViolationError(
            f"input is missing declared columns: {', '.join(missing_columns)}",
            operator_id=manifest.operator_id,
        )
    input_spec = manifest.raw["inputs"]
    asset_count = normalized.input_panel["code"].nunique()
    if asset_count < input_spec["min_assets"]:
        raise InsufficientCrossSectionError(
            f"input has {asset_count} assets; minimum is {input_spec['min_assets']}",
            operator_id=manifest.operator_id,
        )
    min_history = input_spec["min_history"]
    history = normalized.input_panel.groupby("code", sort=False, observed=True).size()
    if not history.empty and int(history.min()) < min_history:
        raise InsufficientHistoryError(
            f"input history is shorter than minimum {min_history}",
            operator_id=manifest.operator_id,
            details={"rows_by_code": {str(code): int(rows) for code, rows in history.items()}},
        )
    for column in required_columns:
        actual_dtype = str(normalized.input_panel[column].dtype)
        allowed_dtypes = input_spec["dtypes"][column]
        if actual_dtype not in allowed_dtypes:
            raise ContractViolationError(
                f"input column {column} dtype {actual_dtype} is not declared",
                operator_id=manifest.operator_id,
                details={"allowed": list(allowed_dtypes)},
            )
    checks = ["request"]

    snapshot = normalized.input_panel.copy(deep=True)
    result = operator(normalized)
    try:
        pd.testing.assert_frame_equal(normalized.input_panel, snapshot)
    except AssertionError as exc:
        raise ContractViolationError(
            "operator mutated input_panel",
            operator_id=manifest.operator_id,
        ) from exc
    checks.append("input_immutability")

    _validate_result(manifest, normalized, result)
    checks.extend(("output_contract", "provenance"))

    repeated = operator(_copy_request(normalized))
    _validate_result(manifest, normalized, repeated)
    _require_equal_data(result.data, repeated.data, manifest, "operator output must be deterministic")
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
    checks.append("determinism")

    shuffled_panel = normalized.input_panel.sample(frac=1, random_state=731).reset_index(drop=True)
    shuffled_request = _copy_request(normalized, panel=shuffled_panel)
    shuffled_result = operator(shuffled_request)
    _validate_result(manifest, shuffled_request, shuffled_result)
    _require_equal_data(
        _canonical(result.data),
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
            symbol_result = operator(symbol_request)
            _validate_result(manifest, symbol_request, symbol_result)
            pieces.append(symbol_result.data)
        per_symbol = pd.concat(pieces, ignore_index=True) if pieces else result.data.iloc[0:0].copy()
        _require_equal_data(
            _canonical(result.data),
            _canonical(per_symbol),
            manifest,
            "batch output must match per-symbol output",
        )
        checks.append("batch_consistency")

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
    expected_fields = {
        field["name_template"].format(**request.parameters)
        for field in outputs["fields"]
    }
    actual_fields = set(result.data.columns) - {"date", "code"}
    if actual_fields != expected_fields:
        raise ContractViolationError(
            "operator output must contain exactly the declared fields",
            operator_id=manifest.operator_id,
            details={"expected": sorted(expected_fields), "actual": sorted(actual_fields)},
        )
    declarations = {
        field["name_template"].format(**request.parameters): field
        for field in outputs["fields"]
    }
    for name, declaration in declarations.items():
        actual_dtype = str(result.data[name].dtype)
        if actual_dtype != declaration["dtype"]:
            raise ContractViolationError(
                f"output column {name} dtype {actual_dtype} does not match declared dtype {declaration['dtype']}",
                operator_id=manifest.operator_id,
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
        input_panel=request.input_panel.copy(deep=True) if panel is None else panel.copy(deep=True),
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
    try:
        pd.testing.assert_frame_equal(left, right)
    except AssertionError as exc:
        raise ContractViolationError(message, operator_id=manifest.operator_id) from exc
