from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from oxq.operators.errors import (
    CausalityViolationError,
    ContractViolationError,
    DuplicateKeyError,
    InsufficientCrossSectionError,
    InsufficientHistoryError,
    InvalidManifestError,
    InvalidPanelError,
    InvalidParameterError,
    MissingColumnError,
)
from oxq.operators.manifest import load_operator_manifest
from oxq.operators.panel import QuantPanelAdapter
from oxq.operators.testing import verify_operator_contract as _verify_operator_contract
from oxq.operators.types import OperatorRequest, OperatorScope
from tests.operators.fake_provider import IMPLEMENTATION_DIGEST
from tests.operators.fake_provider import sma as _sma


def sma(request: OperatorRequest):
    if "close" not in request.input_panel:
        raise MissingColumnError(
            "missing required input column close",
            operator_id=request.operator_id,
            details={"column": "close"},
        )
    return _sma(request)


def verify_operator_contract(
    manifest,
    operator,
    request,
    *,
    expected_distribution_version="1.0.0",
    expected_implementation_digest=IMPLEMENTATION_DIGEST,
):
    def required_column_aware_operator(provider_request: OperatorRequest):
        for column in manifest.raw["inputs"]["required_columns"]:
            if column not in provider_request.input_panel:
                raise MissingColumnError(
                    f"missing required input column {column}",
                    operator_id=provider_request.operator_id,
                    details={"column": column},
                )
        _reject_empty_input(manifest, provider_request)
        return operator(provider_request)

    return _verify_operator_contract(
        manifest,
        required_column_aware_operator,
        request,
        expected_distribution_version=expected_distribution_version,
        expected_implementation_digest=expected_implementation_digest,
    )


def _reject_empty_input(manifest, request: OperatorRequest) -> None:
    if not request.input_panel.empty:
        return
    input_spec = manifest.raw["inputs"]
    if manifest.execution_scope is OperatorScope.CROSS_SECTION or input_spec["min_history"] == 0:
        raise InsufficientCrossSectionError(
            "empty input does not satisfy the declared asset minimum",
            operator_id=request.operator_id,
            details={"min_assets": input_spec["min_assets"], "available_assets": 0},
        )
    raise InsufficientHistoryError(
        "empty input does not satisfy the declared history minimum",
        operator_id=request.operator_id,
        details={"min_history": input_spec["min_history"], "available_history": 0},
    )


def _artifact_wide_payload(valid_manifest_payload, *, period: int = 2):
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["missing_value_policy"] = {"kind": "require_complete"}
    period_declaration = payload["parameters"]["period"]
    period_declaration["affects_warmup"] = False
    period_declaration["affects_output_fields"] = False
    if payload["outputs"]["fields"][0]["name_template"] == "sma_{period}":
        payload["outputs"]["fields"][0]["name_template"] = f"sma_{period}"
    warmup = payload["outputs"]["warmup"]
    if warmup["kind"] == "parameter" and warmup["parameter"] == "period":
        payload["outputs"]["warmup"] = {"kind": "fixed", "rows": period - 1}
    return payload


def _load_contract_manifest(payload, *, period: int = 2):
    return load_operator_manifest(_artifact_wide_payload(payload, period=period))


def test_fake_provider_passes_reusable_contract_suite(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    manifest = _load_contract_manifest(valid_manifest_payload)
    report = verify_operator_contract(
        manifest,
        sma,
        request,
        expected_distribution_version="2.4.1",
    )

    assert report.operator_id == request.operator_id
    assert report.operator_version == manifest.operator_version
    assert report.distribution == manifest.distribution
    assert report.distribution_version == "2.4.1"
    assert report.manifest_digest == manifest.digest
    assert report.implementation_digest == IMPLEMENTATION_DIGEST
    assert report.parameters == {"period": 2}
    assert report.parameters_digest == "sha256:" + hashlib.sha256(b'{"period":2}').hexdigest()
    with pytest.raises(TypeError):
        report.parameters["period"] = 3
    expected_input_dtypes = {
        "close": "float64",
        "code": "string",
        "date": "datetime64[ns]",
    }
    canonical_input_dtypes = json.dumps(
        expected_input_dtypes,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert report.input_dtypes == expected_input_dtypes
    assert report.input_dtypes_digest == "sha256:" + hashlib.sha256(canonical_input_dtypes.encode()).hexdigest()
    with pytest.raises(TypeError):
        report.input_dtypes["close"] = "float32"
    expected_context = {
        "calendar": "XSHG",
        "currency": "CNY",
        "data_version": "fixture-v1",
        "evaluation_time": "close_t",
        "frequency": "1d",
        "price_adjustment": "forward_adjusted",
        "source": "fake",
        "timestamp_semantics": "session_date",
        "timezone": "Asia/Shanghai",
    }
    canonical_context = json.dumps(expected_context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    assert report.context == daily_context
    assert report.context_digest == "sha256:" + hashlib.sha256(canonical_context.encode()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        report.context.source = "mutated"
    assert report.passed is True
    assert report.checks == (
        "request",
        "input_immutability",
        "output_contract",
        "provenance",
        "required_inputs",
        "empty_input",
        "determinism",
        "causality",
        "unordered_input",
        "batch_consistency",
    )


def test_contract_suite_probes_each_required_data_column(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["inputs"]["required_columns"] = ["close", "volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    probed_columns: list[str] = []
    manifest = load_operator_manifest(payload)

    def structured_missing_column_provider(provider_request: OperatorRequest):
        for column in payload["inputs"]["required_columns"]:
            if column not in provider_request.input_panel:
                probed_columns.append(column)
                raise MissingColumnError(
                    f"missing required input column {column}",
                    operator_id=provider_request.operator_id,
                    details={"column": column},
                )
        _reject_empty_input(manifest, provider_request)
        return sma(provider_request)

    report = _verify_operator_contract(
        manifest,
        structured_missing_column_provider,
        request,
        expected_distribution_version="1.0.0",
        expected_implementation_digest=IMPLEMENTATION_DIGEST,
    )

    assert report.passed is True
    assert "required_inputs" in report.checks
    assert probed_columns == ["close", "volume"]


@pytest.mark.parametrize("behavior", ["silent", "key_error", "wrong_column"])
def test_contract_suite_rejects_unstructured_missing_required_column_behavior(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
    behavior: str,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )

    def noncompliant_provider(provider_request: OperatorRequest):
        if "close" not in provider_request.input_panel:
            if behavior == "key_error":
                raise KeyError("close")
            if behavior == "wrong_column":
                raise MissingColumnError(
                    "missing required input column",
                    operator_id=provider_request.operator_id,
                    details={"column": "volume"},
                )
            return _sma(
                OperatorRequest(
                    operator_id=provider_request.operator_id,
                    parameters=provider_request.parameters,
                    input_panel=provider_request.input_panel.assign(close=0.0),
                    context=provider_request.context,
                )
            )
        return _sma(provider_request)

    with pytest.raises(ContractViolationError, match="required input column close.*MissingColumnError") as exc_info:
        _verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            noncompliant_provider,
            request,
            expected_distribution_version="1.0.0",
            expected_implementation_digest=IMPLEMENTATION_DIGEST,
        )

    assert exc_info.value.to_dict()["details"] == {"column": "close"}


def test_contract_suite_rejects_unexpected_implementation_digest(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="implementation_digest mismatch"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            sma,
            request,
            expected_implementation_digest="sha256:" + "d" * 64,
        )


@pytest.mark.parametrize("version", ["1.0", "1.0.0-01", "1\u0660.0.0"])
def test_contract_suite_rejects_invalid_distribution_semantic_versions(
    version,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="semantic versioning") as exc_info:
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            tracked_provider,
            request,
            expected_distribution_version=version,
        )

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {"distribution_version": version}


def test_contract_suite_detects_input_mutation(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    def mutating_provider(mutating_request: OperatorRequest):
        mutating_request.input_panel.loc[:, "close"] = 0.0
        return sma(mutating_request)

    with pytest.raises(ContractViolationError, match="mutated input_panel"):
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), mutating_provider, request)


def test_contract_suite_structures_baseline_provider_exceptions(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def failing_provider(provider_request: OperatorRequest):
        raise RuntimeError("baseline failed")

    with pytest.raises(ContractViolationError, match="operator invocation failed") as exc_info:
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), failing_provider, request)

    assert exc_info.value.operator_id == "fake.indicators.sma"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_contract_suite_preserves_provider_operator_errors(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_error = InvalidPanelError("provider rejected input", operator_id="provider.identity")

    def failing_provider(provider_request: OperatorRequest):
        raise provider_error

    with pytest.raises(InvalidPanelError) as exc_info:
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), failing_provider, request)

    assert exc_info.value is provider_error


@pytest.mark.parametrize("phase", ["determinism", "causality", "scope"])
def test_contract_suite_structures_probe_provider_exceptions(
    phase,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["inputs"]["requires_sorted"] = True
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"])
    if phase == "determinism":
        payload["execution_scope"] = "panel"
        payload["causality"] = "future_using"
    elif phase == "causality":
        payload["execution_scope"] = "panel"
    else:
        payload["execution_scope"] = "cross_section"
        payload["causality"] = "future_using"
        payload["inputs"]["min_history"] = 1
        payload["outputs"] = {
            "fields": [{"name_template": "identity", "dtype": "float64"}],
            "alignment": "canonical_order",
            "warmup": {"kind": "fixed", "rows": 0},
            "nan_policy": "none",
            "multiple": False,
        }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    calls = 0
    full_rows = len(panel)

    def phase_failing_provider(provider_request: OperatorRequest):
        nonlocal calls
        calls += 1
        if phase == "determinism" and calls == 2:
            raise RuntimeError("repeat failed")
        if phase == "causality" and len(provider_request.input_panel) < full_rows:
            raise RuntimeError("causality failed")
        if phase == "scope" and len(provider_request.input_panel) < full_rows:
            raise RuntimeError("scope failed")
        if phase != "scope":
            return sma(provider_request)
        ordered = provider_request.input_panel.sort_values(["date", "code"], kind="stable", ignore_index=True)
        base = sma(provider_request)
        output = ordered[["date", "code"]].copy()
        output["identity"] = ordered["close"].astype("float64")
        diagnostics = type(base.diagnostics)(input_rows=len(ordered), output_rows=len(output))
        return type(base)(data=output, diagnostics=diagnostics, provenance=base.provenance)

    with pytest.raises(ContractViolationError, match="operator invocation failed") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), phase_failing_provider, request)

    assert exc_info.value.operator_id == "fake.indicators.sma"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_contract_suite_detects_undeclared_output(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def extra_output_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data["surprise"] = 1.0
        return result

    with pytest.raises(ContractViolationError, match="declared fields"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            extra_output_provider,
            request,
        )


def test_contract_suite_detects_nondeterminism(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    counter = 0

    def unstable_provider(provider_request: OperatorRequest):
        nonlocal counter
        result = sma(provider_request)
        counter += 1
        result.data.loc[result.data.index[-1], "sma_2"] += counter
        return result

    with pytest.raises(ContractViolationError, match="deterministic"):
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), unstable_provider, request)


def test_contract_suite_detects_nondeterministic_diagnostics(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    counter = 0

    def unstable_provider(provider_request: OperatorRequest):
        nonlocal counter
        result = sma(provider_request)
        counter += 1
        return type(result)(
            data=result.data,
            diagnostics=type(result.diagnostics)(
                input_rows=result.diagnostics.input_rows,
                output_rows=result.diagnostics.output_rows,
                warmup_rows=result.diagnostics.warmup_rows,
                warnings=(f"call-{counter}",),
            ),
            provenance=result.provenance,
        )

    with pytest.raises(ContractViolationError, match="diagnostics"):
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), unstable_provider, request)


def test_contract_report_binds_all_resolved_parameters(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["parameters"]["options"] = {
        "type": "object",
        "default": {"labels": ["alpha", "beta"]},
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    report = verify_operator_contract(load_operator_manifest(payload), sma, request)

    expected = {"options": {"labels": ["alpha", "beta"]}, "period": 2}
    canonical = json.dumps(expected, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    assert report.parameters == {"period": 2, "options": {"labels": ("alpha", "beta")}}
    assert report.parameters_digest == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    with pytest.raises(TypeError):
        report.parameters["options"]["labels"] = ()


def test_contract_suite_requires_exact_manifest_identity(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="another.operator",
        parameters={},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    with pytest.raises(ContractViolationError, match="does not match manifest"):
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), sma, request)


def test_contract_suite_enforces_declared_input_dtype_and_history(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    invalid_dtype = panel.copy()
    invalid_dtype["close"] = invalid_dtype["close"].astype("string")
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=invalid_dtype,
        context=daily_context,
    )
    manifest = _load_contract_manifest(valid_manifest_payload)
    with pytest.raises(ContractViolationError, match="dtype"):
        verify_operator_contract(manifest, sma, request)

    short = panel.groupby("code", sort=False, observed=True).head(1).reset_index(drop=True)
    short_request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=short,
        context=daily_context,
    )
    with pytest.raises(InsufficientHistoryError, match="history"):
        verify_operator_contract(manifest, sma, short_request)


def test_contract_suite_rejects_nonfinite_supplied_numeric_parameters(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["period"].update({"type": "number", "default": 2.0})
    payload["outputs"]["warmup"] = {"kind": "fixed", "rows": 1}
    payload["outputs"]["fields"][0]["name_template"] = "sma"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": float("nan")},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(InvalidParameterError, match="finite"):
        verify_operator_contract(_load_contract_manifest(payload), sma, request)


def test_contract_suite_enforces_dtype_of_present_optional_columns(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel["volume"] = panel["volume"].astype("string")
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="volume dtype"):
        verify_operator_contract(_load_contract_manifest(payload), sma, request)


def test_contract_suite_probes_every_valid_optional_column_subset(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["volume", "quality"]
    payload["inputs"]["dtypes"].update({"volume": ["int64"], "quality": ["float64"]})
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel["quality"] = 1.0
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    observed_optional_subsets: set[tuple[str, ...]] = set()

    def optional_aware_provider(provider_request: OperatorRequest):
        present = tuple(column for column in ("volume", "quality") if column in provider_request.input_panel)
        observed_optional_subsets.add(present)
        return sma(provider_request)

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        optional_aware_provider,
        request,
    )

    assert report.passed is True
    assert "optional_inputs" in report.checks
    assert observed_optional_subsets == {(), ("volume",), ("quality",), ("volume", "quality")}


def test_contract_suite_builds_optional_shapes_without_undeclared_columns(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel["quality"] = 1.0
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    def undeclared_dependent_optional_provider(provider_request: OperatorRequest):
        if "volume" not in provider_request.input_panel:
            provider_request.input_panel["quality"]
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="optional input column volume") as exc_info:
        verify_operator_contract(
            load_operator_manifest(payload),
            undeclared_dependent_optional_provider,
            request,
        )

    assert isinstance(exc_info.value.__cause__, KeyError)


def test_contract_suite_requires_fixture_to_cover_every_declared_optional_column(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["volume", "quality"]
    payload["inputs"]["dtypes"].update({"volume": ["int64"], "quality": ["float64"]})
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="fixture.*optional") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {"columns": ["quality"]}


def test_contract_suite_rejects_unbounded_optional_input_shape_certification(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    optional_columns = [f"optional_{index}" for index in range(9)]
    payload["inputs"]["optional_columns"] = optional_columns
    payload["inputs"]["dtypes"].update({column: ["float64"] for column in optional_columns})
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    for column in optional_columns:
        panel[column] = 1.0
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="at most 8 optional") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {"actual": 9, "maximum": 8}


@pytest.mark.parametrize("phase", ["causality", "scope", "unordered"])
def test_contract_suite_runs_behavioral_probes_for_each_optional_input_shape(
    phase,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    if phase == "causality":
        payload["execution_scope"] = "panel"
        payload["inputs"]["requires_sorted"] = True
    elif phase == "scope":
        payload["causality"] = "future_using"
        payload["inputs"]["requires_sorted"] = True
    else:
        payload["execution_scope"] = "panel"
        payload["causality"] = "future_using"
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    full_rows = len(panel)
    full_assets = int(panel["code"].nunique())

    def shape_dependent_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        provider_panel = provider_request.input_panel
        optional_absent = "volume" not in provider_panel
        keys = provider_panel[["date", "code"]].reset_index(drop=True)
        canonical_keys = keys.sort_values(["date", "code"], kind="stable", ignore_index=True)
        violates_probe = (
            (phase == "causality" and len(provider_panel) < full_rows)
            or (phase == "scope" and provider_panel["code"].nunique() < full_assets)
            or (phase == "unordered" and not keys.equals(canonical_keys))
        )
        if optional_absent and violates_probe:
            result.data.loc[:, "sma_2"] += 100.0
        return result

    expected_message = {
        "causality": "past_only",
        "scope": "other symbols",
        "unordered": "incidental input order",
    }[phase]
    with pytest.raises(ContractViolationError, match=expected_message):
        verify_operator_contract(load_operator_manifest(payload), shape_dependent_provider, request)


@pytest.mark.parametrize("nonfinite", [float("inf"), float("-inf")])
def test_contract_suite_rejects_nonfinite_numeric_output(
    nonfinite,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )

    def nonfinite_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data.loc[result.data.index[-1], "sma_2"] = nonfinite
        return result

    with pytest.raises(ContractViolationError, match="finite"):
        verify_operator_contract(load_operator_manifest(payload), nonfinite_provider, request)


def test_contract_suite_checks_optional_probe_determinism(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    absent_calls = 0

    def unstable_optional_provider(provider_request: OperatorRequest):
        nonlocal absent_calls
        result = sma(provider_request)
        if "volume" not in provider_request.input_panel:
            absent_calls += 1
            result.data.loc[result.data.index[-1], "sma_2"] += absent_calls
        return result

    with pytest.raises(ContractViolationError, match="optional input column volume.*deterministic"):
        verify_operator_contract(load_operator_manifest(payload), unstable_optional_provider, request)

    assert absent_calls == 2


def test_contract_suite_rejects_unconditional_reads_of_optional_columns(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def unconditional_optional_read(provider_request: OperatorRequest):
        provider_request.input_panel["volume"]
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="optional input column") as exc_info:
        verify_operator_contract(
            _load_contract_manifest(payload),
            unconditional_optional_read,
            request,
        )

    assert exc_info.value.to_dict()["details"] == {"column": "volume"}
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_contract_suite_preserves_operator_errors_from_optional_probes(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_error = InvalidPanelError("optional input rejected", operator_id="provider.identity")

    def optional_error_provider(provider_request: OperatorRequest):
        if "volume" not in provider_request.input_panel:
            raise provider_error
        return sma(provider_request)

    with pytest.raises(InvalidPanelError) as exc_info:
        verify_operator_contract(load_operator_manifest(payload), optional_error_provider, request)

    assert exc_info.value is provider_error


def test_contract_suite_probes_with_only_declared_input_columns(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    observed_columns: list[tuple[str, ...]] = []

    def tracking_provider(provider_request: OperatorRequest):
        observed_columns.append(tuple(provider_request.input_panel.columns))
        return sma(provider_request)

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        tracking_provider,
        request,
    )

    assert report.passed is True
    assert observed_columns.count(("date", "code", "close")) > 1


@pytest.mark.parametrize("optional_column", [None, "quality"])
@pytest.mark.parametrize("component", ["data", "diagnostics", "provenance", "metadata"])
def test_contract_suite_repeats_declared_only_full_shape(
    component,
    optional_column,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    declared_full_columns = {"date", "code", "close"}
    if optional_column is not None:
        payload["inputs"]["optional_columns"] = [optional_column]
        payload["inputs"]["dtypes"][optional_column] = ["float64"]
        panel[optional_column] = 1.0
        declared_full_columns.add(optional_column)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    declared_full_calls = 0

    def unstable_declared_full_provider(provider_request: OperatorRequest):
        nonlocal declared_full_calls
        result = sma(provider_request)
        if set(provider_request.input_panel.columns) != declared_full_columns:
            return result
        declared_full_calls += 1
        if declared_full_calls != 2:
            return result
        if component == "data":
            result.data.loc[result.data.index[-1], "sma_2"] += 1.0
            return result
        if component == "diagnostics":
            return replace(result, diagnostics=replace(result.diagnostics, warnings=("changed",)))
        if component == "metadata":
            return replace(result, metadata={"changed": True})

        class UnequalProvenance(type(result.provenance)):
            def __eq__(self, other):
                return self is other

        return replace(
            result,
            provenance=UnequalProvenance(
                operator_id=result.provenance.operator_id,
                operator_version=result.provenance.operator_version,
                implementation_digest=result.provenance.implementation_digest,
            ),
        )

    with pytest.raises(ContractViolationError, match=component if component != "data" else "deterministic"):
        verify_operator_contract(
            load_operator_manifest(payload),
            unstable_declared_full_provider,
            request,
        )

    assert declared_full_calls == 2


@pytest.mark.parametrize("optional_column", [None, "quality"])
@pytest.mark.parametrize("phase", ["causality", "unordered", "batch", "scope"])
def test_contract_suite_runs_behavioral_probes_for_declared_only_full_shape(
    phase,
    optional_column,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    declared_full_columns = {"date", "code", "close"}
    if optional_column is not None:
        payload["inputs"]["optional_columns"] = [optional_column]
        payload["inputs"]["dtypes"][optional_column] = ["float64"]
        panel[optional_column] = 1.0
        declared_full_columns.add(optional_column)
    if phase == "causality":
        payload["execution_scope"] = "panel"
        payload["inputs"]["requires_sorted"] = True
    elif phase == "unordered":
        payload["execution_scope"] = "panel"
        payload["causality"] = "future_using"
    elif phase == "batch":
        payload["causality"] = "future_using"
        payload["inputs"]["requires_sorted"] = True
    else:
        payload["execution_scope"] = "cross_section"
        payload["causality"] = "future_using"
        payload["inputs"]["requires_sorted"] = True
        payload["inputs"]["min_history"] = 1
        payload["outputs"] = {
            "fields": [{"name_template": "identity", "dtype": "float64"}],
            "alignment": "canonical_order",
            "warmup": {"kind": "fixed", "rows": 0},
            "nan_policy": "none",
            "multiple": False,
        }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    full_rows = len(panel)
    full_assets = int(panel["code"].nunique())
    full_dates = int(panel["date"].nunique())

    def probe_violating_declared_full_provider(provider_request: OperatorRequest):
        provider_panel = provider_request.input_panel
        if phase == "scope":
            ordered = provider_panel.sort_values(["date", "code"], kind="stable", ignore_index=True)
            base = sma(provider_request)
            output = ordered[["date", "code"]].copy()
            output["identity"] = ordered["close"].astype("float64")
            result = type(base)(
                data=output,
                diagnostics=replace(base.diagnostics, output_rows=len(output), warmup_rows=0),
                provenance=base.provenance,
            )
        else:
            result = sma(provider_request)
        if set(provider_panel.columns) != declared_full_columns:
            return result
        keys = provider_panel[["date", "code"]].reset_index(drop=True)
        canonical_keys = keys.sort_values(["date", "code"], kind="stable", ignore_index=True)
        violates_probe = (
            (phase == "causality" and len(provider_panel) < full_rows)
            or (phase == "unordered" and not keys.equals(canonical_keys))
            or (phase == "batch" and provider_panel["code"].nunique() < full_assets)
            or (phase == "scope" and provider_panel["date"].nunique() < full_dates)
        )
        if violates_probe:
            output_field = "identity" if phase == "scope" else "sma_2"
            result.data.loc[:, output_field] += 100.0
        return result

    expected_message = {
        "causality": "past_only",
        "unordered": "incidental input order",
        "batch": "other symbols",
        "scope": "per-date",
    }[phase]
    with pytest.raises(ContractViolationError, match=expected_message):
        verify_operator_contract(
            load_operator_manifest(payload),
            probe_violating_declared_full_provider,
            request,
        )


def test_contract_suite_rejects_unconditional_reads_of_undeclared_columns(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def undeclared_input_provider(provider_request: OperatorRequest):
        provider_request.input_panel["volume"]
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="undeclared input columns") as exc_info:
        verify_operator_contract(
            _load_contract_manifest(payload),
            undeclared_input_provider,
            request,
        )

    assert exc_info.value.to_dict()["details"] == {"columns": ["volume"]}
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_contract_suite_rejects_output_dependence_on_undeclared_columns(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def undeclared_dependent_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        if "volume" in provider_request.input_panel:
            result.data["sma_2"] += 1.0
        return result

    with pytest.raises(ContractViolationError, match="depend on undeclared input columns"):
        verify_operator_contract(
            load_operator_manifest(payload),
            undeclared_dependent_provider,
            request,
        )


def test_contract_suite_rejects_empty_explicit_keyed_baseline(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["outputs"]["alignment"] = "explicit_keyed_output"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    provider_calls = 0

    def empty_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        base = sma(provider_request)
        output = base.data.iloc[:0].copy()
        diagnostics = type(base.diagnostics)(
            input_rows=len(provider_request.input_panel),
            output_rows=0,
            warmup_rows=2,
            dropped_rows=len(provider_request.input_panel),
        )
        return type(base)(data=output, diagnostics=diagnostics, provenance=base.provenance)

    with pytest.raises(ContractViolationError, match="baseline.*at least one output row") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), empty_provider, request)

    assert provider_calls == 1
    assert exc_info.value.to_dict()["details"] == {"alignment": "explicit_keyed_output"}


def test_contract_suite_allows_empty_explicit_keyed_optional_probe(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"]["volume"] = ["int64"]
    payload["outputs"]["alignment"] = "explicit_keyed_output"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    observed_rows: list[int] = []

    def subset_provider(provider_request: OperatorRequest):
        base = sma(provider_request)
        output = base.data.iloc[: 1 if "volume" in provider_request.input_panel else 0].copy()
        observed_rows.append(len(output))
        diagnostics = type(base.diagnostics)(
            input_rows=len(provider_request.input_panel),
            output_rows=len(output),
            warmup_rows=2,
            dropped_rows=len(provider_request.input_panel) - len(output),
        )
        return type(base)(data=output, diagnostics=diagnostics, provenance=base.provenance)

    report = verify_operator_contract(load_operator_manifest(payload), subset_provider, request)

    assert report.passed is True
    assert "optional_inputs" in report.checks
    assert 0 in observed_rows


def test_contract_suite_enforces_require_complete_input_policy(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["missing_value_policy"] = {"kind": "require_complete"}
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel.loc[panel.index[-1], "close"] = float("nan")
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="require_complete"):
        verify_operator_contract(_load_contract_manifest(payload), sma, request)


def test_contract_suite_rejects_incomplete_declared_cross_sections(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["requires_complete_cross_section"] = True
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(index=0).reset_index(drop=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    with pytest.raises(InsufficientCrossSectionError, match="complete cross section"):
        verify_operator_contract(_load_contract_manifest(payload), sma, request)


def test_contract_suite_does_not_shuffle_for_sorted_only_operators(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["requires_sorted"] = True
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def sorted_only_provider(provider_request: OperatorRequest):
        keys = provider_request.input_panel[["date", "code"]].reset_index(drop=True)
        canonical = keys.sort_values(["date", "code"], kind="stable", ignore_index=True)
        if not keys.equals(canonical):
            raise AssertionError("provider received unsorted input")
        return sma(provider_request)

    report = verify_operator_contract(_load_contract_manifest(payload), sorted_only_provider, request)

    assert "unordered_input" not in report.checks


def test_contract_suite_rejects_unordered_probe_with_fewer_than_two_rows(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["min_history"] = 1
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).iloc[:1].reset_index(drop=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 1},
        input_panel=panel,
        context=daily_context,
    )

    def unexpected_provider(provider_request: OperatorRequest):
        raise AssertionError("provider must not be called for an invalid unordered probe fixture")

    with pytest.raises(ContractViolationError, match="unordered input probe") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload, period=1), unexpected_provider, request)

    assert exc_info.value.to_dict()["details"] == {
        "required_rows": 2,
        "available_rows": 1,
    }


def test_contract_suite_uses_reversal_for_two_row_unordered_probe(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["min_history"] = 1
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).iloc[:2].reset_index(drop=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 1},
        input_panel=panel,
        context=daily_context,
    )

    observed_keys: list[list[tuple[pd.Timestamp, str]]] = []

    def tracking_provider(provider_request: OperatorRequest):
        observed_keys.append(list(provider_request.input_panel[["date", "code"]].itertuples(index=False, name=None)))
        return sma(provider_request)

    report = verify_operator_contract(_load_contract_manifest(payload, period=1), tracking_provider, request)

    baseline_keys = list(panel[["date", "code"]].itertuples(index=False, name=None))
    assert report.passed is True
    assert observed_keys[-1] == [baseline_keys[1], baseline_keys[0]]


def test_contract_suite_uses_three_distinct_unordered_probes_for_three_rows(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload, period=1)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["min_history"] = 1
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).iloc[:3].reset_index(drop=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 1},
        input_panel=panel,
        context=daily_context,
    )
    observed_keys: list[list[tuple[pd.Timestamp, str]]] = []

    def tracking_provider(provider_request: OperatorRequest):
        observed_keys.append(list(provider_request.input_panel[["date", "code"]].itertuples(index=False, name=None)))
        return sma(provider_request)

    report = verify_operator_contract(load_operator_manifest(payload), tracking_provider, request)

    baseline = list(panel[["date", "code"]].itertuples(index=False, name=None))
    assert report.passed is True
    assert observed_keys[-3] == list(reversed(baseline))
    assert observed_keys[-2] == baseline[1:] + baseline[:1]
    assert observed_keys[-1] == [baseline[0], baseline[2], baseline[1]]


def test_contract_suite_rejects_circular_neighbor_dependence_with_adjacent_swap(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload, period=1)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["min_history"] = 1
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 1},
        input_panel=panel,
        context=daily_context,
    )

    def circular_neighbor_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        raw = provider_request.input_panel.reset_index(drop=True)
        neighbors = raw[["date", "code"]].copy()
        neighbors["sma_1"] = raw["close"].shift(1, fill_value=raw["close"].iloc[-1]) + raw["close"].shift(
            -1, fill_value=raw["close"].iloc[0]
        )
        return replace(
            result,
            data=result.data.drop(columns=["sma_1"]).merge(
                neighbors,
                on=["date", "code"],
                how="left",
                validate="one_to_one",
            ),
        )

    with pytest.raises(ContractViolationError, match="incidental input order"):
        verify_operator_contract(
            load_operator_manifest(payload),
            circular_neighbor_provider,
            request,
        )


def test_contract_suite_applies_declared_tolerance_to_unordered_probe(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["determinism"] = {
        "bitwise": False,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 0.0,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def order_sensitive_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        keys = provider_request.input_panel[["date", "code"]].reset_index(drop=True)
        canonical = keys.sort_values(["date", "code"], kind="stable", ignore_index=True)
        if not keys.equals(canonical):
            result.data.loc[result.data.index[-1], "sma_2"] += 1e-7
        return result

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        order_sensitive_provider,
        request,
    )

    assert report.passed is True


def test_contract_suite_rejects_use_before_declared_availability(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=replace(daily_context, evaluation_time="open_t"),
    )

    with pytest.raises(CausalityViolationError, match="not available"):
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), sma, request)


def test_contract_suite_enriches_input_panel_errors_with_manifest_operator_id(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    duplicate_panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=duplicate_panel,
        context=daily_context,
    )

    with pytest.raises(DuplicateKeyError) as exc_info:
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), sma, request)

    assert exc_info.value.to_dict() == {
        "code": "duplicate_key",
        "operator_id": "fake.indicators.sma",
        "message": "QuantPanel contains duplicate (date, code) keys",
        "details": {"count": 2},
        "retryable": False,
    }
    assert isinstance(exc_info.value.__cause__, DuplicateKeyError)
    assert exc_info.value.__cause__.operator_id is None


def test_contract_suite_enriches_output_panel_errors_with_manifest_operator_id(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def invalid_output_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data.loc[result.data.index[0], "code"] = ""
        return result

    with pytest.raises(InvalidPanelError) as exc_info:
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            invalid_output_provider,
            request,
        )

    assert exc_info.value.to_dict() == {
        "code": "invalid_panel",
        "operator_id": "fake.indicators.sma",
        "message": "QuantPanel code must contain non-empty strings",
        "details": {},
        "retryable": False,
    }
    assert type(exc_info.value.__cause__) is InvalidPanelError
    assert exc_info.value.__cause__.operator_id is None


def test_contract_suite_rejects_non_string_output_column_labels_structurally(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def non_string_column_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data[1] = 0.0
        return result

    with pytest.raises(ContractViolationError, match="column labels must be strings") as exc_info:
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            non_string_column_provider,
            request,
        )

    assert exc_info.value.to_dict() == {
        "code": "contract_violation",
        "operator_id": "fake.indicators.sma",
        "message": "operator output column labels must be strings",
        "details": {"columns": [{"position": 3, "type": "int"}]},
        "retryable": False,
    }


def test_contract_suite_rejects_object_output_fields_for_bitwise_certification(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["outputs"]["fields"][0]["dtype"] = "object"
    provider_calls = 0
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="object output fields.*bitwise") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {"fields": ["sma_2"]}


def test_contract_suite_enforces_declared_output_bounds(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0]["minimum"] = 0.0
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def out_of_bounds_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data.loc[result.data.index[-1], "sma_2"] = -1.0
        return result

    with pytest.raises(ContractViolationError, match="minimum"):
        verify_operator_contract(_load_contract_manifest(payload), out_of_bounds_provider, request)


def test_contract_suite_enforces_output_nan_policy_none(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["nan_policy"] = "none"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="nan_policy=none"):
        verify_operator_contract(_load_contract_manifest(payload), sma, request)


def test_contract_suite_confines_warmup_nans_to_declared_rows(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def late_nan_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data.loc[result.data.index[-1], "sma_2"] = float("nan")
        return result

    with pytest.raises(ContractViolationError, match="outside declared warmup"):
        verify_operator_contract(_load_contract_manifest(valid_manifest_payload), late_nan_provider, request)


def test_manifest_boundary_rejects_negative_resolved_warmup_without_nans(
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["warmup"]["offset"] = -2
    payload["outputs"]["nan_policy"] = "none"

    with pytest.raises(InvalidManifestError, match="minimum plus offset must be non-negative"):
        load_operator_manifest(payload)


def test_contract_suite_validates_dropped_row_count(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def invalid_diagnostics_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        return type(result)(
            data=result.data,
            diagnostics=type(result.diagnostics)(
                input_rows=result.diagnostics.input_rows,
                output_rows=result.diagnostics.output_rows,
                warmup_rows=2,
                dropped_rows=1,
            ),
            provenance=result.provenance,
        )

    with pytest.raises(ContractViolationError, match="dropped_rows"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            invalid_diagnostics_provider,
            request,
        )


def test_contract_suite_refuses_parameterized_duplicate_output_fields(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["alias"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"].append({"name_template": "sma_{alias}", "dtype": "float64", "minimum": 0.0})
    payload["outputs"]["multiple"] = True
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2, "alias": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="parameters that affect output fields") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), sma, request)

    assert exc_info.value.to_dict()["details"] == {"parameters": ["alias"]}


def test_contract_suite_refuses_parameterized_reserved_output_fields(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["field"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{field}"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2, "field": "date"},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="parameters that affect output fields") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), sma, request)

    assert exc_info.value.to_dict()["details"] == {"parameters": ["field"]}


def test_contract_suite_refuses_parameterized_empty_output_fields(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["field"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{field}"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2, "field": ""},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def empty_output_name_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        output = result.data.rename(columns={"sma_2": provider_request.parameters["field"]})
        return type(result)(data=output, diagnostics=result.diagnostics, provenance=result.provenance)

    with pytest.raises(ContractViolationError, match="parameters that affect output fields") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), empty_output_name_provider, request)

    assert exc_info.value.to_dict()["details"] == {"parameters": ["field"]}


def test_contract_suite_checks_immutability_on_unordered_probe(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def unordered_mutating_provider(provider_request: OperatorRequest):
        keys = provider_request.input_panel[["date", "code"]].reset_index(drop=True)
        canonical = keys.sort_values(["date", "code"], kind="stable", ignore_index=True)
        if not keys.equals(canonical):
            provider_request.input_panel.sort_values(["date", "code"], kind="stable", inplace=True)
            provider_request.input_panel.reset_index(drop=True, inplace=True)
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="mutated input_panel"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            unordered_mutating_provider,
            request,
        )


def test_contract_suite_snapshots_first_result_before_repeat_probe(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    shared = None
    calls = 0

    def shared_result_provider(provider_request: OperatorRequest):
        nonlocal calls, shared
        result = sma(provider_request)
        if shared is None:
            shared = result.data
        calls += 1
        shared.loc[shared.index[-1], "sma_2"] = float(calls)
        return type(result)(data=shared, diagnostics=result.diagnostics, provenance=result.provenance)

    with pytest.raises(ContractViolationError, match="deterministic"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            shared_result_provider,
            request,
        )


def test_contract_suite_includes_metadata_in_determinism_check(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    calls = 0

    def varying_metadata_provider(provider_request: OperatorRequest):
        nonlocal calls
        result = sma(provider_request)
        calls += 1
        return type(result)(
            data=result.data,
            diagnostics=result.diagnostics,
            provenance=result.provenance,
            metadata={"call": calls},
        )

    with pytest.raises(ContractViolationError, match="metadata"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            varying_metadata_provider,
            request,
        )


def test_contract_suite_compares_array_metadata_without_ambiguous_truth_values(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def array_metadata_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        return type(result)(
            data=result.data,
            diagnostics=result.diagnostics,
            provenance=result.provenance,
            metadata={"weights": np.array([1.0, 2.0])},
        )

    report = verify_operator_contract(
        _load_contract_manifest(valid_manifest_payload),
        array_metadata_provider,
        request,
    )

    assert report.passed is True


@pytest.mark.parametrize(
    "metadata_factory",
    [
        pytest.param(lambda value: pd.DataFrame({"score": [value]}), id="dataframe"),
        pytest.param(lambda value: pd.Series([value], name="score"), id="series"),
    ],
)
def test_contract_suite_compares_structured_metadata_representation_wise_for_bitwise_determinism(
    metadata_factory,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    values = iter((0.0, -0.0))

    def representation_changing_metadata_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        return type(result)(
            data=result.data,
            diagnostics=result.diagnostics,
            provenance=result.provenance,
            metadata={"statistics": [({"nested": metadata_factory(next(values))},)]},
        )

    with pytest.raises(ContractViolationError, match="metadata.*deterministic"):
        verify_operator_contract(
            load_operator_manifest(payload),
            representation_changing_metadata_provider,
            request,
        )


@pytest.mark.parametrize(
    "metadata_factory",
    [
        pytest.param(lambda value: pd.DataFrame({"score": [value]}), id="dataframe"),
        pytest.param(lambda value: pd.Series([value], name="score"), id="series"),
    ],
)
def test_contract_suite_applies_declared_tolerance_to_structured_metadata(
    metadata_factory,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["determinism"] = {
        "bitwise": False,
        "absolute_tolerance": 1e-3,
        "relative_tolerance": 0.0,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    values = iter((0.0, 5e-4))

    def tolerant_metadata_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        return type(result)(
            data=result.data,
            diagnostics=result.diagnostics,
            provenance=result.provenance,
            metadata={"statistics": [({"nested": metadata_factory(next(values))},)]},
        )

    report = verify_operator_contract(
        load_operator_manifest(payload),
        tolerant_metadata_provider,
        request,
    )

    assert report.passed is True


@pytest.mark.parametrize(
    "metadata_array",
    [
        np.array([1.0, np.nan]),
        np.array([1, pd.NA], dtype=object),
        np.array([np.datetime64("2024-01-01"), np.datetime64("NaT")]),
    ],
    ids=["nan", "pandas-na", "nat"],
)
def test_contract_suite_treats_matching_missing_values_in_array_metadata_as_equal(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
    metadata_array,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def array_metadata_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        return type(result)(
            data=result.data,
            diagnostics=result.diagnostics,
            provenance=result.provenance,
            metadata={"values": metadata_array.copy()},
        )

    report = verify_operator_contract(
        _load_contract_manifest(valid_manifest_payload),
        array_metadata_provider,
        request,
    )

    assert report.passed is True


def test_contract_suite_snapshots_nested_read_only_metadata(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def nested_read_only_metadata_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        return type(result)(
            data=result.data,
            diagnostics=result.diagnostics,
            provenance=result.provenance,
            metadata={
                "config": MappingProxyType(
                    {
                        "window": 2,
                        "nested": MappingProxyType({"label": "stable"}),
                    }
                )
            },
        )

    report = verify_operator_contract(
        _load_contract_manifest(valid_manifest_payload),
        nested_read_only_metadata_provider,
        request,
    )

    assert report.passed is True


@pytest.mark.parametrize("missing", [float("nan"), pd.NA])
def test_contract_suite_treats_repeated_missing_metadata_as_equal(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
    missing,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def missing_metadata_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        return type(result)(
            data=result.data,
            diagnostics=result.diagnostics,
            provenance=result.provenance,
            metadata={"score": missing},
        )

    report = verify_operator_contract(
        _load_contract_manifest(valid_manifest_payload),
        missing_metadata_provider,
        request,
    )

    assert report.passed is True


def test_contract_suite_deeply_checks_object_cell_immutability(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["attributes"]
    payload["inputs"]["dtypes"]["attributes"] = ["object"]
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel["attributes"] = [[{"source": "raw"}] for _ in range(len(panel))]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    def object_mutating_provider(provider_request: OperatorRequest):
        provider_request.input_panel.loc[0, "attributes"][0]["source"] = "mutated"
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="mutated input_panel"):
        verify_operator_contract(_load_contract_manifest(payload), object_mutating_provider, request)


def test_contract_suite_detects_mapping_type_change_in_object_input(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["required_columns"].append("attributes")
    payload["inputs"]["dtypes"]["attributes"] = ["object"]
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"])
    panel["attributes"] = [{"source": "raw"} for _ in range(len(panel))]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    def mapping_type_mutating_provider(provider_request: OperatorRequest):
        provider_request.input_panel.at[0, "attributes"] = OrderedDict(source="raw")
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="mutated input_panel"):
        verify_operator_contract(
            _load_contract_manifest(payload),
            mapping_type_mutating_provider,
            request,
        )


@pytest.mark.parametrize(
    "representation",
    ["python_signed_zero", "python_nan_payload", "numpy_scalar", "numpy_array"],
)
def test_contract_suite_detects_representation_only_mutation_in_nested_object_input(
    representation,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["required_columns"].append("attributes")
    payload["inputs"]["dtypes"]["attributes"] = ["object"]
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"])
    first_nan = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)[0].item()
    second_nan = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)[0].item()

    def make_attributes():
        if representation == "python_nan_payload":
            value = first_nan
        elif representation == "numpy_scalar":
            value = np.float64(0.0)
        elif representation == "numpy_array":
            value = np.array([0.0], dtype=np.float64)
        else:
            value = 0.0
        return {"weights": [(value,)]}

    panel["attributes"] = [make_attributes() for _ in range(len(panel))]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    def representation_mutating_provider(provider_request: OperatorRequest):
        if "attributes" not in provider_request.input_panel:
            raise MissingColumnError(
                "missing required input column attributes",
                operator_id=provider_request.operator_id,
                details={"column": "attributes"},
            )
        weights = provider_request.input_panel.loc[0, "attributes"]["weights"]
        if representation == "python_nan_payload":
            weights[0] = (second_nan,)
        elif representation == "numpy_scalar":
            weights[0] = (np.float64(-0.0),)
        elif representation == "numpy_array":
            weights[0][0][0] = -0.0
        else:
            weights[0] = (-0.0,)
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="mutated input_panel"):
        verify_operator_contract(_load_contract_manifest(payload), representation_mutating_provider, request)


def test_contract_suite_rejects_uncertifiable_object_input_before_invocation(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["required_columns"].append("attributes")
    payload["inputs"]["dtypes"]["attributes"] = ["object"]
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"])
    panel["attributes"] = [object() for _ in range(len(panel))]
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="cannot certify object input.*representation-wise"):
        verify_operator_contract(_load_contract_manifest(payload), tracked_provider, request)

    assert provider_calls == 0


def test_contract_suite_compares_input_immutability_representation_wise(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    def one_ulp_mutating_provider(provider_request: OperatorRequest):
        current = provider_request.input_panel.loc[0, "close"]
        provider_request.input_panel.loc[0, "close"] = np.nextafter(current, np.inf)
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="mutated input_panel"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            one_ulp_mutating_provider,
            request,
        )


def test_contract_suite_verifies_past_only_causality_with_truncated_histories(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def future_using_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        ordered = provider_request.input_panel.sort_values(["code", "date"], kind="stable").copy()
        ordered["sma_2"] = ordered.groupby("code", sort=False)["close"].shift(-1).fillna(ordered["close"])
        result.data.loc[:, "sma_2"] = ordered.sort_values(["date", "code"], kind="stable")["sma_2"].to_numpy()
        return result

    with pytest.raises(ContractViolationError, match="past_only"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            future_using_provider,
            request,
        )


def test_contract_suite_applies_declared_tolerance_to_past_only_causality(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["determinism"] = {
        "bitwise": False,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 0.0,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def subtly_future_using_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        final_close = provider_request.input_panel.groupby("code", sort=False)["close"].last()
        future_adjustment = result.data["code"].map(final_close).astype("float64") * 1e-7
        result.data.loc[:, "sma_2"] += future_adjustment
        return result

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        subtly_future_using_provider,
        request,
    )

    assert report.passed is True


def test_contract_suite_verifies_past_only_causality_for_panel_scope(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def future_using_panel_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        ordered = provider_request.input_panel.sort_values(["code", "date"], kind="stable").copy()
        ordered["sma_2"] = ordered.groupby("code", sort=False)["close"].shift(-1).fillna(ordered["close"])
        result.data.loc[:, "sma_2"] = ordered.sort_values(["date", "code"], kind="stable")["sma_2"].to_numpy()
        return result

    with pytest.raises(ContractViolationError, match="past_only"):
        verify_operator_contract(_load_contract_manifest(payload), future_using_panel_provider, request)


def test_contract_suite_verifies_past_only_causality_for_research_only_scope(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "research_only"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def future_using_research_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        ordered = provider_request.input_panel.sort_values(["code", "date"], kind="stable").copy()
        ordered["sma_2"] = ordered.groupby("code", sort=False)["close"].shift(-1).fillna(ordered["close"])
        result.data.loc[:, "sma_2"] = ordered.sort_values(["date", "code"], kind="stable")["sma_2"].to_numpy()
        return result

    with pytest.raises(ContractViolationError, match="past_only"):
        verify_operator_contract(
            _load_contract_manifest(payload),
            future_using_research_provider,
            request,
        )


def test_contract_suite_rejects_past_only_fixture_without_a_valid_proper_prefix(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["min_history"] = 3
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="causality probe") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), sma, request)

    assert exc_info.value.to_dict()["details"] == {
        "min_history": 3,
        "available_dates": 3,
    }


def test_contract_suite_skips_invalid_past_only_prefixes_for_staggered_panel(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["inputs"]["min_assets"] = 2
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    original_last_date = panel["date"].max()
    last_session = panel.loc[panel["date"] == original_last_date]
    extensions = []
    for offset in range(1, 4):
        extension = last_session.copy(deep=True)
        extension["date"] = extension["date"] + pd.offsets.BDay(offset)
        extension["close"] = extension["close"] + float(offset)
        extensions.append(extension)
    panel = pd.concat([panel, *extensions], ignore_index=True)
    staggered_code = panel["code"].max()
    panel = panel.loc[~((panel["code"] == staggered_code) & (panel["date"] <= original_last_date))].sort_values(
        ["date", "code"], kind="stable", ignore_index=True
    )
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    full_last_date = panel["date"].max()
    probed_prefixes = []

    def minimum_enforcing_provider(provider_request: OperatorRequest):
        provider_panel = provider_request.input_panel
        if provider_panel["date"].max() < full_last_date:
            history = provider_panel.groupby("code", sort=False, observed=True).size()
            if provider_panel["code"].nunique() < 2 or int(history.min()) < 2:
                raise AssertionError("provider received an invalid past_only prefix")
            probed_prefixes.append(provider_panel["date"].max())
        return sma(provider_request)

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        minimum_enforcing_provider,
        request,
    )

    assert "causality" in report.checks
    expected_prefix = panel["date"].drop_duplicates().sort_values().iloc[-2]
    assert probed_prefixes == [expected_prefix, expected_prefix]


@pytest.mark.parametrize("window_alignment", ["centered", "trailing"])
def test_contract_suite_refuses_parameterized_causality_before_provider_call(
    window_alignment,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["window_alignment"] = {
        "type": "string",
        "default": "trailing",
        "required": False,
        "enum": ["centered", "trailing"],
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": True,
        "affects_availability": False,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2, "window_alignment": window_alignment},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="parameters that affect causality") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict() == {
        "code": "contract_violation",
        "operator_id": "fake.indicators.sma",
        "message": "contract checker cannot certify parameters that affect causality",
        "details": {"parameters": ["window_alignment"]},
        "retryable": False,
    }


def test_contract_suite_refuses_parameterized_availability_before_provider_call(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["release_lag"] = {
        "type": "integer",
        "default": 0,
        "required": False,
        "unit": "sessions",
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": True,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2, "release_lag": 0},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="parameters that affect availability") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict() == {
        "code": "contract_violation",
        "operator_id": "fake.indicators.sma",
        "message": "contract checker cannot certify parameters that affect availability",
        "details": {"parameters": ["release_lag"]},
        "retryable": False,
    }


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("affects_output_fields", "parameters that affect output fields"),
        ("affects_warmup", "parameters that affect warmup"),
    ],
)
def test_contract_suite_refuses_artifact_wide_parameter_effects_before_provider_call(
    flag,
    message,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["parameters"]["period"][flag] = True
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match=message) as exc_info:
        verify_operator_contract(load_operator_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {"parameters": ["period"]}


@pytest.mark.parametrize("nan_policy", ["propagate", "declared_missing"])
def test_contract_suite_refuses_uncertified_nan_policies_before_provider_call(
    nan_policy,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["outputs"]["nan_policy"] = nan_policy
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="cannot certify nan_policy") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {"nan_policy": nan_policy}


@pytest.mark.parametrize(
    ("policy", "expected_details"),
    [
        ({"kind": "propagate"}, {"missing_value_policy": "propagate"}),
        ({"kind": "skip_window"}, {"missing_value_policy": "skip_window"}),
        ({"kind": "explicit_fill", "value": 0.0}, {"missing_value_policy": "explicit_fill"}),
    ],
)
def test_contract_suite_refuses_uncertified_input_missing_policies_before_provider_call(
    policy,
    expected_details,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["inputs"]["missing_value_policy"] = policy
    provider_calls = 0
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="cannot certify input missing_value_policy") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == expected_details


def test_contract_suite_enforces_cross_section_min_assets_at_every_date(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "cross_section"
    payload["inputs"]["min_assets"] = 2
    payload["inputs"]["min_history"] = 1
    payload["outputs"] = {
        "fields": [{"name_template": "identity", "dtype": "float64"}],
        "alignment": "canonical_order",
        "warmup": {"kind": "fixed", "rows": 0},
        "nan_policy": "none",
        "multiple": False,
    }
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    last_date = panel["date"].max()
    last_code = panel["code"].max()
    panel = panel.loc[~((panel["date"] == last_date) & (panel["code"] == last_code))].reset_index(drop=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    def identity_provider(provider_request: OperatorRequest):
        ordered = provider_request.input_panel.sort_values(["date", "code"], kind="stable", ignore_index=True)
        output = ordered[["date", "code"]].copy()
        output["identity"] = ordered["close"].astype("float64")
        base = sma(provider_request)
        return type(base)(
            data=output,
            diagnostics=type(base.diagnostics)(input_rows=len(ordered), output_rows=len(output)),
            provenance=base.provenance,
        )

    with pytest.raises(InsufficientCrossSectionError, match="minimum 2 at every date") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), identity_provider, request)

    assert exc_info.value.details["min_assets"] == 2
    assert 1 in exc_info.value.details["assets_by_date"].values()


def test_contract_suite_verifies_cross_section_scope_per_date(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "cross_section"
    payload["inputs"]["min_history"] = 1
    payload["outputs"] = {
        "fields": [{"name_template": "centered", "dtype": "float64"}],
        "alignment": "canonical_order",
        "warmup": {"kind": "fixed", "rows": 0},
        "nan_policy": "none",
        "multiple": False,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def time_mixing_provider(provider_request: OperatorRequest):
        panel = provider_request.input_panel.sort_values(["date", "code"], kind="stable", ignore_index=True)
        output = panel[["date", "code"]].copy()
        output["centered"] = panel["close"] - panel["close"].mean()
        base = sma(
            type(provider_request)(
                operator_id=provider_request.operator_id,
                parameters={"period": 2},
                input_panel=provider_request.input_panel,
                context=provider_request.context,
            )
        )
        return type(base)(
            data=output,
            diagnostics=type(base.diagnostics)(input_rows=len(panel), output_rows=len(output)),
            provenance=base.provenance,
        )

    with pytest.raises(ContractViolationError, match="per-date"):
        verify_operator_contract(_load_contract_manifest(payload), time_mixing_provider, request)


def test_contract_suite_applies_declared_tolerance_to_cross_section_scope_probe(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "cross_section"
    payload["causality"] = "future_using"
    payload["inputs"]["min_history"] = 1
    payload["outputs"] = {
        "fields": [{"name_template": "identity", "dtype": "float64"}],
        "alignment": "canonical_order",
        "warmup": {"kind": "fixed", "rows": 0},
        "nan_policy": "none",
        "multiple": False,
    }
    payload["determinism"] = {
        "bitwise": False,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 0.0,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def date_mixing_provider(provider_request: OperatorRequest):
        panel = provider_request.input_panel.sort_values(["date", "code"], kind="stable", ignore_index=True)
        output = panel[["date", "code"]].copy()
        output["identity"] = panel["close"].astype("float64") + panel["date"].nunique() * 1e-7
        base = sma(provider_request)
        return type(base)(
            data=output,
            diagnostics=type(base.diagnostics)(input_rows=len(panel), output_rows=len(output)),
            provenance=base.provenance,
        )

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        date_mixing_provider,
        request,
    )

    assert report.passed is True


def test_contract_suite_rejects_cross_section_scope_probe_with_one_date(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "cross_section"
    payload["inputs"]["min_history"] = 1
    payload["inputs"]["requires_sorted"] = True
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel = panel.loc[panel["date"] == panel["date"].min()].reset_index(drop=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="cross_section scope probe") as exc_info:
        verify_operator_contract(
            _load_contract_manifest(payload),
            tracked_provider,
            request,
        )

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {
        "required_dates": 2,
        "available_dates": 1,
    }


def test_contract_suite_rejects_time_series_scope_probe_with_one_asset(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel = panel.loc[panel["code"] == panel["code"].min()].reset_index(drop=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="time_series scope probe") as exc_info:
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            sma,
            request,
        )

    assert exc_info.value.to_dict()["details"] == {
        "required_assets": 2,
        "available_assets": 1,
    }


def test_contract_suite_time_series_scope_probes_retain_manifest_min_assets(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["min_assets"] = 2
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    extra_symbol = panel.loc[panel["code"] == panel["code"].min()].copy()
    extra_symbol.loc[:, "code"] = "300001.SZ"
    extra_symbol.loc[:, "close"] += 5.0
    panel = pd.concat([panel, extra_symbol], ignore_index=True).sort_values(["date", "code"], kind="stable", ignore_index=True)
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    probed_asset_counts: list[int] = []

    def min_assets_enforcing_provider(provider_request: OperatorRequest):
        asset_count = int(provider_request.input_panel["code"].nunique())
        if asset_count < 2:
            raise AssertionError("provider received a probe below manifest min_assets")
        probed_asset_counts.append(asset_count)
        return sma(provider_request)

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        min_assets_enforcing_provider,
        request,
    )

    assert report.passed is True
    assert min(probed_asset_counts) == 2


def test_contract_suite_rejects_time_series_mixing_in_specific_min_asset_subset(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["min_assets"] = 2
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    first_symbol = panel.loc[panel["code"] == panel["code"].min()].copy()
    third_symbol = first_symbol.copy()
    third_symbol.loc[:, "code"] = "300001.SZ"
    third_symbol.loc[:, "close"] += 5.0
    fourth_symbol = first_symbol.copy()
    fourth_symbol.loc[:, "code"] = "300002.SZ"
    fourth_symbol.loc[:, "close"] += 10.0
    panel = pd.concat([panel, third_symbol, fourth_symbol], ignore_index=True).sort_values(
        ["date", "code"],
        kind="stable",
        ignore_index=True,
    )
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    mixed_pair = frozenset(("000001.SZ", "300002.SZ"))

    def pair_specific_mixing_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        symbols = frozenset(provider_request.input_panel["code"].unique())
        if symbols == mixed_pair:
            result.data.loc[:, "sma_2"] += 1.0
        return result

    with pytest.raises(ContractViolationError, match="other symbols"):
        verify_operator_contract(
            _load_contract_manifest(payload),
            pair_specific_mixing_provider,
            request,
        )


def test_contract_suite_bounds_time_series_subset_probes_for_thirty_symbols(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["min_assets"] = 2
    payload["inputs"]["optional_columns"] = []
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"])
    template = panel.loc[panel["code"] == panel["code"].min()].copy()
    extra_symbols = []
    for offset in range(1, 29):
        symbol = template.copy()
        symbol.loc[:, "code"] = f"300{offset:03d}.SZ"
        symbol.loc[:, "close"] += float(offset)
        extra_symbols.append(symbol)
    panel = pd.concat([panel, *extra_symbols], ignore_index=True).sort_values(
        ["date", "code"],
        kind="stable",
        ignore_index=True,
    )
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    all_codes = set(panel["code"].unique())
    probed_codes_by_size: dict[int, set[str]] = {}
    subset_probe_calls = 0

    def bounded_probe_provider(provider_request: OperatorRequest):
        nonlocal subset_probe_calls
        included_codes = set(provider_request.input_panel["code"].unique())
        if len(included_codes) < len(all_codes):
            subset_probe_calls += 1
            if subset_probe_calls > 300:
                raise AssertionError("time-series subset probe count exceeded its bound")
            probed_codes_by_size.setdefault(len(included_codes), set()).update(included_codes)
        return sma(provider_request)

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        bounded_probe_provider,
        request,
    )

    assert report.passed is True
    assert subset_probe_calls <= 300
    assert set(probed_codes_by_size) == set(range(2, 30))
    assert all(probed_codes == all_codes for probed_codes in probed_codes_by_size.values())


def test_contract_suite_applies_declared_tolerance_to_time_series_scope_probe(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["determinism"] = {
        "bitwise": False,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 0.0,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def symbol_mixing_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data.loc[:, "sma_2"] += provider_request.input_panel["code"].nunique() * 1e-7
        return result

    report = verify_operator_contract(
        _load_contract_manifest(payload),
        symbol_mixing_provider,
        request,
    )

    assert report.passed is True


def test_contract_suite_rejects_time_series_scope_probe_without_support_assets(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["min_assets"] = 2
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    provider_calls = 0

    def tracked_provider(provider_request: OperatorRequest):
        nonlocal provider_calls
        provider_calls += 1
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="time_series scope probe") as exc_info:
        verify_operator_contract(_load_contract_manifest(payload), tracked_provider, request)

    assert provider_calls == 0
    assert exc_info.value.to_dict()["details"] == {
        "required_assets": 3,
        "available_assets": 2,
    }


def test_contract_suite_uses_exact_determinism_by_default(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    calls = 0

    def subtly_unstable_provider(provider_request: OperatorRequest):
        nonlocal calls
        result = sma(provider_request)
        calls += 1
        result.data.loc[result.data.index[-1], "sma_2"] += calls * 1e-7
        return result

    with pytest.raises(ContractViolationError, match="deterministic"):
        verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            subtly_unstable_provider,
            request,
        )


@pytest.mark.parametrize(
    ("first_bits", "second_bits"),
    [
        (0x0000000000000000, 0x8000000000000000),
        (0x7FF8000000000001, 0x7FF8000000000002),
    ],
)
def test_contract_suite_bitwise_determinism_compares_float_payload_bits(
    first_bits,
    second_bits,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    payloads = iter((first_bits, second_bits))

    def bit_changing_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data["sma_2"].to_numpy(copy=False).view(np.uint64)[0] = next(payloads)
        return result

    with pytest.raises(ContractViolationError, match="deterministic"):
        verify_operator_contract(load_operator_manifest(payload), bit_changing_provider, request)


def test_contract_suite_accepts_timezone_aware_dates_in_bitwise_input_checks(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    context = replace(
        daily_context,
        frequency="1min",
        timestamp_semantics="bar_close",
        price_adjustment="raw",
    )
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, context).drop(columns=["volume"]),
        context=context,
    )

    report = verify_operator_contract(
        _load_contract_manifest(valid_manifest_payload),
        sma,
        request,
    )

    assert report.passed is True
    assert report.input_dtypes["date"] == "datetime64[ns, UTC]"


def test_contract_suite_bitwise_compares_undeclared_column_outputs_by_representation(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def representation_dependent_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        sign = 0.0 if "volume" in provider_request.input_panel else -0.0
        result.data.loc[result.data["sma_2"].notna(), "sma_2"] = sign
        return result

    with pytest.raises(ContractViolationError, match="depend on undeclared input columns"):
        verify_operator_contract(
            load_operator_manifest(payload),
            representation_dependent_provider,
            request,
        )


def test_contract_suite_applies_tolerance_to_undeclared_column_output_comparison(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["determinism"] = {
        "bitwise": False,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 0.0,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def tolerance_dependent_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        if "volume" in provider_request.input_panel:
            result.data.loc[result.data["sma_2"].notna(), "sma_2"] += 1e-7
        return result

    report = verify_operator_contract(
        load_operator_manifest(payload),
        tolerance_dependent_provider,
        request,
    )

    assert report.passed is True


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_contract_suite_accepts_finite_nullable_numeric_extension_outputs(
    dtype,
    monkeypatch,
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["outputs"]["fields"][0]["dtype"] = dtype
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"])
    panel["close"] *= 2
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    original_to_numpy = pd.Series.to_numpy

    def extension_as_object(series, *args, **kwargs):
        if series.name == "sma_2" and str(series.dtype) == dtype and not args and not kwargs:
            return np.asarray(series.tolist(), dtype=object)
        return original_to_numpy(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "to_numpy", extension_as_object)

    def nullable_numeric_provider(provider_request: OperatorRequest):
        result = sma(provider_request)
        result.data["sma_2"] = result.data["sma_2"].astype(dtype)
        return result

    report = verify_operator_contract(
        load_operator_manifest(payload),
        nullable_numeric_provider,
        request,
    )

    assert report.passed is True


def test_contract_suite_required_failure_probes_use_all_declared_optional_shapes(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = _artifact_wide_payload(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["volume", "quality"]
    payload["inputs"]["dtypes"].update({"volume": ["int64"], "quality": ["float64"]})
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    panel["quality"] = 1.0
    panel["helper"] = 2.0
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel,
        context=daily_context,
    )
    observed_shapes: set[tuple[str, ...]] = set()
    manifest = load_operator_manifest(payload)

    def shape_recording_provider(provider_request: OperatorRequest):
        if "close" not in provider_request.input_panel:
            assert "helper" not in provider_request.input_panel
            observed_shapes.add(tuple(column for column in ("volume", "quality") if column in provider_request.input_panel))
            raise MissingColumnError(
                "missing required input column close",
                operator_id=provider_request.operator_id,
                details={"column": "close"},
            )
        _reject_empty_input(manifest, provider_request)
        return sma(provider_request)

    report = _verify_operator_contract(
        manifest,
        shape_recording_provider,
        request,
        expected_distribution_version="1.0.0",
        expected_implementation_digest=IMPLEMENTATION_DIGEST,
    )

    assert report.passed is True
    assert observed_shapes == {(), ("volume",), ("quality",), ("volume", "quality")}


def test_contract_suite_rejects_required_failure_behavior_dependent_on_undeclared_columns(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def undeclared_dependent_failure_provider(provider_request: OperatorRequest):
        if "close" not in provider_request.input_panel:
            if "volume" not in provider_request.input_panel:
                raise KeyError("close")
            raise MissingColumnError(
                "missing required input column close",
                operator_id=provider_request.operator_id,
                details={"column": "close"},
            )
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="required input column close.*MissingColumnError"):
        _verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            undeclared_dependent_failure_provider,
            request,
            expected_distribution_version="1.0.0",
            expected_implementation_digest=IMPLEMENTATION_DIGEST,
        )


def test_contract_suite_checks_input_immutability_when_required_column_is_missing(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )

    def mutating_failure_provider(provider_request: OperatorRequest):
        if "close" not in provider_request.input_panel:
            provider_request.input_panel.loc[0, "code"] = "MUTATED"
            raise MissingColumnError(
                "missing required input column close",
                operator_id=provider_request.operator_id,
                details={"column": "close"},
            )
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="mutated input_panel"):
        _verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            mutating_failure_provider,
            request,
            expected_distribution_version="1.0.0",
            expected_implementation_digest=IMPLEMENTATION_DIGEST,
        )


def test_contract_suite_binds_quant_panel_key_dtypes_in_report_identity(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"])
    object_request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel.assign(code=panel["code"].astype(object)),
        context=daily_context,
    )
    string_request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=panel.assign(code=panel["code"].astype("string")),
        context=daily_context,
    )
    manifest = _load_contract_manifest(valid_manifest_payload)

    object_report = verify_operator_contract(manifest, sma, object_request)
    string_report = verify_operator_contract(manifest, sma, string_request)

    assert object_report.input_dtypes["code"] == "object"
    assert string_report.input_dtypes["code"] == "string"
    assert object_report.input_dtypes_digest != string_report.input_dtypes_digest


def test_contract_suite_applies_declared_determinism_tolerances(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "panel"
    payload["causality"] = "future_using"
    payload["inputs"]["requires_sorted"] = True
    payload["determinism"] = {
        "bitwise": False,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 0.0,
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context).drop(columns=["volume"]),
        context=daily_context,
    )
    calls = 0
    full_input_rows = len(request.input_panel)

    def tolerant_provider(provider_request: OperatorRequest):
        nonlocal calls
        result = sma(provider_request)
        if len(provider_request.input_panel) == full_input_rows:
            calls += 1
            result.data.loc[result.data.index[-1], "sma_2"] += calls * 1e-7
        return result

    report = verify_operator_contract(_load_contract_manifest(payload), tolerant_provider, request)

    assert report.passed is True


def test_fake_provider_matches_expected_values(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    result = sma(request)
    expected = pd.Series([None, None, 10.5, 19.0, 11.5, 19.5], name="sma_2", dtype="float64")
    pd.testing.assert_series_equal(result.data["sma_2"], expected)


def test_contract_suite_probes_a_structurally_valid_empty_quant_panel(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )
    manifest = _load_contract_manifest(valid_manifest_payload)
    empty_shapes: list[tuple[tuple[str, str], ...]] = []

    def empty_aware_provider(provider_request: OperatorRequest):
        if "close" not in provider_request.input_panel:
            raise MissingColumnError(
                "missing required input column close",
                operator_id=provider_request.operator_id,
                details={"column": "close"},
            )
        if provider_request.input_panel.empty:
            QuantPanelAdapter.validate_panel(provider_request.input_panel, provider_request.context)
            empty_shapes.append(tuple((column, str(provider_request.input_panel[column].dtype)) for column in provider_request.input_panel))
            raise InsufficientHistoryError(
                "empty input does not satisfy the declared history minimum",
                operator_id=provider_request.operator_id,
                details={"min_history": 2, "available_history": 0},
            )
        return sma(provider_request)

    report = _verify_operator_contract(
        manifest,
        empty_aware_provider,
        request,
        expected_distribution_version="1.0.0",
        expected_implementation_digest=IMPLEMENTATION_DIGEST,
    )

    assert report.passed is True
    assert "empty_input" in report.checks
    assert empty_shapes == [
        (
            ("date", "datetime64[ns]"),
            ("code", "string"),
            ("close", "float64"),
        )
    ]


@pytest.mark.parametrize(
    "behavior",
    ["index_error", "silent", "wrong_error", "missing_operator_id", "wrong_minimum"],
)
def test_contract_suite_rejects_noncompliant_empty_input_behavior(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
    behavior: str,
) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    def noncompliant_provider(provider_request: OperatorRequest):
        if "close" not in provider_request.input_panel:
            raise MissingColumnError(
                "missing required input column close",
                operator_id=provider_request.operator_id,
                details={"column": "close"},
            )
        if provider_request.input_panel.empty:
            if behavior == "index_error":
                raise IndexError("single positional indexer is out-of-bounds")
            if behavior == "wrong_error":
                raise InsufficientCrossSectionError(
                    "wrong insufficient-input category for time_series scope",
                    operator_id=provider_request.operator_id,
                )
            if behavior == "missing_operator_id":
                raise InsufficientHistoryError("missing operator identity")
            if behavior == "wrong_minimum":
                raise InsufficientHistoryError(
                    "history minimum does not match the manifest",
                    operator_id=provider_request.operator_id,
                    details={"min_history": 999},
                )
        return sma(provider_request)

    with pytest.raises(ContractViolationError, match="empty input"):
        _verify_operator_contract(
            _load_contract_manifest(valid_manifest_payload),
            noncompliant_provider,
            request,
            expected_distribution_version="1.0.0",
            expected_implementation_digest=IMPLEMENTATION_DIGEST,
        )
