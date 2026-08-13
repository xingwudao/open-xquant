from __future__ import annotations

import copy
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from oxq.operators.errors import (
    CausalityViolationError,
    ContractViolationError,
    InsufficientCrossSectionError,
    InsufficientHistoryError,
    InvalidParameterError,
)
from oxq.operators.manifest import load_operator_manifest
from oxq.operators.panel import QuantPanelAdapter
from oxq.operators.testing import verify_operator_contract as _verify_operator_contract
from oxq.operators.types import OperatorRequest
from tests.operators.fake_provider import IMPLEMENTATION_DIGEST, sma


def verify_operator_contract(
    manifest,
    operator,
    request,
    *,
    expected_implementation_digest=IMPLEMENTATION_DIGEST,
):
    return _verify_operator_contract(
        manifest,
        operator,
        request,
        expected_implementation_digest=expected_implementation_digest,
    )


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

    report = verify_operator_contract(load_operator_manifest(valid_manifest_payload), sma, request)

    assert report.operator_id == request.operator_id
    assert report.passed is True
    assert report.checks == (
        "request",
        "input_immutability",
        "output_contract",
        "provenance",
        "determinism",
        "causality",
        "unordered_input",
        "batch_consistency",
    )


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
            load_operator_manifest(valid_manifest_payload),
            sma,
            request,
            expected_implementation_digest="sha256:" + "d" * 64,
        )


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
        verify_operator_contract(load_operator_manifest(valid_manifest_payload), mutating_provider, request)


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
            load_operator_manifest(valid_manifest_payload),
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
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
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
        verify_operator_contract(load_operator_manifest(valid_manifest_payload), unstable_provider, request)


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
        verify_operator_contract(load_operator_manifest(valid_manifest_payload), unstable_provider, request)


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
        verify_operator_contract(load_operator_manifest(valid_manifest_payload), sma, request)


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
    manifest = load_operator_manifest(valid_manifest_payload)
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
        verify_operator_contract(load_operator_manifest(payload), sma, request)


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
        verify_operator_contract(load_operator_manifest(payload), sma, request)


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
        verify_operator_contract(load_operator_manifest(payload), sma, request)


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
        verify_operator_contract(load_operator_manifest(payload), sma, request)


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

    report = verify_operator_contract(load_operator_manifest(payload), sorted_only_provider, request)

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
        verify_operator_contract(load_operator_manifest(payload), unexpected_provider, request)

    assert exc_info.value.to_dict()["details"] == {
        "required_rows": 2,
        "available_rows": 1,
    }


def test_contract_suite_rejects_unordered_probe_that_does_not_change_key_order(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
    monkeypatch,
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

    def unchanged_sample(frame, *, frac, random_state):
        assert frac == 1
        assert random_state == 731
        return frame.copy()

    monkeypatch.setattr(pd.DataFrame, "sample", unchanged_sample)

    with pytest.raises(ContractViolationError, match="did not change key order") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), sma, request)

    assert exc_info.value.to_dict()["details"] == {"available_rows": 2}


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
        verify_operator_contract(load_operator_manifest(valid_manifest_payload), sma, request)


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
        verify_operator_contract(load_operator_manifest(payload), out_of_bounds_provider, request)


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
        verify_operator_contract(load_operator_manifest(payload), sma, request)


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
        verify_operator_contract(load_operator_manifest(valid_manifest_payload), late_nan_provider, request)


def test_contract_suite_rejects_negative_resolved_warmup_without_nans(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["warmup"]["offset"] = -2
    payload["outputs"]["nan_policy"] = "none"
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 1},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="non-negative integer"):
        verify_operator_contract(load_operator_manifest(payload), sma, request)


@pytest.mark.parametrize(
    ("warmup_rows", "dropped_rows", "message"),
    [(999, 0, "warmup_rows"), (2, 1, "dropped_rows")],
)
def test_contract_suite_validates_all_diagnostic_row_counts(
    daily_context,
    daily_symbol_frames,
    valid_manifest_payload,
    warmup_rows,
    dropped_rows,
    message,
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
                warmup_rows=warmup_rows,
                dropped_rows=dropped_rows,
            ),
            provenance=result.provenance,
        )

    with pytest.raises(ContractViolationError, match=message):
        verify_operator_contract(
            load_operator_manifest(valid_manifest_payload),
            invalid_diagnostics_provider,
            request,
        )


def test_contract_suite_rejects_duplicate_resolved_output_names(
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
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 2, "alias": 2},
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
        context=daily_context,
    )

    with pytest.raises(ContractViolationError, match="duplicate resolved output fields"):
        verify_operator_contract(load_operator_manifest(payload), sma, request)


def test_contract_suite_rejects_dynamic_reserved_output_names(
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

    with pytest.raises(ContractViolationError, match="reserved QuantPanel key"):
        verify_operator_contract(load_operator_manifest(payload), sma, request)


def test_contract_suite_rejects_required_parameter_resolving_to_empty_output_name(
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

    with pytest.raises(ContractViolationError, match="non-empty") as exc_info:
        verify_operator_contract(load_operator_manifest(payload), empty_output_name_provider, request)

    assert exc_info.value.to_dict()["details"] == {"fields": [""]}


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
            load_operator_manifest(valid_manifest_payload),
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
        input_panel=QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context),
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
            load_operator_manifest(valid_manifest_payload),
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
            load_operator_manifest(valid_manifest_payload),
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
        load_operator_manifest(valid_manifest_payload),
        array_metadata_provider,
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
        load_operator_manifest(valid_manifest_payload),
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
        load_operator_manifest(valid_manifest_payload),
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
        load_operator_manifest(valid_manifest_payload),
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
        verify_operator_contract(load_operator_manifest(payload), object_mutating_provider, request)


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
            load_operator_manifest(valid_manifest_payload),
            future_using_provider,
            request,
        )


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
        verify_operator_contract(load_operator_manifest(payload), future_using_panel_provider, request)


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
            load_operator_manifest(payload),
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
        verify_operator_contract(load_operator_manifest(payload), sma, request)

    assert exc_info.value.to_dict()["details"] == {
        "min_history": 3,
        "available_dates": 3,
    }


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
        verify_operator_contract(load_operator_manifest(payload), identity_provider, request)

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
        verify_operator_contract(load_operator_manifest(payload), time_mixing_provider, request)


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
            load_operator_manifest(payload),
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
            load_operator_manifest(valid_manifest_payload),
            sma,
            request,
        )

    assert exc_info.value.to_dict()["details"] == {
        "required_assets": 2,
        "available_assets": 1,
    }


def test_contract_suite_uses_exact_determinism_by_default(
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

    def subtly_unstable_provider(provider_request: OperatorRequest):
        nonlocal calls
        result = sma(provider_request)
        calls += 1
        result.data.loc[result.data.index[-1], "sma_2"] += calls * 1e-7
        return result

    with pytest.raises(ContractViolationError, match="deterministic"):
        verify_operator_contract(
            load_operator_manifest(valid_manifest_payload),
            subtly_unstable_provider,
            request,
        )


def test_contract_suite_applies_declared_determinism_tolerances(
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
    calls = 0

    def tolerant_provider(provider_request: OperatorRequest):
        nonlocal calls
        result = sma(provider_request)
        calls += 1
        result.data.loc[result.data.index[-1], "sma_2"] += calls * 1e-7
        return result

    report = verify_operator_contract(load_operator_manifest(payload), tolerant_provider, request)

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
