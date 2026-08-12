from __future__ import annotations

import pandas as pd
import pytest

from oxq.operators.errors import ContractViolationError, InsufficientHistoryError
from oxq.operators.manifest import load_operator_manifest
from oxq.operators.panel import QuantPanelAdapter
from oxq.operators.testing import verify_operator_contract
from oxq.operators.types import OperatorRequest
from tests.operators.fake_provider import sma


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
        "unordered_input",
        "batch_consistency",
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
        verify_operator_contract(load_operator_manifest(valid_manifest_payload), extra_output_provider, request)


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
