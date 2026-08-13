from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from oxq.operators.manifest import load_operator_manifest
from oxq.operators.testing import _copy_request
from oxq.operators.types import (
    FittedOperatorState,
    OperatorAvailability,
    OperatorCausality,
    OperatorDiagnostics,
    OperatorLifecycle,
    OperatorProvenance,
    OperatorRequest,
    OperatorResult,
    OperatorScope,
)


def test_operator_enums_match_contract_values() -> None:
    assert {item.value for item in OperatorScope} == {
        "time_series",
        "cross_section",
        "panel",
        "research_only",
    }
    assert {item.value for item in OperatorLifecycle} == {
        "stateless",
        "fit_transform",
        "evaluation",
        "data_access",
        "visualization",
    }
    assert {item.value for item in OperatorCausality} == {
        "past_only",
        "label_dependent",
        "future_using",
    }
    assert {item.value for item in OperatorAvailability} == {
        "pre_open_t",
        "open_t",
        "intraday_t",
        "close_t",
        "after_close_t",
        "publication_time",
    }


def test_operator_request_snapshots_parameters_and_is_frozen(daily_context) -> None:
    parameters = {
        "period": 2,
        "columns": ["close"],
        "options": {"winsorize": [0.01, 0.99]},
        "groups": {"large", "small"},
    }
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters=parameters,
        input_panel=pd.DataFrame({"date": [], "code": []}),
        context=daily_context,
    )

    parameters["period"] = 99
    parameters["columns"].append("volume")
    parameters["options"]["winsorize"].append(1.0)
    parameters["groups"].add("micro")
    assert request.parameters == {
        "period": 2,
        "columns": ("close",),
        "options": {"winsorize": (0.01, 0.99)},
        "groups": frozenset({"large", "small"}),
    }
    with pytest.raises(TypeError):
        request.parameters["period"] = 3  # type: ignore[index]
    with pytest.raises(AttributeError):
        request.parameters["columns"].append("volume")
    with pytest.raises(TypeError):
        request.parameters["options"]["winsorize"] = (0.0, 1.0)
    with pytest.raises(AttributeError):
        request.parameters["groups"].add("micro")
    with pytest.raises(FrozenInstanceError):
        request.operator_id = "changed"  # type: ignore[misc]


def test_frozen_request_parameters_remain_compatible_with_manifest_validation_and_copy(
    daily_context,
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"].update(
        {
            "labels": {
                "type": "array",
                "default": ["fast", "slow"],
                "required": False,
                "unit": None,
                "affects_warmup": False,
                "affects_output_fields": False,
                "affects_causality": False,
                "affects_availability": False,
            },
            "options": {
                "type": "object",
                "default": {"adjust": True},
                "required": False,
                "unit": None,
                "affects_warmup": False,
                "affects_output_fields": False,
                "affects_causality": False,
                "affects_availability": False,
            },
        }
    )
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={"period": 3, "labels": ["fast"], "options": {"adjust": False}},
        input_panel=pd.DataFrame({"date": [], "code": []}),
        context=daily_context,
    )

    resolved = load_operator_manifest(payload).validate_parameters(request.parameters)
    copied = _copy_request(request)

    assert resolved == {"period": 3, "labels": ["fast"], "options": {"adjust": False}}
    assert copied.parameters == request.parameters


def test_diagnostics_reject_impossible_row_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        OperatorDiagnostics(input_rows=-1, output_rows=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        OperatorDiagnostics(input_rows=2, output_rows=1, dropped_rows=3)


@pytest.mark.parametrize("field", ["input_rows", "output_rows", "warmup_rows", "dropped_rows"])
@pytest.mark.parametrize("invalid", [True, 1.0])
def test_diagnostics_require_actual_integer_row_counts(field, invalid) -> None:
    counts = {"input_rows": 2, "output_rows": 1, "warmup_rows": 0, "dropped_rows": 0}
    counts[field] = invalid

    with pytest.raises(TypeError, match="integers"):
        OperatorDiagnostics(**counts)


def test_fitted_state_snapshots_training_metadata() -> None:
    learned = {"mean": [1.0, 2.0]}
    state = FittedOperatorState(
        operator_id="fake.preprocessing.standardize",
        operator_version="1.0.0",
        training_start="2025-01-01",
        training_end="2025-12-31",
        training_data_digest="sha256:" + "a" * 64,
        training_data_summary={"rows": 252},
        feature_order=("value",),
        parameters={"ddof": 1},
        learned_state=learned,
        random_seed=7,
        dependency_versions={"numpy": "2.4.2"},
        state_digest="sha256:" + "b" * 64,
    )

    learned["mean"].append(3.0)
    assert state.learned_state == {"mean": (1.0, 2.0)}
    with pytest.raises(TypeError):
        state.learned_state["mean"] = (0.0,)  # type: ignore[index]


def test_fitted_state_copies_numpy_arrays_and_makes_them_read_only() -> None:
    coefficients = np.array([1.0, 2.0])
    state = FittedOperatorState(
        operator_id="fake.preprocessing.standardize",
        operator_version="1.0.0",
        training_start="2025-01-01",
        training_end="2025-12-31",
        training_data_digest="sha256:" + "a" * 64,
        training_data_summary={"rows": 252},
        feature_order=("value",),
        parameters={"ddof": 1},
        learned_state={"coefficients": coefficients},
        random_seed=7,
        dependency_versions={"numpy": "2.4.2"},
        state_digest="sha256:" + "b" * 64,
    )

    frozen_coefficients = state.learned_state["coefficients"]
    coefficients[0] = 99.0

    assert isinstance(frozen_coefficients, np.ndarray)
    assert frozen_coefficients is not coefficients
    assert frozen_coefficients.tolist() == [1.0, 2.0]
    assert frozen_coefficients.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        frozen_coefficients[0] = 0.0
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        frozen_coefficients.setflags(write=True)


def test_fitted_state_rejects_object_dtype_arrays() -> None:
    with pytest.raises(TypeError, match="object-dtype arrays"):
        FittedOperatorState(
            operator_id="fake.models.pca",
            operator_version="1.0.0",
            training_start="2025-01-01",
            training_end="2025-12-31",
            training_data_digest="sha256:" + "a" * 64,
            training_data_summary={"rows": 252},
            feature_order=("value",),
            parameters={},
            learned_state={"labels": np.array([["mutable"]], dtype=object)},
            random_seed=7,
            dependency_versions={"numpy": "2.4.2"},
            state_digest="sha256:" + "b" * 64,
        )


def test_operator_result_requires_matching_provenance(daily_context) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={},
        input_panel=pd.DataFrame({"date": [], "code": []}),
        context=daily_context,
    )
    diagnostics = OperatorDiagnostics(input_rows=0, output_rows=0)
    provenance = OperatorProvenance(
        operator_id="another.operator",
        operator_version="1.0.0",
        implementation_digest="sha256:" + "a" * 64,
    )

    with pytest.raises(ValueError, match="operator_id"):
        OperatorResult.for_request(
            request,
            data=pd.DataFrame({"date": [], "code": []}),
            diagnostics=diagnostics,
            provenance=provenance,
        )
