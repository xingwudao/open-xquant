from __future__ import annotations

import copy
import json
import re
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from oxq.operators.errors import OperatorError
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


def _fitted_state(**overrides: Any) -> FittedOperatorState:
    values: dict[str, Any] = {
        "operator_id": "fake.preprocessing.standardize",
        "operator_version": "1.0.0",
        "training_start": "2025-01-01",
        "training_end": "2025-12-31",
        "training_data_digest": "sha256:" + "a" * 64,
        "training_data_summary": {"rows": 252},
        "feature_order": ("value",),
        "parameters": {"ddof": 1},
        "learned_state": {"mean": (1.0, 2.0)},
        "random_seed": 7,
        "dependency_versions": {"numpy": "2.4.2"},
        "state_digest": "sha256:" + "b" * 64,
    }
    values.update(overrides)
    return FittedOperatorState(**values)


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


def test_operator_error_snapshots_nested_structured_details() -> None:
    columns = ["close"]
    groups = {"large", "small"}
    thresholds = np.array([[0.1, 0.9]])
    nested = {
        "columns": columns,
        "groups": groups,
        "thresholds": thresholds,
    }
    error = OperatorError("invalid output", details={"output": nested})
    expected = {
        "code": "operator_error",
        "operator_id": None,
        "message": "invalid output",
        "details": {
            "output": {
                "columns": ["close"],
                "groups": ["large", "small"],
                "thresholds": [[0.1, 0.9]],
            }
        },
        "retryable": False,
    }

    columns.append("volume")
    groups.add("micro")
    thresholds[0, 0] = -1.0
    nested["status"] = "changed"

    assert error.to_dict() == expected
    assert json.loads(json.dumps(error.to_dict(), sort_keys=True)) == expected


def test_operator_error_recursively_freezes_exposed_details() -> None:
    error = OperatorError(
        "invalid output",
        details={
            "output": {
                "columns": ["close"],
                "groups": {"large", "small"},
                "thresholds": np.array([0.1, 0.9]),
            }
        },
    )
    expected = error.to_dict()
    output = error.details["output"]

    with pytest.raises(TypeError):
        output["status"] = "changed"
    with pytest.raises(AttributeError):
        output["columns"].append("volume")
    with pytest.raises(AttributeError):
        output["groups"].add("micro")
    with pytest.raises(ValueError):
        output["thresholds"][0] = -1.0
    with pytest.raises(ValueError):
        output["thresholds"].setflags(write=True)

    assert error.to_dict() == expected


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


def test_operator_request_rejects_non_string_nested_parameter_keys(daily_context) -> None:
    with pytest.raises(TypeError, match=r"parameters\['options'\].*mapping keys must be strings.*int.*1"):
        OperatorRequest(
            operator_id="fake.indicators.sma",
            parameters={"options": {1: "numeric", "1": "string"}},
            input_panel=pd.DataFrame({"date": [], "code": []}),
            context=daily_context,
        )


@pytest.mark.parametrize(
    "parameters",
    [[("period", 2)], ("period", 2), "period=2"],
    ids=["list", "tuple", "string"],
)
def test_operator_request_requires_parameters_mapping(daily_context, parameters: Any) -> None:
    with pytest.raises(TypeError, match="parameters must be a mapping"):
        OperatorRequest(
            operator_id="fake.indicators.sma",
            parameters=parameters,
            input_panel=pd.DataFrame({"date": [], "code": []}),
            context=daily_context,
        )


@pytest.mark.parametrize("context", [{"timezone": "Asia/Shanghai"}, SimpleNamespace(timezone="Asia/Shanghai")])
def test_operator_request_requires_concrete_operator_context(context: Any) -> None:
    with pytest.raises(TypeError, match="context must be an OperatorContext"):
        OperatorRequest(
            operator_id="fake.indicators.sma",
            parameters={"period": 2},
            input_panel=pd.DataFrame({"date": [], "code": []}),
            context=context,
        )


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


def test_diagnostics_snapshots_warning_sequences_as_tuples() -> None:
    warnings = ["insufficient history", "partial output"]

    diagnostics = OperatorDiagnostics(input_rows=2, output_rows=1, warnings=warnings)  # type: ignore[arg-type]
    warnings.append("late mutation")

    assert diagnostics.warnings == ("insufficient history", "partial output")


@pytest.mark.parametrize(
    "warnings",
    [
        "single warning",
        {"warning": "mapped warning"},
        ["valid warning", 1],
        (None,),
        (warning for warning in ["generated warning"]),
    ],
    ids=["string", "mapping", "non-string-item-list", "none-item-tuple", "generator"],
)
def test_diagnostics_require_a_sequence_of_string_warnings(warnings: Any) -> None:
    with pytest.raises(TypeError, match="warnings must be a sequence of strings"):
        OperatorDiagnostics(input_rows=1, output_rows=1, warnings=warnings)


@pytest.mark.parametrize(
    "field",
    ["timezone", "calendar", "frequency", "currency", "data_version", "source"],
)
def test_operator_context_identity_fields_require_strings(daily_context, field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        replace(daily_context, **{field: 1})


@pytest.mark.parametrize("timezone", ["UTC", "America/New_York", "Asia/Shanghai"])
def test_operator_context_accepts_supported_zoneinfo_timezones(daily_context, timezone: str) -> None:
    context = replace(daily_context, timezone=timezone)

    assert context.timezone == timezone


@pytest.mark.parametrize("timezone", ["Mars/Olympus", "/etc/passwd"])
def test_operator_context_rejects_unsupported_zoneinfo_timezones(daily_context, timezone: str) -> None:
    with pytest.raises(ValueError, match=rf"timezone.*{re.escape(timezone)}.*zoneinfo"):
        replace(daily_context, timezone=timezone)


def test_fitted_state_snapshots_training_metadata() -> None:
    learned = {"mean": [1.0, 2.0], "serialized": b"model"}
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
    assert state.learned_state == {"mean": (1.0, 2.0), "serialized": b"model"}
    with pytest.raises(TypeError):
        state.learned_state["mean"] = (0.0,)  # type: ignore[index]


@pytest.mark.parametrize("feature_order", [(1,), ("",), ("value", None)])
def test_fitted_state_requires_non_empty_string_feature_names(feature_order: tuple[Any, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="feature_order"):
        _fitted_state(feature_order=feature_order)


@pytest.mark.parametrize(
    "feature_order",
    ["value", b"value", {"value"}, {"value": 1}, (feature for feature in ["value"]), 1],
    ids=["string", "bytes", "set", "mapping", "generator", "integer"],
)
def test_fitted_state_requires_feature_order_sequence(feature_order: Any) -> None:
    with pytest.raises(TypeError, match="feature_order must be a non-string sequence"):
        _fitted_state(feature_order=feature_order)


def test_fitted_state_snapshots_feature_order_sequence() -> None:
    feature_order = ["open", "close"]

    state = _fitted_state(feature_order=feature_order)
    feature_order.append("volume")

    assert state.feature_order == ("open", "close")


@pytest.mark.parametrize("random_seed", [None, 0, -7])
def test_fitted_state_accepts_none_or_exact_integer_random_seed(random_seed: int | None) -> None:
    state = _fitted_state(random_seed=random_seed)

    assert state.random_seed == random_seed


@pytest.mark.parametrize("field", ["operator_id", "operator_version", "training_start", "training_end"])
@pytest.mark.parametrize(
    "invalid",
    [1, ["mutable"], {"truthy": "mapping"}],
    ids=["integer", "list", "mapping"],
)
def test_fitted_state_identity_fields_require_strings(field: str, invalid: Any) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be a string"):
        _fitted_state(**{field: invalid})


@pytest.mark.parametrize("field", ["operator_id", "operator_version", "training_start", "training_end"])
def test_fitted_state_identity_fields_reject_empty_strings(field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        _fitted_state(**{field: ""})


@pytest.mark.parametrize(
    ("training_start", "training_end"),
    [
        ("2025-01-01", "2025-12-31"),
        ("2025-01-01T10:00:00+08:00", "2025-01-01T03:00:00Z"),
        ("2025-01-01T10:00:00.123456+08:00", "2025-01-01T10:00:00.123456+08:00"),
    ],
    ids=["dates", "timezone-normalized-datetimes", "equal-datetimes"],
)
def test_fitted_state_accepts_portable_iso_8601_training_boundaries(
    training_start: str,
    training_end: str,
) -> None:
    state = _fitted_state(training_start=training_start, training_end=training_end)

    assert state.training_start == training_start
    assert state.training_end == training_end


@pytest.mark.parametrize(
    ("training_start", "training_end"),
    [
        ("2025-02-29", "2025-12-31"),
        ("2025-01-01 00:00:00Z", "2025-12-31T00:00:00Z"),
        ("2025-01-01T00:00:00", "2025-12-31T00:00:00"),
        ("2025-01-01T00:00:00+0800", "2025-12-31T00:00:00+0800"),
        ("２０２５-01-01", "2025-12-31"),
        ("2025-01-01", "2025-12-31T00:00:00Z"),
    ],
    ids=["invalid-date", "space-separator", "naive-datetime", "compact-offset", "non-ascii-digits", "mixed-kinds"],
)
def test_fitted_state_rejects_non_portable_training_boundaries(
    training_start: str,
    training_end: str,
) -> None:
    with pytest.raises(ValueError, match="training boundaries must be ISO 8601"):
        _fitted_state(training_start=training_start, training_end=training_end)


@pytest.mark.parametrize(
    ("training_start", "training_end"),
    [
        ("2025-12-31", "2025-01-01"),
        ("2025-01-01T10:00:00+08:00", "2025-01-01T01:00:00Z"),
    ],
    ids=["dates", "timezone-aware-datetimes"],
)
def test_fitted_state_rejects_reversed_training_boundaries(
    training_start: str,
    training_end: str,
) -> None:
    with pytest.raises(ValueError, match="training_start must not be after training_end"):
        _fitted_state(training_start=training_start, training_end=training_end)


@pytest.mark.parametrize(
    "random_seed",
    [True, 7.0, "7", [], {}],
    ids=["bool", "float", "string", "list", "mapping"],
)
def test_fitted_state_rejects_non_integer_random_seed(random_seed: Any) -> None:
    with pytest.raises(TypeError, match="random_seed must be an integer or None"):
        _fitted_state(random_seed=random_seed)


def test_fitted_state_permanently_freezes_numpy_arrays_in_all_state_mappings() -> None:
    summary_array = np.array([252, 253])
    parameter_array = np.array([0.25, 0.75])
    learned_array = np.array([1.0, 2.0])
    state = _fitted_state(
        training_data_summary={"rows_by_year": summary_array},
        parameters={"quantiles": parameter_array},
        learned_state={"coefficients": learned_array},
    )

    frozen_arrays = (
        state.training_data_summary["rows_by_year"],
        state.parameters["quantiles"],
        state.learned_state["coefficients"],
    )
    summary_array[0] = 999
    parameter_array[0] = 0.0
    learned_array[0] = 99.0

    assert [array.tolist() for array in frozen_arrays] == [[252, 253], [0.25, 0.75], [1.0, 2.0]]
    for frozen_array in frozen_arrays:
        assert isinstance(frozen_array, np.ndarray)
        assert frozen_array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            frozen_array[0] = 0.0
        with pytest.raises(ValueError, match="WRITEABLE flag"):
            frozen_array.setflags(write=True)


@pytest.mark.parametrize("field", ["training_data_summary", "parameters", "learned_state"])
def test_fitted_state_rejects_object_dtype_arrays_in_all_state_mappings(field: str) -> None:
    with pytest.raises(TypeError, match="object-dtype arrays"):
        _fitted_state(**{field: {"labels": np.array([["mutable"]], dtype=object)}})


@pytest.mark.parametrize("field", ["training_data_summary", "parameters", "learned_state"])
def test_fitted_state_rejects_non_string_nested_keys_in_all_state_mappings(field: str) -> None:
    with pytest.raises(TypeError, match=rf"{field}\['nested'\].*mapping keys must be strings.*int.*1"):
        _fitted_state(**{field: {"nested": {1: "numeric", "1": "string"}}})


@pytest.mark.parametrize("field", ["training_data_summary", "parameters", "learned_state"])
@pytest.mark.parametrize(
    "value",
    [[("key", "value")], ("key", "value"), "key=value"],
    ids=["list", "tuple", "string"],
)
def test_fitted_state_requires_state_mappings(field: str, value: Any) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be a mapping"):
        _fitted_state(**{field: value})


@pytest.mark.parametrize(
    "mutable_leaf",
    [bytearray(b"model"), pd.DataFrame({"weight": [1.0]})],
    ids=["bytearray", "dataframe"],
)
@pytest.mark.parametrize("field", ["training_data_summary", "parameters", "learned_state"])
def test_fitted_state_rejects_unknown_mutable_leaves_in_all_state_mappings(field: str, mutable_leaf: Any) -> None:
    with pytest.raises(
        TypeError,
        match=rf"{field}\['model'\].*unsupported leaf type (bytearray|DataFrame)",
    ):
        _fitted_state(**{field: {"model": mutable_leaf}})


def test_fitted_state_snapshots_dependency_versions_as_read_only_string_mapping() -> None:
    dependency_versions = {"numpy": "2.4.2"}
    state = _fitted_state(dependency_versions=dependency_versions)

    dependency_versions["numpy"] = "3.0.0"

    assert state.dependency_versions == {"numpy": "2.4.2"}
    with pytest.raises(TypeError):
        state.dependency_versions["numpy"] = "3.0.0"  # type: ignore[index]


@pytest.mark.parametrize(
    "dependency_versions",
    [{1: "2.4.2"}, {"numpy": 2.4}],
    ids=["non-string-key", "non-string-value"],
)
def test_fitted_state_requires_string_dependency_names_and_versions(dependency_versions: dict[Any, Any]) -> None:
    with pytest.raises(TypeError, match="dependency_versions.*strings"):
        _fitted_state(dependency_versions=dependency_versions)


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


class _DiagnosticsLike:
    output_rows = 1


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("diagnostics", _DiagnosticsLike(), "diagnostics must be an OperatorDiagnostics"),
        ("provenance", object(), "provenance must be an OperatorProvenance"),
        ("metadata", [("source", "test")], "metadata must be a mapping"),
    ],
    ids=["diagnostics", "provenance", "metadata"],
)
def test_operator_result_requires_concrete_contract_types(field: str, invalid: Any, message: str) -> None:
    values: dict[str, Any] = {
        "data": pd.DataFrame({"value": [1.0]}),
        "diagnostics": OperatorDiagnostics(input_rows=1, output_rows=1),
        "provenance": OperatorProvenance(
            operator_id="fake.indicators.sma",
            operator_version="1.0.0",
            implementation_digest="sha256:" + "a" * 64,
        ),
        "metadata": {},
    }
    values[field] = invalid

    with pytest.raises(TypeError, match=message):
        OperatorResult(**values)


def test_operator_result_factory_rejects_non_mapping_metadata(daily_context) -> None:
    request = OperatorRequest(
        operator_id="fake.indicators.sma",
        parameters={},
        input_panel=pd.DataFrame({"value": [1.0]}),
        context=daily_context,
    )

    with pytest.raises(TypeError, match="metadata must be a mapping"):
        OperatorResult.for_request(
            request,
            data=pd.DataFrame({"value": [1.0]}),
            diagnostics=OperatorDiagnostics(input_rows=1, output_rows=1),
            provenance=OperatorProvenance(
                operator_id=request.operator_id,
                operator_version="1.0.0",
                implementation_digest="sha256:" + "a" * 64,
            ),
            metadata=[],  # type: ignore[arg-type]
        )


def test_operator_result_metadata_allows_arbitrary_values() -> None:
    buffer = bytearray(b"metadata")
    frame = pd.DataFrame({"value": [1.0]})
    result = OperatorResult(
        data=pd.DataFrame({"value": [1.0]}),
        diagnostics=OperatorDiagnostics(input_rows=1, output_rows=1),
        provenance=OperatorProvenance(
            operator_id="fake.indicators.sma",
            operator_version="1.0.0",
            implementation_digest="sha256:" + "a" * 64,
        ),
        metadata={"buffer": buffer, "frame": frame},
    )

    buffer[0] = ord("M")
    frame.iloc[0, 0] = 2.0

    assert result.metadata["buffer"] == bytearray(b"metadata")
    pd.testing.assert_frame_equal(result.metadata["frame"], pd.DataFrame({"value": [1.0]}))
