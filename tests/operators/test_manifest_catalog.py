from __future__ import annotations

import copy
import json
from datetime import date
from types import MappingProxyType

import pytest
import yaml  # type: ignore[import-untyped]

from oxq.operators.catalog import load_operator_catalog
from oxq.operators.errors import InvalidManifestError, InvalidParameterError
from oxq.operators.manifest import load_operator_manifest


def test_manifest_is_closed_and_does_not_import_provider(valid_manifest_payload, monkeypatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr("importlib.import_module", lambda name: imported.append(name))
    payload = copy.deepcopy(valid_manifest_payload)
    payload["surprise"] = True

    with pytest.raises(InvalidManifestError, match="surprise"):
        load_operator_manifest(payload)
    assert imported == []


def test_manifest_validates_parameters_and_rejects_unknowns(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)

    assert manifest.validate_parameters({}) == {"period": 2}
    assert manifest.validate_parameters({"period": 5}) == {"period": 5}
    with pytest.raises(InvalidParameterError, match="unknown"):
        manifest.validate_parameters({"unexpected": 1})
    with pytest.raises(InvalidParameterError, match="minimum"):
        manifest.validate_parameters({"period": 0})


def test_manifest_rejects_non_string_supplied_parameter_keys(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)

    with pytest.raises(InvalidParameterError, match="parameter names must be strings") as exc_info:
        manifest.validate_parameters({1: 5})

    assert exc_info.value.code == "invalid_parameter"
    assert exc_info.value.operator_id == "fake.indicators.sma"
    assert exc_info.value.to_dict()["details"] == {"parameters": ["1"]}


@pytest.mark.parametrize("supplied", [None, [], [("period", 5)], "period"])
def test_manifest_rejects_non_mapping_supplied_parameters(valid_manifest_payload, supplied) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)

    with pytest.raises(TypeError, match="supplied parameters must be a mapping"):
        manifest.validate_parameters(supplied)


def test_manifest_resolves_output_name_from_arbitrary_required_string_at_parameter_boundary(
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
    payload["outputs"]["fields"] = [{"name_template": "{field}", "dtype": "float64"}]
    manifest = load_operator_manifest(payload)

    assert manifest.validate_parameters({"field": "custom_signal"}) == {
        "period": 2,
        "field": "custom_signal",
    }


@pytest.mark.parametrize("resolved_name", ["", "date", "code"], ids=["empty", "date", "code"])
def test_manifest_rejects_invalid_output_name_resolved_from_required_parameter(
    valid_manifest_payload,
    resolved_name,
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
    payload["outputs"]["fields"] = [{"name_template": "{field}", "dtype": "float64"}]
    manifest = load_operator_manifest(payload)

    with pytest.raises(InvalidParameterError) as exc_info:
        manifest.validate_parameters({"field": resolved_name})

    assert exc_info.value.code == "invalid_parameter"
    assert exc_info.value.operator_id == "fake.indicators.sma"
    assert "output field" in str(exc_info.value)


def test_manifest_rejects_duplicate_output_name_resolved_from_required_parameter(
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
    payload["outputs"]["fields"].append({"name_template": "{field}", "dtype": "float64"})
    payload["outputs"]["multiple"] = True
    manifest = load_operator_manifest(payload)

    with pytest.raises(InvalidParameterError, match="duplicate output field name") as exc_info:
        manifest.validate_parameters({"field": "sma_2"})

    assert exc_info.value.code == "invalid_parameter"
    assert exc_info.value.operator_id == "fake.indicators.sma"


def test_manifest_rejects_dynamic_output_format_incompatible_with_required_parameter_type(
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    for name in ("value", "spec"):
        payload["parameters"][name] = {
            "type": "string",
            "required": True,
            "unit": None,
            "affects_warmup": False,
            "affects_output_fields": True,
            "affects_causality": False,
            "affects_availability": False,
        }
    payload["parameters"]["spec"]["enum"] = ["d"]
    payload["outputs"]["fields"] = [{"name_template": "{value:{spec}}", "dtype": "float64"}]

    with pytest.raises(InvalidManifestError, match="format.*declared parameter domain"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    ("parameter_type", "supplied"),
    [
        ("object", {"nested": [float("inf")]}),
        ("object", {1: "invalid"}),
        ("object", {"nested": (1, 2)}),
        ("object", {"nested": date(2026, 8, 13)}),
        ("array", (1, 2)),
    ],
    ids=["nonfinite", "non-string-key", "nested-tuple", "custom-leaf", "tuple-array"],
)
def test_manifest_rejects_supplied_composite_parameters_outside_finite_json_tree(
    valid_manifest_payload,
    parameter_type,
    supplied,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["options"] = {
        "type": parameter_type,
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    manifest = load_operator_manifest(payload)

    with pytest.raises(InvalidParameterError, match="JSON|finite|string keys|array"):
        manifest.validate_parameters({"options": supplied})


def test_manifest_rejects_recursive_supplied_composite_parameters(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["options"] = {
        "type": "object",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    recursive: list[object] = []
    recursive.append(recursive)
    manifest = load_operator_manifest(payload)

    with pytest.raises(InvalidParameterError, match="finite JSON tree"):
        manifest.validate_parameters({"options": {"nested": recursive}})


def test_manifest_accepts_supplied_composite_parameters_as_plain_json_tree(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["options"] = {
        "type": "object",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    manifest = load_operator_manifest(payload)

    supplied = {"nested": [1, 2.5, True, None, "value"]}
    assert manifest.validate_parameters({"options": supplied}) == {"period": 2, "options": supplied}


def test_manifest_preserves_json_collection_types_when_resolving_defaults(valid_manifest_payload) -> None:
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

    resolved = load_operator_manifest(payload).validate_parameters({})

    assert resolved["labels"] == ["fast", "slow"]
    assert isinstance(resolved["labels"], list)
    assert resolved["options"] == {"adjust": True}
    assert isinstance(resolved["options"], dict)


def test_manifest_rejects_non_string_keys_in_nested_mappings(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["options"] = {
        "type": "object",
        "default": {"nested": [{1: "invalid"}]},
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }

    with pytest.raises(InvalidManifestError, match="mapping keys must be strings"):
        load_operator_manifest(payload)


def test_manifest_rejects_non_json_leaves_loaded_from_yaml(valid_manifest_payload, tmp_path) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["options"] = {
        "type": "object",
        "default": {"nested": [date(2026, 8, 13)]},
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    source = tmp_path / "operator.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InvalidManifestError, match=r"JSON.*parameters\.options\.default\.nested\[0\]"):
        load_operator_manifest(source)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_manifest_rejects_nested_duplicate_mapping_keys(valid_manifest_payload, tmp_path, suffix) -> None:
    source = tmp_path / f"operator{suffix}"
    if suffix == ".json":
        raw = json.dumps(valid_manifest_payload, separators=(",", ":"))
        raw = raw.replace('"min_assets":1', '"min_assets":1,"min_assets":2')
    else:
        raw = yaml.safe_dump(valid_manifest_payload, sort_keys=False)
        raw = raw.replace("  min_assets: 1\n", "  min_assets: 1\n  min_assets: 2\n")
    source.write_text(raw, encoding="utf-8")

    with pytest.raises(InvalidManifestError, match=r"manifest\.inputs.*duplicate.*min_assets") as exc_info:
        load_operator_manifest(source)

    assert exc_info.value.to_dict()["code"] == "invalid_manifest"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_manifest_accepts_normal_file_mapping_keys(valid_manifest_payload, tmp_path, suffix) -> None:
    source = tmp_path / f"operator{suffix}"
    if suffix == ".json":
        source.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")
    else:
        source.write_text(yaml.safe_dump(valid_manifest_payload, sort_keys=False), encoding="utf-8")

    assert load_operator_manifest(source).operator_id == valid_manifest_payload["operator_id"]


def test_manifest_preserves_safe_yaml_merge_keys(valid_manifest_payload, tmp_path) -> None:
    raw = yaml.safe_dump(valid_manifest_payload, sort_keys=False)
    raw = raw.replace("  period:\n", "  period: &period\n", 1)
    raw = raw.replace("outputs:\n", "  alias:\n    <<: *period\noutputs:\n", 1)
    source = tmp_path / "operator.yaml"
    source.write_text(raw, encoding="utf-8")

    manifest = load_operator_manifest(source)

    assert manifest.raw["parameters"]["alias"] == manifest.raw["parameters"]["period"]


@pytest.mark.parametrize("version", ["1.0.0-01", "1.0.0-alpha..1", "1.0.0-"])
def test_manifest_rejects_invalid_semantic_versions(valid_manifest_payload, version) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["operator_version"] = version

    with pytest.raises(InvalidManifestError, match="semantic versioning"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0-alpha.1", "1.0.0-1a", "1.0.0+001"])
def test_manifest_accepts_valid_semantic_versions(valid_manifest_payload, version) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["operator_version"] = version

    assert load_operator_manifest(payload).operator_version == version


def test_manifest_metadata_snapshot_is_deeply_read_only(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)

    with pytest.raises(TypeError):
        manifest.raw["inputs"]["min_assets"] = 99  # type: ignore[index]
    with pytest.raises(AttributeError):
        manifest.raw["inputs"]["required_columns"].append("volume")


def test_manifest_can_reload_its_deeply_read_only_raw_snapshot(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)

    reloaded = load_operator_manifest(manifest.raw)

    assert reloaded.operator_id == manifest.operator_id
    assert reloaded.digest == manifest.digest


def test_fit_transform_manifest_requires_serializable_state(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["lifecycle"] = "fit_transform"

    with pytest.raises(InvalidManifestError, match="fitted_state"):
        load_operator_manifest(payload)


def test_manifest_rejects_invalid_default_and_warmup_reference(valid_manifest_payload) -> None:
    invalid_default = copy.deepcopy(valid_manifest_payload)
    invalid_default["parameters"]["period"]["default"] = 0
    with pytest.raises(InvalidManifestError, match="default"):
        load_operator_manifest(invalid_default)

    invalid_warmup = copy.deepcopy(valid_manifest_payload)
    invalid_warmup["outputs"]["warmup"]["parameter"] = "missing"
    with pytest.raises(InvalidManifestError, match="warmup"):
        load_operator_manifest(invalid_warmup)


def test_manifest_rejects_integer_bounds_with_no_integer_value(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    declaration = payload["parameters"]["period"]
    declaration["minimum"] = 0.1
    declaration["maximum"] = 0.9
    declaration.pop("default")
    declaration["required"] = True

    with pytest.raises(InvalidManifestError, match="integer.*domain.*empty"):
        load_operator_manifest(payload)


def test_manifest_rejects_negative_warmup_resolved_from_parameter_default(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["warmup"]["offset"] = -3

    with pytest.raises(InvalidManifestError, match="warmup.*default.*non-negative"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    ("domain_kind", "offset"),
    [
        ("required-minimum", -2),
        ("nondefault-minimum", -1),
        ("enum-member", -2),
        ("unbounded-domain", -1),
    ],
)
def test_manifest_rejects_parameter_warmup_domains_that_can_resolve_negative(
    valid_manifest_payload,
    domain_kind,
    offset,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    declaration = payload["parameters"]["period"]
    if domain_kind == "required-minimum":
        declaration["required"] = True
        declaration.pop("default")
    elif domain_kind == "nondefault-minimum":
        declaration["minimum"] = 0
    elif domain_kind == "enum-member":
        declaration["enum"] = [1, 2]
    else:
        declaration.pop("minimum")
    payload["outputs"]["warmup"]["offset"] = offset

    with pytest.raises(InvalidManifestError, match="warmup.*non-negative"):
        load_operator_manifest(payload)


def test_manifest_accepts_parameter_warmup_with_non_negative_integer_domain(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["period"]["minimum"] = 0.5
    payload["outputs"]["warmup"]["offset"] = -1

    manifest = load_operator_manifest(payload)

    assert manifest.validate_parameters({"period": 1}) == {"period": 1}


@pytest.mark.parametrize("rows", [1.0, True], ids=["integral-float", "boolean"])
def test_manifest_rejects_non_int_fixed_warmup_rows(valid_manifest_payload, rows) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["warmup"] = {"kind": "fixed", "rows": rows}

    with pytest.raises(InvalidManifestError, match="warmup"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("offset", [0.0, False], ids=["integral-float", "boolean"])
def test_manifest_rejects_non_int_parameter_warmup_offset(valid_manifest_payload, offset) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["warmup"]["offset"] = offset

    with pytest.raises(InvalidManifestError, match="warmup"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("min_history", [0, 2])
def test_manifest_requires_exactly_one_history_row_for_cross_section_scope(
    valid_manifest_payload,
    min_history,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "cross_section"
    payload["inputs"]["min_history"] = min_history

    with pytest.raises(InvalidManifestError, match="cross_section.*min_history=1"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    "distribution",
    [
        "Fake-Quant-Operators",
        "fake_quant_operators",
        "fake.quant.operators",
        "fake--quant-operators",
        "fake-quant-operators-",
    ],
    ids=["uppercase", "underscore", "dot", "repeated-separator", "trailing-separator"],
)
def test_manifest_rejects_noncanonical_python_distribution_names(
    valid_manifest_payload,
    distribution,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["distribution"] = distribution

    with pytest.raises(InvalidManifestError, match="distribution"):
        load_operator_manifest(payload)


def test_manifest_rejects_bounds_on_nonnumeric_parameters(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["label"] = {
        "type": "string",
        "required": False,
        "minimum": 1,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }

    with pytest.raises(InvalidManifestError, match="numeric"):
        load_operator_manifest(payload)


def test_manifest_rejects_inverted_parameter_bounds(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["period"].update({"minimum": 5, "maximum": 2})

    with pytest.raises(InvalidManifestError, match="minimum must not exceed maximum"):
        load_operator_manifest(payload)


def test_manifest_rejects_inverted_output_bounds(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0].update({"minimum": 2, "maximum": 1})

    with pytest.raises(InvalidManifestError, match="output field minimum must not exceed maximum"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("dtype", ["string", "boolean", "object", "datetime64[ns]", "complex128"])
def test_manifest_rejects_output_bounds_for_non_ordered_numeric_dtypes(valid_manifest_payload, dtype) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0].update({"dtype": dtype, "minimum": 0})

    with pytest.raises(InvalidManifestError, match="output field bounds require a numeric dtype"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("dtype", ["int64", "float64", "Int64", "Float64"])
def test_manifest_accepts_output_bounds_for_numeric_dtypes(valid_manifest_payload, dtype) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0].update({"dtype": dtype, "minimum": 0, "maximum": 1})

    assert load_operator_manifest(payload).raw["outputs"]["fields"][0]["dtype"] == dtype


@pytest.mark.parametrize(
    ("multiple", "fields"),
    [
        (True, [{"name_template": "sma_{period}", "dtype": "float64"}]),
        (
            False,
            [
                {"name_template": "sma_{period}", "dtype": "float64"},
                {"name_template": "sma_signal", "dtype": "boolean"},
            ],
        ),
    ],
    ids=["single-field-declared-multiple", "multiple-fields-declared-single"],
)
def test_manifest_requires_outputs_multiple_to_match_field_count(valid_manifest_payload, multiple, fields) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["multiple"] = multiple
    payload["outputs"]["fields"] = fields

    with pytest.raises(InvalidManifestError, match="outputs multiple.*fields"):
        load_operator_manifest(payload)


def test_manifest_rejects_enum_members_with_the_wrong_parameter_type(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["period"]["enum"] = ["fast", "slow"]

    with pytest.raises(InvalidManifestError, match="enum member"):
        load_operator_manifest(payload)


def test_manifest_rejects_nonfinite_numeric_defaults_and_enum_members(valid_manifest_payload) -> None:
    invalid_default = copy.deepcopy(valid_manifest_payload)
    invalid_default["parameters"]["period"]["type"] = "number"
    invalid_default["parameters"]["period"]["default"] = float("nan")
    with pytest.raises(InvalidManifestError, match="finite"):
        load_operator_manifest(invalid_default)

    invalid_enum = copy.deepcopy(valid_manifest_payload)
    invalid_enum["parameters"]["period"]["type"] = "number"
    invalid_enum["parameters"]["period"]["enum"] = [1.0, float("inf")]
    with pytest.raises(InvalidManifestError, match="finite"):
        load_operator_manifest(invalid_enum)


def test_manifest_accepts_arbitrary_size_integer_default(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    huge_integer = 10**1000
    payload["parameters"]["period"]["default"] = huge_integer
    payload["outputs"]["fields"][0]["name_template"] = "sma"

    manifest = load_operator_manifest(payload)

    assert manifest.validate_parameters({}) == {"period": huge_integer}


def test_manifest_accepts_arbitrary_size_integer_enum_member(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    huge_integer = 10**1000
    payload["parameters"]["period"]["enum"] = [2, huge_integer]

    manifest = load_operator_manifest(payload)

    assert huge_integer in manifest.raw["parameters"]["period"]["enum"]


def test_manifest_accepts_arbitrary_size_integer_request(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0]["name_template"] = "sma"
    manifest = load_operator_manifest(payload)
    huge_integer = 10**1000

    assert manifest.validate_parameters({"period": huge_integer}) == {"period": huge_integer}


def test_manifest_rejects_overlapping_input_columns(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["close"]

    with pytest.raises(InvalidManifestError, match="required_columns"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("reserved_name", ["date", "code"])
def test_manifest_rejects_quant_panel_keys_as_optional_inputs(valid_manifest_payload, reserved_name) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = [reserved_name]
    payload["inputs"]["dtypes"][reserved_name] = ["object"]

    with pytest.raises(InvalidManifestError, match="optional_columns.*reserved.*date.*code"):
        load_operator_manifest(payload)


def test_manifest_rejects_dtype_declarations_for_undeclared_columns(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["dtypes"]["volume"] = ["int64"]

    with pytest.raises(InvalidManifestError, match="dtypes.*unexpected.*volume"):
        load_operator_manifest(payload)


def test_manifest_rejects_recursive_in_memory_container_graph(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    recursive: list[object] = []
    recursive.append(recursive)
    payload["parameters"]["options"] = {
        "type": "array",
        "default": recursive,
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }

    with pytest.raises(InvalidManifestError, match="recursive|cyclic") as exc_info:
        load_operator_manifest(payload)

    assert exc_info.value.to_dict()["code"] == "invalid_manifest"


def test_manifest_rejects_recursive_yaml_alias(valid_manifest_payload, tmp_path) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    recursive: list[object] = []
    recursive.append(recursive)
    payload["parameters"]["options"] = {
        "type": "array",
        "default": recursive,
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    source = tmp_path / "operator.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InvalidManifestError, match="recursive|cyclic") as exc_info:
        load_operator_manifest(source)

    assert exc_info.value.to_dict()["code"] == "invalid_manifest"


def test_manifest_accepts_non_recursive_yaml_alias(valid_manifest_payload, tmp_path) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    shared_dtypes = ["float64"]
    payload["inputs"]["optional_columns"] = ["volume"]
    payload["inputs"]["dtypes"] = {"close": shared_dtypes, "volume": shared_dtypes}
    source = tmp_path / "operator.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    manifest = load_operator_manifest(source)

    assert manifest.raw["inputs"]["dtypes"]["close"] == ("float64",)
    assert manifest.raw["inputs"]["dtypes"]["volume"] == ("float64",)


def test_manifest_rejects_invalid_output_template(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0]["name_template"] = "sma_{period"

    with pytest.raises(InvalidManifestError, match="template"):
        load_operator_manifest(payload)


def test_manifest_requires_output_template_parameters_to_be_resolvable(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["suffix"] = {
        "type": "string",
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "sma_{period}_{suffix}"

    with pytest.raises(InvalidManifestError, match="required or declare a default"):
        load_operator_manifest(payload)


def test_manifest_requires_template_parameters_to_affect_output_fields(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["period"]["affects_output_fields"] = False

    with pytest.raises(InvalidManifestError, match="affects_output_fields"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    ("parameter_type", "default"),
    [
        ("object", {"fast": 1, "slow": 2}),
        ("object", {"slow": 2, "fast": 1}),
        ("array", ["fast", "slow"]),
    ],
    ids=["object", "object-reversed", "array"],
)
def test_manifest_rejects_composite_default_parameters_in_output_templates(
    valid_manifest_payload,
    parameter_type,
    default,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["suffix"] = {
        "type": parameter_type,
        "default": default,
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "sma_{suffix}"

    with pytest.raises(InvalidManifestError, match="template parameters must be scalar: suffix"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("parameter_type", ["object", "array"])
def test_manifest_rejects_required_composite_parameters_in_output_templates(
    valid_manifest_payload,
    parameter_type,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["suffix"] = {
        "type": parameter_type,
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "sma_{suffix}"

    with pytest.raises(InvalidManifestError, match="template parameters must be scalar: suffix"):
        load_operator_manifest(payload)


def test_manifest_validates_nested_output_format_references(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0]["name_template"] = "sma_{period:0{width}d}"

    with pytest.raises(InvalidManifestError, match="unknown parameters: width"):
        load_operator_manifest(payload)


def test_manifest_converts_excessive_dynamic_format_nesting_to_structured_error(
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["spec"] = {
        "type": "string",
        "default": "d",
        "required": False,
        "enum": ["d"],
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    nested_spec = "{spec}"
    for _ in range(1_100):
        nested_spec = "{spec:" + nested_spec + "}"
    payload["outputs"]["fields"][0]["name_template"] = "{period:" + nested_spec + "}"

    with pytest.raises(InvalidManifestError, match="nesting depth") as exc_info:
        load_operator_manifest(payload)

    assert exc_info.value.code == "invalid_manifest"
    assert exc_info.value.operator_id == "fake.indicators.sma"
    assert exc_info.value.to_dict()["details"] == {"maximum_depth": 8}


@pytest.mark.parametrize("reserved_name", ["date", "code"])
def test_manifest_rejects_reserved_static_output_names(valid_manifest_payload, reserved_name) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0]["name_template"] = reserved_name

    with pytest.raises(InvalidManifestError, match="reserved QuantPanel key"):
        load_operator_manifest(payload)


def test_manifest_rejects_reserved_output_name_resolved_from_defaults(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["field"] = {
        "type": "string",
        "default": "date",
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{field}"

    with pytest.raises(InvalidManifestError, match="reserved QuantPanel key"):
        load_operator_manifest(payload)


def test_manifest_rejects_empty_output_name_resolved_from_defaults(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["field"] = {
        "type": "string",
        "default": "",
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{field}"

    with pytest.raises(InvalidManifestError, match="output field name must not be empty"):
        load_operator_manifest(payload)


def test_manifest_rejects_duplicate_default_resolved_output_names(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"].append({"name_template": "sma_2", "dtype": "float64"})
    payload["outputs"]["multiple"] = True

    with pytest.raises(InvalidManifestError, match="duplicate output field name: sma_2"):
        load_operator_manifest(payload)


def test_manifest_rejects_identical_output_templates_with_required_parameters(
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
    payload["outputs"]["fields"] = [
        {"name_template": "{field}", "dtype": "float64"},
        {"name_template": "{field}", "dtype": "float64"},
    ]
    payload["outputs"]["multiple"] = True

    with pytest.raises(InvalidManifestError, match="duplicate output field template: \\{field\\}") as exc_info:
        load_operator_manifest(payload)

    assert exc_info.value.code == "invalid_manifest"
    assert exc_info.value.operator_id == "fake.indicators.sma"


def test_catalog_requires_and_verifies_manifest_digests(valid_manifest_payload) -> None:
    standalone = load_operator_manifest(valid_manifest_payload)
    catalog_payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": standalone.digest}],
    }

    catalog = load_operator_catalog(catalog_payload)
    assert catalog.get("fake.indicators.sma").semantic_name == "SMA"
    assert catalog.operator_ids == ("fake.indicators.sma",)
    assert catalog.digest.startswith("sha256:")
    assert json.loads(catalog.to_json())["catalog_digest"] == catalog.digest


@pytest.mark.parametrize(
    "distribution",
    [
        "Fake-Quant-Operators",
        "fake_quant_operators",
        "fake.quant.operators",
        "fake--quant-operators",
        "fake-quant-operators-",
    ],
    ids=["uppercase", "underscore", "dot", "repeated-separator", "trailing-separator"],
)
def test_catalog_rejects_noncanonical_package_distribution_when_operators_are_empty(
    distribution: str,
) -> None:
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": distribution,
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [],
    }

    with pytest.raises(InvalidManifestError, match="package.distribution") as exc_info:
        load_operator_catalog(payload)

    assert exc_info.value.to_dict()["code"] == "invalid_manifest"


def test_catalog_accepts_nested_read_only_mapping_input_without_mutability_leakage(
    valid_manifest_payload,
) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    package_backing = {
        "distribution": "fake-quant-operators",
        "version": "1.0.0",
        "source_commit": "a" * 40,
        "source_tree_digest": "sha256:" + "b" * 64,
        "build_identifier": "ci-42",
    }
    operator_backing = {**valid_manifest_payload, "manifest_digest": manifest.digest}
    source = MappingProxyType(
        {
            "schema_version": 1,
            "contract_version": 1,
            "package": MappingProxyType(package_backing),
            "operators": (MappingProxyType(operator_backing),),
        }
    )

    catalog = load_operator_catalog(source)
    package_backing["build_identifier"] = "mutated"
    operator_backing["semantic_name"] = "mutated"

    assert catalog.package["build_identifier"] == "ci-42"
    assert catalog.get("fake.indicators.sma").semantic_name == "SMA"
    assert json.loads(catalog.to_json())["operators"][0]["semantic_name"] == "SMA"


def test_catalog_raw_snapshot_rejects_nested_mutation_without_changing_serialization(
    valid_manifest_payload,
) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    catalog = load_operator_catalog(
        {
            "schema_version": 1,
            "contract_version": 1,
            "package": {
                "distribution": "fake-quant-operators",
                "version": "1.0.0",
                "source_commit": "a" * 40,
                "source_tree_digest": "sha256:" + "b" * 64,
                "build_identifier": "ci-42",
            },
            "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
        }
    )
    serialized = catalog.to_json()

    with pytest.raises(TypeError):
        catalog._raw["package"]["build_identifier"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog._raw["operators"][0]["semantic_name"] = "mutated"
    with pytest.raises(AttributeError):
        catalog._raw["operators"][0]["inputs"]["required_columns"].append("volume")

    assert catalog.to_json() == serialized
    assert json.loads(serialized)["catalog_digest"] == catalog.digest


@pytest.mark.parametrize("source_kind", ["mapping", "yaml"])
@pytest.mark.parametrize("cycle_location", ["package", "operator"])
def test_catalog_rejects_recursive_container_graphs_before_traversal(
    valid_manifest_payload,
    tmp_path,
    source_kind,
    cycle_location,
) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
    }
    if cycle_location == "package":
        payload["package"]["build_identifier"] = payload["package"]
        expected_path = "catalog.package.build_identifier"
    else:
        recursive: list[object] = []
        recursive.append(recursive)
        payload["operators"][0]["parameters"]["period"]["default"] = recursive
        expected_path = "catalog.operators[0].parameters.period.default[0]"

    source = payload
    if source_kind == "yaml":
        source = tmp_path / "catalog.yaml"
        source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InvalidManifestError, match="recursive|cyclic") as exc_info:
        load_operator_catalog(source)

    assert expected_path in str(exc_info.value)
    assert exc_info.value.to_dict()["code"] == "invalid_manifest"


def test_catalog_rejects_deep_nonrecursive_mapping_without_leaking_recursion_error() -> None:
    nested: object = "leaf"
    for _ in range(1_100):
        nested = {"nested": nested}
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [],
        "unexpected": nested,
    }

    with pytest.raises(InvalidManifestError, match="nested too deeply") as exc_info:
        load_operator_catalog(payload)

    assert exc_info.value.to_dict()["code"] == "invalid_manifest"


@pytest.mark.parametrize("source_kind", ["mapping", "yaml"])
def test_catalog_accepts_shared_non_recursive_aliases(valid_manifest_payload, tmp_path, source_kind) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    shared_dtypes = ["float64", "float32", "int64"]
    payload["inputs"]["optional_columns"] = ["open"]
    payload["inputs"]["dtypes"] = {"close": shared_dtypes, "open": shared_dtypes}
    manifest = load_operator_manifest(payload)
    catalog_payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**payload, "manifest_digest": manifest.digest}],
    }

    source = catalog_payload
    if source_kind == "yaml":
        source = tmp_path / "catalog.yaml"
        source.write_text(yaml.safe_dump(catalog_payload, sort_keys=False), encoding="utf-8")

    catalog = load_operator_catalog(source)

    assert catalog.operator_ids == ("fake.indicators.sma",)


@pytest.mark.parametrize("field", ["schema_version", "contract_version"])
@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_catalog_requires_integer_version_fields(valid_manifest_payload, field, invalid_version) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
    }
    payload[field] = invalid_version

    with pytest.raises(InvalidManifestError, match=f"{field}.*integer"):
        load_operator_catalog(payload)


def test_catalog_digest_is_stable_across_mapping_order(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    package = {
        "distribution": "fake-quant-operators",
        "version": "1.0.0",
        "source_commit": "a" * 40,
        "source_tree_digest": "sha256:" + "b" * 64,
        "build_identifier": "ci-42",
    }
    left = load_operator_catalog(
        {
            "schema_version": 1,
            "contract_version": 1,
            "package": package,
            "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
        }
    )
    right = load_operator_catalog(
        {
            "operators": [{**dict(reversed(valid_manifest_payload.items())), "manifest_digest": manifest.digest}],
            "package": dict(reversed(package.items())),
            "contract_version": 1,
            "schema_version": 1,
        }
    )
    assert left.digest == right.digest


def test_catalog_rejects_duplicate_operator_ids(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    item = {**valid_manifest_payload, "manifest_digest": manifest.digest}
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [item, item],
    }

    with pytest.raises(InvalidManifestError, match="duplicate operator_id"):
        load_operator_catalog(payload)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_catalog_rejects_duplicate_keys_inside_nested_operator_manifest(
    valid_manifest_payload,
    tmp_path,
    suffix,
) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
    }
    source = tmp_path / f"catalog{suffix}"
    if suffix == ".json":
        raw = json.dumps(payload, separators=(",", ":"))
        raw = raw.replace('"min_assets":1', '"min_assets":1,"min_assets":1')
    else:
        raw = yaml.safe_dump(payload, sort_keys=False)
        raw = raw.replace("    min_assets: 1\n", "    min_assets: 1\n    min_assets: 1\n")
    source.write_text(raw, encoding="utf-8")

    with pytest.raises(
        InvalidManifestError,
        match=r"catalog\.operators\[0\]\.inputs.*duplicate.*min_assets",
    ) as exc_info:
        load_operator_catalog(source)

    assert exc_info.value.to_dict()["code"] == "invalid_manifest"


def test_catalog_rejects_invalid_package_semantic_version(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0-01",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
    }

    with pytest.raises(InvalidManifestError, match="semantic versioning"):
        load_operator_catalog(payload)


def test_catalog_rejects_unicode_digits_in_semantic_version(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1\u0660.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
    }

    with pytest.raises(InvalidManifestError, match="semantic versioning"):
        load_operator_catalog(payload)


@pytest.mark.parametrize("module", ["vendor..operators", "vendor.1operators", "vendor."])
def test_manifest_rejects_invalid_module_segments(valid_manifest_payload, module) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["module"] = module

    with pytest.raises(InvalidManifestError, match="module"):
        load_operator_manifest(payload)


def test_manifest_converts_default_output_format_failures_to_manifest_errors(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["suffix"] = {
        "type": "string",
        "default": "fast",
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "sma_{suffix:d}"

    with pytest.raises(InvalidManifestError, match="format"):
        load_operator_manifest(payload)


def test_manifest_rejects_output_format_incompatible_with_required_parameter_type(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["suffix"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "sma_{suffix:d}"

    with pytest.raises(InvalidManifestError, match="format.*parameter type"):
        load_operator_manifest(payload)


def test_manifest_accepts_output_format_compatible_with_required_parameter_type(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["suffix"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "sma_{suffix:>4s}"

    manifest = load_operator_manifest(payload)

    assert manifest.raw["outputs"]["fields"][0]["name_template"] == "sma_{suffix:>4s}"


@pytest.mark.parametrize(
    "domain",
    [
        {"minimum": 1_114_112},
        {"minimum": -1, "maximum": 65},
        {"enum": [65, 1_114_112]},
    ],
    ids=["minimum-too-large", "minimum-negative", "enum-member-outside-unicode"],
)
def test_manifest_rejects_value_sensitive_output_format_outside_required_parameter_domain(
    valid_manifest_payload,
    domain,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        **domain,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{value:c}"

    with pytest.raises(InvalidManifestError, match="format.*declared parameter domain"):
        load_operator_manifest(payload)


def test_manifest_requires_bounded_or_enumerated_domain_for_value_sensitive_output_format(
    valid_manifest_payload,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{value:c}"

    with pytest.raises(InvalidManifestError, match="format.*declared parameter domain"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    "domain",
    [
        {"minimum": 0, "maximum": 1_114_111},
        {"minimum": -0.5, "maximum": 1_114_111.9},
        {"enum": [0, 65, 1_114_111]},
    ],
    ids=["bounded", "fractional-bounds", "enum"],
)
def test_manifest_accepts_provably_safe_domain_for_value_sensitive_output_format(
    valid_manifest_payload,
    domain,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        **domain,
    }
    payload["outputs"]["fields"][0]["name_template"] = "prefix_{value:c}"

    manifest = load_operator_manifest(payload)

    assert manifest.raw["outputs"]["fields"][0]["name_template"] == "prefix_{value:c}"


@pytest.mark.parametrize("format_specs", [["c"], ["d", "c"]], ids=["c", "mixed-with-c"])
def test_manifest_rejects_dynamic_value_sensitive_output_format_outside_parameter_domain(
    valid_manifest_payload,
    format_specs,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        "minimum": 1_114_112,
    }
    payload["parameters"]["spec"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        "enum": format_specs,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{value:{spec}}"

    with pytest.raises(InvalidManifestError, match="format.*declared parameter domain"):
        load_operator_manifest(payload)


def test_manifest_rejects_dynamic_output_format_without_finite_spec_domain(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["parameters"]["spec"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{value:{spec}}"

    with pytest.raises(InvalidManifestError, match="format.*declared parameter domain"):
        load_operator_manifest(payload)


def test_manifest_rejects_excessive_dynamic_output_format_combinations(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    for name in ("width", "precision"):
        payload["parameters"][name] = {
            "type": "integer",
            "required": True,
            "unit": None,
            "affects_warmup": False,
            "affects_output_fields": True,
            "affects_causality": False,
            "affects_availability": False,
            "enum": list(range(17)),
        }
    payload["outputs"]["fields"][0]["name_template"] = "{value:{width}.{precision}f}"

    with pytest.raises(InvalidManifestError, match="format.*declared parameter domain"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    ("name_template", "size_parameter"),
    [
        ("{value:{width}d}", "width"),
        ("{value:.{precision}f}", "precision"),
    ],
    ids=["width", "precision"],
)
def test_manifest_rejects_oversized_dynamic_format_sizes_before_rendering(
    valid_manifest_payload,
    monkeypatch,
    name_template,
    size_parameter,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer" if size_parameter == "width" else "number",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["parameters"][size_parameter] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        "enum": [1_000_000_000],
    }
    payload["outputs"]["fields"][0]["name_template"] = name_template

    def fail_if_rendered(*args, **kwargs):
        raise AssertionError("format() must not run before size validation")

    monkeypatch.setattr("builtins.format", fail_if_rendered)

    with pytest.raises(InvalidManifestError, match="resource limit.*width|resource limit.*precision"):
        load_operator_manifest(payload)


def test_manifest_rejects_oversized_resolved_output_name_before_rendering(
    valid_manifest_payload,
    monkeypatch,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "x" * 250 + "{value:10d}"

    def fail_if_rendered(*args, **kwargs):
        raise AssertionError("format() must not run before output-name length validation")

    monkeypatch.setattr("builtins.format", fail_if_rendered)

    with pytest.raises(InvalidManifestError, match="resource limit.*maximum length"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    ("format_specs", "value_domain"),
    [
        (["c"], {"minimum": 0, "maximum": 1_114_111}),
        (["d", "08x"], {}),
    ],
    ids=["unicode-code-point", "non-value-sensitive"],
)
def test_manifest_accepts_provably_safe_dynamic_output_format_domains(
    valid_manifest_payload,
    format_specs,
    value_domain,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        **value_domain,
    }
    payload["parameters"]["spec"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        "enum": format_specs,
    }
    payload["outputs"]["fields"][0]["name_template"] = "{value:{spec}}"

    manifest = load_operator_manifest(payload)

    assert manifest.raw["outputs"]["fields"][0]["name_template"] == "{value:{spec}}"


def test_manifest_rejects_dynamic_output_format_incompatible_with_value_type(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["parameters"]["spec"] = {
        "type": "string",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
        "enum": ["s"],
    }
    payload["outputs"]["fields"][0]["name_template"] = "{value:{spec}}"

    with pytest.raises(InvalidManifestError, match="format.*declared parameter domain"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("format_spec", ["d", "08x", ",d", ">12d"])
def test_manifest_accepts_common_integer_output_formats_without_bounded_domain(
    valid_manifest_payload,
    format_spec,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["value"] = {
        "type": "integer",
        "required": True,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = f"value_{{value:{format_spec}}}"

    manifest = load_operator_manifest(payload)

    assert manifest.raw["outputs"]["fields"][0]["name_template"] == f"value_{{value:{format_spec}}}"


def test_manifest_rejects_composite_parameter_attribute_access_in_output_template(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["options"] = {
        "type": "object",
        "default": {},
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = "sma_{options.name}"

    with pytest.raises(InvalidManifestError, match="reference parameters directly"):
        load_operator_manifest(payload)


@pytest.mark.parametrize("name_template", ["sma_{suffix.upper}", "sma_{suffix[0]}"])
def test_manifest_requires_output_template_fields_to_reference_parameters_directly(
    valid_manifest_payload,
    name_template,
) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["parameters"]["suffix"] = {
        "type": "string",
        "default": "fast",
        "required": False,
        "unit": None,
        "affects_warmup": False,
        "affects_output_fields": True,
        "affects_causality": False,
        "affects_availability": False,
    }
    payload["outputs"]["fields"][0]["name_template"] = name_template

    with pytest.raises(InvalidManifestError, match="reference parameters directly"):
        load_operator_manifest(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [(("outputs", "fields", 0, "minimum"), float("nan")), (("determinism", "absolute_tolerance"), float("inf"))],
)
def test_manifest_converts_nonfinite_declarations_to_manifest_errors(valid_manifest_payload, path, value) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    if path[0] == "outputs":
        payload["outputs"]["fields"][0]["minimum"] = value
    else:
        payload["determinism"] = {"absolute_tolerance": value}

    with pytest.raises(InvalidManifestError, match="finite"):
        load_operator_manifest(payload)


def test_catalog_rejects_unsupported_source_commit_lengths(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 41,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
    }

    with pytest.raises(InvalidManifestError, match="full hexadecimal commit"):
        load_operator_catalog(payload)


@pytest.mark.parametrize("source_kind", ["mapping", "yaml"])
@pytest.mark.parametrize("key_location", ["top_level", "nested"])
def test_catalog_rejects_non_string_mapping_keys_before_field_processing(
    valid_manifest_payload,
    tmp_path,
    source_kind,
    key_location,
) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    payload = {
        "schema_version": 1,
        "contract_version": 1,
        "package": {
            "distribution": "fake-quant-operators",
            "version": "1.0.0",
            "source_commit": "a" * 40,
            "source_tree_digest": "sha256:" + "b" * 64,
            "build_identifier": "ci-42",
        },
        "operators": [{**valid_manifest_payload, "manifest_digest": manifest.digest}],
    }
    if key_location == "top_level":
        payload["unknown"] = "field"
        payload[1] = "invalid"
        expected_path = "catalog[1]"
    else:
        payload["unknown"] = {"nested": [{1: "invalid"}]}
        expected_path = "catalog.unknown.nested[0][1]"

    source = payload
    if source_kind == "yaml":
        source = tmp_path / "catalog.yaml"
        source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InvalidManifestError, match="catalog mapping keys must be strings") as exc_info:
        load_operator_catalog(source)

    assert expected_path in str(exc_info.value)
    assert exc_info.value.to_dict()["code"] == "invalid_manifest"
