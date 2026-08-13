from __future__ import annotations

import copy
import json
from datetime import date

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


def test_manifest_rejects_negative_warmup_resolved_from_parameter_default(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["warmup"]["offset"] = -3

    with pytest.raises(InvalidManifestError, match="warmup.*default.*non-negative"):
        load_operator_manifest(payload)


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


def test_manifest_rejects_cross_section_scope_with_multi_row_history(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["execution_scope"] = "cross_section"

    with pytest.raises(InvalidManifestError, match="cross_section.*min_history"):
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

    manifest = load_operator_manifest(payload)

    assert manifest.validate_parameters({}) == {"period": huge_integer}


def test_manifest_accepts_arbitrary_size_integer_enum_member(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    huge_integer = 10**1000
    payload["parameters"]["period"]["enum"] = [2, huge_integer]

    manifest = load_operator_manifest(payload)

    assert huge_integer in manifest.raw["parameters"]["period"]["enum"]


def test_manifest_accepts_arbitrary_size_integer_request(valid_manifest_payload) -> None:
    manifest = load_operator_manifest(valid_manifest_payload)
    huge_integer = 10**1000

    assert manifest.validate_parameters({"period": huge_integer}) == {"period": huge_integer}


def test_manifest_rejects_overlapping_input_columns(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["inputs"]["optional_columns"] = ["close"]

    with pytest.raises(InvalidManifestError, match="required_columns"):
        load_operator_manifest(payload)


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


def test_manifest_validates_nested_output_format_references(valid_manifest_payload) -> None:
    payload = copy.deepcopy(valid_manifest_payload)
    payload["outputs"]["fields"][0]["name_template"] = "sma_{period:0{width}d}"

    with pytest.raises(InvalidManifestError, match="unknown parameters: width"):
        load_operator_manifest(payload)


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
