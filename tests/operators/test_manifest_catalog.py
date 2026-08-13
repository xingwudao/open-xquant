from __future__ import annotations

import copy
import json

import pytest

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
