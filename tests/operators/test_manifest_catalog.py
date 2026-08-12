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
