"""Strict packaged contracts for operator distribution and installation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from oxq.operators.formats import (
    canonical_json_bytes,
    safe_relative_path,
    sha256_bytes,
    strict_json_object,
)
from oxq.operators.resources import (
    materialize_operator_distribution_profile,
    materialize_operator_install_profile,
)


def _schemas() -> dict[str, dict[str, object]]:
    with materialize_operator_distribution_profile() as distribution:
        with materialize_operator_install_profile() as install:
            return {
                name: json.loads(path.read_text(encoding="utf-8"))
                for name, path in {**distribution, **install}.items()
                if name not in {"official_providers", "official_environment_providers"}
            }


def test_operator_distribution_resources_are_packaged_and_strict() -> None:
    schemas = _schemas()

    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False

    record = schemas["certification_record_v2"]
    assert "target" in record["required"]
    assert "baseline_sets" in record["required"]
    case = record["$defs"]["operator"]["properties"]["baseline_cases"]["items"]
    assert {"baseline_path", "case_index", "case_digest"} <= set(case["required"])


@pytest.mark.parametrize(
    "name",
    ["operator_release"],
)
def test_install_schemas_bound_size_bytes_and_reject_extra_fields(name: str) -> None:
    schema = _schemas()[name]
    serialized = json.dumps(schema, sort_keys=True)
    assert '"size_bytes"' in serialized
    assert '"maximum"' in serialized

    invalid = {key: None for key in schema["required"]}
    invalid["unexpected"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid)


def test_official_provider_map_uses_canonical_provider_name() -> None:
    with materialize_operator_install_profile() as paths:
        provider_map = json.loads(paths["official_providers"].read_text(encoding="utf-8"))
    assert provider_map == {
        "schema_version": 1,
        "providers": {
            "equant-py": {
                "repository": "xingwudao/equant-py",
                "release_asset": "operator-release-v1.json",
            }
        },
    }


def test_shared_strict_formats_reject_ambiguous_json_and_paths() -> None:
    assert strict_json_object(b'{"key": 1}') == {"key": 1}
    assert canonical_json_bytes({"z": 1, "a": "x"}) == b'{"a":"x","z":1}\n'
    assert sha256_bytes(b"open-xquant") == (
        "sha256:2068031c336a3d98dbdfce0325431289d6ef75167af65e043f5a20ae8f1093ec"
    )
    assert safe_relative_path("publication/record.json").as_posix() == "publication/record.json"

    for raw in (b'[]', b'{"key": NaN}', b'{"key": 1, "key": 2}'):
        with pytest.raises(ValueError):
            strict_json_object(raw)
    for path in (
        "",
        ".",
        "dir/.",
        "dir//file.json",
        "/absolute.json",
        "../escape.json",
        "one\\two.json",
        "nul\x00.json",
    ):
        with pytest.raises(ValueError):
            safe_relative_path(path)


@pytest.mark.parametrize(
    "invalid",
    [
        "",
        ".",
        "dir/.",
        "dir//file.json",
        "/absolute.json",
        "../escape.json",
        "one\\two.json",
        "nul\x00.json",
        "trailing/",
    ],
)
def test_certification_record_v2_paths_are_canonical_posix_components(
    invalid: str,
) -> None:
    path_schema = _schemas()["certification_record_v2"]["$defs"]["path"]

    with pytest.raises(ValidationError):
        Draft202012Validator(path_schema).validate(invalid)


def test_certification_record_v2_rejects_nul_artifact_filename() -> None:
    filename_schema = _schemas()["certification_record_v2"]["$defs"]["artifact"]["properties"]["filename"]

    with pytest.raises(ValidationError):
        Draft202012Validator(filename_schema).validate("provider\x00.whl")


def test_release_index_accepts_closed_wheel_entries() -> None:
    schema = _schemas()["operator_release"]
    digest = "sha256:" + "a" * 64
    asset = {
        "filename": "bundle.zip",
        "url": "https://github.com/xingwudao/equant-py/releases/download/v1.0.0/bundle.zip",
        "size_bytes": 1,
        "digest": digest,
    }
    index = {
        "schema_version": 1,
        "release_type": "open-xquant-operator-release",
        "provider": "equant-py",
        "release": "1.0.0",
        "submission_commit": "git-sha1:" + "a" * 40,
        "source_commit": "git-sha1:" + "b" * 40,
        "certification_state": "research-certified",
        "operator_count": 1,
        "targets": [
            {
                "python_tag": "cp312",
                "abi_tag": "cp312",
                "platform_tag": "macosx_14_0_arm64",
                "bundle": asset,
                "wheels": [
                    {
                        **asset,
                        "filename": "equant_ttr-1.0.0-py3-none-any.whl",
                        "distribution": "equant-ttr",
                        "version": "1.0.0",
                        "role": "implementation",
                        "tags": ["py3-none-any"],
                    }
                ],
            }
        ],
    }

    Draft202012Validator(schema).validate(index)


@pytest.mark.parametrize(
    ("path", "filename"),
    [
        (("targets", 0, "bundle", "filename"), "bundle\x00.zip"),
        (("targets", 0, "wheels", 0, "filename"), "equant_ttr\x00-1.0.0-py3-none-any.whl"),
    ],
)
def test_release_index_rejects_nul_asset_filenames(
    path: tuple[object, ...],
    filename: str,
) -> None:
    schema = _schemas()["operator_release"]
    digest = "sha256:" + "a" * 64
    asset = {
        "filename": "bundle.zip",
        "url": "https://github.com/xingwudao/equant-py/releases/download/v1.0.0/bundle.zip",
        "size_bytes": 1,
        "digest": digest,
    }
    index = {
        "schema_version": 1,
        "release_type": "open-xquant-operator-release",
        "provider": "equant-py",
        "release": "1.0.0",
        "submission_commit": "git-sha1:" + "a" * 40,
        "source_commit": "git-sha1:" + "b" * 40,
        "certification_state": "research-certified",
        "operator_count": 1,
        "targets": [
            {
                "python_tag": "cp312",
                "abi_tag": "cp312",
                "platform_tag": "macosx_14_0_arm64",
                "bundle": asset.copy(),
                "wheels": [
                    {
                        **asset,
                        "filename": "equant_ttr-1.0.0-py3-none-any.whl",
                        "distribution": "equant-ttr",
                        "version": "1.0.0",
                        "role": "implementation",
                        "tags": ["py3-none-any"],
                    }
                ],
            }
        ],
    }
    target: object = index
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = filename  # type: ignore[index]

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(index)


def test_release_index_accepts_compressed_wheel_tags() -> None:
    tag_schema = _schemas()["operator_release"]["$defs"]["wheel"]["properties"]["tags"]["items"]

    Draft202012Validator(tag_schema).validate("py3.py312-none-any")
    Draft202012Validator(tag_schema).validate("py3-none-any.macosx_14_0_arm64")


def test_runtime_protocol_schema_matches_implemented_messages() -> None:
    schema = _schemas()["runtime_protocol"]
    validator = Draft202012Validator(schema)
    request = {
        "implementation_artifact": "/tmp/provider.whl",
        "dependency_artifacts": ["/tmp/dependency.whl"],
        "module": "ettr",
        "callable": "sma",
        "parameters": {"n": 2},
        "input": {
            "schema_version": 1,
            "primary_key": ["date", "code"],
            "columns": [{"name": "close", "dtype": "float64", "required": True}],
            "context": {
                "timezone": "Asia/Shanghai",
                "calendar": "XSHG",
                "frequency": "1d",
                "timestamp_semantics": "bar_close",
                "currency": "CNY",
                "price_adjustment": "raw",
                "data_version": "v1",
                "source": "literal-test",
            },
            "alignment": "preserve_input_order",
            "records": [{"date": "2026-08-24", "code": "000001.SZ", "close": 1.0}],
        },
        "output_fields": [
            {
                "name": "sma_2",
                "dtype": "float64",
            }
        ],
        "output_alignment": "preserve_input_order",
    }
    ok_response = {
        "status": "ok",
        "outputs": {"sma_2": [None, 1.5]},
        "repeated_outputs": {"sma_2": [None, 1.5]},
    }
    error_response = {
        "status": "error",
        "code": "provider_import_failed",
    }

    for message in (request, ok_response, error_response):
        validator.validate(message)

    with pytest.raises(ValidationError):
        validator.validate({**request, "test_runtime_paths": ["/tmp/runtime"]})


def test_runtime_protocol_rejects_extra_output_field_properties() -> None:
    schema = _schemas()["runtime_protocol"]
    request = {
        "implementation_artifact": "/tmp/provider.whl",
        "dependency_artifacts": [],
        "module": "ettr",
        "callable": "sma",
        "parameters": {},
        "input": {
            "schema_version": 1,
            "primary_key": ["date", "code"],
            "columns": [{"name": "close", "dtype": "float64", "required": True}],
            "context": {"timezone": "Asia/Shanghai"},
            "alignment": "preserve_input_order",
            "records": [{"date": "2026-08-24", "code": "000001.SZ", "close": 1.0}],
        },
        "output_fields": [{"name": "sma_2", "dtype": "float64", "extra": True}],
        "output_alignment": "preserve_input_order",
    }

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(request)


def test_readme_certification_workflow_does_not_document_removed_output_dir() -> None:
    readme = Path(__file__).resolve().parents[2].joinpath("README.md").read_text(
        encoding="utf-8"
    )

    assert "--output-dir" not in readme
    assert ".open-xquant/certifications/<provider>/<release>/" not in readme


def test_environment_install_plan_uses_split_equant_distributions() -> None:
    plan = Path(__file__).resolve().parents[2].joinpath(
        "docs/superpowers/plans/2026-08-27-certified-operator-one-step-install.md"
    ).read_text(encoding="utf-8")

    assert "pip install equant-core==1.0.0 equant-ttr==1.0.0" in plan
    assert "pip install equant-py==1.0.0" not in plan


@pytest.mark.parametrize("invalid", ["1.2.3foo", "1.2.3.4"])
def test_public_release_versions_require_exact_semver(invalid: str) -> None:
    schemas = _schemas()
    semver = schemas["operator_release"]["$defs"]["semver"]
    with pytest.raises(ValidationError):
        Draft202012Validator(semver).validate(invalid)


@pytest.mark.parametrize("invalid", ["not-a-version", "1...0"])
def test_distribution_versions_require_pep440(invalid: str) -> None:
    schemas = _schemas()
    for name, definition in (("certification_record_v2", "artifact"), ("operator_release", "wheel")):
        schema = schemas[name]
        assert schema["$defs"][definition]["properties"]["version"] == {
            "$ref": "#/$defs/pythonPackageVersion"
        }
        with pytest.raises(ValidationError):
            Draft202012Validator(schema["$defs"]["pythonPackageVersion"]).validate(invalid)
