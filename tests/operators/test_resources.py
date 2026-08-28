import hashlib
import json
import subprocess
import zipfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from oxq.operators.resources import (
    materialize_certification_profile,
    materialize_contract_surface,
    materialize_operator_distribution_profile,
    materialize_operator_install_profile,
)

EXPECTED_SURFACE_DIGESTS = {
    "quant_panel_schema": "sha256:fd6fcd7f3102cdd63913644f87a154a22713c0286a6e9e1cc16e84ca6b283a9c",
    "operator_manifest_schema": "sha256:b50b15c446d8940a358ff32c31e191407a7841ab57a30f2fcbafb9a15b9f8793",
    "operator_binding_schema": "sha256:1d0e3ed12acde2a2d0c1fe2309f9a090ea7b0f8193bc0f3f6fd659c178047de6",
    "reference_validator": "sha256:33528a97f6405809ead8a9f542c1c2dae9d89cb6c966943def6ca80097e8f67a",
}
PROFILE_NAMES = {
    "provider_catalog",
    "candidate_build",
    "numerical_baseline",
    "certification_record",
}
FROZEN_WHEEL_MEMBERS = {
    "oxq/operators/contracts/v1/quant-panel-v1.schema.json",
    "oxq/operators/contracts/v1/operator-manifest-v1.schema.json",
    "oxq/operators/contracts/v1/operator-binding-v1.schema.json",
    "oxq/operators/contracts/v1/reference_validator_v1.py",
}
PROFILE_WHEEL_MEMBERS = {
    "oxq/operators/certification_profile/v1/provider-catalog-v1.schema.json",
    "oxq/operators/certification_profile/v1/candidate-build-v1.schema.json",
    "oxq/operators/certification_profile/v1/numerical-baseline-v1.schema.json",
    "oxq/operators/certification_profile/v1/certification-record-v1.schema.json",
}


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    return _digest(path.read_bytes())


def _valid_quant_panel() -> dict[str, object]:
    return {
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
            "source": "test",
        },
        "alignment": "preserve_input_order",
        "records": [{"date": "2026-08-26", "code": "000001.SZ", "close": 10.0}],
    }


def _valid_profile_instances() -> dict[str, dict[str, object]]:
    digest = "sha256:" + "a" * 64
    return {
        "provider_catalog": {
            "schema_version": 1,
            "provider": {"name": "equant-py", "release": "1.0.0"},
            "contract_version": "1.0.0",
            "status": "candidate",
            "build_record": "candidate-build-v1.json",
            "operators": {
                "equant.ttr.sma@1.0.0": {
                    "manifest": "manifests/equant.ttr.sma.operator.json",
                    "baseline": "numerical_baselines/technical-v1.json",
                }
            },
        },
        "candidate_build": {
            "schema_version": 1,
            "provider": "equant-py",
            "release": "1.0.0",
            "source_commit": "git-sha1:" + "a" * 40,
            "python": "3.12.0",
            "build_command": "uv build",
            "artifacts": [
                {
                    "distribution": "equant-ttr",
                    "version": "1.0.0",
                    "filename": "equant_ttr-1.0.0-py3-none-any.whl",
                    "role": "implementation",
                    "build_identifier": "build-20260826-equant-ttr",
                    "digest": digest,
                }
            ],
        },
        "numerical_baseline": {
            "schema_version": 1,
            "provider": "equant-py",
            "release": "1.0.0",
            "cases": [
                {
                    "case_id": "sma-window-3",
                    "operator_id": "equant.ttr.sma",
                    "operator_version": "1.0.0",
                    "parameters": {"window": 3},
                    "input": _valid_quant_panel(),
                    "expected": {"sma_3": [None, 10.0]},
                    "tolerance": {"absolute": 0.0, "relative": 0.0},
                }
            ],
        },
        "certification_record": {
            "schema_version": 1,
            "certifier": "open-xquant-local",
            "certified_at": "2026-08-26T12:00:00Z",
            "provider": "equant-py",
            "release": "1.0.0",
            "submission_commit": "git-sha1:" + "b" * 40,
            "source_commit": "git-sha1:" + "a" * 40,
            "state": "research-certified",
            "artifacts": [
                {
                    "distribution": "equant-ttr",
                    "version": "1.0.0",
                    "filename": "equant_ttr-1.0.0-py3-none-any.whl",
                    "role": "implementation",
                    "build_identifier": "build-20260826-equant-ttr",
                    "digest": digest,
                }
            ],
            "operators": [
                {
                    "operator_id": "equant.ttr.sma",
                    "operator_version": "1.0.0",
                    "manifest_digest": digest,
                    "implementation_digest": digest,
                    "binding_digest": digest,
                    "baseline_cases": [{"case_id": "sma-3", "status": "passed"}],
                }
            ],
        },
    }


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _validate_profile_schemas(schemas: dict[str, dict[str, object]]) -> None:
    for name, instance in _valid_profile_instances().items():
        Draft202012Validator(
            schemas[name],
            format_checker=FormatChecker(),
        ).validate(instance)


def test_certification_artifact_version_policy_matches_artifact_roles() -> None:
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in (Path(__file__).parents[2] / "contracts" / "operator-certification").glob("*.schema.json")
    }
    instances = _valid_profile_instances()

    candidate_build = deepcopy(instances["candidate_build"])
    candidate_build["artifacts"][0]["version"] = "1.0.0.post1"  # type: ignore[index]
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas["candidate-build-v1.schema.json"]).validate(candidate_build)

    runtime_candidate_build = deepcopy(instances["candidate_build"])
    runtime_candidate_build["artifacts"].append(  # type: ignore[union-attr]
        {
            "distribution": "helper-pkg",
            "version": "1.0.0.post1",
            "filename": "helper_pkg-1.0.0.post1-py3-none-any.whl",
            "role": "runtime-dependency",
            "build_identifier": "helper-build",
            "digest": "sha256:" + "b" * 64,
        }
    )
    Draft202012Validator(schemas["candidate-build-v1.schema.json"]).validate(runtime_candidate_build)
    for invalid_version in ("1_", "1..."):
        invalid_runtime_candidate_build = deepcopy(runtime_candidate_build)
        invalid_runtime_candidate_build["artifacts"][1]["version"] = invalid_version  # type: ignore[index]
        with pytest.raises(ValidationError):
            Draft202012Validator(schemas["candidate-build-v1.schema.json"]).validate(invalid_runtime_candidate_build)

    certification_record = deepcopy(instances["certification_record"])
    certification_record["artifacts"][0]["version"] = "1.0.0.post1"  # type: ignore[index]
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas["certification-record-v1.schema.json"]).validate(certification_record)

    runtime_certification_record = deepcopy(instances["certification_record"])
    runtime_certification_record["artifacts"].append(  # type: ignore[union-attr]
        {
            "distribution": "helper-pkg",
            "version": "1.0.0.post1",
            "filename": "helper_pkg-1.0.0.post1-py3-none-any.whl",
            "role": "runtime-dependency",
            "build_identifier": "helper-build",
            "digest": "sha256:" + "b" * 64,
        }
    )
    Draft202012Validator(schemas["certification-record-v1.schema.json"]).validate(runtime_certification_record)
    for invalid_version in ("1_", "1..."):
        invalid_runtime_certification_record = deepcopy(runtime_certification_record)
        invalid_runtime_certification_record["artifacts"][1]["version"] = invalid_version  # type: ignore[index]
        with pytest.raises(ValidationError):
            Draft202012Validator(schemas["certification-record-v1.schema.json"]).validate(invalid_runtime_certification_record)


def test_materialized_frozen_surface_has_exact_digests() -> None:
    with materialize_contract_surface() as paths:
        assert set(paths) == set(EXPECTED_SURFACE_DIGESTS)
        assert {
            name: sha256_file(path) for name, path in paths.items()
        } == EXPECTED_SURFACE_DIGESTS


def test_certification_profile_schemas_are_valid_draft_2020_12() -> None:
    with materialize_certification_profile() as paths:
        assert set(paths) == PROFILE_NAMES
        schemas = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
        for schema in schemas.values():
            Draft202012Validator.check_schema(schema)
        _validate_profile_schemas(schemas)


def test_operator_distribution_and_install_profiles_are_packaged() -> None:
    with materialize_operator_distribution_profile() as distribution:
        with materialize_operator_install_profile() as install:
            paths = {**distribution, **install}

    assert set(paths) == {
        "certification_record_v2",
        "operator_release",
        "runtime_protocol",
        "official_providers",
        "official_environment_providers",
    }
    for path in paths.values():
        assert path.is_file()


def test_environment_install_phase_does_not_package_installed_release_schema() -> None:
    with materialize_operator_install_profile() as paths:
        assert "installed_release" not in paths


def test_catalog_uses_operator_identity_version_keys() -> None:
    with materialize_certification_profile() as paths:
        schema = json.loads(paths["provider_catalog"].read_text(encoding="utf-8"))
    instance = _valid_profile_instances()["provider_catalog"]
    Draft202012Validator(schema).validate(instance)

    invalid_identity = deepcopy(instance)
    invalid_identity["operators"] = {
        "equant.ttr.sma@1.0": instance["operators"]["equant.ttr.sma@1.0.0"]
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid_identity)


def test_catalog_rejects_duplicate_operator_identity_json_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        json.loads(
            '{"equant.ttr.sma@1.0.0": {}, "equant.ttr.sma@1.0.0": {}}',
            object_pairs_hook=_strict_json_object,
        )


@pytest.mark.parametrize(
    ("profile_name", "mutate"),
    [
        (
            "provider_catalog",
            lambda instance: instance.update({"build_record": "../candidate-build-v1.json"}),
        ),
        (
            "provider_catalog",
            lambda instance: instance["operators"]["equant.ttr.sma@1.0.0"].update({"manifest": "/manifest.json"}),
        ),
        (
            "candidate_build",
            lambda instance: instance.update({"source_commit": "git-sha1:" + "A" * 40}),
        ),
        (
            "candidate_build",
            lambda instance: instance["artifacts"][0].update({"role": "unknown"}),
        ),
        (
            "candidate_build",
            lambda instance: instance["artifacts"][0].update({"digest": "sha256:bad"}),
        ),
        (
            "candidate_build",
            lambda instance: instance["artifacts"][0].update({"build_identifier": ""}),
        ),
        (
            "candidate_build",
            lambda instance: instance["artifacts"][0].pop("build_identifier"),
        ),
        (
            "numerical_baseline",
            lambda instance: instance["cases"][0].pop("case_id"),
        ),
        (
            "numerical_baseline",
            lambda instance: instance["cases"][0]["expected"].update({"sma_3": [[10.0]]}),
        ),
        (
            "numerical_baseline",
            lambda instance: instance["cases"][0]["tolerance"].update({"absolute": -0.1}),
        ),
        (
            "certification_record",
            lambda instance: instance.update({"certifier": "another-certifier"}),
        ),
        (
            "certification_record",
            lambda instance: instance.pop("submission_commit"),
        ),
        (
            "certification_record",
            lambda instance: instance.update({"artifacts": []}),
        ),
        (
            "certification_record",
            lambda instance: instance.update({"certified_at": "2026-08-26T12:00:00+08:00"}),
        ),
        (
            "certification_record",
            lambda instance: instance.update({"certified_at": "not-a-dateZ"}),
        ),
    ],
)
def test_profile_schemas_reject_invalid_certification_inputs(
    profile_name: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    with materialize_certification_profile() as paths:
        schema = json.loads(paths[profile_name].read_text(encoding="utf-8"))
    instance = deepcopy(_valid_profile_instances()[profile_name])
    mutate(instance)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)


def test_wheel_contains_operator_contract_and_profile_resources(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[2]
    profile_source_directory = repository_root / "contracts" / "operator-certification"
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=repository_root,
    )
    wheel_path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        members = set(wheel.namelist())
        frozen_members = {
            member for member in members if member.startswith("oxq/operators/contracts/v1/")
        }
        profile_members = {
            member
            for member in members
            if member.startswith("oxq/operators/certification_profile/v1/")
        }
        frozen_digests = {
            name: _digest(wheel.read(member))
            for name, member in {
                "quant_panel_schema": "oxq/operators/contracts/v1/quant-panel-v1.schema.json",
                "operator_manifest_schema": "oxq/operators/contracts/v1/operator-manifest-v1.schema.json",
                "operator_binding_schema": "oxq/operators/contracts/v1/operator-binding-v1.schema.json",
                "reference_validator": "oxq/operators/contracts/v1/reference_validator_v1.py",
            }.items()
        }
        profile_schemas = {
            member.rsplit("/", 1)[-1].removesuffix("-v1.schema.json").replace("-", "_"): json.loads(wheel.read(member))
            for member in PROFILE_WHEEL_MEMBERS
        }
        wheel_profile_bytes = {
            member: wheel.read(member) for member in PROFILE_WHEEL_MEMBERS
        }

    source_profile_bytes = {
        f"oxq/operators/certification_profile/v1/{filename}": (
            profile_source_directory / filename
        ).read_bytes()
        for filename in (
            "provider-catalog-v1.schema.json",
            "candidate-build-v1.schema.json",
            "numerical-baseline-v1.schema.json",
            "certification-record-v1.schema.json",
        )
    }

    assert frozen_members == FROZEN_WHEEL_MEMBERS
    assert profile_members == PROFILE_WHEEL_MEMBERS
    assert frozen_digests == EXPECTED_SURFACE_DIGESTS
    assert wheel_profile_bytes == source_profile_bytes
    _validate_profile_schemas(profile_schemas)
