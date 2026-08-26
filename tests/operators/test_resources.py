import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from jsonschema import Draft202012Validator

from oxq.operators.resources import (
    materialize_certification_profile,
    materialize_contract_surface,
)

EXPECTED_SURFACE_DIGESTS = {
    "quant_panel_schema": "sha256:fd6fcd7f3102cdd63913644f87a154a22713c0286a6e9e1cc16e84ca6b283a9c",
    "operator_manifest_schema": "sha256:adea87a6caec3984d65d9fbaaa0ba132be76e5609ed17407de5e8b85c38bf82e",
    "operator_binding_schema": "sha256:1d0e3ed12acde2a2d0c1fe2309f9a090ea7b0f8193bc0f3f6fd659c178047de6",
    "reference_validator": "sha256:48099f887ebfc9fd9857ba8cececaa8b52c1dd5a2020ccc5eca21c3120664d9a",
}


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_materialized_frozen_surface_has_exact_digests() -> None:
    with materialize_contract_surface() as paths:
        assert set(paths) == set(EXPECTED_SURFACE_DIGESTS)
        assert {
            name: sha256_file(path) for name, path in paths.items()
        } == EXPECTED_SURFACE_DIGESTS


def test_certification_profile_schemas_are_valid_draft_2020_12() -> None:
    with materialize_certification_profile() as paths:
        assert set(paths) == {
            "provider_catalog",
            "candidate_build",
            "numerical_baseline",
            "certification_record",
        }
        for path in paths.values():
            Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_wheel_contains_operator_contract_and_profile_resources(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[2]
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

    assert {
        "oxq/operators/contracts/v1/quant-panel-v1.schema.json",
        "oxq/operators/contracts/v1/operator-manifest-v1.schema.json",
        "oxq/operators/contracts/v1/operator-binding-v1.schema.json",
        "oxq/operators/contracts/v1/reference_validator_v1.py",
        "oxq/operators/certification_profile/v1/provider-catalog-v1.schema.json",
        "oxq/operators/certification_profile/v1/candidate-build-v1.schema.json",
        "oxq/operators/certification_profile/v1/numerical-baseline-v1.schema.json",
        "oxq/operators/certification_profile/v1/certification-record-v1.schema.json",
    } <= members
