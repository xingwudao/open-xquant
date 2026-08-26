"""Contract validation tests for loaded local provider submissions."""

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import oxq.operators.certification as certification
from oxq.operators.certification import validate_provider_contract
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.resources import materialize_contract_surface
from oxq.operators.submission import load_provider_submission
from tests.operators.helpers import rewrite_json, write_provider_repository

EXPECTED_SURFACE_DIGESTS = {
    "quant_panel_schema": "sha256:fd6fcd7f3102cdd63913644f87a154a22713c0286a6e9e1cc16e84ca6b283a9c",
    "operator_manifest_schema": "sha256:adea87a6caec3984d65d9fbaaa0ba132be76e5609ed17407de5e8b85c38bf82e",
    "operator_binding_schema": "sha256:1d0e3ed12acde2a2d0c1fe2309f9a090ea7b0f8193bc0f3f6fd659c178047de6",
    "reference_validator": "sha256:48099f887ebfc9fd9857ba8cececaa8b52c1dd5a2020ccc5eca21c3120664d9a",
}
EXPECTED_BINDING_FIELDS = {
    "binding_version",
    "operator_id",
    "operator_version",
    "distribution",
    "distribution_version",
    "source_commit",
    "source_tree_digest",
    "schema_id",
    "schema_release",
    "schema_digest",
    "manifest_digest",
    "implementation_digest",
    "surface_release",
    "contract_surface",
    "certification_state",
}


def sha256_file(path: Path) -> str:
    """Return a frozen-contract digest for exact file bytes."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _rewrite_manifest(
    repository: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    rewrite_json(
        repository / "manifests" / "equant.ttr.sma.operator.json",
        mutate,
    )


def _assert_contract_error(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    expected_code: str,
) -> None:
    fixture = write_provider_repository(
        tmp_path,
        mutate=lambda repository: _rewrite_manifest(repository, mutate),
    )
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)
    assert caught.value.code == expected_code
    assert caught.value.operator_id == "equant.ttr.sma"


def test_constructs_an_exact_contract_valid_binding_from_real_artifacts(
    tmp_path: Path,
) -> None:
    fixture = write_provider_repository(tmp_path)
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        result = validate_provider_contract(submission)
        candidate = result.operators[0]

        assert result.provider == "equant-py"
        assert result.release == "1.0.0"
        assert candidate.manifest["operator_id"] == "equant.ttr.sma"
        assert candidate.binding["certification_state"] == "contract-valid"
        assert set(candidate.binding) == EXPECTED_BINDING_FIELDS
        assert candidate.binding["manifest_digest"] == sha256_file(
            candidate.manifest_path
        )
        assert candidate.binding["implementation_digest"] == sha256_file(
            candidate.implementation_artifact
        )
        assert candidate.binding["contract_surface"] == {
            name: {"release": "1.0.0", "digest": digest}
            for name, digest in EXPECTED_SURFACE_DIGESTS.items()
        }


def test_hashes_the_implementation_commit_archive_not_submission_metadata(
    tmp_path: Path,
) -> None:
    fixture = write_provider_repository(tmp_path)
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        metadata_source = submission.archive_root / "src" / "equant_ttr" / "sma.py"
        metadata_source.write_text("tampered submission metadata\n", encoding="utf-8")

        result = validate_provider_contract(submission)

    assert result.operators[0].binding["certification_state"] == "contract-valid"


def test_rejects_manifest_schema_before_standalone_semantics(tmp_path: Path) -> None:
    def invalidate_schema_and_semantics(manifest: dict[str, object]) -> None:
        del manifest["semantic_name"]
        manifest["input"]["optional_columns"] = ["close"]  # type: ignore[index]

    _assert_contract_error(
        tmp_path,
        invalidate_schema_and_semantics,
        "manifest_schema_invalid",
    )


def test_rejects_standalone_manifest_semantic_failure(tmp_path: Path) -> None:
    _assert_contract_error(
        tmp_path,
        lambda manifest: manifest["input"].update(  # type: ignore[union-attr]
            {"optional_columns": ["close"]}
        ),
        "manifest_semantics_invalid",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update({"operator_id": "equant.ttr.ema"}),
        lambda manifest: manifest.update({"operator_version": "2.0.0"}),
        lambda manifest: manifest.update({"distribution": "other-ttr"}),
        lambda manifest: manifest["implementation"].update(  # type: ignore[union-attr]
            {"package_version": "2.0.0"}
        ),
        lambda manifest: manifest["implementation"].update(  # type: ignore[union-attr]
            {"source_commit": "git-sha1:" + "1" * 40}
        ),
        lambda manifest: manifest["implementation"].update(  # type: ignore[union-attr]
            {"build_identifier": "unrelated-build"}
        ),
    ],
    ids=[
        "catalog-operator-id",
        "catalog-operator-version",
        "build-distribution",
        "build-package-version",
        "implementation-source-commit",
        "build-identifier",
    ],
)
def test_rejects_catalog_build_and_manifest_identity_mismatches(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    _assert_contract_error(tmp_path, mutate, "manifest_identity_mismatch")


@pytest.mark.parametrize(
    ("field", "expected_code"),
    [
        ("source_tree_digest", "binding_validation_failed"),
        ("implementation_digest", "manifest_identity_mismatch"),
    ],
)
def test_rejects_declared_source_tree_and_implementation_digest_mismatches(
    tmp_path: Path,
    field: str,
    expected_code: str,
) -> None:
    _assert_contract_error(
        tmp_path,
        lambda manifest: manifest["implementation"].update(  # type: ignore[union-attr]
            {field: "sha256:" + "1" * 64}
        ),
        expected_code,
    )


def test_recomputes_the_real_wheel_digest_during_binding_validation(
    tmp_path: Path,
) -> None:
    fixture = write_provider_repository(tmp_path)
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        submission.artifacts[0].wheel_path.write_bytes(b"changed after submission load")
        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)

    assert caught.value.code == "binding_validation_failed"
    assert caught.value.stage == "binding"


def test_rejects_a_manifest_and_wheel_that_disagree_with_the_build_record(
    tmp_path: Path,
) -> None:
    fixture = write_provider_repository(tmp_path)
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        replacement_wheel = b"replacement wheel bytes"
        submission.artifacts[0].wheel_path.write_bytes(replacement_wheel)
        manifest_path = submission.operators[0].manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["implementation"]["implementation_digest"] = (  # type: ignore[index]
            f"sha256:{hashlib.sha256(replacement_wheel).hexdigest()}"
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )

        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)

    assert caught.value.code == "manifest_identity_mismatch"
    assert caught.value.stage == "manifest"


@pytest.mark.parametrize("artifact", sorted(EXPECTED_SURFACE_DIGESTS))
def test_rejects_any_contract_surface_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    fixture = write_provider_repository(tmp_path / "provider-fixture")
    copied_paths: dict[str, Path] = {}
    with materialize_contract_surface() as paths:
        for name, path in paths.items():
            copied = tmp_path / "surface" / path.name
            copied.parent.mkdir(exist_ok=True)
            copied.write_bytes(path.read_bytes())
            copied_paths[name] = copied
    copied_paths[artifact].write_bytes(
        copied_paths[artifact].read_bytes() + b"\n"
    )

    @contextmanager
    def tampered_surface() -> Iterator[dict[str, Path]]:
        yield copied_paths

    monkeypatch.setattr(
        certification,
        "materialize_contract_surface",
        tampered_surface,
    )
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)

    assert caught.value.code == "binding_validation_failed"
    assert caught.value.stage == "binding"
    assert caught.value.message == "frozen contract surface validation failed"


@pytest.mark.parametrize(
    "invalid_bytes",
    [
        b"\xff",
        b'{"schema_version": 1, "schema_version": 1}',
        b"NaN",
    ],
    ids=["invalid-utf8", "duplicate-json-key", "nonstandard-number"],
)
def test_normalizes_strict_manifest_decode_failures(
    tmp_path: Path,
    invalid_bytes: bytes,
) -> None:
    fixture = write_provider_repository(tmp_path)
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        submission.operators[0].manifest_path.write_bytes(invalid_bytes)
        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)

    assert caught.value.as_dict() == {
        "status": "fail",
        "stage": "manifest",
        "code": "manifest_schema_invalid",
        "message": "operator manifest is not strict UTF-8 JSON",
        "operator_id": "equant.ttr.sma",
    }
