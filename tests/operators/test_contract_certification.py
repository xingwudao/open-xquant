"""Contract validation tests for loaded local provider submissions."""

import hashlib
import json
import os
import py_compile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import oxq.operators.certification as certification
from oxq.operators.certification import validate_provider_contract
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.resources import materialize_contract_surface
from oxq.operators.submission import load_provider_submission
from tests.operators.helpers import (
    COMPATIBILITY_ROOT,
    rewrite_json,
    write_provider_repository,
)

EXPECTED_SURFACE_DIGESTS = {
    "quant_panel_schema": "sha256:fd6fcd7f3102cdd63913644f87a154a22713c0286a6e9e1cc16e84ca6b283a9c",
    "operator_manifest_schema": "sha256:adea87a6caec3984d65d9fbaaa0ba132be76e5609ed17407de5e8b85c38bf82e",
    "operator_binding_schema": "sha256:1d0e3ed12acde2a2d0c1fe2309f9a090ea7b0f8193bc0f3f6fd659c178047de6",
    "reference_validator": "sha256:b863570a443f5dd1e8f26ab94b2b5421dd3a52331d1b8c60bbfeb88d40653524",
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


def _copy_contract_surface(root: Path) -> dict[str, Path]:
    copied_paths: dict[str, Path] = {}
    with materialize_contract_surface() as paths:
        for name, path in paths.items():
            copied = root / path.name
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(path.read_bytes())
            copied_paths[name] = copied
    return copied_paths


def _materialized_surface(paths: dict[str, Path]) -> Callable[[], object]:
    @contextmanager
    def copied_surface() -> Iterator[dict[str, Path]]:
        yield paths

    return copied_surface


def _rewrite_manifest(
    repository: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    rewrite_json(
        repository
        / COMPATIBILITY_ROOT
        / "manifests"
        / "equant.ttr.sma.operator.json",
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


def test_accepts_an_explicit_build_identifier_that_is_not_derived(
    tmp_path: Path,
) -> None:
    fixture = write_provider_repository(tmp_path)
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        assert submission.artifacts[0].build_identifier == (
            "build-20260826-equant-ttr"
        )
        result = validate_provider_contract(submission)

    implementation = result.operators[0].manifest["implementation"]
    assert implementation["build_identifier"] == "build-20260826-equant-ttr"  # type: ignore[index]


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
    copied_paths = _copy_contract_surface(tmp_path / "surface")
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


def test_executes_verified_validator_source_instead_of_a_valid_bytecode_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = write_provider_repository(tmp_path / "provider-fixture")
    copied_paths = _copy_contract_surface(tmp_path / "surface")
    validator_path = copied_paths["reference_validator"]
    verified_source = validator_path.read_bytes()
    malicious_prefix = b'raise RuntimeError("cached validator executed")\n'
    malicious_source = malicious_prefix + b"#" * (
        len(verified_source) - len(malicious_prefix)
    )
    assert len(malicious_source) == len(verified_source)
    timestamp = 1_700_000_000
    validator_path.write_bytes(malicious_source)
    os.utime(validator_path, (timestamp, timestamp))
    py_compile.compile(str(validator_path), doraise=True)
    validator_path.write_bytes(verified_source)
    os.utime(validator_path, (timestamp, timestamp))
    monkeypatch.setattr(
        certification,
        "materialize_contract_surface",
        _materialized_surface(copied_paths),
    )

    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        result = validate_provider_contract(submission)

    assert result.operators[0].binding["certification_state"] == "contract-valid"


def test_binding_validation_uses_private_snapshot_after_materialized_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = write_provider_repository(tmp_path / "provider-fixture")
    copied_paths = _copy_contract_surface(tmp_path / "surface")
    monkeypatch.setattr(
        certification,
        "materialize_contract_surface",
        _materialized_surface(copied_paths),
    )
    validate_semantics = certification._validate_manifest_semantics

    def replace_materialized_path(*args: object) -> None:
        validate_semantics(*args)  # type: ignore[arg-type]
        copied_paths["quant_panel_schema"].write_bytes(b"swapped after verification")

    monkeypatch.setattr(
        certification,
        "_validate_manifest_semantics",
        replace_materialized_path,
    )

    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        result = validate_provider_contract(submission)

    assert result.operators[0].binding["certification_state"] == "contract-valid"


def test_private_snapshot_is_read_only_during_binding_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = write_provider_repository(tmp_path)
    validate_binding = certification._validate_binding_semantics

    def assert_snapshot_is_read_only(*args: object) -> None:
        snapshot_paths = args[-1]
        assert isinstance(snapshot_paths, dict)
        with pytest.raises(OSError):
            snapshot_paths["quant_panel_schema"].write_bytes(b"replacement")
        validate_binding(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(
        certification,
        "_validate_binding_semantics",
        assert_snapshot_is_read_only,
    )
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        result = validate_provider_contract(submission)

    assert result.operators[0].binding["certification_state"] == "contract-valid"


def test_reads_each_materialized_surface_artifact_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = write_provider_repository(tmp_path / "provider-fixture")
    copied_paths = _copy_contract_surface(tmp_path / "surface")
    reads = {path: 0 for path in copied_paths.values()}
    read_bytes = Path.read_bytes

    def tracked_read_bytes(path: Path) -> bytes:
        if path in reads:
            reads[path] += 1
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    monkeypatch.setattr(
        certification,
        "materialize_contract_surface",
        _materialized_surface(copied_paths),
    )
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        validate_provider_contract(submission)

    assert set(reads.values()) == {1}


@pytest.mark.parametrize("phase", ["enter", "cleanup"])
def test_normalizes_contract_surface_materialization_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    fixture = write_provider_repository(tmp_path)

    @contextmanager
    def broken_surface() -> Iterator[dict[str, Path]]:
        if phase == "enter":
            raise OSError("raw materialization failure")
        with materialize_contract_surface() as paths:
            yield paths
        raise OSError("raw cleanup failure")

    monkeypatch.setattr(
        certification,
        "materialize_contract_surface",
        broken_surface,
    )
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)

    assert caught.value.as_dict() == {
        "status": "fail",
        "stage": "binding",
        "code": "binding_validation_failed",
        "message": "frozen contract resources are unavailable",
    }


def test_preserves_existing_certification_error_from_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = write_provider_repository(tmp_path)
    expected = OperatorCertificationError(
        "sentinel",
        "already normalized",
        stage="resource",
    )

    @contextmanager
    def broken_surface() -> Iterator[dict[str, Path]]:
        raise expected
        yield {}

    monkeypatch.setattr(
        certification,
        "materialize_contract_surface",
        broken_surface,
    )
    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)

    assert caught.value is expected


def test_normalizes_schema_parser_recursion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = write_provider_repository(tmp_path)

    def recurse(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RecursionError("raw schema parser failure")

    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        monkeypatch.setattr(certification.json, "loads", recurse)
        with pytest.raises(OperatorCertificationError) as caught:
            validate_provider_contract(submission)

    assert caught.value.as_dict() == {
        "status": "fail",
        "stage": "manifest",
        "code": "manifest_schema_invalid",
        "message": "operator manifest schema is unavailable",
    }


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
