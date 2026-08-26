"""Atomic publication and lookup tests for operator certifications."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import oxq.operators.registry as registry_module
from oxq.operators.certification import (
    _issue_research_certification,
    certify_provider,
)
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BaselineCase,
    BaselineResult,
    BuildArtifact,
    ContractCandidate,
    ContractCertification,
    ResearchCertification,
)
from oxq.operators.registry import CertificationRegistry, publish_certification
from oxq.operators.resources import materialize_certification_profile
from oxq.operators.submission import load_provider_submission
from tests.operators.helpers import _manifest as _valid_manifest
from tests.operators.test_baseline_runner import _write_certifiable_provider

_SURFACE_DIGESTS = {
    "quant_panel_schema": "sha256:fd6fcd7f3102cdd63913644f87a154a22713c0286a6e9e1cc16e84ca6b283a9c",
    "operator_manifest_schema": "sha256:adea87a6caec3984d65d9fbaaa0ba132be76e5609ed17407de5e8b85c38bf82e",
    "operator_binding_schema": "sha256:1d0e3ed12acde2a2d0c1fe2309f9a090ea7b0f8193bc0f3f6fd659c178047de6",
    "reference_validator": "sha256:4758be90c907f636f6751174d6a9a7f0e1b9422e15ee873a77ba6d599d7cd7bc",
}


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _source_tree_digest(root: Path, source_files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(source_files):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256((root / relative_path).read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _binding(
    operator_id: str = "equant.ttr.sma",
    operator_version: str = "1.0.0",
    *,
    state: str = "research-certified",
) -> dict[str, object]:
    return {
        "binding_version": 1,
        "operator_id": operator_id,
        "operator_version": operator_version,
        "distribution": "equant-ttr",
        "distribution_version": "1.0.0",
        "source_commit": "git-sha1:" + "a" * 40,
        "source_tree_digest": "sha256:" + "b" * 64,
        "schema_id": ("https://open-xquant.dev/contracts/quant-operators/operator-manifest-v1.schema.json"),
        "schema_release": "1.0.0",
        "schema_digest": _SURFACE_DIGESTS["operator_manifest_schema"],
        "manifest_digest": "sha256:" + "d" * 64,
        "implementation_digest": "sha256:" + "e" * 64,
        "surface_release": "1.0.0",
        "contract_surface": {name: {"release": "1.0.0", "digest": digest} for name, digest in _SURFACE_DIGESTS.items()},
        "certification_state": state,
    }


def _result(
    tmp_path: Path,
    *,
    provider: str = "equant-py",
    release: str = "1.0.0",
    operator_id: str = "equant.ttr.sma",
    operator_version: str = "1.0.0",
    state: str = "research-certified",
    status: str = "passed",
) -> ResearchCertification:
    source_root = tmp_path / f"{provider}-{release}-source"
    source_root.mkdir(parents=True, exist_ok=True)
    source_files = ["src/equant_ttr/sma.py"]
    source_file = source_root / source_files[0]
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("SECRET_PROVIDER_SOURCE = True\n")
    source_tree_digest = _source_tree_digest(source_root, source_files)
    wheel = tmp_path / f"{provider}-{release}.whl"
    wheel.write_bytes(b"exact provider wheel")
    wheel_digest = _sha256(wheel.read_bytes())
    manifest = _valid_manifest(
        "a" * 40,
        source_tree_digest,
        wheel_digest,
        "registry-test-build",
    )
    manifest["operator_id"] = operator_id
    manifest["operator_version"] = operator_version
    manifest_path = source_root / "operator.json"
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    candidate = ContractCandidate(
        manifest=manifest,
        binding={
            **_binding(operator_id, operator_version, state=state),
            "source_tree_digest": source_tree_digest,
            "manifest_digest": _sha256(manifest_bytes),
            "implementation_digest": wheel_digest,
        },
        manifest_path=manifest_path,
        implementation_artifact=wheel,
    )
    baseline = BaselineCase(
        case_id="sma-window-3",
        operator_id=operator_id,
        operator_version=operator_version,
        parameters={"window": 3},
        input={},
        expected={"sma_3": [1.0]},
        tolerance={"absolute": 0.0, "relative": 0.0},
    )
    baseline_result = BaselineResult(
        operator_id=operator_id,
        operator_version=operator_version,
        case_id="sma-window-3",
        status=status,
    )
    artifact = BuildArtifact(
        distribution="equant-ttr",
        version="1.0.0",
        filename=wheel.name,
        role="implementation",
        build_identifier="registry-test-build",
        digest=wheel_digest,
        wheel_path=wheel,
    )
    return _issue_research_certification(
        ResearchCertification(
            provider=provider,
            release=release,
            submission_commit="git-sha1:" + "f" * 40,
            source_commit="git-sha1:" + "a" * 40,
            source_root=source_root,
            operators=(candidate,),
            artifacts=(artifact,),
            baseline_cases=(baseline,),
            baseline_results=(baseline_result,),
        )
    )


def _assert_canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    parsed = cast(dict[str, object], json.loads(raw))
    assert raw == (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode()
    return parsed


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _with_manifest(
    result: ResearchCertification,
    manifest: dict[str, object],
) -> ResearchCertification:
    candidate = result.operators[0]
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    candidate.manifest_path.write_bytes(manifest_bytes)
    binding = dict(candidate.binding)
    binding["manifest_digest"] = _sha256(manifest_bytes)
    return _issue_research_certification(
        replace(
            result,
            operators=(replace(candidate, manifest=manifest, binding=binding),),
        )
    )


def _with_changed_manifest(
    result: ResearchCertification,
    marker: str,
) -> ResearchCertification:
    manifest = dict(result.operators[0].manifest)
    manifest["semantic_name"] = marker
    return _with_manifest(result, manifest)


def test_publishes_exact_canonical_layout_and_looks_up_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / ".open-xquant" / "certifications"
    result = _result(tmp_path)
    monkeypatch.setattr(
        registry_module,
        "_utc_now",
        lambda: "2026-08-26T12:34:56.123456Z",
    )

    published = publish_certification(result, output)

    release_dir = output / "equant-py" / "1.0.0"
    assert published.release_dir == release_dir
    assert sorted(path.relative_to(release_dir).as_posix() for path in release_dir.rglob("*")) == [
        "bindings",
        "bindings/equant.ttr.sma@1.0.0.binding.json",
        "certification-record.json",
        "registry-entry.json",
    ]
    binding_path = release_dir / "bindings/equant.ttr.sma@1.0.0.binding.json"
    binding = _assert_canonical_json(binding_path)
    record = _assert_canonical_json(release_dir / "certification-record.json")
    entry = _assert_canonical_json(release_dir / "registry-entry.json")
    assert "binding_digest" not in binding
    assert published.record["certified_at"] == record["certified_at"]
    assert record["certified_at"] == "2026-08-26T12:34:56.123456Z"
    assert record["certifier"] == "open-xquant-local"
    assert record["submission_commit"] == "git-sha1:" + "f" * 40
    assert record["source_commit"] == "git-sha1:" + "a" * 40
    assert record["artifacts"] == [
        {
            "build_identifier": "registry-test-build",
            "digest": result.artifacts[0].digest,
            "distribution": "equant-ttr",
            "filename": result.artifacts[0].filename,
            "role": "implementation",
            "version": "1.0.0",
        }
    ]
    assert record["operators"] == [
        {
            "baseline_cases": [{"case_id": "sma-window-3", "status": "passed"}],
            "binding_digest": _sha256(binding_path.read_bytes()),
            "implementation_digest": result.artifacts[0].digest,
            "manifest_digest": result.operators[0].binding["manifest_digest"],
            "operator_id": "equant.ttr.sma",
            "operator_version": "1.0.0",
        }
    ]
    with materialize_certification_profile() as paths:
        schema = json.loads(paths["certification_record"].read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    assert entry["submission_commit"] == result.submission_commit
    assert CertificationRegistry(output).get("equant.ttr.sma", "1.0.0") == binding
    assert b"SECRET_PROVIDER_SOURCE" not in b"".join(_tree_bytes(release_dir).values())


def test_real_submission_certification_publishes_and_resolves(tmp_path: Path) -> None:
    fixture = _write_certifiable_provider(tmp_path, expected=[None, None, 2.0])
    output = tmp_path / "certifications"

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        certified = certify_provider(submission)
        assert certified.submission_commit == f"git-sha1:{fixture.submission_commit}"
        assert certified.source_commit == f"git-sha1:{fixture.implementation_commit}"
        published = publish_certification(certified, output)

    binding = CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")
    assert published.record["submission_commit"] == certified.submission_commit
    assert published.record["source_commit"] == certified.source_commit
    assert binding is not None
    assert binding["certification_state"] == "research-certified"


def test_rejects_a_directly_constructed_research_certification_clone(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(tmp_path, expected=[None, None, 2.0])
    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        certified = certify_provider(submission)
        forged = ResearchCertification(
            provider=certified.provider,
            release=certified.release,
            submission_commit=certified.submission_commit,
            source_commit=certified.source_commit,
            source_root=certified.source_root,
            operators=certified.operators,
            artifacts=certified.artifacts,
            baseline_cases=certified.baseline_cases,
            baseline_results=certified.baseline_results,
        )

        with pytest.raises(OperatorCertificationError) as caught:
            publish_certification(forged, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"


def test_rejects_mutated_contents_of_an_issued_certification(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    object.__setattr__(result, "release", "1.0.1")

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"


def test_failure_before_atomic_replace_leaves_no_release_or_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "certifications"
    result = _result(tmp_path)

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("injected final publication failure")

    monkeypatch.setattr(registry_module.os, "replace", fail_replace)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, output)

    assert caught.value.code == "certification_publish_failed"
    assert caught.value.stage == "registry"
    assert not (output / "equant-py" / "1.0.0").exists()
    assert not list(output.rglob("registry-entry.json"))
    assert not list(output.rglob("*.staging-*"))


def test_post_rename_directory_fsync_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "certifications"
    provider_dir = output / "equant-py"
    real_fsync = registry_module._fsync_directory
    real_replace = registry_module.os.replace
    renamed = False
    post_rename_attempts = 0

    def observed_replace(source: Path, target: Path) -> None:
        nonlocal renamed
        real_replace(source, target)
        renamed = True

    def fail_after_rename(path: Path) -> None:
        nonlocal post_rename_attempts
        if renamed and path == provider_dir:
            post_rename_attempts += 1
            if post_rename_attempts == 1:
                raise OSError("injected post-rename fsync failure")
        real_fsync(path)

    monkeypatch.setattr(registry_module.os, "replace", observed_replace)
    monkeypatch.setattr(registry_module, "_fsync_directory", fail_after_rename)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(_result(tmp_path), output)

    assert caught.value.code == "certification_publish_failed"
    assert (provider_dir / "1.0.0" / "registry-entry.json").is_file()
    retried = publish_certification(_result(tmp_path), output)
    assert retried.release_dir == provider_dir / "1.0.0"
    assert post_rename_attempts == 2


def test_identical_republication_returns_original_without_new_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "certifications"
    result = _result(tmp_path)
    monkeypatch.setattr(
        registry_module,
        "_utc_now",
        lambda: "2026-08-26T01:02:03Z",
    )
    first = publish_certification(result, output)
    original = _tree_bytes(first.release_dir)

    def timestamp_must_not_be_generated() -> str:
        raise AssertionError("idempotent publication generated a new timestamp")

    monkeypatch.setattr(registry_module, "_utc_now", timestamp_must_not_be_generated)
    second = publish_certification(result, output)

    assert second.record == first.record
    assert _tree_bytes(first.release_dir) == original


def test_conflicting_republication_preserves_original_release(tmp_path: Path) -> None:
    output = tmp_path / "certifications"
    result = _result(tmp_path)
    first = publish_certification(result, output)
    original = _tree_bytes(first.release_dir)
    conflict = _with_changed_manifest(result, "conflicting-manifest")

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(conflict, output)

    assert caught.value.code == "certification_conflict"
    assert _tree_bytes(first.release_dir) == original


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("contract-valid", "passed"),
        ("runtime-certified", "passed"),
        ("failed", "passed"),
        ("research-certified", "failed"),
    ],
)
def test_rejects_non_research_or_failed_results(
    tmp_path: Path,
    state: str,
    status: str,
) -> None:
    result = _result(tmp_path, state=state, status=status)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"
    assert not (tmp_path / "certifications").exists()


def test_rejects_contract_certification_and_empty_research_result(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    contract = ContractCertification(
        provider=result.provider,
        release=result.release,
        submission_commit=result.submission_commit,
        source_commit=result.source_commit,
        source_root=result.source_root,
        operators=result.operators,
        artifacts=result.artifacts,
        baseline_cases=result.baseline_cases,
    )
    empty = replace(result, operators=(), baseline_cases=(), baseline_results=())

    for invalid in (contract, empty):
        with pytest.raises(OperatorCertificationError) as caught:
            publish_certification(
                cast(ResearchCertification, invalid),
                tmp_path / "certifications",
            )
        assert caught.value.code == "certification_input_invalid"


@pytest.mark.parametrize(("provider", "release"), [("../escape", "1.0.0"), ("equant-py", "../escape")])
def test_rejects_provider_release_path_escape(
    tmp_path: Path,
    provider: str,
    release: str,
) -> None:
    result = _result(tmp_path, provider=provider, release=release)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"
    assert not (tmp_path / "escape").exists()


def test_rejects_duplicate_operator_identity(tmp_path: Path) -> None:
    result = _result(tmp_path)
    duplicate = replace(
        result,
        operators=(result.operators[0], result.operators[0]),
    )

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(duplicate, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "runtime-only",
        "unrelated-implementation",
        "duplicate-digest",
        "filename-path-mismatch",
    ],
)
def test_rejects_invalid_artifact_sets(tmp_path: Path, mutation: str) -> None:
    result = _result(tmp_path)
    artifact = result.artifacts[0]
    if mutation == "empty":
        artifacts: tuple[BuildArtifact, ...] = ()
    elif mutation == "runtime-only":
        artifacts = (replace(artifact, role="runtime-dependency"),)
    elif mutation == "unrelated-implementation":
        artifacts = (replace(artifact, distribution="unrelated-wheel"),)
    elif mutation == "filename-path-mismatch":
        artifacts = (replace(artifact, filename="declared-other.whl"),)
    else:
        duplicate_path = tmp_path / "duplicate.whl"
        duplicate_path.write_bytes(artifact.wheel_path.read_bytes())
        artifacts = (
            artifact,
            replace(
                artifact,
                filename=duplicate_path.name,
                role="runtime-dependency",
                build_identifier="duplicate-build",
                wheel_path=duplicate_path,
            ),
        )
    invalid = replace(result, artifacts=artifacts)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(invalid, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"


def test_allows_release_artifacts_to_share_one_build_invocation_identifier(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    dependency_path = tmp_path / "equant_core-1.0.0-py3-none-any.whl"
    dependency_path.write_bytes(b"exact shared-build dependency wheel")
    dependency = BuildArtifact(
        distribution="equant-core",
        version="1.0.0",
        filename=dependency_path.name,
        role="runtime-dependency",
        build_identifier=result.artifacts[0].build_identifier,
        digest=_sha256(dependency_path.read_bytes()),
        wheel_path=dependency_path,
    )
    output = tmp_path / "certifications"

    publication = publish_certification(
        _issue_research_certification(replace(result, artifacts=(*result.artifacts, dependency))),
        output,
    )

    assert [artifact["build_identifier"] for artifact in publication.record["artifacts"]] == [
        "registry-test-build",
        "registry-test-build",
    ]
    binding = CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")
    assert binding is not None
    assert binding["certification_state"] == "research-certified"


def test_rejects_unrelated_extra_implementation_artifact(tmp_path: Path) -> None:
    result = _result(tmp_path)
    unrelated_path = tmp_path / "unrelated.whl"
    unrelated_path.write_bytes(b"unrelated implementation")
    unrelated = BuildArtifact(
        distribution="unrelated-wheel",
        version="2.0.0",
        filename=unrelated_path.name,
        role="implementation",
        build_identifier="unrelated-build",
        digest=_sha256(unrelated_path.read_bytes()),
        wheel_path=unrelated_path,
    )

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(
            replace(result, artifacts=(*result.artifacts, unrelated)),
            tmp_path / "certifications",
        )

    assert caught.value.code == "certification_input_invalid"


def test_rejects_manifest_artifact_mismatch_and_changed_wheel_bytes(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    implementation = dict(result.operators[0].manifest["implementation"])  # type: ignore[arg-type]
    implementation["build_identifier"] = "different-build"
    manifest = dict(result.operators[0].manifest)
    manifest["implementation"] = implementation
    mismatched = replace(
        result,
        operators=(replace(result.operators[0], manifest=manifest),),
    )
    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(mismatched, tmp_path / "manifest-output")
    assert caught.value.code == "certification_input_invalid"

    result.artifacts[0].wheel_path.write_bytes(b"changed after certification")
    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, tmp_path / "wheel-output")
    assert caught.value.code == "certification_input_invalid"


@pytest.mark.parametrize(
    "tamper",
    [
        "manifest-bytes",
        "declared-manifest-digest",
        "retained-manifest",
        "source-tree-bytes",
        "non-normalized-source-path",
        "missing-manifest",
        "missing-source",
    ],
)
def test_rejects_manifest_and_source_tree_provenance_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    result = _result(tmp_path)
    candidate = result.operators[0]
    if tamper == "manifest-bytes":
        candidate.manifest_path.write_bytes(candidate.manifest_path.read_bytes() + b" ")
    elif tamper == "declared-manifest-digest":
        binding = dict(candidate.binding)
        binding["manifest_digest"] = "sha256:" + "9" * 64
        result = replace(result, operators=(replace(candidate, binding=binding),))
    elif tamper == "retained-manifest":
        retained = dict(candidate.manifest)
        retained["semantic_name"] = "forged-retained-value"
        result = replace(result, operators=(replace(candidate, manifest=retained),))
    elif tamper == "source-tree-bytes":
        (result.source_root / "src/equant_ttr/sma.py").write_text("SECRET_PROVIDER_SOURCE = False\n")
    elif tamper == "non-normalized-source-path":
        manifest = dict(candidate.manifest)
        implementation = dict(manifest["implementation"])  # type: ignore[arg-type]
        source_files = ["src//equant_ttr/sma.py"]
        implementation["source_files"] = source_files
        implementation["source_tree_digest"] = _source_tree_digest(
            result.source_root,
            source_files,
        )
        manifest["implementation"] = implementation
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        candidate.manifest_path.write_bytes(manifest_bytes)
        binding = dict(candidate.binding)
        binding["manifest_digest"] = _sha256(manifest_bytes)
        binding["source_tree_digest"] = implementation["source_tree_digest"]
        result = replace(
            result,
            operators=(replace(candidate, manifest=manifest, binding=binding),),
        )
    elif tamper == "missing-manifest":
        candidate.manifest_path.unlink()
    else:
        (result.source_root / "src/equant_ttr/sma.py").unlink()

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"
    assert caught.value.stage == "registry"


@pytest.mark.parametrize("invalidity", ["schema", "semantics", "binding"])
def test_revalidates_full_frozen_manifest_and_binding_contract(
    tmp_path: Path,
    invalidity: str,
) -> None:
    result = _result(tmp_path)
    candidate = result.operators[0]
    if invalidity == "binding":
        binding = dict(candidate.binding)
        binding["schema_digest"] = "sha256:" + "9" * 64
        result = replace(result, operators=(replace(candidate, binding=binding),))
    else:
        manifest = json.loads(json.dumps(candidate.manifest))
        if invalidity == "schema":
            del manifest["semantic_name"]
        else:
            manifest["input"]["optional_columns"] = ["close"]
        result = _with_manifest(result, manifest)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, tmp_path / "certifications")

    assert caught.value.code == "certification_input_invalid"
    assert caught.value.stage == "registry"


def test_source_tree_read_cannot_escape_through_parent_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result(tmp_path / "result")
    candidate = result.operators[0]
    relative_source = "src/equant_ttr/sma.py"
    source_parent = result.source_root / "src/equant_ttr"
    outside_root = tmp_path / "outside"
    outside_source = outside_root / relative_source
    outside_source.parent.mkdir(parents=True)
    outside_source.write_text("SECRET_PROVIDER_SOURCE = OUTSIDE\n")
    outside_digest = _source_tree_digest(outside_root, [relative_source])
    manifest = dict(candidate.manifest)
    implementation = dict(manifest["implementation"])  # type: ignore[arg-type]
    implementation["source_tree_digest"] = outside_digest
    manifest["implementation"] = implementation
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    candidate.manifest_path.write_bytes(manifest_bytes)
    binding = dict(candidate.binding)
    binding["manifest_digest"] = _sha256(manifest_bytes)
    binding["source_tree_digest"] = outside_digest
    result = _issue_research_certification(
        replace(
            result,
            operators=(replace(candidate, manifest=manifest, binding=binding),),
        )
    )
    real_open = registry_module.os.open
    swapped = False

    def swap_parent_before_leaf_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and Path(path).name == "sma.py":
            swapped = True
            source_parent.rename(source_parent.with_name("equant_ttr-original"))
            source_parent.symlink_to(outside_source.parent, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(registry_module.os, "open", swap_parent_before_leaf_open)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(result, tmp_path / "certifications")

    assert swapped
    assert caught.value.code == "certification_input_invalid"


def test_source_root_descriptor_closes_when_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    real_open = registry_module.os.open
    real_fstat = registry_module.os.fstat
    opened: list[int] = []

    def observed_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def fail_fstat(descriptor: int) -> os.stat_result:
        del descriptor
        raise OSError("injected fstat failure")

    monkeypatch.setattr(registry_module.os, "open", observed_open)
    monkeypatch.setattr(registry_module.os, "fstat", fail_fstat)

    with pytest.raises(OSError, match="injected fstat failure"):
        registry_module._open_directory_no_follow(source_root)

    assert len(opened) == 1
    with pytest.raises(OSError):
        real_fstat(opened[0])


def test_descriptor_cleanup_attempts_every_close_after_one_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[int] = []

    def flaky_close(descriptor: int) -> None:
        attempted.append(descriptor)
        if descriptor == 22:
            raise OSError("injected close failure")

    monkeypatch.setattr(registry_module.os, "close", flaky_close)

    with pytest.raises(OSError, match="injected close failure"):
        registry_module._close_descriptors([11, 22, 33])

    assert attempted == [11, 22, 33]


def test_windows_source_digest_and_directory_sync_avoid_directory_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_path = source_root / "src/equant_ttr/sma.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"provider source\n")

    monkeypatch.setattr(registry_module, "_is_windows", lambda: True, raising=False)
    monkeypatch.setattr(
        registry_module,
        "_read_relative_regular_file_windows",
        lambda root, parts: root.joinpath(*parts).read_bytes(),
    )
    monkeypatch.setattr(
        registry_module,
        "_open_windows_directory_handle",
        lambda path: 44,
    )
    monkeypatch.setattr(
        registry_module,
        "_close_windows_handle",
        lambda handle: None,
    )

    def reject_directory_open(*args: object, **kwargs: object) -> int:
        raise AssertionError(f"Windows directory descriptor opened: {args!r} {kwargs!r}")

    monkeypatch.setattr(registry_module.os, "open", reject_directory_open)

    assert registry_module._source_tree_digest(
        source_root,
        ["src/equant_ttr/sma.py"],
    ) == _source_tree_digest(source_root, ["src/equant_ttr/sma.py"])
    registry_module._fsync_directory(source_root)


def test_windows_directory_replace_uses_write_through_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "staging"
    target = tmp_path / "release"
    observed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(registry_module, "_is_windows", lambda: True, raising=False)
    monkeypatch.setattr(
        registry_module,
        "_move_file_write_through",
        lambda left, right: observed.append((left, right)),
    )

    registry_module._replace_directory(source, target)

    assert observed == [(source, target)]


def test_windows_source_digest_rejects_reparse_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sma.py").write_bytes(b"outside\n")
    source_root.mkdir()
    (source_root / "src").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(registry_module, "_is_windows", lambda: True, raising=False)

    def reject_reparse_chain(root: Path, parts: list[str]) -> list[int]:
        current = root
        for part in parts:
            current /= part
            if current.is_symlink():
                raise OSError(f"directory is a Windows reparse point: {current}")
        return []

    monkeypatch.setattr(
        registry_module,
        "_pin_windows_directory_chain",
        reject_reparse_chain,
    )

    with pytest.raises(OSError, match="reparse"):
        registry_module._source_tree_digest(source_root, ["src/sma.py"])


def test_windows_source_digest_rejects_reparse_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sma.py").write_bytes(b"outside\n")
    source_root = tmp_path / "source"
    source_root.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(registry_module, "_is_windows", lambda: True, raising=False)
    monkeypatch.setattr(
        registry_module,
        "_pin_windows_directory_chain",
        lambda root, parts: (_ for _ in ()).throw(OSError(f"directory is a Windows reparse point: {root}")),
    )

    with pytest.raises(OSError, match="reparse"):
        registry_module._source_tree_digest(source_root, ["sma.py"])


def test_windows_source_digest_rejects_final_handle_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_path = source_root / "sma.py"
    source_root.mkdir()
    source_path.write_bytes(b"inside\n")
    outside_path = tmp_path / "outside.py"
    outside_path.write_bytes(b"outside\n")
    monkeypatch.setattr(registry_module, "_is_windows", lambda: True, raising=False)
    monkeypatch.setattr(
        registry_module,
        "_pin_windows_directory_chain",
        lambda root, parts: [11],
    )
    monkeypatch.setattr(
        registry_module,
        "_open_windows_file_handle",
        lambda path: 22,
    )
    monkeypatch.setattr(
        registry_module,
        "_windows_handle_attributes",
        lambda handle: 0,
    )
    monkeypatch.setattr(
        registry_module,
        "_windows_final_path",
        lambda handle: source_root if handle == 11 else outside_path,
    )
    monkeypatch.setattr(
        registry_module,
        "_close_windows_handle",
        lambda handle: None,
    )

    with pytest.raises(OSError, match="escaped"):
        registry_module._source_tree_digest(source_root, ["sma.py"])


def test_allows_multiple_operators_bound_to_one_implementation_artifact(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    first = result.operators[0]
    second_manifest = dict(first.manifest)
    second_manifest["operator_id"] = "equant.ttr.ema"
    second_manifest_path = result.source_root / "ema.operator.json"
    second_manifest_bytes = (json.dumps(second_manifest, sort_keys=True) + "\n").encode()
    second_manifest_path.write_bytes(second_manifest_bytes)
    second_binding = dict(first.binding)
    second_binding["operator_id"] = "equant.ttr.ema"
    second_binding["manifest_digest"] = _sha256(second_manifest_bytes)
    second = replace(
        first,
        manifest=second_manifest,
        binding=second_binding,
        manifest_path=second_manifest_path,
    )
    second_case = replace(
        result.baseline_cases[0],
        case_id="ema-window-3",
        operator_id="equant.ttr.ema",
        expected={"ema_3": [1.0]},
    )
    second_pass = replace(
        result.baseline_results[0],
        case_id="ema-window-3",
        operator_id="equant.ttr.ema",
    )
    shared = _issue_research_certification(
        replace(
            result,
            operators=(first, second),
            baseline_cases=(*result.baseline_cases, second_case),
            baseline_results=(*result.baseline_results, second_pass),
        )
    )
    output = tmp_path / "certifications"

    publish_certification(shared, output)

    assert CertificationRegistry(output).get("equant.ttr.sma", "1.0.0") is not None
    assert CertificationRegistry(output).get("equant.ttr.ema", "1.0.0") is not None


def test_rejects_passed_case_identity_not_declared_by_baseline(tmp_path: Path) -> None:
    result = _result(tmp_path)
    invented = replace(result.baseline_results[0], case_id="invented-case")

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(
            replace(result, baseline_results=(invented,)),
            tmp_path / "certifications",
        )

    assert caught.value.code == "certification_input_invalid"


def test_rejects_invalid_runtime_field_types_with_stable_error(tmp_path: Path) -> None:
    result = _result(tmp_path)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(
            replace(result, provider=cast(str, 1)),
            tmp_path / "certifications",
        )

    assert caught.value.code == "certification_input_invalid"
    assert caught.value.stage == "registry"


def test_registry_rejects_corrupt_entry_and_path_escape(tmp_path: Path) -> None:
    output = tmp_path / "certifications"
    release = publish_certification(_result(tmp_path), output).release_dir
    entry_path = release / "registry-entry.json"
    entry = json.loads(entry_path.read_text())
    entry["operators"][0]["binding"] = "../../outside.binding.json"
    entry_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")

    with pytest.raises(OperatorCertificationError) as caught:
        CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")

    assert caught.value.code == "registry_invalid"
    assert caught.value.stage == "registry"


@pytest.mark.parametrize("operation", ["lookup", "republish"])
def test_normalizes_deeply_nested_registry_json(
    tmp_path: Path,
    operation: str,
) -> None:
    output = tmp_path / "certifications"
    result = _result(tmp_path)
    release = publish_certification(result, output).release_dir
    nested = "[" * 10000 + "0" + "]" * 10000
    (release / "registry-entry.json").write_text(
        f'{{"nested":{nested}}}',
        encoding="utf-8",
    )

    with pytest.raises(OperatorCertificationError) as caught:
        if operation == "lookup":
            CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")
        else:
            publish_certification(result, output)

    expected = "registry_invalid" if operation == "lookup" else "certification_conflict"
    assert caught.value.code == expected


@pytest.mark.parametrize("field", ["submission_commit", "artifacts"])
def test_registry_rejects_entry_provenance_that_differs_from_record(
    tmp_path: Path,
    field: str,
) -> None:
    output = tmp_path / "certifications"
    release = publish_certification(_result(tmp_path), output).release_dir
    entry_path = release / "registry-entry.json"
    entry = json.loads(entry_path.read_text())
    if field == "submission_commit":
        entry[field] = "git-sha1:" + "9" * 40
    else:
        entry[field][0]["digest"] = "sha256:" + "9" * 64
    _write_canonical_json(entry_path, entry)

    with pytest.raises(OperatorCertificationError) as caught:
        CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")

    assert caught.value.code == "registry_invalid"


def test_registry_rejects_binding_source_commit_rewrite(tmp_path: Path) -> None:
    output = tmp_path / "certifications"
    release = publish_certification(_result(tmp_path), output).release_dir
    binding_path = release / "bindings/equant.ttr.sma@1.0.0.binding.json"
    record_path = release / "certification-record.json"
    entry_path = release / "registry-entry.json"
    binding = json.loads(binding_path.read_text())
    record = json.loads(record_path.read_text())
    entry = json.loads(entry_path.read_text())
    binding["source_commit"] = "git-sha1:" + "9" * 40
    _write_canonical_json(binding_path, binding)
    binding_digest = _sha256(binding_path.read_bytes())
    record["operators"][0]["binding_digest"] = binding_digest
    _write_canonical_json(record_path, record)
    entry["operators"][0]["binding_digest"] = binding_digest
    entry["certification_record_digest"] = _sha256(record_path.read_bytes())
    _write_canonical_json(entry_path, entry)

    with pytest.raises(OperatorCertificationError) as caught:
        CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")

    assert caught.value.code == "registry_invalid"


@pytest.mark.parametrize("state", ["runtime-certified", "ml-certified"])
def test_registry_rejects_binding_state_that_disagrees_with_record(
    tmp_path: Path,
    state: str,
) -> None:
    output = tmp_path / "certifications"
    release = publish_certification(_result(tmp_path), output).release_dir
    binding_path = release / "bindings/equant.ttr.sma@1.0.0.binding.json"
    record_path = release / "certification-record.json"
    entry_path = release / "registry-entry.json"
    binding = json.loads(binding_path.read_text())
    record = json.loads(record_path.read_text())
    entry = json.loads(entry_path.read_text())
    binding["certification_state"] = state
    _write_canonical_json(binding_path, binding)
    binding_digest = _sha256(binding_path.read_bytes())
    record["operators"][0]["binding_digest"] = binding_digest
    _write_canonical_json(record_path, record)
    entry["operators"][0]["binding_digest"] = binding_digest
    entry["certification_record_digest"] = _sha256(record_path.read_bytes())
    _write_canonical_json(entry_path, entry)

    with pytest.raises(OperatorCertificationError) as caught:
        CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")

    assert caught.value.code == "registry_invalid"


def test_registry_rejects_cross_release_identity_collision(tmp_path: Path) -> None:
    output = tmp_path / "certifications"
    publish_certification(_result(tmp_path / "one"), output)
    publish_certification(
        _result(tmp_path / "two", provider="second-provider", release="2.0.0"),
        output,
    )

    with pytest.raises(OperatorCertificationError) as caught:
        CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")

    assert caught.value.code == "registry_identity_collision"


@pytest.mark.parametrize(
    "schema_loader",
    ["_snapshot_contract_surface", "_certification_record_schema"],
)
def test_normalizes_unavailable_publication_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_loader: str,
) -> None:
    def unavailable() -> object:
        raise OperatorCertificationError(
            "binding_validation_failed",
            "injected schema failure",
            stage="binding",
        )

    monkeypatch.setattr(registry_module, schema_loader, unavailable)

    with pytest.raises(OperatorCertificationError) as caught:
        publish_certification(_result(tmp_path), tmp_path / "certifications")

    assert caught.value.code == "certification_publish_failed"
    assert caught.value.stage == "registry"
    assert not list((tmp_path / "certifications").rglob("registry-entry.json"))


def test_normalizes_unavailable_registry_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "certifications"
    publish_certification(_result(tmp_path), output)

    def unavailable() -> object:
        raise OperatorCertificationError(
            "binding_validation_failed",
            "injected schema failure",
            stage="binding",
        )

    monkeypatch.setattr(registry_module, "_binding_schema", unavailable)

    with pytest.raises(OperatorCertificationError) as caught:
        CertificationRegistry(output).get("equant.ttr.sma", "1.0.0")

    assert caught.value.code == "registry_invalid"
    assert caught.value.stage == "registry"


def test_concurrent_identical_publications_converge_on_one_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "certifications"
    result = _result(tmp_path)
    barrier = threading.Barrier(2)

    monkeypatch.setattr(
        registry_module,
        "_utc_now",
        lambda: "2026-08-26T02:03:04Z",
    )
    published: list[object] = []
    failures: list[BaseException] = []

    def run() -> None:
        try:
            barrier.wait(timeout=5)
            published.append(publish_certification(result, output))
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert len(published) == 2
    assert len(list(output.rglob("certification-record.json"))) == 1
    assert not list(output.rglob("*.staging-*"))
    assert sorted(path.name for path in (output / "equant-py").iterdir()) == ["1.0.0"]


def test_concurrent_conflicting_publications_have_one_winner_and_no_residue(
    tmp_path: Path,
) -> None:
    output = tmp_path / "certifications"
    first = _result(tmp_path / "first")
    second = _result(tmp_path / "second")
    second = _with_changed_manifest(second, "concurrent-conflict")
    barrier = threading.Barrier(2)
    successes: list[object] = []
    failures: list[OperatorCertificationError] = []

    def run(result: ResearchCertification) -> None:
        barrier.wait(timeout=5)
        try:
            successes.append(publish_certification(result, output))
        except OperatorCertificationError as error:
            failures.append(error)

    threads = [
        threading.Thread(target=run, args=(first,)),
        threading.Thread(target=run, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(successes) == 1
    assert [error.code for error in failures] == ["certification_conflict"]
    assert sorted(path.name for path in (output / "equant-py").iterdir()) == ["1.0.0"]


def test_git_ignores_local_certification_output() -> None:
    repository = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        ["git", "check-ignore", "-q", ".open-xquant/certifications/probe"],
        cwd=repository,
        check=False,
    )

    assert completed.returncode == 0
