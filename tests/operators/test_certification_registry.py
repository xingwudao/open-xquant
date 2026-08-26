"""Atomic publication and lookup tests for operator certifications."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import oxq.operators.registry as registry_module
from oxq.operators.certification import certify_provider
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
from tests.operators.test_baseline_runner import _write_certifiable_provider


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


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
        "schema_id": (
            "https://open-xquant.dev/contracts/quant-operators/"
            "operator-manifest-v1.schema.json"
        ),
        "schema_release": "1.0.0",
        "schema_digest": "sha256:" + "c" * 64,
        "manifest_digest": "sha256:" + "d" * 64,
        "implementation_digest": "sha256:" + "e" * 64,
        "surface_release": "1.0.0",
        "contract_surface": {
            "quant_panel_schema": {
                "release": "1.0.0",
                "digest": "sha256:" + "1" * 64,
            },
            "operator_manifest_schema": {
                "release": "1.0.0",
                "digest": "sha256:" + "2" * 64,
            },
            "operator_binding_schema": {
                "release": "1.0.0",
                "digest": "sha256:" + "3" * 64,
            },
            "reference_validator": {
                "release": "1.0.0",
                "digest": "sha256:" + "4" * 64,
            },
        },
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
    (source_root / "provider.py").write_text("SECRET_PROVIDER_SOURCE = True\n")
    wheel = tmp_path / f"{provider}-{release}.whl"
    wheel.write_bytes(b"exact provider wheel")
    wheel_digest = _sha256(wheel.read_bytes())
    manifest = {
        "operator_id": operator_id,
        "operator_version": operator_version,
        "distribution": "equant-ttr",
        "implementation": {
            "package_version": "1.0.0",
            "source_commit": "git-sha1:" + "a" * 40,
            "build_identifier": "registry-test-build",
            "implementation_digest": wheel_digest,
        },
    }
    manifest_path = source_root / "operator.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = ContractCandidate(
        manifest=manifest,
        binding={
            **_binding(operator_id, operator_version, state=state),
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
    return ResearchCertification(
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


def _assert_canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    parsed = cast(dict[str, object], json.loads(raw))
    assert raw == (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode()
    return parsed


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    assert sorted(
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
    ) == [
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
    assert record["artifacts"] == [{
        "build_identifier": "registry-test-build",
        "digest": result.artifacts[0].digest,
        "distribution": "equant-ttr",
        "filename": result.artifacts[0].filename,
        "role": "implementation",
        "version": "1.0.0",
    }]
    assert record["operators"] == [{
        "baseline_cases": [{"case_id": "sma-window-3", "status": "passed"}],
        "binding_digest": _sha256(binding_path.read_bytes()),
        "implementation_digest": result.artifacts[0].digest,
        "manifest_digest": "sha256:" + "d" * 64,
        "operator_id": "equant.ttr.sma",
        "operator_version": "1.0.0",
    }]
    with materialize_certification_profile() as paths:
        schema = json.loads(paths["certification_record"].read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    assert entry["submission_commit"] == result.submission_commit
    assert CertificationRegistry(output).get("equant.ttr.sma", "1.0.0") == binding
    assert b"SECRET_PROVIDER_SOURCE" not in b"".join(
        _tree_bytes(release_dir).values()
    )


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
    changed_binding = dict(result.operators[0].binding)
    changed_binding["manifest_digest"] = "sha256:" + "9" * 64
    changed_candidate = replace(result.operators[0], binding=changed_binding)
    conflict = replace(result, operators=(changed_candidate,))

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


def test_allows_multiple_operators_bound_to_one_implementation_artifact(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    first = result.operators[0]
    second_manifest = dict(first.manifest)
    second_manifest["operator_id"] = "equant.ttr.ema"
    second_binding = dict(first.binding)
    second_binding["operator_id"] = "equant.ttr.ema"
    second = replace(first, manifest=second_manifest, binding=second_binding)
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
    shared = replace(
        result,
        operators=(first, second),
        baseline_cases=(*result.baseline_cases, second_case),
        baseline_results=(*result.baseline_results, second_pass),
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
    ["_binding_schema", "_certification_record_schema"],
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
    assert sorted(path.name for path in (output / "equant-py").iterdir()) == [
        "1.0.0"
    ]


def test_concurrent_conflicting_publications_have_one_winner_and_no_residue(
    tmp_path: Path,
) -> None:
    output = tmp_path / "certifications"
    first = _result(tmp_path / "first")
    second = _result(tmp_path / "second")
    changed_binding = dict(second.operators[0].binding)
    changed_binding["manifest_digest"] = "sha256:" + "9" * 64
    second = replace(
        second,
        operators=(replace(second.operators[0], binding=changed_binding),),
    )
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
    assert sorted(path.name for path in (output / "equant-py").iterdir()) == [
        "1.0.0"
    ]


def test_git_ignores_local_certification_output() -> None:
    repository = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        ["git", "check-ignore", "-q", ".open-xquant/certifications/probe"],
        cwd=repository,
        check=False,
    )

    assert completed.returncode == 0
