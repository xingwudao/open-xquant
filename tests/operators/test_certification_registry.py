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
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BaselineCase,
    BaselineResult,
    ContractCandidate,
    ContractCertification,
    ResearchCertification,
)
from oxq.operators.registry import CertificationRegistry, publish_certification
from oxq.operators.resources import materialize_certification_profile


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
    manifest = {
        "operator_id": operator_id,
        "operator_version": operator_version,
    }
    manifest_path = source_root / "operator.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = ContractCandidate(
        manifest=manifest,
        binding=_binding(operator_id, operator_version, state=state),
        manifest_path=manifest_path,
        implementation_artifact=wheel,
    )
    baseline = BaselineCase(
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
        case_id="sma-3",
        status=status,
    )
    return ResearchCertification(
        provider=provider,
        release=release,
        submission_commit="f" * 40,
        source_commit="a" * 40,
        source_root=source_root,
        operators=(candidate,),
        artifacts=(),
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
    assert record["source_commit"] == "git-sha1:" + "a" * 40
    assert record["operators"] == [{
        "baseline_cases": [{"case_id": "sma-3", "status": "passed"}],
        "binding_digest": _sha256(binding_path.read_bytes()),
        "implementation_digest": "sha256:" + "e" * 64,
        "manifest_digest": "sha256:" + "d" * 64,
        "operator_id": "equant.ttr.sma",
        "operator_version": "1.0.0",
    }]
    with materialize_certification_profile() as paths:
        schema = json.loads(paths["certification_record"].read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    assert entry["submission_commit"] == "git-sha1:" + "f" * 40
    assert CertificationRegistry(output).get("equant.ttr.sma", "1.0.0") == binding
    assert b"SECRET_PROVIDER_SOURCE" not in b"".join(
        _tree_bytes(release_dir).values()
    )


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


def test_git_ignores_local_certification_output() -> None:
    repository = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        ["git", "check-ignore", "-q", ".open-xquant/certifications/probe"],
        cwd=repository,
        check=False,
    )

    assert completed.returncode == 0
