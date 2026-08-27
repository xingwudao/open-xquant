"""Deterministic, evidence-complete certification bundle tests."""

from __future__ import annotations

import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from oxq.operators.bundle import (
    export_certification_bundle,
    materialize_validated_bundle,
    validate_certification_bundle,
)
from oxq.operators.certification import certify_provider
from oxq.operators.models import CertificationTarget
from oxq.operators.registry import publish_certification
from oxq.operators.submission import load_provider_submission
from tests.operators.test_baseline_runner import _write_certifiable_provider

TARGET = CertificationTarget.parse("cp312-cp312-macosx_14_0_arm64")


def _export_bundle_fixture(root: Path, output: Path):
    fixture = _write_certifiable_provider(root / "fixture", expected=[None, None, 2.0])
    submission = load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir)
    try:
        result = certify_provider(submission)
        publication = publish_certification(result, root / "registry", target=TARGET)
        return export_certification_bundle(
            provider="equant-py",
            release="1.0.0",
            registry_dir=root / "registry",
            manifest_dir=fixture.path / "compat/open_xquant/manifests",
            baseline_files=[fixture.path / "compat/open_xquant/numerical_baselines/technical-v1.json"],
            target=TARGET,
            output_path=output,
        ), publication
    finally:
        submission.__exit__(None, None, None)


def test_export_is_byte_deterministic_and_preserves_source_json(tmp_path: Path) -> None:
    first, publication = _export_bundle_fixture(tmp_path / "one", tmp_path / "one.zip")
    source = tmp_path / "one" / "fixture" / "provider" / "compat/open_xquant"
    second = export_certification_bundle(
        provider="equant-py",
        release="1.0.0",
        registry_dir=tmp_path / "one" / "registry",
        manifest_dir=source / "manifests",
        baseline_files=[source / "numerical_baselines/technical-v1.json"],
        target=TARGET,
        output_path=tmp_path / "two.zip",
    )

    assert first.bundle_path.read_bytes() == second.bundle_path.read_bytes()
    manifest = next((tmp_path / "one" / "fixture" / "provider" / "compat/open_xquant/manifests").glob("*.json"))
    with zipfile.ZipFile(first.bundle_path) as archive:
        assert archive.read("manifests/equant.ttr.sma@1.0.0.operator.json") == manifest.read_bytes()
        assert archive.read("publication/certification-record.json") == (publication.release_dir / "certification-record.json").read_bytes()


def test_export_validates_and_materializes_evidence_complete_bundle(tmp_path: Path) -> None:
    bundle, _ = _export_bundle_fixture(tmp_path, tmp_path / "bundle.zip")

    validated = validate_certification_bundle(bundle.bundle_path)
    destination = tmp_path / "materialized"
    materialize_validated_bundle(validated, destination)

    assert validated.provider == "equant-py"
    assert validated.release == "1.0.0"
    assert validated.target == TARGET
    assert validated.operator_count == 1
    assert (destination / "bundle-manifest.json").read_bytes() == validated.members["bundle-manifest.json"]


@pytest.mark.parametrize(
    ("name", "configure"),
    [
        ("../escape", lambda info: info),
        ("folder\\escape", lambda info: info),
        ("nul\x00name", lambda info: info),
        ("duplicate", lambda info: info),
        ("symlink", lambda info: setattr(info, "external_attr", (stat.S_IFLNK | 0o777) << 16)),
        ("special", lambda info: setattr(info, "external_attr", (stat.S_IFCHR | 0o644) << 16)),
        ("encrypted", lambda info: setattr(info, "flag_bits", info.flag_bits | 1)),
        ("stored", lambda info: info),
    ],
)
def test_validation_rejects_unsafe_members(tmp_path: Path, name: str, configure) -> None:
    safe_name = name.replace("/", "_").replace("\\", "_").replace(chr(0), "_")
    bundle, _ = _export_bundle_fixture(tmp_path / safe_name, tmp_path / f"{safe_name}.zip")
    path = bundle.bundle_path
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED if name == "stored" else zipfile.ZIP_DEFLATED
    configure(info)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(info, b"x")
        if name == "duplicate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("bundle-manifest.json", b"y")

    with pytest.raises(ValueError):
        validate_certification_bundle(path)


def test_validation_rejects_member_and_expansion_limits(tmp_path: Path) -> None:
    bundle, _ = _export_bundle_fixture(tmp_path / "fixture", tmp_path / "limits.zip")
    too_many = bundle.bundle_path
    with zipfile.ZipFile(too_many, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(513):
            archive.writestr(f"member-{index}", b"x")
    with pytest.raises(ValueError):
        validate_certification_bundle(too_many)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        pytest.param("too-large", b"x" * (16 * 1024 * 1024 + 1), id="member-size"),
        pytest.param("ratio", b"x" * 200_000, id="compression-ratio"),
    ],
)
def test_validation_rejects_expansion_and_ratio_limits(tmp_path: Path, name: str, payload: bytes) -> None:
    bundle, _ = _export_bundle_fixture(tmp_path / name, tmp_path / f"{name}.zip")
    with zipfile.ZipFile(bundle.bundle_path, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(name, payload)

    with pytest.raises(ValueError):
        validate_certification_bundle(bundle.bundle_path)
