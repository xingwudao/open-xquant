"""Targeted certification records retain committed submission provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

import oxq.operators.registry as registry_module
from oxq.operators.certification import certify_provider
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.formats import canonical_json_bytes, sha256_bytes, strict_json_object
from oxq.operators.models import CertificationTarget
from oxq.operators.registry import (
    import_certification_publication,
    publish_certification,
    read_certification_publication,
)
from oxq.operators.submission import load_provider_submission
from tests.operators.test_baseline_runner import _write_certifiable_provider
from tests.operators.test_certification_registry import _result


def _certified_result_with_committed_baseline(tmp_path: Path):
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    submission = load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    )
    return submission, certify_provider(submission)


def test_targeted_certification_issues_v2_from_committed_evidence(
    tmp_path: Path,
) -> None:
    submission, result = _certified_result_with_committed_baseline(tmp_path)
    try:
        published = publish_certification(
            result,
            tmp_path / "registry",
            target=CertificationTarget.parse("cp312-cp312-macosx_14_0_arm64"),
        )
        record = strict_json_object(
            (published.release_dir / "certification-record.json").read_bytes()
        )
        assert record["schema_version"] == 2
        assert record["target"] == {
            "python_tag": "cp312",
            "abi_tag": "cp312",
            "platform_tag": "macosx_14_0_arm64",
        }
        assert record["baseline_sets"][0]["digest"].startswith("sha256:")  # type: ignore[index]
        case = record["operators"][0]["baseline_cases"][0]  # type: ignore[index]
        baseline = strict_json_object(result.baseline_cases[0].baseline_path.read_bytes())
        expected_case = baseline["cases"][0]  # type: ignore[index]
        assert case["case_digest"] == sha256_bytes(canonical_json_bytes(expected_case))
    finally:
        submission.__exit__(None, None, None)


def test_targeted_certification_rejects_synthetic_baseline_provenance(
    tmp_path: Path,
) -> None:
    with pytest.raises(OperatorCertificationError, match="committed baseline provenance"):
        publish_certification(
            _result(tmp_path),
            tmp_path / "registry",
            target=CertificationTarget.parse("cp312-cp312-macosx_14_0_arm64"),
        )


def test_reads_and_imports_targeted_publication(tmp_path: Path) -> None:
    submission, result = _certified_result_with_committed_baseline(tmp_path)
    try:
        published = publish_certification(
            result,
            tmp_path / "source",
            target=CertificationTarget.parse("cp312-cp312-macosx_14_0_arm64"),
        )
        assert read_certification_publication(published.release_dir).record["schema_version"] == 2
        imported = import_certification_publication(
            published.release_dir,
            tmp_path / "destination",
        )
        assert imported.record["target"]["python_tag"] == "cp312"  # type: ignore[index]
    finally:
        submission.__exit__(None, None, None)


def test_untargeted_publication_remains_byte_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "_utc_now", lambda: "2026-08-27T00:00:00Z")
    result = _result(tmp_path)
    first = publish_certification(result, tmp_path / "first")
    second = publish_certification(result, tmp_path / "second", target=None)
    first_files = {
        path.relative_to(first.release_dir): path.read_bytes()
        for path in first.release_dir.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.release_dir): path.read_bytes()
        for path in second.release_dir.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


@pytest.mark.parametrize("value", ["cp312-cp312", "CP312-cp312-macosx_14_0_arm64"])
def test_target_parser_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError):
        CertificationTarget.parse(value)
