"""CLI coverage for local, manually trusted operator certification."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from tests.operators.helpers import commit_mutation, rewrite_json
from tests.operators.test_baseline_runner import _write_certifiable_provider

from oxq.cli.main import main


def _explicit_args(
    provider_repo: Path,
    provider_commit: str,
    artifact_dir: Path,
    output_dir: Path,
    *,
    as_json: bool = True,
) -> list[str]:
    args = [
        "operator",
        "certify-provider",
        "--provider-repo",
        str(provider_repo),
        "--provider-commit",
        provider_commit,
        "--artifact-dir",
        str(artifact_dir),
        "--output-dir",
        str(output_dir),
        "--trust-provider-code",
    ]
    if as_json:
        args.append("--json")
    return args


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_certify_provider_with_explicit_paths_emits_stable_json(tmp_path: Path) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    output = tmp_path / "certifications"

    result = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            fixture.submission_commit,
            fixture.artifact_dir,
            output,
        ),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "operator_count": 1,
        "output": str((output / "equant-py" / "1.0.0").resolve()),
        "provider": "equant-py",
        "release": "1.0.0",
        "source_commit": f"git-sha1:{fixture.implementation_commit}",
        "status": "research-certified",
        "submission_commit": f"git-sha1:{fixture.submission_commit}",
    }
    assert (output / "equant-py" / "1.0.0" / "registry-entry.json").is_file()


def test_certify_provider_defaults_to_provider_dist_and_current_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    args = [
        "operator",
        "certify-provider",
        "--provider-repo",
        str(fixture.path),
        "--provider-commit",
        fixture.submission_commit,
        "--trust-provider-code",
        "--json",
    ]
    first = CliRunner().invoke(main, args)
    second = CliRunner().invoke(main, args)

    expected_release = (
        workspace / ".open-xquant" / "certifications" / "equant-py" / "1.0.0"
    ).resolve()
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output)["output"] == str(expected_release)
    assert json.loads(second.output)["output"] == str(expected_release)
    assert len(list(expected_release.rglob("certification-record.json"))) == 1


def test_missing_trust_fails_before_repository_loading(tmp_path: Path) -> None:
    non_git_directory = tmp_path / "not-a-repository"
    non_git_directory.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "operator",
            "certify-provider",
            "--provider-repo",
            str(non_git_directory),
            "--provider-commit",
            "not-a-commit",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "code": "provider_code_trust_required",
        "message": "--trust-provider-code is required to execute provider wheels",
        "stage": "trust",
        "status": "fail",
    }


def test_rejects_remote_provider_url_at_the_cli_boundary(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "operator",
            "certify-provider",
            "--provider-repo",
            "https://github.com/example/provider.git",
            "--provider-commit",
            "a" * 40,
            "--trust-provider-code",
            "--output-dir",
            str(tmp_path / "output"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--provider-repo'" in result.output
    assert not (tmp_path / "output").exists()


def test_non_full_commit_emits_stable_service_error_without_publication(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    output = tmp_path / "certifications"

    result = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            "deadbeef",
            fixture.artifact_dir,
            output,
        ),
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "code": "provider_commit_invalid",
        "message": "provider commit must be a full lowercase SHA-1",
        "stage": "repository",
        "status": "fail",
    }
    assert not output.exists()


def test_empty_provider_release_emits_stable_error_without_publication(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    rewrite_json(
        fixture.path / "provider-catalog-v1.json",
        lambda catalog: catalog.update({"operators": {}}),
    )
    empty_commit = commit_mutation(fixture.path, "empty provider release")
    output = tmp_path / "certifications"

    result = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            empty_commit,
            fixture.artifact_dir,
            output,
        ),
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert set(payload) == {"status", "stage", "code", "message"}
    assert payload["status"] == "fail"
    assert payload["stage"] == "catalog"
    assert payload["code"] == "submission_schema_invalid"
    assert not output.exists()


def test_baseline_failure_is_atomic_and_human_error_has_no_traceback(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 99.0],
    )
    output = tmp_path / "certifications"

    result = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            fixture.submission_commit,
            fixture.artifact_dir,
            output,
            as_json=False,
        ),
    )

    assert result.exit_code == 1
    assert "Certification failed" in result.output
    assert "baseline_mismatch" in result.output
    assert "Traceback" not in result.output
    assert not output.exists() or not list(output.rglob("registry-entry.json"))


def test_human_success_reports_identity_count_output_and_state(tmp_path: Path) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    output = tmp_path / "certifications"

    result = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            fixture.submission_commit,
            fixture.artifact_dir,
            output,
            as_json=False,
        ),
    )

    assert result.exit_code == 0, result.output
    assert "Provider: equant-py" in result.output
    assert "Release: 1.0.0" in result.output
    assert "Operators: 1" in result.output
    assert f"Output: {(output / 'equant-py' / '1.0.0').resolve()}" in result.output
    assert "Status: research-certified" in result.output
    assert "Traceback" not in result.output


def test_conflicting_release_is_rejected_without_overwriting_original(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    output = tmp_path / "certifications"
    first = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            fixture.submission_commit,
            fixture.artifact_dir,
            output,
        ),
    )
    release_dir = output / "equant-py" / "1.0.0"
    before = _tree_bytes(release_dir)
    rewrite_json(
        fixture.path / "manifests" / "equant.ttr.sma.operator.json",
        lambda manifest: manifest.update({"semantic_name": "Simple Moving Average"}),
    )
    conflicting_commit = commit_mutation(fixture.path, "conflicting release")

    conflicting = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            conflicting_commit,
            fixture.artifact_dir,
            output,
        ),
    )

    assert first.exit_code == 0, first.output
    assert conflicting.exit_code == 1
    assert json.loads(conflicting.output)["code"] == "certification_conflict"
    assert _tree_bytes(release_dir) == before
