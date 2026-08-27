"""CLI coverage for local, manually trusted operator certification."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from click.testing import CliRunner
from tests.operators.helpers import (
    CATALOG_NAME,
    COMPATIBILITY_ROOT,
    commit_mutation,
    rewrite_json,
)
from tests.operators.test_baseline_runner import _write_certifiable_provider

import oxq.operators.baseline_runner as baseline_runner
import oxq.operators.runtime_protocol as runtime_protocol
from oxq.cli.main import main


@pytest.fixture(autouse=True)
def _provide_explicit_fixture_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_paths = [
        path for path in sys.path if "site-packages" in Path(path).parts
    ]

    def run_fixture_request(
        request: Mapping[str, object],
        wheel_snapshots: Sequence[str | Path],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        return runtime_protocol.run_exact_wheel_request(
            request,
            wheel_snapshots,
            timeout_seconds=timeout_seconds,
            _test_runtime_paths=runtime_paths,
        )

    monkeypatch.setattr(
        baseline_runner,
        "run_exact_wheel_request",
        run_fixture_request,
    )


def _explicit_args(
    provider_repo: Path,
    provider_commit: str,
    artifact_dir: Path,
    output_dir: Path | None = None,
    *,
    as_json: bool = True,
) -> list[str]:
    del output_dir
    args = [
        "operator",
        "certify-provider",
        "--provider-repo",
        str(provider_repo),
        "--provider-commit",
        provider_commit,
        "--artifact-dir",
        str(artifact_dir),
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


def test_certify_provider_reads_compat_layout_and_ignores_root_catalogs(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    (fixture.path / "provider-catalog-v1.json").write_text(
        "legacy root catalog must not be read",
        encoding="utf-8",
    )
    (fixture.path / CATALOG_NAME).write_text(
        "root operator catalog must not be read",
        encoding="utf-8",
    )
    submission_commit = commit_mutation(
        fixture.path,
        "add ignored root catalogs",
    )
    output = tmp_path / "certifications"

    result = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            submission_commit,
            fixture.artifact_dir,
            output,
        ),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "operator_count": 1,
        "provider": "equant-py",
        "release": "1.0.0",
        "source_commit": f"git-sha1:{fixture.implementation_commit}",
        "status": "research-certified",
        "submission_commit": f"git-sha1:{submission_commit}",
    }
    assert not output.exists()


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

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output)["status"] == "research-certified"
    assert json.loads(second.output)["status"] == "research-certified"
    assert not (workspace / ".open-xquant" / "certifications").exists()


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


def test_invalid_target_fails_before_provider_repository_loading(tmp_path: Path) -> None:
    repository = tmp_path / "not-a-repository"
    repository.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "operator",
            "certify-provider",
            "--provider-repo",
            str(repository),
            "--provider-commit",
            "a" * 40,
            "--trust-provider-code",
            "--target",
            "invalid",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "code": "certification_target_invalid",
        "message": "certification target must be python-abi-platform",
        "stage": "target",
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
        fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME,
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


def test_compat_catalog_reference_cannot_escape_compat_directory(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[None, None, 2.0],
    )
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME,
        lambda catalog: catalog["operators"]["equant.ttr.sma@1.0.0"].update(  # type: ignore[index]
            {"manifest": "../manifests/equant.ttr.sma.operator.json"}
        ),
    )
    escaped_commit = commit_mutation(fixture.path, "escape compat directory")
    output = tmp_path / "certifications"

    result = CliRunner().invoke(
        main,
        _explicit_args(
            fixture.path,
            escaped_commit,
            fixture.artifact_dir,
            output,
        ),
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "code": "submission_schema_invalid",
        "message": "submission does not match its schema",
        "stage": "catalog",
        "status": "fail",
    }
    assert not output.exists()


def test_baseline_failure_json_includes_loaded_identity_and_is_atomic(
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
        ),
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "code": "baseline_mismatch",
        "message": "provider output does not match the numerical baseline",
        "operator_id": "equant.ttr.sma",
        "provider": "equant-py",
        "release": "1.0.0",
        "stage": "baseline",
        "status": "fail",
    }
    assert not output.exists()


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**63])
def test_out_of_range_int64_baseline_fails_through_cli(
    tmp_path: Path,
    value: int,
) -> None:
    provider_source = (
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        f"    return pd.Series([{value}, None, {value}], "
        "index=frame.index, name=f'sma_{window}', dtype='object')\n"
    )
    fixture = _write_certifiable_provider(
        tmp_path / "fixture",
        expected=[value, None, value],
        provider_source=provider_source,
        output_dtype="int64",
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

    assert result.exit_code == 1
    assert json.loads(result.output)["code"] == "baseline_mismatch"
    assert not output.exists()


def test_human_failure_reports_loaded_identity_without_traceback(
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
    assert "Certification failed for equant-py 1.0.0" in result.output
    assert "baseline_mismatch" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()


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
    assert "Output:" not in result.output
    assert "Status: research-certified" in result.output
    assert "Traceback" not in result.output
