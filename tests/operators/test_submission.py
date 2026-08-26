"""Integration tests for exact local provider submission loading."""

import json
import subprocess
from pathlib import Path

import pytest

from oxq.operators.errors import OperatorCertificationError
from oxq.operators.submission import load_provider_submission
from tests.operators.helpers import (
    BUILD_IDENTIFIER,
    commit_mutation,
    rewrite_json,
    sha256,
    write_provider_repository,
)


def test_loads_a_committed_submission_into_an_independent_archive(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)

    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        assert submission.provider == "equant-py"
        assert submission.release == "1.0.0"
        assert submission.submission_commit == f"git-sha1:{fixture.submission_commit}"
        assert submission.source_commit == f"git-sha1:{fixture.implementation_commit}"
        assert submission.archive_root != fixture.path
        assert submission.source_root != submission.archive_root
        assert [entry.operator_id for entry in submission.operators] == ["equant.ttr.sma"]
        assert [entry.operator_version for entry in submission.operators] == ["1.0.0"]
        assert submission.operators[0].manifest_path.is_file()
        assert submission.operators[0].baseline_path.is_file()
        assert submission.artifacts[0].wheel_path == fixture.artifact_dir / fixture.wheel_name
        assert submission.artifacts[0].build_identifier == BUILD_IDENTIFIER


def test_uses_committed_data_not_a_dirty_working_tree(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / "provider-catalog-v1.json",
        lambda value: value["provider"].update({"name": "dirty-provider"}),  # type: ignore[index,union-attr]
    )

    with load_provider_submission(
        fixture.path, fixture.submission_commit, fixture.artifact_dir
    ) as submission:
        assert submission.provider == "equant-py"


@pytest.mark.parametrize("revision", ["deadbeef", "A" * 40, "not-a-commit"])
def test_rejects_a_noncanonical_provider_commit(tmp_path: Path, revision: str) -> None:
    fixture = write_provider_repository(tmp_path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, revision, fixture.artifact_dir),
        "provider_commit_invalid",
    )


def test_rejects_a_missing_git_repository(tmp_path: Path) -> None:
    _assert_error(
        lambda: load_provider_submission(tmp_path / "missing", "0" * 40, tmp_path),
        "provider_repo_invalid",
    )


def test_rejects_an_existing_directory_that_is_not_a_git_repository(tmp_path: Path) -> None:
    directory = tmp_path / "not-a-repository"
    directory.mkdir()

    _assert_error(
        lambda: load_provider_submission(directory, "0" * 40, tmp_path),
        "provider_repo_invalid",
    )


@pytest.mark.parametrize(
    "path_value",
    ["/candidate-build-v1.json", "../candidate-build-v1.json", "dir\\build.json", "./build.json"],
)
def test_rejects_unsafe_catalog_paths(tmp_path: Path, path_value: str) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / "provider-catalog-v1.json",
        lambda value: value.update({"build_record": path_value}),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_schema_invalid",
    )


def test_rejects_duplicate_json_keys_and_nonstandard_constants(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    (fixture.path / "provider-catalog-v1.json").write_text(
        '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
    )
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_json_invalid",
    )

    fixture = write_provider_repository(tmp_path / "nan")
    (fixture.path / "candidate-build-v1.json").write_text("NaN", encoding="utf-8")
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_json_invalid",
    )


def test_rejects_duplicate_keyed_catalog_operator_identity(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    catalog_path = fixture.path / "provider-catalog-v1.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    operator_key = "equant.ttr.sma@1.0.0"
    entry = json.dumps(catalog.pop("operators")[operator_key], sort_keys=True)
    catalog_path.write_text(
        f"{json.dumps(catalog, sort_keys=True)[:-1]}, \"operators\": {{\"{operator_key}\": {entry}, \"{operator_key}\": {entry}}}}}",
        encoding="utf-8",
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_json_invalid",
    )

def test_rejects_schema_and_identity_mismatches(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / "numerical_baselines" / "technical-v1.json",
        lambda value: value.update({"provider": "another-provider"}),
    )
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_identity_mismatch",
    )

    fixture = write_provider_repository(tmp_path / "schema")
    rewrite_json(
        fixture.path / "candidate-build-v1.json",
        lambda value: value.update({"unexpected": True}),
    )
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_schema_invalid",
    )


def test_rejects_baseline_case_identity_that_differs_from_catalog(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / "numerical_baselines" / "technical-v1.json",
        lambda value: value["cases"][0].update({"operator_id": "equant.ttr.ema"}),  # type: ignore[index]
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_identity_mismatch",
    )


def test_rejects_duplicate_global_baseline_case_identity(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)

    def duplicate_case(value: dict[str, object]) -> None:
        first = dict(value["cases"][0])  # type: ignore[index]
        first["parameters"] = {"window": 4}
        value["cases"].append(first)  # type: ignore[union-attr]

    rewrite_json(
        fixture.path / "numerical_baselines" / "technical-v1.json",
        duplicate_case,
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_identity_mismatch",
    )


@pytest.mark.parametrize(
    "source_commit",
    ["git-sha1:" + "a" * 40, "git-sha1:deadbeef"],
)
def test_rejects_missing_or_nonfull_implementation_commit(
    tmp_path: Path, source_commit: str
) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / "candidate-build-v1.json",
        lambda value: value.update({"source_commit": source_commit}),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_schema_invalid" if source_commit.endswith("deadbeef") else "submission_identity_mismatch",
    )


def test_rejects_an_implementation_commit_not_ancestral_to_submission(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    tree = subprocess.run(
        ["git", "-C", str(fixture.path), "rev-parse", "HEAD^{tree}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    unrelated = subprocess.run(
        ["git", "-C", str(fixture.path), "commit-tree", tree, "-m", "unrelated source"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    rewrite_json(
        fixture.path / "candidate-build-v1.json",
        lambda value: value.update({"source_commit": f"git-sha1:{unrelated}"}),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_identity_mismatch",
    )


def test_rejects_duplicate_or_missing_artifacts(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / "candidate-build-v1.json",
        lambda value: value["artifacts"].append(  # type: ignore[index,union-attr]
            {
                "distribution": "equant-ttr",
                "version": "1.0.0",
                "filename": fixture.wheel_name,
                "role": "runtime-dependency",
                "build_identifier": "duplicate-test-build",
                "digest": sha256(b"another artifact"),
            }
        ),
    )
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_schema_invalid",
    )

    fixture = write_provider_repository(tmp_path / "missing")
    (fixture.artifact_dir / fixture.wheel_name).unlink()
    _assert_error(
        lambda: load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir),
        "artifact_missing",
    )


def test_rejects_invalid_artifact_files_and_digests(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    (fixture.artifact_dir / fixture.wheel_name).write_bytes(b"changed")
    _assert_error(
        lambda: load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir),
        "artifact_digest_mismatch",
    )

    fixture = write_provider_repository(tmp_path / "directory")
    (fixture.artifact_dir / fixture.wheel_name).unlink()
    (fixture.artifact_dir / fixture.wheel_name).mkdir()
    _assert_error(
        lambda: load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir),
        "artifact_missing",
    )


def test_normalizes_an_artifact_read_race_to_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_provider_repository(tmp_path)
    wheel_path = fixture.artifact_dir / fixture.wheel_name
    path_open = Path.open

    def lose_artifact_race(path: Path, *args: object, **kwargs: object) -> object:
        if path == wheel_path:
            wheel_path.unlink()
            raise FileNotFoundError("artifact replaced during verification")
        return path_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", lose_artifact_race)

    _assert_error(
        lambda: load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir),
        "artifact_missing",
    )


def test_rejects_referenced_symlinks_from_the_committed_archive(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    manifest = fixture.path / "manifests" / "equant.ttr.sma.operator.json"
    manifest.unlink()
    manifest.symlink_to("../candidate-build-v1.json")
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_path_invalid",
    )


@pytest.mark.parametrize("source_path", ["/source.py", "../source.py", "dir\\source.py", "./source.py"])
def test_rejects_unsafe_manifest_source_paths(tmp_path: Path, source_path: str) -> None:
    fixture = write_provider_repository(tmp_path)
    manifest_path = fixture.path / "manifests" / "equant.ttr.sma.operator.json"
    rewrite_json(
        manifest_path,
        lambda value: value["implementation"].update({"source_files": [source_path]}),  # type: ignore[index,union-attr]
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_path_invalid",
    )


def test_rejects_a_symlinked_implementation_source_file(tmp_path: Path) -> None:
    def make_source_symlink(repository: Path) -> None:
        source = repository / "src" / "equant_ttr" / "sma.py"
        target = source.with_name("shared.py")
        target.write_text("def sma():\n    return 10.0\n", encoding="utf-8")
        source.unlink()
        source.symlink_to(target.name)

    fixture = write_provider_repository(tmp_path, implementation_mutate=make_source_symlink)

    _assert_error(
        lambda: load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir),
        "submission_path_invalid",
    )


def _assert_error(action: object, code: str) -> None:
    with pytest.raises(OperatorCertificationError) as caught:
        action()  # type: ignore[operator]
    assert caught.value.code == code
