"""Integration tests for exact local provider submission loading."""

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import oxq.operators.submission as submission_module
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.submission import load_provider_submission
from tests.operators.helpers import (
    BUILD_IDENTIFIER,
    CATALOG_NAME,
    COMPATIBILITY_ROOT,
    commit_mutation,
    rewrite_json,
    sha256,
    write_provider_repository,
)


def test_loads_a_committed_submission_into_an_independent_archive(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)

    with load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir) as submission:
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
        fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME,
        lambda value: value["provider"].update({"name": "dirty-provider"}),  # type: ignore[index,union-attr]
    )

    with load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir) as submission:
        assert submission.provider == "equant-py"


def test_ignores_legacy_catalog_files_outside_compatibility_root(
    tmp_path: Path,
) -> None:
    def add_root_decoys(repository: Path) -> None:
        (repository / "provider-catalog-v1.json").write_text(
            "legacy root catalog must not be read",
            encoding="utf-8",
        )
        (repository / CATALOG_NAME).write_text(
            "root operator catalog must not be read",
            encoding="utf-8",
        )

    fixture = write_provider_repository(tmp_path, mutate=add_root_decoys)

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        assert submission.provider == "equant-py"
        assert submission.release == "1.0.0"


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
        fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME,
        lambda value: value.update({"build_record": path_value}),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_schema_invalid",
    )


@pytest.mark.parametrize("field", ["manifest", "baseline"])
def test_rejects_catalog_references_that_escape_compatibility_root(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = write_provider_repository(tmp_path)

    def escape_reference(catalog: dict[str, object]) -> None:
        operators = catalog["operators"]  # type: ignore[assignment]
        entry = operators["equant.ttr.sma@1.0.0"]  # type: ignore[index]
        entry[field] = f"../{field}.json"  # type: ignore[index]

    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME,
        escape_reference,
    )
    commit = commit_mutation(fixture.path, f"escape {field} reference")

    _assert_error(
        lambda: load_provider_submission(
            fixture.path,
            commit,
            fixture.artifact_dir,
        ),
        "submission_schema_invalid",
    )


def test_rejects_duplicate_json_keys_and_nonstandard_constants(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    (fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME).write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_json_invalid",
    )

    fixture = write_provider_repository(tmp_path / "nan")
    (fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json").write_text("NaN", encoding="utf-8")
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_json_invalid",
    )


def test_rejects_duplicate_keyed_catalog_operator_identity(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    catalog_path = fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    operator_key = "equant.ttr.sma@1.0.0"
    entry = json.dumps(catalog.pop("operators")[operator_key], sort_keys=True)
    catalog_path.write_text(
        f'{json.dumps(catalog, sort_keys=True)[:-1]}, "operators": {{"{operator_key}": {entry}, "{operator_key}": {entry}}}}}',
        encoding="utf-8",
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_json_invalid",
    )


def test_rejects_a_catalog_for_an_unsupported_contract_release(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME,
        lambda value: value.update({"contract_version": "2.0.0"}),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_schema_invalid",
    )


def test_rejects_schema_and_identity_mismatches(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json",
        lambda value: value.update({"provider": "another-provider"}),
    )
    commit = commit_mutation(fixture.path)
    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_identity_mismatch",
    )

    fixture = write_provider_repository(tmp_path / "schema")
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
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
        fixture.path / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json",
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
        fixture.path / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json",
        duplicate_case,
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_identity_mismatch",
    )


@pytest.mark.parametrize("operator_count", [2, 5])
def test_loads_shared_baseline_file_once_for_all_referencing_catalog_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator_count: int,
) -> None:
    operator_ids = [
        "equant.ttr.sma",
        "equant.ttr.ema",
        "equant.ttr.rsi",
        "equant.ttr.atr",
        "equant.ttr.momentum",
    ][:operator_count]

    def share_baseline(repository: Path) -> None:
        catalog_path = repository / COMPATIBILITY_ROOT / CATALOG_NAME
        baseline_path = repository / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json"
        catalog = json.loads(catalog_path.read_text())
        baseline = json.loads(baseline_path.read_text())
        original_case = baseline["cases"][0]
        baseline["cases"] = []
        for operator_id in operator_ids:
            catalog["operators"][f"{operator_id}@1.0.0"] = {
                "manifest": "manifests/equant.ttr.sma.operator.json",
                "baseline": "numerical_baselines/technical-v1.json",
            }
            case = dict(original_case)
            case["operator_id"] = operator_id
            case["case_id"] = operator_id.rsplit(".", 1)[-1] + "-window-3"
            baseline["cases"].append(case)
        catalog_path.write_text(json.dumps(catalog, sort_keys=True))
        baseline_path.write_text(json.dumps(baseline, sort_keys=True))

    fixture = write_provider_repository(tmp_path, mutate=share_baseline)
    baseline_reads = 0
    real_read_json = submission_module._read_json

    def observed_read_json(path: Path, operator_id: str | None = None) -> object:
        nonlocal baseline_reads
        if path.name == "technical-v1.json":
            baseline_reads += 1
        return real_read_json(path, operator_id)

    monkeypatch.setattr(submission_module, "_read_json", observed_read_json)

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        assert [case.operator_id for case in submission.baseline_cases] == operator_ids
        assert [case.case_id for case in submission.baseline_cases] == [
            operator_id.rsplit(".", 1)[-1] + "-window-3" for operator_id in operator_ids
        ]

    assert baseline_reads == 1


def test_rejects_shared_baseline_missing_a_referencing_catalog_identity(
    tmp_path: Path,
) -> None:
    def add_uncovered_operator(repository: Path) -> None:
        rewrite_json(
            repository / COMPATIBILITY_ROOT / CATALOG_NAME,
            lambda catalog: catalog["operators"].update(  # type: ignore[union-attr]
                {
                    "equant.ttr.ema@1.0.0": {
                        "manifest": "manifests/equant.ttr.sma.operator.json",
                        "baseline": "numerical_baselines/technical-v1.json",
                    }
                }
            ),
        )

    fixture = write_provider_repository(tmp_path, mutate=add_uncovered_operator)

    _assert_error(
        lambda: load_provider_submission(
            fixture.path,
            fixture.submission_commit,
            fixture.artifact_dir,
        ),
        "submission_identity_mismatch",
    )


@pytest.mark.parametrize(
    "source_commit",
    ["git-sha1:" + "a" * 40, "git-sha1:deadbeef"],
)
def test_rejects_missing_or_nonfull_implementation_commit(tmp_path: Path, source_commit: str) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
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
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
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
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
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


def test_rejects_an_artifact_that_is_not_a_wheel(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    wheel_path = fixture.artifact_dir / fixture.wheel_name
    wheel_path.write_bytes(b"not a wheel")
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
        lambda value: value["artifacts"][0].update(  # type: ignore[index]
            {"digest": sha256(wheel_path.read_bytes())}
        ),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "artifact_invalid",
    )


@pytest.mark.parametrize(
    ("metadata_name", "metadata_version"),
    [("another-project", "1.0.0"), ("equant-ttr", "2.0.0")],
    ids=["distribution", "version"],
)
def test_rejects_wheel_metadata_that_differs_from_the_build_record(
    tmp_path: Path,
    metadata_name: str,
    metadata_version: str,
) -> None:
    fixture = write_provider_repository(tmp_path)
    wheel_path = fixture.artifact_dir / fixture.wheel_name
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("equant_ttr/__init__.py", "")
        archive.writestr(
            "equant_ttr-1.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "equant_ttr-1.0.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: {metadata_version}\n",
        )
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
        lambda value: value["artifacts"][0].update(  # type: ignore[index]
            {"digest": sha256(wheel_path.read_bytes())}
        ),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "artifact_identity_mismatch",
    )


@pytest.mark.parametrize("location", ["filename", "wheel-header"])
def test_rejects_wheel_tags_incompatible_with_certifier_runtime(
    tmp_path: Path,
    location: str,
) -> None:
    fixture = write_provider_repository(tmp_path)
    wheel_path = fixture.artifact_dir / fixture.wheel_name
    filename = fixture.wheel_name
    if location == "filename":
        filename = filename.replace("py3-none-any", "cp999-cp999-any")
        incompatible_path = wheel_path.with_name(filename)
        wheel_path.rename(incompatible_path)
        wheel_path = incompatible_path
    else:
        with zipfile.ZipFile(wheel_path, "w") as archive:
            archive.writestr("equant_ttr/__init__.py", "")
            archive.writestr(
                "equant_ttr-1.0.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: cp999-cp999-any\n",
            )
            archive.writestr(
                "equant_ttr-1.0.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: equant-ttr\nVersion: 1.0.0\n",
            )
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
        lambda value: value["artifacts"][0].update(  # type: ignore[index]
            {
                "filename": filename,
                "digest": sha256(wheel_path.read_bytes()),
            }
        ),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(
            fixture.path,
            commit,
            fixture.artifact_dir,
        ),
        "artifact_invalid",
    )


def test_rejects_wheel_filename_identity_that_differs_from_metadata(
    tmp_path: Path,
) -> None:
    fixture = write_provider_repository(tmp_path)
    wheel_path = fixture.artifact_dir / fixture.wheel_name
    filename = "other_project-9.9.0-py3-none-any.whl"
    wheel_path.rename(wheel_path.with_name(filename))
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
        lambda value: value["artifacts"][0].update(  # type: ignore[index]
            {"filename": filename}
        ),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(
            fixture.path,
            commit,
            fixture.artifact_dir,
        ),
        "artifact_identity_mismatch",
    )


@pytest.mark.parametrize("wheel_version", ["invalid", "99.0"])
def test_rejects_unsupported_wheel_format_versions(
    tmp_path: Path,
    wheel_version: str,
) -> None:
    fixture = write_provider_repository(tmp_path)
    wheel_path = fixture.artifact_dir / fixture.wheel_name
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("equant_ttr/__init__.py", "")
        archive.writestr(
            "equant_ttr-1.0.0.dist-info/WHEEL",
            f"Wheel-Version: {wheel_version}\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "equant_ttr-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: equant-ttr\nVersion: 1.0.0\n",
        )
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
        lambda value: value["artifacts"][0].update(  # type: ignore[index]
            {"digest": sha256(wheel_path.read_bytes())}
        ),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(
            fixture.path,
            commit,
            fixture.artifact_dir,
        ),
        "artifact_invalid",
    )


def test_accepts_normalized_prerelease_wheel_filename_version(
    tmp_path: Path,
) -> None:
    filename = "equant_ttr-1.0.0rc1-py3-none-any.whl"
    wheel_path = tmp_path / filename
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("equant_ttr/__init__.py", "")
        archive.writestr(
            "equant_ttr-1.0.0rc1.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "equant_ttr-1.0.0rc1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: equant-ttr\nVersion: 1.0.0-rc.1\n",
        )

    submission_module._verify_wheel_identity(
        wheel_path,
        "equant-ttr",
        "1.0.0-rc.1",
        filename,
    )


@pytest.mark.parametrize(
    "dist_info_directory",
    ["other_project-1.0.0.dist-info", "equant_ttr-9.0.0.dist-info"],
)
def test_rejects_dist_info_directory_that_differs_from_wheel_identity(
    tmp_path: Path,
    dist_info_directory: str,
) -> None:
    fixture = write_provider_repository(tmp_path)
    wheel_path = fixture.artifact_dir / fixture.wheel_name
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("equant_ttr/__init__.py", "")
        archive.writestr(
            f"{dist_info_directory}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            f"{dist_info_directory}/METADATA",
            "Metadata-Version: 2.1\nName: equant-ttr\nVersion: 1.0.0\n",
        )
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / "candidate-build-v1.json",
        lambda value: value["artifacts"][0].update(  # type: ignore[index]
            {"digest": sha256(wheel_path.read_bytes())}
        ),
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(
            fixture.path,
            commit,
            fixture.artifact_dir,
        ),
        "artifact_identity_mismatch",
    )


def test_intake_accepts_multi_output_expected_mapping(tmp_path: Path) -> None:
    def add_second_output(repository: Path) -> None:
        rewrite_json(
            repository / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json",
            lambda value: value["cases"][0]["expected"].update(  # type: ignore[index]
                {"ema_3": [None, 10.0]}
            ),
        )

    fixture = write_provider_repository(tmp_path, mutate=add_second_output)

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        assert set(submission.baseline_cases[0].expected) == {"sma_3", "ema_3"}


def test_ignores_local_git_replacement_objects(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    rewrite_json(
        fixture.path / COMPATIBILITY_ROOT / CATALOG_NAME,
        lambda value: value["provider"].update({"name": "replacement-provider"}),  # type: ignore[union-attr]
    )
    replacement_commit = commit_mutation(fixture.path)
    subprocess.run(
        [
            "git",
            "-C",
            str(fixture.path),
            "replace",
            fixture.submission_commit,
            replacement_commit,
        ],
        check=True,
    )

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        assert submission.provider == "equant-py"


def test_normalizes_an_artifact_read_race_to_artifact_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    manifest = fixture.path / COMPATIBILITY_ROOT / "manifests" / "equant.ttr.sma.operator.json"
    manifest.unlink()
    manifest.symlink_to("../candidate-build-v1.json")
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_path_invalid",
    )


def test_allows_an_unreferenced_symlink_in_the_committed_repository(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    target = fixture.path / "docs" / "target.md"
    target.parent.mkdir()
    target.write_text("documentation\n", encoding="utf-8")
    (target.parent / "latest.md").symlink_to(target.name)
    commit = commit_mutation(fixture.path)

    with load_provider_submission(
        fixture.path,
        commit,
        fixture.artifact_dir,
    ) as submission:
        assert submission.provider == "equant-py"


def test_reads_exact_source_blobs_without_export_substitution(tmp_path: Path) -> None:
    source_bytes = b'BUILD = "$Format:%H$"\ndef sma():\n    return 10.0\n'

    def add_archive_attribute(repository: Path) -> None:
        (repository / "src" / "equant_ttr" / "sma.py").write_bytes(source_bytes)
        (repository / ".gitattributes").write_text(
            "src/equant_ttr/sma.py export-subst\n",
            encoding="utf-8",
        )

    fixture = write_provider_repository(
        tmp_path,
        implementation_mutate=add_archive_attribute,
    )

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        assert (submission.source_root / "src" / "equant_ttr" / "sma.py").read_bytes() == source_bytes


def test_reads_referenced_blobs_even_when_export_ignore_is_set(tmp_path: Path) -> None:
    def ignore_manifest_in_archives(repository: Path) -> None:
        (repository / ".gitattributes").write_text(
            "compat/open_xquant/manifests/equant.ttr.sma.operator.json export-ignore\n",
            encoding="utf-8",
        )

    fixture = write_provider_repository(tmp_path, mutate=ignore_manifest_in_archives)

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        assert submission.operators[0].manifest_path.is_file()


@pytest.mark.parametrize("source_path", ["/source.py", "../source.py", "dir\\source.py", "./source.py"])
def test_rejects_unsafe_manifest_source_paths(tmp_path: Path, source_path: str) -> None:
    fixture = write_provider_repository(tmp_path)
    manifest_path = fixture.path / COMPATIBILITY_ROOT / "manifests" / "equant.ttr.sma.operator.json"
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


def test_normalizes_deeply_nested_json_to_an_intake_error(tmp_path: Path) -> None:
    fixture = write_provider_repository(tmp_path)
    baseline_path = fixture.path / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json"
    baseline = baseline_path.read_text(encoding="utf-8")
    marker = '"parameters": {"window": 3}'
    assert marker in baseline
    nested = "[" * 10000 + "0" + "]" * 10000
    baseline_path.write_text(
        baseline.replace(marker, f'"parameters": {{"nested": {nested}}}'),
        encoding="utf-8",
    )
    commit = commit_mutation(fixture.path)

    _assert_error(
        lambda: load_provider_submission(fixture.path, commit, fixture.artifact_dir),
        "submission_json_invalid",
    )


def _assert_error(action: object, code: str) -> None:
    with pytest.raises(OperatorCertificationError) as caught:
        action()  # type: ignore[operator]
    assert caught.value.code == code
