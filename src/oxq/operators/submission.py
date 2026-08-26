"""Load one provider submission from an exact, local Git commit."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker, ValidationError  # type: ignore[import-untyped]

from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BaselineCase,
    BuildArtifact,
    CatalogEntry,
    ProviderSubmission,
)
from oxq.operators.resources import materialize_certification_profile

_FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_CATALOG_FILE = "provider-catalog-v1.json"


def load_provider_submission(
    provider_repo: Path | str,
    provider_commit: str,
    artifact_dir: Path | str,
) -> ProviderSubmission:
    """Archive and validate a provider's exact committed submission."""
    repository = Path(provider_repo)
    if not repository.is_dir():
        raise _error("provider_repo_invalid", "provider repository does not exist", "repository")
    if not _is_git_repository(repository):
        raise _error("provider_repo_invalid", "provider repository is not a Git repository", "repository")
    if _FULL_SHA1.fullmatch(provider_commit) is None:
        raise _error("provider_commit_invalid", "provider commit must be a full lowercase SHA-1", "repository")
    resolved_commit = _resolve_commit(repository, provider_commit)
    if resolved_commit != provider_commit:
        raise _error("provider_commit_invalid", "provider commit did not resolve exactly", "repository")

    archive = TemporaryDirectory(prefix="oxq-provider-submission-")
    try:
        archive_root = Path(archive.name) / "submission"
        archive_root.mkdir()
        tar_path = Path(archive.name) / "submission.tar"
        _archive_commit(repository, provider_commit, tar_path)
        _extract_archive(tar_path, archive_root)
        return _load_archive(
            archive=archive,
            archive_root=archive_root,
            repository=repository,
            provider_commit=provider_commit,
            artifact_dir=Path(artifact_dir),
        )
    except BaseException:
        archive.cleanup()
        raise


def _resolve_commit(repository: Path, provider_commit: str) -> str:
    result = _git(
        repository,
        ["rev-parse", "--verify", f"{provider_commit}^{{commit}}"],
    )
    if result.returncode != 0:
        raise _error("provider_commit_invalid", "provider commit is not a local commit", "repository")
    return result.stdout.strip()


def _is_git_repository(repository: Path) -> bool:
    return _git(repository, ["rev-parse", "--git-dir"]).returncode == 0


def _archive_commit(repository: Path, provider_commit: str, tar_path: Path) -> None:
    result = _git(
        repository,
        ["archive", "--format=tar", f"--output={tar_path}", provider_commit],
    )
    if result.returncode != 0 or not tar_path.is_file():
        raise _error("provider_commit_invalid", "cannot archive provider commit", "repository")


def _git(repository: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def _extract_archive(tar_path: Path, archive_root: Path) -> None:
    try:
        with tarfile.open(tar_path, "r") as archive:
            members = archive.getmembers()
            for member in members:
                _validate_tar_member(member)
            archive.extractall(archive_root, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise _error("submission_path_invalid", "provider archive is unsafe", "archive") from exc


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or not member.name
        or any(part in {"", ".", ".."} for part in path.parts)
        or member.issym()
        or member.islnk()
        or not (member.isfile() or member.isdir())
    ):
        raise _error("submission_path_invalid", "provider archive contains an unsafe member", "archive")


def _load_archive(
    *,
    archive: TemporaryDirectory[str],
    archive_root: Path,
    repository: Path,
    provider_commit: str,
    artifact_dir: Path,
) -> ProviderSubmission:
    catalog = _read_json(_contained_file(archive_root, _CATALOG_FILE, "catalog"))
    schemas = _certification_schemas()
    _validate_schema(catalog, schemas["provider_catalog"], "catalog")
    catalog_data = _mapping(catalog, "catalog")
    provider = _mapping(catalog_data["provider"], "catalog")
    provider_name = _string(provider["name"], "catalog")
    release = _string(provider["release"], "catalog")

    build_path = _contained_file(
        archive_root, _string(catalog_data["build_record"], "catalog"), "build"
    )
    build = _read_json(build_path)
    _validate_schema(build, schemas["candidate_build"], "build")
    build_data = _mapping(build, "build")
    _verify_build_identity(build_data, provider_name, release)
    source_commit = _source_commit(build_data)
    _verify_source_ancestor(repository, source_commit, provider_commit)
    source_root = Path(archive.name) / "source"
    source_root.mkdir()
    source_tar = Path(archive.name) / "source.tar"
    _archive_commit(repository, source_commit, source_tar)
    _extract_archive(source_tar, source_root)

    operators = _load_operators(archive_root, source_root, catalog_data)
    baselines = _load_baselines(operators, schemas["numerical_baseline"], provider_name, release)
    artifacts = _load_artifacts(build_data, artifact_dir)
    return ProviderSubmission(
        provider=provider_name,
        release=release,
        submission_commit=f"git-sha1:{provider_commit}",
        source_commit=f"git-sha1:{source_commit}",
        archive_root=archive_root,
        source_root=source_root,
        operators=operators,
        artifacts=artifacts,
        baseline_cases=baselines,
        _archive=archive,
    )


def _certification_schemas() -> dict[str, dict[str, object]]:
    with materialize_certification_profile() as paths:
        return {name: _mapping(_read_json(path), "schema") for name, path in paths.items()}


def _load_operators(
    archive_root: Path, source_root: Path, catalog: Mapping[str, object]
) -> tuple[CatalogEntry, ...]:
    raw_operators = _mapping(catalog["operators"], "catalog")
    entries: list[CatalogEntry] = []
    for key, raw_entry in raw_operators.items():
        operator_id, separator, operator_version = key.rpartition("@")
        if not separator or not operator_id or not operator_version:
            raise _error("submission_schema_invalid", "catalog operator key is invalid", "catalog")
        entry = _mapping(raw_entry, "catalog")
        manifest_path = _contained_file(
            archive_root, _string(entry["manifest"], "catalog"), "manifest", operator_id
        )
        _validate_manifest_sources(source_root, manifest_path, operator_id)
        baseline_path = _contained_file(
            archive_root, _string(entry["baseline"], "catalog"), "baseline", operator_id
        )
        entries.append(CatalogEntry(operator_id, operator_version, manifest_path, baseline_path))
    return tuple(entries)


def _validate_manifest_sources(source_root: Path, manifest_path: Path, operator_id: str) -> None:
    manifest = _mapping(_read_json(manifest_path), "manifest", operator_id)
    implementation = manifest.get("implementation")
    if not isinstance(implementation, dict):
        return
    source_files = implementation.get("source_files")
    if not isinstance(source_files, list):
        return
    for source_file in source_files:
        if not isinstance(source_file, str):
            raise _error("submission_path_invalid", "manifest source path is invalid", "manifest", operator_id)
        _contained_file(source_root, source_file, "source", operator_id)


def _load_baselines(
    operators: tuple[CatalogEntry, ...],
    schema: Mapping[str, object],
    provider: str,
    release: str,
) -> tuple[BaselineCase, ...]:
    cases: list[BaselineCase] = []
    for entry in operators:
        baseline = _read_json(entry.baseline_path, entry.operator_id)
        _validate_schema(baseline, schema, "baseline", entry.operator_id)
        baseline_data = _mapping(baseline, "baseline", entry.operator_id)
        if (
            baseline_data["provider"] != provider
            or baseline_data["release"] != release
        ):
            raise _error("submission_identity_mismatch", "baseline identity does not match catalog", "baseline", entry.operator_id)
        for raw_case in cast(list[object], baseline_data["cases"]):
            case = _mapping(raw_case, "baseline", entry.operator_id)
            cases.append(
                BaselineCase(
                    operator_id=_string(case["operator_id"], "baseline", entry.operator_id),
                    operator_version=_string(case["operator_version"], "baseline", entry.operator_id),
                    parameters=_mapping(case["parameters"], "baseline", entry.operator_id),
                    input=_mapping(case["input"], "baseline", entry.operator_id),
                    expected=_mapping(case["expected"], "baseline", entry.operator_id),
                    tolerance=_mapping(case["tolerance"], "baseline", entry.operator_id),
                )
            )
    return tuple(cases)


def _verify_build_identity(build: Mapping[str, object], provider: str, release: str) -> None:
    if build["provider"] != provider or build["release"] != release:
        raise _error("submission_identity_mismatch", "build identity does not match catalog", "build")


def _source_commit(build: Mapping[str, object]) -> str:
    source_commit = _string(build["source_commit"], "build")
    if not source_commit.startswith("git-sha1:"):
        raise _error("submission_identity_mismatch", "source commit is not a SHA-1 commit", "build")
    source_hash = source_commit.removeprefix("git-sha1:")
    if _FULL_SHA1.fullmatch(source_hash) is None:
        raise _error("submission_identity_mismatch", "source commit is not a full lowercase SHA-1", "build")
    return source_hash


def _verify_source_ancestor(repository: Path, source_commit: str, submission_commit: str) -> None:
    result = _git(repository, ["rev-parse", "--verify", f"{source_commit}^{{commit}}"])
    if result.returncode != 0 or result.stdout.strip() != source_commit:
        raise _error("submission_identity_mismatch", "source commit did not resolve exactly", "build")
    result = _git(repository, ["merge-base", "--is-ancestor", source_commit, submission_commit])
    if result.returncode != 0:
        raise _error("submission_identity_mismatch", "source commit is not an ancestor of submission", "build")


def _load_artifacts(build: Mapping[str, object], artifact_dir: Path) -> tuple[BuildArtifact, ...]:
    raw_artifacts = cast(list[object], build["artifacts"])
    filenames: set[str] = set()
    artifacts: list[BuildArtifact] = []
    for raw_artifact in raw_artifacts:
        artifact = _mapping(raw_artifact, "artifact")
        filename = _string(artifact["filename"], "artifact")
        if filename in filenames or Path(filename).name != filename:
            raise _error("submission_schema_invalid", "artifact filenames must be unique basenames", "artifact")
        filenames.add(filename)
        wheel_path = artifact_dir / filename
        try:
            if wheel_path.is_symlink() or not wheel_path.exists() or not _is_regular_file(wheel_path):
                raise _error("artifact_missing", f"artifact is missing: {filename}", "artifact")
            actual_digest = _sha256_file(wheel_path)
        except OSError as exc:
            raise _error("artifact_missing", f"artifact is missing: {filename}", "artifact") from exc
        digest = _string(artifact["digest"], "artifact")
        if actual_digest != digest:
            raise _error("artifact_digest_mismatch", f"artifact digest differs: {filename}", "artifact")
        artifacts.append(
            BuildArtifact(
                distribution=_string(artifact["distribution"], "artifact"),
                version=_string(artifact["version"], "artifact"),
                filename=filename,
                role=_string(artifact["role"], "artifact"),
                digest=digest,
                wheel_path=wheel_path,
            )
        )
    return tuple(artifacts)


def _contained_file(
    root: Path, relative_path: str, stage: str, operator_id: str | None = None
) -> Path:
    pure_path = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise _error("submission_path_invalid", "submission path is not normalized", stage, operator_id)
    candidate = root.joinpath(*pure_path.parts)
    if (
        not candidate.is_relative_to(root)
        or candidate.is_symlink()
        or not candidate.exists()
        or not _is_regular_file(candidate)
    ):
        raise _error("submission_path_invalid", "submission path is not a regular archive file", stage, operator_id)
    return candidate


def _read_json(path: Path, operator_id: str | None = None) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _error("submission_json_invalid", f"invalid JSON: {path.name}", "json", operator_id) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validate_schema(
    instance: object, schema: Mapping[str, object], stage: str, operator_id: str | None = None
) -> None:
    try:
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)
    except ValidationError as exc:
        raise _error("submission_schema_invalid", "submission does not match its schema", stage, operator_id) from exc


def _mapping(value: object, stage: str, operator_id: str | None = None) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _error("submission_schema_invalid", "submission object is required", stage, operator_id)
    return cast(dict[str, object], value)


def _string(value: object, stage: str, operator_id: str | None = None) -> str:
    if not isinstance(value, str):
        raise _error("submission_schema_invalid", "submission string is required", stage, operator_id)
    return value


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _error(
    code: str, message: str, stage: str, operator_id: str | None = None
) -> OperatorCertificationError:
    return OperatorCertificationError(code, message, stage=stage, operator_id=operator_id)
