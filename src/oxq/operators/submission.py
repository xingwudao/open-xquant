"""Load one provider submission from an exact, local Git commit."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import zipfile
from collections.abc import Mapping
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker, ValidationError  # type: ignore[import-untyped]
from packaging.tags import parse_tag, sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BaselineCase,
    BuildArtifact,
    CatalogEntry,
    ProviderSubmission,
)
from oxq.operators.resources import materialize_certification_profile

_FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_COMPATIBILITY_ROOT = PurePosixPath("compat/open_xquant")
_CATALOG_FILE = "operator_catalog.json"
_CONTRACT_RELEASE = "1.0.0"
_REGULAR_GIT_MODES = {"100644": 0o644, "100755": 0o755}


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
        _materialize_commit(repository, provider_commit, archive_root)
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


def _git(repository: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def _git_bytes(
    repository: Path,
    args: list[str],
    *,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        input=input_bytes,
        capture_output=True,
    )


def _materialize_commit(repository: Path, commit: str, destination: Path) -> None:
    tree = _git_bytes(repository, ["ls-tree", "-rz", "--full-tree", commit])
    if tree.returncode != 0:
        raise _error(
            "provider_commit_invalid",
            "cannot read provider commit tree",
            "repository",
        )
    entries: list[tuple[str, str, str]] = []
    for raw_entry in tree.stdout.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = header.decode("ascii").split(" ", 2)
            relative_path = raw_path.decode("utf-8")
        except (UnicodeError, ValueError):
            continue
        if object_type != "blob" or mode not in _REGULAR_GIT_MODES:
            continue
        if not _safe_git_path(relative_path):
            continue
        entries.append((mode, object_id, relative_path))

    blobs = _read_git_blobs(repository, [object_id for _, object_id, _ in entries])
    try:
        for (mode, _, relative_path), blob in zip(entries, blobs, strict=True):
            path = destination.joinpath(*PurePosixPath(relative_path).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            path.chmod(_REGULAR_GIT_MODES[mode])
    except OSError as exc:
        raise _error(
            "submission_path_invalid",
            "provider commit tree cannot be materialized safely",
            "archive",
        ) from exc


def _safe_git_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return bool(
        relative_path and "\\" not in relative_path and not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _read_git_blobs(repository: Path, object_ids: list[str]) -> list[bytes]:
    if not object_ids:
        return []
    result = _git_bytes(
        repository,
        ["cat-file", "--batch"],
        input_bytes=("\n".join(object_ids) + "\n").encode("ascii"),
    )
    if result.returncode != 0:
        raise _error(
            "provider_commit_invalid",
            "cannot read provider commit blobs",
            "repository",
        )
    blobs: list[bytes] = []
    offset = 0
    try:
        for expected_id in object_ids:
            header_end = result.stdout.index(b"\n", offset)
            header = result.stdout[offset:header_end].decode("ascii")
            object_id, object_type, raw_size = header.split(" ", 2)
            if object_id != expected_id or object_type != "blob":
                raise ValueError("unexpected Git object")
            size = int(raw_size)
            start = header_end + 1
            end = start + size
            if result.stdout[end : end + 1] != b"\n":
                raise ValueError("truncated Git object")
            blobs.append(result.stdout[start:end])
            offset = end + 1
        if offset != len(result.stdout):
            raise ValueError("unexpected trailing Git output")
    except (UnicodeError, ValueError):
        raise _error(
            "provider_commit_invalid",
            "provider commit contains an unreadable blob",
            "repository",
        ) from None
    return blobs


def _load_archive(
    *,
    archive: TemporaryDirectory[str],
    archive_root: Path,
    repository: Path,
    provider_commit: str,
    artifact_dir: Path,
) -> ProviderSubmission:
    catalog_path = _COMPATIBILITY_ROOT / _CATALOG_FILE
    catalog = _read_json(_contained_file(archive_root, catalog_path.as_posix(), "catalog"))
    compatibility_root = archive_root.joinpath(*_COMPATIBILITY_ROOT.parts)
    schemas = _certification_schemas()
    _validate_schema(catalog, schemas["provider_catalog"], "catalog")
    catalog_data = _mapping(catalog, "catalog")
    if catalog_data["contract_version"] != _CONTRACT_RELEASE:
        raise _error(
            "submission_schema_invalid",
            "catalog contract release is unsupported",
            "catalog",
        )
    provider = _mapping(catalog_data["provider"], "catalog")
    provider_name = _string(provider["name"], "catalog")
    release = _string(provider["release"], "catalog")

    build_path = _contained_file(
        compatibility_root,
        _string(catalog_data["build_record"], "catalog"),
        "build",
    )
    build = _read_json(build_path)
    _validate_schema(build, schemas["candidate_build"], "build")
    build_data = _mapping(build, "build")
    _verify_build_identity(build_data, provider_name, release)
    source_commit = _source_commit(build_data)
    _verify_source_ancestor(repository, source_commit, provider_commit)
    source_root = Path(archive.name) / "source"
    source_root.mkdir()
    _materialize_commit(repository, source_commit, source_root)

    operators = _load_operators(compatibility_root, source_root, catalog_data)
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


def _load_operators(archive_root: Path, source_root: Path, catalog: Mapping[str, object]) -> tuple[CatalogEntry, ...]:
    raw_operators = _mapping(catalog["operators"], "catalog")
    entries: list[CatalogEntry] = []
    for key, raw_entry in raw_operators.items():
        operator_id, separator, operator_version = key.rpartition("@")
        if not separator or not operator_id or not operator_version:
            raise _error("submission_schema_invalid", "catalog operator key is invalid", "catalog")
        entry = _mapping(raw_entry, "catalog")
        manifest_path = _contained_file(archive_root, _string(entry["manifest"], "catalog"), "manifest", operator_id)
        _validate_manifest_sources(source_root, manifest_path, operator_id)
        baseline_path = _contained_file(archive_root, _string(entry["baseline"], "catalog"), "baseline", operator_id)
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
    case_identities: set[tuple[str, str, str]] = set()
    entries_by_path: dict[Path, list[CatalogEntry]] = {}
    for entry in operators:
        entries_by_path.setdefault(entry.baseline_path, []).append(entry)
    for baseline_path, entries in entries_by_path.items():
        error_operator_id = entries[0].operator_id
        referenced_identities = {(entry.operator_id, entry.operator_version) for entry in entries}
        covered_identities: set[tuple[str, str]] = set()
        baseline = _read_json(baseline_path, error_operator_id)
        _validate_schema(baseline, schema, "baseline", error_operator_id)
        baseline_data = _mapping(baseline, "baseline", error_operator_id)
        if baseline_data["provider"] != provider or baseline_data["release"] != release:
            raise _error(
                "submission_identity_mismatch",
                "baseline identity does not match catalog",
                "baseline",
                error_operator_id,
            )
        for raw_case in cast(list[object], baseline_data["cases"]):
            case = _mapping(raw_case, "baseline", error_operator_id)
            case_id = _string(case["case_id"], "baseline", error_operator_id)
            operator_id = _string(case["operator_id"], "baseline", error_operator_id)
            operator_version = _string(case["operator_version"], "baseline", error_operator_id)
            operator_identity = (operator_id, operator_version)
            identity = (operator_id, operator_version, case_id)
            if operator_identity not in referenced_identities or identity in case_identities:
                raise _error(
                    "submission_identity_mismatch",
                    "baseline case identity does not match its catalog entry",
                    "baseline",
                    error_operator_id,
                )
            covered_identities.add(operator_identity)
            case_identities.add(identity)
            cases.append(
                BaselineCase(
                    case_id=case_id,
                    operator_id=operator_id,
                    operator_version=operator_version,
                    parameters=_mapping(case["parameters"], "baseline", error_operator_id),
                    input=_mapping(case["input"], "baseline", error_operator_id),
                    expected=_mapping(case["expected"], "baseline", error_operator_id),
                    tolerance=_mapping(case["tolerance"], "baseline", error_operator_id),
                )
            )
        if covered_identities != referenced_identities:
            raise _error(
                "submission_identity_mismatch",
                "shared baseline does not cover every catalog identity",
                "baseline",
                error_operator_id,
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
        distribution = _string(artifact["distribution"], "artifact")
        version = _string(artifact["version"], "artifact")
        _verify_wheel_identity(wheel_path, distribution, version, filename)
        artifacts.append(
            BuildArtifact(
                distribution=distribution,
                version=version,
                filename=filename,
                role=_string(artifact["role"], "artifact"),
                build_identifier=_string(artifact["build_identifier"], "artifact"),
                digest=digest,
                wheel_path=wheel_path,
            )
        )
    return tuple(artifacts)


def _verify_wheel_identity(
    wheel_path: Path,
    distribution: str,
    version: str,
    filename: str,
) -> None:
    try:
        if wheel_path.suffix != ".whl":
            raise zipfile.BadZipFile("artifact filename is not a wheel")
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel_files = [name for name in wheel.namelist() if _is_dist_info_file(name, "WHEEL")]
            metadata_files = [name for name in wheel.namelist() if _is_dist_info_file(name, "METADATA")]
            if (
                len(wheel_files) != 1
                or len(metadata_files) != 1
                or PurePosixPath(wheel_files[0]).parent != PurePosixPath(metadata_files[0]).parent
            ):
                raise zipfile.BadZipFile("wheel metadata files are invalid")
            wheel_metadata = BytesParser(policy=policy.default).parsebytes(wheel.read(wheel_files[0]))
            package_metadata = BytesParser(policy=policy.default).parsebytes(wheel.read(metadata_files[0]))
            if not wheel_metadata.get("Wheel-Version"):
                raise zipfile.BadZipFile("wheel version header is missing")
            wheel_tags = set()
            for header in wheel_metadata.get_all("Tag", []):
                wheel_tags.update(parse_tag(header))
            _, _, _, filename_tags = parse_wheel_filename(filename)
            compatible_tags = set(sys_tags())
            if not wheel_tags or not wheel_tags.intersection(compatible_tags) or not filename_tags.intersection(compatible_tags):
                raise zipfile.BadZipFile("wheel tags are incompatible")
            metadata_name = package_metadata.get("Name")
            metadata_version = package_metadata.get("Version")
    except (
        InvalidWheelFilename,
        KeyError,
        OSError,
        UnicodeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise _error(
            "artifact_invalid",
            f"artifact is not a valid wheel: {filename}",
            "artifact",
        ) from exc
    if (
        not isinstance(metadata_name, str)
        or _canonical_distribution(metadata_name) != _canonical_distribution(distribution)
        or metadata_version != version
    ):
        raise _error(
            "artifact_identity_mismatch",
            f"wheel metadata differs from build record: {filename}",
            "artifact",
        )


def _is_dist_info_file(name: str, filename: str) -> bool:
    parts = PurePosixPath(name).parts
    return len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == filename


def _canonical_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _contained_file(root: Path, relative_path: str, stage: str, operator_id: str | None = None) -> Path:
    pure_path = PurePosixPath(relative_path)
    if not relative_path or "\\" in relative_path or pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise _error("submission_path_invalid", "submission path is not normalized", stage, operator_id)
    candidate = root.joinpath(*pure_path.parts)
    if not candidate.is_relative_to(root) or candidate.is_symlink() or not candidate.exists() or not _is_regular_file(candidate):
        raise _error("submission_path_invalid", "submission path is not a regular archive file", stage, operator_id)
    return candidate


def _read_json(path: Path, operator_id: str | None = None) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
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


def _validate_schema(instance: object, schema: Mapping[str, object], stage: str, operator_id: str | None = None) -> None:
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


def _error(code: str, message: str, stage: str, operator_id: str | None = None) -> OperatorCertificationError:
    return OperatorCertificationError(code, message, stage=stage, operator_id=operator_id)
