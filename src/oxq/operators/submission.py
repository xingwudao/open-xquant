"""Load one provider submission from an exact, local Git commit."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import stat
import subprocess
import zipfile
from collections.abc import Mapping
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import IO, cast

from jsonschema import Draft202012Validator, FormatChecker, ValidationError  # type: ignore[import-untyped]
from packaging.metadata import Metadata
from packaging.requirements import Requirement
from packaging.tags import Tag, parse_tag, sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

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
        ["git", "--no-replace-objects", "-C", str(repository), *args],
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
        ["git", "--no-replace-objects", "-C", str(repository), *args],
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

    _stream_git_blobs(repository, entries, destination)


def _safe_git_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return bool(
        relative_path and "\\" not in relative_path and not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _stream_git_blobs(
    repository: Path,
    entries: list[tuple[str, str, str]],
    destination: Path,
) -> None:
    if not entries:
        return
    try:
        process = subprocess.Popen(
            ["git", "--no-replace-objects", "-C", str(repository), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise _unreadable_git_blob_error() from exc
    if process.stdin is None or process.stdout is None:
        _abort_git_batch(process)
        raise _unreadable_git_blob_error()

    try:
        for mode, object_id, relative_path in entries:
            _request_git_blob(process.stdin, object_id)
            size = _read_git_blob_header(process.stdout, object_id)
            path = destination.joinpath(*PurePosixPath(relative_path).parts)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as output:
                    _copy_git_blob(process.stdout, output, size)
                path.chmod(_REGULAR_GIT_MODES[mode])
            except OSError as exc:
                raise _error(
                    "submission_path_invalid",
                    "provider commit tree cannot be materialized safely",
                    "archive",
                ) from exc
            _read_git_blob_terminator(process.stdout)
        try:
            process.stdin.close()
        except OSError as exc:
            raise _unreadable_git_blob_error() from exc
        if process.wait() != 0:
            raise _unreadable_git_blob_error()
    finally:
        _abort_git_batch(process)


def _request_git_blob(stream: IO[bytes], object_id: str) -> None:
    try:
        stream.write(object_id.encode("ascii") + b"\n")
        stream.flush()
    except (OSError, UnicodeError) as exc:
        raise _unreadable_git_blob_error() from exc


def _read_git_blob_header(stream: IO[bytes], expected_id: str) -> int:
    try:
        header = stream.readline().decode("ascii").removesuffix("\n")
        object_id, object_type, raw_size = header.split(" ", 2)
        size = int(raw_size)
        if object_id != expected_id or object_type != "blob" or size < 0:
            raise ValueError("unexpected Git object")
    except (OSError, UnicodeError, ValueError) as exc:
        raise _unreadable_git_blob_error() from exc
    return size


def _copy_git_blob(stream: IO[bytes], output: IO[bytes], size: int) -> None:
    remaining = size
    while remaining:
        try:
            chunk = stream.read(min(1024 * 1024, remaining))
        except OSError as exc:
            raise _unreadable_git_blob_error() from exc
        if not chunk:
            raise _unreadable_git_blob_error()
        output.write(chunk)
        remaining -= len(chunk)


def _read_git_blob_terminator(stream: IO[bytes]) -> None:
    try:
        terminator = stream.read(1)
    except OSError as exc:
        raise _unreadable_git_blob_error() from exc
    if terminator != b"\n":
        raise _unreadable_git_blob_error()


def _abort_git_batch(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait()
    except OSError:
        pass


def _unreadable_git_blob_error() -> OperatorCertificationError:
    return _error(
        "provider_commit_invalid",
        "provider commit contains an unreadable blob",
        "repository",
    )


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
    requirements_by_filename: dict[str, tuple[Requirement, ...]] = {}
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
        requirements_by_filename[filename] = _verify_wheel_identity(
            wheel_path,
            distribution,
            version,
            filename,
        )
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
    _verify_artifact_closure(artifacts, requirements_by_filename)
    return tuple(artifacts)


def _verify_wheel_identity(
    wheel_path: Path,
    distribution: str,
    version: str,
    filename: str,
) -> tuple[Requirement, ...]:
    try:
        if wheel_path.suffix != ".whl":
            raise zipfile.BadZipFile("artifact filename is not a wheel")
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel_files = [name for name in wheel.namelist() if _is_dist_info_file(name, "WHEEL")]
            metadata_files = [name for name in wheel.namelist() if _is_dist_info_file(name, "METADATA")]
            record_files = [name for name in wheel.namelist() if _is_dist_info_file(name, "RECORD")]
            if (
                len(wheel_files) != 1
                or len(metadata_files) != 1
                or len(record_files) != 1
                or PurePosixPath(wheel_files[0]).parent != PurePosixPath(metadata_files[0]).parent
                or PurePosixPath(wheel_files[0]).parent != PurePosixPath(record_files[0]).parent
            ):
                raise zipfile.BadZipFile("wheel metadata files are invalid")
            wheel_metadata = BytesParser(policy=policy.default).parsebytes(wheel.read(wheel_files[0]))
            package_metadata = Metadata.from_email(
                wheel.read(metadata_files[0]),
                validate=False,
            )
            wheel_version = wheel_metadata.get("Wheel-Version")
            wheel_version_match = re.fullmatch(r"([0-9]+)\.([0-9]+)", wheel_version) if isinstance(wheel_version, str) else None
            if wheel_version_match is None or int(wheel_version_match.group(1)) != 1:
                raise zipfile.BadZipFile("wheel version header is unsupported")
            wheel_tags: set[Tag] = set()
            for header in wheel_metadata.get_all("Tag", []):
                wheel_tags.update(parse_tag(header))
            filename_distribution, filename_version, _, filename_tags = parse_wheel_filename(filename)
            expected_dist_info_directory = (
                f"{str(filename_distribution).replace('-', '_')}-{str(filename_version).replace('-', '_')}.dist-info"
            )
            dist_info_directory = PurePosixPath(wheel_files[0]).parent.name
            build_version = Version(version)
            current_python = Version(platform.python_version())
            if package_metadata.requires_python is not None and current_python not in package_metadata.requires_python:
                raise ValueError("wheel requires a different Python runtime")
            compatible_tags = set(sys_tags())
            if (
                not wheel_tags
                or wheel_tags != filename_tags
                or not wheel_tags.intersection(compatible_tags)
                or not filename_tags.intersection(compatible_tags)
            ):
                raise zipfile.BadZipFile("wheel tags are incompatible")
            metadata_name = package_metadata.name
            metadata_version = package_metadata.version
            requirements = tuple(package_metadata.requires_dist or ())
    except (
        ExceptionGroup,
        InvalidWheelFilename,
        InvalidVersion,
        KeyError,
        NotImplementedError,
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
        or canonicalize_name(metadata_name) != canonicalize_name(distribution)
        or metadata_version != build_version
        or canonicalize_name(str(filename_distribution)) != canonicalize_name(distribution)
        or filename_version != build_version
        or dist_info_directory != expected_dist_info_directory
    ):
        raise _error(
            "artifact_identity_mismatch",
            f"wheel metadata differs from build record: {filename}",
            "artifact",
        )
    return requirements


def _verify_artifact_closure(
    artifacts: list[BuildArtifact],
    requirements_by_filename: Mapping[str, tuple[Requirement, ...]],
) -> None:
    available: dict[str, set[Version]] = {}
    requirements_by_distribution: dict[str, list[Requirement]] = {}
    try:
        for artifact in artifacts:
            artifact_distribution = canonicalize_name(artifact.distribution)
            available.setdefault(artifact_distribution, set()).add(Version(artifact.version))
            requirements_by_distribution.setdefault(artifact_distribution, []).extend(
                requirements_by_filename[artifact.filename]
            )
        if any(requirement.url is not None for requirements in requirements_by_distribution.values() for requirement in requirements):
            raise ValueError("direct-reference wheel dependencies are forbidden")

        pending_contexts = [(distribution, "") for distribution in available]
        evaluated_contexts = set(pending_contexts)
        while pending_contexts:
            distribution, extra = pending_contexts.pop()
            for requirement in requirements_by_distribution[distribution]:
                if requirement.marker is not None and not requirement.marker.evaluate({"extra": extra}):
                    continue
                dependency = canonicalize_name(requirement.name)
                versions = available.get(dependency, set())
                if not any(version in requirement.specifier for version in versions):
                    raise ValueError("wheel dependency is absent from artifact closure")
                for requested_extra in requirement.extras:
                    context = (dependency, canonicalize_name(requested_extra))
                    if context not in evaluated_contexts:
                        evaluated_contexts.add(context)
                        pending_contexts.append(context)
    except (InvalidVersion, KeyError, ValueError) as exc:
        raise _error(
            "artifact_invalid",
            "artifact dependency is not satisfied by verified wheels",
            "artifact",
        ) from exc


def _is_dist_info_file(name: str, filename: str) -> bool:
    parts = PurePosixPath(name).parts
    return len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == filename


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
