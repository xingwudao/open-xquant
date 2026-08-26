"""Atomic local publication and lookup for certified operator bindings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast

from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    FormatChecker,
    SchemaError,
    ValidationError,
)

from oxq.operators.certification import (
    _load_reference_validator,
    _snapshot_contract_surface,
    _validate_binding_semantics,
    _validate_manifest_semantics,
)
from oxq.operators.certification import (
    _load_schema as _load_contract_schema,
)
from oxq.operators.certification import (
    _validate_schema as _validate_contract_schema,
)
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BaselineCase,
    BaselineResult,
    BuildArtifact,
    CatalogEntry,
    ContractCandidate,
    ResearchCertification,
)
from oxq.operators.resources import materialize_certification_profile

_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_SEMVER_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:(?:0|[1-9][0-9]*)|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:(?:0|[1-9][0-9]*)|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_OPERATOR_ID_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
    r"(?:\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*)+$"
)
_SHA1_PATTERN = re.compile(r"^git-sha1:[0-9a-f]{40}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_RESEARCH_STATES = {
    "research-certified",
    "runtime-certified",
    "ml-certified",
}
_ENTRY_FIELDS = {
    "schema_version",
    "provider",
    "release",
    "submission_commit",
    "source_commit",
    "certification_record",
    "certification_record_digest",
    "artifacts",
    "operators",
}
_ENTRY_OPERATOR_FIELDS = {
    "operator_id",
    "operator_version",
    "binding",
    "binding_digest",
}
_ENTRY_ARTIFACT_FIELDS = {
    "distribution",
    "version",
    "filename",
    "role",
    "build_identifier",
    "digest",
}


@dataclass(frozen=True)
class PublishedCertification:
    """One immutable provider-release publication."""

    release_dir: Path
    record: Mapping[str, object]
    registry_entry: Mapping[str, object]
    bindings: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class _PreparedCertification:
    result: ResearchCertification
    binding_bytes: Mapping[str, bytes]
    record_operators: tuple[dict[str, object], ...]
    entry_artifacts: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _RenderedPublication:
    record: dict[str, object]
    entry: dict[str, object]
    files: Mapping[str, bytes]


def publish_certification(
    result: ResearchCertification,
    output_dir: str | Path,
) -> PublishedCertification:
    """Atomically publish one fully passed research certification."""
    prepared = _prepare_certification(result)
    output_root = Path(output_dir).expanduser().resolve()
    provider_dir = output_root / result.provider
    release_dir = provider_dir / result.release
    _require_contained(output_root, release_dir)

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        provider_dir.mkdir(exist_ok=True)
        if not _is_real_directory(output_root) or not _is_real_directory(provider_dir):
            raise OSError("certification output is not a directory")
        _fsync_directory(output_root)
    except OSError:
        raise _error(
            "certification_publish_failed",
            "certification output directory is unavailable",
        ) from None

    if _lexists(release_dir):
        return _durable_existing_or_conflict(prepared, release_dir, provider_dir)

    rendered = _render_publication(prepared, _utc_now())
    staging_dir: Path | None = None
    try:
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{result.release}.staging-",
                dir=provider_dir,
            )
        )
        _write_staging(staging_dir, rendered)
        _fsync_directory(provider_dir)
        try:
            os.replace(staging_dir, release_dir)
        except OSError:
            if _lexists(release_dir):
                return _durable_existing_or_conflict(
                    prepared,
                    release_dir,
                    provider_dir,
                )
            raise _error(
                "certification_publish_failed",
                "certification publication failed before atomic commit",
            ) from None
        staging_dir = None
        try:
            _fsync_directory(provider_dir)
        except OSError:
            raise _error(
                "certification_publish_failed",
                "certification was renamed but directory durability was not confirmed",
            ) from None
    except OperatorCertificationError:
        raise
    except OSError:
        raise _error(
            "certification_publish_failed",
            "certification publication failed before atomic commit",
        ) from None
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)

    return _published_from_rendered(release_dir, rendered)


class CertificationRegistry:
    """Deterministically scan immutable provider-release registry entries."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()

    def get(
        self,
        operator_id: str,
        operator_version: str,
    ) -> Mapping[str, object] | None:
        """Return a verified research-capable binding, or ``None``."""
        if not _valid_operator_identity(operator_id, operator_version):
            raise _error("registry_invalid", "registry lookup identity is invalid")
        if not _lexists(self.output_dir):
            return None
        if not _is_real_directory(self.output_dir):
            raise _error("registry_invalid", "certification registry is invalid")

        matches: list[Mapping[str, object]] = []
        try:
            binding_schema = _binding_schema()
            record_schema = _certification_record_schema()
        except (OperatorCertificationError, OSError, ValueError, TypeError, UnicodeError):
            raise _error(
                "registry_invalid",
                "certification registry schemas are unavailable",
            ) from None
        try:
            release_dirs = tuple(_iter_release_directories(self.output_dir))
            for release_dir in release_dirs:
                publication = _read_publication(
                    release_dir,
                    binding_schema=binding_schema,
                    record_schema=record_schema,
                )
                for binding in publication.bindings:
                    if (
                        binding["operator_id"] == operator_id
                        and binding["operator_version"] == operator_version
                    ):
                        matches.append(binding)
        except OperatorCertificationError:
            raise
        except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
            raise _error("registry_invalid", "certification registry is invalid") from None

        if len(matches) > 1:
            raise _error(
                "registry_identity_collision",
                "operator identity appears in more than one certification release",
                operator_id,
            )
        return matches[0] if matches else None


def _prepare_certification(result: ResearchCertification) -> _PreparedCertification:
    if not isinstance(result, ResearchCertification):
        raise _input_error("publisher requires a research certification result")
    if (
        not isinstance(result.provider, str)
        or not isinstance(result.release, str)
        or not isinstance(result.submission_commit, str)
        or not isinstance(result.source_commit, str)
        or not isinstance(result.source_root, Path)
        or not isinstance(result.operators, tuple)
        or not isinstance(result.artifacts, tuple)
        or not isinstance(result.baseline_cases, tuple)
        or not isinstance(result.baseline_results, tuple)
    ):
        raise _input_error("research certification field types are invalid")
    if not _valid_provider_release(result.provider, result.release):
        raise _input_error("provider or release identity is invalid")
    if not _SHA1_PATTERN.fullmatch(result.submission_commit) or not _SHA1_PATTERN.fullmatch(
        result.source_commit
    ):
        raise _input_error("certification commit identity is invalid")
    if not result.operators or not result.baseline_cases or not result.baseline_results:
        raise _input_error("research certification must not be empty")

    entry_artifacts, implementation_artifacts = _prepare_artifacts(result.artifacts)

    identities, binding_values, binding_bytes = _prepare_candidates(
        result,
        implementation_artifacts,
    )

    case_counts = {identity: 0 for identity in identities}
    declared_cases: set[tuple[str, str, str]] = set()
    for case in result.baseline_cases:
        if (
            not isinstance(case, BaselineCase)
            or not isinstance(case.case_id, str)
            or not case.case_id
            or not isinstance(case.operator_id, str)
            or not isinstance(case.operator_version, str)
            or not isinstance(case.parameters, Mapping)
            or not isinstance(case.input, Mapping)
            or not isinstance(case.expected, Mapping)
            or not isinstance(case.tolerance, Mapping)
        ):
            raise _input_error("declared numerical baseline is invalid")
        identity = (case.operator_id, case.operator_version)
        case_identity = (*identity, case.case_id)
        if identity not in case_counts or case_identity in declared_cases:
            raise _input_error("baseline case does not identify a certified operator")
        declared_cases.add(case_identity)
        case_counts[identity] += 1

    cases_by_identity: dict[tuple[str, str], list[dict[str, object]]] = {
        identity: [] for identity in identities
    }
    passed_cases: set[tuple[str, str, str]] = set()
    for baseline in result.baseline_results:
        if (
            not isinstance(baseline, BaselineResult)
            or not isinstance(baseline.operator_id, str)
            or not isinstance(baseline.operator_version, str)
            or not isinstance(baseline.case_id, str)
            or not isinstance(baseline.status, str)
        ):
            raise _input_error("numerical baseline result fields are invalid")
        identity = (baseline.operator_id, baseline.operator_version)
        result_key = (*identity, baseline.case_id)
        if (
            identity not in identities
            or baseline.status != "passed"
            or not baseline.case_id
            or result_key in passed_cases
        ):
            raise _input_error("numerical baseline results are incomplete or invalid")
        passed_cases.add(result_key)
        cases_by_identity[identity].append(
            {"case_id": baseline.case_id, "status": "passed"}
        )
    if declared_cases != passed_cases or any(
        case_counts[identity] == 0 for identity in identities
    ):
        raise _input_error("not every certified operator baseline passed")

    record_operators: list[dict[str, object]] = []
    for identity in sorted(identities):
        binding = binding_values[identity]
        filename = _binding_filename(*identity)
        manifest_digest = binding.get("manifest_digest")
        implementation_digest = binding.get("implementation_digest")
        if not isinstance(manifest_digest, str) or not isinstance(
            implementation_digest, str
        ):
            raise _input_error("binding provenance digest is invalid", identity[0])
        record_operators.append(
            {
                "operator_id": identity[0],
                "operator_version": identity[1],
                "manifest_digest": manifest_digest,
                "implementation_digest": implementation_digest,
                "binding_digest": _sha256(binding_bytes[filename]),
                "baseline_cases": sorted(
                    cases_by_identity[identity],
                    key=lambda item: cast(str, item["case_id"]),
                ),
            }
        )

    return _PreparedCertification(
        result=result,
        binding_bytes=MappingProxyType(binding_bytes),
        record_operators=tuple(record_operators),
        entry_artifacts=entry_artifacts,
    )


def _prepare_candidates(
    result: ResearchCertification,
    implementation_artifacts: tuple[BuildArtifact, ...],
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], Mapping[str, object]],
    dict[str, bytes],
]:
    identities: set[tuple[str, str]] = set()
    matched_implementation_artifacts: set[tuple[str, str, str]] = set()
    binding_bytes: dict[str, bytes] = {}
    binding_values: dict[tuple[str, str], Mapping[str, object]] = {}
    try:
        with _snapshot_contract_surface() as (surface_bytes, surface_paths):
            manifest_schema = _load_contract_schema(
                surface_bytes["operator_manifest_schema"],
                "manifest_schema_invalid",
                "manifest",
            )
            binding_schema = _load_contract_schema(
                surface_bytes["operator_binding_schema"],
                "binding_validation_failed",
                "binding",
            )
            validator = _load_reference_validator(
                surface_bytes["reference_validator"],
                surface_paths["reference_validator"],
            )
            for candidate in result.operators:
                if (
                    not isinstance(candidate, ContractCandidate)
                    or not isinstance(candidate.manifest, Mapping)
                    or not isinstance(candidate.binding, Mapping)
                    or not isinstance(candidate.manifest_path, Path)
                    or not isinstance(candidate.implementation_artifact, Path)
                ):
                    raise _input_error("certified operator fields are invalid")
                binding = candidate.binding
                operator_id = binding.get("operator_id")
                operator_version = binding.get("operator_version")
                if not isinstance(operator_id, str) or not isinstance(
                    operator_version, str
                ):
                    raise _input_error("certified binding identity is invalid")
                identity = (operator_id, operator_version)
                if identity in identities or not _valid_operator_identity(*identity):
                    raise _input_error(
                        "certified operator identity is duplicated or invalid",
                        operator_id,
                    )
                if binding.get("certification_state") != "research-certified":
                    raise _input_error(
                        "publisher accepts only research-certified bindings",
                        operator_id,
                    )
                if binding.get("source_commit") != result.source_commit:
                    raise _input_error(
                        "binding source commit does not match certification",
                        operator_id,
                    )
                manifest, manifest_bytes = _read_candidate_manifest(
                    candidate,
                    operator_id,
                    operator_version,
                )
                matched = _verify_candidate_provenance(
                    result,
                    candidate,
                    implementation_artifacts,
                    operator_id,
                    operator_version,
                )
                try:
                    _validate_contract_schema(
                        manifest,
                        manifest_schema,
                        code="manifest_schema_invalid",
                        message="operator manifest does not match frozen schema",
                        stage="manifest",
                        operator_id=operator_id,
                    )
                    _validate_manifest_semantics(validator, manifest, operator_id)
                    _validate_contract_schema(
                        _thaw_json_mapping(binding),
                        binding_schema,
                        code="binding_validation_failed",
                        message="operator binding validation failed",
                        stage="binding",
                        operator_id=operator_id,
                    )
                    with tempfile.TemporaryDirectory(
                        prefix="oxq-certified-manifest-"
                    ) as directory:
                        manifest_snapshot = Path(directory) / "operator.json"
                        manifest_snapshot.write_bytes(manifest_bytes)
                        entry = CatalogEntry(
                            operator_id=operator_id,
                            operator_version=operator_version,
                            manifest_path=manifest_snapshot,
                            baseline_path=manifest_snapshot,
                        )
                        _validate_binding_semantics(
                            validator,
                            binding,
                            manifest,
                            entry,
                            result.source_root,
                            matched.wheel_path,
                            surface_paths,
                        )
                except OperatorCertificationError:
                    raise _input_error(
                        "certified manifest or binding violates the frozen contract",
                        operator_id,
                    ) from None
                matched = _verify_candidate_provenance(
                    result,
                    candidate,
                    implementation_artifacts,
                    operator_id,
                    operator_version,
                )
                matched_implementation_artifacts.add(
                    (matched.distribution, matched.version, matched.filename)
                )
                identities.add(identity)
                binding_values[identity] = binding
                filename = _binding_filename(*identity)
                if filename in binding_bytes:
                    raise _input_error(
                        "certified binding filename collides",
                        operator_id,
                    )
                binding_bytes[filename] = _json_bytes(binding)
    except OperatorCertificationError as error:
        if error.code == "certification_input_invalid":
            raise
        raise _error(
            "certification_publish_failed",
            "frozen certification contract is unavailable",
        ) from None
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        raise _error(
            "certification_publish_failed",
            "frozen certification contract is unavailable",
        ) from None

    expected_implementation_artifacts = {
        (artifact.distribution, artifact.version, artifact.filename)
        for artifact in implementation_artifacts
    }
    if matched_implementation_artifacts != expected_implementation_artifacts:
        raise _input_error("certification contains an unrelated implementation artifact")
    return identities, binding_values, binding_bytes


def _read_candidate_manifest(
    candidate: ContractCandidate,
    operator_id: str,
    operator_version: str,
) -> tuple[dict[str, object], bytes]:
    try:
        manifest_bytes = _read_regular_file(candidate.manifest_path)
        manifest = _strict_json_object(manifest_bytes)
        retained_manifest = _thaw_json_mapping(candidate.manifest)
    except (
        OSError,
        ValueError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
    ):
        raise _input_error(
            "certified operator manifest is unavailable or invalid",
            operator_id,
        ) from None
    if (
        manifest != retained_manifest
        or candidate.binding.get("manifest_digest") != _sha256(manifest_bytes)
        or manifest.get("operator_id") != operator_id
        or manifest.get("operator_version") != operator_version
        or manifest.get("distribution") != candidate.binding.get("distribution")
    ):
        raise _input_error(
            "manifest bytes do not match certified binding provenance",
            operator_id,
        )
    return manifest, manifest_bytes


def _prepare_artifacts(
    artifacts: tuple[BuildArtifact, ...],
) -> tuple[tuple[dict[str, object], ...], tuple[BuildArtifact, ...]]:
    if not artifacts or not all(isinstance(item, BuildArtifact) for item in artifacts):
        raise _input_error("certification requires verified build artifacts")
    identities: set[tuple[str, str, str]] = set()
    filenames: set[str] = set()
    digests: set[str] = set()
    build_identifiers: set[str] = set()
    records: list[dict[str, object]] = []
    implementations: list[BuildArtifact] = []
    for artifact in artifacts:
        value: dict[str, object] = {
            "distribution": artifact.distribution,
            "version": artifact.version,
            "filename": artifact.filename,
            "role": artifact.role,
            "build_identifier": artifact.build_identifier,
            "digest": artifact.digest,
        }
        identity = (artifact.distribution, artifact.version, artifact.role)
        if (
            not isinstance(artifact.wheel_path, Path)
            or not _valid_entry_artifact(value)
            or artifact.wheel_path.name != artifact.filename
            or identity in identities
            or artifact.filename in filenames
            or artifact.digest in digests
            or artifact.build_identifier in build_identifiers
        ):
            raise _input_error("certification artifact identity is invalid or duplicated")
        try:
            actual_digest = _sha256(_read_regular_file(artifact.wheel_path))
        except OSError:
            raise _input_error("certification artifact is unavailable") from None
        if actual_digest != artifact.digest:
            raise _input_error("certification artifact digest changed after validation")
        identities.add(identity)
        filenames.add(artifact.filename)
        digests.add(artifact.digest)
        build_identifiers.add(artifact.build_identifier)
        records.append(value)
        if artifact.role == "implementation":
            implementations.append(artifact)
    if not implementations:
        raise _input_error("certification requires an implementation artifact")
    return (
        tuple(
            sorted(
                records,
                key=lambda item: (
                    cast(str, item["role"]),
                    cast(str, item["distribution"]),
                    cast(str, item["version"]),
                    cast(str, item["filename"]),
                ),
            )
        ),
        tuple(implementations),
    )


def _verify_candidate_provenance(
    result: ResearchCertification,
    candidate: ContractCandidate,
    artifacts: tuple[BuildArtifact, ...],
    operator_id: str,
    operator_version: str,
) -> BuildArtifact:
    manifest, _ = _read_candidate_manifest(
        candidate,
        operator_id,
        operator_version,
    )
    binding = candidate.binding
    implementation = manifest.get("implementation")
    if (
        not isinstance(implementation, Mapping)
    ):
        raise _input_error(
            "manifest bytes do not match certified binding provenance",
            operator_id,
        )
    source_files = implementation.get("source_files")
    if (
        not isinstance(source_files, list)
        or not source_files
        or not all(isinstance(path, str) for path in source_files)
        or len(source_files) != len(set(cast(list[str], source_files)))
    ):
        raise _input_error("manifest source file list is invalid", operator_id)
    try:
        actual_source_tree_digest = _source_tree_digest(
            result.source_root,
            cast(list[str], source_files),
        )
    except (OSError, ValueError, TypeError):
        raise _input_error(
            "certified source tree is unavailable or invalid",
            operator_id,
        ) from None
    if (
        implementation.get("source_commit") != result.source_commit
        or implementation.get("source_commit") != binding.get("source_commit")
        or implementation.get("source_tree_digest") != actual_source_tree_digest
        or implementation.get("source_tree_digest")
        != binding.get("source_tree_digest")
    ):
        raise _input_error(
            "manifest source provenance does not match certified source tree",
            operator_id,
        )
    try:
        candidate_path = candidate.implementation_artifact.resolve(strict=True)
    except OSError:
        raise _input_error("implementation artifact path is unavailable", operator_id) from None
    matching_artifacts: list[BuildArtifact] = []
    for artifact in artifacts:
        try:
            artifact_path = artifact.wheel_path.resolve(strict=True)
        except OSError:
            continue
        if (
            candidate_path == artifact_path
            and manifest.get("distribution") == artifact.distribution
            and binding.get("distribution") == artifact.distribution
            and binding.get("distribution_version") == artifact.version
            and binding.get("implementation_digest") == artifact.digest
            and implementation.get("package_version") == artifact.version
            and implementation.get("implementation_digest") == artifact.digest
            and implementation.get("build_identifier") == artifact.build_identifier
        ):
            matching_artifacts.append(artifact)
    if len(matching_artifacts) != 1:
        raise _input_error(
            "certified binding does not match exactly one implementation artifact",
            operator_id,
        )
    return matching_artifacts[0]


def _source_tree_digest(root: Path, source_files: list[str]) -> str:
    normalized_paths: list[tuple[str, list[str]]] = []
    for relative in sorted(source_files):
        path = PurePosixPath(relative)
        parts = relative.split("/")
        if (
            not relative
            or path.is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("source path is not normalized")
        normalized_paths.append((relative, parts))

    root_descriptor = _open_directory_no_follow(root)
    digest = hashlib.sha256()
    try:
        for relative, parts in normalized_paths:
            source_bytes = _read_relative_regular_file(root_descriptor, parts)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(source_bytes).hexdigest().encode("ascii"))
            digest.update(b"\n")
    finally:
        _close_descriptors([root_descriptor])
    return f"sha256:{digest.hexdigest()}"


def _open_directory_no_follow(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("path is not a real directory")
    except BaseException:
        _close_descriptors([descriptor])
        raise
    return descriptor


def _read_relative_regular_file(root_descriptor: int, parts: list[str]) -> bytes:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    opened_directories: list[int] = []
    current_descriptor = root_descriptor
    leaf_descriptor: int | None = None
    try:
        for part in parts[:-1]:
            current_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=current_descriptor,
            )
            opened_directories.append(current_descriptor)
            if not stat.S_ISDIR(os.fstat(current_descriptor).st_mode):
                raise OSError("source parent is not a real directory")
        leaf_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current_descriptor,
        )
        return _read_regular_descriptor(leaf_descriptor)
    finally:
        descriptors = (
            [leaf_descriptor] if leaf_descriptor is not None else []
        ) + list(reversed(opened_directories))
        _close_descriptors(descriptors)


def _close_descriptors(descriptors: Sequence[int]) -> None:
    first_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _render_publication(
    prepared: _PreparedCertification,
    certified_at: str,
) -> _RenderedPublication:
    result = prepared.result
    record: dict[str, object] = {
        "schema_version": 1,
        "certifier": "open-xquant-local",
        "certified_at": certified_at,
        "provider": result.provider,
        "release": result.release,
        "submission_commit": result.submission_commit,
        "source_commit": result.source_commit,
        "state": "research-certified",
        "artifacts": list(prepared.entry_artifacts),
        "operators": list(prepared.record_operators),
    }
    try:
        record_schema = _certification_record_schema()
    except (
        OperatorCertificationError,
        OSError,
        ValueError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        raise _error(
            "certification_publish_failed",
            "certification schemas are unavailable",
        ) from None
    _validate_schema(
        record,
        record_schema,
        code="certification_input_invalid",
        message="certification record does not match its schema",
    )
    record_bytes = _json_bytes(record)
    entry_operators = [
        {
            "operator_id": operator["operator_id"],
            "operator_version": operator["operator_version"],
            "binding": "bindings/"
            + _binding_filename(
                cast(str, operator["operator_id"]),
                cast(str, operator["operator_version"]),
            ),
            "binding_digest": operator["binding_digest"],
        }
        for operator in prepared.record_operators
    ]
    entry: dict[str, object] = {
        "schema_version": 1,
        "provider": result.provider,
        "release": result.release,
        "submission_commit": result.submission_commit,
        "source_commit": result.source_commit,
        "certification_record": "certification-record.json",
        "certification_record_digest": _sha256(record_bytes),
        "artifacts": list(prepared.entry_artifacts),
        "operators": entry_operators,
    }
    files = {
        "certification-record.json": record_bytes,
        "registry-entry.json": _json_bytes(entry),
        **{
            f"bindings/{filename}": value
            for filename, value in prepared.binding_bytes.items()
        },
    }
    return _RenderedPublication(record=record, entry=entry, files=files)


def _write_staging(staging_dir: Path, rendered: _RenderedPublication) -> None:
    bindings_dir = staging_dir / "bindings"
    bindings_dir.mkdir()
    for relative_path, value in sorted(rendered.files.items()):
        path = staging_dir / relative_path
        _require_contained(staging_dir, path)
        _write_file(path, value)
    _fsync_directory(bindings_dir)
    _fsync_directory(staging_dir)


def _write_file(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _existing_or_conflict(
    prepared: _PreparedCertification,
    release_dir: Path,
) -> PublishedCertification:
    try:
        existing = _read_publication(
            release_dir,
            binding_schema=_binding_schema(),
            record_schema=_certification_record_schema(),
        )
        certified_at = existing.record.get("certified_at")
        if not isinstance(certified_at, str):
            raise ValueError("existing timestamp is invalid")
        expected = _render_publication(prepared, certified_at)
        if _publication_files(release_dir) == dict(expected.files):
            return existing
    except (OperatorCertificationError, OSError, ValueError, TypeError, UnicodeError):
        pass
    raise _error(
        "certification_conflict",
        "provider release already has different certification bytes",
    )


def _durable_existing_or_conflict(
    prepared: _PreparedCertification,
    release_dir: Path,
    provider_dir: Path,
) -> PublishedCertification:
    publication = _existing_or_conflict(prepared, release_dir)
    try:
        _fsync_directory(provider_dir)
    except OSError:
        raise _error(
            "certification_publish_failed",
            "existing certification directory durability was not confirmed",
        ) from None
    return publication


def _read_publication(
    release_dir: Path,
    *,
    binding_schema: Mapping[str, object],
    record_schema: Mapping[str, object],
) -> PublishedCertification:
    try:
        if not _is_real_directory(release_dir):
            raise ValueError("release is not a real directory")
        provider = release_dir.parent.name
        release = release_dir.name
        if not _valid_provider_release(provider, release):
            raise ValueError("release path identity is invalid")
        actual_files = _publication_files(release_dir)
        entry_bytes = actual_files.get("registry-entry.json")
        record_bytes = actual_files.get("certification-record.json")
        if entry_bytes is None or record_bytes is None:
            raise ValueError("publication metadata is missing")
        entry = _strict_json_object(entry_bytes)
        record = _strict_json_object(record_bytes)
        if _json_bytes(entry) != entry_bytes or _json_bytes(record) != record_bytes:
            raise ValueError("publication metadata is not canonical JSON")
        _validate_registry_entry(entry, provider, release)
        _validate_schema(
            record,
            record_schema,
            code="registry_invalid",
            message="certification record is invalid",
        )
        if (
            entry["certification_record_digest"] != _sha256(record_bytes)
            or entry["certification_record"] != "certification-record.json"
            or entry["provider"] != record["provider"]
            or entry["release"] != record["release"]
            or entry["submission_commit"] != record["submission_commit"]
            or entry["source_commit"] != record["source_commit"]
            or entry["artifacts"] != record["artifacts"]
            or record["certifier"] != "open-xquant-local"
            or record["state"] != "research-certified"
        ):
            raise ValueError("registry entry and record do not match")

        record_artifacts = cast(list[dict[str, object]], record["artifacts"])
        implementation_artifacts = _validate_record_artifacts(record_artifacts)

        entry_operators = cast(list[dict[str, object]], entry["operators"])
        record_operators = cast(list[dict[str, object]], record["operators"])
        record_by_identity: dict[tuple[str, str], dict[str, object]] = {}
        for operator in record_operators:
            identity = _operator_identity(operator)
            if identity in record_by_identity:
                raise ValueError("record operator identity collision")
            record_by_identity[identity] = operator

        bindings: list[Mapping[str, object]] = []
        expected_files = {"certification-record.json", "registry-entry.json"}
        seen: set[tuple[str, str]] = set()
        matched_implementation_artifacts: set[tuple[str, str, str]] = set()
        for operator in entry_operators:
            identity = _operator_identity(operator)
            if identity in seen:
                raise ValueError("registry operator identity collision")
            seen.add(identity)
            expected_relative = "bindings/" + _binding_filename(*identity)
            if operator["binding"] != expected_relative:
                raise ValueError("registry binding path is invalid")
            expected_files.add(expected_relative)
            raw_binding = actual_files.get(expected_relative)
            if raw_binding is None or operator["binding_digest"] != _sha256(raw_binding):
                raise ValueError("registry binding digest is invalid")
            binding = _strict_json_object(raw_binding)
            if _json_bytes(binding) != raw_binding:
                raise ValueError("binding is not canonical JSON")
            _validate_schema(
                binding,
                binding_schema,
                code="registry_invalid",
                message="published binding is invalid",
                operator_id=identity[0],
            )
            if _operator_identity(binding) != identity or binding.get(
                "certification_state"
            ) not in _RESEARCH_STATES:
                raise ValueError("binding is not research capable")
            matching_artifacts = [
                artifact
                for artifact in implementation_artifacts
                if artifact["distribution"] == binding.get("distribution")
                and artifact["version"] == binding.get("distribution_version")
                and artifact["digest"] == binding.get("implementation_digest")
            ]
            if (
                binding.get("source_commit") != record["source_commit"]
                or len(matching_artifacts) != 1
            ):
                raise ValueError("binding does not match certified implementation provenance")
            matched = matching_artifacts[0]
            matched_implementation_artifacts.add(
                (
                    cast(str, matched["distribution"]),
                    cast(str, matched["version"]),
                    cast(str, matched["filename"]),
                )
            )
            record_operator = record_by_identity.get(identity)
            if (
                record_operator is None
                or record_operator.get("binding_digest") != operator["binding_digest"]
                or record_operator.get("manifest_digest")
                != binding.get("manifest_digest")
                or record_operator.get("implementation_digest")
                != binding.get("implementation_digest")
            ):
                raise ValueError("record and binding provenance do not match")
            bindings.append(_freeze_json_mapping(binding))
        expected_implementation_artifacts = {
            (
                cast(str, artifact["distribution"]),
                cast(str, artifact["version"]),
                cast(str, artifact["filename"]),
            )
            for artifact in implementation_artifacts
        }
        if (
            seen != set(record_by_identity)
            or set(actual_files) != expected_files
            or matched_implementation_artifacts != expected_implementation_artifacts
        ):
            raise ValueError("publication layout is not exact")
        return PublishedCertification(
            release_dir=release_dir,
            record=_freeze_json_mapping(record),
            registry_entry=_freeze_json_mapping(entry),
            bindings=tuple(bindings),
        )
    except OperatorCertificationError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        raise _error("registry_invalid", "certification registry entry is invalid") from None


def _validate_registry_entry(
    entry: Mapping[str, object],
    provider: str,
    release: str,
) -> None:
    if set(entry) != _ENTRY_FIELDS or entry.get("schema_version") != 1:
        raise ValueError("registry entry fields are invalid")
    if (
        entry.get("provider") != provider
        or entry.get("release") != release
        or entry.get("certification_record") != "certification-record.json"
        or not isinstance(entry.get("submission_commit"), str)
        or not re.fullmatch(r"git-sha1:[0-9a-f]{40}", cast(str, entry["submission_commit"]))
        or not isinstance(entry.get("source_commit"), str)
        or not re.fullmatch(r"git-sha1:[0-9a-f]{40}", cast(str, entry["source_commit"]))
        or not isinstance(entry.get("certification_record_digest"), str)
        or not _DIGEST_PATTERN.fullmatch(cast(str, entry["certification_record_digest"]))
    ):
        raise ValueError("registry entry identity is invalid")
    artifacts = entry.get("artifacts")
    operators = entry.get("operators")
    if (
        not isinstance(artifacts, list)
        or not all(isinstance(item, dict) and _valid_entry_artifact(item) for item in artifacts)
        or not isinstance(operators, list)
        or not operators
    ):
        raise ValueError("registry entry collections are invalid")
    for operator in operators:
        if not isinstance(operator, dict) or set(operator) != _ENTRY_OPERATOR_FIELDS:
            raise ValueError("registry operator entry is invalid")
        operator_id = operator.get("operator_id")
        operator_version = operator.get("operator_version")
        binding = operator.get("binding")
        digest = operator.get("binding_digest")
        if (
            not isinstance(operator_id, str)
            or not isinstance(operator_version, str)
            or not _valid_operator_identity(operator_id, operator_version)
            or not isinstance(binding, str)
            or not isinstance(digest, str)
            or not _DIGEST_PATTERN.fullmatch(digest)
        ):
            raise ValueError("registry operator identity is invalid")


def _validate_record_artifacts(
    artifacts: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not artifacts:
        raise ValueError("certification artifacts are empty")
    identities: set[tuple[str, str, str]] = set()
    filenames: set[str] = set()
    digests: set[str] = set()
    build_identifiers: set[str] = set()
    implementations: list[dict[str, object]] = []
    for artifact in artifacts:
        if not _valid_entry_artifact(artifact):
            raise ValueError("certification artifact is invalid")
        identity = (
            cast(str, artifact["distribution"]),
            cast(str, artifact["version"]),
            cast(str, artifact["role"]),
        )
        filename = cast(str, artifact["filename"])
        digest = cast(str, artifact["digest"])
        build_identifier = cast(str, artifact["build_identifier"])
        if (
            identity in identities
            or filename in filenames
            or digest in digests
            or build_identifier in build_identifiers
        ):
            raise ValueError("certification artifact identity is duplicated")
        identities.add(identity)
        filenames.add(filename)
        digests.add(digest)
        build_identifiers.add(build_identifier)
        if artifact["role"] == "implementation":
            implementations.append(artifact)
    if not implementations:
        raise ValueError("certification implementation artifact is missing")
    return implementations


def _valid_entry_artifact(value: Mapping[str, object]) -> bool:
    if set(value) != _ENTRY_ARTIFACT_FIELDS:
        return False
    strings = [value.get(field) for field in _ENTRY_ARTIFACT_FIELDS]
    if not all(isinstance(item, str) and item for item in strings):
        return False
    filename = cast(str, value["filename"])
    return (
        value["role"] in {"implementation", "runtime-dependency"}
        and _PROVIDER_PATTERN.fullmatch(cast(str, value["distribution"])) is not None
        and _SEMVER_PATTERN.fullmatch(cast(str, value["version"])) is not None
        and Path(filename).name == filename
        and "/" not in filename
        and "\\" not in filename
        and _DIGEST_PATTERN.fullmatch(cast(str, value["digest"])) is not None
    )


def _publication_files(release_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for item in release_dir.iterdir():
        mode = item.lstat().st_mode
        if item.name == "bindings" and stat.S_ISDIR(mode):
            for binding in item.iterdir():
                if not stat.S_ISREG(binding.lstat().st_mode):
                    raise ValueError("binding path is not a regular file")
                files[f"bindings/{binding.name}"] = _read_regular_file(binding)
        elif stat.S_ISREG(mode):
            files[item.name] = _read_regular_file(item)
        else:
            raise ValueError("publication contains a link or unexpected directory")
    return files


def _iter_release_directories(output_root: Path) -> Iterator[Path]:
    for provider_dir in sorted(output_root.iterdir(), key=lambda path: path.name):
        if provider_dir.name.startswith("."):
            continue
        if not _is_real_directory(provider_dir) or not _PROVIDER_PATTERN.fullmatch(
            provider_dir.name
        ):
            raise ValueError("registry provider directory is invalid")
        for release_dir in sorted(provider_dir.iterdir(), key=lambda path: path.name):
            if release_dir.name.startswith("."):
                continue
            if not _is_real_directory(release_dir) or not _SEMVER_PATTERN.fullmatch(
                release_dir.name
            ):
                raise ValueError("registry release directory is invalid")
            yield release_dir


def _binding_schema() -> Mapping[str, object]:
    with _snapshot_contract_surface() as (surface_bytes, _):
        return _strict_json_object(surface_bytes["operator_binding_schema"])


def _certification_record_schema() -> Mapping[str, object]:
    with materialize_certification_profile() as paths:
        return _strict_json_object(paths["certification_record"].read_bytes())


def _validate_schema(
    value: Mapping[str, object],
    schema: Mapping[str, object],
    *,
    code: str,
    message: str,
    operator_id: str | None = None,
) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).validate(_thaw_json_mapping(value))
    except (SchemaError, ValidationError, TypeError, ValueError, RecursionError):
        raise _error(code, message, operator_id) from None


def _strict_json_object(value: bytes) -> dict[str, object]:
    parsed = json.loads(
        value.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonstandard_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("JSON root is not an object")
    return cast(dict[str, object], parsed)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    del value
    raise ValueError("non-standard JSON number")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                _thaw_json_mapping(value),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise _input_error("certification contains unsupported JSON values") from None


def _thaw_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _thaw_json_value(item) for key, item in value.items()}


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return _thaw_json_mapping(cast(Mapping[str, object], value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("unsupported JSON value")


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {key: _freeze_json_value(item) for key, item in value.items()}
    )


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_json_mapping(cast(Mapping[str, object], value))
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _published_from_rendered(
    release_dir: Path,
    rendered: _RenderedPublication,
) -> PublishedCertification:
    bindings: list[Mapping[str, object]] = []
    operators = cast(list[dict[str, object]], rendered.record["operators"])
    for operator in operators:
        filename = _binding_filename(
            cast(str, operator["operator_id"]),
            cast(str, operator["operator_version"]),
        )
        binding = _strict_json_object(rendered.files[f"bindings/{filename}"])
        bindings.append(_freeze_json_mapping(binding))
    return PublishedCertification(
        release_dir=release_dir,
        record=_freeze_json_mapping(rendered.record),
        registry_entry=_freeze_json_mapping(rendered.entry),
        bindings=tuple(bindings),
    )


def _operator_identity(value: Mapping[str, object]) -> tuple[str, str]:
    operator_id = value.get("operator_id")
    operator_version = value.get("operator_version")
    if not isinstance(operator_id, str) or not isinstance(operator_version, str):
        raise ValueError("operator identity is invalid")
    if not _valid_operator_identity(operator_id, operator_version):
        raise ValueError("operator identity is invalid")
    return operator_id, operator_version


def _binding_filename(operator_id: str, operator_version: str) -> str:
    if not _valid_operator_identity(operator_id, operator_version):
        raise ValueError("operator identity is invalid")
    return f"{operator_id}@{operator_version}.binding.json"


def _valid_provider_release(provider: object, release: object) -> bool:
    return bool(
        isinstance(provider, str)
        and isinstance(release, str)
        and _PROVIDER_PATTERN.fullmatch(provider)
        and _SEMVER_PATTERN.fullmatch(release)
    )


def _valid_operator_identity(operator_id: object, operator_version: object) -> bool:
    return bool(
        isinstance(operator_id, str)
        and isinstance(operator_version, str)
        and _OPERATOR_ID_PATTERN.fullmatch(operator_id)
        and _SEMVER_PATTERN.fullmatch(operator_version)
    )


def _require_contained(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        raise _input_error("certification output path escapes its root") from None


def _is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        return _read_regular_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_descriptor(descriptor: int) -> bytes:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise OSError("path is not a regular file")
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _input_error(
    message: str,
    operator_id: str | None = None,
) -> OperatorCertificationError:
    return _error("certification_input_invalid", message, operator_id)


def _error(
    code: str,
    message: str,
    operator_id: str | None = None,
) -> OperatorCertificationError:
    return OperatorCertificationError(
        code,
        message,
        stage="registry",
        operator_id=operator_id,
    )
