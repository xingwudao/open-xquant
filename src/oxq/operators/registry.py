"""Atomic local publication and lookup for certified operator bindings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    FormatChecker,
    SchemaError,
    ValidationError,
)

from oxq.operators.certification import _snapshot_contract_surface
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import ResearchCertification
from oxq.operators.resources import materialize_certification_profile

try:
    import fcntl
except ImportError:  # pragma: no cover - supported release targets are POSIX
    fcntl = None  # type: ignore[assignment]


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
_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
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
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


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

    with _publication_lock(provider_dir):
        if _lexists(release_dir):
            return _existing_or_conflict(prepared, release_dir)

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
            os.replace(staging_dir, release_dir)
            staging_dir = None
            _fsync_directory(provider_dir)
        except OSError:
            if _lexists(release_dir):
                return _existing_or_conflict(prepared, release_dir)
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
    if not _valid_provider_release(result.provider, result.release):
        raise _input_error("provider or release identity is invalid")
    if not _SHA1_PATTERN.fullmatch(result.submission_commit) or not _SHA1_PATTERN.fullmatch(
        result.source_commit
    ):
        raise _input_error("certification commit identity is invalid")
    if not result.operators or not result.baseline_cases or not result.baseline_results:
        raise _input_error("research certification must not be empty")

    try:
        binding_schema = _binding_schema()
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

    identities: set[tuple[str, str]] = set()
    binding_bytes: dict[str, bytes] = {}
    binding_values: dict[tuple[str, str], Mapping[str, object]] = {}
    for candidate in result.operators:
        binding = candidate.binding
        operator_id = binding.get("operator_id")
        operator_version = binding.get("operator_version")
        if not isinstance(operator_id, str) or not isinstance(operator_version, str):
            raise _input_error("certified binding identity is invalid")
        identity = (operator_id, operator_version)
        if identity in identities or not _valid_operator_identity(*identity):
            raise _input_error("certified operator identity is duplicated or invalid", operator_id)
        if binding.get("certification_state") != "research-certified":
            raise _input_error("publisher accepts only research-certified bindings", operator_id)
        if (
            candidate.manifest.get("operator_id") != operator_id
            or candidate.manifest.get("operator_version") != operator_version
        ):
            raise _input_error("manifest and binding identities do not match", operator_id)
        if binding.get("source_commit") != f"git-sha1:{result.source_commit}":
            raise _input_error("binding source commit does not match certification", operator_id)
        _validate_schema(
            binding,
            binding_schema,
            code="certification_input_invalid",
            message="research-certified binding is invalid",
            operator_id=operator_id,
        )
        identities.add(identity)
        binding_values[identity] = binding
        filename = _binding_filename(*identity)
        if filename in binding_bytes:
            raise _input_error("certified binding filename collides", operator_id)
        binding_bytes[filename] = _json_bytes(binding)

    case_counts = {identity: 0 for identity in identities}
    for case in result.baseline_cases:
        identity = (case.operator_id, case.operator_version)
        if identity not in case_counts:
            raise _input_error("baseline case does not identify a certified operator")
        case_counts[identity] += 1

    result_counts = {identity: 0 for identity in identities}
    cases_by_identity: dict[tuple[str, str], list[dict[str, object]]] = {
        identity: [] for identity in identities
    }
    result_keys: set[tuple[str, str, str]] = set()
    for baseline in result.baseline_results:
        identity = (baseline.operator_id, baseline.operator_version)
        result_key = (*identity, baseline.case_id)
        if (
            identity not in result_counts
            or baseline.status != "passed"
            or not baseline.case_id
            or result_key in result_keys
        ):
            raise _input_error("numerical baseline results are incomplete or invalid")
        result_keys.add(result_key)
        result_counts[identity] += 1
        cases_by_identity[identity].append(
            {"case_id": baseline.case_id, "status": "passed"}
        )
    if len(result.baseline_results) != len(result.baseline_cases) or any(
        case_counts[identity] == 0
        or case_counts[identity] != result_counts[identity]
        for identity in identities
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

    entry_artifacts: tuple[dict[str, object], ...] = tuple(
        {
            "distribution": artifact.distribution,
            "version": artifact.version,
            "filename": artifact.filename,
            "role": artifact.role,
            "build_identifier": artifact.build_identifier,
            "digest": artifact.digest,
        }
        for artifact in sorted(
            result.artifacts,
            key=lambda item: (item.role, item.distribution, item.version, item.filename),
        )
    )
    if any(not _valid_entry_artifact(item) for item in entry_artifacts):
        raise _input_error("certification artifact provenance is invalid")

    return _PreparedCertification(
        result=result,
        binding_bytes=MappingProxyType(binding_bytes),
        record_operators=tuple(record_operators),
        entry_artifacts=entry_artifacts,
    )


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
        "source_commit": f"git-sha1:{result.source_commit}",
        "state": "research-certified",
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
        "submission_commit": f"git-sha1:{result.submission_commit}",
        "source_commit": f"git-sha1:{result.source_commit}",
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
            or entry["source_commit"] != record["source_commit"]
            or record["state"] != "research-certified"
        ):
            raise ValueError("registry entry and record do not match")

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
        if seen != set(record_by_identity) or set(actual_files) != expected_files:
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
                files[f"bindings/{binding.name}"] = binding.read_bytes()
        elif stat.S_ISREG(mode):
            files[item.name] = item.read_bytes()
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
            if release_dir.name.startswith(".") or release_dir.name == ".publish.lock":
                continue
            if not _is_real_directory(release_dir) or not _SEMVER_PATTERN.fullmatch(
                release_dir.name
            ):
                raise ValueError("registry release directory is invalid")
            yield release_dir


@contextmanager
def _publication_lock(provider_dir: Path) -> Iterator[None]:
    lock_path = provider_dir / ".publish.lock"
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(lock_path, threading.Lock())
    with thread_lock:
        try:
            with lock_path.open("a+b") as stream:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OperatorCertificationError:
            raise
        except OSError:
            raise _error(
                "certification_publish_failed",
                "certification publication lock is unavailable",
            ) from None


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
        ).validate(value)
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


def _valid_provider_release(provider: str, release: str) -> bool:
    return bool(
        _PROVIDER_PATTERN.fullmatch(provider)
        and _SEMVER_PATTERN.fullmatch(release)
    )


def _valid_operator_identity(operator_id: str, operator_version: str) -> bool:
    return bool(
        _OPERATOR_ID_PATTERN.fullmatch(operator_id)
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
