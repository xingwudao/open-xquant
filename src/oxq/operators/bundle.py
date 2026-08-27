"""Strict deterministic certification bundle export and validation."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker, SchemaError, ValidationError  # type: ignore[import-untyped]

from oxq.operators.errors import OperatorCertificationError
from oxq.operators.formats import canonical_json_bytes, safe_relative_path, sha256_bytes, strict_json_object
from oxq.operators.models import CertificationTarget
from oxq.operators.registry import PublishedCertification, import_certification_publication, read_certification_publication
from oxq.operators.resources import materialize_operator_distribution_profile

_MAX_MEMBERS = 512
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_RATIO = 100
_MAX_BUNDLE_BYTES = 32 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ValidatedCertificationBundle:
    """One evidence-complete, hostile-input-validated certification ZIP."""

    bundle_path: Path
    provider: str
    release: str
    target: CertificationTarget
    operator_count: int
    digest: str
    members: Mapping[str, bytes]


def export_certification_bundle(
    *,
    provider: str,
    release: str,
    registry_dir: str | Path,
    manifest_dir: str | Path,
    baseline_files: Sequence[str | Path],
    target: CertificationTarget,
    output_path: str | Path,
) -> ValidatedCertificationBundle:
    """Write one deterministic ZIP from a targeted local certification record."""
    if not isinstance(target, CertificationTarget):
        raise ValueError("certification target is invalid")
    publication = read_certification_publication(Path(registry_dir).expanduser().resolve() / provider / release)
    record = publication.record
    if record.get("schema_version") != 2 or _record_target(record) != target:
        raise ValueError("publication does not match the requested target")
    source_members = _export_members(publication.release_dir, record, manifest_dir, baseline_files)
    source_members["bundle-manifest.json"] = canonical_json_bytes(
        _bundle_manifest(provider, release, target, publication.release_dir, record, source_members)
    )
    output = Path(output_path).expanduser().resolve()
    _write_zip(output, source_members)
    return validate_certification_bundle(output)


def validate_certification_bundle(bundle_path: str | Path) -> ValidatedCertificationBundle:
    """Read a ZIP without extraction and prove its complete v2 evidence closure."""
    path = Path(bundle_path).expanduser().resolve()
    try:
        if not path.is_file() or path.stat().st_size > _MAX_BUNDLE_BYTES:
            raise ValueError("bundle size is invalid")
        with zipfile.ZipFile(path) as archive:
            if archive.comment:
                raise ValueError("bundle comment is forbidden")
            infos = archive.infolist()
            if len(infos) > _MAX_MEMBERS:
                raise ValueError("bundle has too many members")
            members = _read_members(archive, infos)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise ValueError("certification bundle is not a valid ZIP") from None
    try:
        manifest_bytes = members["bundle-manifest.json"]
        manifest = strict_json_object(manifest_bytes)
        if canonical_json_bytes(manifest) != manifest_bytes:
            raise ValueError("bundle manifest is not canonical JSON")
        _validate_bundle_manifest(manifest)
        provider = _string(manifest, "provider")
        release = _string(manifest, "release")
        target = _record_target(manifest)
        operator_count = manifest.get("operator_count")
        if not isinstance(operator_count, int) or isinstance(operator_count, bool):
            raise ValueError("bundle operator count is invalid")
        _validate_evidence(members, manifest, provider, release, target, operator_count)
    except (KeyError, TypeError, ValueError, SchemaError, ValidationError):
        raise ValueError("certification bundle evidence is invalid") from None
    return ValidatedCertificationBundle(
        bundle_path=path,
        provider=provider,
        release=release,
        target=target,
        operator_count=operator_count,
        digest=sha256_bytes(path.read_bytes()),
        members=MappingProxyType(dict(members)),
    )


def materialize_validated_bundle(bundle: ValidatedCertificationBundle, destination: Path) -> None:
    """Materialize already validated members without ever extracting the ZIP."""
    if not isinstance(bundle, ValidatedCertificationBundle):
        raise ValueError("validated bundle is required")
    root = destination.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        for name, value in sorted(bundle.members.items()):
            relative = safe_relative_path(name)
            path = root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(value)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def import_certification_bundle(
    bundle_path: str | Path,
    output_dir: str | Path,
    *,
    trust_unsigned_bundle: bool,
    bundle_store: str | Path | None = None,
) -> PublishedCertification:
    """Validate and atomically import a portable certification bundle.

    A bundle is audit evidence, not an installed runtime.  Only its validated
    publication is committed to the supplied local certification registry.
    """
    if not trust_unsigned_bundle:
        raise OperatorCertificationError(
            "bundle_trust_required",
            "--trust-unsigned-bundle is required to import a bundle",
            stage="trust",
        )
    source = Path(bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if _is_within(source, output):
        raise OperatorCertificationError(
            "bundle_input_invalid",
            "bundle input must be outside the destination registry",
            stage="input",
        )
    if bundle_store is not None and _is_within(Path(bundle_store).expanduser().resolve(), output):
        raise OperatorCertificationError(
            "bundle_store_invalid",
            "audit bundle store must be outside the destination registry",
            stage="output",
        )
    try:
        bundle = validate_certification_bundle(source)
    except ValueError as exc:
        raise OperatorCertificationError(
            "bundle_invalid",
            "certification bundle is invalid",
            stage="validation",
        ) from exc

    staging: Path | None = None
    try:
        output.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{bundle.release}.bundle-import-", dir=output))
        staging.rmdir()
        materialize_validated_bundle(bundle, staging)
        # The full materialization is useful only for validation.  The registry
        # transaction copies and publishes the publication subtree atomically.
        provider_dir = staging / bundle.provider
        provider_dir.mkdir()
        publication = provider_dir / bundle.release
        (staging / "publication").replace(publication)
        imported = import_certification_publication(publication, output)
    except OperatorCertificationError:
        raise
    except (OSError, ValueError):
        raise OperatorCertificationError(
            "bundle_import_failed",
            "certification bundle import failed before registry commit",
            stage="import",
        ) from None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    if bundle_store is not None:
        try:
            _store_bundle(source, Path(bundle_store).expanduser().resolve())
        except OSError:
            raise OperatorCertificationError(
                "bundle_store_failed",
                "certification bundle was imported but audit storage failed",
                stage="audit",
            ) from None
    return imported


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _store_bundle(source: Path, store: Path) -> None:
    """Durably publish a byte-identical audit copy after registry success."""
    store.mkdir(parents=True, exist_ok=True)
    destination = store / source.name
    source_bytes = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != source_bytes:
            raise OSError("audit bundle name conflicts with different bytes")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{source.name}.", dir=store)
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "wb", closefd=True) as stream:
            stream.write(source_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != source_bytes:
                raise OSError("audit bundle name conflicts with different bytes") from None
        _fsync_directory(store)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _export_members(
    release_dir: Path,
    record: Mapping[str, object],
    manifest_dir: str | Path,
    baseline_files: Sequence[str | Path],
) -> dict[str, bytes]:
    members = {
        "publication/" + path.relative_to(release_dir).as_posix(): path.read_bytes()
        for path in sorted(release_dir.rglob("*"))
        if path.is_file()
    }
    expected_baselines = _file_descriptors(record, "baseline_sets")
    supplied: dict[str, bytes] = {}
    for value in baseline_files:
        path = Path(value).expanduser().resolve()
        raw = path.read_bytes()
        supplied[sha256_bytes(raw)] = raw
    for original, digest in expected_baselines.items():
        raw = supplied.get(digest)
        if raw is None:
            raise ValueError("required baseline evidence is missing")
        members["baselines/" + original] = raw
    root = Path(manifest_dir).expanduser().resolve()
    record_operators = _record_operators(record)
    for identity, operator in record_operators.items():
        filename = f"{identity[0]}@{identity[1]}.operator.json"
        matches: list[bytes] = []
        for path in sorted(root.rglob("*.json")):
            raw = path.read_bytes()
            try:
                parsed = strict_json_object(raw)
            except ValueError:
                continue
            if parsed.get("operator_id") == identity[0] and parsed.get("operator_version") == identity[1]:
                matches.append(raw)
        if len(matches) != 1:
            raise ValueError("required manifest evidence is missing or ambiguous")
        raw = matches[0]
        if sha256_bytes(raw) != operator.get("manifest_digest"):
            raise ValueError("manifest evidence digest does not match publication")
        members["manifests/" + filename] = raw
    return members


def _bundle_manifest(
    provider: str,
    release: str,
    target: CertificationTarget,
    release_dir: Path,
    record: Mapping[str, object],
    members: Mapping[str, bytes],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "bundle_type": "open-xquant-certification",
        "provider": provider,
        "release": release,
        "target": {"python_tag": target.python_tag, "abi_tag": target.abi_tag, "platform_tag": target.platform_tag},
        "publication_path": "publication",
        "registry_entry_digest": sha256_bytes(members["publication/registry-entry.json"]),
        "manifests": [
            {"path": f"manifests/{identity[0]}@{identity[1]}.operator.json", "digest": operator["manifest_digest"]}
            for identity, operator in sorted(_record_operators(record).items())
        ],
        "baseline_sets": [
            {"path": "baselines/" + path, "digest": digest} for path, digest in sorted(_file_descriptors(record, "baseline_sets").items())
        ],
        "operator_count": len(_record_operators(record)),
    }


def _read_members(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    total = 0
    for info in infos:
        _validate_info(info, members)
        total += info.file_size
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("bundle expanded size exceeds limit")
        with archive.open(info, "r") as stream:
            value = stream.read(info.file_size + 1)
        if len(value) != info.file_size:
            raise ValueError("bundle member size differs from its ZIP entry")
        members[info.filename] = value
    return members


def _validate_info(info: zipfile.ZipInfo, members: Mapping[str, bytes]) -> None:
    safe_relative_path(info.filename)
    if info.filename in members or info.is_dir() or info.extra or info.comment:
        raise ValueError("bundle member metadata is invalid")
    if info.flag_bits & 1 or info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError("bundle member compression is invalid")
    if stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG:
        raise ValueError("bundle member is not a regular file")
    if info.file_size > _MAX_MEMBER_BYTES:
        raise ValueError("bundle member exceeds expanded size limit")
    if info.file_size and (not info.compress_size or info.file_size > info.compress_size * _MAX_RATIO):
        raise ValueError("bundle member compression ratio exceeds limit")


def _validate_evidence(
    members: Mapping[str, bytes],
    manifest: Mapping[str, object],
    provider: str,
    release: str,
    target: CertificationTarget,
    operator_count: int,
) -> None:
    if manifest.get("publication_path") != "publication":
        raise ValueError("publication path is invalid")
    publication_members = {name.removeprefix("publication/"): value for name, value in members.items() if name.startswith("publication/")}
    if not publication_members:
        raise ValueError("publication evidence is missing")
    with tempfile.TemporaryDirectory(prefix="oxq-certification-bundle-") as temporary:
        root = Path(temporary) / provider / release
        root.mkdir(parents=True)
        for name, value in publication_members.items():
            safe_relative_path(name)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        publication = read_certification_publication(root)
    record = publication.record
    if record.get("schema_version") != 2 or _record_target(record) != target:
        raise ValueError("publication target is invalid")
    if sha256_bytes(members.get("publication/registry-entry.json", b"")) != manifest.get("registry_entry_digest"):
        raise ValueError("registry entry digest is invalid")
    operators = _record_operators(record)
    if len(operators) != operator_count:
        raise ValueError("operator count does not match publication")
    manifest_files = _file_descriptors(manifest, "manifests")
    baseline_files = _file_descriptors(manifest, "baseline_sets")
    expected_manifest_paths = {f"manifests/{key[0]}@{key[1]}.operator.json" for key in operators}
    expected_baseline_paths = {"baselines/" + key for key in _file_descriptors(record, "baseline_sets")}
    if set(manifest_files) != expected_manifest_paths or set(baseline_files) != expected_baseline_paths:
        raise ValueError("bundle evidence layout is incomplete")
    if set(members) != {"bundle-manifest.json", *{"publication/" + key for key in publication_members}, *manifest_files, *baseline_files}:
        raise ValueError("bundle layout is not exact")
    baseline_by_original = _file_descriptors(record, "baseline_sets")
    for path, digest in baseline_files.items():
        original = path.removeprefix("baselines/")
        raw = members[path]
        if digest != baseline_by_original.get(original) or sha256_bytes(raw) != digest:
            raise ValueError("baseline evidence digest is invalid")
    bindings = {(cast(str, item["operator_id"]), cast(str, item["operator_version"])): item for item in publication.bindings}
    artifacts = cast(list[Mapping[str, object]], record["artifacts"])
    for identity, operator in operators.items():
        path = f"manifests/{identity[0]}@{identity[1]}.operator.json"
        raw = members[path]
        parsed = strict_json_object(raw)
        binding = bindings.get(identity)
        if (
            binding is None
            or sha256_bytes(raw) != operator.get("manifest_digest")
            or manifest_files[path] != operator.get("manifest_digest")
        ):
            raise ValueError("manifest evidence digest is invalid")
        if parsed.get("operator_id") != identity[0] or parsed.get("operator_version") != identity[1]:
            raise ValueError("manifest identity is invalid")
        implementation = parsed.get("implementation")
        if (
            not isinstance(implementation, Mapping)
            or parsed.get("distribution") != binding.get("distribution")
            or implementation.get("source_commit") != record.get("source_commit")
            or implementation.get("implementation_digest") != binding.get("implementation_digest")
        ):
            raise ValueError("manifest provenance is invalid")
        if not any(
            artifact.get("distribution") == binding.get("distribution")
            and artifact.get("version") == binding.get("distribution_version")
            and artifact.get("digest") == binding.get("implementation_digest")
            for artifact in artifacts
        ):
            raise ValueError("implementation evidence is invalid")
        cases = operator.get("baseline_cases")
        if not isinstance(cases, tuple) or not cases:
            raise ValueError("operator does not have certified baseline evidence")
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("baseline case is invalid")
            original = case.get("baseline_path")
            index = case.get("case_index")
            if not isinstance(original, str) or not isinstance(index, int) or original not in baseline_by_original:
                raise ValueError("baseline case provenance is invalid")
            baseline = strict_json_object(members["baselines/" + original])
            all_cases = baseline.get("cases")
            if not isinstance(all_cases, list) or not 0 <= index < len(all_cases) or not isinstance(all_cases[index], dict):
                raise ValueError("baseline case location is invalid")
            evidence = cast(dict[str, object], all_cases[index])
            if (
                evidence.get("case_id") != case.get("case_id")
                or evidence.get("operator_id") != identity[0]
                or evidence.get("operator_version") != identity[1]
                or sha256_bytes(canonical_json_bytes(evidence)) != case.get("case_digest")
            ):
                raise ValueError("baseline case digest is invalid")


def _file_descriptors(value: Mapping[str, object], name: str) -> dict[str, str]:
    raw = value.get(name)
    if not isinstance(raw, (list, tuple)):
        raise ValueError("file descriptors are invalid")
    result: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("digest"), str):
            raise ValueError("file descriptor is invalid")
        path = item["path"]
        safe_relative_path(path)
        if path in result:
            raise ValueError("file descriptor path is duplicated")
        result[path] = cast(str, item["digest"])
    return result


def _record_operators(record: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, object]]:
    raw = record.get("operators")
    if not isinstance(raw, (list, tuple)):
        raise ValueError("record operators are invalid")
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("operator_id"), str)
            or not isinstance(item.get("operator_version"), str)
        ):
            raise ValueError("record operator is invalid")
        identity = (cast(str, item["operator_id"]), cast(str, item["operator_version"]))
        if identity in result:
            raise ValueError("record operator is duplicated")
        result[identity] = item
    return result


def _record_target(value: Mapping[str, object]) -> CertificationTarget:
    raw = value.get("target")
    if not isinstance(raw, Mapping):
        raise ValueError("target is invalid")
    return CertificationTarget.parse("-".join((_string(raw, "python_tag"), _string(raw, "abi_tag"), _string(raw, "platform_tag"))))


def _string(value: Mapping[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str):
        raise ValueError(f"{name} is invalid")
    return item


def _validate_bundle_manifest(value: Mapping[str, object]) -> None:
    with materialize_operator_distribution_profile() as paths:
        schema = strict_json_object(paths["certification_bundle_manifest"].read_bytes())
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    except (SchemaError, ValidationError, TypeError, ValueError):
        raise ValueError("bundle manifest does not match its schema") from None


def _write_zip(path: Path, members: Mapping[str, bytes]) -> None:
    if path.exists():
        raise ValueError("bundle output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info.extra = b""
            archive.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
