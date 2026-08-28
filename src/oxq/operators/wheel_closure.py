"""Validate an exact operator wheel closure before any wheel metadata is used."""

from __future__ import annotations

import base64
import csv
import hashlib
import stat
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.metadata import Metadata
from packaging.requirements import Requirement
from packaging.tags import Tag, parse_tag
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from oxq.operators.install_errors import install_error
from oxq.operators.release_index import ReleaseTarget, ReleaseWheel

_MAX_MEMBERS = 4096
_MAX_MEMBER_BYTES = 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_RATIO = 100


@dataclass(frozen=True)
class VerifiedWheel:
    """One archive whose bytes, ZIP structure, and wheel identity were checked."""

    path: Path
    distribution: str
    version: str
    filename: str
    requirements: tuple[Requirement, ...]


@dataclass(frozen=True)
class VerifiedWheelClosure:
    """The only wheel set eligible for later isolated extraction."""

    target: ReleaseTarget
    wheels: tuple[VerifiedWheel, ...]


def verify_wheel_closure(
    target: ReleaseTarget,
    wheel_paths: Iterable[str | Path],
    *,
    certified_artifacts: Iterable[object],
) -> VerifiedWheelClosure:
    """Fail closed unless local wheels exactly match both certified declarations."""
    try:
        declared = tuple(target.wheels)
        certified = tuple(certified_artifacts)
        paths = tuple(Path(path) for path in wheel_paths)
        _require_declared_artifacts_match(declared, certified)
        _require_exact_paths(declared, paths)
        verified = tuple(_verify_one(path, _wheel_by_filename(declared)[path.name]) for path in paths)
        _verify_dependency_closure(verified)
        return VerifiedWheelClosure(target=target, wheels=verified)
    except Exception as exc:
        if isinstance(exc, _WheelValidationFailure):
            raise install_error("operator_wheel_invalid", "operator wheel closure is invalid", stage="wheel") from None
        raise install_error("operator_wheel_invalid", "operator wheel closure is invalid", stage="wheel") from None


class _WheelValidationFailure(ValueError):  # noqa: N818
    pass


def _wheel_by_filename(wheels: Iterable[ReleaseWheel]) -> dict[str, ReleaseWheel]:
    result = {wheel.filename: wheel for wheel in wheels}
    if len(result) != len(tuple(wheels)):
        raise _WheelValidationFailure("duplicate declared wheel")
    return result


def _require_exact_paths(declared: tuple[ReleaseWheel, ...], paths: tuple[Path, ...]) -> None:
    expected = {wheel.filename for wheel in declared}
    actual = [path.name for path in paths]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise _WheelValidationFailure("downloaded wheel closure differs")
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.suffix != ".whl":
            raise _WheelValidationFailure("wheel path is unsafe")


def _require_declared_artifacts_match(declared: tuple[ReleaseWheel, ...], certified: tuple[object, ...]) -> None:
    expected = {_artifact_identity(wheel) for wheel in declared}
    actual = {_artifact_identity(wheel) for wheel in certified}
    if len(expected) != len(declared) or len(actual) != len(certified) or actual != expected:
        raise _WheelValidationFailure("certification artifacts differ")


def _artifact_identity(value: object) -> tuple[str, str, str, str, str]:
    def read(name: str) -> Any:
        if isinstance(value, Mapping):
            return value[name]
        return getattr(value, name)

    try:
        return (
            str(read("filename")),
            str(read("distribution")),
            str(read("version")),
            str(read("digest")),
            str(read("role")),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _WheelValidationFailure("invalid certified artifact") from exc


def _verify_one(path: Path, declared: ReleaseWheel) -> VerifiedWheel:
    if _sha256(path) != _digest(declared.digest) or path.stat().st_size != declared.size_bytes:
        raise _WheelValidationFailure("wheel bytes differ")
    try:
        with zipfile.ZipFile(path) as archive:
            members = _validate_members(archive)
            record_name, metadata_name, wheel_name = _dist_info_members(members)
            contents = {name: archive.read(name) for name in members}
    except (OSError, RuntimeError, UnicodeError, zipfile.BadZipFile, _WheelValidationFailure) as exc:
        if isinstance(exc, _WheelValidationFailure):
            raise
        raise _WheelValidationFailure("invalid wheel zip") from exc
    _validate_record(record_name, contents)
    requirements, metadata_distribution, metadata_version = _metadata(contents[metadata_name])
    tags = _wheel_tags(contents[wheel_name])
    _validate_identity(path.name, declared, metadata_distribution, metadata_version, tags, wheel_name)
    return VerifiedWheel(path.resolve(), declared.distribution, declared.version, path.name, requirements)


def _validate_members(archive: zipfile.ZipFile) -> tuple[str, ...]:
    infos = archive.infolist()
    if not infos or len(infos) > _MAX_MEMBERS:
        raise _WheelValidationFailure("member count")
    names: list[str] = []
    total = 0
    for info in infos:
        name = info.filename
        pure = PurePosixPath(name)
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if (
            not name
            or "\\" in name
            or pure.is_absolute()
            or name.endswith("/")
            or any(part in {"", ".", ".."} for part in pure.parts)
            or info.flag_bits & 0x1
            or kind not in {0, stat.S_IFREG}
            or info.file_size > _MAX_MEMBER_BYTES
        ):
            raise _WheelValidationFailure("unsafe member")
        compressed = max(1, info.compress_size)
        if info.file_size > compressed * _MAX_RATIO:
            raise _WheelValidationFailure("compression ratio")
        total += info.file_size
        if total > _MAX_TOTAL_BYTES:
            raise _WheelValidationFailure("expanded size")
        names.append(name)
    if len(names) != len(set(names)):
        raise _WheelValidationFailure("duplicate member")
    return tuple(names)


def _dist_info_members(members: tuple[str, ...]) -> tuple[str, str, str]:
    found: dict[str, list[str]] = {name: [] for name in ("RECORD", "METADATA", "WHEEL")}
    for member in members:
        parts = PurePosixPath(member).parts
        if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] in found:
            found[parts[1]].append(member)
    if any(len(values) != 1 for values in found.values()):
        raise _WheelValidationFailure("dist-info members")
    parents = {PurePosixPath(values[0]).parent for values in found.values()}
    if len(parents) != 1:
        raise _WheelValidationFailure("dist-info parents")
    return found["RECORD"][0], found["METADATA"][0], found["WHEEL"][0]


def _validate_record(record_name: str, contents: Mapping[str, bytes]) -> None:
    try:
        rows = list(csv.reader(StringIO(contents[record_name].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise _WheelValidationFailure("record encoding") from exc
    seen: set[str] = set()
    if not rows:
        raise _WheelValidationFailure("empty record")
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in seen or row[0] not in contents:
            raise _WheelValidationFailure("record entry")
        seen.add(row[0])
        digest, size = row[1], row[2]
        if row[0] == record_name:
            if digest or size:
                raise _WheelValidationFailure("record self entry")
            continue
        if not digest.startswith("sha256=") or not size.isdecimal() or int(size) != len(contents[row[0]]):
            raise _WheelValidationFailure("record hash or size")
        try:
            actual = base64.urlsafe_b64encode(hashlib.sha256(contents[row[0]]).digest()).rstrip(b"=").decode("ascii")
        except UnicodeError as exc:
            raise _WheelValidationFailure("record digest") from exc
        if digest.removeprefix("sha256=") != actual:
            raise _WheelValidationFailure("record digest")
    if seen != set(contents):
        raise _WheelValidationFailure("record incomplete")


def _metadata(value: bytes) -> tuple[tuple[Requirement, ...], str, Version]:
    try:
        parsed = Metadata.from_email(value, validate=True)
        return (
            tuple(item if isinstance(item, Requirement) else Requirement(item) for item in parsed.requires_dist or ()),
            parsed.name,
            parsed.version,
        )
    except (Exception, InvalidVersion, ValueError) as exc:
        raise _WheelValidationFailure("metadata") from exc


def _wheel_tags(value: bytes) -> set[Tag]:
    try:
        headers = BytesParser(policy=policy.default).parsebytes(value)
        if headers.defects:
            raise ValueError("wheel header parser defect")
        version = headers.get("Wheel-Version")
        generator = headers.get("Generator")
        root_is_purelib = headers.get("Root-Is-Purelib")
        singleton_headers = ("Wheel-Version", "Generator", "Root-Is-Purelib", "Build")
        if (
            any(len(headers.get_all(header, [])) > 1 for header in singleton_headers)
            or not isinstance(version, str)
            or not version.startswith("1.")
            or not isinstance(generator, str)
            or not generator.strip()
            or root_is_purelib not in {"true", "false"}
        ):
            raise ValueError("wheel version")
        tags: set[Tag] = set()
        for header in headers.get_all("Tag", []):
            tags.update(parse_tag(header))
        if not tags:
            raise ValueError("wheel tags")
        return tags
    except (TypeError, ValueError) as exc:
        raise _WheelValidationFailure("wheel metadata") from exc


def _validate_identity(
    filename: str,
    declared: ReleaseWheel,
    metadata_name: str,
    metadata_version: Version,
    tags: set[Tag],
    wheel_member: str,
) -> None:
    try:
        filename_distribution, filename_version, _, filename_tags = parse_wheel_filename(filename)
        expected_info = f"{str(filename_distribution).replace('-', '_')}-{str(filename_version).replace('-', '_')}.dist-info"
        declared_tags = {tag for value in declared.tags for tag in parse_tag(value)}
        if (
            canonicalize_name(metadata_name) != canonicalize_name(declared.distribution)
            or metadata_version != Version(declared.version)
            or canonicalize_name(str(filename_distribution)) != canonicalize_name(declared.distribution)
            or filename_version != Version(declared.version)
            or PurePosixPath(wheel_member).parent.name != expected_info
            or tags != filename_tags
            or tags != declared_tags
        ):
            raise ValueError("identity")
    except (InvalidWheelFilename, InvalidVersion, ValueError) as exc:
        raise _WheelValidationFailure("wheel identity") from exc


def _verify_dependency_closure(wheels: tuple[VerifiedWheel, ...]) -> None:
    available = {canonicalize_name(wheel.distribution): Version(wheel.version) for wheel in wheels}
    if len(available) != len(wheels):
        raise _WheelValidationFailure("duplicate distribution")
    try:
        requirements = {canonicalize_name(wheel.distribution): wheel.requirements for wheel in wheels}
        if any(requirement.url is not None for items in requirements.values() for requirement in items):
            raise _WheelValidationFailure("direct reference dependency")
        pending = [(distribution, "") for distribution in available]
        evaluated = set(pending)
        while pending:
            distribution, extra = pending.pop()
            for requirement in requirements[distribution]:
                if requirement.marker is not None and not requirement.marker.evaluate({"extra": extra}):
                    continue
                dependency = canonicalize_name(requirement.name)
                version = available.get(dependency)
                if version is None or version not in requirement.specifier:
                    raise _WheelValidationFailure("dependency closure")
                for requested_extra in requirement.extras:
                    context = (dependency, canonicalize_name(requested_extra))
                    if context not in evaluated:
                        evaluated.add(context)
                        pending.append(context)
    except ValueError as exc:
        raise _WheelValidationFailure("dependency marker") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str) -> str:
    prefix, separator, digest = value.partition(":")
    if prefix != "sha256" or not separator or len(digest) != 64:
        raise _WheelValidationFailure("declared digest")
    return digest
