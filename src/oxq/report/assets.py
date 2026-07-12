"""Report asset manifest management."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import secrets
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from oxq.process_lock import (
    ProcessFileLock,
    stable_filesystem_identity,
    stable_path_location_identity,
    verified_user_runtime_root,
)
from oxq.run_digests import _portable_component_key, _portable_posix_relative_parts
from oxq.selection_lock import final_selection_lock_path, hold_final_selection_lock

EMBEDDED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
MANIFEST_SCHEMA_VERSION = 1
URL_RESERVED_PATH_CHARS = {"#", "?", "%", " ", "(", ")", "[", "]", "&"}
ASSET_KIND_SUBDIR = {"figure": "figures", "attachment": "attachments"}
_REPORT_TRANSACTION_SCHEMA_VERSION = 1
_REPORT_TRANSACTION_JOURNAL_NAME = ".oxq-report-transaction.json"
_REPORT_TRANSACTION_ID_BYTES = 16


@dataclass(frozen=True)
class AssetSource:
    script: str | None = None
    input_artifacts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> AssetSource:
        if not isinstance(raw, dict):
            return cls()
        artifacts = raw.get("input_artifacts", [])
        return cls(
            script=raw.get("script") if isinstance(raw.get("script"), str) else None,
            input_artifacts=[str(item) for item in artifacts] if isinstance(artifacts, list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.script:
            data["script"] = self.script
        if self.input_artifacts:
            data["input_artifacts"] = self.input_artifacts
        return data


@dataclass(frozen=True)
class ReportAsset:
    id: str
    kind: str
    path: str
    title: str
    caption: str = ""
    section: str = "results"
    order: int = 100
    mime_type: str = "application/octet-stream"
    sha256: str = ""
    source: AssetSource = field(default_factory=AssetSource)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReportAsset:
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            path=str(raw["path"]),
            title=str(raw["title"]),
            caption=str(raw.get("caption", "")),
            section=str(raw.get("section", "results")),
            order=int(raw.get("order", 100)),
            mime_type=str(raw.get("mime_type", "application/octet-stream")),
            sha256=str(raw.get("sha256", "")),
            source=AssetSource.from_dict(raw.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "path": self.path,
            "title": self.title,
            "caption": self.caption,
            "section": self.section,
            "order": self.order,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
        }
        source = self.source.to_dict()
        if source:
            data["source"] = source
        return data


@dataclass(frozen=True)
class ReportAssetBatchEntry:
    asset_id: str
    file_path: Path
    title: str
    caption: str = ""
    section: str = "results"
    order: int = 100
    source_script: Path | None = None
    source_artifacts: list[str] = field(default_factory=list)


ReportPublication = Mapping[str | Path, bytes | None]
ReportPublicationBuilder = Callable[[], ReportPublication]


class ReportPublicationError(ValueError):
    """Raised when a report publication cannot be committed safely."""


@dataclass
class _PublicationTarget:
    relative: str
    target: Path
    content: bytes | None
    existed: bool
    baseline: bytes | None
    staged: Path | None = None
    backup: Path | None = None


@dataclass(frozen=True)
class _ValidatedPublication:
    relative: str
    parts: tuple[str, ...]
    content: bytes | None


@dataclass(frozen=True)
class _ReportRootPlan:
    supplied: Path
    discovery_subject: Path
    existing_ancestor: Path
    existing_ancestor_identity: str
    root: Path
    missing_parts: tuple[str, ...]


@dataclass(frozen=True)
class _FileIdentity:
    size: int
    sha256: str


@dataclass(frozen=True)
class _RecoveryTarget:
    relative: str
    target: Path
    staged: Path
    backup: Path
    old: _FileIdentity | None
    new: _FileIdentity | None


def publish_report_artifacts(
    report_dir: str | Path,
    artifacts: ReportPublication | ReportPublicationBuilder,
    *,
    lock_subject: str | Path | None = None,
) -> None:
    """Atomically publish one report file set under the canonical final lock.

    ``artifacts`` maps safe paths relative to ``report_dir`` to complete file
    bytes; ``None`` removes a file. A callable is evaluated while the final
    selection lock is held, allowing a batch to derive its manifest from one
    locked baseline. Every target is replaced as one rollback-capable batch.
    """
    preflighted = None if callable(artifacts) else _preflight_publication(artifacts)
    plan = _plan_report_root(report_dir)
    subject = _validated_lock_subject(lock_subject) if lock_subject is not None else plan.discovery_subject
    selection_lock = final_selection_lock_path(subject)
    with hold_final_selection_lock(selection_lock):
        if callable(artifacts) and plan.missing_parts:
            with _hold_report_publication_locks(plan) as hold_location_lock:
                root_metadata = _lstat_or_none(plan.supplied, label="report publication directory")
                if root_metadata is None:
                    preflighted = _preflight_publication(artifacts())
                    root = _materialize_report_root(
                        plan,
                        create=True,
                        hold_location_lock=hold_location_lock,
                    )
                    hold_location_lock()
                    _recover_report_publication_locked(root)
                else:
                    root = _materialize_report_root(
                        plan,
                        create=False,
                        hold_location_lock=hold_location_lock,
                    )
                    hold_location_lock()
                    _recover_report_publication_locked(root)
                    preflighted = _preflight_publication(artifacts())
                _publish_preflighted_report(root, preflighted)
            return
        with _hold_report_publication_root(plan, create=True) as root:
            if callable(artifacts):
                preflighted = _preflight_publication(artifacts())
            assert preflighted is not None
            _publish_preflighted_report(root, preflighted)


def _publish_preflighted_report(root: Path, requested: tuple[_ValidatedPublication, ...]) -> None:
    targets = _prepare_publication_targets(root, requested)
    changed = [item for item in targets if item.existed or item.content is not None]
    if not changed:
        return
    transaction_id = _assign_publication_paths(changed)
    _publish_transaction_locked(root, changed, transaction_id)


@contextmanager
def report_publication_read_transaction(report_dir: str | Path) -> Iterator[Path]:
    """Lock and recover one existing report package for coherent reads."""
    plan = _plan_report_root(report_dir)
    with _hold_report_publication_root(plan, create=False) as root:
        yield root


@contextmanager
def _hold_report_publication_root(plan: _ReportRootPlan, *, create: bool) -> Iterator[Path]:
    with _hold_report_publication_locks(plan) as hold_current_location_lock:
        root = _materialize_report_root(
            plan,
            create=create,
            hold_location_lock=hold_current_location_lock,
        )
        hold_current_location_lock()
        _recover_report_publication_locked(root)
        yield root


@contextmanager
def _hold_report_publication_locks(plan: _ReportRootPlan) -> Iterator[Callable[[], None]]:
    with ExitStack() as locks:
        held_lock_paths: set[Path] = set()

        def hold_current_location_lock() -> None:
            # Missing-path identity can change as ancestors appear; retain each
            # identity until the final directory inode lock is held.
            lock_path = _report_publication_lock_path(plan.root)
            if lock_path not in held_lock_paths:
                locks.enter_context(ProcessFileLock(lock_path))
                held_lock_paths.add(lock_path)

        hold_current_location_lock()
        yield hold_current_location_lock


def _plan_report_root(report_dir: str | Path) -> _ReportRootPlan:
    _require_no_link_or_reparse_components(Path(report_dir), label="report publication directory")
    supplied = _absolute_lexical_path(report_dir)
    current = supplied
    missing: list[str] = []
    while _lstat_or_none(current, label="report publication directory") is None:
        if current.parent == current:
            raise ReportPublicationError(f"report publication directory has no existing ancestor: {supplied}")
        missing.append(current.name)
        current = current.parent
    current_metadata = _lstat_or_none(current, label="report publication directory")
    assert current_metadata is not None
    if not stat.S_ISDIR(current_metadata.st_mode):
        raise ReportPublicationError(f"report publication directory ancestor is not a directory: {current}")
    try:
        existing_ancestor = current.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReportPublicationError(f"report publication directory could not be resolved: {supplied}") from exc
    root = existing_ancestor.joinpath(*reversed(missing))
    return _ReportRootPlan(
        supplied=supplied,
        discovery_subject=current,
        existing_ancestor=existing_ancestor,
        existing_ancestor_identity=stable_filesystem_identity(existing_ancestor),
        root=root,
        missing_parts=tuple(reversed(missing)),
    )


def _materialize_report_root(
    plan: _ReportRootPlan,
    *,
    create: bool,
    hold_location_lock: Callable[[], None] | None = None,
) -> Path:
    _require_no_link_or_reparse_components(plan.supplied, label="report publication directory")
    if stable_filesystem_identity(plan.existing_ancestor) != plan.existing_ancestor_identity:
        raise ReportPublicationError(
            f"report publication directory ancestor changed during lock acquisition: {plan.supplied}"
        )
    current = plan.existing_ancestor
    for part in plan.missing_parts:
        candidate = current / part
        metadata = _lstat_or_none(candidate, label="report publication directory")
        if not create and metadata is None:
            raise ReportPublicationError(f"report publication directory does not exist: {plan.supplied}")
        if metadata is None:
            try:
                candidate.mkdir()
            except FileExistsError:
                pass
            else:
                _fsync_directory(current)
            metadata = _lstat_or_none(candidate, label="report publication directory")
        if metadata is None or _metadata_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ReportPublicationError(f"report publication directory is not a directory: {candidate}")
        try:
            if candidate.resolve(strict=True) != candidate:
                raise ReportPublicationError(
                    f"report publication directory contains a symlink or reparse point: {candidate}"
                )
        except (OSError, RuntimeError) as exc:
            raise ReportPublicationError(f"report publication directory could not be resolved: {candidate}") from exc
        current = candidate
        if hold_location_lock is not None:
            hold_location_lock()
    try:
        resolved = plan.supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReportPublicationError(f"report publication directory could not be resolved: {plan.supplied}") from exc
    _require_no_link_or_reparse_components(plan.supplied, label="report publication directory")
    if resolved != plan.root or not resolved.is_dir():
        raise ReportPublicationError(f"report publication directory changed during lock acquisition: {plan.supplied}")
    return resolved


def _report_publication_lock_path(root: Path) -> Path:
    location_identity = stable_path_location_identity(root)
    identity = hashlib.sha256(location_identity.encode("utf-8")).hexdigest()
    return verified_user_runtime_root() / "report-publication" / f"{identity}.lock"


def _absolute_lexical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path)))


def _validated_lock_subject(lock_subject: str | Path) -> Path:
    _require_no_link_or_reparse_components(Path(lock_subject), label="report publication lock subject")
    return _absolute_lexical_path(lock_subject)


def _require_no_link_or_reparse_components(path: Path, *, label: str) -> None:
    if path.is_absolute():
        current = Path(path.anchor)
        components = path.parts[1:]
    else:
        working_directory = Path.cwd()
        current = Path(working_directory.anchor)
        components = (*working_directory.parts[1:], *path.parts)
    root_metadata = _lstat_or_none(current, label=label)
    if root_metadata is not None and _metadata_is_link_or_reparse(root_metadata):
        raise ReportPublicationError(f"{label} contains a symlink or reparse point: {current}")
    for part in components:
        if part == "..":
            current = current.parent
            continue
        current /= part
        metadata = _lstat_or_none(current, label=label)
        if metadata is not None and _metadata_is_link_or_reparse(metadata):
            raise ReportPublicationError(f"{label} contains a symlink or reparse point: {current}")


def _lstat_or_none(path: Path, *, label: str) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReportPublicationError(f"{label} could not be inspected: {path}: {exc}") from exc


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _is_link_or_reparse(path: Path) -> bool:
    metadata = _lstat_or_none(path, label="report publication path")
    return metadata is not None and _metadata_is_link_or_reparse(metadata)


def _path_entry_present(path: Path) -> bool:
    return _lstat_or_none(path, label="report publication path") is not None


def _preflight_publication(artifacts: ReportPublication) -> tuple[_ValidatedPublication, ...]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ReportPublicationError("report publication requires at least one target")
    requested: list[_ValidatedPublication] = []
    for raw_relative, raw_content in artifacts.items():
        try:
            raw_path = os.fspath(raw_relative)
        except TypeError as exc:
            raise ReportPublicationError(f"report publication target must be a portable relative path: {raw_relative}") from exc
        parts = _portable_posix_relative_parts(raw_path)
        if not isinstance(raw_path, str) or parts is None:
            raise ReportPublicationError(f"report publication target must be a safe relative path: {raw_relative}")
        _require_unreserved_report_parts(parts, label="report publication target")
        if raw_content is not None and not isinstance(raw_content, bytes):
            raise ReportPublicationError(f"report publication content must be bytes or None: {raw_path}")
        requested.append(
            _ValidatedPublication(
                relative="/".join(parts),
                parts=parts,
                content=raw_content,
            )
        )
    _require_no_portable_path_collisions(
        [item.relative for item in requested],
        label="report publication target",
        allow_exact_reuse=False,
        error_type=ReportPublicationError,
    )
    return tuple(requested)


def _prepare_publication_targets(
    root: Path,
    artifacts: tuple[_ValidatedPublication, ...],
) -> list[_PublicationTarget]:
    _require_no_existing_portable_target_aliases(root, artifacts)
    targets: list[_PublicationTarget] = []
    for item in artifacts:
        relative = Path(*item.parts)
        target = root / relative
        _require_safe_publication_target(root, target)
        existed = target.exists()
        baseline = target.read_bytes() if existed else None
        targets.append(
            _PublicationTarget(
                relative=item.relative,
                target=target,
                content=item.content,
                existed=existed,
                baseline=baseline,
            )
        )
    return targets


def _require_no_existing_portable_target_aliases(
    root: Path,
    artifacts: tuple[_ValidatedPublication, ...],
) -> None:
    for item in artifacts:
        current = root
        for part in item.parts:
            metadata = _lstat_or_none(current, label="report publication target parent")
            if metadata is None or _metadata_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                break
            try:
                with os.scandir(current) as directory_entries:
                    entries = tuple(directory_entries)
            except OSError as exc:
                raise ReportPublicationError(f"report publication target parent could not be listed: {current}: {exc}") from exc
            matching = [entry.name for entry in entries if _portable_component_key(entry.name) == _portable_component_key(part)]
            aliases = [name for name in matching if name != part]
            if aliases:
                raise ReportPublicationError(
                    f"report publication target has a portable alias collision: {item.relative!r} and {aliases[0]!r}"
                )
            if part not in matching:
                break
            current /= part


def _require_unreserved_report_parts(parts: tuple[str, ...], *, label: str) -> None:
    for part in parts:
        portable = _portable_component_key(part)
        if portable.startswith(".") and ".oxq-report-" in portable:
            raise ReportPublicationError(f"{label} uses a reserved report transaction namespace: {'/'.join(parts)}")


def _require_no_portable_path_collisions(
    names: Iterable[str],
    *,
    label: str,
    allow_exact_reuse: bool,
    error_type: type[ValueError],
) -> None:
    keyed: list[tuple[tuple[str, ...], str]] = []
    for name in names:
        parts = _portable_posix_relative_parts(name)
        if parts is None:
            raise error_type(f"{label} must be a portable relative path: {name!r}")
        key = tuple(_portable_component_key(part) for part in parts)
        for previous_key, previous_name in keyed:
            if allow_exact_reuse and name == previous_name:
                continue
            shared = min(len(key), len(previous_key))
            if key[:shared] == previous_key[:shared]:
                raise error_type(f"{label} portable path collision: {previous_name!r} and {name!r}")
        keyed.append((key, name))


def _require_safe_publication_target(root: Path, target: Path) -> None:
    current = root
    relative = target.relative_to(root)
    for part in relative.parts[:-1]:
        current /= part
        metadata = _lstat_or_none(current, label="report publication target")
        if metadata is None:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            else:
                _fsync_directory(current.parent)
            metadata = _lstat_or_none(current, label="report publication target")
        if metadata is None or _metadata_is_link_or_reparse(metadata):
            raise ReportPublicationError(f"report publication target contains a symlink or reparse point: {target}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ReportPublicationError(f"report publication target parent is not a directory: {current}")
    target_metadata = _lstat_or_none(target, label="report publication target")
    if target_metadata is not None and (
        _metadata_is_link_or_reparse(target_metadata) or not stat.S_ISREG(target_metadata.st_mode)
    ):
        raise ReportPublicationError(f"report publication target must be a regular non-symlink file: {target}")


def _assign_publication_paths(targets: list[_PublicationTarget]) -> str:
    target_paths = {item.target for item in targets}
    for _ in range(100):
        transaction_id = secrets.token_hex(_REPORT_TRANSACTION_ID_BYTES)
        assigned: list[tuple[_PublicationTarget, Path | None, Path | None]] = []
        internal_paths: set[Path] = set()
        occupied = False
        for item in targets:
            staged = _transaction_internal_path(item.target, transaction_id, "new") if item.content is not None else None
            backup = _transaction_internal_path(item.target, transaction_id, "old") if item.existed else None
            for internal in (staged, backup):
                if internal is not None and (
                    internal in target_paths
                    or internal in internal_paths
                    or internal.exists()
                    or _is_link_or_reparse(internal)
                ):
                    occupied = True
                    break
                if internal is not None:
                    internal_paths.add(internal)
            if occupied:
                break
            assigned.append((item, staged, backup))
        if occupied:
            continue
        for item, staged, backup in assigned:
            item.staged = staged
            item.backup = backup
        return transaction_id
    raise ReportPublicationError("could not reserve internal report publication paths")


def _transaction_internal_path(target: Path, transaction_id: str, state: str) -> Path:
    return target.parent / f".{target.name}.oxq-report-{transaction_id}.{state}"


def _stage_publication_targets(targets: list[_PublicationTarget]) -> None:
    staged_directories: set[Path] = set()
    for item in targets:
        if item.content is None:
            continue
        assert item.staged is not None
        with item.staged.open("xb") as handle:
            _report_publication_precommit_boundary(f"{item.relative}.stage-created")
            handle.write(item.content)
            handle.flush()
            os.fsync(handle.fileno())
        _report_publication_precommit_boundary(f"{item.relative}.staged")
        staged_directories.add(item.staged.parent)
    for directory in staged_directories:
        _fsync_directory(directory)


def _require_unchanged_publication_baseline(targets: list[_PublicationTarget]) -> None:
    for item in targets:
        exists = item.target.exists()
        if _is_link_or_reparse(item.target) or exists != item.existed:
            raise ReportPublicationError(f"report publication baseline changed: {item.relative}")
        if exists and item.target.read_bytes() != item.baseline:
            raise ReportPublicationError(f"report publication baseline changed: {item.relative}")


def _publish_transaction_locked(
    root: Path,
    targets: list[_PublicationTarget],
    transaction_id: str,
) -> None:
    journal_path = _report_transaction_journal_path(root)
    payload = _publication_journal_payload(
        root,
        targets,
        transaction_id,
        recovery="rollback",
        staging_complete=False,
    )
    try:
        _write_report_transaction_journal(journal_path, payload, replace_existing=False)
        _report_publication_precommit_boundary("journal.created")
        _stage_publication_targets(targets)
        _report_publication_precommit_boundary("staging.complete")
        _require_unchanged_publication_baseline(targets)
        _report_publication_precommit_boundary("baseline.validated")
        payload["staging_complete"] = True
        _write_report_transaction_journal(journal_path, payload, replace_existing=True)
        _report_publication_precommit_boundary("journal.staged")
        _commit_publication_targets(root, targets, payload)
    except BaseException as cause:
        if journal_path.exists() or _is_link_or_reparse(journal_path):
            try:
                _recover_report_publication_locked(root)
            except BaseException as recovery_error:
                raise ReportPublicationError(
                    f"report publication failed and could not be recovered: {recovery_error}"
                ) from cause
        raise


def _commit_publication_targets(
    root: Path,
    targets: list[_PublicationTarget],
    payload: dict[str, Any],
) -> None:
    journal_path = _report_transaction_journal_path(root)
    try:
        for item in targets:
            if item.existed:
                assert item.backup is not None
                os.replace(item.target, item.backup)
                _fsync_directory(item.target.parent)
            _report_publication_boundary(f"{item.relative}.replace")
            if item.content is not None:
                assert item.staged is not None
                os.replace(item.staged, item.target)
                _fsync_directory(item.target.parent)
        payload["recovery"] = "commit"
        _write_report_transaction_journal(journal_path, payload, replace_existing=True)
        _report_publication_precommit_boundary("journal.committed")
        _recover_report_publication_locked(root)
    except BaseException as cause:
        if journal_path.exists() or _is_link_or_reparse(journal_path):
            try:
                _recover_report_publication_locked(root)
            except BaseException as recovery_error:
                raise ReportPublicationError(
                    f"report publication failed and could not be recovered: {recovery_error}"
                ) from cause
        raise


def _publication_journal_payload(
    root: Path,
    targets: list[_PublicationTarget],
    transaction_id: str,
    *,
    recovery: str,
    staging_complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "recovery": recovery,
        "staging_complete": staging_complete,
        "report_root_identity": stable_filesystem_identity(root),
        "targets": [
            {
                "path": item.relative,
                "old": _identity_payload(item.baseline) if item.existed else None,
                "new": _identity_payload(item.content),
            }
            for item in targets
        ],
    }


def _identity_payload(content: bytes | None) -> dict[str, Any] | None:
    if content is None:
        return None
    return {"size": len(content), "sha256": _sha256_bytes(content)}


def _report_transaction_journal_path(root: Path) -> Path:
    return root / _REPORT_TRANSACTION_JOURNAL_NAME


def _write_report_transaction_journal(
    path: Path,
    payload: Mapping[str, Any],
    *,
    replace_existing: bool,
) -> None:
    present = path.exists() or _is_link_or_reparse(path)
    if replace_existing:
        if not present:
            raise ReportPublicationError(f"report publication journal disappeared: {path}")
        _require_regular_journal_file(path)
    elif present:
        raise ReportPublicationError(f"report publication journal already exists: {path}")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _recover_report_publication_locked(root: Path) -> None:
    journal_path = _report_transaction_journal_path(root)
    _require_no_report_journal_alias(root, journal_path)
    if not journal_path.exists() and not _is_link_or_reparse(journal_path):
        return
    payload = _read_report_transaction_journal(journal_path)
    recovery, targets = _validate_report_transaction_journal(root, journal_path, payload)
    ordered = targets if recovery == "commit" else list(reversed(targets))
    for item in ordered:
        if recovery == "commit":
            _recover_committed_target(item, journal_path)
        else:
            _recover_rolled_back_target(item, journal_path)
    _clear_report_transaction_journal(journal_path)


def _require_no_report_journal_alias(root: Path, journal_path: Path) -> None:
    try:
        with os.scandir(root) as entries:
            aliases = [
                entry.name
                for entry in entries
                if entry.name != journal_path.name and _portable_component_key(entry.name) == _portable_component_key(journal_path.name)
            ]
    except OSError as exc:
        raise ReportPublicationError(f"report publication journal directory could not be listed: {root}: {exc}") from exc
    if aliases:
        raise ReportPublicationError(f"report publication journal has a portable alias collision: {journal_path.name!r} and {aliases[0]!r}")


def _read_report_transaction_journal(path: Path) -> dict[str, Any]:
    _require_regular_journal_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportPublicationError(f"report publication journal is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportPublicationError(f"report publication journal must contain an object: {path}")
    return payload


def _require_regular_journal_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReportPublicationError(f"report publication journal could not be inspected: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or _metadata_is_link_or_reparse(metadata) or metadata.st_nlink != 1:
        raise ReportPublicationError(f"report publication journal path is unsafe: {path}")


def _validate_report_transaction_journal(
    root: Path,
    journal_path: Path,
    payload: Mapping[str, Any],
) -> tuple[str, list[_RecoveryTarget]]:
    if payload.get("schema_version") != _REPORT_TRANSACTION_SCHEMA_VERSION:
        raise ReportPublicationError(f"report publication journal has an unsupported schema: {journal_path}")
    transaction_id = payload.get("transaction_id")
    if not _is_transaction_id(transaction_id):
        raise ReportPublicationError(f"report publication journal has an unsafe transaction id: {journal_path}")
    if payload.get("report_root_identity") != stable_filesystem_identity(root):
        raise ReportPublicationError(f"report publication journal has an unsafe report root: {journal_path}")
    recovery = payload.get("recovery")
    if recovery not in {"commit", "rollback"}:
        raise ReportPublicationError(f"report publication journal has an invalid recovery mode: {journal_path}")
    staging_complete = payload.get("staging_complete")
    if not isinstance(staging_complete, bool) or (recovery == "commit" and not staging_complete):
        raise ReportPublicationError(f"report publication journal has an invalid staging state: {journal_path}")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ReportPublicationError(f"report publication journal has no targets: {journal_path}")

    validated_targets: list[tuple[str, _FileIdentity | None, _FileIdentity | None]] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping):
            raise ReportPublicationError(f"report publication journal contains an invalid target: {journal_path}")
        relative = _validated_journal_relative_path(raw_target.get("path"), journal_path)
        validated_targets.append(
            (
                relative,
                _validated_file_identity(raw_target.get("old"), journal_path),
                _validated_file_identity(raw_target.get("new"), journal_path),
            )
        )
    _require_no_portable_path_collisions(
        [relative for relative, _, _ in validated_targets],
        label="report publication journal target",
        allow_exact_reuse=False,
        error_type=ReportPublicationError,
    )

    targets: list[_RecoveryTarget] = []
    used_paths = {journal_path}
    for relative, old_identity, new_identity in validated_targets:
        target = root / Path(*PurePosixPath(relative).parts)
        _require_safe_recovery_target(root, target, journal_path)
        staged = _transaction_internal_path(target, transaction_id, "new")
        backup = _transaction_internal_path(target, transaction_id, "old")
        expanded = (target, staged, backup)
        if any(path in used_paths for path in expanded):
            raise ReportPublicationError(f"report publication journal contains overlapping targets: {journal_path}")
        used_paths.update(expanded)
        targets.append(
            _RecoveryTarget(
                relative=relative,
                target=target,
                staged=staged,
                backup=backup,
                old=old_identity,
                new=new_identity,
            )
        )
    return recovery, targets


def _is_transaction_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _REPORT_TRANSACTION_ID_BYTES * 2
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_journal_relative_path(value: object, journal_path: Path) -> str:
    parts = _portable_posix_relative_parts(value)
    if not isinstance(value, str) or parts is None:
        raise ReportPublicationError(f"report publication journal contains an unsafe target: {journal_path}")
    _require_unreserved_report_parts(parts, label="report publication journal target")
    return value


def _validated_file_identity(value: object, journal_path: Path) -> _FileIdentity | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"size", "sha256"}:
        raise ReportPublicationError(f"report publication journal contains an invalid file identity: {journal_path}")
    size = value.get("size")
    sha256 = value.get("sha256")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(sha256, str)
        or not sha256.startswith("sha256:")
        or len(sha256) != 71
        or any(character not in "0123456789abcdef" for character in sha256[7:])
    ):
        raise ReportPublicationError(f"report publication journal contains an invalid file identity: {journal_path}")
    return _FileIdentity(size=size, sha256=sha256)


def _require_safe_recovery_target(root: Path, target: Path, journal_path: Path) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ReportPublicationError(f"report publication journal target escapes the report root: {journal_path}") from exc
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if _is_link_or_reparse(current) or not current.is_dir():
            raise ReportPublicationError(f"report publication journal target path is unsafe: {journal_path}")
    if _is_link_or_reparse(target) or (target.exists() and not target.is_file()):
        raise ReportPublicationError(f"report publication journal target path is unsafe: {journal_path}")


def _recover_rolled_back_target(item: _RecoveryTarget, journal_path: Path) -> None:
    _remove_internal_file(item.staged, item.new, journal_path, allow_partial=True)
    backup_present = _path_entry_present(item.backup)
    target_present = _path_entry_present(item.target)
    if item.old is None:
        if backup_present:
            raise ReportPublicationError(f"report publication journal has an unexpected backup: {journal_path}")
        if target_present:
            _require_file_identity(item.target, item.new, journal_path)
            item.target.unlink()
            _fsync_directory(item.target.parent)
    elif backup_present:
        _require_file_identity(item.backup, item.old, journal_path)
        if target_present and _file_matches_identity(item.target, item.old):
            item.backup.unlink()
            _fsync_directory(item.backup.parent)
        else:
            if target_present:
                _require_file_identity(item.target, item.new, journal_path)
            os.replace(item.backup, item.target)
            _fsync_directory(item.target.parent)
    else:
        _require_file_identity(item.target, item.old, journal_path)


def _recover_committed_target(item: _RecoveryTarget, journal_path: Path) -> None:
    if item.new is None:
        if _path_entry_present(item.target):
            raise ReportPublicationError(f"report publication journal commit target is not removed: {journal_path}")
    elif not _file_matches_identity(item.target, item.new):
        if _path_entry_present(item.target):
            raise ReportPublicationError(f"report publication journal commit target changed: {journal_path}")
        _require_file_identity(item.staged, item.new, journal_path)
        os.replace(item.staged, item.target)
        _fsync_directory(item.target.parent)
    _remove_internal_file(item.staged, item.new, journal_path)
    _remove_internal_file(item.backup, item.old, journal_path)


def _remove_internal_file(
    path: Path,
    expected: _FileIdentity | None,
    journal_path: Path,
    *,
    allow_partial: bool = False,
) -> None:
    if not _path_entry_present(path):
        return
    if allow_partial:
        _require_regular_internal_file(path, journal_path)
    else:
        _require_file_identity(path, expected, journal_path)
    path.unlink()
    _fsync_directory(path.parent)


def _require_regular_internal_file(path: Path, journal_path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReportPublicationError(f"report publication recovery file is unsafe: {journal_path}") from exc
    if _metadata_is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ReportPublicationError(f"report publication recovery file is unsafe: {journal_path}")


def _require_file_identity(path: Path, expected: _FileIdentity | None, journal_path: Path) -> None:
    if expected is None or not _file_matches_identity(path, expected):
        raise ReportPublicationError(f"report publication journal recovery file changed: {journal_path}")


def _file_matches_identity(path: Path, expected: _FileIdentity) -> bool:
    try:
        return (
            not _is_link_or_reparse(path)
            and path.is_file()
            and path.stat().st_size == expected.size
            and _sha256(path) == expected.sha256
        )
    except OSError:
        return False


def _clear_report_transaction_journal(path: Path) -> None:
    _require_regular_journal_file(path)
    path.unlink()
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _report_publication_boundary(label: str) -> None:
    """Test hook for failures at report replacement boundaries."""


def _report_publication_precommit_boundary(label: str) -> None:
    """Test hook for crashes before report replacement boundaries."""


def report_assets_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "report_assets"


def manifest_path(run_dir: str | Path) -> Path:
    return report_assets_dir(run_dir) / "manifest.json"


def safe_asset_id(asset_id: str) -> str:
    candidate = asset_id
    parts = _portable_posix_relative_parts(candidate)
    if parts != (candidate,) or any(char in candidate for char in URL_RESERVED_PATH_CHARS) or _uses_reserved_report_namespace(candidate):
        raise ValueError(f"invalid asset id: {asset_id}")
    return candidate


def list_report_assets(run_dir: str | Path) -> list[ReportAsset]:
    with report_publication_read_transaction(run_dir) as run_path:
        return _list_report_assets_locked(run_path)


def _list_report_assets_locked(run_dir: Path) -> list[ReportAsset]:
    parsed = [_validate_manifest_asset(run_dir, asset) for asset in _load_manifest_assets(run_dir)]
    return sorted(parsed, key=_asset_sort_key)


def _list_report_assets_excluding(run_dir: Path, replacing_asset_ids: set[str]) -> list[ReportAsset]:
    parsed = []
    for asset in _load_manifest_assets(run_dir):
        if asset.id in replacing_asset_ids:
            continue
        parsed.append(_validate_manifest_asset(run_dir, asset))
    return sorted(parsed, key=_asset_sort_key)


def _load_manifest_assets(run_dir: Path) -> list[ReportAsset]:
    path = manifest_path(run_dir)
    try:
        content = _read_stable_report_asset_input(run_dir, path)
    except FileNotFoundError:
        return []
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid report asset manifest: {path}: {exc}") from exc
    items = raw.get("assets", []) if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"invalid report asset manifest: {path}")
    assets: list[ReportAsset] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            asset = ReportAsset.from_dict(item)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid report asset manifest entry: {path}: {exc}") from exc
        safe_asset_id(asset.id)
        _validate_manifest_asset_path(asset)
        _validate_manifest_source_script_path(asset)
        assets.append(asset)
    _validate_manifest_asset_collisions(assets)
    return assets


def _validate_manifest_asset_collisions(assets: list[ReportAsset]) -> None:
    for asset in assets:
        safe_asset_id(asset.id)
        _validate_manifest_asset_path(asset)
        _validate_manifest_source_script_path(asset)
    _require_no_portable_path_collisions(
        [asset.id for asset in assets],
        label="report asset id",
        allow_exact_reuse=False,
        error_type=ValueError,
    )
    referenced_paths = [asset.path for asset in assets]
    referenced_paths.extend(asset.source.script for asset in assets if asset.source.script)
    _require_no_portable_path_collisions(
        referenced_paths,
        label="report asset path",
        allow_exact_reuse=True,
        error_type=ValueError,
    )


def add_report_asset(
    run_dir: str | Path,
    file_path: str | Path,
    *,
    asset_id: str,
    title: str,
    caption: str = "",
    section: str = "results",
    order: int = 100,
    source_script: str | Path | None = None,
    source_artifacts: list[str] | None = None,
) -> ReportAsset:
    return add_report_assets(
        run_dir,
        [
            {
                "id": asset_id,
                "file_path": str(file_path),
                "title": title,
                "caption": caption,
                "section": section,
                "order": order,
                "source_script": str(source_script) if source_script is not None else None,
                "source_artifacts": source_artifacts or [],
            }
        ],
    )[0]


def add_report_assets(run_dir: str | Path, entries: Iterable[Mapping[str, Any]]) -> list[ReportAsset]:
    """Register multiple report assets in one manifest update."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"run directory not found: {run_path}")

    parsed_entries = [_parse_batch_entry(raw) for raw in entries]
    if not parsed_entries:
        raise ValueError("report asset batch is empty")
    _require_no_portable_path_collisions(
        [entry.asset_id for entry in parsed_entries],
        label="report asset id",
        allow_exact_reuse=False,
        error_type=ValueError,
    )

    replacing_ids = {entry.asset_id for entry in parsed_entries}
    published_assets: list[ReportAsset] = []

    def build_publication() -> ReportPublication:
        existing = _list_report_assets_excluding(run_path, replacing_ids)
        previous = _replaced_manifest_assets(run_path, replacing_ids)
        publications: dict[str | Path, bytes | None] = {}
        reserved_destinations: set[Path] = set()
        assets = [
            _build_report_asset(
                run_path,
                entry.file_path,
                asset_id=entry.asset_id,
                title=entry.title,
                caption=entry.caption,
                section=entry.section,
                order=entry.order,
                source_script=entry.source_script,
                source_artifacts=entry.source_artifacts,
                publications=publications,
                reserved_destinations=reserved_destinations,
            )
            for entry in parsed_entries
        ]
        _validate_manifest_asset_collisions(existing + assets)
        published_assets[:] = assets
        retained_paths = _referenced_asset_paths(existing + assets)
        for old_asset in previous:
            for old_path in _report_asset_file_paths(old_asset):
                if old_path not in retained_paths and old_path not in publications:
                    publications[old_path] = None
        publications[Path("report_assets/manifest.json")] = _manifest_content(existing + assets)
        return publications

    publish_report_artifacts(run_path, build_publication)
    return published_assets


def _parse_batch_entry(raw: Mapping[str, Any]) -> ReportAssetBatchEntry:
    if not isinstance(raw, Mapping):
        raise ValueError("report asset batch item must be an object")
    missing = [key for key in ("id", "file_path", "title") if key not in raw]
    if missing:
        raise ValueError(f"report asset batch item missing required field(s): {', '.join(missing)}")

    asset_id = safe_asset_id(str(raw["id"]))
    file_path = Path(str(raw["file_path"]))
    if not file_path.exists():
        raise FileNotFoundError(f"asset file not found: {file_path}")

    source_script = None
    if raw.get("source_script") is not None:
        source_script = Path(str(raw["source_script"]))
        if not source_script.exists():
            raise FileNotFoundError(f"source script not found: {source_script}")

    source_artifacts = raw.get("source_artifacts", [])
    if not isinstance(source_artifacts, list):
        raise ValueError("source_artifacts must be a list")

    try:
        order = int(raw.get("order", 100))
    except (TypeError, ValueError) as exc:
        raise ValueError("order must be an integer") from exc

    return ReportAssetBatchEntry(
        asset_id=asset_id,
        file_path=file_path,
        title=str(raw["title"]),
        caption=str(raw.get("caption", "")),
        section=str(raw.get("section", "results")),
        order=order,
        source_script=source_script,
        source_artifacts=[str(item) for item in source_artifacts],
    )


def _build_report_asset(
    run_path: Path,
    source_path: Path,
    *,
    asset_id: str,
    title: str,
    caption: str,
    section: str,
    order: int,
    source_script: str | Path | None,
    source_artifacts: list[str],
    publications: dict[str | Path, bytes | None],
    reserved_destinations: set[Path],
) -> ReportAsset:
    suffix = source_path.suffix.lower()
    kind = "figure" if suffix in EMBEDDED_IMAGE_EXTENSIONS else "attachment"
    subdir = "figures" if kind == "figure" else "attachments"
    destination = report_assets_dir(run_path) / subdir / f"{asset_id}{suffix}"
    content = source_path.read_bytes()
    relative_destination = Path("report_assets") / subdir / destination.name
    publications[relative_destination] = content
    reserved_destinations.add(destination.resolve(strict=False))

    copied_script = _copy_source_script(
        run_path,
        source_script,
        asset_id,
        publications=publications,
        reserved_destinations=reserved_destinations,
    )
    asset = ReportAsset(
        id=asset_id,
        kind=kind,
        path=_relative_to_report_assets(run_path, destination),
        title=title,
        caption=caption,
        section=section,
        order=order,
        mime_type=mimetypes.guess_type(destination.name)[0] or "application/octet-stream",
        sha256=_sha256_bytes(content),
        source=AssetSource(
            script=copied_script,
            input_artifacts=[str(item) for item in source_artifacts],
        ),
    )
    return asset


def _copy_source_script(
    run_dir: Path,
    source_script: str | Path | None,
    asset_id: str,
    *,
    publications: dict[str | Path, bytes | None],
    reserved_destinations: set[Path],
) -> str | None:
    if source_script is None:
        return None
    script_path = Path(source_script)
    if not script_path.exists():
        raise FileNotFoundError(f"source script not found: {script_path}")

    assets_dir = report_assets_dir(run_dir)
    assets_dir_resolved = assets_dir.resolve()
    script_path_resolved = script_path.resolve()
    try:
        return script_path_resolved.relative_to(assets_dir_resolved).as_posix()
    except ValueError:
        destination = assets_dir / "scripts" / script_path.name
        destination = _available_source_script_destination(
            destination,
            script_path_resolved,
            asset_id,
            reserved_destinations,
        )
        publications[Path("report_assets") / "scripts" / destination.name] = script_path.read_bytes()
        reserved_destinations.add(destination.resolve(strict=False))
        return destination.relative_to(assets_dir).as_posix()


def _available_source_script_destination(
    default_destination: Path,
    source_resolved: Path,
    asset_id: str,
    reserved_destinations: set[Path],
) -> Path:
    if (
        default_destination.resolve(strict=False) not in reserved_destinations
        and (not default_destination.exists() or default_destination.resolve() == source_resolved)
    ):
        return default_destination

    candidate = default_destination.with_name(f"{asset_id}_{default_destination.name}")
    if candidate.resolve(strict=False) not in reserved_destinations and (
        not candidate.exists() or candidate.resolve() == source_resolved
    ):
        return candidate

    counter = 2
    while True:
        candidate = default_destination.with_name(f"{asset_id}_{counter}_{default_destination.name}")
        if candidate.resolve(strict=False) not in reserved_destinations and (
            not candidate.exists() or candidate.resolve() == source_resolved
        ):
            return candidate
        counter += 1


def _relative_to_report_assets(run_dir: Path, path: Path) -> str:
    return path.relative_to(report_assets_dir(run_dir)).as_posix()


def _validate_manifest_asset(run_dir: Path, asset: ReportAsset) -> ReportAsset:
    relative_path = _validate_manifest_asset_path(asset)
    asset_path = report_assets_dir(run_dir) / relative_path
    try:
        content = _read_stable_report_asset_input(run_dir, asset_path)
    except FileNotFoundError as exc:
        raise ValueError(f"missing report asset file: {asset_path}") from exc
    if asset.sha256:
        actual_sha256 = _sha256_bytes(content)
        if actual_sha256 != asset.sha256:
            raise ValueError(f"hash mismatch for report asset {asset.id}: expected {asset.sha256}, got {actual_sha256}")
    return ReportAsset(
        id=asset.id,
        kind=asset.kind,
        path=relative_path,
        title=asset.title,
        caption=asset.caption,
        section=asset.section,
        order=asset.order,
        mime_type=asset.mime_type,
        sha256=asset.sha256,
        source=asset.source,
    )


def _validate_manifest_asset_path(asset: ReportAsset) -> str:
    raw_path = asset.path
    parts = _portable_report_path_parts(raw_path)
    if parts is None:
        raise ValueError(f"invalid report asset path for {asset.id}: {raw_path}")
    expected_subdir = ASSET_KIND_SUBDIR.get(asset.kind)
    if expected_subdir is None or not parts or parts[0] != expected_subdir:
        raise ValueError(f"invalid report asset path for {asset.id}: {raw_path}")
    return "/".join(parts)


def _validate_manifest_source_script_path(asset: ReportAsset) -> None:
    raw_path = asset.source.script
    if raw_path is None:
        return
    parts = _portable_report_path_parts(raw_path)
    if parts is None or not parts or parts[0] != "scripts":
        raise ValueError(f"invalid report asset source script path for {asset.id}: {raw_path}")


def _portable_report_path_parts(value: object) -> tuple[str, ...] | None:
    parts = _portable_posix_relative_parts(value)
    if not isinstance(value, str) or parts is None:
        return None
    if any(character in value for character in URL_RESERVED_PATH_CHARS):
        return None
    if any(_uses_reserved_report_namespace(part) for part in parts):
        return None
    return parts


def _uses_reserved_report_namespace(component: str) -> bool:
    portable = _portable_component_key(component)
    return portable.startswith(".") and ".oxq-report-" in portable


def _manifest_content(assets: list[ReportAsset]) -> bytes:
    data = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "assets": [asset.to_dict() for asset in sorted(assets, key=_asset_sort_key)],
    }
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()


def _replaced_manifest_assets(run_dir: Path, replacing_ids: set[str]) -> list[ReportAsset]:
    replaced: list[ReportAsset] = []
    for asset in _load_manifest_assets(run_dir):
        if asset.id in replacing_ids:
            replaced.append(asset)
    return replaced


def _report_asset_file_paths(asset: ReportAsset) -> set[Path]:
    paths = {Path("report_assets") / asset.path}
    if asset.source.script:
        paths.add(Path("report_assets") / asset.source.script)
    return paths


def _referenced_asset_paths(assets: list[ReportAsset]) -> set[Path]:
    paths: set[Path] = set()
    for asset in assets:
        paths.update(_report_asset_file_paths(asset))
    return paths


def _read_stable_report_asset_input(run_dir: Path, path: Path) -> bytes:
    run_root = _absolute_lexical_path(run_dir)
    candidate = _absolute_lexical_path(path)
    try:
        relative = candidate.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"report asset input must stay within the run directory: {candidate}") from exc
    if not relative.parts:
        raise ValueError(f"report asset input must identify a file: {candidate}")

    root_metadata = _report_asset_input_lstat(run_root)
    if root_metadata is None:
        raise FileNotFoundError(candidate)
    if _metadata_is_link_or_reparse(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"report asset input root must be a non-symlink directory: {run_root}")

    current = run_root
    for part in relative.parts[:-1]:
        _require_no_portable_child_alias(current, part, candidate)
        current /= part
        metadata = _report_asset_input_lstat(current)
        if metadata is None:
            raise FileNotFoundError(candidate)
        if _metadata_is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"report asset input path contains a symlink, reparse point, or non-directory: {current}")

    _require_no_portable_child_alias(current, relative.parts[-1], candidate)
    before = _report_asset_input_lstat(candidate)
    if before is None:
        raise FileNotFoundError(candidate)
    if _metadata_is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"report asset input must be a non-symlink regular file: {candidate}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError(f"report asset input must be a non-symlink regular file: {candidate}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_report_asset_input_file(before, opened):
            raise ValueError(f"report asset input changed during read: {candidate}")
        _report_asset_input_read_boundary(candidate, "opened")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        _report_asset_input_read_boundary(candidate, "read")
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    after_path = _report_asset_input_lstat(candidate)
    if (
        after_path is None
        or _metadata_is_link_or_reparse(after_path)
        or not stat.S_ISREG(after_path.st_mode)
        or not _same_report_asset_input_file(before, after_path)
        or _report_asset_input_coherence(before) != _report_asset_input_coherence(opened)
        or _report_asset_input_coherence(opened) != _report_asset_input_coherence(after_descriptor)
        or _report_asset_input_coherence(after_descriptor) != _report_asset_input_coherence(after_path)
        or len(content) != after_descriptor.st_size
    ):
        raise ValueError(f"report asset input changed during read and is not coherent: {candidate}")
    return content


def _require_no_portable_child_alias(parent: Path, child_name: str, candidate: Path) -> None:
    try:
        with os.scandir(parent) as entries:
            aliases = [
                entry.name
                for entry in entries
                if entry.name != child_name and _portable_component_key(entry.name) == _portable_component_key(child_name)
            ]
    except FileNotFoundError:
        raise FileNotFoundError(candidate) from None
    except OSError as exc:
        raise ValueError(f"report asset input parent could not be listed safely: {parent}: {exc}") from exc
    if aliases:
        raise ValueError(f"report asset input has a portable path collision: {child_name!r} and {aliases[0]!r}")


def _report_asset_input_lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"report asset input could not be inspected safely: {path}: {exc}") from exc


def _same_report_asset_input_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _report_asset_input_coherence(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _report_asset_input_read_boundary(path: Path, stage: str) -> None:
    """Test hook for report-asset descriptor replacement races."""


def _asset_sort_key(asset: ReportAsset) -> tuple[str, int, str]:
    return (asset.section, asset.order, asset.id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
