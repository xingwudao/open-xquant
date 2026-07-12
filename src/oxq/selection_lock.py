"""Canonical governed-workspace final-selection lock discovery and holding."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from oxq.process_lock import ProcessFileLock

FINAL_SELECTION_LOCK_RELATIVE_PATH = Path(".open-xquant/locks/final-selection.lock")
_WORKSPACE_CONFIG_RELATIVE_PATH = Path(".open-xquant/workspace.yaml")


class SelectionLockError(ValueError):
    """Raised when governed workspace lock discovery cannot proceed safely."""


def governing_workspace_root(path: str | Path) -> Path | None:
    """Return the nearest canonical governed workspace, or ``None`` for legacy runs.

    Discovery is read-only. A present workspace marker is authoritative: malformed
    configuration fails closed instead of falling through to an outer workspace or
    silently treating the subject as a standalone run.
    """
    subject = _resolve_subject(path)
    start = subject if subject.is_dir() else subject.parent
    for candidate in (start, *start.parents):
        config_path = candidate / _WORKSPACE_CONFIG_RELATIVE_PATH
        if not config_path.exists() and not config_path.is_symlink():
            continue
        workspace = candidate.resolve(strict=True)
        config = _read_workspace_config(workspace, config_path)
        if not _is_governed_workspace(workspace, config):
            return None
        return workspace
    return None


def final_selection_lock_path(path: str | Path) -> Path | None:
    """Resolve the canonical final-selection lock path without creating it."""
    workspace = governing_workspace_root(path)
    if workspace is None:
        return None
    lock_path = workspace / FINAL_SELECTION_LOCK_RELATIVE_PATH
    locks_dir = lock_path.parent
    if locks_dir.is_symlink() or (locks_dir.exists() and not locks_dir.is_dir()):
        raise SelectionLockError(f"workspace final-selection lock directory is unsafe: {locks_dir}")
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise SelectionLockError(f"workspace final-selection lock path is unsafe: {lock_path}")
    return lock_path.resolve(strict=False)


@contextmanager
def hold_final_selection_lock(lock_path: str | Path | None) -> Iterator[None]:
    """Hold a canonical final-selection lock, revalidating it at acquisition.

    Pre-discovered paths are untrusted by the time this function is entered.
    The complete parent must still be canonical, the lock directory and file
    must not be symlinks/reparse points, and the opened file must remain the
    same regular file acquired by the cross-platform process lock.
    """
    if lock_path is None:
        yield
        return
    path = Path(lock_path)
    if not path.is_absolute() or path.name != "final-selection.lock" or path.parent.name != "locks":
        raise SelectionLockError(f"workspace final-selection lock path is not canonical: {path}")
    parent_identity = _prepare_lock_parent(path)
    descriptor = _open_selection_lock_file(path)
    try:
        lock = ProcessFileLock(path)
        with lock:
            _require_lock_parent_identity(path.parent, parent_identity)
            _require_same_regular_file(path, descriptor, getattr(lock, "descriptor", None))
            yield
    finally:
        os.close(descriptor)


def _prepare_lock_parent(path: Path) -> tuple[int, int]:
    locks_dir = path.parent
    config_dir = locks_dir.parent
    if config_dir.name != ".open-xquant":
        raise SelectionLockError(f"workspace final-selection lock path is not canonical: {path}")
    if _is_link_or_reparse(config_dir) or not config_dir.is_dir():
        raise SelectionLockError(f"workspace final-selection lock directory is unsafe: {locks_dir}")
    if locks_dir.exists() or locks_dir.is_symlink():
        if _is_link_or_reparse(locks_dir) or not locks_dir.is_dir():
            raise SelectionLockError(f"workspace final-selection lock directory is unsafe: {locks_dir}")
    else:
        try:
            os.mkdir(locks_dir, mode=0o700)
        except OSError as exc:
            raise SelectionLockError(
                f"workspace final-selection lock directory could not be created safely: {locks_dir}: {exc}"
            ) from exc
    try:
        if locks_dir.resolve(strict=True) != locks_dir:
            raise SelectionLockError(f"workspace final-selection lock directory is unsafe: {locks_dir}")
        os.chmod(locks_dir, 0o700)
        metadata = locks_dir.stat(follow_symlinks=False)
    except OSError as exc:
        raise SelectionLockError(
            f"workspace final-selection lock directory could not be secured: {locks_dir}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise SelectionLockError(f"workspace final-selection lock directory is unsafe: {locks_dir}")
    return metadata.st_dev, metadata.st_ino


def _open_selection_lock_file(path: Path) -> int:
    if _is_link_or_reparse(path):
        raise SelectionLockError(f"workspace final-selection lock path is unsafe: {path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SelectionLockError(f"workspace final-selection lock path is unsafe: {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SelectionLockError(f"workspace final-selection lock path is unsafe: {path}")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        else:
            os.chmod(path, 0o600)
        _require_same_regular_file(path, descriptor, None)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _require_lock_parent_identity(path: Path, expected: tuple[int, int]) -> None:
    if _is_link_or_reparse(path) or not path.is_dir() or path.resolve(strict=True) != path:
        raise SelectionLockError(f"workspace final-selection lock directory is unsafe: {path}")
    metadata = path.stat(follow_symlinks=False)
    if (metadata.st_dev, metadata.st_ino) != expected:
        raise SelectionLockError(f"workspace final-selection lock directory changed during acquisition: {path}")


def _require_same_regular_file(path: Path, descriptor: int, lock_descriptor: int | None) -> None:
    if _is_link_or_reparse(path):
        raise SelectionLockError(f"workspace final-selection lock path is unsafe: {path}")
    try:
        path_metadata = path.stat(follow_symlinks=False)
        descriptor_metadata = os.fstat(descriptor)
    except OSError as exc:
        raise SelectionLockError(f"workspace final-selection lock path is unsafe: {path}: {exc}") from exc
    identities = {(path_metadata.st_dev, path_metadata.st_ino), (descriptor_metadata.st_dev, descriptor_metadata.st_ino)}
    if lock_descriptor is not None:
        lock_metadata = os.fstat(lock_descriptor)
        identities.add((lock_metadata.st_dev, lock_metadata.st_ino))
    if (
        len(identities) != 1
        or not stat.S_ISREG(path_metadata.st_mode)
        or not stat.S_ISREG(descriptor_metadata.st_mode)
        or descriptor_metadata.st_nlink != 1
    ):
        raise SelectionLockError(f"workspace final-selection lock path changed during acquisition: {path}")


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _resolve_subject(path: str | Path) -> Path:
    supplied = Path(path)
    try:
        return supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SelectionLockError(f"workspace discovery subject could not be resolved: {supplied}: {exc}") from exc


def _read_workspace_config(workspace: Path, config_path: Path) -> Mapping[str, Any]:
    config_dir = workspace / ".open-xquant"
    canonical_config = config_dir / "workspace.yaml"
    if config_dir.is_symlink() or not config_dir.is_dir() or config_path != canonical_config:
        raise SelectionLockError(f"workspace configuration path is unsafe: {config_path}")
    if config_path.is_symlink() or not config_path.is_file():
        raise SelectionLockError(f"workspace configuration must be a regular non-symlink file: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SelectionLockError(f"workspace configuration is invalid: {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SelectionLockError(f"workspace configuration must contain a mapping: {config_path}")
    return payload


def _is_governed_workspace(workspace: Path, config: Mapping[str, Any]) -> bool:
    workflow = config.get("workflow")
    if workflow is not None and not isinstance(workflow, Mapping):
        raise SelectionLockError("workspace workflow configuration must contain a mapping")
    paths = config.get("paths")
    if paths is not None and not isinstance(paths, Mapping):
        raise SelectionLockError("workspace paths configuration must contain a mapping")

    versions_dir_configured = isinstance(paths, Mapping) and "versions_dir" in paths
    governed = (isinstance(workflow, Mapping) and workflow.get("layout") == "version_governed") or versions_dir_configured
    if not governed:
        return False

    raw_versions_dir = paths.get("versions_dir", "versions") if isinstance(paths, Mapping) else "versions"
    if not isinstance(raw_versions_dir, str) or not raw_versions_dir:
        raise SelectionLockError("workspace paths.versions_dir must be a non-empty string")
    versions_dir = Path(raw_versions_dir)
    if versions_dir.is_absolute() or PureWindowsPath(raw_versions_dir).is_absolute() or ".." in versions_dir.parts:
        raise SelectionLockError("workspace paths.versions_dir must be a safe relative path")
    candidate = workspace / versions_dir
    current = workspace
    for part in versions_dir.parts:
        current /= part
        if current.is_symlink():
            raise SelectionLockError("workspace paths.versions_dir must not contain symlink components")
    try:
        candidate.resolve(strict=False).relative_to(workspace)
    except ValueError as exc:
        raise SelectionLockError("workspace paths.versions_dir must stay within the workspace") from exc
    return True
