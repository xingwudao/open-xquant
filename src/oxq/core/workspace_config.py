"""Canonical, read-only governed workspace configuration loading."""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

WORKSPACE_CONFIG_RELATIVE_PATH = Path(".open-xquant/workspace.yaml")


class WorkspaceConfigError(ValueError):
    """Raised when a present workspace marker cannot be trusted or parsed."""


@dataclass(frozen=True, slots=True)
class DiscoveredWorkspaceConfig:
    root: Path
    path: Path
    config: dict[str, Any]


def load_workspace_config(
    path: str | Path,
    *,
    missing_ok: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Load the canonical workspace YAML object without following symlinks."""

    config_path = Path(path)
    if config_path.name != "workspace.yaml" or config_path.parent.name != ".open-xquant":
        raise WorkspaceConfigError(f"workspace configuration path is not canonical: {config_path}")
    try:
        config_text = _read_regular_file_nofollow(config_path)
    except FileNotFoundError as exc:
        if missing_ok:
            return {}
        raise WorkspaceConfigError(f"workspace configuration does not exist: {config_path}") from exc
    except WorkspaceConfigError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkspaceConfigError(f"workspace configuration could not be read: {config_path}: {exc}") from exc
    try:
        payload = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise WorkspaceConfigError(f"workspace configuration contains invalid YAML: {config_path}: {exc}") from exc
    if payload is None and allow_empty:
        return {}
    if not isinstance(payload, dict):
        raise WorkspaceConfigError(f"workspace configuration must contain an object: {config_path}")
    return payload


def _read_regular_file_nofollow(path: Path) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(nofollow, int) or nofollow == 0 or not isinstance(directory, int) or directory == 0:
        raise _nofollow_unavailable()

    absolute = path.absolute()
    directory_flags = os.O_RDONLY | nofollow | directory
    directory_descriptor = _open_directory_nofollow(
        absolute.anchor,
        directory_flags,
        display_path=Path(absolute.anchor),
    )
    current_path = Path(absolute.anchor)

    try:
        for component in absolute.parts[1:-1]:
            current_path /= component
            child_descriptor = _open_directory_nofollow(
                component,
                directory_flags,
                display_path=current_path,
                dir_fd=directory_descriptor,
            )
            previous_descriptor = directory_descriptor
            directory_descriptor = child_descriptor
            os.close(previous_descriptor)
        return _read_regular_file_at(absolute.name, path, directory_descriptor, nofollow)
    finally:
        os.close(directory_descriptor)


def _open_directory_nofollow(
    path: str,
    flags: int,
    *,
    display_path: Path,
    dir_fd: int | None = None,
) -> int:
    try:
        if dir_fd is None:
            descriptor = os.open(path, flags)
        else:
            descriptor = os.open(path, flags, dir_fd=dir_fd)
    except (NotImplementedError, TypeError) as exc:
        raise _nofollow_unavailable() from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise WorkspaceConfigError(
                f"workspace configuration path component must be a directory and must not be a symlink: {display_path}"
            ) from exc
        raise

    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise WorkspaceConfigError(f"workspace configuration path component must be a directory: {display_path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_file_at(name: str, path: Path, directory_descriptor: int, nofollow: int) -> str:
    try:
        descriptor = os.open(name, os.O_RDONLY | nofollow, dir_fd=directory_descriptor)
    except (NotImplementedError, TypeError) as exc:
        raise _nofollow_unavailable() from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise WorkspaceConfigError(f"workspace configuration must not be a symlink: {path}") from exc
        raise

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WorkspaceConfigError(f"workspace configuration must be a regular file: {path}")
        stream = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = -1
        with stream:
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _nofollow_unavailable() -> WorkspaceConfigError:
    return WorkspaceConfigError("workspace configuration cannot be read: atomic directory-descriptor nofollow protection is unavailable")


def discover_workspace_config(path: str | Path) -> DiscoveredWorkspaceConfig | None:
    """Find and load the nearest canonical workspace marker."""

    subject = Path(path).absolute()
    start = subject if subject.is_dir() else subject.parent
    for candidate in (start, *start.parents):
        config_path = candidate / WORKSPACE_CONFIG_RELATIVE_PATH
        if not config_path.exists() and not config_path.is_symlink():
            continue
        config = load_workspace_config(config_path)
        return DiscoveredWorkspaceConfig(root=candidate.resolve(), path=config_path, config=config)
    return None
