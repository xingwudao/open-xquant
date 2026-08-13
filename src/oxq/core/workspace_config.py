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
    unsafe_component = _first_link_component(config_path)
    if unsafe_component is not None:
        raise WorkspaceConfigError(f"workspace configuration path component must not be a symlink: {unsafe_component}")
    if not config_path.exists() and not config_path.is_symlink():
        if missing_ok:
            return {}
        raise WorkspaceConfigError(f"workspace configuration does not exist: {config_path}")
    if config_path.parent.is_symlink():
        raise WorkspaceConfigError(f"workspace configuration directory must not be a symlink: {config_path.parent}")
    if not config_path.parent.is_dir():
        raise WorkspaceConfigError(f"workspace configuration directory is invalid: {config_path.parent}")
    if config_path.is_symlink():
        raise WorkspaceConfigError(f"workspace configuration must not be a symlink: {config_path}")
    try:
        payload = yaml.safe_load(_read_regular_file_nofollow(config_path))
    except yaml.YAMLError as exc:
        raise WorkspaceConfigError(f"workspace configuration contains invalid YAML: {config_path}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkspaceConfigError(f"workspace configuration could not be read: {config_path}: {exc}") from exc
    if payload is None and allow_empty:
        return {}
    if not isinstance(payload, dict):
        raise WorkspaceConfigError(f"workspace configuration must contain an object: {config_path}")
    return payload


def _read_regular_file_nofollow(path: Path) -> str:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise WorkspaceConfigError("workspace configuration cannot be read: atomic nofollow protection is unavailable")

    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
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


def _first_link_component(path: Path) -> Path | None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            return current
    return None


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
