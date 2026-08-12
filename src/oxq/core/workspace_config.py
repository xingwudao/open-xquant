"""Canonical, read-only governed workspace configuration loading."""

from __future__ import annotations

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
    if not config_path.is_file():
        raise WorkspaceConfigError(f"workspace configuration must be a regular file: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise WorkspaceConfigError(f"workspace configuration contains invalid YAML: {config_path}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkspaceConfigError(f"workspace configuration could not be read: {config_path}: {exc}") from exc
    if payload is None and allow_empty:
        return {}
    if not isinstance(payload, dict):
        raise WorkspaceConfigError(f"workspace configuration must contain an object: {config_path}")
    return payload


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
