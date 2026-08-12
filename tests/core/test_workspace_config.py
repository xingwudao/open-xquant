from __future__ import annotations

from pathlib import Path

import pytest

from oxq.core.workspace_config import (
    WorkspaceConfigError,
    discover_workspace_config,
    load_workspace_config,
)


def test_load_workspace_config_accepts_canonical_mapping(tmp_path: Path) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("paths:\n  versions_dir: research_versions\n", encoding="utf-8")

    config = load_workspace_config(config_path)

    assert config == {"paths": {"versions_dir": "research_versions"}}


def test_load_workspace_config_missing_behavior_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / ".open-xquant" / "workspace.yaml"
    assert load_workspace_config(path, missing_ok=True) == {}
    with pytest.raises(WorkspaceConfigError, match="does not exist"):
        load_workspace_config(path)


def test_load_workspace_config_empty_document_behavior_is_explicit(tmp_path: Path) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    path = config_dir / "workspace.yaml"
    path.write_text("", encoding="utf-8")

    assert load_workspace_config(path, allow_empty=True) == {}
    with pytest.raises(WorkspaceConfigError, match="must contain an object"):
        load_workspace_config(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [("workflow: [\n", "invalid YAML"), ("- not\n- an\n- object\n", "must contain an object")],
)
def test_load_workspace_config_rejects_invalid_documents(tmp_path: Path, content: str, message: str) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    path = config_dir / "workspace.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(WorkspaceConfigError, match=message):
        load_workspace_config(path)


def test_load_workspace_config_rejects_noncanonical_path(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(WorkspaceConfigError, match="canonical"):
        load_workspace_config(path)


def test_load_workspace_config_rejects_symlink_file_and_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "workspace.yaml"
    target.write_text("{}\n", encoding="utf-8")
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    (config_dir / "workspace.yaml").symlink_to(target)
    with pytest.raises(WorkspaceConfigError, match="symlink"):
        load_workspace_config(config_dir / "workspace.yaml")

    second = tmp_path / "second"
    second.mkdir()
    (second / ".open-xquant").symlink_to(target_dir, target_is_directory=True)
    with pytest.raises(WorkspaceConfigError, match="symlink"):
        load_workspace_config(second / ".open-xquant" / "workspace.yaml")


def test_load_workspace_config_rejects_broken_symlink_marker(tmp_path: Path) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    path = config_dir / "workspace.yaml"
    path.symlink_to(tmp_path / "missing.yaml")

    with pytest.raises(WorkspaceConfigError, match="symlink"):
        load_workspace_config(path, missing_ok=True)


def test_discover_workspace_config_uses_nearest_canonical_workspace(tmp_path: Path) -> None:
    outer_config = tmp_path / ".open-xquant"
    outer_config.mkdir()
    (outer_config / "workspace.yaml").write_text("name: outer\n", encoding="utf-8")
    inner = tmp_path / "research" / "run"
    inner.mkdir(parents=True)
    inner_config = tmp_path / "research" / ".open-xquant"
    inner_config.mkdir()
    (inner_config / "workspace.yaml").write_text("name: inner\n", encoding="utf-8")

    discovered = discover_workspace_config(inner / "result.json")

    assert discovered is not None
    assert discovered.root == tmp_path / "research"
    assert discovered.config == {"name": "inner"}


def test_discover_workspace_config_treats_unsafe_nearest_marker_as_authoritative(tmp_path: Path) -> None:
    outer_config = tmp_path / ".open-xquant"
    outer_config.mkdir()
    (outer_config / "workspace.yaml").write_text("name: outer\n", encoding="utf-8")
    inner = tmp_path / "research" / "run"
    inner.mkdir(parents=True)
    inner_config = tmp_path / "research" / ".open-xquant"
    inner_config.mkdir()
    (inner_config / "workspace.yaml").write_text("workflow: [\n", encoding="utf-8")

    with pytest.raises(WorkspaceConfigError, match="invalid YAML"):
        discover_workspace_config(inner)
