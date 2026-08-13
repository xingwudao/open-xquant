from __future__ import annotations

import os
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


def test_load_workspace_config_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    config_dir = target / ".open-xquant"
    config_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text("name: outside\n", encoding="utf-8")
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(WorkspaceConfigError, match="symlink"):
        load_workspace_config(workspace_link / ".open-xquant" / "workspace.yaml")


def test_load_workspace_config_rejects_symlink_replacement_before_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    link_target = tmp_path / "outside.yaml"
    link_target.write_text("name: link-target\n", encoding="utf-8")
    real_open = os.open
    replaced = False

    def replace_config() -> None:
        nonlocal replaced
        if replaced:
            return
        config_path.unlink()
        config_path.symlink_to(link_target)
        replaced = True

    def racing_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if Path(path) == config_path or (dir_fd is not None and path == config_path.name):
            replace_config()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)

    try:
        config = load_workspace_config(config_path)
    except WorkspaceConfigError:
        pass
    else:
        pytest.fail(f"read replacement symlink target: {config!r}")
    assert replaced


def test_load_workspace_config_reads_open_descriptor_after_path_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    link_target = tmp_path / "outside.yaml"
    link_target.write_text("name: link-target\n", encoding="utf-8")
    real_open = os.open
    replaced = False

    def replace_config() -> None:
        nonlocal replaced
        if replaced:
            return
        config_path.unlink()
        config_path.symlink_to(link_target)
        replaced = True

    def racing_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == config_path or (dir_fd is not None and path == config_path.name):
            replace_config()
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)

    assert load_workspace_config(config_path) == {"name": "trusted"}
    assert replaced


def test_load_workspace_config_rejects_ancestor_symlink_replacement_before_directory_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace-under-test"
    config_dir = workspace / ".open-xquant"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside_config_dir = outside / ".open-xquant"
    outside_config_dir.mkdir(parents=True)
    (outside_config_dir / "workspace.yaml").write_text("name: link-target\n", encoding="utf-8")
    held_workspace = tmp_path / "held-workspace"
    real_open = os.open
    replaced = False

    def replace_ancestor() -> None:
        nonlocal replaced
        if replaced:
            return
        workspace.rename(held_workspace)
        workspace.symlink_to(outside, target_is_directory=True)
        replaced = True

    def racing_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if (dir_fd is None and Path(path) == config_path) or (dir_fd is not None and path == workspace.name):
            replace_ancestor()
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)

    with pytest.raises(WorkspaceConfigError, match="symlink"):
        load_workspace_config(config_path)
    assert replaced


def test_load_workspace_config_reads_from_open_ancestor_after_path_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace-under-test"
    config_dir = workspace / ".open-xquant"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside_config_dir = outside / ".open-xquant"
    outside_config_dir.mkdir(parents=True)
    (outside_config_dir / "workspace.yaml").write_text("name: link-target\n", encoding="utf-8")
    held_workspace = tmp_path / "held-workspace"
    real_open = os.open
    replaced = False

    def replace_ancestor() -> None:
        nonlocal replaced
        if replaced:
            return
        workspace.rename(held_workspace)
        workspace.symlink_to(outside, target_is_directory=True)
        replaced = True

    def racing_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is None and Path(path) == config_path:
            replace_ancestor()
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is not None and path == workspace.name:
            replace_ancestor()
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)

    assert load_workspace_config(config_path) == {"name": "trusted"}
    assert replaced


def test_load_workspace_config_fstats_and_closes_every_opened_descriptor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "workspace" / ".open-xquant"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    real_open = os.open
    real_fstat = os.fstat
    opened: list[int] = []
    fstatted: list[int] = []

    def tracking_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def tracking_fstat(descriptor: int) -> os.stat_result:
        fstatted.append(descriptor)
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "fstat", tracking_fstat)

    assert load_workspace_config(config_path) == {"name": "trusted"}
    assert len(opened) >= 3
    assert fstatted == opened
    for descriptor in set(opened):
        with pytest.raises(OSError):
            real_fstat(descriptor)


def test_load_workspace_config_fails_closed_without_nofollow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    with pytest.raises(WorkspaceConfigError, match="nofollow"):
        load_workspace_config(config_path)


def test_load_workspace_config_fails_closed_without_directory_nofollow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)

    with pytest.raises(WorkspaceConfigError, match="nofollow"):
        load_workspace_config(config_path)


def test_load_workspace_config_fails_closed_without_directory_descriptor_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    config_path = config_dir / "workspace.yaml"
    config_path.write_text("name: trusted\n", encoding="utf-8")
    real_open = os.open

    def no_directory_descriptor_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is not None:
            raise NotImplementedError("dir_fd unavailable")
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", no_directory_descriptor_open)

    with pytest.raises(WorkspaceConfigError, match="nofollow"):
        load_workspace_config(config_path)


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
