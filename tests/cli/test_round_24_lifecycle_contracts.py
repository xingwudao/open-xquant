from __future__ import annotations

import os
from pathlib import Path

import pytest

from oxq.cli import agent as agent_module


@pytest.mark.parametrize("alias_kind", ["dotdot", "symlink"])
def test_agent_lifecycle_lock_uses_config_root_filesystem_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    alias_kind: str,
) -> None:
    config_root = tmp_path / "home/.config/open-xquant"
    config_root.mkdir(parents=True)
    if alias_kind == "dotdot":
        alias_parent = config_root.parent / "path-alias"
        alias_parent.mkdir()
        alias = alias_parent / ".." / config_root.name
    else:
        alias = tmp_path / "config-root-alias"
        try:
            alias.symlink_to(config_root, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlink unavailable: {exc}")

    monkeypatch.setattr(agent_module, "verified_user_runtime_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(agent_module, "config_dir", lambda: config_root)
    canonical_lock = agent_module.lifecycle_lock_path()
    monkeypatch.setattr(agent_module, "config_dir", lambda: alias)

    assert agent_module.lifecycle_lock_path() == canonical_lock


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory rename semantics")
def test_agent_commit_config_root_swap_keeps_journal_off_replacement_and_rolls_back_skill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    config_root = home / ".config/open-xquant"
    config_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    skills_root = home / ".cursor/skills"
    destination = skills_root / "build-strategy-spec"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("original skill\n", encoding="utf-8")

    staging_root = tmp_path / "staging"
    staged_skill = staging_root / destination.name
    staged_skill.mkdir(parents=True)
    (staged_skill / "SKILL.md").write_text("replacement skill\n", encoding="utf-8")
    staged_manifest = staging_root / "agent-install.json"
    manifest = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(skills_root),
                "agents_dir": None,
                "instruction_file": None,
                "config_file": None,
                "installed_paths": [str(destination)],
                "skills": [{"dest": str(destination / "SKILL.md")}],
                "agent_roles": [],
                "managed_blocks": [],
            }
        },
        "sdk_bundles": [],
    }
    agent_module.write_json_file(agent_module.manifest_path(), manifest)
    agent_module.write_json_file(staged_manifest, manifest)

    displaced_config = tmp_path / "displaced-config-root"
    replacement_sentinel = config_root / "replacement-owner.txt"
    original_parent_identity = agent_module._PosixRecoveryMutations.parent_identity
    journal_identity_calls = 0
    swapped = False

    def parent_identity_then_swap(
        self: agent_module._PosixRecoveryMutations,
        path: Path,
    ) -> dict[str, int]:
        nonlocal journal_identity_calls, swapped
        identity = original_parent_identity(self, path)
        if path == agent_module.lifecycle_transaction_path():
            journal_identity_calls += 1
            if journal_identity_calls == 3:
                config_root.replace(displaced_config)
                config_root.mkdir()
                replacement_sentinel.write_text("keep replacement root\n", encoding="utf-8")
                swapped = True
        return identity

    monkeypatch.setattr(
        agent_module._PosixRecoveryMutations,
        "parent_identity",
        parent_identity_then_swap,
    )

    with pytest.raises(Exception, match="identity changed|changed after validation"):
        agent_module._commit_target_upgrade(
            [
                (destination, staged_skill),
                (agent_module.manifest_path(), staged_manifest),
            ]
        )

    assert swapped
    assert (destination / "SKILL.md").read_text(encoding="utf-8") == "original skill\n"
    assert replacement_sentinel.read_text(encoding="utf-8") == "keep replacement root\n"
    assert not (config_root / "agent-lifecycle-transaction.json").exists()
    assert not (config_root / "agent-lifecycle-manifest-witness.json").exists()
    assert not (config_root / "agent-lifecycle-manifest-witness.sha256").exists()
    assert not list(skills_root.glob(f".{destination.name}.backup-*"))
    assert not list(skills_root.glob(f".{destination.name}.install-*"))
