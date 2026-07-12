from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oxq import process_lock
from oxq.cli import agent as agent_module
from oxq.cli import research as research_module


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX rename semantics")
@pytest.mark.parametrize("had_destination", [False, True])
def test_agent_commit_rejects_ordinary_parent_replacement_after_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    had_destination: bool,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    parent = home / ".cursor/skills"
    parent.mkdir(parents=True)
    destination = parent / "build-strategy-spec"
    if had_destination:
        destination.mkdir()
        (destination / "SKILL.md").write_text("original\n", encoding="utf-8")

    staging_root = tmp_path / "staging"
    staged_skill = staging_root / destination.name
    staged_skill.mkdir(parents=True)
    (staged_skill / "SKILL.md").write_text("managed\n", encoding="utf-8")
    staged_manifest = staging_root / "agent-install.json"
    agent_module.write_json_file(
        staged_manifest,
        {
            "schema_version": 1,
            "targets": {
                "cursor": {
                    "installed": True,
                    "skills_dir": str(parent),
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
        },
    )

    displaced = tmp_path / "displaced-skills"
    original_prepare = agent_module._prepare_lifecycle_operations

    def prepare_then_replace(operations):
        prepared = original_prepare(operations)
        parent.replace(displaced)
        parent.mkdir()
        (parent / "unrelated.txt").write_text("keep\n", encoding="utf-8")
        if had_destination:
            destination.mkdir()
            (destination / "keep.txt").write_text("unrelated\n", encoding="utf-8")
        return prepared

    monkeypatch.setattr(
        agent_module,
        "_prepare_lifecycle_operations",
        prepare_then_replace,
    )

    with pytest.raises(Exception, match="identity|parent.*changed"):
        agent_module._commit_target_upgrade(
            [
                (destination, staged_skill),
                (agent_module.manifest_path(), staged_manifest),
            ]
        )

    assert (parent / "unrelated.txt").read_text(encoding="utf-8") == "keep\n"
    if had_destination:
        assert (destination / "keep.txt").read_text(encoding="utf-8") == "unrelated\n"
    else:
        assert not destination.exists()
    assert agent_module.lifecycle_transaction_path().is_file()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX rename semantics")
@pytest.mark.parametrize("had_destination", [False, True])
def test_agent_recovery_rejects_ordinary_parent_replacement_after_journal_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    had_destination: bool,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    parent = home / ".cursor/skills"
    parent.mkdir(parents=True)
    destination = parent / "build-strategy-spec"
    destination.mkdir()
    (destination / "installed.txt").write_text("installed\n", encoding="utf-8")
    backup = parent / ".build-strategy-spec.backup-deadbeef"
    if had_destination:
        backup.mkdir()
        (backup / "original.txt").write_text("original\n", encoding="utf-8")

    config_root = home / ".config/open-xquant"
    config_root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(parent),
                "agents_dir": None,
                "instruction_file": None,
                "config_file": None,
                "installed_paths": [str(destination)],
                "skills": [],
                "agent_roles": [],
                "managed_blocks": [],
            }
        },
        "sdk_bundles": [],
    }
    agent_module.manifest_path().write_text(json.dumps(manifest), encoding="utf-8")
    agent_module._write_lifecycle_manifest_witness(manifest)
    transaction = agent_module.lifecycle_transaction_path()
    with agent_module._secure_recovery_mutations([destination, backup]) as mutations:
        replacement_evidence = mutations.path_evidence(destination)
        original_evidence = mutations.path_evidence(backup) if had_destination else {"kind": "absent"}
    transaction.write_text(
        json.dumps(
            {
                "schema_version": agent_module.LIFECYCLE_TRANSACTION_SCHEMA_VERSION,
                "transaction_type": "agent-lifecycle",
                "phase": "prepared",
                "staging_complete": True,
                "created_at": "2026-01-01T00:00:00Z",
                "journal_parent_identity": agent_module._path_parent_identity(transaction),
                "operations": [
                    {
                        "destination": str(destination),
                        "staged": str(tmp_path / "consumed-stage"),
                        "local_staged": str(parent / ".build-strategy-spec.install-deadbeef"),
                        "backup": str(backup) if had_destination else None,
                        "had_destination": had_destination,
                        "relative_name": destination.name,
                        "parent_identity": agent_module._path_parent_identity(destination),
                        "original_evidence": original_evidence,
                        "replacement_evidence": replacement_evidence,
                    }
                ],
                "created_parents": [],
                "rollback_cleanup_paths": [],
                "trusted_roots": agent_module._lifecycle_trusted_root_identities(
                    agent_module._purge_trusted_roots(manifest)
                ),
            }
        ),
        encoding="utf-8",
    )

    displaced = tmp_path / "displaced-skills"
    replacement = tmp_path / "replacement-skills"
    replacement.mkdir()
    replacement_destination = replacement / destination.name
    replacement_destination.mkdir()
    (replacement_destination / "keep.txt").write_text("keep\n", encoding="utf-8")
    if had_destination:
        replacement_backup = replacement / backup.name
        replacement_backup.mkdir()
        (replacement_backup / "keep.txt").write_text("backup keep\n", encoding="utf-8")
    original_validate = agent_module._validated_pending_lifecycle_transaction

    def validate_then_replace(path: Path) -> dict:
        payload = original_validate(path)
        parent.replace(displaced)
        replacement.replace(parent)
        return payload

    monkeypatch.setattr(
        agent_module,
        "_validated_pending_lifecycle_transaction",
        validate_then_replace,
    )

    with pytest.raises(Exception, match="identity|parent.*changed"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    if had_destination:
        assert (parent / backup.name / "keep.txt").read_text(encoding="utf-8") == "backup keep\n"
    assert transaction.is_file()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX rename semantics")
@pytest.mark.parametrize("had_destination", [False, True])
def test_agent_finish_rejects_ordinary_parent_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    had_destination: bool,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    parent = home / ".cursor/skills"
    parent.mkdir(parents=True)
    destination = parent / "build-strategy-spec"
    if had_destination:
        destination.write_text("original\n", encoding="utf-8")
    staged = tmp_path / "staged"
    staged.write_text("managed\n", encoding="utf-8")
    staged_manifest = tmp_path / "agent-install.json"
    agent_module.write_json_file(
        staged_manifest,
        {
            "schema_version": 1,
            "targets": {
                "cursor": {
                    "installed": True,
                    "skills_dir": str(parent),
                    "agents_dir": None,
                    "instruction_file": None,
                    "config_file": None,
                    "installed_paths": [str(destination)],
                    "skills": [{"dest": str(destination)}],
                    "agent_roles": [],
                    "managed_blocks": [],
                }
            },
            "sdk_bundles": [],
        },
    )

    committed = agent_module._commit_target_upgrade(
        [
            (destination, staged),
            (agent_module.manifest_path(), staged_manifest),
        ]
    )
    backup = committed[0][1]
    displaced = tmp_path / "displaced-skills"
    parent.replace(displaced)
    parent.mkdir()
    (parent / "keep.txt").write_text("keep\n", encoding="utf-8")
    if backup is not None:
        replacement_backup = parent / backup.name
        replacement_backup.write_text("unrelated backup\n", encoding="utf-8")

    with pytest.raises(Exception, match="identity|parent.*changed"):
        agent_module._finish_committed_lifecycle_transaction(committed)

    assert (parent / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    if backup is not None:
        assert (parent / backup.name).read_text(encoding="utf-8") == "unrelated backup\n"
    assert agent_module.lifecycle_transaction_path().is_file()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX rename semantics")
@pytest.mark.parametrize("had_original", [False, True])
def test_governance_commit_rejects_ordinary_parent_replacement_after_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    had_original: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="round-23")
    parent = workspace / "versions/v001"
    destination = parent / "phase_state.json"
    if not had_original:
        destination.unlink()
    displaced = workspace / "versions/v001-displaced"
    original_write_journal = research_module._write_governance_transaction_journal
    swapped = False

    def write_journal_then_replace(root: Path, journal: dict[str, object]) -> None:
        nonlocal swapped
        original_write_journal(root, journal)
        if journal.get("state") != "prepared" or swapped:
            return
        swapped = True
        parent.replace(displaced)
        parent.mkdir()
        (parent / "unrelated.txt").write_text("keep\n", encoding="utf-8")
        if had_original:
            destination.write_text("unrelated destination\n", encoding="utf-8")

    monkeypatch.setattr(
        research_module,
        "_write_governance_transaction_journal",
        write_journal_then_replace,
    )

    with pytest.raises(Exception, match="identity|parent.*changed|journal preserved"):
        research_module._write_governance_files_atomically(
            workspace,
            {destination: json.dumps({"status": "managed"}) + "\n"},
        )

    assert swapped
    assert (parent / "unrelated.txt").read_text(encoding="utf-8") == "keep\n"
    if had_original:
        assert destination.read_text(encoding="utf-8") == "unrelated destination\n"
    else:
        assert not destination.exists()
    assert (workspace / ".open-xquant/governance-transaction.json").is_file()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX rename semantics")
@pytest.mark.parametrize("had_original", [False, True])
def test_governance_recovery_rejects_ordinary_parent_replacement_after_journal_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    had_original: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="round-23-recovery")
    parent = workspace / "versions/v001"
    destination = parent / "phase_state.json"
    destination.write_text("installed\n", encoding="utf-8")
    transaction_id = "a" * 32
    stage, backup = research_module._governance_transaction_artifacts(
        destination,
        transaction_id,
    )
    if had_original:
        backup.write_text("original\n", encoding="utf-8")
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": research_module._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "state": "prepared",
                "journal_parent_identity": research_module._governance_identity_payload(journal_path.parent.stat()),
                "entries": [
                    {
                        "destination": "versions/v001/phase_state.json",
                        "had_original": had_original,
                        "parent_identity": research_module._governance_identity_payload(parent.stat()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    displaced = workspace / "versions/v001-displaced"
    replacement = workspace / "versions/v001-replacement"
    replacement.mkdir()
    (replacement / "phase_state.json").write_text(
        "unrelated destination\n",
        encoding="utf-8",
    )
    if had_original:
        (replacement / backup.name).write_text("unrelated backup\n", encoding="utf-8")
    original_read = research_module._read_governance_transaction_journal

    def read_then_replace(path: Path) -> dict[str, object]:
        journal = original_read(path)
        parent.replace(displaced)
        replacement.replace(parent)
        return journal

    monkeypatch.setattr(
        research_module,
        "_read_governance_transaction_journal",
        read_then_replace,
    )

    with pytest.raises(Exception, match="identity|parent.*changed|journal preserved"):
        research_module._recover_governance_transaction(workspace)

    assert destination.read_text(encoding="utf-8") == "unrelated destination\n"
    if had_original:
        assert (parent / backup.name).read_text(encoding="utf-8") == "unrelated backup\n"
    assert journal_path.is_file()


def test_agent_and_research_locks_share_verified_user_runtime_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(process_lock.tempfile, "gettempdir", lambda: str(tmp_path))

    agent_path = agent_module.lifecycle_lock_path()
    research_path = research_module._workspace_init_lock_path(tmp_path / "workspace")

    assert agent_path.parent.parent == research_path.parent.parent
    assert agent_path.parent.name == "agent"
    assert research_path.parent.name == "research"
