from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
import pytest

from oxq.cli import agent as agent_module


def _commit_with_bundle_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    bundle_root = agent_module.config_dir() / "sdk-bundles" / "bundle-round26"
    bundle_root.mkdir(parents=True)
    (bundle_root / "runner.txt").write_text("owned bundle\n", encoding="utf-8")
    staged_manifest = tmp_path / "staging" / "agent-install.json"
    staged_manifest.parent.mkdir()
    staged_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "targets": {},
                "sdk_bundles": [{"id": "bundle-round26", "root": str(bundle_root)}],
            }
        ),
        encoding="utf-8",
    )
    agent_module._commit_target_upgrade(
        [(agent_module.manifest_path(), staged_manifest)],
        rollback_cleanup_paths=[bundle_root],
    )
    return bundle_root, agent_module.lifecycle_transaction_path()


def test_rollback_cleanup_root_records_complete_ownership_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_root, transaction = _commit_with_bundle_cleanup(monkeypatch, tmp_path)

    record = json.loads(transaction.read_text(encoding="utf-8"))["rollback_cleanup_paths"][0]

    assert record["path"] == str(bundle_root)
    assert set(record["evidence"]) == {
        "kind",
        "device",
        "inode",
        "mode",
        "tree_sha256",
    }
    assert record["evidence"]["kind"] == "directory"


def test_prepared_recovery_preserves_replaced_sdk_cleanup_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_root, transaction = _commit_with_bundle_cleanup(monkeypatch, tmp_path)
    agent_module._mark_lifecycle_transaction_prepared()
    shutil.rmtree(bundle_root)
    bundle_root.mkdir()
    replacement = bundle_root / "keep.txt"
    replacement.write_text("unrelated replacement\n", encoding="utf-8")

    with pytest.raises(click.ClickException, match="changed|evidence|generation"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert replacement.read_text(encoding="utf-8") == "unrelated replacement\n"
    assert transaction.is_file()


_CRASH_DURING_LOCAL_COPY = r"""
import os
import sys
from pathlib import Path

from oxq.cli import agent as agent_module

home = Path(sys.argv[1])
staged = Path(sys.argv[2])
os.environ["HOME"] = str(home)
os.environ["USERPROFILE"] = str(home)
agent_module.config_dir().mkdir(parents=True)
mutation_type = (
    agent_module._WindowsRecoveryMutations
    if os.name == "nt"
    else agent_module._PosixRecoveryMutations
)


def crash_with_partial_copy(self, source, destination):
    del self, source
    if not (destination.exists() or destination.is_symlink()):
        os._exit(72)
    if staged.is_dir() != destination.is_dir():
        os._exit(74)
    if destination.is_dir():
        (destination / "partial.txt").write_text("partial copy\n", encoding="utf-8")
    else:
        destination.write_text("partial copy\n", encoding="utf-8")
    os._exit(73)


mutation_type.copy_from = crash_with_partial_copy
agent_module._commit_target_upgrade([(agent_module.agent_config_path(), staged)])
"""


@pytest.mark.parametrize("artifact_kind", ["file", "directory"])
@pytest.mark.parametrize("replace_generation", [False, True])
def test_incomplete_local_copy_uses_journaled_placeholder_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_kind: str,
    replace_generation: bool,
) -> None:
    home = tmp_path / "home"
    staged = tmp_path / "staging" / "agent.yaml"
    staged.parent.mkdir()
    if artifact_kind == "directory":
        staged.mkdir()
        (staged / "complete.txt").write_text("complete source\n", encoding="utf-8")
    else:
        staged.write_text("complete source\n", encoding="utf-8")

    crashed = subprocess.run(
        [sys.executable, "-c", _CRASH_DURING_LOCAL_COPY, str(home), str(staged)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=15,
        check=False,
    )
    assert crashed.returncode == 73, crashed.stdout + crashed.stderr

    monkeypatch.setenv("HOME", str(home))
    transaction = agent_module.lifecycle_transaction_path()
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    assert payload["staging_complete"] is False
    operation = payload["operations"][0]
    local_staged = Path(operation["local_staged"])
    evidence = operation["local_staged_evidence"]
    assert evidence["kind"] == artifact_kind
    assert set(evidence) == {
        "kind",
        "device",
        "inode",
        "mode",
        "tree_sha256",
    }

    if replace_generation:
        if local_staged.is_dir():
            shutil.rmtree(local_staged)
            local_staged.mkdir()
            replacement = local_staged / "keep.txt"
        else:
            local_staged.unlink()
            replacement = local_staged
        replacement.write_text("unrelated replacement\n", encoding="utf-8")

        with pytest.raises(click.ClickException, match="changed|evidence|generation"):
            agent_module._recover_pending_lifecycle_transaction(dry_run=False)

        assert replacement.read_text(encoding="utf-8") == "unrelated replacement\n"
        assert transaction.is_file()
    else:
        assert agent_module._recover_pending_lifecycle_transaction(dry_run=False)
        assert not local_staged.exists()
        assert not transaction.exists()


def _write_purge_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    backup_kind: str,
) -> tuple[Path, Path]:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_root = agent_module.config_dir()
    skills_root = home / ".cursor" / "skills"
    skill_destination = skills_root / "build-strategy-spec"
    manifest_payload = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(skills_root),
                "agents_dir": None,
                "instruction_file": None,
                "config_file": None,
                "installed_paths": [str(skill_destination)],
                "agent_roles": [],
                "managed_blocks": [],
            }
        },
        "sdk_bundles": [],
    }
    manifest_backup = config_root / ".agent-install.json.backup-round26"
    manifest_backup.parent.mkdir(parents=True)
    manifest_backup.write_text(json.dumps(manifest_payload), encoding="utf-8")

    if backup_kind == "skill":
        destination = skill_destination
        backup = skills_root / ".build-strategy-spec.backup-round26"
        backup.mkdir(parents=True)
        (backup / "SKILL.md").write_text("retained skill\n", encoding="utf-8")
    elif backup_kind == "config":
        destination = agent_module.agent_config_path()
        backup = config_root / ".agent.yaml.backup-round26"
        backup.write_text("agent_profile: standalone-agent\n", encoding="utf-8")
    else:
        raise AssertionError(f"unsupported backup kind: {backup_kind}")

    agent_module._write_pending_purge_cleanup(
        [
            (destination, backup),
            (agent_module.manifest_path(), manifest_backup),
        ]
    )
    return backup, agent_module.purge_transaction_path()


@pytest.mark.parametrize("backup_kind", ["skill", "config"])
def test_purge_sidecar_v3_records_every_backup_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backup_kind: str,
) -> None:
    _backup, transaction = _write_purge_sidecar(
        monkeypatch,
        tmp_path,
        backup_kind=backup_kind,
    )

    payload = json.loads(transaction.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 3
    assert len(payload["backups"]) == 2
    for record in payload["backups"]:
        assert set(record["evidence"]) == {
            "kind",
            "device",
            "inode",
            "mode",
            "tree_sha256",
        }


@pytest.mark.parametrize("backup_kind", ["skill", "config"])
def test_purge_recovery_preserves_replaced_retained_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backup_kind: str,
) -> None:
    backup, transaction = _write_purge_sidecar(
        monkeypatch,
        tmp_path,
        backup_kind=backup_kind,
    )
    if backup.is_dir():
        shutil.rmtree(backup)
        backup.mkdir()
        replacement = backup / "keep.txt"
    else:
        backup.unlink()
        replacement = backup
    replacement.write_text("unrelated replacement\n", encoding="utf-8")

    with pytest.raises(click.ClickException, match="backup|evidence|identity|changed"):
        agent_module._recover_pending_purge_cleanup(dry_run=False)

    assert replacement.read_text(encoding="utf-8") == "unrelated replacement\n"
    assert transaction.is_file()


_CRASH_BEFORE_PURGE_BACKUP_CLEANUP = r"""
import os
import sys
from pathlib import Path

from oxq.cli import agent as agent_module

home = Path(sys.argv[1])
os.environ["HOME"] = str(home)
os.environ["USERPROFILE"] = str(home)


def crash_before_cleanup(path):
    if ".backup-" in path.name:
        os._exit(75)
    raise AssertionError(f"unexpected purge cleanup path: {path}")


agent_module._remove_upgrade_path = crash_before_cleanup
agent_module._recover_pending_purge_cleanup(dry_run=False)
"""


def test_purge_recovery_after_crash_rejects_replaced_skill_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup, transaction = _write_purge_sidecar(
        monkeypatch,
        tmp_path,
        backup_kind="skill",
    )
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH_BEFORE_PURGE_BACKUP_CLEANUP,
            str(tmp_path / "home"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=15,
        check=False,
    )
    assert crashed.returncode == 75, crashed.stdout + crashed.stderr
    assert transaction.is_file()
    shutil.rmtree(backup)
    backup.mkdir()
    replacement = backup / "keep.txt"
    replacement.write_text("replacement after crash\n", encoding="utf-8")

    with pytest.raises(click.ClickException, match="backup|evidence|identity|changed"):
        agent_module._recover_pending_purge_cleanup(dry_run=False)

    assert replacement.read_text(encoding="utf-8") == "replacement after crash\n"
    assert transaction.is_file()


def _arm_replacement_after_evidence(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> dict[str, object]:
    mutation_type = agent_module._WindowsRecoveryMutations if os.name == "nt" else agent_module._PosixRecoveryMutations
    original_evidence = mutation_type.path_evidence
    original_remove = mutation_type.remove
    original_quarantine = mutation_type.quarantine
    state: dict[str, object] = {
        "armed": False,
        "swapped": False,
        "replacement": None,
    }

    def swap_generation() -> None:
        if state["swapped"]:
            return
        state["swapped"] = True
        if target.is_dir():
            shutil.rmtree(target)
            target.mkdir()
            replacement = target / "keep.txt"
        else:
            target.unlink()
            replacement = target
        replacement.write_text("barrier replacement\n", encoding="utf-8")
        state["replacement"] = replacement

    def evidence_then_arm(self, path: Path) -> dict[str, object]:
        evidence = original_evidence(self, path)
        if path == target and evidence != {"kind": "absent"}:
            state["armed"] = True
        return evidence

    def remove_after_barrier(self, path: Path) -> None:
        if path == target and state["armed"]:
            swap_generation()
        original_remove(self, path)

    def quarantine_after_barrier(self, source: Path, destination: Path) -> None:
        if source == target and state["armed"]:
            swap_generation()
        original_quarantine(self, source, destination)

    monkeypatch.setattr(mutation_type, "path_evidence", evidence_then_arm)
    monkeypatch.setattr(mutation_type, "remove", remove_after_barrier)
    monkeypatch.setattr(mutation_type, "quarantine", quarantine_after_barrier)
    return state


def _assert_barrier_replacement_survived(state: dict[str, object]) -> None:
    assert state["armed"] is True
    assert state["swapped"] is True
    replacement = state["replacement"]
    assert isinstance(replacement, Path)
    assert replacement.read_text(encoding="utf-8") == "barrier replacement\n"


def test_rollback_destination_quarantines_after_evidence_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.parent.mkdir(parents=True)
    staged = tmp_path / "staging" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text("agent_profile: multi-agent\n", encoding="utf-8")
    agent_module._commit_target_upgrade([(destination, staged)])
    agent_module._mark_lifecycle_transaction_prepared()
    transaction = agent_module.lifecycle_transaction_path()
    state = _arm_replacement_after_evidence(monkeypatch, destination)

    with pytest.raises(click.ClickException, match="changed|evidence|quarantine"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    _assert_barrier_replacement_survived(state)
    assert transaction.is_file()


def _write_incomplete_local_stage_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.parent.mkdir(parents=True)
    local_staged = destination.parent / ".agent.yaml.install-deadbeef"
    local_staged.mkdir()
    (local_staged / "partial.txt").write_text("partial copy\n", encoding="utf-8")
    with agent_module._secure_recovery_mutations([local_staged]) as mutations:
        local_staged_evidence = mutations.path_evidence(local_staged)
    manifest = {"targets": {}, "sdk_bundles": []}
    agent_module._write_lifecycle_manifest_witness(manifest)
    transaction = agent_module.lifecycle_transaction_path()
    transaction.write_text(
        json.dumps(
            {
                "schema_version": agent_module.LIFECYCLE_TRANSACTION_SCHEMA_VERSION,
                "transaction_type": "agent-lifecycle",
                "phase": "prepared",
                "staging_complete": False,
                "created_at": "2026-01-01T00:00:00Z",
                "journal_parent_identity": agent_module._path_parent_identity(transaction),
                "operations": [
                    {
                        "destination": str(destination),
                        "staged": str(tmp_path / "consumed-stage"),
                        "local_staged": str(local_staged),
                        "backup": None,
                        "had_destination": False,
                        "relative_name": destination.name,
                        "parent_identity": agent_module._path_parent_identity(destination),
                        "original_evidence": {"kind": "absent"},
                        "local_staged_evidence": local_staged_evidence,
                        "replacement_evidence": None,
                    }
                ],
                "created_parents": [],
                "rollback_cleanup_paths": [],
                "trusted_roots": agent_module._lifecycle_trusted_root_identities(agent_module._purge_trusted_roots(manifest)),
            }
        ),
        encoding="utf-8",
    )
    return local_staged, transaction


def test_prepared_local_stage_quarantines_after_evidence_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_staged, transaction = _write_incomplete_local_stage_journal(
        monkeypatch,
        tmp_path,
    )
    state = _arm_replacement_after_evidence(monkeypatch, local_staged)

    with pytest.raises(click.ClickException, match="changed|evidence|quarantine"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    _assert_barrier_replacement_survived(state)
    assert transaction.is_file()


def test_prepared_sdk_cleanup_quarantines_after_evidence_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_root, transaction = _commit_with_bundle_cleanup(monkeypatch, tmp_path)
    agent_module._mark_lifecycle_transaction_prepared()
    state = _arm_replacement_after_evidence(monkeypatch, bundle_root)

    with pytest.raises(click.ClickException, match="changed|evidence|quarantine"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    _assert_barrier_replacement_survived(state)
    assert transaction.is_file()


def test_committed_backup_quarantines_after_evidence_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.parent.mkdir(parents=True)
    destination.write_text("agent_profile: standalone-agent\n", encoding="utf-8")
    staged = tmp_path / "staging" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text("agent_profile: multi-agent\n", encoding="utf-8")
    committed = agent_module._commit_target_upgrade([(destination, staged)])
    backup = committed[0][1]
    assert backup is not None
    transaction = agent_module.lifecycle_transaction_path()
    state = _arm_replacement_after_evidence(monkeypatch, backup)

    with pytest.raises(click.ClickException, match="changed|evidence|quarantine"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    _assert_barrier_replacement_survived(state)
    assert transaction.is_file()


_CRASH_AFTER_QUARANTINE = r"""
import os
import sys
from pathlib import Path

from oxq.cli import agent as agent_module

home = Path(sys.argv[1])
recovery_kind = sys.argv[2]
os.environ["HOME"] = str(home)
os.environ["USERPROFILE"] = str(home)
mutation_type = (
    agent_module._WindowsRecoveryMutations
    if os.name == "nt"
    else agent_module._PosixRecoveryMutations
)
original_quarantine = mutation_type.quarantine


def quarantine_then_crash(self, source, destination):
    original_quarantine(self, source, destination)
    os._exit(76)


mutation_type.quarantine = quarantine_then_crash
if recovery_kind == "lifecycle":
    agent_module._recover_pending_lifecycle_transaction(dry_run=False)
else:
    agent_module._recover_pending_purge_cleanup(dry_run=False)
"""


def _crash_recovery_after_quarantine(home: Path, recovery_kind: str) -> None:
    crashed = subprocess.run(
        [sys.executable, "-c", _CRASH_AFTER_QUARANTINE, str(home), recovery_kind],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=15,
        check=False,
    )
    assert crashed.returncode == 76, crashed.stdout + crashed.stderr


def test_lifecycle_retry_recovers_crash_after_atomic_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.parent.mkdir(parents=True)
    staged = tmp_path / "staging" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text("agent_profile: multi-agent\n", encoding="utf-8")
    agent_module._commit_target_upgrade([(destination, staged)])
    agent_module._mark_lifecycle_transaction_prepared()

    _crash_recovery_after_quarantine(home, "lifecycle")

    assert agent_module.lifecycle_transaction_path().is_file()
    assert list(destination.parent.glob(".oxq-quarantine-*"))
    assert agent_module._recover_pending_lifecycle_transaction(dry_run=False)
    assert not list(destination.parent.glob(".oxq-quarantine-*"))
    assert not agent_module.lifecycle_transaction_path().exists()


def test_purge_retry_recovers_crash_after_atomic_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup, transaction = _write_purge_sidecar(
        monkeypatch,
        tmp_path,
        backup_kind="skill",
    )

    _crash_recovery_after_quarantine(tmp_path / "home", "purge")

    assert transaction.is_file()
    assert list(backup.parent.glob(".oxq-quarantine-*"))
    assert agent_module._recover_pending_purge_cleanup(dry_run=False)
    assert not list(backup.parent.glob(".oxq-quarantine-*"))
    assert not transaction.exists()
