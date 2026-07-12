from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click
import pytest

from oxq.cli import agent as agent_module


def test_commit_rejects_destination_edit_after_local_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.parent.mkdir(parents=True)
    destination.write_text("agent_profile: standalone-agent\n", encoding="utf-8")
    staged = tmp_path / "stage" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text("agent_profile: multi-agent\n", encoding="utf-8")
    original_write = agent_module._write_lifecycle_transaction
    edited = False

    def write_then_edit(**kwargs: object) -> None:
        nonlocal edited
        original_write(**kwargs)
        if kwargs["phase"] == "prepared" and kwargs["staging_complete"] is True and not edited:
            destination.write_text("agent_profile: external-edit\n", encoding="utf-8")
            edited = True

    monkeypatch.setattr(agent_module, "_write_lifecycle_transaction", write_then_edit)

    with pytest.raises(click.ClickException, match="changed|evidence|unrecognized"):
        agent_module._commit_target_upgrade([(destination, staged)])

    assert edited
    assert destination.read_text(encoding="utf-8") == "agent_profile: external-edit\n"
    assert agent_module.lifecycle_transaction_path().is_file()


def test_commit_rejects_nested_tree_edit_after_local_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.mkdir(parents=True)
    original_child = destination / "nested" / "profile.txt"
    original_child.parent.mkdir()
    original_child.write_text("original\n", encoding="utf-8")
    staged = tmp_path / "stage" / "agent.yaml"
    staged.mkdir(parents=True)
    (staged / "profile.txt").write_text("replacement\n", encoding="utf-8")
    original_write = agent_module._write_lifecycle_transaction

    def write_then_edit(**kwargs: object) -> None:
        original_write(**kwargs)
        if kwargs["phase"] == "prepared" and kwargs["staging_complete"] is True:
            original_child.write_text("external nested edit\n", encoding="utf-8")

    monkeypatch.setattr(agent_module, "_write_lifecycle_transaction", write_then_edit)

    with pytest.raises(click.ClickException, match="changed|evidence|unrecognized"):
        agent_module._commit_target_upgrade([(destination, staged)])

    assert original_child.read_text(encoding="utf-8") == "external nested edit\n"
    assert agent_module.lifecycle_transaction_path().is_file()


def test_prepared_recovery_preserves_fresh_destination_created_after_install_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    staged = tmp_path / "stage" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text("agent_profile: multi-agent\n", encoding="utf-8")
    code = r"""
import os
import sys
from pathlib import Path

from oxq.cli import agent as agent_module

home = Path(sys.argv[1])
staged = Path(sys.argv[2])
os.environ["HOME"] = str(home)
agent_module.config_dir().mkdir(parents=True)
mutation_type = (
    agent_module._WindowsRecoveryMutations
    if os.name == "nt"
    else agent_module._PosixRecoveryMutations
)
original_replace = mutation_type.replace

def crash_after_install(self, source, destination):
    result = original_replace(self, source, destination)
    if source.name.startswith(".agent.yaml.install-") and destination == agent_module.agent_config_path():
        os._exit(73)
    return result

mutation_type.replace = crash_after_install
agent_module._commit_target_upgrade([(agent_module.agent_config_path(), staged)])
"""
    crashed = subprocess.run(
        [sys.executable, "-c", code, str(home), str(staged)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=15,
        check=False,
    )
    assert crashed.returncode == 73, crashed.stdout + crashed.stderr

    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    assert destination.read_text(encoding="utf-8") == "agent_profile: multi-agent\n"
    destination.unlink()
    destination.write_text("agent_profile: external-fresh\n", encoding="utf-8")
    transaction = agent_module.lifecycle_transaction_path()

    with pytest.raises(click.ClickException, match="changed|evidence|unrecognized"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert destination.read_text(encoding="utf-8") == "agent_profile: external-fresh\n"
    assert transaction.is_file()


@pytest.mark.parametrize("alias_kind", ["case", "ancestor-alias"])
def test_missing_config_root_aliases_share_bootstrap_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    alias_kind: str,
) -> None:
    monkeypatch.setattr(agent_module, "verified_user_runtime_root", lambda: tmp_path / "runtime")
    if alias_kind == "case":
        canonical = tmp_path / "Config" / "Open-XQuant"
        alternate = tmp_path / "config" / "open-xquant"
    else:
        backing = tmp_path / "backing"
        backing.mkdir()
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(backing, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks are unavailable: {exc}")
        canonical = backing / "Config" / "Open-XQuant"
        alternate = alias / "config" / "open-xquant"

    monkeypatch.setattr(agent_module, "config_dir", lambda: canonical)
    first_bootstrap = agent_module._lifecycle_bootstrap_lock_path()
    first_lifecycle = agent_module.lifecycle_lock_path()
    monkeypatch.setattr(agent_module, "config_dir", lambda: alternate)

    assert agent_module._lifecycle_bootstrap_lock_path() == first_bootstrap
    assert agent_module.lifecycle_lock_path() == first_lifecycle


def test_lifecycle_lock_transitions_from_missing_location_to_filesystem_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "missing-home" / ".config" / "open-xquant"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setattr(agent_module, "config_dir", lambda: config_root)
    monkeypatch.setattr(agent_module, "verified_user_runtime_root", lambda: runtime_root)
    initial_bootstrap = agent_module._lifecycle_bootstrap_lock_path()
    acquired: list[Path] = []

    class RecordingLock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> RecordingLock:
            acquired.append(self.path)
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(agent_module, "ProcessFileLock", RecordingLock)

    with agent_module.agent_lifecycle_lock():
        assert not config_root.exists()
        config_root.mkdir(parents=True)

    with agent_module.agent_lifecycle_lock():
        pass

    assert acquired == [
        agent_module._lifecycle_transition_lock_path(),
        initial_bootstrap,
        agent_module._lifecycle_transition_lock_path(),
        agent_module._lifecycle_bootstrap_lock_path(),
        agent_module.lifecycle_lock_path(),
    ]


@pytest.mark.parametrize("phase", ["prepared", "committed"])
def test_lifecycle_recovery_accepts_an_alternate_config_root_spelling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
) -> None:
    config_root = tmp_path / "home" / ".config" / "open-xquant"
    config_root.mkdir(parents=True)
    alias_parent = config_root.parent / "path-alias"
    alias_parent.mkdir()
    alias = alias_parent / ".." / config_root.name
    monkeypatch.setattr(agent_module, "config_dir", lambda: config_root)
    destination = agent_module.agent_config_path()
    destination.write_text("agent_profile: standalone-agent\n", encoding="utf-8")
    staged = tmp_path / "stage" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text("agent_profile: multi-agent\n", encoding="utf-8")

    committed = agent_module._commit_target_upgrade([(destination, staged)])
    backup = committed[0][1]
    assert backup is not None and backup.is_file()
    journal = json.loads(agent_module.lifecycle_transaction_path().read_text(encoding="utf-8"))
    operation = journal["operations"][0]
    assert operation["original_evidence"]["kind"] == "file"
    assert operation["replacement_evidence"]["kind"] == "file"
    if phase == "prepared":
        agent_module._mark_lifecycle_transaction_prepared()

    monkeypatch.setattr(agent_module, "config_dir", lambda: alias)
    assert agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    expected = "standalone-agent" if phase == "prepared" else "multi-agent"
    assert destination.read_text(encoding="utf-8") == f"agent_profile: {expected}\n"
    assert not backup.exists()
    assert not (config_root / "agent-lifecycle-transaction.json").exists()


def test_read_recovered_agent_profile_recovers_before_reading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.parent.mkdir(parents=True)
    destination.write_text(
        "schema_version: 1\nagent_profile: standalone-agent\n",
        encoding="utf-8",
    )
    staged = tmp_path / "stage" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text(
        "schema_version: 1\nagent_profile: multi-agent\n",
        encoding="utf-8",
    )
    agent_module._commit_target_upgrade([(destination, staged)])
    agent_module._mark_lifecycle_transaction_prepared()

    assert agent_module.read_recovered_agent_profile() == agent_module.AGENT_PROFILE_STANDALONE
    assert not agent_module.lifecycle_transaction_path().exists()
    assert "standalone-agent" in destination.read_text(encoding="utf-8")


def test_read_recovered_agent_profile_uses_default_and_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    assert agent_module.read_recovered_agent_profile() == agent_module.AGENT_PROFILE_MULTI

    path = agent_module.agent_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("schema_version: 1\nagent_profile: invalid\n", encoding="utf-8")
    with pytest.raises(click.ClickException, match="agent profile"):
        agent_module.read_recovered_agent_profile()

    path.write_text("- standalone-agent\n", encoding="utf-8")
    with pytest.raises(click.ClickException, match="agent profile"):
        agent_module.read_recovered_agent_profile()


def test_schema_two_lifecycle_journal_fails_closed_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    destination = agent_module.agent_config_path()
    destination.parent.mkdir(parents=True)
    destination.write_text("agent_profile: standalone-agent\n", encoding="utf-8")
    staged = tmp_path / "stage" / "agent.yaml"
    staged.parent.mkdir()
    staged.write_text("agent_profile: multi-agent\n", encoding="utf-8")
    committed = agent_module._commit_target_upgrade([(destination, staged)])
    backup = committed[0][1]
    assert backup is not None
    transaction = agent_module.lifecycle_transaction_path()
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    transaction.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(click.ClickException, match="unsupported schema_version"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert destination.read_text(encoding="utf-8") == "agent_profile: multi-agent\n"
    assert backup.read_text(encoding="utf-8") == "agent_profile: standalone-agent\n"
    assert transaction.is_file()
