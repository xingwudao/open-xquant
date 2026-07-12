from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from oxq.cli import agent as agent_module
from oxq.cli.main import main


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory descriptors")
def test_commit_parent_swap_before_first_rename_preserves_unrelated_data_and_journal(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    destination_parent = home / ".cursor/skills"
    destination = destination_parent / "build-strategy-spec"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("old skill\n", encoding="utf-8")
    staging_root = tmp_path / "system-stage"
    staged_skill = staging_root / "build-strategy-spec"
    staged_skill.mkdir(parents=True)
    (staged_skill / "SKILL.md").write_text("new skill\n", encoding="utf-8")
    staged_manifest = staging_root / "agent-install.json"
    monkeypatch.setenv("HOME", str(home))
    manifest = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(destination_parent),
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
    agent_module.write_json_file(staged_manifest, manifest)
    displaced_parent = tmp_path / "displaced-skills"
    external_parent = tmp_path / "external-skills"
    original_replace = os.replace
    swapped = False

    def swap_before_replace(source, target, *args, **kwargs) -> None:
        nonlocal swapped
        if not swapped and Path(source).name == destination.name and Path(target).name.startswith(f".{destination.name}.backup-"):
            swapped = True
            original_replace(destination_parent, displaced_parent)
            destination_parent.mkdir(parents=True)
            unrelated = destination_parent / destination.name
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("keep me\n", encoding="utf-8")
        original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(agent_module.os, "replace", swap_before_replace)

    with pytest.raises(Exception, match="identity changed|changed after validation"):
        agent_module._commit_target_upgrade(
            [
                (destination, staged_skill),
                (agent_module.manifest_path(), staged_manifest),
            ]
        )

    assert swapped
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep me\n"
    transaction = agent_module.lifecycle_transaction_path()
    assert transaction.is_file()

    original_replace(destination_parent, external_parent)
    original_replace(displaced_parent, destination_parent)
    monkeypatch.setattr(agent_module.os, "replace", original_replace)
    assert agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert (destination / "SKILL.md").read_text(encoding="utf-8") == "old skill\n"
    assert (external_parent / destination.name / "keep.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not transaction.exists()
    assert not list(destination_parent.glob(f".{destination.name}.install-*"))


def test_commit_copies_system_temp_stage_before_atomic_install_on_exdev(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    destination = home / ".cursor/skills/build-strategy-spec"
    destination.parent.mkdir(parents=True)
    staged = tmp_path / "other-mount/system-stage/build-strategy-spec"
    staged.mkdir(parents=True)
    (staged / "SKILL.md").write_text("new skill\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    original_replace = os.replace

    def reject_cross_directory_replace(source, target, *args, **kwargs) -> None:
        if not kwargs.get("src_dir_fd") and not kwargs.get("dst_dir_fd"):
            source_path = Path(source)
            target_path = Path(target)
            if source_path.is_absolute() and target_path.is_absolute():
                if source_path.parent != target_path.parent:
                    raise OSError(errno.EXDEV, "injected cross-device rename")
        original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(agent_module.os, "replace", reject_cross_directory_replace)

    committed = agent_module._commit_target_upgrade([(destination, staged)])
    agent_module._finish_committed_lifecycle_transaction(committed)

    assert (destination / "SKILL.md").read_text(encoding="utf-8") == "new skill\n"
    assert (staged / "SKILL.md").read_text(encoding="utf-8") == "new skill\n"
    assert not list(destination.parent.glob(f".{destination.name}.install-*"))
    assert not agent_module.lifecycle_transaction_path().exists()


def test_recovery_of_incomplete_local_staging_without_evidence_fails_closed(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    destination = home / ".cursor/skills/build-strategy-spec"
    destination.mkdir(parents=True)
    (destination / "keep.txt").write_text("keep me\n", encoding="utf-8")
    local_staged = destination.parent / f".{destination.name}.install-deadbeef"
    local_staged.mkdir()
    (local_staged / "partial.txt").write_text("partial\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    manifest = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(destination.parent),
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
                        "staged": str(tmp_path / "system-stage"),
                        "local_staged": str(local_staged),
                        "backup": None,
                        "had_destination": False,
                        "relative_name": destination.name,
                        "parent_identity": agent_module._path_parent_identity(destination),
                        "original_evidence": {"kind": "absent"},
                        "replacement_evidence": None,
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

    with pytest.raises(Exception, match="no recorded ownership evidence"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep me\n"
    assert (local_staged / "partial.txt").read_text(encoding="utf-8") == "partial\n"
    assert transaction.is_file()


CHILD_SCRIPT = r"""
import hashlib
import json
import os
from pathlib import Path

from oxq.cli import agent as agent_module
from oxq.cli.main import main


def fake_bundle(source_root, config_root, *, dry_run=False):
    bundle_root = config_root / "sdk-bundles/bundle-test"
    runner = bundle_root / "runner/.venv/bin/oxq"
    python = bundle_root / "runner/.venv/bin/python"
    wheel = bundle_root / "dist/open_xquant-0.1.0-py3-none-any.whl"
    lock = bundle_root / "requirements.lock.txt"
    packages = bundle_root / "packages.json"
    if not dry_run:
        runner.parent.mkdir(parents=True, exist_ok=True)
        wheel.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        wheel.write_text("wheel", encoding="utf-8")
        lock.write_text("lock\n", encoding="utf-8")
        packages.write_text("[]\n", encoding="utf-8")
    return {
        "id": "bundle-test",
        "root": str(bundle_root),
        "wheel": {"path": str(wheel), "sha256": hashlib.sha256(b"wheel").hexdigest()},
        "dependencies": {
            "lock_file": str(lock),
            "lock_sha256": hashlib.sha256(b"lock\n").hexdigest(),
            "packages_file": str(packages),
        },
        "runner": {
            "venv": str(bundle_root / "runner/.venv"),
            "python": str(python),
            "oxq": str(runner),
            "argv": [str(runner)],
        },
    }


agent_module.build_sdk_bundle = fake_bundle
mutation_type = (
    agent_module._WindowsRecoveryMutations
    if os.name == "nt"
    else agent_module._PosixRecoveryMutations
)
original_replace = mutation_type.replace
event_log = Path(os.environ["OXQ_EVENT_LOG"])
crash_index = int(os.environ.get("OXQ_CRASH_INDEX", "-1"))
event_index = 0


def observed_replace(self, source_path, destination_path):
    global event_index
    result = original_replace(self, source_path, destination_path)
    transaction = agent_module.lifecycle_transaction_path()
    if transaction.is_file():
        payload = json.loads(transaction.read_text(encoding="utf-8"))
        source = str(source_path)
        destination = str(destination_path)
        event = None
        for operation in payload["operations"]:
            if source == operation.get("local_staged") and destination == operation["destination"]:
                event = f"install:{Path(destination).name}"
                break
            if source == operation["destination"] and destination == operation.get("backup"):
                kind = "backup" if operation.get("staged") is not None else "removal"
                event = f"{kind}:{Path(source).name}"
                break
        if event is not None:
            with event_log.open("a", encoding="utf-8") as stream:
                stream.write(event + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if event_index == crash_index:
                os._exit(91)
            event_index += 1
    return result


mutation_type.replace = observed_replace
main.main(
    args=json.loads(os.environ["OXQ_ARGS"]),
    prog_name="oxq",
    standalone_mode=False,
)
"""


def _write_source(root: Path, body: str) -> None:
    skill = root / "agent/skills/build-strategy-spec"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: build-strategy-spec\ndescription: Test skill\n---\n\n" + body + "\n",
        encoding="utf-8",
    )


def _snapshot(root: Path, *, exclude_events: bool = True) -> dict[str, tuple[str, bytes]]:
    if not root.exists():
        return {}
    result: dict[str, tuple[str, bytes]] = {}
    for path in sorted(root.rglob("*")):
        if exclude_events and path.name == "events.log":
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = ("symlink", os.fsencode(path.readlink()))
        elif path.is_file():
            result[relative] = ("file", path.read_bytes())
    return result


def _run_child(
    repo: Path,
    home: Path,
    args: list[str],
    event_log: Path,
    *,
    crash_index: int = -1,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "CODEX_HOME": str(home / ".codex"),
            "OXQ_ARGS": json.dumps(args),
            "OXQ_CRASH_INDEX": str(crash_index),
            "OXQ_EVENT_LOG": str(event_log),
            "PYTHONPATH": str(repo / "src"),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _recover_with_status(repo: Path, home: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "CODEX_HOME": str(home / ".codex"),
            "PYTHONPATH": str(repo / "src"),
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from oxq.cli.main import main; main()",
            "agent",
            "status",
            "--json",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("command", ["install", "upgrade", "uninstall"])
def test_lifecycle_recovers_after_process_exit_at_every_commit_boundary(tmp_path, command: str) -> None:
    repo = Path.cwd()
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    install_args = [
        "agent",
        "install",
        "--target",
        "codex",
        "--profile",
        "standalone-agent",
        "--from-local",
        str(source_v1),
        "--yes",
    ]
    command_args = {
        "install": install_args,
        "upgrade": [
            "agent",
            "upgrade",
            "--target",
            "codex",
            "--from-local",
            str(source_v2),
            "--yes",
        ],
        "uninstall": ["agent", "uninstall", "--target", "codex", "--yes"],
    }[command]

    probe_home = tmp_path / f"probe-{command}"
    probe_events = tmp_path / f"probe-{command}-events.log"
    if command == "install":
        probe_agents = probe_home / ".codex/AGENTS.md"
        probe_agents.parent.mkdir(parents=True)
        probe_agents.write_text("user instructions\n", encoding="utf-8")
    else:
        setup = _run_child(repo, probe_home, install_args, tmp_path / f"setup-{command}.log")
        assert setup.returncode == 0, setup.stderr
    probe = _run_child(repo, probe_home, command_args, probe_events)
    assert probe.returncode == 0, probe.stderr
    events = probe_events.read_text(encoding="utf-8").splitlines()
    assert events
    assert any(event.startswith("backup:") for event in events)
    assert any(event.startswith("install:") for event in events)
    if command == "uninstall":
        assert any(event.startswith("removal:") for event in events)

    for crash_index, event in enumerate(events):
        home = tmp_path / f"{command}-{crash_index}"
        event_log = home / "events.log"
        if command == "install":
            agents = home / ".codex/AGENTS.md"
            agents.parent.mkdir(parents=True)
            agents.write_text("user instructions\n", encoding="utf-8")
        else:
            setup = _run_child(repo, home, install_args, tmp_path / f"setup-{command}-{crash_index}.log")
            assert setup.returncode == 0, setup.stderr
        before = _snapshot(home)

        crashed = _run_child(repo, home, command_args, event_log, crash_index=crash_index)
        assert crashed.returncode == 91, (event, crashed.stdout, crashed.stderr)
        recovered = _recover_with_status(repo, home)

        assert recovered.returncode == 0, (event, recovered.stdout, recovered.stderr)
        json.loads(recovered.stdout)
        assert _snapshot(home) == before, event
        assert not (home / ".config/open-xquant/agent-lifecycle-transaction.json").exists()
        assert not any(".backup-" in path.name for path in home.rglob("*"))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_commit_rolls_back_base_exception(monkeypatch, tmp_path, interruption: BaseException) -> None:
    home = tmp_path / "home"
    destination = home / ".codex/AGENTS.md"
    staged = tmp_path / "staged-AGENTS.md"
    destination.parent.mkdir(parents=True)
    destination.write_text("old instructions\n", encoding="utf-8")
    staged.write_text("new instructions\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    mutation_type = agent_module._WindowsRecoveryMutations if os.name == "nt" else agent_module._PosixRecoveryMutations
    original_replace = mutation_type.replace

    def interrupt_after_install(self, source: Path, target: Path) -> None:
        original_replace(self, source, target)
        if source.name.startswith(f".{destination.name}.install-") and target == destination:
            raise interruption

    monkeypatch.setattr(mutation_type, "replace", interrupt_after_install)

    with pytest.raises(type(interruption)):
        agent_module._commit_target_upgrade([(destination, staged)])

    assert destination.read_text(encoding="utf-8") == "old instructions\n"
    assert not agent_module.lifecycle_transaction_path().exists()
    assert not any(".backup-" in path.name for path in home.rglob("*"))


def test_fresh_install_status_recovers_after_commit_and_rollback_failures(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    staging_root = tmp_path / "staging"
    destination = home / ".cursor/skills/build-strategy-spec"
    staged_skill = staging_root / "build-strategy-spec"
    staged_manifest = staging_root / "agent-install.json"
    staged_skill.mkdir(parents=True)
    (staged_skill / "SKILL.md").write_text("new skill\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    manifest = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(destination.parent),
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
    }
    agent_module.write_json_file(staged_manifest, manifest)
    mutation_type = agent_module._WindowsRecoveryMutations if os.name == "nt" else agent_module._PosixRecoveryMutations
    original_replace = mutation_type.replace
    original_rollback = agent_module._rollback_lifecycle_operations

    def fail_manifest_commit(self, source: Path, target: Path) -> None:
        if target == agent_module.manifest_path():
            raise OSError("injected manifest commit failure")
        original_replace(self, source, target)

    def fail_rollback(_operations: list[dict]) -> None:
        raise OSError("injected rollback failure")

    monkeypatch.setattr(mutation_type, "replace", fail_manifest_commit)
    monkeypatch.setattr(agent_module, "_rollback_lifecycle_operations", fail_rollback)

    with pytest.raises(OSError, match="manifest commit failure"):
        agent_module._commit_target_upgrade(
            [
                (destination, staged_skill),
                (agent_module.manifest_path(), staged_manifest),
            ]
        )

    assert destination.is_dir()
    assert agent_module.lifecycle_transaction_path().is_file()
    shutil.rmtree(staging_root)
    monkeypatch.setattr(mutation_type, "replace", original_replace)
    monkeypatch.setattr(agent_module, "_rollback_lifecycle_operations", original_rollback)

    status = agent_module._status_payload()

    assert status["targets"] == {}
    assert not destination.exists()
    assert not agent_module.lifecycle_transaction_path().exists()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory descriptors")
def test_commit_rejects_destination_parent_replacement_before_sync(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    destination_parent = home / ".cursor/skills"
    displaced_parent = tmp_path / "displaced-skills"
    staged = tmp_path / "staged-skill"
    destination = destination_parent / "build-strategy-spec"
    destination_parent.mkdir(parents=True)
    staged.write_text("new skill\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    mutation_type = agent_module._WindowsRecoveryMutations if os.name == "nt" else agent_module._PosixRecoveryMutations
    original_replace = mutation_type.replace

    def replace_then_swap(self, source: Path, target: Path) -> None:
        original_replace(self, source, target)
        if source.name.startswith(f".{destination.name}.install-") and target == destination:
            destination_parent.replace(displaced_parent)
            destination_parent.mkdir(parents=True)

    monkeypatch.setattr(mutation_type, "replace", replace_then_swap)

    with pytest.raises(Exception, match="identity changed|changed after validation"):
        agent_module._commit_target_upgrade([(destination, staged)])

    assert (displaced_parent / destination.name).read_text(encoding="utf-8") == "new skill\n"
    assert not destination.exists()
    assert agent_module.lifecycle_transaction_path().exists()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory descriptors")
def test_recovery_rollback_rejects_parent_replacement_before_sync(tmp_path) -> None:
    parent = tmp_path / "skills"
    displaced_parent = tmp_path / "displaced-skills"
    destination = parent / "skill"
    backup = parent / ".skill.backup-deadbeef"
    parent.mkdir()
    destination.write_text("new\n", encoding="utf-8")
    backup.write_text("old\n", encoding="utf-8")
    operation = {
        "destination": str(destination),
        "staged": str(tmp_path / "consumed-stage"),
        "backup": str(backup),
        "had_destination": True,
    }

    with agent_module._secure_recovery_mutations([destination, backup]) as mutations:
        operation.update(
            {
                "original_evidence": mutations.path_evidence(backup),
                "replacement_evidence": mutations.path_evidence(destination),
            }
        )
        original_replace = mutations.replace

        def replace_then_swap(source: Path, target: Path) -> None:
            original_replace(source, target)
            parent.replace(displaced_parent)
            parent.mkdir()

        mutations.replace = replace_then_swap  # type: ignore[method-assign]
        with pytest.raises(Exception, match="identity changed|changed after validation"):
            agent_module._secure_rollback_lifecycle_operations([operation], mutations)

    assert (displaced_parent / destination.name).read_text(encoding="utf-8") == "old\n"


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory descriptors")
@pytest.mark.parametrize("artifact", ["backup", "journal"])
def test_recovery_cleanup_rejects_parent_replacement_before_sync(
    monkeypatch,
    tmp_path,
    artifact: str,
) -> None:
    parent = tmp_path / artifact
    displaced_parent = tmp_path / f"displaced-{artifact}"
    parent.mkdir()
    path = parent / (".skill.backup-deadbeef" if artifact == "backup" else "transaction.json")
    path.write_text("retained\n", encoding="utf-8")
    original_unlink = os.unlink
    swapped = False

    def unlink_then_swap(target, *args, **kwargs) -> None:
        nonlocal swapped
        original_unlink(target, *args, **kwargs)
        if not swapped and Path(target).name == path.name:
            swapped = True
            parent.replace(displaced_parent)
            parent.mkdir()

    monkeypatch.setattr(agent_module.os, "unlink", unlink_then_swap)
    with agent_module._secure_recovery_mutations([path]) as mutations:
        with pytest.raises(Exception, match="identity changed|changed after validation"):
            if artifact == "backup":
                agent_module._secure_cleanup_upgrade_backups(
                    [(parent / "skill", path)],
                    mutations,
                )
            else:
                agent_module._secure_remove_transaction_metadata(path, mutations)

    assert swapped
    assert not (displaced_parent / path.name).exists()


def _write_pending_recovery_journal(
    home: Path,
    *,
    phase: str,
) -> tuple[Path, Path, Path, Path]:
    config_root = home / ".config/open-xquant"
    skills_root = home / ".cursor/skills"
    destination = skills_root / "build-strategy-spec"
    backup = skills_root / ".build-strategy-spec.backup-deadbeef"
    manifest = config_root / "agent-install.json"
    transaction = config_root / "agent-lifecycle-transaction.json"
    destination.mkdir(parents=True)
    (destination / "managed.txt").write_text("managed\n", encoding="utf-8")
    config_root.mkdir(parents=True)
    manifest_payload = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(skills_root),
                "agents_dir": None,
                "instruction_file": None,
                "config_file": None,
                "installed_paths": [str(destination)],
                "agent_roles": [],
                "managed_blocks": [],
            }
        },
        "sdk_bundles": [],
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    agent_module._write_lifecycle_manifest_witness(manifest_payload)
    target_operation = {
        "destination": str(destination),
        "staged": str(home / "consumed-stage"),
        "backup": None,
        "had_destination": False,
        "relative_name": destination.name,
        "parent_identity": agent_module._path_parent_identity(destination),
        "original_evidence": {"kind": "absent"},
        "replacement_evidence": None,
    }
    if phase == "committed":
        backup.mkdir()
        (backup / "managed.txt").write_text("backup\n", encoding="utf-8")
        target_operation.update({"backup": str(backup), "had_destination": True})
    with agent_module._secure_recovery_mutations([destination, backup]) as mutations:
        target_operation["replacement_evidence"] = mutations.path_evidence(destination)
        if phase == "committed":
            target_operation["original_evidence"] = mutations.path_evidence(backup)
    transaction.write_text(
        json.dumps(
            {
                "schema_version": agent_module.LIFECYCLE_TRANSACTION_SCHEMA_VERSION,
                "transaction_type": "agent-lifecycle",
                "phase": phase,
                "created_at": "2026-01-01T00:00:00Z",
                "journal_parent_identity": agent_module._path_parent_identity(transaction),
                "operations": [
                    target_operation,
                    {
                        "destination": str(manifest),
                        "staged": None,
                        "backup": None,
                        "had_destination": False,
                        "relative_name": manifest.name,
                        "parent_identity": agent_module._path_parent_identity(manifest),
                        "original_evidence": {"kind": "absent"},
                        "replacement_evidence": None,
                    },
                ],
                "created_parents": [],
                "rollback_cleanup_paths": [],
                "trusted_roots": agent_module._lifecycle_trusted_root_identities(
                    agent_module._purge_trusted_roots(manifest_payload)
                ),
            }
        ),
        encoding="utf-8",
    )
    return skills_root, destination, backup, transaction


@pytest.mark.parametrize("phase", ["prepared", "committed"])
def test_recovery_preserves_journal_and_external_data_when_parent_swaps_after_validation(
    monkeypatch,
    tmp_path,
    phase: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    skills_root, destination, backup, transaction = _write_pending_recovery_journal(
        home,
        phase=phase,
    )
    displaced_root = tmp_path / "displaced-skills"
    external_root = tmp_path / "external-skills"
    external_target = external_root / destination.name if phase == "prepared" else external_root / backup.name
    external_target.mkdir(parents=True)
    external_sentinel = external_target / "keep.txt"
    external_sentinel.write_text("keep me\n", encoding="utf-8")
    original_validate = agent_module._validated_pending_lifecycle_transaction

    def validate_then_swap(path: Path) -> dict:
        payload = original_validate(path)
        skills_root.replace(displaced_root)
        skills_root.symlink_to(external_root, target_is_directory=True)
        return payload

    monkeypatch.setattr(
        agent_module,
        "_validated_pending_lifecycle_transaction",
        validate_then_swap,
    )

    with pytest.raises(Exception, match="changed|reparse|symlink|secure recovery"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert external_sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert transaction.exists()
    assert (displaced_root / destination.name / "managed.txt").exists()
    if phase == "committed":
        assert (displaced_root / backup.name / "managed.txt").exists()


@pytest.mark.parametrize("damage", ["missing", "altered"])
def test_recovery_preserves_journal_when_manifest_witness_is_unavailable_or_altered(
    monkeypatch,
    tmp_path,
    damage: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    _skills_root, destination, _backup, transaction = _write_pending_recovery_journal(
        home,
        phase="prepared",
    )
    witness = agent_module.lifecycle_manifest_witness_path()
    if damage == "missing":
        witness.unlink()
    else:
        witness.write_text("{}\n", encoding="utf-8")

    with pytest.raises(Exception, match="manifest witness"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert transaction.is_file()
    assert (destination / "managed.txt").read_text(encoding="utf-8") == "managed\n"


def test_purge_recovery_preserves_sidecar_and_external_data_after_validation_swap(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    skills_root = home / ".cursor/skills"
    destination = skills_root / "build-strategy-spec"
    backup = skills_root / ".build-strategy-spec.backup-round19"
    backup.mkdir(parents=True)
    (backup / "managed.txt").write_text("backup\n", encoding="utf-8")
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    transaction.parent.mkdir(parents=True)
    transaction.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_type": "agent-uninstall-purge",
                "phase": "committed",
                "created_at": "2026-01-01T00:00:00Z",
                "backups": [{"destination": str(destination), "backup": str(backup)}],
            }
        ),
        encoding="utf-8",
    )
    displaced_root = tmp_path / "displaced-skills"
    external_root = tmp_path / "external-skills"
    external_backup = external_root / backup.name
    external_backup.mkdir(parents=True)
    external_sentinel = external_backup / "keep.txt"
    external_sentinel.write_text("keep me\n", encoding="utf-8")
    original_validate = agent_module._validated_pending_purge_cleanup

    def validate_then_swap(path: Path) -> list[tuple[Path, Path | None]]:
        committed = original_validate(path)
        skills_root.replace(displaced_root)
        skills_root.symlink_to(external_root, target_is_directory=True)
        return committed

    monkeypatch.setattr(agent_module, "_validated_pending_purge_cleanup", validate_then_swap)

    with pytest.raises(Exception, match="changed|reparse|symlink|secure recovery"):
        agent_module._recover_pending_purge_cleanup(dry_run=False)

    assert external_sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert transaction.exists()
    assert (displaced_root / backup.name / "managed.txt").exists()


def test_fsync_directory_does_not_open_directories_on_windows(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(agent_module, "_is_windows", lambda: True, raising=False)

    def reject_directory_open(*_args, **_kwargs):
        raise OSError("Windows does not support opening this directory with os.open")

    monkeypatch.setattr(agent_module.os, "open", reject_directory_open)

    agent_module._fsync_directory(tmp_path)


def _fake_windows_bundle(source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict:
    del source_root
    bundle_root = config_root / "sdk-bundles/bundle-windows"
    runner = bundle_root / "runner/.venv/bin/oxq"
    if not dry_run:
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("runner\n", encoding="utf-8")
    return {
        "id": "bundle-windows",
        "root": str(bundle_root),
        "runner": {"oxq": str(runner), "argv": [str(runner)]},
    }


def test_windows_directory_fsync_behavior_covers_all_lifecycle_flows(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(agent_module, "build_sdk_bundle", _fake_windows_bundle)
    windows_checks = 0

    def windows() -> bool:
        nonlocal windows_checks
        windows_checks += 1
        return True

    monkeypatch.setattr(agent_module, "_is_windows", windows, raising=False)
    commands = [
        [
            "agent",
            "install",
            "--target",
            "cursor",
            "--profile",
            "standalone-agent",
            "--from-local",
            str(source_v1),
            "--yes",
        ],
        [
            "agent",
            "upgrade",
            "--target",
            "cursor",
            "--from-local",
            str(source_v2),
            "--yes",
        ],
        ["agent", "uninstall", "--target", "cursor", "--yes"],
    ]

    for command in commands:
        checks_before = windows_checks
        result = CliRunner().invoke(main, command)
        assert result.exit_code == 0, result.output
        assert windows_checks > checks_before


def test_uninstall_base_exception_during_second_backup_cleanup_stays_committed(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    source = tmp_path / "source"
    _write_source(source, "installed workflow")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(agent_module, "build_sdk_bundle", _fake_windows_bundle)
    install = CliRunner().invoke(
        main,
        [
            "agent",
            "install",
            "--target",
            "cursor",
            "--profile",
            "standalone-agent",
            "--from-local",
            str(source),
            "--yes",
        ],
    )
    assert install.exit_code == 0, install.output
    installed_skill = home / ".cursor/skills/build-strategy-spec"
    transaction = home / ".config/open-xquant/agent-lifecycle-transaction.json"
    manifest_path = home / ".config/open-xquant/agent-install.json"
    original_remove = agent_module._remove_upgrade_path
    deleted_backups: list[Path] = []

    def interrupt_after_second_backup(path: Path) -> None:
        original_remove(path)
        if ".backup-" in path.name:
            deleted_backups.append(path)
            if len(deleted_backups) == 2:
                raise KeyboardInterrupt

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", interrupt_after_second_backup)

    with pytest.raises(KeyboardInterrupt):
        agent_module.uninstall.callback(
            target="cursor",
            all_targets=False,
            dry_run=False,
            purge_config=False,
            yes=True,
        )

    assert len(deleted_backups) == 2
    assert transaction.is_file()
    assert json.loads(transaction.read_text(encoding="utf-8"))["phase"] == "committed"
    committed_manifest = manifest_path.read_bytes()
    assert json.loads(committed_manifest)["targets"]["cursor"]["installed"] is False
    assert not installed_skill.exists()

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", original_remove)
    status = CliRunner().invoke(main, ["agent", "status", "--json"])

    assert status.exit_code == 0, status.output
    json.loads(status.output)
    assert manifest_path.read_bytes() == committed_manifest
    assert not installed_skill.exists()
    assert not transaction.exists()
    assert not agent_module.lifecycle_manifest_witness_path().exists()
    assert not agent_module.lifecycle_manifest_witness_digest_path().exists()
    assert not any(".backup-" in path.name for path in home.rglob("*"))


def test_fresh_install_recovery_rejects_forged_roots_after_staging_cleanup(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    staging_root = tmp_path / "staging"
    destination = home / ".cursor/skills/build-strategy-spec"
    staged_skill = staging_root / "build-strategy-spec"
    staged_manifest = staging_root / "agent-install.json"
    staged_skill.mkdir(parents=True)
    (staged_skill / "SKILL.md").write_text("new skill\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    manifest = {
        "schema_version": 1,
        "targets": {
            "cursor": {
                "installed": True,
                "skills_dir": str(destination.parent),
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
    }
    agent_module.write_json_file(staged_manifest, manifest)
    mutation_type = agent_module._WindowsRecoveryMutations if os.name == "nt" else agent_module._PosixRecoveryMutations
    original_replace = mutation_type.replace
    original_rollback = agent_module._rollback_lifecycle_operations

    def fail_manifest_commit(self, source: Path, target: Path) -> None:
        if target == agent_module.manifest_path():
            raise OSError("injected manifest commit failure")
        original_replace(self, source, target)

    def fail_rollback(_operations: list[dict]) -> None:
        raise OSError("injected rollback failure")

    monkeypatch.setattr(mutation_type, "replace", fail_manifest_commit)
    monkeypatch.setattr(agent_module, "_rollback_lifecycle_operations", fail_rollback)
    with pytest.raises(OSError, match="manifest commit failure"):
        agent_module._commit_target_upgrade(
            [
                (destination, staged_skill),
                (agent_module.manifest_path(), staged_manifest),
            ]
        )
    transaction = agent_module.lifecycle_transaction_path()
    original_journal = transaction.read_bytes()
    shutil.rmtree(staging_root)
    monkeypatch.setattr(mutation_type, "replace", original_replace)
    monkeypatch.setattr(agent_module, "_rollback_lifecycle_operations", original_rollback)

    external_root = tmp_path / "unrelated"
    external_root.mkdir()
    external_file = external_root / "notes"
    external_file.write_text("keep me\n", encoding="utf-8")
    forged = json.loads(original_journal)
    forged["operations"][0].update(
        {
            "destination": str(external_file),
            "staged": str(tmp_path / "consumed-external-stage"),
            "local_staged": str(external_root / ".notes.install-deadbeef"),
            "backup": None,
            "had_destination": False,
            "relative_name": external_file.name,
            "parent_identity": agent_module._path_parent_identity(external_file),
        }
    )
    forged["trusted_roots"]["targets"]["cursor"]["skills_dir"] = str(external_root)
    forged["created_parents"] = []
    transaction.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(Exception, match="witness|authoritative|trusted roots"):
        agent_module._recover_pending_lifecycle_transaction(dry_run=False)

    assert external_file.read_text(encoding="utf-8") == "keep me\n"
    assert transaction.is_file()
    transaction.write_bytes(original_journal)

    status = agent_module._status_payload()

    assert status["targets"] == {}
    assert not destination.exists()
    assert not transaction.exists()
    assert not agent_module.lifecycle_manifest_witness_path().exists()
    assert not agent_module.lifecycle_manifest_witness_digest_path().exists()
