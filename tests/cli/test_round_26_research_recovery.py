from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from oxq.cli import research

_INITIALIZER_WITH_CREATION_BARRIER = r"""
import pathlib
import sys
import time

from oxq.cli import research

workspace = pathlib.Path(sys.argv[1])
paused = pathlib.Path(sys.argv[2])
release = pathlib.Path(sys.argv[3])
original_write_workspace_config = research._write_workspace_config
did_pause = False

def pause_after_workspace_creation(cwd, path, payload):
    global did_pause
    if not did_pause and pathlib.Path(path).name == "workspace.yaml":
        did_pause = True
        if not workspace.is_dir():
            raise AssertionError("workspace was not created before the barrier")
        paused.write_text("created\n", encoding="utf-8")
        deadline = time.monotonic() + 20
        while not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("release was not signaled")
            time.sleep(0.02)
    return original_write_workspace_config(cwd, path, payload)

research._write_workspace_config = pause_after_workspace_creation
research.initialize_workspace(workspace, name="round-26-first")
"""


_SECOND_INITIALIZER = r"""
import pathlib
import sys

from oxq.cli import research

workspace = pathlib.Path(sys.argv[1])
started = pathlib.Path(sys.argv[2])
completed = pathlib.Path(sys.argv[3])
started.write_text("started\n", encoding="utf-8")
research.initialize_workspace(workspace, name="round-26-second")
completed.write_text("completed\n", encoding="utf-8")
"""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_evidenced_governance_journal(
    workspace: Path,
    *,
    state: str,
) -> tuple[Path, Path, Path, bytes]:
    transaction_id = "2" * 32
    destination = workspace / "versions/v001/phase_state.json"
    stage, backup = research._governance_transaction_artifacts(
        destination,
        transaction_id,
    )
    original = destination.read_bytes()
    replacement = b'{"transaction": "replacement"}\n'

    destination.replace(backup)
    stage.write_bytes(replacement)
    replacement_identity = research._governance_identity_payload(stage.stat())
    stage.replace(destination)

    journal_path = research._governance_transaction_path(workspace)
    journal = {
        "schema_version": research._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "state": state,
        "journal_parent_identity": research._governance_identity_payload(journal_path.parent.stat()),
        "entries": [
            {
                "destination": "versions/v001/phase_state.json",
                "had_original": True,
                "parent_identity": research._governance_identity_payload(destination.parent.stat()),
                "progress": "installed",
                "replacement_identity": replacement_identity,
                "replacement_sha256": _sha256(replacement),
                "original_identity": research._governance_identity_payload(backup.stat()),
                "original_sha256": _sha256(original),
            }
        ],
    }
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    return destination, backup, journal_path, replacement


def _assert_recovery_evidence_unchanged(
    backup: Path,
    journal_path: Path,
    *,
    backup_before: bytes,
    journal_before: bytes,
) -> None:
    assert backup.read_bytes() == backup_before
    assert journal_path.read_bytes() == journal_before


def test_committed_recovery_preserves_backup_and_journal_when_destination_is_missing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research.initialize_workspace(workspace, name="round-26-missing")
    destination, backup, journal_path, _replacement = _write_evidenced_governance_journal(workspace, state="committed")
    destination.unlink()
    backup_before = backup.read_bytes()
    journal_before = journal_path.read_bytes()

    with pytest.raises(Exception, match="missing destination"):
        research._recover_governance_transaction(workspace)

    assert not destination.exists()
    _assert_recovery_evidence_unchanged(
        backup,
        journal_path,
        backup_before=backup_before,
        journal_before=journal_before,
    )


def test_committed_recovery_preserves_same_content_external_replacement(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research.initialize_workspace(workspace, name="round-26-replaced")
    destination, backup, journal_path, replacement = _write_evidenced_governance_journal(workspace, state="committed")
    expected_identity = research._governance_identity_payload(destination.stat())
    external = destination.parent / ".external-phase-state"
    external.write_bytes(replacement)
    os.replace(external, destination)
    external_identity = research._governance_identity_payload(destination.stat())
    assert external_identity != expected_identity
    backup_before = backup.read_bytes()
    journal_before = journal_path.read_bytes()

    with pytest.raises(Exception, match="destination.*unrecognized"):
        research._recover_governance_transaction(workspace)

    assert destination.read_bytes() == replacement
    assert research._governance_identity_payload(destination.stat()) == external_identity
    _assert_recovery_evidence_unchanged(
        backup,
        journal_path,
        backup_before=backup_before,
        journal_before=journal_before,
    )


def test_committed_recovery_requires_installed_progress_before_cleanup(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research.initialize_workspace(workspace, name="round-26-progress")
    destination, backup, journal_path, replacement = _write_evidenced_governance_journal(workspace, state="committed")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["entries"][0]["progress"] = "staged"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    backup_before = backup.read_bytes()
    journal_before = journal_path.read_bytes()

    with pytest.raises(Exception, match="progress.*installed"):
        research._recover_governance_transaction(workspace)

    assert destination.read_bytes() == replacement
    _assert_recovery_evidence_unchanged(
        backup,
        journal_path,
        backup_before=backup_before,
        journal_before=journal_before,
    )


def _force_path_mutation_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == "nt":
        pytest.skip("the native Windows mutation backend already uses paths")
    original_init = research._GovernanceMutationParent.__init__

    def initialize_without_directory_descriptor(
        self: research._GovernanceMutationParent,
        cwd: Path,
        path: Path,
    ) -> None:
        original_init(self, cwd, path)
        assert self.descriptor is not None
        descriptor = self.descriptor
        self.descriptor = None
        os.close(descriptor)

    monkeypatch.setattr(
        research._GovernanceMutationParent,
        "__init__",
        initialize_without_directory_descriptor,
    )


@pytest.mark.parametrize(
    "race_kind",
    ["inode-replacement", "content-mutation"],
)
@pytest.mark.parametrize(
    "mutation_backend",
    ["native", "path"],
)
def test_conditional_unlink_quarantines_raced_artifact_and_preserves_recovery_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    race_kind: str,
    mutation_backend: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research.initialize_workspace(workspace, name="round-26-unlink-race")
    destination, backup, journal_path, _replacement = _write_evidenced_governance_journal(workspace, state="prepared")
    backup_before = backup.read_bytes()
    journal_before = journal_path.read_bytes()
    raced_content = b'{"external": "raced"}\n'
    raced_source = destination.parent / ".external-raced-phase-state"
    if race_kind == "inode-replacement":
        raced_source.write_bytes(raced_content)
    if mutation_backend == "path":
        _force_path_mutation_backend(monkeypatch)

    original_replace = os.replace
    original_unlink = os.unlink
    race_injected = False

    def inject_race(directory_descriptor: int | None) -> None:
        nonlocal race_injected
        assert not race_injected
        if race_kind == "inode-replacement":
            if directory_descriptor is None:
                original_replace(raced_source, destination)
            else:
                original_replace(
                    raced_source.name,
                    destination.name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                )
        else:
            destination.write_bytes(raced_content)
        race_injected = True

    def race_before_replace(
        source: object,
        target: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if not race_injected and Path(source).name == destination.name and ".quarantine-" in Path(target).name:
            inject_race(src_dir_fd)
        original_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def race_before_unlink(
        path: object,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if not race_injected and Path(path).name == destination.name:
            inject_race(dir_fd)
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(research.os, "replace", race_before_replace)
    monkeypatch.setattr(research.os, "unlink", race_before_unlink)

    with pytest.raises(Exception, match="changed|unrecognized"):
        research._recover_governance_transaction(workspace)

    assert race_injected
    _assert_recovery_evidence_unchanged(
        backup,
        journal_path,
        backup_before=backup_before,
        journal_before=journal_before,
    )
    quarantines = list(destination.parent.glob(f".{destination.name}.quarantine-*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == raced_content


def _wait_for_process_path(
    path: Path,
    process: subprocess.Popen[str],
    *,
    timeout: float = 10,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and process.poll() is None:
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    if path.exists():
        return
    stdout, stderr = process.communicate(timeout=1)
    pytest.fail(f"initializer did not reach {path.name} (exit={process.returncode}):\n{stdout}{stderr}")


def _assert_initializers_serialize_across_creation(
    first_workspace: Path,
    second_workspace: Path,
    tmp_path: Path,
) -> None:
    assert not first_workspace.exists()
    assert not second_workspace.exists()
    paused = tmp_path / "first-paused-after-creation"
    release = tmp_path / "release-first-initializer"
    second_started = tmp_path / "second-started"
    second_completed = tmp_path / "second-completed"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _INITIALIZER_WITH_CREATION_BARRIER,
            str(first_workspace),
            str(paused),
            str(release),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    second: subprocess.Popen[str] | None = None
    first_stdout = ""
    first_stderr = ""
    second_stdout = ""
    second_stderr = ""
    blocked_after_creation = False
    try:
        _wait_for_process_path(paused, first)
        assert first_workspace.is_dir()
        second = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _SECOND_INITIALIZER,
                str(second_workspace),
                str(second_started),
                str(second_completed),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        _wait_for_process_path(second_started, second)
        time.sleep(0.6)
        blocked_after_creation = second.poll() is None and not second_completed.exists()
        release.write_text("continue\n", encoding="utf-8")
        first_stdout, first_stderr = first.communicate(timeout=20)
        second_stdout, second_stderr = second.communicate(timeout=20)
    finally:
        release.touch()
        if first.poll() is None:
            first.kill()
            first_stdout, first_stderr = first.communicate()
        if second is not None and second.poll() is None:
            second.kill()
            second_stdout, second_stderr = second.communicate()

    assert blocked_after_creation, f"the second initializer crossed the missing-to-existing lock transition\n{second_stdout}{second_stderr}"
    assert first.returncode == 0, first_stdout + first_stderr
    assert second is not None
    assert second.returncode == 0, second_stdout + second_stderr
    assert second_completed.is_file()


def test_workspace_transition_lock_path_is_invariant_across_directory_creation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    before_creation = research._workspace_init_transition_lock_path(workspace)
    workspace.mkdir()
    after_creation = research._workspace_init_transition_lock_path(workspace)

    assert before_creation == after_creation


def test_workspace_transition_lock_serializes_initializer_after_directory_creation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    _assert_initializers_serialize_across_creation(workspace, workspace, tmp_path)


def test_workspace_transition_lock_serializes_case_alias_after_directory_creation(
    tmp_path: Path,
) -> None:
    canonical_parent = tmp_path / "CaseSensitiveProbe"
    canonical_parent.mkdir()
    alias_parent = tmp_path / "casesensitiveprobe"
    if not alias_parent.exists() or not canonical_parent.samefile(alias_parent):
        pytest.skip("the test filesystem is case-sensitive")

    _assert_initializers_serialize_across_creation(
        canonical_parent / "MissingWorkspace",
        alias_parent / "missingworkspace",
        tmp_path,
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="requires Linux bind mounts",
)
def test_workspace_transition_lock_serializes_bind_alias_after_directory_creation(
    tmp_path: Path,
) -> None:
    mount = shutil.which("mount")
    umount = shutil.which("umount")
    if mount is None or umount is None:
        pytest.skip("mount utilities are unavailable")
    canonical_parent = tmp_path / "canonical-parent"
    alias_parent = tmp_path / "bind-parent"
    canonical_parent.mkdir()
    alias_parent.mkdir()
    mounted = subprocess.run(
        [mount, "--bind", str(canonical_parent), str(alias_parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    if mounted.returncode != 0:
        pytest.skip(f"bind mounts are unavailable: {mounted.stderr.strip()}")
    try:
        _assert_initializers_serialize_across_creation(
            canonical_parent / "missing-workspace",
            alias_parent / "missing-workspace",
            tmp_path,
        )
    finally:
        unmounted = subprocess.run(
            [umount, str(alias_parent)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert unmounted.returncode == 0, unmounted.stderr
