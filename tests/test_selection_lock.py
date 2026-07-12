from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from oxq import run_digests
from oxq.run_digests import publish_run_artifacts, run_digest_transaction, update_artifact_hashes_and_run_digest
from oxq.selection_lock import (
    SelectionLockError,
    final_selection_lock_path,
    governing_workspace_root,
    hold_final_selection_lock,
)


def _write_standalone_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    content = b"baseline\n"
    (run_dir / "metrics.txt").write_bytes(content)
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps({"metrics.txt": run_digests._hash_bytes(content)}),
        encoding="utf-8",
    )
    update_artifact_hashes_and_run_digest(run_dir, lambda manifest: None)


def _write_workspace_config(workspace: Path, content: str) -> Path:
    config_path = workspace / ".open-xquant" / "workspace.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _governed_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "research_versions" / "v001" / "09_backtests" / "run-a"
    _write_standalone_run(run_dir)
    _write_workspace_config(
        workspace,
        "schema_version: 1\nworkflow:\n  layout: version_governed\npaths:\n  versions_dir: research_versions\n",
    )
    return workspace, run_dir


def _wait_for_path(path: Path, process: subprocess.Popen[bytes], timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"subprocess exited before readiness: {process.returncode}; "
                f"stdout={stdout.decode(errors='replace')!r}; stderr={stderr.decode(errors='replace')!r}"
            )
        time.sleep(0.01)
    process.kill()
    stdout, stderr = process.communicate()
    raise AssertionError(
        f"subprocess did not become ready; stdout={stdout.decode(errors='replace')!r}; stderr={stderr.decode(errors='replace')!r}"
    )


def _publication_process(run_dir: Path, ready_path: Path, artifact_name: str) -> subprocess.Popen[bytes]:
    script = """
import sys
from pathlib import Path
from oxq.run_digests import publish_run_artifacts

run_dir = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
artifact_name = sys.argv[3]
ready_path.write_text("ready\\n", encoding="utf-8")
publish_run_artifacts(run_dir, {artifact_name: b"published\\n"})
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(run_dir), str(ready_path), artifact_name],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_selection_lock_discovery_resolves_canonical_workspace_through_alias(tmp_path) -> None:
    workspace, run_dir = _governed_workspace(tmp_path)
    alias_parent = tmp_path / "aliases"
    alias_parent.mkdir()
    alias = alias_parent / "run-alias"
    alias.symlink_to(run_dir, target_is_directory=True)

    assert governing_workspace_root(alias) == workspace.resolve()
    assert final_selection_lock_path(alias) == workspace.resolve() / ".open-xquant/locks/final-selection.lock"


@pytest.mark.parametrize(
    "workspace_config",
    [
        None,
        "schema_version: 1\n",
        "schema_version: 1\nworkflow:\n  layout: legacy\npaths:\n  runs_dir: runs\n",
    ],
    ids=["no-workspace", "unclassified-workspace", "legacy-workspace"],
)
def test_selection_lock_discovery_skips_legitimate_standalone_runs(
    tmp_path,
    workspace_config: str | None,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / "run-a"
    _write_standalone_run(run_dir)
    if workspace_config is not None:
        _write_workspace_config(workspace, workspace_config)

    assert governing_workspace_root(run_dir) is None
    assert final_selection_lock_path(run_dir) is None


def test_selection_lock_discovery_uses_nearest_workspace_marker(tmp_path) -> None:
    outer = tmp_path / "outer"
    inner = outer / "nested"
    run_dir = inner / "runs" / "run-a"
    _write_standalone_run(run_dir)
    _write_workspace_config(outer, "workflow:\n  layout: version_governed\n")
    _write_workspace_config(inner, "schema_version: 1\n")

    assert final_selection_lock_path(run_dir) is None


@pytest.mark.parametrize(
    "workspace_config",
    [
        "workflow: [\n",
        "- not-a-mapping\n",
        "workflow: invalid\n",
        "paths: invalid\n",
        "paths:\n  versions_dir:\n",
        "paths:\n  versions_dir: ../outside\n",
        "workflow:\n  layout: version_governed\npaths:\n  versions_dir: /outside\n",
    ],
    ids=[
        "invalid-yaml",
        "non-mapping",
        "invalid-workflow",
        "invalid-paths",
        "empty-versions-dir",
        "escaping-versions-dir",
        "absolute-versions-dir",
    ],
)
def test_governed_workspace_discovery_fails_closed_before_publication_mutation(
    tmp_path,
    workspace_config: str,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "versions" / "v001" / "09_backtests" / "run-a"
    _write_standalone_run(run_dir)
    run_lock_path = run_dir.parent / "run_digests.jsonl.lock"
    run_lock_path.unlink()
    manifest_path = run_dir / "artifact_hashes.json"
    digest_path = run_dir.parent / "run_digests.jsonl"
    original_manifest = manifest_path.read_bytes()
    original_digest = digest_path.read_bytes()
    _write_workspace_config(workspace, workspace_config)

    with pytest.raises(SelectionLockError, match="workspace"):
        publish_run_artifacts(run_dir, {"blocked.txt": b"blocked\n"})

    assert manifest_path.read_bytes() == original_manifest
    assert digest_path.read_bytes() == original_digest
    assert not (run_dir / "blocked.txt").exists()
    assert not run_lock_path.exists()
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()
    assert not (workspace / ".open-xquant/locks").exists()


def test_runtime_transaction_acquires_run_lock_before_final_selection_lock(monkeypatch, tmp_path) -> None:
    workspace, run_dir = _governed_workspace(tmp_path)
    run_lock_path = (run_dir.parent / "run_digests.jsonl.lock").resolve(strict=False)
    selection_path = workspace.resolve() / ".open-xquant/locks/final-selection.lock"
    events: list[tuple[str, Path]] = []

    class RecordingLock:
        def __init__(self, path: str | Path) -> None:
            self.path = Path(path).resolve(strict=False)

        def __enter__(self):
            events.append(("enter", self.path))
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            events.append(("exit", self.path))

    monkeypatch.setattr(run_digests, "ProcessFileLock", RecordingLock)
    monkeypatch.setattr("oxq.selection_lock.ProcessFileLock", RecordingLock)

    with run_digest_transaction(run_dir):
        assert events == [("enter", run_lock_path), ("enter", selection_path)]

    assert events == [
        ("enter", run_lock_path),
        ("enter", selection_path),
        ("exit", selection_path),
        ("exit", run_lock_path),
    ]


def test_hold_final_selection_lock_rejects_replaced_locks_parent(tmp_path) -> None:
    workspace, run_dir = _governed_workspace(tmp_path)
    lock_path = final_selection_lock_path(run_dir)
    assert lock_path is not None
    attacker_dir = tmp_path / "attacker-locks"
    attacker_dir.mkdir()
    lock_path.parent.symlink_to(attacker_dir, target_is_directory=True)

    with pytest.raises(SelectionLockError, match="lock directory is unsafe"):
        with hold_final_selection_lock(lock_path):
            pass

    assert not (attacker_dir / lock_path.name).exists()


def test_hold_final_selection_lock_enforces_restrictive_permissions(tmp_path) -> None:
    _workspace, run_dir = _governed_workspace(tmp_path)
    lock_path = final_selection_lock_path(run_dir)
    assert lock_path is not None

    with hold_final_selection_lock(lock_path):
        assert lock_path.is_file()

    assert stat.S_IMODE(lock_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_process_holding_final_selection_lock_blocks_publication_until_release(tmp_path) -> None:
    workspace, run_dir = _governed_workspace(tmp_path)
    lock_path = final_selection_lock_path(run_dir)
    assert lock_path == workspace.resolve() / ".open-xquant/locks/final-selection.lock"
    ready_path = tmp_path / "publisher-ready"
    artifact_path = run_dir / "published.txt"

    with hold_final_selection_lock(lock_path):
        process = _publication_process(run_dir, ready_path, artifact_path.name)
        _wait_for_path(ready_path, process)
        time.sleep(0.1)
        assert process.poll() is None
        assert not artifact_path.exists()

    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    assert artifact_path.read_bytes() == b"published\n"


def test_process_holding_final_selection_lock_blocks_journal_recovery_until_release(tmp_path) -> None:
    workspace, run_dir = _governed_workspace(tmp_path)
    marker_path = run_dir / "recovery-marker.txt"
    marker_path.write_bytes(b"old\n")
    target = run_digests._journal_target("artifact", marker_path, b"recovered\n", name=marker_path.name)
    journal_path = run_dir.parent / "run_digests.jsonl.journal"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recovery": "commit",
                "run_id": run_dir.name,
                "targets": [target],
            }
        ),
        encoding="utf-8",
    )
    ready_path = tmp_path / "recovery-ready"
    script = """
import sys
from pathlib import Path
from oxq.run_digests import run_digest_transaction

run_dir = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
ready_path.write_text("ready\\n", encoding="utf-8")
with run_digest_transaction(run_dir):
    pass
"""
    lock_path = final_selection_lock_path(run_dir)
    assert lock_path == workspace.resolve() / ".open-xquant/locks/final-selection.lock"

    with hold_final_selection_lock(lock_path):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(run_dir), str(ready_path)],
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _wait_for_path(ready_path, process)
        time.sleep(0.1)
        assert process.poll() is None
        assert marker_path.read_bytes() == b"old\n"
        assert journal_path.exists()

    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    assert marker_path.read_bytes() == b"recovered\n"
    assert not journal_path.exists()
