from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from oxq.cli import agent as agent_module
from oxq.cli import research

_LOCK_PROCESS = r"""
import pathlib
import sys
import time

from oxq.cli import research
from oxq.process_lock import ProcessFileLock

workspace = pathlib.Path(sys.argv[1])
acquired = pathlib.Path(sys.argv[2])
release = pathlib.Path(sys.argv[3]) if len(sys.argv) == 4 else None

with ProcessFileLock(research._workspace_init_lock_path(workspace)):
    acquired.write_text("acquired\n", encoding="utf-8")
    if release is not None:
        deadline = time.monotonic() + 15
        while not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("release was not signaled")
            time.sleep(0.02)
"""


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
    pytest.fail(
        f"lock process did not acquire in time (exit={process.returncode}):\n"
        f"{stdout}{stderr}"
    )


def _assert_missing_workspace_aliases_serialize(
    first_workspace: Path,
    second_workspace: Path,
    tmp_path: Path,
) -> None:
    assert not first_workspace.exists()
    assert not second_workspace.exists()
    first_acquired = tmp_path / "first-acquired"
    second_acquired = tmp_path / "second-acquired"
    release = tmp_path / "release-first"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _LOCK_PROCESS,
            str(first_workspace),
            str(first_acquired),
            str(release),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    second: subprocess.Popen[str] | None = None
    try:
        _wait_for_process_path(first_acquired, first)
        second = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _LOCK_PROCESS,
                str(second_workspace),
                str(second_acquired),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        time.sleep(0.4)
        assert second.poll() is None, second.communicate(timeout=1)
        assert not second_acquired.exists()
        release.write_text("continue\n", encoding="utf-8")
        first_stdout, first_stderr = first.communicate(timeout=15)
        second_stdout, second_stderr = second.communicate(timeout=15)
    finally:
        release.touch()
        if first.poll() is None:
            first.kill()
            first.communicate()
        if second is not None and second.poll() is None:
            second.kill()
            second.communicate()

    assert first.returncode == 0, first_stdout + first_stderr
    assert second is not None
    assert second.returncode == 0, second_stdout + second_stderr
    assert second_acquired.is_file()


def test_missing_workspace_lock_serializes_case_aliases_when_supported(
    tmp_path: Path,
) -> None:
    canonical_parent = tmp_path / "CaseSensitiveProbe"
    canonical_parent.mkdir()
    alias_parent = tmp_path / "casesensitiveprobe"
    if not alias_parent.exists() or not canonical_parent.samefile(alias_parent):
        pytest.skip("the test filesystem is case-sensitive")

    _assert_missing_workspace_aliases_serialize(
        canonical_parent / "MissingWorkspace",
        alias_parent / "missingworkspace",
        tmp_path,
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="requires Linux bind mounts",
)
def test_missing_workspace_lock_serializes_bind_mount_aliases_when_supported(
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
        _assert_missing_workspace_aliases_serialize(
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


def test_research_init_recovers_agent_profile_before_workspace_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".config" / "open-xquant"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    config_path = agent_module.agent_config_path()
    config_path.write_text(
        "schema_version: 1\nagent_profile: standalone-agent\n",
        encoding="utf-8",
    )
    staged_config = tmp_path / "staged-agent.yaml"
    staged_config.write_text(
        "schema_version: 1\nagent_profile: multi-agent\n",
        encoding="utf-8",
    )
    committed = agent_module._commit_target_upgrade([(config_path, staged_config)])
    backup = committed[0][1]
    assert backup is not None
    journal = json.loads(
        agent_module.lifecycle_transaction_path().read_text(encoding="utf-8")
    )
    local_staged = Path(journal["operations"][0]["local_staged"])
    agent_module._mark_lifecycle_transaction_prepared()
    config_path.replace(local_staged)
    assert not config_path.exists()
    assert backup.is_file()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text(
        "<!-- open-xquant-subagents:begin -->\n"
        "stale policy\n"
        "<!-- open-xquant-subagents:end -->\n",
        encoding="utf-8",
    )
    original_recover = research._recover_governance_transaction

    def assert_agent_recovered_before_workspace_recovery(root: Path) -> None:
        assert config_path.is_file()
        assert "standalone-agent" in config_path.read_text(encoding="utf-8")
        assert not agent_module.lifecycle_transaction_path().exists()
        original_recover(root)

    monkeypatch.setattr(
        research,
        "_recover_governance_transaction",
        assert_agent_recovered_before_workspace_recovery,
    )

    research.initialize_workspace(workspace)

    agents_text = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "open-xquant-workspace:begin" in agents_text
    assert "open-xquant-subagents:begin" not in agents_text
    assert not backup.exists()
    assert not local_staged.exists()
    assert not agent_module.lifecycle_transaction_path().exists()
