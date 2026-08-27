"""Contained child-process controller tests."""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys

import pytest

from oxq.operators import child_process


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-S", "-c", "pass"],
        [sys.executable, "-I", "-c", "pass"],
    ],
    ids=["missing-isolated-mode", "missing-no-site"],
)
def test_rejects_child_command_without_required_python_startup_flags(
    command: list[str],
) -> None:
    """Removing either isolation flag must prevent a provider from starting."""
    with pytest.raises(AssertionError, match="-I and -S"):
        child_process.run_contained_child(
            command,
            timeout_seconds=5,
            response_secret=None,
            environment={},
        )


def test_discards_child_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing DEVNULL streams would expose provider output to the controller."""
    observed: dict[str, object] = {}
    popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        observed.setdefault("stdout", kwargs.get("stdout"))
        observed.setdefault("stderr", kwargs.get("stderr"))
        return popen(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(child_process.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(child_process, "_contained_posix_command", lambda command: command)

    assert (
        child_process.run_contained_child(
            [sys.executable, "-I", "-S", "-c", "pass"],
            timeout_seconds=5,
            response_secret=None,
            environment={},
        )
        == 0
    )
    assert observed == {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def test_child_environment_removes_import_paths_credentials_and_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaking these variables would let a provider escape its wheel boundary."""
    observed: dict[str, str] = {}

    class FakeProcess:
        pid = 123
        returncode = 0
        stdin = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

        def poll(self) -> int:
            return self.returncode

    def recording_popen(*args: object, **kwargs: object) -> FakeProcess:
        del args
        observed.update(kwargs["env"])  # type: ignore[arg-type]
        return FakeProcess()

    monkeypatch.setattr(child_process, "_platform_name", lambda: "nt")
    monkeypatch.setattr(child_process.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(child_process, "_open_windows_kill_on_close_job", lambda process: 1)
    monkeypatch.setattr(child_process, "_close_windows_job", lambda handle: None)

    assert (
        child_process.run_contained_child(
            [sys.executable, "-I", "-S", "-c", "pass"],
            timeout_seconds=5,
            response_secret=None,
            environment={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "/unsafe",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "HTTP_PROXY": "http://proxy.invalid",
                "HTTPS_PROXY": "https://proxy.invalid",
                "LANG": "C",
            },
        )
        == 0
    )
    assert observed["LANG"] == "C"
    assert "PYTHONPATH" not in observed
    assert "AWS_SECRET_ACCESS_KEY" not in observed
    assert "HTTP_PROXY" not in observed
    assert "HTTPS_PROXY" not in observed


def test_macos_process_containment_fails_closed_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(child_process, "_posix_platform", lambda: "darwin")
    monkeypatch.setattr(child_process.Path, "is_file", lambda path: True)
    monkeypatch.setattr(
        child_process.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1),
    )
    with pytest.raises(OSError, match="sandbox is unavailable"):
        child_process._contained_posix_command([sys.executable, "-I", "-S", "-c", "pass"])


def test_linux_supervisor_always_cleans_children_when_launch_is_interrupted() -> None:
    tree = ast.parse(child_process._LINUX_SUBREAPER_SCRIPT)
    supervision = next(
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "Popen" for child in ast.walk(node)
        )
    )
    assert any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "terminate_adopted_children"
        for statement in supervision.finalbody
        for child in ast.walk(statement)
    )


def test_linux_guardian_inserts_launcher_between_itself_and_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(child_process, "_posix_platform", lambda: "linux")
    provider = [sys.executable, "-I", "-S", "-c", "pass"]
    command = child_process._contained_posix_command(provider)
    assert command[5:10] == [sys.executable, "-I", "-S", "-c", child_process._LINUX_LAUNCHER_SCRIPT]
    assert command[10:] == provider


def test_success_closes_windows_kill_on_close_job(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, int]] = []

    class FakeProcess:
        pid = 123
        returncode = 0
        stdin = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            events.append(("wait", self.pid))
            return self.returncode

        def poll(self) -> int:
            return self.returncode

    monkeypatch.setattr(child_process, "_platform_name", lambda: "nt")
    monkeypatch.setattr(child_process.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(child_process, "_open_windows_kill_on_close_job", lambda process: events.append(("open", process.pid)) or 456)
    monkeypatch.setattr(child_process, "_close_windows_job", lambda handle: events.append(("close", handle)))
    assert (
        child_process.run_contained_child(
            [sys.executable, "-I", "-S", "-c", "pass"], timeout_seconds=5, response_secret=None, environment={}
        )
        == 0
    )
    assert events == [("open", 123), ("wait", 123), ("close", 456)]


def test_process_tree_cleanup_expands_collected_descendant_subtrees(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 123

        def poll(self) -> int:
            return -signal.SIGKILL

        def kill(self) -> None:
            raise AssertionError("reaped guardian must not be killed again")

    children = {123: set(), 200: {300}, 300: set()}
    killed: list[set[int]] = []
    monkeypatch.setattr(child_process, "_platform_name", lambda: "posix")
    monkeypatch.setattr(child_process, "_posix_descendant_pids", lambda root_pid: children[root_pid])
    monkeypatch.setattr(child_process.os, "killpg", lambda *args: None)
    monkeypatch.setattr(child_process.os, "kill", lambda *args: None)
    monkeypatch.setattr(child_process, "_kill_posix_processes", lambda process_ids: killed.append(set(process_ids)))
    child_process._kill_process_tree(FakeProcess(), {200})  # type: ignore[arg-type]
    assert killed == [{200, 300}]
