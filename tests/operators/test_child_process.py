"""Contained child-process controller tests."""

from __future__ import annotations

import ast
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from oxq.operators import child_process


@pytest.mark.parametrize("timeout_seconds", [0, -1, math.nan, math.inf])
def test_rejects_non_positive_or_non_finite_child_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: float,
) -> None:
    def fail_if_started(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("child process started before timeout validation")

    monkeypatch.setattr(child_process.subprocess, "Popen", fail_if_started)

    with pytest.raises(ValueError, match="timeout must be finite and positive"):
        child_process.run_contained_child(
            [sys.executable, "-I", "-S", "-c", "pass"],
            timeout_seconds=timeout_seconds,
            response_secret=None,
            environment={},
        )


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
    with pytest.raises(ValueError, match="-I and -S"):
        child_process.run_contained_child(
            command,
            timeout_seconds=5,
            response_secret=None,
            environment={},
        )


def test_rejects_child_command_with_python_startup_flags_after_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_started(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("child process started before interpreter flag ordering validation")

    monkeypatch.setattr(child_process.subprocess, "Popen", fail_if_started)

    with pytest.raises(ValueError, match="interpreter-option prefix"):
        child_process.run_contained_child(
            [sys.executable, "-c", "pass", "-I", "-S"],
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


def test_windows_launch_uses_controller_job_gate_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}
    events: list[str] = []

    class FakeProcess:
        pid = 123
        returncode = 0
        stdin = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

        def poll(self) -> int:
            return self.returncode

    def recording_popen(command: list[str], **kwargs: object) -> FakeProcess:
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(child_process, "_platform_name", lambda: "nt")
    monkeypatch.setattr(child_process.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(child_process, "_open_windows_kill_on_close_job", lambda process: events.append("job") or 456)
    monkeypatch.setattr(child_process, "_close_windows_job", lambda handle: events.append(f"close:{handle}"))

    assert (
        child_process.run_contained_child(
            [sys.executable, "-I", "-S", "-c", "pass"],
            timeout_seconds=5,
            response_secret=None,
            environment={},
        )
        == 0
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[:4] == [sys.executable, "-I", "-S", "-c"]
    assert command[4] == child_process._WINDOWS_JOB_GATE_SCRIPT
    assert command[5:] == [sys.executable, "-I", "-S", "-c", "pass"]
    env = observed["env"]
    assert isinstance(env, dict)
    assert "OXQ_BASELINE_WINDOWS_JOB_GATE" in env
    assert events == ["job", "close:456"]


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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper assertion")
def test_timeout_reaps_detached_provider_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "descendant.pid"
    script = (
        "import pathlib, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        f"'import os, pathlib, time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)'], "
        "start_new_session=True)\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        child_process.run_contained_child(
            [sys.executable, "-I", "-S", "-c", script],
            timeout_seconds=0.5,
            response_secret=None,
            environment={},
        )
    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guardian assertion")
def test_guardian_reaps_provider_after_provider_kills_launcher(tmp_path: Path) -> None:
    pid_path = tmp_path / "provider.pid"
    script = (
        "import os, pathlib, signal, time\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "os.kill(os.getppid(), signal.SIGKILL)\n"
        "time.sleep(60)\n"
    )
    returncode = child_process.run_contained_child(
        [sys.executable, "-I", "-S", "-c", script],
        timeout_seconds=5,
        response_secret=None,
        environment={},
    )
    assert returncode != 0
    provider_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(provider_pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper assertion")
def test_success_reaps_detached_provider_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "successful-descendant.pid"
    script = (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        f"'import os, pathlib, time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)'], "
        "start_new_session=True)\n"
        "time.sleep(0.2)\n"
    )
    assert child_process.run_contained_child(
        [sys.executable, "-I", "-S", "-c", script],
        timeout_seconds=5,
        response_secret=None,
        environment={},
    ) == 0
    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX containment assertion")
def test_success_contains_detached_atexit_descendant_when_tracking_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "detached-atexit-descendant.pid"
    script = (
        "import atexit, pathlib, subprocess, sys\n"
        "def spawn_at_exit():\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        f"    pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
        "atexit.register(spawn_at_exit)\n"
    )
    monkeypatch.setattr(child_process, "_posix_descendant_pids", lambda root_pid: set())
    assert child_process.run_contained_child(
        [sys.executable, "-I", "-S", "-c", script],
        timeout_seconds=5,
        response_secret=None,
        environment={},
    ) == 0
    if pid_path.exists():
        descendant_pid = int(pid_path.read_text(encoding="utf-8"))
        try:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(descendant_pid)],
                check=False,
                capture_output=True,
                text=True,
            )
            assert state.returncode != 0 or state.stdout.lstrip().startswith("Z")
        finally:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object assertion")
def test_windows_success_reaps_provider_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "windows-descendant.pid"
    script = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        f"'import os, pathlib, time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)'], "
        "creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))\n"
        "time.sleep(0.2)\n"
    )
    assert child_process.run_contained_child(
        [sys.executable, "-I", "-S", "-c", script],
        timeout_seconds=5,
        response_secret=None,
        environment={},
    ) == 0
    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {descendant_pid}", "/NH"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert str(descendant_pid) not in completed.stdout
