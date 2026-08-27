"""Contained child-process controller tests."""

from __future__ import annotations

import os
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
