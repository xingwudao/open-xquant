"""Contained controller for isolated operator child processes."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

_CHILD_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TZ",
        "WINDIR",
        "OXQ_EXACT_TEST_RUNTIME",
        "OXQ_EXACT_TEST_RUNTIME_PATHS",
    },
)
_DARWIN_SANDBOX_PROFILE = "(version 1) (allow default) (deny process-fork)"
_LINUX_LAUNCHER_SCRIPT = r"""
import os
import signal
import subprocess
import sys

target = subprocess.Popen(sys.argv[1:])
returncode = target.wait()
if returncode < 0:
    signal.signal(-returncode, signal.SIG_DFL)
    os.kill(os.getpid(), -returncode)
raise SystemExit(returncode)
"""
_WINDOWS_JOB_GATE_SCRIPT = r"""
import os
import pathlib
import subprocess
import sys
import time

gate = pathlib.Path(os.environ["OXQ_BASELINE_WINDOWS_JOB_GATE"])
deadline = time.monotonic() + 30
while not gate.exists():
    if time.monotonic() >= deadline:
        raise SystemExit(124)
    time.sleep(0.01)
raise SystemExit(subprocess.call(sys.argv[1:]))
"""
_LINUX_SUBREAPER_SCRIPT = r"""
import ctypes
import os
import signal
import subprocess
import sys


def request_termination(signum, frame):
    global termination_requested
    del signum, frame
    termination_requested = True


def child_pids():
    path = f"/proc/{os.getpid()}/task/{os.getpid()}/children"
    value = open(path, encoding="ascii").read().strip()
    return [int(pid) for pid in value.split()]


def terminate_adopted_children():
    while True:
        children = child_pids()
        if not children:
            return
        for pid in children:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        for pid in children:
            try:
                os.waitpid(pid, os.WUNTRACED)
            except ChildProcessError:
                pass
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in children:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
termination_requested = False
signal.signal(signal.SIGTERM, request_termination)
target = None
returncode = 1
try:
    target = subprocess.Popen(sys.argv[1:])
    while True:
        if termination_requested:
            target.kill()
            target.wait()
            returncode = 124
            break
        try:
            returncode = target.wait(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            continue
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if target is not None and target.poll() is None:
        target.kill()
        target.wait()
    terminate_adopted_children()
if returncode < 0:
    signal.signal(-returncode, signal.SIG_DFL)
    os.kill(os.getpid(), -returncode)
raise SystemExit(returncode)
"""


def run_contained_child(
    command: list[str],
    *,
    timeout_seconds: float,
    response_secret: bytes | None,
    environment: Mapping[str, str],
) -> int:
    """Run a Python child with platform containment and a scrubbed environment."""
    _validate_python_isolation_flags(command)
    timeout_seconds = _validated_timeout(timeout_seconds)
    platform_name = _platform_name()
    provider_command = command
    child_environment = _contained_environment(environment)
    windows_job: int | None = None
    windows_gate_directory: TemporaryDirectory[str] | None = None
    if platform_name == "nt":
        windows_gate_directory = TemporaryDirectory(prefix="oxq-windows-job-")
        gate_path = Path(windows_gate_directory.name) / "assigned"
        child_environment["OXQ_BASELINE_WINDOWS_JOB_GATE"] = str(gate_path)
        try:
            process = subprocess.Popen(
                _contained_windows_command(command),
                stdin=(subprocess.PIPE if response_secret is not None else subprocess.DEVNULL),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=cast(int, getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)),
                env=child_environment,
            )
        except BaseException:
            windows_gate_directory.cleanup()
            raise
    else:
        process = subprocess.Popen(
            _contained_posix_command(command),
            stdin=(subprocess.PIPE if response_secret is not None else subprocess.DEVNULL),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=child_environment,
        )
    try:
        if response_secret is not None:
            if process.stdin is None:
                raise OSError("provider response authentication pipe is unavailable")
            try:
                process.stdin.write(response_secret.hex().encode("ascii") + b"\n")
                process.stdin.flush()
            finally:
                process.stdin.close()
        if platform_name == "nt":
            try:
                windows_job = _open_windows_kill_on_close_job(process)
                gate_path.write_bytes(b"assigned\n")
            except OSError:
                _kill_process_tree(process)
                process.wait()
                raise
        descendants: set[int] = set()
        deadline = time.monotonic() + timeout_seconds
        while True:
            if platform_name != "nt":
                descendants.update(_posix_descendant_pids(process.pid))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if _posix_platform() == "linux":
                    _terminate_linux_supervisor(process, descendants)
                else:
                    _kill_process_tree(process, descendants)
                process.wait()
                raise subprocess.TimeoutExpired(provider_command, timeout_seconds)
            try:
                process.wait(timeout=min(0.02, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if platform_name != "nt" and _posix_platform() == "linux" and process.returncode != 0:
            _kill_process_tree(process, descendants)
        return int(process.returncode)
    finally:
        try:
            if windows_job is not None:
                _close_windows_job(windows_job)
        finally:
            if windows_gate_directory is not None:
                windows_gate_directory.cleanup()


def _validated_timeout(timeout_seconds: float) -> float:
    if type(timeout_seconds) not in {int, float}:
        raise TypeError("timeout is not numeric")
    normalized = float(timeout_seconds)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout must be finite and positive")
    return normalized


def _validate_python_isolation_flags(command: list[str]) -> None:
    try:
        executable = Path(command[0]).resolve(strict=True)
        expected = Path(sys.executable).resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise ValueError("contained child command must use the expected Python executable") from exc
    if executable != expected:
        raise ValueError("contained child command must use the expected Python executable")
    option_prefix: list[str] = []
    for argument in command[1:]:
        if argument in {"-c", "-m", "-"} or not argument.startswith("-"):
            break
        option_prefix.append(argument)
    if "-I" not in option_prefix or "-S" not in option_prefix:
        raise ValueError("contained child commands must include -I and -S in the interpreter-option prefix")


def _contained_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in environment.items() if name in _CHILD_ENVIRONMENT_ALLOWLIST}


def _kill_process_tree(process: subprocess.Popen[bytes], known_descendants: set[int] | None = None) -> None:
    if _platform_name() == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True)
    else:
        descendants = _expand_posix_descendants(set(known_descendants or ()) | _posix_descendant_pids(process.pid))
        try:
            os.killpg(process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        descendants = _expand_posix_descendants(descendants | _posix_descendant_pids(process.pid))
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _kill_posix_processes(descendants)
    if process.poll() is None:
        process.kill()


def _kill_posix_processes(process_ids: set[int]) -> None:
    for pid in process_ids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _expand_posix_descendants(process_ids: set[int]) -> set[int]:
    descendants = set(process_ids)
    pending = list(process_ids)
    while pending:
        children = _posix_descendant_pids(pending.pop()).difference(descendants)
        descendants.update(children)
        pending.extend(children)
    return descendants


def _platform_name() -> str:
    return os.name


def _posix_platform() -> str:
    return sys.platform


def _contained_windows_command(command: list[str]) -> list[str]:
    return [sys.executable, "-I", "-S", "-c", _WINDOWS_JOB_GATE_SCRIPT, *command]


def _contained_posix_command(command: list[str]) -> list[str]:
    platform = _posix_platform()
    if platform == "linux":
        return [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _LINUX_SUBREAPER_SCRIPT,
            sys.executable,
            "-I",
            "-S",
            "-c",
            _LINUX_LAUNCHER_SCRIPT,
            *command,
        ]
    if platform == "darwin":
        sandbox = Path("/usr/bin/sandbox-exec")
        if not sandbox.is_file():
            raise OSError("macOS process sandbox is unavailable")
        probe = subprocess.run(
            [str(sandbox), "-p", _DARWIN_SANDBOX_PROFILE, "/usr/bin/true"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        if probe.returncode != 0:
            raise OSError("macOS process sandbox is unavailable")
        return [str(sandbox), "-p", _DARWIN_SANDBOX_PROFILE, *command]
    raise OSError(f"unsupported POSIX process-containment platform: {platform}")


def _terminate_linux_supervisor(process: subprocess.Popen[bytes], known_descendants: set[int]) -> None:
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process, known_descendants)


def _open_windows_kill_on_close_job(process: subprocess.Popen[bytes]) -> int:
    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise _windows_error()
    try:
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(information), ctypes.sizeof(information)):
            raise _windows_error()
        process_handle = getattr(process, "_handle", None)
        if process_handle is None or not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(process_handle))):
            raise _windows_error()
        return int(job)
    except BaseException:
        kernel32.CloseHandle(job)
        raise


def _close_windows_job(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = wintypes.HANDLE(handle)
    termination_error: OSError | None = None
    if not kernel32.TerminateJobObject(job, 1):
        termination_error = _windows_error()
    elif kernel32.WaitForSingleObject(job, 5000) != 0:
        termination_error = OSError("Windows provider job did not terminate")
    if not kernel32.CloseHandle(job):
        raise _windows_error()
    if termination_error is not None:
        raise termination_error


def _windows_error() -> OSError:
    import ctypes

    error = int(getattr(ctypes, "get_last_error")())
    return OSError(error, os.strerror(error))


def _posix_descendant_pids(root_pid: int) -> set[int]:
    try:
        result = subprocess.run(["ps", "-axo", "pid=,ppid="], check=False, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    children_by_parent: dict[int, set[int]] = {}
    for line in result.stdout.splitlines():
        try:
            pid_text, parent_text = line.split()
            pid = int(pid_text)
            parent = int(parent_text)
        except ValueError:
            continue
        children_by_parent.setdefault(parent, set()).add(pid)
    descendants: set[int] = set()
    pending = list(children_by_parent.get(root_pid, set()))
    while pending:
        pid = pending.pop()
        if pid not in descendants:
            descendants.add(pid)
            pending.extend(children_by_parent.get(pid, set()))
    return descendants
