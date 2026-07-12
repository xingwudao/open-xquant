from __future__ import annotations

import errno
import os
import select
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from oxq import process_lock
from oxq.process_lock import ProcessFileLock, ProcessLockError


class _FakeMsvcrt:
    LK_LOCK = 3
    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def locking(self, descriptor: int, operation: int, length: int) -> None:
        self.calls.append((descriptor, operation, length))


def test_process_lock_uses_windows_backend_without_importing_fcntl(monkeypatch, tmp_path) -> None:
    fake_msvcrt = _FakeMsvcrt()
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        if name == "msvcrt":
            return fake_msvcrt
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(process_lock, "_platform_name", lambda: "win32")
    monkeypatch.setattr(process_lock.importlib, "import_module", import_module)

    with ProcessFileLock(tmp_path / "state.lock"):
        pass

    assert imported == ["msvcrt"]
    assert [operation for _, operation, _ in fake_msvcrt.calls] == [
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_UNLCK,
    ]
    assert all(length == 1 for _, _, length in fake_msvcrt.calls)


def test_windows_process_lock_retries_contention_beyond_native_retry_limit(
    monkeypatch,
    tmp_path,
) -> None:
    class ContendedMsvcrt(_FakeMsvcrt):
        def __init__(self) -> None:
            super().__init__()
            self.acquire_attempts = 0

        def locking(self, descriptor: int, operation: int, length: int) -> None:
            super().locking(descriptor, operation, length)
            if operation == self.LK_NBLCK:
                self.acquire_attempts += 1
                if self.acquire_attempts <= 12:
                    raise OSError(errno.EACCES, "lock is held")

    fake_msvcrt = ContendedMsvcrt()
    sleeps: list[float] = []
    monkeypatch.setattr(process_lock, "_platform_name", lambda: "win32")
    monkeypatch.setattr(process_lock.importlib, "import_module", lambda _name: fake_msvcrt)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with ProcessFileLock(tmp_path / "state.lock"):
        pass

    assert fake_msvcrt.acquire_attempts == 13
    assert len(sleeps) == 12
    assert [operation for _, operation, _ in fake_msvcrt.calls][-1] == fake_msvcrt.LK_UNLCK


def test_windows_process_lock_does_not_retry_unexpected_os_error(monkeypatch, tmp_path) -> None:
    class FailingMsvcrt(_FakeMsvcrt):
        def locking(self, descriptor: int, operation: int, length: int) -> None:
            super().locking(descriptor, operation, length)
            if operation == self.LK_NBLCK:
                raise OSError(errno.EBADF, "bad descriptor")

    fake_msvcrt = FailingMsvcrt()
    sleeps: list[float] = []
    monkeypatch.setattr(process_lock, "_platform_name", lambda: "win32")
    monkeypatch.setattr(process_lock.importlib, "import_module", lambda _name: fake_msvcrt)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with pytest.raises(OSError) as exc_info:
        with ProcessFileLock(tmp_path / "state.lock"):
            pass

    assert exc_info.value.errno == errno.EBADF
    assert sleeps == []
    assert len(fake_msvcrt.calls) == 1


def test_process_lock_fails_closed_when_platform_backend_is_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(process_lock, "_platform_name", lambda: "linux")

    def unavailable(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(process_lock.importlib, "import_module", unavailable)

    with pytest.raises(ProcessLockError, match="fcntl"):
        with ProcessFileLock(tmp_path / "state.lock"):
            pass


def test_process_lock_is_reentrant_and_releases_and_closes_on_exception(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, int]] = []

    class FakeBackend:
        def acquire(self, handle) -> None:
            calls.append(("acquire", handle.fileno()))

        def release(self, handle) -> None:
            calls.append(("release", handle.fileno()))

    monkeypatch.setattr(process_lock, "_load_backend", lambda: FakeBackend())
    descriptor = -1

    with pytest.raises(RuntimeError, match="boom"):
        with ProcessFileLock(tmp_path / "state.lock") as outer:
            descriptor = outer.descriptor
            with ProcessFileLock(tmp_path / "state.lock") as inner:
                assert inner.descriptor == descriptor
                raise RuntimeError("boom")

    assert calls == [("acquire", descriptor), ("release", descriptor)]
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_relative_process_lock_keeps_constructor_path_after_chdir(monkeypatch, tmp_path) -> None:
    constructor_dir = tmp_path / "constructor"
    entered_dir = tmp_path / "entered"
    constructor_dir.mkdir()
    entered_dir.mkdir()
    monkeypatch.chdir(constructor_dir)
    lock = ProcessFileLock("state.lock")

    monkeypatch.chdir(entered_dir)
    with lock:
        with ProcessFileLock(constructor_dir / "state.lock") as alias:
            assert alias.descriptor == lock.descriptor

    assert (constructor_dir / "state.lock").is_file()
    assert not (entered_dir / "state.lock").exists()


def test_nested_hard_link_alias_is_rejected_before_second_backend_acquire(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    class FakeBackend:
        def acquire(self, _handle) -> None:
            calls.append("acquire")

        def release(self, _handle) -> None:
            calls.append("release")

    monkeypatch.setattr(process_lock, "_load_backend", lambda: FakeBackend())
    lock_path = tmp_path / "state.lock"
    alias_path = tmp_path / "state-hard-link.lock"

    with ProcessFileLock(lock_path):
        os.link(lock_path, alias_path)
        with pytest.raises(ProcessLockError, match="multiple hard links"):
            with ProcessFileLock(alias_path):
                pass

    assert calls == ["acquire", "release"]


def test_mixed_case_aliases_reuse_existing_inode_state_at_entry(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[tuple[str, int]] = []

    class FakeBackend:
        def acquire(self, handle) -> None:
            calls.append(("acquire", handle.fileno()))

        def release(self, handle) -> None:
            calls.append(("release", handle.fileno()))

    upper = tmp_path / "State.lock"
    lower = tmp_path / "state.lock"
    original_open = Path.open

    def case_insensitive_open(path: Path, *args, **kwargs):
        return original_open(upper if path == lower else path, *args, **kwargs)

    monkeypatch.setattr(process_lock, "_load_backend", lambda: FakeBackend())
    monkeypatch.setattr(Path, "open", case_insensitive_open)

    with ProcessFileLock(upper) as outer:
        with ProcessFileLock(lower) as inner:
            assert inner.descriptor == outer.descriptor

    assert [event for event, _descriptor in calls] == ["acquire", "release"]


def test_stable_path_location_identity_casefolds_a_missing_suffix(tmp_path) -> None:
    upper = tmp_path / "Config" / "Open-XQuant"
    lower = tmp_path / "config" / "open-xquant"

    assert process_lock.stable_path_location_identity(upper) == process_lock.stable_path_location_identity(lower)


def test_stable_path_location_identity_uses_existing_ancestor_identity_for_aliases(
    tmp_path,
) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(backing, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    canonical = backing / "Config" / "Open-XQuant"
    alternate = alias / "config" / "open-xquant"

    assert process_lock.stable_path_location_identity(canonical) == process_lock.stable_path_location_identity(alternate)


def test_stable_path_location_identity_collapses_an_existing_final_alias(tmp_path) -> None:
    location = tmp_path / "Open-XQuant"
    location.mkdir()
    alias = tmp_path / "agent-config-alias"
    try:
        alias.symlink_to(location, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    assert process_lock.stable_path_location_identity(alias) == process_lock.stable_path_location_identity(location)


def test_verified_user_runtime_root_is_owner_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(process_lock, "_stable_runtime_base_path", lambda: tmp_path)

    root = process_lock.verified_user_runtime_root()

    assert stat.S_IMODE(root.stat().st_mode) == 0o700


@pytest.mark.parametrize("unsafe", ["mode", "owner"])
def test_verified_user_runtime_root_rejects_unsafe_preexisting_directory(
    monkeypatch,
    tmp_path,
    unsafe: str,
) -> None:
    monkeypatch.setattr(process_lock, "_stable_runtime_base_path", lambda: tmp_path)
    root = process_lock._user_runtime_root_path()
    root.mkdir(mode=0o700)
    if unsafe == "mode":
        root.chmod(0o755)
    else:
        monkeypatch.setattr(
            process_lock,
            "_runtime_owner_id",
            lambda: root.stat().st_uid + 1,
        )

    with pytest.raises(ProcessLockError, match="runtime root"):
        process_lock.verified_user_runtime_root()


def test_user_runtime_roots_are_isolated_by_effective_identity(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(process_lock, "_stable_runtime_base_path", lambda: tmp_path)
    monkeypatch.setattr(process_lock, "_effective_user_identity", lambda: "uid-1001")
    first = process_lock.verified_user_runtime_root()
    monkeypatch.setattr(process_lock, "_effective_user_identity", lambda: "uid-1002")
    second = process_lock.verified_user_runtime_root()

    assert first != second
    assert first.parent == second.parent == tmp_path


def test_non_posix_user_identity_uses_sid_instead_of_login_environment(monkeypatch) -> None:
    monkeypatch.setattr(process_lock, "_runtime_owner_id", lambda: None)
    monkeypatch.setattr(
        process_lock,
        "_windows_user_sid",
        lambda: "S-1-5-21-111-222-333-1001",
        raising=False,
    )
    monkeypatch.setenv("USERNAME", "first-name")
    monkeypatch.setenv("USERDOMAIN", "first-domain")
    first = process_lock._effective_user_identity()
    monkeypatch.setenv("USERNAME", "second-name")
    monkeypatch.setenv("USERDOMAIN", "second-domain")

    assert process_lock._effective_user_identity() == first
    assert first == "sid-S-1-5-21-111-222-333-1001"


def test_user_runtime_lock_contends_across_different_temp_environments(tmp_path) -> None:
    first_temp = tmp_path / "first-temp"
    second_temp = tmp_path / "second-temp"
    first_temp.mkdir()
    second_temp.mkdir()
    release = tmp_path / "release"
    acquired = tmp_path / "acquired"
    lock_name = f"temp-env-{tmp_path.name}.lock"
    code = """
import pathlib
import sys
import time

from oxq.process_lock import ProcessFileLock, verified_user_runtime_root

mode, signal, lock_name = sys.argv[1:]
signal_path = pathlib.Path(signal)
lock_path = verified_user_runtime_root() / "tests" / lock_name
print(lock_path, flush=True)
with ProcessFileLock(lock_path):
    if mode == "hold":
        print("held", flush=True)
        deadline = time.monotonic() + 15
        while not signal_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("release was not signaled")
            time.sleep(0.02)
    else:
        signal_path.write_text("acquired", encoding="utf-8")
"""

    def temp_env(path: Path) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(path),
            "TEMP": str(path),
            "TMP": str(path),
        }

    first = subprocess.Popen(
        [sys.executable, "-c", code, "hold", str(release), lock_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=temp_env(first_temp),
    )
    second: subprocess.Popen[str] | None = None
    try:
        first_lock_path = first.stdout.readline().strip() if first.stdout is not None else ""
        assert first.stdout is not None
        assert first.stdout.readline().strip() == "held"

        second = subprocess.Popen(
            [sys.executable, "-c", code, "acquire", str(acquired), lock_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=temp_env(second_temp),
        )
        assert second.stdout is not None
        second_lock_path = second.stdout.readline().strip()

        assert second_lock_path == first_lock_path
        time.sleep(0.4)
        assert not acquired.exists(), "different temp environments bypassed the per-user lock"

        release.write_text("release", encoding="utf-8")
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
    assert second.returncode == 0, second_stdout + second_stderr
    assert acquired.read_text(encoding="utf-8") == "acquired"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
@pytest.mark.parametrize("alias", ["relative", "absolute", "symlink"])
def test_forked_child_reacquires_parent_held_lock_for_all_path_aliases(
    monkeypatch,
    tmp_path,
    alias: str,
) -> None:
    lock_path = tmp_path / "state.lock"
    lock_path.touch()
    symlink_path = tmp_path / "state-link.lock"
    symlink_path.symlink_to(lock_path)
    monkeypatch.chdir(tmp_path)
    child_path = {
        "relative": Path("state.lock"),
        "absolute": lock_path,
        "symlink": symlink_path,
    }[alias]
    read_fd, write_fd = os.pipe()
    child_pid = -1

    try:
        with ProcessFileLock(lock_path):
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                try:
                    os.write(write_fd, b"R")
                    with ProcessFileLock(child_path):
                        os.write(write_fd, b"A")
                except BaseException as exc:
                    os.write(write_fd, b"E" + repr(exc).encode("utf-8", errors="replace"))
                finally:
                    os.close(write_fd)
                    os._exit(0)

            os.close(write_fd)
            assert os.read(read_fd, 1) == b"R"
            readable, _, _ = select.select([read_fd], [], [], 0.25)
            assert readable == [], "forked child bypassed the parent-held OS lock"

        readable, _, _ = select.select([read_fd], [], [], 5)
        assert readable == [read_fd]
        assert os.read(read_fd, 4096) == b"A"
    finally:
        os.close(read_fd)
        if child_pid > 0:
            waited_pid, status = os.waitpid(child_pid, 0)
            assert waited_pid == child_pid
            assert os.waitstatus_to_exitcode(status) == 0
