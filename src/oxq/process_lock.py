"""Cross-platform, reentrant process file locks."""

from __future__ import annotations

import errno
import hashlib
import importlib
import os
import stat
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Protocol, Self


class ProcessLockError(RuntimeError):
    """Raised when the platform process-lock backend is unavailable."""


class _LockBackend(Protocol):
    def acquire(self, handle: BinaryIO) -> None: ...

    def release(self, handle: BinaryIO) -> None: ...


class _PosixLockBackend:
    def __init__(self, fcntl_module: Any) -> None:
        self._fcntl = fcntl_module

    def acquire(self, handle: BinaryIO) -> None:
        self._fcntl.flock(handle.fileno(), self._fcntl.LOCK_EX)

    def release(self, handle: BinaryIO) -> None:
        self._fcntl.flock(handle.fileno(), self._fcntl.LOCK_UN)


class _WindowsLockBackend:
    _CONTENTION_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EDEADLK})
    _RETRY_INTERVAL_SECONDS = 0.05

    def __init__(self, msvcrt_module: Any) -> None:
        self._msvcrt = msvcrt_module

    def acquire(self, handle: BinaryIO) -> None:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        while True:
            try:
                self._msvcrt.locking(handle.fileno(), self._msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno not in self._CONTENTION_ERRNOS:
                    raise
                time.sleep(self._RETRY_INTERVAL_SECONDS)
            else:
                return

    def release(self, handle: BinaryIO) -> None:
        handle.seek(0)
        self._msvcrt.locking(handle.fileno(), self._msvcrt.LK_UNLCK, 1)


def _platform_name() -> str:
    return sys.platform


def _load_backend() -> _LockBackend:
    if _platform_name() == "win32":
        module_name = "msvcrt"
        backend_type = _WindowsLockBackend
    else:
        module_name = "fcntl"
        backend_type = _PosixLockBackend
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProcessLockError(f"required process-lock backend {module_name!r} is unavailable on {_platform_name()}") from exc
    return backend_type(module)


@dataclass
class _LockState:
    mutex: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0
    handle: BinaryIO | None = None
    backend: _LockBackend | None = None
    identity: tuple[int, int] | None = None


_STATE_GUARD = threading.Lock()
_LOCK_STATES: dict[str, _LockState] = {}
_IDENTITY_STATES: dict[tuple[int, int], _LockState] = {}
_REGISTRY_PID = os.getpid()


def _reset_after_fork() -> None:
    global _IDENTITY_STATES, _LOCK_STATES, _REGISTRY_PID, _STATE_GUARD

    for state in tuple(_LOCK_STATES.values()):
        handle = state.handle
        state.depth = 0
        state.handle = None
        state.backend = None
        state.identity = None
        state.mutex = threading.RLock()
        if handle is not None:
            handle.close()
    _STATE_GUARD = threading.Lock()
    _LOCK_STATES = {}
    _IDENTITY_STATES = {}
    _REGISTRY_PID = os.getpid()


def _ensure_current_process_registry() -> None:
    if _REGISTRY_PID != os.getpid():
        _reset_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _state_for(path: Path) -> _LockState:
    _ensure_current_process_registry()
    key = os.path.normcase(str(path.resolve(strict=False)))
    with _STATE_GUARD:
        return _LOCK_STATES.setdefault(key, _LockState())


def _state_for_identity(
    candidate: _LockState,
    identity: tuple[int, int],
) -> _LockState:
    with _STATE_GUARD:
        previous = candidate.identity
        if previous is not None and previous != identity and _IDENTITY_STATES.get(previous) is candidate:
            del _IDENTITY_STATES[previous]
        existing = _IDENTITY_STATES.get(identity)
        if existing is not None:
            return existing
        candidate.identity = identity
        _IDENTITY_STATES[identity] = candidate
        return candidate


def _runtime_owner_id() -> int | None:
    get_effective_uid = getattr(os, "geteuid", None)
    return int(get_effective_uid()) if get_effective_uid is not None else None


def _effective_user_identity() -> str:
    owner_id = _runtime_owner_id()
    if owner_id is not None:
        return f"uid-{owner_id}"
    return f"sid-{_windows_user_sid()}"


def _windows_user_sid() -> str:
    import ctypes
    from ctypes import wintypes

    token_query = 0x0008
    token_user_class = 1
    error_insufficient_buffer = 122
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = ()
    get_current_process.restype = wintypes.HANDLE
    open_process_token = advapi32.OpenProcessToken
    open_process_token.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    open_process_token.restype = wintypes.BOOL
    get_token_information = advapi32.GetTokenInformation
    get_token_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    get_token_information.restype = wintypes.BOOL
    convert_sid = advapi32.ConvertSidToStringSidW
    convert_sid.argtypes = (wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR))
    convert_sid.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = (ctypes.c_void_p,)
    local_free.restype = ctypes.c_void_p

    token = wintypes.HANDLE()
    if not open_process_token(get_current_process(), token_query, ctypes.byref(token)):
        error = ctypes.get_last_error()
        raise ProcessLockError(f"cannot open the current Windows user token: {ctypes.WinError(error)}")
    try:
        required = wintypes.DWORD()
        get_token_information(
            token,
            token_user_class,
            None,
            0,
            ctypes.byref(required),
        )
        error = ctypes.get_last_error()
        if error != error_insufficient_buffer or required.value == 0:
            raise ProcessLockError(f"cannot size the current Windows user SID: {ctypes.WinError(error)}")
        buffer = ctypes.create_string_buffer(required.value)
        if not get_token_information(
            token,
            token_user_class,
            buffer,
            required,
            ctypes.byref(required),
        ):
            error = ctypes.get_last_error()
            raise ProcessLockError(f"cannot read the current Windows user SID: {ctypes.WinError(error)}")

        class SidAndAttributes(ctypes.Structure):
            _fields_ = [("sid", wintypes.LPVOID), ("attributes", wintypes.DWORD)]

        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(SidAndAttributes)).contents.sid
        sid_text = wintypes.LPWSTR()
        if not convert_sid(sid_pointer, ctypes.byref(sid_text)):
            error = ctypes.get_last_error()
            raise ProcessLockError(f"cannot format the current Windows user SID: {ctypes.WinError(error)}")
        try:
            return str(sid_text.value)
        finally:
            local_free(ctypes.cast(sid_text, ctypes.c_void_p))
    finally:
        close_handle(token)


def _windows_local_app_data_path() -> Path:
    import ctypes
    from ctypes import wintypes

    csidl_local_app_data = 0x001C
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    get_folder_path = shell32.SHGetFolderPathW
    get_folder_path.argtypes = (
        wintypes.HWND,
        ctypes.c_int,
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
    )
    get_folder_path.restype = ctypes.c_long
    buffer = ctypes.create_unicode_buffer(32768)
    result = get_folder_path(None, csidl_local_app_data, None, 0, buffer)
    if result != 0 or not buffer.value:
        raise ProcessLockError(f"cannot resolve the current Windows Local AppData directory: HRESULT 0x{result & 0xFFFFFFFF:08x}")
    return Path(buffer.value)


def _stable_runtime_base_path() -> Path:
    if os.name == "nt":
        return _windows_local_app_data_path() / "open-xquant"
    system_temporary = Path("/tmp")
    if system_temporary.is_dir():
        return system_temporary
    return Path(tempfile.gettempdir())


def stable_filesystem_identity(path: str | Path) -> str:
    """Return a stable identity for an existing filesystem object."""

    candidate = Path(path)
    try:
        status = candidate.stat()
    except OSError as exc:
        raise ProcessLockError(f"cannot inspect filesystem identity for {candidate}: {exc}") from exc
    if _platform_name() != "win32":
        return f"posix:{int(status.st_dev)}:{int(status.st_ino)}"
    try:
        volume_serial, file_index = _windows_file_identity(candidate)
    except OSError as exc:
        device = int(status.st_dev)
        inode = int(status.st_ino)
        if inode == 0:
            raise ProcessLockError(f"cannot obtain stable Windows filesystem identity for {candidate}: {exc}") from exc
        return f"windows-stat:{device}:{inode}"
    return f"windows:{volume_serial}:{file_index}"


def stable_path_location_identity(path: str | Path) -> str:
    """Return an alias-stable identity for an existing or missing path."""

    candidate = Path(os.path.abspath(Path(path).expanduser()))
    current = candidate
    missing_parts: list[str] = []
    while True:
        try:
            status = current.stat()
        except OSError as exc:
            if exc.errno not in {errno.ENOENT, errno.ENOTDIR}:
                raise ProcessLockError(f"cannot inspect path location for {candidate}: {exc}") from exc
            parent = current.parent
            if parent == current:
                raise ProcessLockError(f"cannot find an existing ancestor for {candidate}") from exc
            missing_parts.append(current.name)
            current = parent
            continue
        break

    if missing_parts and not stat.S_ISDIR(status.st_mode):
        raise ProcessLockError(f"nearest existing path ancestor is not a directory: {current}")
    digest = hashlib.sha256()
    for part in reversed(missing_parts):
        normalized = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", part).casefold())
        encoded = normalized.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"location:{stable_filesystem_identity(current)}:{len(missing_parts)}:{digest.hexdigest()}"


def _windows_file_identity(path: Path) -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    file_read_attributes = 0x00000080
    file_share_read = 0x00000001
    file_share_write = 0x00000002
    file_share_delete = 0x00000004
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    get_file_information = kernel32.GetFileInformationByHandle
    get_file_information.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    )
    get_file_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(path),
        file_read_attributes,
        file_share_read | file_share_write | file_share_delete,
        None,
        open_existing,
        file_flag_backup_semantics,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error), str(path))
    try:
        information = ByHandleFileInformation()
        if not get_file_information(handle, ctypes.byref(information)):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(path))
        file_index = (int(information.file_index_high) << 32) | int(information.file_index_low)
        return int(information.volume_serial_number), file_index
    finally:
        close_handle(handle)


def _user_runtime_root_path() -> Path:
    identity = hashlib.sha256(_effective_user_identity().encode("utf-8")).hexdigest()[:24]
    return _stable_runtime_base_path() / f"open-xquant-runtime-{identity}"


def verified_user_runtime_root() -> Path:
    """Return a private runtime root for the effective user."""

    root = _user_runtime_root_path()
    try:
        root.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        pass
    try:
        status = root.lstat()
    except OSError as exc:
        raise ProcessLockError(f"cannot inspect per-user runtime root {root}: {exc}") from exc
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_point = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400))
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode) or reparse_point:
        raise ProcessLockError(f"unsafe per-user runtime root: {root}")
    owner_id = _runtime_owner_id()
    if owner_id is not None and status.st_uid != owner_id:
        raise ProcessLockError(f"unsafe owner for per-user runtime root: {root}")
    if os.name != "nt" and stat.S_IMODE(status.st_mode) & 0o077:
        raise ProcessLockError(f"unsafe permissions for per-user runtime root: {root}")
    return root


class ProcessFileLock:
    """Hold an exclusive process lock, reusing it for same-thread nesting."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve(strict=False)
        self._state = _state_for(self._path)
        self._entered = False
        self._pid = os.getpid()

    def _refresh_after_fork(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        _ensure_current_process_registry()
        self._state = _state_for(self._path)
        self._entered = False
        self._pid = pid

    @property
    def descriptor(self) -> int:
        self._refresh_after_fork()
        if not self._entered or self._state.handle is None:
            raise ProcessLockError("process lock is not held")
        return self._state.handle.fileno()

    def __enter__(self) -> Self:
        self._refresh_after_fork()
        if self._entered:
            raise ProcessLockError("the same ProcessFileLock instance cannot be entered twice")
        while True:
            state = self._state
            state.mutex.acquire()
            try:
                if state.depth != 0:
                    state.depth += 1
                    self._entered = True
                    return self
                self._path.parent.mkdir(parents=True, exist_ok=True)
                handle = self._path.open("a+b")
                try:
                    status = os.fstat(handle.fileno())
                    if status.st_nlink > 1:
                        raise ProcessLockError(f"process lock file has multiple hard links: {self._path}")
                    canonical = _state_for_identity(
                        state,
                        (int(status.st_dev), int(status.st_ino)),
                    )
                    if canonical is not state:
                        self._state = canonical
                        handle.close()
                        state.mutex.release()
                        continue
                    backend = _load_backend()
                    backend.acquire(handle)
                except BaseException:
                    if not handle.closed:
                        handle.close()
                    raise
                state.handle = handle
                state.backend = backend
                state.depth = 1
                self._entered = True
                return self
            except BaseException:
                state.mutex.release()
                raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._refresh_after_fork()
        if not self._entered:
            return
        try:
            self._state.depth -= 1
            if self._state.depth == 0:
                handle = self._state.handle
                backend = self._state.backend
                self._state.handle = None
                self._state.backend = None
                if handle is not None:
                    try:
                        if backend is not None:
                            backend.release(handle)
                    finally:
                        handle.close()
        finally:
            self._entered = False
            self._state.mutex.release()
