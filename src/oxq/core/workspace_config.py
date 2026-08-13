"""Canonical, read-only governed workspace configuration loading."""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml  # type: ignore[import-untyped]

WORKSPACE_CONFIG_RELATIVE_PATH = Path(".open-xquant/workspace.yaml")


class WorkspaceConfigError(ValueError):
    """Raised when a present workspace marker cannot be trusted or parsed."""


@dataclass(frozen=True, slots=True)
class DiscoveredWorkspaceConfig:
    root: Path
    path: Path
    config: dict[str, Any]


def load_workspace_config(
    path: str | Path,
    *,
    missing_ok: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Load the canonical workspace YAML object without following symlinks."""

    config_path = Path(path)
    if config_path.name != "workspace.yaml" or config_path.parent.name != ".open-xquant":
        raise WorkspaceConfigError(f"workspace configuration path is not canonical: {config_path}")
    try:
        config_text = _read_regular_file_nofollow(config_path)
    except FileNotFoundError as exc:
        if missing_ok:
            return {}
        raise WorkspaceConfigError(f"workspace configuration does not exist: {config_path}") from exc
    except WorkspaceConfigError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkspaceConfigError(f"workspace configuration could not be read: {config_path}: {exc}") from exc
    try:
        payload = _load_workspace_yaml(config_text, path=config_path)
    except yaml.YAMLError as exc:
        raise WorkspaceConfigError(f"workspace configuration contains invalid YAML: {config_path}: {exc}") from exc
    if payload is None and allow_empty:
        return {}
    if not isinstance(payload, dict):
        raise WorkspaceConfigError(f"workspace configuration must contain an object: {config_path}")
    return payload


def _load_workspace_yaml(raw: str, *, path: Path) -> Any:
    loader = yaml.SafeLoader(raw)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _reject_duplicate_workspace_yaml_keys(node, loader=loader, path=path)
        return loader.construct_document(node)
    finally:
        loader.dispose()


def _reject_duplicate_workspace_yaml_keys(
    node: Any,
    *,
    loader: Any,
    path: Path,
    location: str = "workspace",
    active: set[int] | None = None,
    completed: set[int] | None = None,
) -> None:
    active = set() if active is None else active
    completed = set() if completed is None else completed
    identity = id(node)
    if identity in active or identity in completed:
        return
    active.add(identity)
    try:
        if isinstance(node, yaml.MappingNode):
            seen: set[Any] = set()
            for key_node, value_node in node.value:
                key = key_node.value if key_node.tag == "tag:yaml.org,2002:merge" else loader.construct_object(key_node, deep=True)
                child_location = f"{location}.{key}" if isinstance(key, str) else location
                try:
                    duplicate = key in seen
                    seen.add(key)
                except TypeError:
                    duplicate = False
                if duplicate:
                    raise WorkspaceConfigError(
                        f"workspace configuration contains invalid YAML: {path}: {child_location}: duplicate mapping key: {key}"
                    )
                _reject_duplicate_workspace_yaml_keys(
                    value_node,
                    loader=loader,
                    path=path,
                    location=child_location,
                    active=active,
                    completed=completed,
                )
        elif isinstance(node, yaml.SequenceNode):
            for index, item in enumerate(node.value):
                _reject_duplicate_workspace_yaml_keys(
                    item,
                    loader=loader,
                    path=path,
                    location=f"{location}[{index}]",
                    active=active,
                    completed=completed,
                )
        completed.add(identity)
    finally:
        active.remove(identity)


def _read_regular_file_nofollow(path: Path) -> str:
    if _platform_name() == "nt":
        return _read_regular_file_windows_nofollow(path)

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(nofollow, int) or nofollow == 0 or not isinstance(directory, int) or directory == 0:
        raise _nofollow_unavailable()

    absolute = path.absolute()
    directory_flags = os.O_RDONLY | nofollow | directory
    directory_descriptor = _open_directory_nofollow(
        absolute.anchor,
        directory_flags,
        display_path=Path(absolute.anchor),
    )
    current_path = Path(absolute.anchor)

    try:
        for component in absolute.parts[1:-1]:
            current_path /= component
            child_descriptor = _open_directory_nofollow(
                component,
                directory_flags,
                display_path=current_path,
                dir_fd=directory_descriptor,
            )
            previous_descriptor = directory_descriptor
            directory_descriptor = child_descriptor
            os.close(previous_descriptor)
        return _read_regular_file_at(absolute.name, path, directory_descriptor, nofollow)
    finally:
        os.close(directory_descriptor)


class _WindowsWorkspaceFileApi(Protocol):
    def open(self, path: Path, *, directory: bool) -> int: ...

    def attributes(self, handle: int) -> int: ...

    def is_disk_file(self, handle: int) -> bool: ...

    def read_bytes(self, handle: int) -> bytes: ...

    def close(self, handle: int) -> None: ...


class _NativeWindowsWorkspaceFileApi:
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_READ_ATTRIBUTES = 0x00000080
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_TYPE_DISK = 0x0001
    _GENERIC_READ = 0x80000000
    _OPEN_EXISTING = 3

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = [
                ("file_attributes", wintypes.DWORD),
                ("reparse_tag", wintypes.DWORD),
            ]

        self._ctypes = ctypes
        self._file_attribute_tag_info = FileAttributeTagInfo
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        self._create_file = kernel32.CreateFileW
        self._create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        self._create_file.restype = wintypes.HANDLE
        self._get_file_information = kernel32.GetFileInformationByHandleEx
        self._get_file_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        self._get_file_information.restype = wintypes.BOOL
        self._get_file_type = kernel32.GetFileType
        self._get_file_type.argtypes = (wintypes.HANDLE,)
        self._get_file_type.restype = wintypes.DWORD
        self._read_file = kernel32.ReadFile
        self._read_file.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        )
        self._read_file.restype = wintypes.BOOL
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL

    def open(self, path: Path, *, directory: bool) -> int:
        desired_access = self._FILE_READ_ATTRIBUTES if directory else self._GENERIC_READ | self._FILE_READ_ATTRIBUTES
        flags = self._FILE_FLAG_OPEN_REPARSE_POINT
        if directory:
            flags |= self._FILE_FLAG_BACKUP_SEMANTICS
        # Pin each absolute-path component against rename until the leaf handle is open.
        handle = self._create_file(
            str(path),
            desired_access,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            None,
            self._OPEN_EXISTING,
            flags,
            None,
        )
        invalid_handle = self._ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            error = int(getattr(self._ctypes, "get_last_error")())
            if error in {2, 3}:
                raise FileNotFoundError(error, os.strerror(error), str(path))
            raise OSError(error, os.strerror(error), str(path))
        return int(handle)

    def attributes(self, handle: int) -> int:
        info = self._file_attribute_tag_info()
        if not self._get_file_information(handle, 9, self._ctypes.byref(info), self._ctypes.sizeof(info)):
            error = int(getattr(self._ctypes, "get_last_error")())
            raise OSError(error, os.strerror(error))
        return int(info.file_attributes)

    def is_disk_file(self, handle: int) -> bool:
        return int(self._get_file_type(handle)) == self._FILE_TYPE_DISK

    def read_bytes(self, handle: int) -> bytes:
        chunks: list[bytes] = []
        while True:
            buffer = self._ctypes.create_string_buffer(1024 * 1024)
            count = self._ctypes.c_uint32()
            if not self._read_file(handle, buffer, len(buffer), self._ctypes.byref(count), None):
                error = int(getattr(self._ctypes, "get_last_error")())
                raise OSError(error, os.strerror(error))
            if count.value == 0:
                return b"".join(chunks)
            chunks.append(buffer.raw[: count.value])

    def close(self, handle: int) -> None:
        self._close_handle(handle)


def _platform_name() -> str:
    return os.name


def _windows_workspace_file_api() -> _WindowsWorkspaceFileApi:
    return _NativeWindowsWorkspaceFileApi()


def _read_regular_file_windows_nofollow(path: Path) -> str:
    absolute = path.absolute()
    api = _windows_workspace_file_api()
    handles: list[int] = []
    reparse_attribute = 0x00000400
    directory_attribute = 0x00000010
    current = Path(absolute.anchor)
    directories = [current]
    for component in absolute.parts[1:-1]:
        current /= component
        directories.append(current)
    try:
        for directory_path in directories:
            handle = api.open(directory_path, directory=True)
            handles.append(handle)
            attributes = api.attributes(handle)
            if attributes & reparse_attribute:
                raise WorkspaceConfigError(f"workspace configuration path component must not be a reparse point: {directory_path}")
            if not attributes & directory_attribute:
                raise WorkspaceConfigError(f"workspace configuration path component must be a directory: {directory_path}")

        file_handle = api.open(absolute, directory=False)
        handles.append(file_handle)
        attributes = api.attributes(file_handle)
        if attributes & reparse_attribute:
            raise WorkspaceConfigError(f"workspace configuration must not be a symlink or reparse point: {path}")
        if attributes & directory_attribute or not api.is_disk_file(file_handle):
            raise WorkspaceConfigError(f"workspace configuration must be a regular file: {path}")
        return api.read_bytes(file_handle).decode("utf-8")
    finally:
        while handles:
            api.close(handles.pop())


def _open_directory_nofollow(
    path: str,
    flags: int,
    *,
    display_path: Path,
    dir_fd: int | None = None,
) -> int:
    try:
        if dir_fd is None:
            descriptor = os.open(path, flags)
        else:
            descriptor = os.open(path, flags, dir_fd=dir_fd)
    except (NotImplementedError, TypeError) as exc:
        raise _nofollow_unavailable() from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise WorkspaceConfigError(
                f"workspace configuration path component must be a directory and must not be a symlink: {display_path}"
            ) from exc
        raise

    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise WorkspaceConfigError(f"workspace configuration path component must be a directory: {display_path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_file_at(name: str, path: Path, directory_descriptor: int, nofollow: int) -> str:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | nofollow, dir_fd=directory_descriptor)
    except (NotImplementedError, TypeError) as exc:
        raise _nofollow_unavailable() from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise WorkspaceConfigError(f"workspace configuration must not be a symlink: {path}") from exc
        raise

    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WorkspaceConfigError(f"workspace configuration must be a regular file: {path}")
        stream = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = -1
        with stream:
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _nofollow_unavailable() -> WorkspaceConfigError:
    return WorkspaceConfigError("workspace configuration cannot be read: atomic directory-descriptor nofollow protection is unavailable")


def discover_workspace_config(path: str | Path) -> DiscoveredWorkspaceConfig | None:
    """Find and load the nearest canonical workspace marker."""

    subject = Path(path).absolute()
    start = subject if subject.is_dir() else subject.parent
    for candidate in (start, *start.parents):
        config_path = candidate / WORKSPACE_CONFIG_RELATIVE_PATH
        try:
            config_directory_status = config_path.parent.lstat()
        except FileNotFoundError:
            config_directory_status = None
        except OSError as exc:
            raise WorkspaceConfigError(f"workspace configuration directory could not be inspected: {config_path.parent}: {exc}") from exc
        if config_directory_status is not None:
            attributes = getattr(config_directory_status, "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
            if stat.S_ISLNK(config_directory_status.st_mode) or bool(attributes & reparse_flag):
                raise WorkspaceConfigError(
                    f"workspace configuration directory must not be a symlink or reparse point: {config_path.parent}"
                )
            if not stat.S_ISDIR(config_directory_status.st_mode):
                raise WorkspaceConfigError(f"workspace configuration directory must be a directory: {config_path.parent}")
        if not config_path.exists() and not config_path.is_symlink():
            continue
        config = load_workspace_config(config_path)
        return DiscoveredWorkspaceConfig(root=candidate.resolve(), path=config_path, config=config)
    return None
