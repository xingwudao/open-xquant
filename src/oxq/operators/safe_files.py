"""No-follow filesystem primitives shared by operator storage code."""
from __future__ import annotations
import os
import stat
from pathlib import Path

def read_regular_file(path: Path) -> bytes:
    if os.name == "nt":
        return _read_regular_file_windows(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise OSError("path is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024): chunks.append(chunk)
        return b"".join(chunks)
    finally: os.close(fd)

def write_file(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())

def fsync_directory(path: Path) -> None:
    if os.name == "nt": return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(fd)
    finally: os.close(fd)

def replace_directory(source: Path, target: Path) -> None:
    os.replace(source, target)

def _read_regular_file_windows(path: Path) -> bytes:
    """Read only a non-reparse regular file through a pinned Windows handle."""
    import ctypes
    from ctypes import wintypes
    create=ctypes.WinDLL("kernel32",use_last_error=True).CreateFileW
    create.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE); create.restype=wintypes.HANDLE
    handle=create(str(path),0x80000000,3,None,3,0x00200000,None)
    if handle == ctypes.c_void_p(-1).value: raise OSError(ctypes.get_last_error(),"Windows file open failed",str(path))
    try:
        class Info(ctypes.Structure): _fields_=[("attributes",wintypes.DWORD),("tag",wintypes.DWORD)]
        info=Info(); getter=ctypes.WinDLL("kernel32",use_last_error=True).GetFileInformationByHandleEx
        getter.argtypes=(wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD); getter.restype=wintypes.BOOL
        if not getter(handle,9,ctypes.byref(info),ctypes.sizeof(info)) or info.attributes & 0x00000410: raise OSError("path is a Windows reparse point or directory")
        import msvcrt
        fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|getattr(os,"O_BINARY",0)); handle=None
        try:
            chunks=[]
            while chunk:=os.read(fd,1024*1024): chunks.append(chunk)
            return b"".join(chunks)
        finally: os.close(fd)
    finally:
        if handle is not None: ctypes.WinDLL("kernel32",use_last_error=True).CloseHandle(handle)
