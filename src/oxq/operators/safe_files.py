"""No-follow filesystem primitives shared by operator storage code."""
from __future__ import annotations
import os
import stat
from pathlib import Path

def read_regular_file(path: Path) -> bytes:
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
