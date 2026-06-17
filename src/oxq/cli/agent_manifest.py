"""Manifest, hashing, and marker-block helpers for Agent lifecycle commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class MarkerBlockError(ValueError):
    """Raised when a managed marker block is malformed."""


def expand_path(path: str | Path) -> Path:
    """Expand user/env vars and return an absolute path."""

    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    write_text_file(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_yaml_file(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    write_text_file(path, yaml.safe_dump(payload, sort_keys=False, width=1000))


def _markers(marker: str) -> tuple[str, str]:
    return f"<!-- {marker}:begin -->", f"<!-- {marker}:end -->"


def _find_marker_block(text: str, marker: str) -> tuple[int, int] | None:
    begin, end = _markers(marker)
    begin_index = text.find(begin)
    end_index = text.find(end)
    if begin_index == -1 and end_index == -1:
        return None
    if begin_index == -1 or end_index == -1 or end_index < begin_index:
        raise MarkerBlockError(f"Partial marker block for {marker}")
    return begin_index, end_index + len(end)


def upsert_marker_block(path: Path, marker: str, content: str) -> None:
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{_markers(marker)[0]}\n{content.rstrip()}\n{_markers(marker)[1]}"
    found = _find_marker_block(original, marker)
    if found is None:
        prefix = original
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix and not prefix.endswith("\n\n"):
            prefix += "\n"
        updated = prefix + block + "\n"
    else:
        start, stop = found
        updated = original[:start] + block + original[stop:]
        if not updated.endswith("\n"):
            updated += "\n"
    write_text_file(path, updated)


def remove_marker_block(path: Path, marker: str) -> None:
    if not path.exists():
        return
    original = path.read_text(encoding="utf-8")
    found = _find_marker_block(original, marker)
    if found is None:
        return
    start, stop = found
    updated = original[:start].rstrip() + "\n" + original[stop:].lstrip()
    if updated == "\n":
        updated = ""
    write_text_file(path, updated)
