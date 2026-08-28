"""Strict, deterministic formats shared by operator distribution workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import cast


def strict_json_object(raw: bytes) -> dict[str, object]:
    """Parse a JSON object while rejecting ambiguous JSON extensions."""
    value = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return cast(dict[str, object], value)


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Encode one mapping as the canonical JSON bytes used for digests."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a namespaced SHA-256 digest."""
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def safe_relative_path(value: str) -> PurePosixPath:
    """Return one non-empty, portable relative path or raise ``ValueError``."""
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("path must be a non-empty POSIX relative path")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("path must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("path must be a non-empty POSIX relative path")
    return path


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    del value
    raise ValueError("non-standard JSON number")
