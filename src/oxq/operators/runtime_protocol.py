"""Authenticated parent-side protocol for exact-wheel operator children."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from oxq.operators.child_process import run_contained_child

_MAX_RESPONSE_BYTES = 1024 * 1024
_POLICY_EXIT_CODE = 86


def canonical_protocol_bytes(value: object) -> bytes:
    """Encode a protocol value as strict, canonical JSON bytes."""
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("protocol value is not strict JSON") from exc


def run_exact_wheel_request(
    request: Mapping[str, object],
    wheel_snapshots: Sequence[str | Path],
    *,
    timeout_seconds: float,
    _test_runtime_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Execute a request in a contained ``-I -S`` exact-wheel child.

    The caller supplies immutable snapshot paths.  Their paths are bound into
    the child request only after the request itself has been canonicalized.
    """
    paths = [str(Path(path).resolve(strict=True)) for path in wheel_snapshots]
    if not paths:
        raise ValueError("exact wheel closure is empty")
    if "test_runtime_paths" in request:
        raise ValueError("test_runtime_paths is not a production request field")
    payload = dict(request)
    payload["implementation_artifact"] = paths[0]
    payload["dependency_artifacts"] = paths[1:]
    with TemporaryDirectory(prefix="oxq-exact-request-") as directory, TemporaryDirectory(
        prefix="oxq-exact-response-"
    ) as response_directory:
        request_path = Path(directory) / "request.json"
        response_path = Path(response_directory) / "response.json"
        request_path.write_bytes(canonical_protocol_bytes(payload))
        secret = secrets.token_bytes(32)
        environment = dict(os.environ)
        environment.pop("OXQ_EXACT_TEST_RUNTIME", None)
        environment.pop("OXQ_EXACT_TEST_RUNTIME_PATHS", None)
        if _test_runtime_paths:
            environment["OXQ_EXACT_TEST_RUNTIME"] = "1"
            environment["OXQ_EXACT_TEST_RUNTIME_PATHS"] = os.pathsep.join(
                str(Path(item).resolve(strict=True)) for item in _test_runtime_paths
            )
        returncode = run_contained_child(
            [sys.executable, "-I", "-S", str(_child_path()), str(request_path), str(response_path)],
            timeout_seconds=timeout_seconds,
            response_secret=secret,
            environment=environment,
        )
        return _read_authenticated_response(response_path, returncode, secret)


def _child_path() -> Path:
    return Path(__file__).with_name("_exact_wheel_child.py").resolve(strict=True)


def _read_authenticated_response(path: Path, returncode: int, secret: bytes) -> dict[str, object]:
    if returncode == _POLICY_EXIT_CODE:
        return {"status": "error", "code": "provider_import_failed"}
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("response is too large")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
        if returncode != 0 or not isinstance(value, dict):
            raise ValueError("invalid response")
        auth = value.pop("auth", None)
        expected = "hmac-sha256:" + hmac.new(secret, canonical_protocol_bytes(value), hashlib.sha256).hexdigest()
        if not isinstance(auth, str) or not hmac.compare_digest(auth, expected):
            raise ValueError("response authentication failed")
        status = value.get("status")
        if status == "ok" and set(value) == {"status", "outputs", "repeated_outputs"}:
            return cast(dict[str, object], value)
        if status == "error" and set(value) == {"status", "code"}:
            return cast(dict[str, object], value)
        raise ValueError("invalid response fields")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("exact-wheel child response is invalid") from exc


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    del value
    raise ValueError("non-standard JSON number")
