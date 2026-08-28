"""Protocol boundaries for exact-wheel operator children."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from types import TracebackType

import pytest

from oxq.operators import runtime_protocol
from oxq.operators.runtime_protocol import run_exact_wheel_request


def test_canonical_protocol_bytes_are_deterministic_and_reject_non_json() -> None:
    from oxq.operators.runtime_protocol import canonical_protocol_bytes

    assert canonical_protocol_bytes({"b": 2, "a": [True, None]}) == b'{"a":[true,null],"b":2}'
    with pytest.raises(ValueError):
        canonical_protocol_bytes({"value": float("nan")})


def test_active_response_reader_bounds_oversized_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    response = tmp_path / "response.json"
    sizes: list[int] = []

    class OversizedResponse:
        def __enter__(self) -> OversizedResponse:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc_value, traceback

        def read(self, size: int = -1) -> bytes:
            sizes.append(size)
            if size < 0:
                pytest.fail("active response reader attempted an unbounded read")
            return b"x" * size

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: OversizedResponse())
    with pytest.raises(ValueError, match="exact-wheel child response is invalid"):
        runtime_protocol._read_authenticated_response(response, 0, b"x" * 32)

    assert sizes == [1024 * 1024 + 1]


def test_exact_wheel_request_rejects_test_runtime_paths_in_wire_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "provider.whl"
    wheel.write_bytes(b"placeholder")

    def fail_if_child_starts(*args: object, **kwargs: object) -> int:
        del args, kwargs
        pytest.fail("test runtime paths from request payload reached the child")

    monkeypatch.setattr(runtime_protocol, "run_contained_child", fail_if_child_starts)

    with pytest.raises(ValueError, match="test_runtime_paths"):
        run_exact_wheel_request(
            {
                "module": "provider",
                "callable": "sma",
                "parameters": {},
                "input": {},
                "output_fields": [],
                "output_alignment": "preserve_input_order",
                "test_runtime_paths": [],
            },
            [wheel],
            timeout_seconds=1,
        )


def test_exact_wheel_request_clears_inherited_test_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "provider.whl"
    wheel.write_bytes(b"placeholder")
    monkeypatch.setenv("OXQ_EXACT_TEST_RUNTIME", "1")
    monkeypatch.setenv("OXQ_EXACT_TEST_RUNTIME_PATHS", str(tmp_path))

    def fake_child(
        command: list[str],
        *,
        timeout_seconds: float,
        response_secret: bytes | None,
        environment: dict[str, str],
    ) -> int:
        del timeout_seconds
        assert "OXQ_EXACT_TEST_RUNTIME" not in environment
        assert "OXQ_EXACT_TEST_RUNTIME_PATHS" not in environment
        assert response_secret is not None
        response = {"status": "error", "code": "provider_import_failed"}
        payload = {
            **response,
            "auth": "hmac-sha256:"
            + hmac.new(
                response_secret,
                runtime_protocol.canonical_protocol_bytes(response),
                hashlib.sha256,
            ).hexdigest(),
        }
        Path(command[-1]).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    monkeypatch.setattr(runtime_protocol, "run_contained_child", fake_child)

    assert run_exact_wheel_request(
        {
            "module": "provider",
            "callable": "sma",
            "parameters": {},
            "input": {},
            "output_fields": [],
            "output_alignment": "preserve_input_order",
        },
        [wheel],
        timeout_seconds=1,
    ) == {"status": "error", "code": "provider_import_failed"}
