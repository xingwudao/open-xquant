"""Protocol boundaries for exact-wheel operator children."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pytest

from oxq.operators import runtime_protocol


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
