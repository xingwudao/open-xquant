"""Protocol boundaries for exact-wheel operator children."""

from __future__ import annotations

import pytest
from pathlib import Path

from oxq.operators import runtime_protocol


def test_canonical_protocol_bytes_are_deterministic_and_reject_non_json() -> None:
    from oxq.operators.runtime_protocol import canonical_protocol_bytes

    assert canonical_protocol_bytes({"b": 2, "a": [True, None]}) == b'{"a":[true,null],"b":2}'
    with pytest.raises(ValueError):
        canonical_protocol_bytes({"value": float("nan")})


def test_active_response_reader_bounds_oversized_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    response = tmp_path / "response.json"
    response.write_bytes(b"x" * (1024 * 1024 + 1))
    real_open = Path.open
    sizes: list[int] = []

    class Reader:
        def __enter__(self): return self
        def __exit__(self, *args: object): self.stream.close()
        def read(self, size: int) -> bytes: sizes.append(size); return self.stream.read(size)
        def __init__(self, stream: object): self.stream = stream
    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: Reader(real_open(path, *a, **kw)))
    with pytest.raises(ValueError):
        runtime_protocol._read_authenticated_response(response, 0, b"x" * 32)
    assert sizes == [1024 * 1024 + 1]
