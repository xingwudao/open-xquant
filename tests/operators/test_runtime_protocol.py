"""Protocol boundaries for exact-wheel operator children."""

from __future__ import annotations

import pytest


def test_canonical_protocol_bytes_are_deterministic_and_reject_non_json() -> None:
    from oxq.operators.runtime_protocol import canonical_protocol_bytes

    assert canonical_protocol_bytes({"b": 2, "a": [True, None]}) == b'{"a":[true,null],"b":2}'
    with pytest.raises(ValueError):
        canonical_protocol_bytes({"value": float("nan")})
