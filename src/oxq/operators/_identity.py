"""Identity validation shared by public operator boundaries."""

from __future__ import annotations

import re

_CANONICAL_OPERATOR_ID_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?)+$"
)


def is_canonical_operator_id(value: str) -> bool:
    return _CANONICAL_OPERATOR_ID_RE.fullmatch(value) is not None
