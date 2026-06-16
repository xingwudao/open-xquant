"""Shared operator resolution for comparison-style signals."""

from __future__ import annotations

import operator
from collections.abc import Callable

_OPS: dict[str, Callable] = {
    "gt": operator.gt,
    "lt": operator.lt,
    "gte": operator.ge,
    "lte": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}

_ALIASES: dict[str, str] = {
    ">": "gt",
    "<": "lt",
    ">=": "gte",
    "<=": "lte",
    "==": "eq",
    "!=": "ne",
    "<>": "ne",
    "=": "eq",
    "above": "gt",
    "below": "lt",
}


def resolve_op(relationship: str) -> Callable:
    """Return the operator function for *relationship*, accepting aliases."""
    key = relationship.strip()
    canonical = _ALIASES.get(key, key)
    op = _OPS.get(canonical)
    if op is None:
        valid = sorted(_OPS) + sorted(_ALIASES)
        raise ValueError(
            f"Unknown relationship {relationship!r}. "
            f"Valid values: {valid}"
        )
    return op
