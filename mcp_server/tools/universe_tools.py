"""Deprecated: tools moved to oxq.tools.universe. Kept for test compatibility."""

from __future__ import annotations

from oxq.tools.universe import universe_inspect as _inspect  # noqa: F401
from oxq.tools.universe import universe_set as _resolve  # noqa: F401


def register(mcp: object) -> None:
    """No-op: tools now registered via oxq.tools.registry."""
