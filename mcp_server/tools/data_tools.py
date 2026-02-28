"""Deprecated: tools moved to oxq.tools.data. Kept for test compatibility."""

from __future__ import annotations

from oxq.tools.data import inspect_symbol as _inspect  # noqa: F401
from oxq.tools.data import list_symbols as _list_symbols  # noqa: F401
from oxq.tools.data import load_symbols as _load_symbols  # noqa: F401


def register(mcp: object) -> None:
    """No-op: tools now registered via oxq.tools.registry."""
