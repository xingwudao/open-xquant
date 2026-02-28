"""Tool registry for oxq SDK tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolDef:
    """Definition of a registered tool."""

    name: str
    description: str
    fn: Callable[..., dict[str, Any]]


class ToolRegistry:
    """Central registry for SDK tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def tool(self, name: str, description: str) -> Callable[..., Any]:
        """Decorator to register a tool function."""

        def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
            self._tools[name] = ToolDef(name=name, description=description, fn=fn)
            return fn

        return decorator

    def all_tools(self) -> list[ToolDef]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get(self, name: str) -> ToolDef:
        """Get a tool by name."""
        return self._tools[name]


# Module-level singleton — tool modules import this directly.
registry = ToolRegistry()
