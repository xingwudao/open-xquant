from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Filter:
    """Single filter condition for universe screening."""

    column: str
    op: str
    value: float


@dataclass(frozen=True)
class UniverseSnapshot:
    """Immutable snapshot of a resolved universe."""

    as_of_date: str
    symbols: tuple[str, ...]
    source: str
    metadata: dict[str, Any]


@runtime_checkable
class UniverseProvider(Protocol):
    """Interface for universe resolution."""

    def get_universe(self, as_of_date: str) -> UniverseSnapshot: ...

    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]: ...
