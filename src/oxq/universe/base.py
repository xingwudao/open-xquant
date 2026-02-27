from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Filter:
    """Single filter condition for universe screening."""

    column: str
    op: str
    value: float


@dataclass(frozen=True)
class UniverseSnapshot:
    """Immutable snapshot of a resolved universe."""

    symbols: tuple[str, ...]
    source: str
    metadata: dict[str, Any]
