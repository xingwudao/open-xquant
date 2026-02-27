from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd


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


@runtime_checkable
class UniverseProvider(Protocol):
    """Interface for universe resolution."""

    def resolve(self, mktdata: dict[str, pd.DataFrame] | None = None) -> UniverseSnapshot: ...
