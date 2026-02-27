from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oxq.universe.base import UniverseSnapshot


@dataclass(frozen=True)
class StaticUniverse:
    """Fixed symbol list, manually specified."""

    symbols: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.symbols, list):
            object.__setattr__(self, "symbols", tuple(self.symbols))

    def resolve(self, mktdata: dict[str, Any] | None = None) -> UniverseSnapshot:
        return UniverseSnapshot(
            symbols=self.symbols,
            source=f"static:{self.name}" if self.name else "static",
            metadata={},
        )
