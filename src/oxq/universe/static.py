from __future__ import annotations

from dataclasses import dataclass

from oxq.universe.base import UniverseSnapshot


@dataclass(frozen=True)
class StaticUniverse:
    """Fixed symbol list, manually specified."""

    symbols: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.symbols, list):
            object.__setattr__(self, "symbols", tuple(self.symbols))

    def get_universe(self, as_of_date: str) -> UniverseSnapshot:
        return UniverseSnapshot(
            as_of_date=as_of_date,
            symbols=self.symbols,
            source=f"static:{self.name}" if self.name else "static",
            metadata={},
        )

    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]:
        return [self.get_universe(start)]
