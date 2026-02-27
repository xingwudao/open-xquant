from __future__ import annotations

import operator
from dataclasses import dataclass

import pandas as pd

from oxq.universe.base import Filter, UniverseSnapshot

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class FilterUniverse:
    """Rule-based universe screening on mktdata."""

    base: tuple[str, ...]
    filters: tuple[Filter, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.base, list):
            object.__setattr__(self, "base", tuple(self.base))
        if isinstance(self.filters, list):
            object.__setattr__(self, "filters", tuple(self.filters))

    def resolve(self, mktdata: dict[str, pd.DataFrame] | None = None) -> UniverseSnapshot:
        if mktdata is None:
            raise ValueError("FilterUniverse requires mktdata to resolve")

        survivors = []
        for symbol in self.base:
            if symbol not in mktdata:
                continue
            row = mktdata[symbol].iloc[-1]
            if all(_eval_filter(f, row) for f in self.filters):
                survivors.append(symbol)

        return UniverseSnapshot(
            symbols=tuple(survivors),
            source=self._build_source(),
            metadata={"base_count": len(self.base), "filtered_count": len(survivors)},
        )

    def _build_source(self) -> str:
        parts = [f"{f.column}{f.op}{f.value}" for f in self.filters]
        name_part = f":{self.name}" if self.name else ""
        return f"filter{name_part}:{','.join(parts)}"


def _eval_filter(f: Filter, row: pd.Series) -> bool:
    """Evaluate a single filter condition against one row."""
    return _OPS[f.op](row[f.column], f.value)
