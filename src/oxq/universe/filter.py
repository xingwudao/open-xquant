from __future__ import annotations

import operator
from dataclasses import dataclass, field

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
    mktdata: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.base, list):
            object.__setattr__(self, "base", tuple(self.base))
        if isinstance(self.filters, list):
            object.__setattr__(self, "filters", tuple(self.filters))

    def get_universe(self, as_of_date: str) -> UniverseSnapshot:
        if not self.mktdata:
            raise ValueError("FilterUniverse requires mktdata to get_universe")

        as_of = pd.Timestamp(as_of_date)
        survivors = []
        for symbol in self.base:
            if symbol not in self.mktdata:
                continue
            df = self.mktdata[symbol]
            valid = df[df.index <= as_of]
            if valid.empty:
                continue
            row = valid.iloc[-1]
            if all(_eval_filter(f, row) for f in self.filters):
                survivors.append(symbol)

        return UniverseSnapshot(
            as_of_date=as_of_date,
            symbols=tuple(survivors),
            source=self._build_source(),
            metadata={"base_count": len(self.base), "filtered_count": len(survivors)},
        )

    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]:
        return [self.get_universe(start), self.get_universe(end)]

    def _build_source(self) -> str:
        parts = [f"{f.column}{f.op}{f.value}" for f in self.filters]
        name_part = f":{self.name}" if self.name else ""
        return f"filter{name_part}:{','.join(parts)}"


def _eval_filter(f: Filter, row: pd.Series) -> bool:
    """Evaluate a single filter condition against one row."""
    return _OPS[f.op](row[f.column], f.value)
