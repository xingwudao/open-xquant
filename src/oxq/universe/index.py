"""IndexUniverse — resolve index constituents from data sources."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from oxq.universe.base import UniverseSnapshot


def _fetch_csindex(code: str) -> list[str]:
    """Fetch index constituents via akshare's csindex API."""
    ak = importlib.import_module("akshare")
    df = ak.index_stock_cons_csindex(symbol=code)
    col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
    return df[col].tolist()


# Built-in index registry: key -> {code, name, source, fetch_fn}
INDEX_REGISTRY: dict[str, dict[str, Any]] = {
    "csi300": {
        "code": "000300",
        "name": "沪深300",
        "source": "akshare",
        "fetch_fn": _fetch_csindex,
    },
    "csi500": {
        "code": "000905",
        "name": "中证500",
        "source": "akshare",
        "fetch_fn": _fetch_csindex,
    },
    "csi1000": {
        "code": "000852",
        "name": "中证1000",
        "source": "akshare",
        "fetch_fn": _fetch_csindex,
    },
    "sse50": {
        "code": "000016",
        "name": "上证50",
        "source": "akshare",
        "fetch_fn": _fetch_csindex,
    },
}


def register_index(
    key: str,
    code: str,
    name: str,
    fetch_fn: Callable[[str], list[str]],
    source: str = "custom",
) -> None:
    """Register a new index in the global registry."""
    INDEX_REGISTRY[key] = {
        "code": code,
        "name": name,
        "source": source,
        "fetch_fn": fetch_fn,
    }


def list_indexes() -> list[dict[str, str]]:
    """Return info for all registered indexes."""
    return [
        {"key": key, "code": info["code"], "name": info["name"], "source": info["source"]}
        for key, info in INDEX_REGISTRY.items()
    ]


@dataclass
class IndexUniverse:
    """Resolve index constituents from registry or custom fetch function.

    Usage:
        # From registry
        universe = IndexUniverse(key="csi300")

        # Custom fetch function
        universe = IndexUniverse(key="my_idx", fetch_fn=my_fetch)
    """

    key: str
    fetch_fn: Callable[[str], list[str]] | None = None
    _code: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if self.fetch_fn is None:
            if self.key not in INDEX_REGISTRY:
                msg = f"Unknown index '{self.key}'. Available: {sorted(INDEX_REGISTRY)}"
                raise ValueError(msg)
            entry = INDEX_REGISTRY[self.key]
            self.fetch_fn = entry["fetch_fn"]
            self._code = entry["code"]
        else:
            # Custom fetch_fn provided — check registry for code, default to key
            entry = INDEX_REGISTRY.get(self.key, {})
            self._code = entry.get("code", self.key)

    def get_universe(self, as_of_date: str) -> UniverseSnapshot:
        """Fetch current constituents for the index."""
        symbols = self.fetch_fn(self._code)
        return UniverseSnapshot(
            as_of_date=as_of_date,
            symbols=tuple(symbols),
            source=f"index:{self.key}",
            metadata={"index_key": self.key, "count": len(symbols)},
        )

    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]:
        """Return snapshots for start and end dates."""
        return [self.get_universe(start), self.get_universe(end)]
