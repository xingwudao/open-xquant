"""Universe tools — set, inspect, list indexes, and history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oxq.data.loaders import resolve_data_dir
from oxq.tools.registry import registry
from oxq.universe import Filter, FilterUniverse, StaticUniverse


@registry.tool(
    name="universe_set",
    description=(
        "Create a universe from symbols. "
        "type='static' for a fixed list, type='filter' to screen by conditions."
    ),
)
def universe_set(
    type: str,
    symbols: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    code: str | None = None,
    name: str = "",
    as_of_date: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Create and resolve a universe."""
    if type == "static":
        if not symbols:
            return {"error": "type='static' requires 'symbols' list."}
        universe = StaticUniverse(symbols=tuple(symbols), name=name)
        date = as_of_date or _latest_date(symbols, data_dir)
        snapshot = universe.get_universe(date)
        return {
            "symbols": list(snapshot.symbols),
            "count": len(snapshot.symbols),
            "source": snapshot.source,
            "as_of_date": snapshot.as_of_date,
        }

    if type == "filter":
        if not symbols:
            return {"error": "type='filter' requires 'symbols' (base symbols) list."}
        if not filters:
            return {"error": "type='filter' requires 'filters' list."}
        path = resolve_data_dir(Path(data_dir) if data_dir else None)
        mktdata: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        for sym in symbols:
            parquet_path = path / f"{sym}.parquet"
            if parquet_path.exists():
                mktdata[sym] = pd.read_parquet(parquet_path)
            else:
                missing.append(sym)
        filter_objs = tuple(
            Filter(column=f["column"], op=f["op"], value=f["value"]) for f in filters
        )
        filter_universe = FilterUniverse(
            base=tuple(symbols), filters=filter_objs, mktdata=mktdata, name=name,
        )
        date = as_of_date or _latest_date(symbols, data_dir)
        snapshot = filter_universe.get_universe(date)
        result: dict[str, Any] = {
            "symbols": list(snapshot.symbols),
            "count": len(snapshot.symbols),
            "source": snapshot.source,
            "as_of_date": snapshot.as_of_date,
            "base_count": snapshot.metadata.get("base_count", len(symbols)),
            "filtered_count": snapshot.metadata.get("filtered_count", len(snapshot.symbols)),
        }
        if missing:
            result["missing_data"] = missing
        return result

    if type == "index":
        return {"error": "type='index' is planned for Phase 2. Use universe_list_indexes for available indexes."}

    return {"error": f"Unknown type '{type}'. Use 'static', 'filter', or 'index'."}


@registry.tool(
    name="universe_list_indexes",
    description="List available index-based universes (e.g. S&P 500, CSI 300).",
)
def universe_list_indexes() -> dict[str, Any]:
    """List available index universes."""
    return {
        "indexes": [],
        "note": "Index-based universes are planned for Phase 2.",
    }


@registry.tool(
    name="universe_inspect",
    description=(
        "Inspect symbols in a universe — shows data availability, "
        "date range, and latest price/volume for each symbol."
    ),
)
def universe_inspect(
    symbols: list[str],
    as_of_date: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Inspect data availability for universe symbols."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)

    details: list[dict[str, Any]] = []
    available = 0
    for sym in symbols:
        parquet_path = path / f"{sym}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            available += 1
            details.append({
                "symbol": sym,
                "has_data": True,
                "rows": len(df),
                "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
                "latest_close": float(df["close"].iloc[-1]) if "close" in df.columns else None,
                "latest_volume": int(df["volume"].iloc[-1]) if "volume" in df.columns else None,
            })
        else:
            details.append({
                "symbol": sym,
                "has_data": False,
            })

    return {
        "total": len(symbols),
        "available": available,
        "missing": len(symbols) - available,
        "data_dir": str(path),
        "details": details,
    }


@registry.tool(
    name="universe_history",
    description="Get universe snapshots over a date range.",
)
def universe_history(
    symbols: list[str],
    start: str,
    end: str,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Get universe snapshots for start and end dates."""
    universe = StaticUniverse(symbols=tuple(symbols))
    history = universe.get_history(start, end)
    return {
        "snapshots": [
            {
                "as_of_date": snap.as_of_date,
                "symbols": list(snap.symbols),
                "count": len(snap.symbols),
                "source": snap.source,
            }
            for snap in history
        ],
    }


def _latest_date(symbols: list[str], data_dir: str | None) -> str:
    """Find the latest date from local data files, or return today."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    latest = None
    for sym in symbols:
        parquet_path = path / f"{sym}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if not df.empty:
                dt = df.index[-1]
                if latest is None or dt > latest:
                    latest = dt
    if latest is not None:
        return str(latest.date())
    from datetime import date
    return date.today().isoformat()
