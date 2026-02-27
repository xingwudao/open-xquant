from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from oxq.data.loaders import resolve_data_dir
from oxq.universe import Filter, FilterUniverse, StaticUniverse


def _resolve_static(symbols: list[str], name: str = "") -> dict[str, Any]:
    """Create and resolve a static universe."""
    universe = StaticUniverse(symbols=tuple(symbols), name=name)
    snapshot = universe.resolve()
    return {
        "symbols": list(snapshot.symbols),
        "count": len(snapshot.symbols),
        "source": snapshot.source,
    }


def _resolve_filter(
    base_symbols: list[str],
    filters: list[dict[str, Any]],
    name: str = "",
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Create and resolve a filter universe using local market data."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)

    # Load mktdata for base symbols
    mktdata: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in base_symbols:
        parquet_path = path / f"{sym}.parquet"
        if parquet_path.exists():
            mktdata[sym] = pd.read_parquet(parquet_path)
        else:
            missing.append(sym)

    # Parse filters
    filter_objs = tuple(
        Filter(column=f["column"], op=f["op"], value=f["value"])
        for f in filters
    )

    universe = FilterUniverse(base=tuple(base_symbols), filters=filter_objs, name=name)
    snapshot = universe.resolve(mktdata=mktdata)

    result: dict[str, Any] = {
        "symbols": list(snapshot.symbols),
        "count": len(snapshot.symbols),
        "source": snapshot.source,
        "base_count": snapshot.metadata.get("base_count", len(base_symbols)),
        "filtered_count": snapshot.metadata.get("filtered_count", len(snapshot.symbols)),
    }
    if missing:
        result["missing_data"] = missing
    return result


def _inspect(symbols: list[str], data_dir: str | None = None) -> dict[str, Any]:
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


def register(mcp: FastMCP) -> None:
    """Register universe tools on the MCP server instance."""

    @mcp.tool(
        name="universe_resolve_static",
        description=(
            "Create a static universe from a fixed list of symbols. "
            "Returns the symbol list as a resolved universe snapshot."
        ),
    )
    def universe_resolve_static(
        symbols: list[str],
        name: str = "",
    ) -> dict[str, Any]:
        return _resolve_static(symbols, name)

    @mcp.tool(
        name="universe_resolve_filter",
        description=(
            "Filter symbols from local market data by price/volume conditions. "
            "Each filter has a column (e.g. 'close', 'volume'), an operator (>=, <=, >, <, ==, !=), "
            "and a numeric value. All filters are AND-combined. "
            "Requires market data to be downloaded first via data_load_symbols."
        ),
    )
    def universe_resolve_filter(
        base_symbols: list[str],
        filters: list[dict[str, Any]],
        name: str = "",
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        return _resolve_filter(base_symbols, filters, name, data_dir)

    @mcp.tool(
        name="universe_inspect",
        description=(
            "Inspect symbols in a universe — shows data availability, "
            "date range, and latest price/volume for each symbol."
        ),
    )
    def universe_inspect(
        symbols: list[str],
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        return _inspect(symbols, data_dir)
