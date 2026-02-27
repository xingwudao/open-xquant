from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from oxq.data.loaders import Downloader, resolve_data_dir


def _list_symbols(data_dir: str | None = None) -> dict[str, Any]:
    """List locally available symbols."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    if not path.exists():
        return {"symbols": [], "count": 0, "data_dir": str(path)}
    symbols = sorted(p.stem for p in path.glob("*.parquet"))
    return {"symbols": symbols, "count": len(symbols), "data_dir": str(path)}


def _inspect(symbol: str, data_dir: str | None = None) -> dict[str, Any]:
    """Inspect a symbol's local data."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    parquet_path = path / f"{symbol}.parquet"
    if not parquet_path.exists():
        return {"symbol": symbol, "error": f"No data for '{symbol}'. Run data_load_symbols first."}
    df = pd.read_parquet(parquet_path)
    return {
        "symbol": symbol,
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
        "missing_values": int(df.isna().sum().sum()),
        "sample_head": df.head(3).reset_index().astype(str).to_dict(orient="records"),
    }


def _load_symbols(
    symbols: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Download symbols from an external source."""
    dest = Path(data_dir) if data_dir else None

    dl: Downloader
    if source == "yfinance":
        from oxq.data import YFinanceDownloader
        dl = YFinanceDownloader()
    elif source == "akshare":
        from oxq.data import AkShareDownloader
        dl = AkShareDownloader()
    else:
        return {"error": f"Unknown source '{source}'. Use 'yfinance' or 'akshare'."}

    rows: dict[str, int] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            dl.download(sym, start, end, dest_dir=dest)
            df = pd.read_parquet(resolve_data_dir(dest) / f"{sym}.parquet")
            rows[sym] = len(df)
        except Exception as e:
            errors[sym] = str(e)

    result: dict[str, Any] = {
        "symbols": list(rows.keys()),
        "rows": rows,
        "data_dir": str(resolve_data_dir(dest)),
    }
    if errors:
        result["errors"] = errors
    return result


def register(mcp: FastMCP) -> None:
    """Register data tools on the MCP server instance."""

    @mcp.tool(name="data_load_symbols", description="Download market data for given symbols")
    def data_load_symbols(
        symbols: list[str],
        start: str,
        end: str,
        source: str = "yfinance",
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        return _load_symbols(symbols, start, end, source, data_dir)

    @mcp.tool(name="data_list_symbols", description="List locally available market data symbols")
    def data_list_symbols(data_dir: str | None = None) -> dict[str, Any]:
        return _list_symbols(data_dir)

    @mcp.tool(name="data_inspect", description="Inspect data summary for a symbol (rows, date range, missing values)")
    def data_inspect(symbol: str, data_dir: str | None = None) -> dict[str, Any]:
        return _inspect(symbol, data_dir)
