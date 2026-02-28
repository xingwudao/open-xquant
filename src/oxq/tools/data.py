"""Data tools — list, inspect, and load market/factor data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oxq.data.factors import INDICATOR_MAP, resolve_factor_dir
from oxq.data.loaders import Downloader, resolve_data_dir
from oxq.tools.registry import registry


@registry.tool(
    name="data_list_symbols",
    description="List locally available market data symbols",
)
def list_symbols(data_dir: str | None = None) -> dict[str, Any]:
    """List locally available symbols."""
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    if not path.exists():
        return {"symbols": [], "count": 0, "data_dir": str(path)}
    symbols = sorted(p.stem for p in path.glob("*.parquet"))
    return {"symbols": symbols, "count": len(symbols), "data_dir": str(path)}


@registry.tool(
    name="data_inspect",
    description="Inspect data summary for a symbol (rows, date range, missing values)",
)
def inspect_symbol(symbol: str, data_dir: str | None = None) -> dict[str, Any]:
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


@registry.tool(
    name="data_load_symbols",
    description="Download market data for given symbols",
)
def load_symbols(
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


# ---------------------------------------------------------------------------
# Factor tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="factor_download",
    description="Download a macro indicator (gdp, gdp_per_capita, gdp_growth, cpi) from World Bank",
)
def factor_download(
    indicator: str,
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2024,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Download a macro indicator from World Bank and save locally."""
    from oxq.data.factors import WorldBankDownloader

    dest = Path(data_dir) if data_dir else None
    dl = WorldBankDownloader()
    try:
        path = dl.download(indicator, countries, start_year, end_year, dest_dir=dest)
    except (ValueError, Exception) as exc:
        return {"error": str(exc)}

    df = pd.read_parquet(path)
    return {
        "indicator": indicator,
        "countries": list(df.columns),
        "year_range": [int(df.index.min()), int(df.index.max())],
        "rows": len(df),
        "path": str(path),
    }


@registry.tool(
    name="factor_list",
    description="List locally available factor files",
)
def factor_list(data_dir: str | None = None) -> dict[str, Any]:
    """List locally available factor data files."""
    path = resolve_factor_dir(Path(data_dir) if data_dir else None)
    if not path.exists():
        return {"factors": [], "count": 0, "data_dir": str(path)}
    factors = sorted(p.stem for p in path.glob("*.parquet"))
    return {"factors": factors, "count": len(factors), "data_dir": str(path)}


@registry.tool(
    name="factor_inspect",
    description="Inspect a factor file (year range, countries, sample values)",
)
def factor_inspect(
    indicator: str,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Inspect a locally stored factor file."""
    path = resolve_factor_dir(Path(data_dir) if data_dir else None)
    parquet_path = path / f"{indicator}.parquet"
    if not parquet_path.exists():
        return {
            "indicator": indicator,
            "error": f"No data for '{indicator}'. Run factor_download first.",
            "available_indicators": sorted(INDICATOR_MAP),
        }
    df = pd.read_parquet(parquet_path)
    return {
        "indicator": indicator,
        "countries": list(df.columns),
        "year_range": [int(df.index.min()), int(df.index.max())],
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "sample": df.tail(3).reset_index().astype(str).to_dict(orient="records"),
    }
