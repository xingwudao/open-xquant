"""Macro factor data: download from World Bank and read locally."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from oxq.core.errors import DownloadError

# Human-readable name → World Bank indicator code
INDICATOR_MAP: dict[str, str] = {
    "gdp": "NY.GDP.MKTP.CD",  # GDP (current USD)
    "gdp_per_capita": "NY.GDP.PCAP.CD",  # GDP per capita (current USD)
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
    "cpi": "FP.CPI.TOTL.ZG",  # CPI inflation (annual %)
}


def resolve_factor_dir(dest_dir: Path | None = None) -> Path:
    """Resolve factor data directory.

    Priority: parameter > $OXQ_DATA_DIR/factor > ~/.oxq/data/factor.
    """
    if dest_dir is not None:
        return dest_dir
    env = os.environ.get("OXQ_DATA_DIR")
    if env:
        return Path(env) / "factor"
    return Path.home() / ".oxq" / "data" / "factor"


def _fetch_world_bank(
    indicator_code: str,
    countries: list[str],
    start_year: int,
    end_year: int,
    timeout: int = 60,
    retries: int = 3,
) -> list[dict[str, Any]]:
    """Fetch data from World Bank API v2. Returns raw JSON records."""
    import time

    country_str = ";".join(countries)
    url = (
        f"https://api.worldbank.org/v2/country/{country_str}"
        f"/indicator/{indicator_code}"
        f"?date={start_year}:{end_year}&format=json&per_page=10000"
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                body = json.loads(resp.read().decode())
            # World Bank returns [metadata, data] — data is the second element
            if not isinstance(body, list) or len(body) < 2 or body[1] is None:
                return []
            result: list[dict[str, Any]] = body[1]
            return result
        except (TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert World Bank JSON records to a DataFrame (index=year, cols=countries)."""
    rows: dict[int, dict[str, float | None]] = {}
    for rec in records:
        year = int(rec["date"])
        country = rec["countryiso3code"]
        value = rec["value"]
        if year not in rows:
            rows[year] = {}
        rows[year][country] = float(value) if value is not None else None

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "year"
    df = df.sort_index()
    # Reorder columns alphabetically for consistency
    df = df.reindex(sorted(df.columns), axis=1)
    return df


class WorldBankDownloader:
    """Download macro indicators from World Bank Open Data API."""

    def download(
        self,
        indicator: str,
        countries: list[str],
        start_year: int = 2000,
        end_year: int = 2024,
        dest_dir: Path | None = None,
    ) -> Path:
        """Download indicator data and save as parquet.

        Parameters
        ----------
        indicator : str
            Human-readable indicator name (e.g. "gdp", "cpi").
        countries : list[str]
            ISO 3166-1 alpha-3 country codes (e.g. ["USA", "CHN"]).
        start_year : int
            Start year (inclusive).
        end_year : int
            End year (inclusive).
        dest_dir : Path | None
            Override storage directory.

        Returns
        -------
        Path
            Path to the saved parquet file.
        """
        if indicator not in INDICATOR_MAP:
            msg = (
                f"Unknown indicator '{indicator}'. "
                f"Available: {sorted(INDICATOR_MAP)}"
            )
            raise ValueError(msg)

        indicator_code = INDICATOR_MAP[indicator]

        try:
            records = _fetch_world_bank(
                indicator_code, countries, start_year, end_year
            )
        except Exception as exc:
            msg = f"Failed to download '{indicator}' from World Bank: {exc}"
            raise DownloadError(msg) from exc

        if not records:
            msg = (
                f"No data returned for '{indicator}' "
                f"(countries={countries}, {start_year}-{end_year})."
            )
            raise DownloadError(msg)

        df = _records_to_dataframe(records)

        factor_dir = resolve_factor_dir(dest_dir)
        factor_dir.mkdir(parents=True, exist_ok=True)
        path = factor_dir / f"{indicator}.parquet"
        df.to_parquet(path)
        return path


def read_factor(
    indicator: str,
    countries: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Read local factor data.

    Parameters
    ----------
    indicator : str
        Factor name (e.g. "gdp").
    countries : list[str] | None
        Filter to these countries. None returns all available.
    start_year : int | None
        Filter start year (inclusive).
    end_year : int | None
        Filter end year (inclusive).
    data_dir : Path | None
        Override factor data directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with index=year (int), columns=country codes.
    """
    factor_dir = resolve_factor_dir(data_dir)
    path = factor_dir / f"{indicator}.parquet"
    if not path.exists():
        msg = f"Factor file not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(path)

    if countries is not None:
        available = [c for c in countries if c in df.columns]
        df = df[available]

    if start_year is not None:
        df = df[df.index >= start_year]
    if end_year is not None:
        df = df[df.index <= end_year]

    return df
