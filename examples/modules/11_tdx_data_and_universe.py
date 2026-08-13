from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oxq.core.errors import DownloadError
from oxq.data.market import LocalMarketDataProvider
from oxq.data.providers import Downloader
from oxq.universe.static import StaticUniverse


@dataclass(frozen=True)
class TdxDataContext:
    market: LocalMarketDataProvider
    universe: StaticUniverse
    downloaded_paths: dict[str, Path]


def build_tdx_data_context(
    downloader: Downloader,
    symbols: list[str],
    start: str,
    end: str,
    dest_dir: Path,
) -> TdxDataContext:
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(symbols) != len(set(symbols)):
        raise ValueError("symbols must be unique")

    downloaded_paths = downloader.download_many(
        symbols, start, end, dest_dir=dest_dir
    )
    if set(downloaded_paths) != set(symbols):
        raise DownloadError(
            "downloader must return exactly one path for each requested symbol"
        )

    market = LocalMarketDataProvider(data_dir=dest_dir)
    universe = StaticUniverse(tuple(symbols), name="tdx-example")
    snapshot = universe.get_universe(as_of_date=end)
    for symbol in snapshot.symbols:
        market.get_bars(symbol, start, end)

    return TdxDataContext(
        market=market,
        universe=universe,
        downloaded_paths=downloaded_paths,
    )
