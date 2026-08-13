from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from oxq.core.errors import DownloadError
from oxq.data.loaders import resolve_data_dir
from oxq.data.market import LocalMarketDataProvider
from oxq.data.providers import Downloader
from oxq.universe.static import StaticUniverse

if TYPE_CHECKING:
    from examples.modules.pytdx_downloader import PyTdxDownloader
    from examples.modules.tdxquant_downloader import TdxQuantDownloader
elif __package__:
    from .pytdx_downloader import PyTdxDownloader
    from .tdxquant_downloader import TdxQuantDownloader
else:
    from pytdx_downloader import PyTdxDownloader
    from tdxquant_downloader import TdxQuantDownloader


@dataclass(frozen=True)
class TdxDataContext:
    market: LocalMarketDataProvider
    universe: StaticUniverse
    downloaded_paths: dict[str, Path]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("start", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("end", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--dest-dir", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TDX data and build an open-xquant Universe."
    )
    providers = parser.add_subparsers(dest="provider", required=True)

    pytdx = providers.add_parser("pytdx")
    _add_common_arguments(pytdx)
    pytdx.add_argument("--host", required=True)
    pytdx.add_argument("--port", type=int, default=7709)
    pytdx.add_argument("--timeout", type=float, default=5.0)
    pytdx.add_argument("--no-auto-adjust", action="store_true")

    tdxquant = providers.add_parser("tdxquant")
    _add_common_arguments(tdxquant)
    tdxquant.add_argument("--endpoint", default="http://127.0.0.1:17709/")
    tdxquant.add_argument(
        "--dividend-type", choices=("front", "none"), default="front"
    )
    tdxquant.add_argument("--timeout", type=float, default=10.0)
    return parser


def create_downloader(args: argparse.Namespace) -> Downloader:
    if args.provider == "pytdx":
        return PyTdxDownloader(
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            auto_adjust=not args.no_auto_adjust,
        )
    if args.provider == "tdxquant":
        return TdxQuantDownloader(
            endpoint=args.endpoint,
            dividend_type=args.dividend_type,
            timeout=args.timeout,
        )
    raise ValueError(f"unknown TDX provider: {args.provider}")


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
        bars = market.get_bars(symbol, start, end)
        if bars.empty:
            raise DownloadError(
                f"No downloaded bars for '{symbol}' from {start} to {end}."
            )

    return TdxDataContext(
        market=market,
        universe=universe,
        downloaded_paths=downloaded_paths,
    )


def _print_context(
    provider: str,
    context: TdxDataContext,
    start: str,
    end: str,
) -> None:
    snapshot = context.universe.get_universe(as_of_date=end)
    print(f"Provider: {provider}")
    print(f"Universe: {', '.join(snapshot.symbols)}")
    for symbol in snapshot.symbols:
        path = context.downloaded_paths[symbol]
        bars = context.market.get_bars(symbol, start, end)
        first = bars.index[0].date().isoformat()
        last = bars.index[-1].date().isoformat()
        print(f"{symbol}: {len(bars)} rows, {first} to {last}")
        print(f"  Data: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dest_dir = resolve_data_dir(args.dest_dir)
    context = build_tdx_data_context(
        create_downloader(args),
        args.symbols,
        args.start,
        args.end,
        dest_dir,
    )
    _print_context(args.provider, context, args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
