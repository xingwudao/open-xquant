# TDX Downloader Module And Universe Example Design

## Goal

Move both TDX downloader examples directly into `examples/modules/` and add
one complete module example that downloads data, reopens it through
open-xquant, and constructs a `StaticUniverse` that downstream components can
consume.

This work is an incremental branch based on commit `078556f`. It does not add
either downloader to the SDK, dependency set, provider enum, CLI, or doctor.

## File Layout

The downloader implementations keep importable, unnumbered file names:

- `examples/modules/pytdx_downloader.py`
- `examples/modules/tdxquant_downloader.py`

The executable end-to-end module example follows the existing numbered
learning sequence:

- `examples/modules/11_tdx_data_and_universe.py`

The complete `examples/custom_data_sources/` directory is removed. No new
README or TDX-specific subdirectory is created.

Tests remain under `tests/examples/` and gain a focused test module for the
end-to-end workflow.

## Source-Level Documentation

The useful content from `examples/custom_data_sources/README.md` and
`examples/custom_data_sources/PYTDX.md` moves into module docstrings instead
of another Markdown file.

`pytdx_downloader.py` documents:

- direct TCP connection without installing or starting the desktop client;
- explicit server ownership and access requirements;
- `pytdx==1.72` as an optional, archived dependency;
- supported symbols and daily bars;
- yfinance-style ratio adjustment semantics and limitations;
- Python and CLI examples;
- output files, operational limitations, licensing, and data-use boundaries.

`tdxquant_downloader.py` documents:

- the official loopback HTTP transport;
- the supported TdxQuant desktop client and post-market data prerequisites;
- front-adjusted and unadjusted modes;
- Python and CLI examples;
- output files, operational limitations, licensing, and data-use boundaries.

The install hint and all commands use the new `examples/modules/` paths.

## Complete Workflow

`11_tdx_data_and_universe.py` is orchestration code, not a third downloader.
It accepts either `pytdx` or `tdxquant` as the backend and a non-empty list of
symbols.

The workflow is:

1. Construct the selected object through the `Downloader` protocol.
2. Call `download_many(symbols, start, end, dest_dir)`.
3. Construct `LocalMarketDataProvider(data_dir=dest_dir)`.
4. Construct `StaticUniverse(tuple(symbols), name="tdx-example")`.
5. Obtain a snapshot at the inclusive end date.
6. Read every snapshot symbol back through `LocalMarketDataProvider.get_bars`.
7. Print the backend, downloaded paths, universe source and symbols, and each
   frame's row count and actual date range.

The reusable seam is:

```python
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
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("symbols must be non-empty and unique")
    paths = downloader.download_many(symbols, start, end, dest_dir)
    if set(paths) != set(symbols):
        raise DownloadError("downloader returned unexpected symbol paths")
    market = LocalMarketDataProvider(data_dir=dest_dir)
    universe = StaticUniverse(tuple(symbols), name="tdx-example")
    snapshot = universe.get_universe(as_of_date=end)
    for symbol in snapshot.symbols:
        market.get_bars(symbol, start, end)
    return TdxDataContext(market, universe, paths)
```

Downstream examples or user code can pass `context.market` and
`context.universe` to later components without knowing which TDX transport
created the Parquet files.

## Command-Line Interface

The common arguments are positional `provider`, `start`, and `end`, plus a
required multi-value `--symbols` option and optional `--dest-dir` and
`--timeout` options.

PyTdx requires an explicit `--host` and supports `--port` and
`--no-auto-adjust`:

```bash
uv run --with pytdx==1.72 python \
  examples/modules/11_tdx_data_and_universe.py \
  pytdx 2020-05-01 2026-01-01 \
  --symbols 510300.SH 159915.SZ \
  --host YOUR_TDX_HOST --port 7709 \
  --dest-dir data/market
```

TdxQuant supports `--endpoint` and `--dividend-type`:

```bash
uv run python examples/modules/11_tdx_data_and_universe.py \
  tdxquant 2024-01-01 2024-12-31 \
  --symbols 510300.SH 159915.SZ \
  --endpoint http://127.0.0.1:17709/ \
  --dividend-type front \
  --dest-dir data/market
```

Backend-inapplicable flags fail explicitly instead of being silently ignored.
The example does not include a server list, discover servers, or start a
desktop client.

## Validation And Failure Behavior

`build_tdx_data_context` rejects an empty or duplicate symbol list before any
network request. It delegates symbol and date validation to the selected
downloader, then verifies that `download_many` returned exactly one path for
each requested symbol. It reopens every symbol through the local market data
provider so a malformed or missing Parquet file fails before a context is
reported as ready.

Backend construction preserves existing validation:

- PyTdx requires an explicit host and exact optional dependency version.
- TdxQuant permits only a loopback HTTP endpoint on port `17709`.
- Network errors remain `DownloadError` values from the downloader modules.

There is no automatic fallback from one backend to the other. This keeps
transport choice and data provenance explicit.

## Tests

The change follows an import-and-execution RED/GREEN cycle:

1. Add tests that import both downloaders from `examples.modules` and execute
   each new script path with `--help`.
2. Run them and observe failure because the new imports and executable paths do
   not exist yet.
3. Move the files and update all imports, patches, commands, and install hints.
4. Run them again and observe successful imports and exit code zero.

Tests do not assert that the old source directory is absent. Removing it is a
reviewed repository-layout requirement, while the automated contract covers
the user-visible import and execution behavior that depends on the new layout.

Workflow tests use a fake `Downloader` that writes two small, valid Parquet
files. They verify:

- the exact `download_many` call;
- downloaded-path coverage;
- `StaticUniverse` symbol order and source;
- snapshot creation at the end date;
- successful readback through `LocalMarketDataProvider`;
- rejection of empty, duplicate, missing, and extra download results;
- selection and validation of both CLI backends without opening real sockets.

Existing downloader test suites continue to cover transport, decoding,
pagination, adjustment, and error behavior after import paths are changed.
The real PyTdx smoke test remains manual because it depends on an external
server and data permission.

## User-Facing Index

The Chinese and English module-example lists in the repository `README.md`
gain `11_tdx_data_and_universe.py`, described as a selectable TDX downloader,
local market-data readback, and Universe construction example. The index does
not describe either downloader as a built-in SDK provider.

## Non-Goals

- No SDK data-source registration.
- No new mandatory or optional project dependency group.
- No TDX server discovery, benchmarking, failover, or credentials.
- No automatic desktop-client startup.
- No strategy, signal, optimizer, or backtest execution in this example.
- No production guarantees for synchronization, rate limiting, or atomic
  multi-file publication.
