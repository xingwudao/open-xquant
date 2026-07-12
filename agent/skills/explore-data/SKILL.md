---
name: explore-data
description: >-
  Inspect, download, and validate open-xquant market, macro, and financial
  data; use before backtests, factor studies, or any task that depends on local
  parquet data.
---

## Phase Path Containment Preflight

Before any phase artifact read, write, directory creation, command, or handoff,
run this preflight completely and block on any failure:

1. Read `.open-xquant/workspace.yaml`. Resolve `version_root` from
   `paths.versions_dir`, using `versions` only when that key is absent.
   Require `version_root` to be a safe workspace-relative path: reject an
   absolute path or any `..` path segment, resolve it canonically with
   symlinks, and require the result to stay inside the workspace.
2. Read `current.json`. For normal phase work, set `expected_version_id` to
   `current.json.active_version`; it must exist. Only a contract that
   explicitly owns cross-version inspection may instead set
   `expected_version_id` from the referenced version id for that historical
   read. This exception never permits active-version work to consume another
   version.
3. Set the intended version directory to
   `<version_root>/<expected_version_id>/` and resolve it canonically. Require
   the intended version directory to remain inside the canonical version root
   and workspace; otherwise treat it as a symlink escape. Read
   `version_manifest.json` only from that exact directory. The manifest
   `version_id` must equal `expected_version_id`; for normal phase work it
   therefore must also equal `current.json.active_version`.
4. Before using each required `phase_paths` value, require a non-empty
   workspace-relative string. Reject an absolute path and any `..` path
   segment. Resolve `<workspace>/<phase_path>` canonically, including existing
   symlink ancestors even when the leaf will be created, and require the target
   to be the intended version directory or a descendant of it. A symlink escape
   outside that directory is invalid.
5. On any identity or path failure, stop before phase artifact reads, writes,
   directory creation, commands, or handoffs. Do not normalize an unsafe path
   into acceptance and do not fall back to a default phase path.

Block examples when `expected_version_id` is `v001` include
`strategy_store/v001/../v002/04_spec_build`,
`strategy_store/v002/04_spec_build`, `/tmp/04_spec_build`, and
`strategy_store/v001/escape/04_spec_build` when `escape` is a symlink
whose target is outside the intended version directory. An allowed custom
nested phase path is
`strategy_store/v001/custom/phases/04_spec_build` when its canonical
target remains under the intended version directory.

For a new-version bootstrap, only `manage-strategy-version` may proceed before
the new manifest exists or the new id becomes active. It must apply the same
workspace-relative, traversal, canonical-containment, and symlink checks to
every constructed phase path before directory creation, then write a matching
manifest before publishing `current.json` last.

## Version Path Resolution

Before using any `<phase_paths.*>` or `<version_root>` placeholder, read
`.open-xquant/workspace.yaml`. Resolve `version_root` from
`paths.versions_dir`; use `versions` only when that key is absent. Require a
safe relative path whose resolved target stays inside the workspace. Then read
`<version_root>/<version_id>/version_manifest.json` and use its exact
`phase_paths` entry for each phase. For example, a configured root of
`research_versions` must resolve the spec-build phase to
`research_versions/v003/04_spec_build`; never redirect it to a default-root phase path.

# Data Explorer

You prepare data before research. Do not assume data exists or is complete.

## Version-Governed Output Gate

When this skill is used as the data inspection phase for a strategy workflow,
read `current.json` and use `active_version` as `version_id`. Write data
inspection artifacts only under:

```text
<phase_paths.05_data_inspection>/data_inspection_result.json
<phase_paths.05_data_inspection>/data_availability_report.md
```

Do not write root-level `data_inspection_result.json` or
`data_availability_report.md`.

## Runner Python

Use the resolved runner's virtualenv Python in an installed research workspace.
Use the sibling
Python from the same `bin` directory as the runner when importing `oxq`.
Read `~/.config/open-xquant/agent.yaml`, take `preferred_runner`, then run the
sibling `python`. `oxq run python` does not exist.
Do not run `uv run python` in an installed research workspace unless the
current directory is the open-xquant source checkout.

```bash
RUNNER="/path/to/resolved/oxq"
PYTHON="$(dirname "$RUNNER")/python"
"$PYTHON" - <<'PY'
import oxq
print(oxq.__version__)
PY
```

## Market Data

List local parquet files:

```bash
uv run python - <<'PY'
from oxq.tools.data import list_symbols
print(list_symbols(data_dir="/path/to/market"))
PY
```

Inspect one symbol:

```bash
uv run python - <<'PY'
from oxq.tools.data import inspect_symbol
print(inspect_symbol("SPY", data_dir="/path/to/market"))
PY
```

Download with `yfinance`:

```bash
uv run python - <<'PY'
from pathlib import Path
from oxq.data.loaders import YFinanceDownloader

data_dir = Path("/tmp/oxq_agent_data")
YFinanceDownloader().download_many(
    symbols=["SPY", "QQQ"],
    start="2018-01-01",
    end="2026-01-01",
    dest_dir=data_dir,
)
print(data_dir)
PY
```

For A-share market bars, use `AkShareDownloader` and install the `akshare`
extra first.

## Required Data Shape

Every market parquet must have:

- file name `<SYMBOL>.parquet`
- tz-aware `DatetimeIndex`
- columns `open`, `high`, `low`, `close`, `volume`
- enough history for indicator warmup and train/test periods

If `LocalMarketDataProvider` says `No data for '<SYMBOL>'`, stop and fix data
or ask the user for a valid `--data-dir`.

## Macro Data

Use `WorldBankFetcher` with indicator name, year range, and countries:

```python
from oxq.data.factors import WorldBankFetcher

fetcher = WorldBankFetcher()
gdp = fetcher.fetch(
    target="gdp",
    start="2018",
    end="2024",
    countries=["USA", "CHN"],
)
```

Available macro aliases include `gdp`, `gdp_per_capita`, `gdp_growth`, and
`cpi`.

## Financial Data

For A-share fundamentals, use `EastMoneyFetcher` through the factor data layer
and verify available fields before screening. Network and provider coverage can
fail; report provider errors directly.

## Red Lines

- Do not run a formal backtest on unknown or missing data.
- Do not mix data providers inside one run unless the user approves and the
  report states it.
- Do not treat generated mock data or demo downloads as production research
  evidence.
