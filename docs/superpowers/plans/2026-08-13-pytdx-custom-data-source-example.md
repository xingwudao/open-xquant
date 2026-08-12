# PyTdx Custom Data Source Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an example-only `PyTdxDownloader` that directly connects to an explicitly supplied TDX-compatible quote server, writes open-xquant OHLCV artifacts, and defaults to yfinance-style proportional auto-adjustment without requiring a desktop client.

**Architecture:** Keep the integration in one example module with narrow internal boundaries for input validation, lazy dependency loading, connection lifetime, backward pagination, corporate-action parsing, proportional adjustment, and persistence. Reuse open-xquant's existing path, error, and manifest helpers; keep all pytdx calls behind a fakeable API boundary so automated tests never open sockets. Document it beside, but do not merge it with, the existing local TdxQuant HTTP example.

**Tech Stack:** Python 3.12, `pytdx==1.72` as an ephemeral example dependency, pandas, NumPy, PyArrow, pytest, `unittest.mock`, `importlib.metadata`, and existing `oxq.data` helpers.

## Global Constraints

- Work only in the existing `feature/tdx-data-source` linked worktree.
- Create `examples/custom_data_sources/pytdx_downloader.py`, `examples/custom_data_sources/PYTDX.md`, and `tests/examples/test_pytdx_downloader.py`; modify only `examples/custom_data_sources/README.md` outside those files.
- Do not modify `pyproject.toml`, `src/oxq/data/`, SDK exports, CLI source enums, doctor checks, or bundle metadata.
- Require an explicit IPv4 address or hostname plus port; never embed, scan, benchmark, rotate, or recommend quote-server addresses.
- Support only daily bars (`category=9`) and six-digit `.SH` / `.SZ` symbols; reject `.BJ`.
- Interpret `start` and `end` as canonical `YYYY-MM-DD` dates with both endpoints included.
- Default `auto_adjust=True`; apply one cumulative ratio to OHLC and never adjust volume; do not round adjusted prices to cents.
- Anchor adjustment at the newest bar returned by the server, not at the requested `end` date.
- Persist exactly `open`, `high`, `low`, `close`, `volume` with a unique ascending `Asia/Shanghai` index named `date`; OHLC are `float64` and volume is `int64`.
- Validate every external value and the complete adjusted output before creating the destination directory or writing files.
- Automated tests must replace `TdxHq_API` completely and must fail if production code attempts a real socket.
- Code, identifiers, CLI help, exceptions, and comments are English; user documentation may be Chinese.
- Use strict TDD: observe RED before each implementation slice and before each review-finding fix.
- Commit each independently passing slice; do not amend or squash existing commits.

---

## File Map

- Create `examples/custom_data_sources/pytdx_downloader.py`: validation, lazy pytdx loading, session lifetime, pagination, response conversion, adjustment, persistence, `Downloader` methods, and CLI.
- Create `tests/examples/test_pytdx_downloader.py`: fake API plus offline tests for every public and internal boundary.
- Create `examples/custom_data_sources/PYTDX.md`: setup, explicit-host usage, adjustment semantics, output, troubleshooting, smoke test, and licensing boundaries.
- Modify `examples/custom_data_sources/README.md`: turn the top into a two-option index while retaining the complete TdxQuant guide.
- Consume `docs/superpowers/specs/2026-08-13-pytdx-custom-data-source-example-design.md`: authoritative behavior and acceptance criteria.

---

### Task 1: Input contract, lazy dependency, and connection lifetime

**Files:**
- Create: `examples/custom_data_sources/pytdx_downloader.py`
- Create: `tests/examples/test_pytdx_downloader.py`

**Interfaces:**
- Consumes: `oxq.core.errors.DownloadError`.
- Produces: `_normalize_symbol(symbol: str) -> tuple[str, int, str]`, `_parse_date_range(start: str, end: str) -> tuple[date, date]`, `_load_pytdx() -> tuple[type[Any], str]`, `_connected_api(host: str, port: int, timeout: float) -> Iterator[tuple[Any, str]]`, and the validated `PyTdxDownloader` constructor.

- [ ] **Step 1: Write failing validation and dependency tests**

Create the test module with a socket guard and these exact behavior tests:

```python
from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import examples.custom_data_sources.pytdx_downloader as module
from examples.custom_data_sources.pytdx_downloader import PyTdxDownloader
from oxq.core.errors import DownloadError


@pytest.fixture(autouse=True)
def forbid_real_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not open a real socket")

    monkeypatch.setattr(socket, "socket", fail_socket)


@pytest.mark.parametrize(
    "host",
    ["", " host", "host ", "http://host", "host:7709", "user@host", "a/b"],
)
def test_rejects_invalid_host(host: str) -> None:
    with pytest.raises(ValueError, match="host"):
        PyTdxDownloader(host=host)


@pytest.mark.parametrize("port", [True, 0, 65536, -1, 7709.0])
def test_rejects_invalid_port(port: object) -> None:
    with pytest.raises(ValueError, match="port"):
        PyTdxDownloader(host="quote.example", port=port)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        PyTdxDownloader(host="quote.example", timeout=timeout)


def test_rejects_non_boolean_auto_adjust() -> None:
    with pytest.raises(ValueError, match="auto_adjust"):
        PyTdxDownloader(host="quote.example", auto_adjust=1)  # type: ignore[arg-type]


def test_normalizes_supported_symbols() -> None:
    assert module._normalize_symbol("510300.sh") == ("510300.SH", 1, "510300")
    assert module._normalize_symbol("159919.sz") == ("159919.SZ", 0, "159919")


@pytest.mark.parametrize("symbol", ["510300", "SH510300", "510300.BJ", "abc.SH"])
def test_rejects_unsupported_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match="six digits.*SH.*SZ"):
        module._normalize_symbol(symbol)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("20200501", "2026-01-01"),
        ("2020-5-1", "2026-01-01"),
        ("2020-02-30", "2026-01-01"),
        ("2026-01-02", "2026-01-01"),
    ],
)
def test_rejects_invalid_date_range(start: str, end: str) -> None:
    with pytest.raises(ValueError, match="date"):
        module._parse_date_range(start, end)


def test_missing_pytdx_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(DownloadError, match=r"uv run --with pytdx==1\.72"):
        module._load_pytdx()
```

- [ ] **Step 2: Run the new test module and confirm RED**

Run:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py
```

Expected: collection fails because
`examples.custom_data_sources.pytdx_downloader` does not exist.

- [ ] **Step 3: Implement validation and lazy loading**

Create the module with these constants, helpers, and constructor behavior:

```python
from __future__ import annotations

import importlib
import math
import re
from contextlib import contextmanager
from datetime import date, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

from oxq.core.errors import DownloadError

_SYMBOL_PATTERN = re.compile(r"^[0-9]{6}\.(SH|SZ)$")
_HOST_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*$"
)
_INSTALL_HINT = (
    "uv run --with pytdx==1.72 python "
    "examples/custom_data_sources/pytdx_downloader.py --help"
)


def _normalize_symbol(symbol: str) -> tuple[str, int, str]:
    normalized = symbol.upper()
    match = _SYMBOL_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("symbol must contain six digits and a .SH or .SZ suffix")
    suffix = match.group(1)
    return normalized, 1 if suffix == "SH" else 0, normalized[:6]


def _parse_date_range(start: str, end: str) -> tuple[date, date]:
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("start and end must be valid YYYY-MM-DD dates") from exc
    if start != start_date.isoformat() or end != end_date.isoformat():
        raise ValueError("start and end must be canonical YYYY-MM-DD dates")
    if start_date > end_date:
        raise ValueError("start date must not be later than end date")
    return start_date, end_date


def _load_pytdx() -> tuple[type[Any], str]:
    try:
        hq = importlib.import_module("pytdx.hq")
        api_class = hq.TdxHq_API
        version = metadata.version("pytdx")
    except (ModuleNotFoundError, ImportError, AttributeError, metadata.PackageNotFoundError) as exc:
        raise DownloadError(f"pytdx==1.72 is required; run: {_INSTALL_HINT}") from exc
    return api_class, version


class PyTdxDownloader:
    def __init__(
        self,
        *,
        host: str,
        port: int = 7709,
        auto_adjust: bool = True,
        timeout: float = 5.0,
    ) -> None:
        if host != host.strip() or _HOST_PATTERN.fullmatch(host) is None:
            raise ValueError("host must be an IPv4 address or hostname without scheme or port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer from 1 to 65535")
        if not isinstance(auto_adjust, bool):
            raise ValueError("auto_adjust must be a boolean")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        self.host = host
        self.port = port
        self.auto_adjust = auto_adjust
        self.timeout = timeout
```

Use an IPv4/hostname pattern that accepts `127.0.0.1` but rejects schemes,
embedded ports, paths, whitespace, and credentials. Keep imports of pytdx out
of module scope.

- [ ] **Step 4: Add and pass connection-lifetime tests**

Append tests that inject a fake class through `_load_pytdx`:

```python
def test_connected_api_uses_conservative_options_and_disconnects() -> None:
    api = MagicMock()
    api.connect.return_value = api
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with module._connected_api("quote.example", 7709, 4.5) as (opened, version):
            assert opened is api
            assert version == "1.72"

    api_class.assert_called_once_with(
        multithread=False,
        heartbeat=False,
        auto_retry=False,
        raise_exception=True,
    )
    api.connect.assert_called_once_with("quote.example", 7709, time_out=4.5)
    api.disconnect.assert_called_once_with()


def test_connected_api_rejects_false_connect_result() -> None:
    api = MagicMock()
    api.connect.return_value = False
    api_class = MagicMock(return_value=api)
    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(DownloadError, match="Cannot connect"):
            with module._connected_api("quote.example", 7709, 5.0):
                pass
```

Implement `_connected_api` as a context manager. It constructs the API with
the exact conservative options above, wraps connect/disconnect exceptions as
`DownloadError`, calls `disconnect()` once after a successful connect, does not
mask an exception already raised by the body, and never catches
`KeyboardInterrupt` or `SystemExit`. Add `sys` to the imports:

```python
@contextmanager
def _connected_api(host: str, port: int, timeout: float) -> Iterator[tuple[Any, str]]:
    api_class, version = _load_pytdx()
    api = api_class(
        multithread=False,
        heartbeat=False,
        auto_retry=False,
        raise_exception=True,
    )
    try:
        result = api.connect(host, port, time_out=timeout)
    except Exception as exc:
        raise DownloadError(f"Cannot connect to TDX quote server {host}:{port}.") from exc
    if result is False or result is None:
        raise DownloadError(f"Cannot connect to TDX quote server {host}:{port}.")
    try:
        yield api, version
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            api.disconnect()
        except Exception as exc:
            if not active_exception:
                raise DownloadError(
                    f"Cannot disconnect from TDX quote server {host}:{port}."
                ) from exc
```

Add one test where `disconnect()` raises after a clean body and assert a
`DownloadError`, plus one where the body and `disconnect()` both raise and
assert the body's original exception is preserved.

- [ ] **Step 5: Run focused checks and commit**

Run:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py
uv run ruff check examples/custom_data_sources/pytdx_downloader.py \
  tests/examples/test_pytdx_downloader.py
```

Expected: all Task 1 tests pass and Ruff reports no errors.

Commit:

```bash
git add examples/custom_data_sources/pytdx_downloader.py \
  tests/examples/test_pytdx_downloader.py
git commit -m "feat: add PyTdx connection contract"
```

---

### Task 2: Backward pagination and raw OHLCV validation

**Files:**
- Modify: `examples/custom_data_sources/pytdx_downloader.py`
- Modify: `tests/examples/test_pytdx_downloader.py`

**Interfaces:**
- Consumes: a connected object exposing `get_security_bars(category, market, code, offset, count)`.
- Produces: `_fetch_raw_bars(api: Any, market: int, code: str, start_date: date, symbol: str) -> pd.DataFrame` returning complete unique ascending raw history through the server's newest bar.

- [ ] **Step 1: Add a deterministic fake API and failing pagination test**

Add this helper and test:

```python
class FakeApi:
    def __init__(
        self,
        *,
        pages: dict[int, object],
        actions: object | None = None,
    ) -> None:
        self.pages = pages
        self.actions = [] if actions is None else actions
        self.bar_calls: list[tuple[int, int, str, int, int]] = []
        self.action_calls: list[tuple[int, str]] = []

    def get_security_bars(
        self, category: int, market: int, code: str, offset: int, count: int
    ) -> object:
        self.bar_calls.append((category, market, code, offset, count))
        return self.pages.get(offset, [])

    def get_xdxr_info(self, market: int, code: str) -> object:
        self.action_calls.append((market, code))
        return self.actions


def bar(day: str, close: float, volume: object = 1000) -> dict[str, object]:
    return {
        "datetime": f"{day} 15:00",
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "vol": volume,
    }


def test_fetches_800_bar_pages_backward_until_before_start() -> None:
    newest_days = pd.date_range("2022-01-01", periods=800, freq="D")
    newest = [
        bar(day.strftime("%Y-%m-%d"), 10.0 + number / 1000)
        for number, day in enumerate(reversed(newest_days))
    ]
    older = [bar("2020-04-30", 9.0), bar("2020-05-01", 9.1)]
    api = FakeApi(pages={0: newest, 800: older})

    frame = module._fetch_raw_bars(
        api, market=1, code="510300", start_date=module.date(2020, 5, 1), symbol="510300.SH"
    )

    assert api.bar_calls == [
        (9, 1, "510300", 0, 800),
        (9, 1, "510300", 800, 800),
    ]
    assert frame.index.is_monotonic_increasing
    assert frame.index.min().date().isoformat() == "2020-04-30"
    assert frame.index.max().date() == newest_days[-1].date()
```

- [ ] **Step 2: Run the pagination test and confirm RED**

Run:

```bash
uv run pytest -q \
  tests/examples/test_pytdx_downloader.py::test_fetches_800_bar_pages_backward_until_before_start
```

Expected: FAIL because `_fetch_raw_bars` is missing.

- [ ] **Step 3: Implement page parsing and pagination**

Add `Mapping`, `Decimal`, `InvalidOperation`, NumPy, and pandas imports, then
add these constants and helpers:

```python
_BAR_CATEGORY = 9
_PAGE_SIZE = 800
_MAX_PAGES = 128
_OHLC = ("open", "high", "low", "close")
_COLUMNS = (*_OHLC, "volume")


def _price(value: object, symbol: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise DownloadError(f"TDX returned invalid OHLC for '{symbol}'.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DownloadError(f"TDX returned invalid OHLC for '{symbol}'.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise DownloadError(f"TDX returned non-finite or non-positive OHLC for '{symbol}'.")
    return parsed


def _volume(value: object, symbol: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise DownloadError(f"TDX returned unsafe non-negative integer volume for '{symbol}'.")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DownloadError(f"TDX returned unsafe non-negative integer volume for '{symbol}'.") from exc
    limit = np.iinfo(np.int64).max
    if (
        not parsed.is_finite()
        or parsed < 0
        or parsed != parsed.to_integral_value()
        or parsed > limit
    ):
        raise DownloadError(f"TDX returned unsafe non-negative integer volume for '{symbol}'.")
    return int(parsed)


def _parse_bar_page(payload: object, symbol: str) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise DownloadError(f"TDX bar response for '{symbol}' must be a list.")
    rows: list[dict[str, object]] = []
    required = {"datetime", "open", "high", "low", "close", "vol"}
    for item in payload:
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise DownloadError(f"TDX returned a bar without required bar fields for '{symbol}'.")
        raw_datetime = item["datetime"]
        if not isinstance(raw_datetime, str):
            raise DownloadError(f"TDX returned an invalid bar date for '{symbol}'.")
        try:
            timestamp = pd.Timestamp(pd.to_datetime(raw_datetime, errors="raise"))
        except (TypeError, ValueError) as exc:
            raise DownloadError(f"TDX returned an invalid bar date for '{symbol}'.") from exc
        values = {field: _price(item[field], symbol) for field in _OHLC}
        if not (
            values["low"] <= values["open"] <= values["high"]
            and values["low"] <= values["close"] <= values["high"]
        ):
            raise DownloadError(f"TDX returned inconsistent OHLC for '{symbol}'.")
        rows.append(
            {
                "date": timestamp.tz_localize(None).normalize(),
                **values,
                "volume": _volume(item["vol"], symbol),
            }
        )
    if not rows:
        return pd.DataFrame(
            {**{field: pd.Series(dtype="float64") for field in _OHLC},
             "volume": pd.Series(dtype="int64")},
            index=pd.DatetimeIndex([], name="date", tz="Asia/Shanghai"),
        )
    frame = pd.DataFrame(rows).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date").tz_localize("Asia/Shanghai")
    frame[list(_OHLC)] = frame[list(_OHLC)].astype("float64")
    frame["volume"] = frame["volume"].astype("int64")
    return frame.loc[:, list(_COLUMNS)]


def _merge_bar_pages(pages: list[pd.DataFrame], symbol: str) -> pd.DataFrame:
    combined = pd.concat(pages)
    unique_rows: list[pd.DataFrame] = []
    for _, group in combined.groupby(level=0, sort=False):
        if len(group.drop_duplicates()) != 1:
            raise DownloadError(f"TDX returned conflicting bars for '{symbol}'.")
        unique_rows.append(group.iloc[[0]])
    result = pd.concat(unique_rows).sort_index()
    if result.index.has_duplicates:
        raise DownloadError(f"TDX returned duplicate bar dates for '{symbol}'.")
    return result.loc[:, list(_COLUMNS)]


def _fetch_raw_bars(
    api: Any,
    market: int,
    code: str,
    start_date: date,
    symbol: str,
) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    previous_fingerprint: tuple[tuple[object, ...], ...] | None = None
    for page_number in range(_MAX_PAGES):
        offset = page_number * _PAGE_SIZE
        try:
            payload = api.get_security_bars(
                _BAR_CATEGORY, market, code, offset, _PAGE_SIZE
            )
        except Exception as exc:
            raise DownloadError(f"TDX bar request failed for '{symbol}'.") from exc
        if payload is None:
            raise DownloadError(f"TDX returned no bar response for '{symbol}'.")
        page = _parse_bar_page(payload, symbol)
        if page.empty:
            if pages:
                break
            raise DownloadError(f"No data returned for '{symbol}'.")
        fingerprint = tuple(page.reset_index().itertuples(index=False, name=None))
        if fingerprint == previous_fingerprint:
            raise DownloadError(f"TDX repeated a bar page for '{symbol}'.")
        previous_fingerprint = fingerprint
        pages.append(page)
        if page.index.min().date() < start_date or len(page) < _PAGE_SIZE:
            break
    else:
        raise DownloadError(
            f"TDX bar pagination exceeded {_MAX_PAGES} pages for '{symbol}'."
        )
    return _merge_bar_pages(pages, symbol)
```

- [ ] **Step 4: Add malformed response and pagination guard tests**

Append parameterized tests covering these exact cases:

```python
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "no bar response"),
        ({"datetime": "2024-01-01"}, "list"),
        ([{"datetime": "bad"}], "required bar fields"),
        ([bar("2024-01-01", float("nan"))], "finite positive OHLC"),
        ([bar("2024-01-01", 10.0, True)], "safe non-negative integer volume"),
        ([bar("2024-01-01", 10.0, 1.5)], "safe non-negative integer volume"),
    ],
)
def test_rejects_invalid_bar_payload(payload: object, message: str) -> None:
    api = FakeApi(pages={0: payload})
    with pytest.raises(DownloadError, match=message):
        module._fetch_raw_bars(api, 1, "510300", module.date(2020, 5, 1), "510300.SH")
```

Also add separate tests for low/high inconsistency, a repeated full page,
conflicting duplicate dates across pages, exact duplicate overlap, and the
`_MAX_PAGES` exhaustion path. Build each full page with unique calendar dates.

- [ ] **Step 5: Run focused tests, type/lint checks, and commit**

Run:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py
uv run ruff check examples/custom_data_sources/pytdx_downloader.py \
  tests/examples/test_pytdx_downloader.py
uv run mypy examples/custom_data_sources/pytdx_downloader.py
```

Expected: all Task 1-2 tests pass; Ruff and mypy report no errors.

Commit:

```bash
git add examples/custom_data_sources/pytdx_downloader.py \
  tests/examples/test_pytdx_downloader.py
git commit -m "feat: validate and paginate PyTdx daily bars"
```

---

### Task 3: Corporate-action parsing and proportional auto-adjustment

**Files:**
- Modify: `examples/custom_data_sources/pytdx_downloader.py`
- Modify: `tests/examples/test_pytdx_downloader.py`

**Interfaces:**
- Consumes: complete ascending raw bars and `api.get_xdxr_info(market, code)`.
- Produces: `_adjust_bars(frame: pd.DataFrame, payload: object, first_output_date: date, symbol: str) -> tuple[pd.DataFrame, int]` with yfinance-style proportional OHLC adjustment and unchanged volume.

- [ ] **Step 1: Add the failing cash-dividend adjustment test**

Use raw closes `10.0` on 2024-01-01, `10.2` on 2024-01-02, and `9.8` on
the 2024-01-03 ex-date. A `fenhong=2.0` event means cash `0.2` per share;
the event ratio is `(10.2 - 0.2) / 10.2` and applies only before the ex-date:

```python
def xdxr(
    day: str,
    *,
    category: int = 1,
    fenhong: object = 0.0,
    peigujia: object = 0.0,
    songzhuangu: object = 0.0,
    peigu: object = 0.0,
) -> dict[str, object]:
    parsed = module.datetime.strptime(day, "%Y-%m-%d")
    return {
        "year": parsed.year,
        "month": parsed.month,
        "day": parsed.day,
        "category": category,
        "fenhong": fenhong,
        "peigujia": peigujia,
        "songzhuangu": songzhuangu,
        "peigu": peigu,
    }


def test_cash_dividend_adjusts_ohlc_by_one_ratio_and_preserves_volume() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 10.2), bar("2024-01-03", 9.8)],
        "510300.SH",
    )
    original_volume = frame["volume"].copy()

    adjusted, count = module._adjust_bars(
        frame,
        [xdxr("2024-01-03", fenhong=2.0)],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    ratio = 10.0 / 10.2
    assert adjusted.loc["2024-01-02", "close"] == pytest.approx(10.2 * ratio)
    assert adjusted.loc["2024-01-03", "close"] == pytest.approx(9.8)
    assert adjusted.loc["2024-01-02", "open"] == pytest.approx(10.1 * ratio)
    pd.testing.assert_series_equal(adjusted["volume"], original_volume)
    assert count == 1
```

- [ ] **Step 2: Run the cash-dividend test and confirm RED**

Run:

```bash
uv run pytest -q \
  tests/examples/test_pytdx_downloader.py::test_cash_dividend_adjusts_ohlc_by_one_ratio_and_preserves_volume
```

Expected: FAIL because `_adjust_bars` is missing.

- [ ] **Step 3: Implement strict action parsing and cumulative ratios**

Create an immutable action record and these pure helpers:

```python
@dataclass(frozen=True)
class _Action:
    day: date
    fenhong: float
    peigujia: float
    songzhuangu: float
    peigu: float


def _parse_actions(
    payload: object,
    first_output_date: date,
    latest_date: date,
    symbol: str,
) -> list[_Action]:
    if payload is None or not isinstance(payload, list):
        raise DownloadError(f"TDX returned an invalid corporate-action response for '{symbol}'.")
    ignored = {2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14}
    by_day: dict[date, _Action] = {}
    for record in payload:
        if not isinstance(record, Mapping):
            raise DownloadError(f"TDX returned an invalid corporate action for '{symbol}'.")
        parts: list[int] = []
        for key in ("year", "month", "day"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise DownloadError(f"TDX returned an invalid corporate-action date for '{symbol}'.")
            parts.append(value)
        try:
            action_day = date(*parts)
        except ValueError as exc:
            raise DownloadError(f"TDX returned an invalid corporate-action date for '{symbol}'.") from exc
        category = record.get("category")
        if isinstance(category, bool) or not isinstance(category, int):
            raise DownloadError(f"TDX returned an invalid corporate-action category for '{symbol}'.")
        if action_day <= first_output_date or action_day > latest_date:
            continue
        if category in ignored:
            continue
        if category != 1:
            raise DownloadError(
                f"TDX returned unsupported corporate-action category {category} for '{symbol}'."
            )
        parsed: dict[str, float] = {}
        for key in ("fenhong", "peigujia", "songzhuangu", "peigu"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
                raise DownloadError(f"TDX returned invalid adjustment fields for '{symbol}'.")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise DownloadError(f"TDX returned invalid adjustment fields for '{symbol}'.") from exc
            if not math.isfinite(number) or number < 0:
                raise DownloadError(f"TDX returned invalid adjustment fields for '{symbol}'.")
            parsed[key] = number
        action = _Action(day=action_day, **parsed)
        previous = by_day.get(action_day)
        if previous is not None and previous != action:
            raise DownloadError(f"TDX returned conflicting corporate actions for '{symbol}'.")
        by_day[action_day] = action
    return sorted(by_day.values(), key=lambda action: action.day)


def _event_ratio(action: _Action, previous_close: float, symbol: str) -> float:
    if not math.isfinite(previous_close) or previous_close <= 0:
        raise DownloadError(f"TDX returned an invalid previous close for '{symbol}'.")
    cash = action.fenhong / 10.0
    bonus = action.songzhuangu / 10.0
    rights = action.peigu / 10.0
    denominator = 1.0 + bonus + rights
    reference = (
        previous_close - cash + rights * action.peigujia
    ) / denominator
    if not math.isfinite(denominator) or denominator <= 0:
        raise DownloadError(f"TDX returned an invalid adjustment denominator for '{symbol}'.")
    ratio = reference / previous_close
    if not math.isfinite(reference) or reference <= 0 or not math.isfinite(ratio) or ratio <= 0:
        raise DownloadError(f"TDX returned an invalid adjustment factor for '{symbol}'.")
    return ratio


def _adjust_bars(
    frame: pd.DataFrame,
    payload: object,
    first_output_date: date,
    symbol: str,
) -> tuple[pd.DataFrame, int]:
    latest_date = frame.index.max().date()
    actions = _parse_actions(payload, first_output_date, latest_date, symbol)
    events: list[tuple[date, float]] = []
    for action in actions:
        prior = frame.loc[frame.index.date < action.day]
        if prior.empty:
            raise DownloadError(f"No previous close exists for an adjustment of '{symbol}'.")
        events.append(
            (action.day, _event_ratio(action, float(prior.iloc[-1]["close"]), symbol))
        )
    adjusted = frame.copy()
    for timestamp in adjusted.index:
        ratio = math.prod(
            event_ratio
            for event_day, event_ratio in events
            if event_day > timestamp.date()
        )
        adjusted.loc[timestamp, list(_OHLC)] = (
            frame.loc[timestamp, list(_OHLC)].to_numpy(dtype="float64") * ratio
        )
    prices = adjusted.loc[:, list(_OHLC)].to_numpy(dtype="float64")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DownloadError(f"Adjustment produced invalid OHLC for '{symbol}'.")
    adjusted.loc[:, list(_OHLC)] = adjusted.loc[:, list(_OHLC)].astype("float64")
    return adjusted, len(actions)
```

- [ ] **Step 4: Add complete adjustment edge tests**

Add exact numerical tests for:

```text
送转: songzhuangu=2.0 -> denominator 1.2
配股: peigu=2.0, peigujia=5.0 -> numerator adds 1.0, denominator 1.2
累计: two later events -> earlier bars receive ratio_1 * ratio_2
结束日之后: output ends before event, but event still adjusts output history
未来事件: event date after latest raw bar -> ignored
同日重复: identical record -> one applied event
同日冲突: different fields -> DownloadError
```

Add failure tests for `None`, non-list payload, invalid dates, missing fields,
booleans, negative/non-finite amounts, relevant categories `11` and `12`,
unknown relevant category `99`, missing prior close, non-positive reference
price, and a future unsupported event that must be ignored.

- [ ] **Step 5: Run focused tests, lint/type checks, and commit**

Run:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py
uv run ruff check examples/custom_data_sources/pytdx_downloader.py \
  tests/examples/test_pytdx_downloader.py
uv run mypy examples/custom_data_sources/pytdx_downloader.py
```

Expected: all Task 1-3 tests pass; Ruff and mypy report no errors.

Commit:

```bash
git add examples/custom_data_sources/pytdx_downloader.py \
  tests/examples/test_pytdx_downloader.py
git commit -m "feat: add yfinance-style PyTdx adjustment"
```

---

### Task 4: Downloader integration, artifacts, multi-symbol flow, and CLI

**Files:**
- Modify: `examples/custom_data_sources/pytdx_downloader.py`
- Modify: `tests/examples/test_pytdx_downloader.py`

**Interfaces:**
- Consumes: Task 1 session helpers, Task 2 raw frame, Task 3 adjustment, `resolve_data_dir`, and `write_manifest`.
- Produces: the complete `Downloader`-compatible public class and `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Add the failing end-to-end artifact test**

Patch `_connected_api` to return a fake API and version, then assert the exact
contract:

```python
def test_download_writes_standard_adjusted_artifacts(tmp_path: Path) -> None:
    api = FakeApi(
        pages={0: [bar("2024-01-04", 9.9), bar("2024-01-03", 9.8), bar("2024-01-02", 10.2)]},
        actions=[xdxr("2024-01-03", fenhong=2.0)],
    )
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        path = PyTdxDownloader(host="quote.example").download(
            "510300.sh", "2024-01-02", "2024-01-03", tmp_path
        )

    assert path == tmp_path / "510300.SH.parquet"
    frame = pd.read_parquet(path)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert str(frame.index.tz) == "Asia/Shanghai"
    assert frame["volume"].dtype == "int64"

    manifest = read_manifest(tmp_path / "510300.SH.manifest.json")
    assert manifest is not None
    assert manifest["provider"] == "pytdx"
    assert manifest["rows"] == 2
    assert manifest["extra"] == {
        "auto_adjust": True,
        "adjustment_method": "xdxr_ratio_yfinance_semantics",
        "adjustment_reference_date": "2024-01-04",
        "applied_event_count": 1,
        "bar_category": 9,
        "host": "quote.example",
        "period": "1d",
        "port": 7709,
        "pytdx_version": "1.72",
        "transport": "tdx_hq_tcp",
    }
    assert verify_manifest(path).status == "real"
```

- [ ] **Step 2: Run the end-to-end test and confirm RED**

Run:

```bash
uv run pytest -q \
  tests/examples/test_pytdx_downloader.py::test_download_writes_standard_adjusted_artifacts
```

Expected: FAIL because `download` is not implemented.

- [ ] **Step 3: Implement single- and multi-symbol integration**

Add imports for `resolve_data_dir` and `write_manifest`. Add an immutable
prepared request so all caller input is validated before opening a socket, then
implement one private connected method and both public methods:

```python
@dataclass(frozen=True)
class _DownloadRequest:
    symbol: str
    market: int
    code: str
    start_text: str
    end_text: str
    start_date: date
    end_date: date


def _prepare_request(symbol: str, start: str, end: str) -> _DownloadRequest:
    normalized, market, code = _normalize_symbol(symbol)
    start_date, end_date = _parse_date_range(start, end)
    return _DownloadRequest(
        symbol=normalized,
        market=market,
        code=code,
        start_text=start,
        end_text=end,
        start_date=start_date,
        end_date=end_date,
    )


def _download_connected(
    self,
    api: Any,
    pytdx_version: str,
    request: _DownloadRequest,
    dest_dir: Path | None,
) -> Path:
    raw = _fetch_raw_bars(
        api,
        request.market,
        request.code,
        request.start_date,
        request.symbol,
    )
    output_mask = (
        (raw.index.date >= request.start_date)
        & (raw.index.date <= request.end_date)
    )
    requested_raw = raw.loc[output_mask]
    if requested_raw.empty:
        raise DownloadError(
            f"No data returned for '{request.symbol}' "
            f"({request.start_text} to {request.end_text})."
        )
    if self.auto_adjust:
        try:
            actions = api.get_xdxr_info(request.market, request.code)
        except Exception as exc:
            raise DownloadError(
                f"TDX corporate-action request failed for '{request.symbol}'."
            ) from exc
        adjusted, event_count = _adjust_bars(
            raw,
            actions,
            requested_raw.index.min().date(),
            request.symbol,
        )
        adjustment_method = "xdxr_ratio_yfinance_semantics"
    else:
        adjusted = raw.copy()
        event_count = 0
        adjustment_method = "none"
    frame = adjusted.loc[output_mask, list(_COLUMNS)].copy()
    prices = frame.loc[:, list(_OHLC)].to_numpy(dtype="float64")
    if (
        frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
        or not np.isfinite(prices).all()
        or (prices <= 0).any()
        or (frame["volume"] < 0).any()
    ):
        raise DownloadError(f"TDX produced an invalid output frame for '{request.symbol}'.")
    data_dir = resolve_data_dir(dest_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{request.symbol}.parquet"
    frame.to_parquet(path)
    write_manifest(
        parquet_path=path,
        symbol=request.symbol,
        provider="pytdx",
        start=request.start_text,
        end=request.end_text,
        rows=len(frame),
        extra={
            "auto_adjust": self.auto_adjust,
            "adjustment_method": adjustment_method,
            "adjustment_reference_date": raw.index.max().date().isoformat(),
            "applied_event_count": event_count,
            "bar_category": _BAR_CATEGORY,
            "host": self.host,
            "period": "1d",
            "port": self.port,
            "pytdx_version": pytdx_version,
            "transport": "tdx_hq_tcp",
        },
    )
    return path

def download(
    self,
    symbol: str,
    start: str,
    end: str,
    dest_dir: Path | None = None,
) -> Path:
    request = _prepare_request(symbol, start, end)
    with _connected_api(self.host, self.port, self.timeout) as (api, version):
        return self._download_connected(api, version, request, dest_dir)

def download_many(
    self,
    symbols: list[str],
    start: str,
    end: str,
    dest_dir: Path | None = None,
) -> dict[str, Path]:
    requests = [
        (symbol, _prepare_request(symbol, start, end)) for symbol in symbols
    ]
    if not requests:
        return {}
    results: dict[str, Path] = {}
    with _connected_api(self.host, self.port, self.timeout) as (api, version):
        for original_symbol, request in requests:
            results[original_symbol] = self._download_connected(
                api, version, request, dest_dir
            )
    return results
```

For `auto_adjust=False`, do not call `get_xdxr_info`; use method `none`, count
`0`, and still record the newest source bar as `adjustment_reference_date` so
the fetched snapshot remains auditable.

- [ ] **Step 4: Add integration failure, multi-symbol, and protocol tests**

Add tests that prove:

```python
def test_downloader_satisfies_protocol() -> None:
    downloader: Downloader = PyTdxDownloader(host="quote.example")
    assert isinstance(downloader, Downloader)
```

Also prove no destination directory is created for invalid bars, invalid
actions, or an empty inclusive range; `auto_adjust=False` skips actions and
preserves raw OHLC; `download_many` opens one connection, preserves input
order, performs `.SH`/`.SZ` market calls, and stops before the third symbol when
the second fails. Assert files completed before the failure remain present.

- [ ] **Step 5: Add CLI tests and implementation**

Implement:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download daily bars directly from a TDX quote server."
    )
    parser.add_argument("symbol", help="Six-digit symbol with .SH or .SZ")
    parser.add_argument("start", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("end", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--host", required=True, help="Explicit TDX server hostname or IPv4 address")
    parser.add_argument("--port", type=int, default=7709)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--no-auto-adjust", action="store_true")
    parser.add_argument("--dest-dir", type=Path)
    args = parser.parse_args(argv)
    path = PyTdxDownloader(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        auto_adjust=not args.no_auto_adjust,
    ).download(args.symbol, args.start, args.end, args.dest_dir)
    print(path)
    return 0
```

Add tests patching `PyTdxDownloader` to assert required host, defaults, the
`--no-auto-adjust` inversion, positional dates, destination path, printed
output, and exit `0`.

- [ ] **Step 6: Run focused checks and commit**

Run:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py
uv run pytest -q tests/examples/test_tdxquant_downloader.py
uv run ruff check examples/custom_data_sources tests/examples
uv run mypy examples/custom_data_sources/pytdx_downloader.py
```

Expected: both example suites pass; Ruff and mypy report no errors.

Commit:

```bash
git add examples/custom_data_sources/pytdx_downloader.py \
  tests/examples/test_pytdx_downloader.py
git commit -m "feat: complete PyTdx downloader example"
```

---

### Task 5: User documentation and SDK-boundary audit

**Files:**
- Create: `examples/custom_data_sources/PYTDX.md`
- Modify: `examples/custom_data_sources/README.md`

**Interfaces:**
- Consumes: the final CLI and class from Task 4.
- Produces: a two-option entry document and a complete standalone direct-connection guide.

- [ ] **Step 1: Write the PyTdx guide**

Create `PYTDX.md` with these concrete sections and facts:

```text
# 直接连接行情服务器：PyTdxDownloader
定位与非 SDK 声明
前提和 pytdx==1.72 临时安装
显式 host 的 Python 调用
显式 --host 的 CLI 调用
无需启动通达信客户端
默认 yfinance 风格比例前复权
--no-auto-adjust 不复权
比例复权与通达信桌面仿射复权的差异
双端包含日期和 Asia/Shanghai schema
manifest 字段
连接、空数据、分页、事件和依赖错误排查
510300.SH 人工 smoke 命令
非生产、服务器访问、数据许可和再分发警告
pytdx 已归档、个人学习定位和非商业使用声明
```

All commands must use `uv run --with pytdx==1.72`; every executable example
must contain `--host YOUR_TDX_HOST` rather than a real or default address.

- [ ] **Step 2: Convert README into a two-option entry without deleting TdxQuant details**

Add an opening section that links `PYTDX.md` and describes:

```text
TdxQuantDownloader: official local HTTP, supported TQ desktop client required.
PyTdxDownloader: direct third-party protocol client, desktop client not required,
explicit server required, higher compatibility/licensing/availability risk.
```

Retitle the existing guide as the TdxQuant section and preserve all existing
commands, warnings, and official links.

- [ ] **Step 3: Audit forbidden SDK changes and documentation commands**

Run:

```bash
git diff 5c79715 -- pyproject.toml src/oxq
rg -n "YOUR_TDX_HOST|pytdx==1.72|510300.SH|无需.*客户端|auto_adjust" \
  examples/custom_data_sources/README.md \
  examples/custom_data_sources/PYTDX.md
rg -n "[0-9]{1,3}(\.[0-9]{1,3}){3}(:7709)?" \
  examples/custom_data_sources/PYTDX.md \
  examples/custom_data_sources/pytdx_downloader.py
```

Expected: the SDK diff is empty; required terms exist; no embedded IPv4 server
address is present except validation examples such as `127.0.0.1` in tests,
which are not a recommended endpoint.

- [ ] **Step 4: Run focused verification and commit docs**

Run:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py \
  tests/examples/test_tdxquant_downloader.py
uv run ruff check examples/custom_data_sources tests/examples
uv run mypy examples/custom_data_sources/pytdx_downloader.py
git diff --check
```

Expected: all commands exit `0`.

Commit:

```bash
git add examples/custom_data_sources/README.md \
  examples/custom_data_sources/PYTDX.md
git commit -m "docs: explain direct PyTdx data source example"
```

---

### Task 6: Implementation review, regression fixes, and final verification

**Files:**
- Review: every file changed after commit `5c79715`.
- Modify: only files implicated by verified review findings.
- Test: add regression coverage to `tests/examples/test_pytdx_downloader.py` before each fix.

**Interfaces:**
- Consumes: design spec, implementation plan, committed implementation, and test evidence.
- Produces: a reviewed branch with no remaining actionable finding and a clean full-suite result.

- [ ] **Step 1: Review the committed diff against requirements**

Run:

```bash
git diff --stat 5c79715..HEAD
git diff --check 5c79715..HEAD
git diff 5c79715..HEAD -- \
  examples/custom_data_sources/pytdx_downloader.py \
  examples/custom_data_sources/PYTDX.md \
  examples/custom_data_sources/README.md \
  tests/examples/test_pytdx_downloader.py
```

Inspect every branch for: design-spec coverage, inclusive dates, current-date
adjustment anchoring, pagination termination, duplicate/conflict handling,
numeric type confusion including booleans, close-before-event lookup, future
events, connection cleanup, exception masking, partial writes, secrets/server
addresses, CLI truthfulness, and tests that genuinely execute each failure
path.

- [ ] **Step 2: Fix every verified finding with RED/GREEN evidence**

For each actionable finding, first add the smallest test that fails for the
observed reason:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py -x
```

Record the expected assertion or exception mismatch, implement the narrow
fix, rerun the same node until it passes, then rerun the entire PyTdx test
module. If review has no actionable findings, make no synthetic code change
and proceed with the review evidence.

- [ ] **Step 3: Commit review fixes**

When findings required changes:

```bash
git add examples/custom_data_sources/pytdx_downloader.py \
  examples/custom_data_sources/PYTDX.md \
  examples/custom_data_sources/README.md \
  tests/examples/test_pytdx_downloader.py
git commit -m "fix: address PyTdx example review findings"
```

Do not create an empty commit when no code or documentation changed.

- [ ] **Step 4: Run complete verification from a clean index**

Run and retain exit codes:

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py
uv run pytest -q tests/examples/test_tdxquant_downloader.py
uv run ruff check examples/custom_data_sources tests/examples
uv run mypy examples/custom_data_sources/pytdx_downloader.py
uv run pytest
git diff --check
git status --short
```

Expected: all checks exit `0`; the full suite has no new failures; status is
clean before the live smoke output is created outside the repository.

---

### Task 7: Live `510300.SH` acceptance test

**Files:**
- Do not create or modify repository files.
- Write smoke artifacts only below a directory created with `mktemp -d`.

**Interfaces:**
- Consumes: the reviewed CLI, `pytdx==1.72`, and one explicitly selected reachable compatible endpoint.
- Produces: independently verified adjusted `510300.SH` Parquet and manifest evidence for `2020-05-01` through `2026-01-01`.

- [ ] **Step 1: Select a reachable endpoint without changing the example contract**

Obtain candidate endpoints from a current, attributable public source for the
one-time smoke test. Test candidates serially with short timeouts and stop at
the first endpoint that accepts a pytdx connection. Do not write candidates
to source, docs, test fixtures, shell history files, or commits; do not scan
address ranges.

- [ ] **Step 2: Download the fixed ETF and date range**

Create an external temporary directory, then run with the chosen endpoint:

```bash
SMOKE_DIR=$(mktemp -d)
uv run --with pytdx==1.72 python \
  examples/custom_data_sources/pytdx_downloader.py \
  510300.SH 2020-05-01 2026-01-01 \
  --host "$TDX_SMOKE_HOST" --port "$TDX_SMOKE_PORT" \
  --dest-dir "$SMOKE_DIR"
```

`TDX_SMOKE_HOST` and `TDX_SMOKE_PORT` are task-specific variables; do not use
`HOME` or another system variable.

- [ ] **Step 3: Verify the artifacts independently**

Run a read-only verifier that loads the generated files and asserts:

```python
assert len(frame) > 0
assert frame.index.is_unique
assert frame.index.is_monotonic_increasing
assert str(frame.index.tz) == "Asia/Shanghai"
assert frame.index.min().date() >= date(2020, 5, 1)
assert frame.index.max().date() <= date(2026, 1, 1)
assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
assert frame[["open", "high", "low", "close"]].notna().all().all()
assert np.isfinite(frame[["open", "high", "low", "close"]].to_numpy()).all()
assert (frame["volume"] >= 0).all()
assert manifest["provider"] == "pytdx"
assert manifest["symbol"] == "510300.SH"
assert manifest["start"] == "2020-05-01"
assert manifest["end"] == "2026-01-01"
assert manifest["rows"] == len(frame)
assert manifest["extra"]["auto_adjust"] is True
assert manifest["extra"]["adjustment_method"] == "xdxr_ratio_yfinance_semantics"
assert verify_manifest(parquet_path).status == "real"
```

Print only endpoint-independent evidence: row count, first/last dates, dtypes,
applied event count, adjustment reference date, and manifest verification
status. Do not print or commit raw records or the selected endpoint.

- [ ] **Step 4: Reconfirm repository cleanliness and completion**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: the feature branch is clean, all implementation/review commits are
present, and no smoke artifacts are tracked. Only after this evidence and the
full verification in Task 6 may the objective be marked complete.
