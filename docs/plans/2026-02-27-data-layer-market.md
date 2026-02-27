# Data Layer: Market OHLCV Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement market OHLCV data download (yfinance + akshare) and local Parquet-based reading, following the Download-then-Read separation pattern.

**Architecture:** Two independent concerns — Downloaders fetch from external APIs and persist to Parquet; LocalMarketDataProvider reads from Parquet and serves the strategy pipeline. See `docs/plans/2026-02-27-data-layer-market-design.md` for full design.

**Tech Stack:** Python 3.12+, pandas, pyarrow, yfinance (optional), akshare (optional), pytest

---

### Task 1: Core Error Types

**Files:**
- Modify: `src/oxq/core/errors.py`
- Create: `tests/core/test_errors.py`

**Step 1: Write the failing test**

```python
# tests/core/test_errors.py
from oxq.core.errors import OxqError, SymbolNotFoundError, DownloadError


def test_oxq_error_is_exception():
    assert issubclass(OxqError, Exception)


def test_symbol_not_found_error_is_oxq_error():
    err = SymbolNotFoundError("AAPL")
    assert isinstance(err, OxqError)
    assert "AAPL" in str(err)


def test_download_error_is_oxq_error():
    err = DownloadError("timeout")
    assert isinstance(err, OxqError)
    assert "timeout" in str(err)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_errors.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/oxq/core/errors.py
class OxqError(Exception):
    """Framework base exception."""


class SymbolNotFoundError(OxqError):
    """Local data file not found for symbol."""


class DownloadError(OxqError):
    """Data download failed."""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_errors.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/oxq/core/errors.py tests/core/test_errors.py
git commit -m "feat(core): add OxqError, SymbolNotFoundError, DownloadError"
```

---

### Task 2: MarketDataProvider Protocol

**Files:**
- Modify: `src/oxq/data/providers.py`
- Create: `tests/data/test_providers.py`

**Step 1: Write the failing test**

```python
# tests/data/test_providers.py
import pandas as pd
from oxq.data.providers import MarketDataProvider


class FakeProvider:
    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_latest(self, symbol: str) -> pd.Series:
        return pd.Series()


def test_fake_provider_satisfies_protocol():
    provider: MarketDataProvider = FakeProvider()
    assert isinstance(provider, MarketDataProvider)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_providers.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/oxq/data/providers.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    """Market data interface: the only entry point for strategy to access market data."""

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...

    def get_latest(self, symbol: str) -> pd.Series: ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_providers.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add src/oxq/data/providers.py tests/data/test_providers.py
git commit -m "feat(data): add MarketDataProvider Protocol"
```

---

### Task 3: resolve_data_dir Utility

**Files:**
- Modify: `src/oxq/data/loaders.py`
- Create: `tests/data/test_loaders.py`

**Step 1: Write the failing test**

```python
# tests/data/test_loaders.py
from pathlib import Path

from oxq.data.loaders import resolve_data_dir


def test_resolve_with_explicit_dir(tmp_path: Path):
    result = resolve_data_dir(tmp_path / "custom")
    assert result == tmp_path / "custom"


def test_resolve_with_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "env"))
    result = resolve_data_dir()
    assert result == tmp_path / "env" / "market"


def test_resolve_default(monkeypatch):
    monkeypatch.delenv("OXQ_DATA_DIR", raising=False)
    result = resolve_data_dir()
    assert result == Path.home() / ".oxq" / "data" / "market"


def test_explicit_dir_overrides_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "env"))
    result = resolve_data_dir(tmp_path / "explicit")
    assert result == tmp_path / "explicit"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_loaders.py::test_resolve_with_explicit_dir -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# src/oxq/data/loaders.py
from __future__ import annotations

import os
from pathlib import Path


def resolve_data_dir(dest_dir: Path | None = None) -> Path:
    """Resolve data storage directory. Priority: parameter > OXQ_DATA_DIR > default."""
    if dest_dir is not None:
        return dest_dir
    env = os.environ.get("OXQ_DATA_DIR")
    if env:
        return Path(env) / "market"
    return Path.home() / ".oxq" / "data" / "market"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_loaders.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/oxq/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): add resolve_data_dir utility"
```

---

### Task 4: LocalMarketDataProvider

**Files:**
- Modify: `src/oxq/data/market.py`
- Modify: `tests/data/test_loaders.py` (add helper to create test parquet)
- Create: `tests/data/test_market.py`

**Context:** This provider reads Parquet files from a local directory. It needs a helper to create test fixtures. We'll create a `conftest.py` with a shared fixture for a sample Parquet file.

**Step 1: Create test conftest with sample data fixture**

```python
# tests/data/conftest.py
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a tmp dir with a sample AAPL.parquet file."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B", name="date")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )
    df.to_parquet(tmp_path / "AAPL.parquet")
    return tmp_path
```

**Step 2: Write the failing tests**

```python
# tests/data/test_market.py
from pathlib import Path

import pandas as pd
import pytest

from oxq.core.errors import SymbolNotFoundError
from oxq.data.market import LocalMarketDataProvider
from oxq.data.providers import MarketDataProvider


def test_satisfies_protocol(sample_data_dir: Path):
    provider: MarketDataProvider = LocalMarketDataProvider(data_dir=sample_data_dir)
    assert isinstance(provider, MarketDataProvider)


def test_get_bars_full_range(sample_data_dir: Path):
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    df = provider.get_bars("AAPL", "2024-01-01", "2024-12-31")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert len(df) == 5


def test_get_bars_date_filter(sample_data_dir: Path):
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    df = provider.get_bars("AAPL", "2024-01-03", "2024-01-04")
    assert len(df) == 2


def test_get_bars_out_of_range_returns_available(sample_data_dir: Path):
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    df = provider.get_bars("AAPL", "2020-01-01", "2020-12-31")
    assert len(df) == 0


def test_get_bars_symbol_not_found(sample_data_dir: Path):
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    with pytest.raises(SymbolNotFoundError, match="MSFT"):
        provider.get_bars("MSFT", "2024-01-01", "2024-12-31")


def test_get_latest(sample_data_dir: Path):
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    s = provider.get_latest("AAPL")
    assert isinstance(s, pd.Series)
    assert s["close"] == 108.0


def test_get_latest_symbol_not_found(sample_data_dir: Path):
    provider = LocalMarketDataProvider(data_dir=sample_data_dir)
    with pytest.raises(SymbolNotFoundError):
        provider.get_latest("MSFT")


def test_uses_resolve_data_dir_default(monkeypatch, sample_data_dir: Path):
    monkeypatch.setenv("OXQ_DATA_DIR", str(sample_data_dir.parent))
    monkeypatch.setattr(
        "oxq.data.market.resolve_data_dir",
        lambda dest_dir=None: sample_data_dir,
    )
    provider = LocalMarketDataProvider()
    df = provider.get_bars("AAPL", "2024-01-01", "2024-12-31")
    assert len(df) == 5
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/data/test_market.py -v`
Expected: FAIL with ImportError

**Step 4: Write minimal implementation**

```python
# src/oxq/data/market.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from oxq.core.errors import SymbolNotFoundError
from oxq.data.loaders import resolve_data_dir


class LocalMarketDataProvider:
    """Read market data from local Parquet files. Implements MarketDataProvider Protocol."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = resolve_data_dir(data_dir)

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self._data_dir / f"{symbol}.parquet"
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        return df.loc[start:end]

    def get_latest(self, symbol: str) -> pd.Series:
        path = self._data_dir / f"{symbol}.parquet"
        if not path.exists():
            msg = f"No data for '{symbol}'. Run downloader first."
            raise SymbolNotFoundError(msg)
        df = pd.read_parquet(path)
        return df.iloc[-1]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_market.py -v`
Expected: 8 passed

**Step 6: Commit**

```bash
git add tests/data/conftest.py tests/data/test_market.py src/oxq/data/market.py
git commit -m "feat(data): add LocalMarketDataProvider (Parquet reader)"
```

---

### Task 5: Downloader Protocol + YFinanceDownloader

**Files:**
- Modify: `src/oxq/data/loaders.py` (add Downloader Protocol + YFinanceDownloader)
- Modify: `tests/data/test_loaders.py` (add downloader tests)

**Step 1: Write the failing tests**

Append to `tests/data/test_loaders.py`:

```python
# append to tests/data/test_loaders.py
from unittest.mock import patch, MagicMock

import pandas as pd

from oxq.data.loaders import Downloader, YFinanceDownloader


def test_yfinance_downloader_satisfies_protocol():
    downloader: Downloader = YFinanceDownloader()
    assert isinstance(downloader, Downloader)


def test_yfinance_download_saves_parquet(tmp_path):
    mock_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [104.0, 105.0],
            "Volume": [1000, 1100],
        },
        index=pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03"], name="Date"
        ),
    )
    with patch("oxq.data.loaders.yfinance") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        downloader = YFinanceDownloader()
        path = downloader.download("AAPL", "2024-01-02", "2024-01-03", dest_dir=tmp_path)

    assert path == tmp_path / "AAPL.parquet"
    assert path.exists()
    result = pd.read_parquet(path)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert len(result) == 2


def test_yfinance_download_many(tmp_path):
    mock_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [104.0],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
    )
    with patch("oxq.data.loaders.yfinance") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        downloader = YFinanceDownloader()
        paths = downloader.download_many(
            ["AAPL", "MSFT"], "2024-01-02", "2024-01-03", dest_dir=tmp_path
        )

    assert set(paths.keys()) == {"AAPL", "MSFT"}
    assert all(p.exists() for p in paths.values())


def test_yfinance_download_empty_raises(tmp_path):
    with patch("oxq.data.loaders.yfinance") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        downloader = YFinanceDownloader()
        with pytest.raises(DownloadError, match="AAPL"):
            downloader.download("AAPL", "2024-01-02", "2024-01-03", dest_dir=tmp_path)
```

Add missing import at top of file:

```python
import pytest
from oxq.core.errors import DownloadError
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_loaders.py::test_yfinance_downloader_satisfies_protocol -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Append to `src/oxq/data/loaders.py`:

```python
# append to src/oxq/data/loaders.py
from typing import Protocol, runtime_checkable

import pandas as pd

from oxq.core.errors import DownloadError


@runtime_checkable
class Downloader(Protocol):
    """Data download protocol: fetch from external source and persist."""

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path: ...

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]: ...


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw API DataFrame to standard schema."""
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    cols = ["open", "high", "low", "close", "volume"]
    df = df[cols]
    df["volume"] = df["volume"].astype("int64")
    return df


class YFinanceDownloader:
    """Download market data via yfinance. Covers US and global equities."""

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        import yfinance  # noqa: F811 — lazy import, optional dep

        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        ticker = yfinance.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        if df.empty:
            msg = f"No data returned for '{symbol}' ({start} to {end})."
            raise DownloadError(msg)

        df = _normalize_df(df)
        path = data_dir / f"{symbol}.parquet"
        df.to_parquet(path)
        return path

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return {s: self.download(s, start, end, dest_dir) for s in symbols}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_loaders.py -v`
Expected: 8 passed (4 resolve + 4 yfinance)

**Step 5: Commit**

```bash
git add src/oxq/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): add Downloader Protocol + YFinanceDownloader"
```

---

### Task 6: AkShareDownloader

**Files:**
- Modify: `src/oxq/data/loaders.py` (add AkShareDownloader)
- Modify: `tests/data/test_loaders.py` (add akshare tests)

**Context:** akshare uses `ak.stock_zh_a_hist(symbol, start_date, end_date)` which returns a DataFrame with Chinese column names like `日期`, `开盘`, `收盘`, `最高`, `最低`, `成交量`. The symbol format is bare code like `"600519"` (no suffix).

**Step 1: Write the failing tests**

Append to `tests/data/test_loaders.py`:

```python
from oxq.data.loaders import AkShareDownloader


def test_akshare_downloader_satisfies_protocol():
    downloader: Downloader = AkShareDownloader()
    assert isinstance(downloader, Downloader)


def test_akshare_download_saves_parquet(tmp_path):
    mock_df = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [1800.0, 1810.0],
            "最高": [1850.0, 1860.0],
            "最低": [1790.0, 1800.0],
            "收盘": [1840.0, 1850.0],
            "成交量": [50000, 51000],
        }
    )
    with patch("oxq.data.loaders.akshare") as mock_ak:
        mock_ak.stock_zh_a_hist.return_value = mock_df

        downloader = AkShareDownloader()
        path = downloader.download("600519", "20240102", "20240103", dest_dir=tmp_path)

    assert path == tmp_path / "600519.parquet"
    assert path.exists()
    result = pd.read_parquet(path)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert len(result) == 2


def test_akshare_download_empty_raises(tmp_path):
    with patch("oxq.data.loaders.akshare") as mock_ak:
        mock_ak.stock_zh_a_hist.return_value = pd.DataFrame()

        downloader = AkShareDownloader()
        with pytest.raises(DownloadError, match="600519"):
            downloader.download("600519", "20240102", "20240103", dest_dir=tmp_path)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_loaders.py::test_akshare_downloader_satisfies_protocol -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Append to `src/oxq/data/loaders.py`:

```python
class AkShareDownloader:
    """Download A-share market data via akshare."""

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        import akshare  # noqa: F811 — lazy import, optional dep

        data_dir = resolve_data_dir(dest_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        df = akshare.stock_zh_a_hist(
            symbol=symbol,
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df.empty:
            msg = f"No data returned for '{symbol}' ({start} to {end})."
            raise DownloadError(msg)

        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[["open", "high", "low", "close", "volume"]]
        df["volume"] = df["volume"].astype("int64")

        path = data_dir / f"{symbol}.parquet"
        df.to_parquet(path)
        return path

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return {s: self.download(s, start, end, dest_dir) for s in symbols}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_loaders.py -v`
Expected: 11 passed (4 resolve + 4 yfinance + 3 akshare)

**Step 5: Commit**

```bash
git add src/oxq/data/loaders.py tests/data/test_loaders.py
git commit -m "feat(data): add AkShareDownloader"
```

---

### Task 7: Package Exports + Dependencies

**Files:**
- Modify: `src/oxq/data/__init__.py`
- Modify: `src/oxq/core/__init__.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

```python
# tests/data/test_init.py

def test_data_public_api():
    from oxq.data import (
        MarketDataProvider,
        LocalMarketDataProvider,
        Downloader,
        YFinanceDownloader,
        AkShareDownloader,
        resolve_data_dir,
    )
    assert MarketDataProvider is not None
    assert LocalMarketDataProvider is not None
    assert Downloader is not None
    assert YFinanceDownloader is not None
    assert AkShareDownloader is not None
    assert resolve_data_dir is not None


def test_core_errors_public_api():
    from oxq.core import OxqError, SymbolNotFoundError, DownloadError
    assert OxqError is not None
    assert SymbolNotFoundError is not None
    assert DownloadError is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_init.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementations**

```python
# src/oxq/data/__init__.py
from oxq.data.loaders import (
    AkShareDownloader,
    Downloader,
    YFinanceDownloader,
    resolve_data_dir,
)
from oxq.data.market import LocalMarketDataProvider
from oxq.data.providers import MarketDataProvider

__all__ = [
    "AkShareDownloader",
    "Downloader",
    "LocalMarketDataProvider",
    "MarketDataProvider",
    "YFinanceDownloader",
    "resolve_data_dir",
]
```

```python
# src/oxq/core/__init__.py
from oxq.core.errors import DownloadError, OxqError, SymbolNotFoundError

__all__ = [
    "DownloadError",
    "OxqError",
    "SymbolNotFoundError",
]
```

Update `pyproject.toml` dependencies:

```toml
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
yfinance = [
    "yfinance>=0.2",
]
akshare = [
    "akshare>=1.10",
]
mcp = [
    "mcp",
]
talib = [
    "TA-Lib",
]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff",
    "mypy",
    "pandas-stubs",
    "pyarrow-stubs",
]
```

**Step 4: Install updated deps and run all tests**

Run: `pip install -e ".[dev]" && pytest -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/oxq/data/__init__.py src/oxq/core/__init__.py pyproject.toml tests/data/test_init.py
git commit -m "feat(data): add package exports and pyarrow/yfinance/akshare deps"
```

---

### Task 8: Lint + Type Check

**Step 1: Run ruff**

Run: `ruff check src/oxq/core/errors.py src/oxq/data/`
Expected: No issues (or fix any that appear)

**Step 2: Run mypy**

Run: `mypy src/oxq/core/errors.py src/oxq/data/`
Expected: No issues (or fix any that appear)

**Step 3: Run full test suite**

Run: `pytest -v`
Expected: All tests pass

**Step 4: Commit any fixes**

```bash
git add -A && git commit -m "chore: fix lint and type issues"
```

(Skip commit if no fixes needed.)
