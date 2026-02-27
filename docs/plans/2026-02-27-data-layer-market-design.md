# Data Layer: Market OHLCV Data — Design Document

Date: 2026-02-27

## Overview

实现数据层的市场量价数据获取功能。分为两个独立职责：

- **Downloader（数据下载）** — side-effectful，从外部 API 获取数据并持久化为 Parquet 文件。策略运行前的准备步骤，不参与回测/交易管道。
- **MarketDataProvider（数据消费）** — pure，只从本地 Parquet 文件读取，返回标准 DataFrame。策略管道中的唯一数据入口。

```
用户/Agent
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│  Downloader  │────▶│  ~/.oxq/data/    │
│  (yfinance)  │     │  market/         │
│  (akshare)   │     │  AAPL.parquet    │
└─────────────┘     │  600519.SS.par.. │
                     └────────┬─────────┘
                              │ read only
                              ▼
                     ┌──────────────────┐
                     │ LocalProvider     │
                     │ (MarketData      │
                     │  Provider)       │
                     └──────────────────┘
                              │
                              ▼
                        mktdata DataFrame
```

## Design Decisions

- **目标市场**: US stocks (yfinance) + A-shares (akshare)，各自独立 Provider
- **存储格式**: Parquet only（MVP 阶段）
- **缓存策略**: Download-then-read 分离，Provider 只读本地文件
- **数据库支持**: Protocol 天然支持扩展，MVP 不内置
- **存储路径**: 默认 `~/.oxq/data/market/`，支持环境变量和参数覆盖

## Protocol & Data Types

### MarketDataProvider Protocol

```python
# src/oxq/data/providers.py

class MarketDataProvider(Protocol):
    """行情接口：策略获取行情数据的唯一入口"""
    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...
    def get_latest(self, symbol: str) -> pd.Series: ...
```

### DataFrame Schema

- **Index**: `DatetimeIndex`, name = `"date"`
- **Columns**: `open`, `high`, `low`, `close`, `volume` (全小写)
- **dtype**: `open/high/low/close` = `float64`, `volume` = `int64`
- **排序**: 按日期升序

`get_latest` 返回最近一个交易日的 `pd.Series`，keys 同上。

## Downloader Design

```python
# src/oxq/data/loaders.py

class Downloader(Protocol):
    """数据下载协议：从外部源获取数据并持久化"""
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
```

### Implementations

- `YFinanceDownloader` — 封装 `yfinance.download()`，适用于美股及全球市场
- `AkShareDownloader` — 封装 akshare 日线接口，适用于 A 股

### Responsibilities

- 下载原始数据 → 统一为标准 schema → 存为 Parquet
- 列名映射、时区处理、复权处理在 Downloader 内部完成
- `dest_dir` 默认 `~/.oxq/data/market/`，可通过参数或 `OXQ_DATA_DIR` 覆盖
- 文件命名: `{symbol}.parquet`

## LocalMarketDataProvider

```python
# src/oxq/data/market.py

class LocalMarketDataProvider:
    def __init__(self, data_dir: Path | None = None):
        # 优先级: 参数 > OXQ_DATA_DIR > ~/.oxq/data/market/
        ...

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        # 读取 Parquet → 按日期过滤 → 返回标准 DataFrame
        # 文件不存在 → SymbolNotFoundError
        ...

    def get_latest(self, symbol: str) -> pd.Series:
        # 读取 Parquet → 返回最后一行
        ...
```

- 日期范围超出本地数据 → 返回可用部分，不报错
- 无缓存逻辑、无网络调用

## Error Types

```python
# src/oxq/core/errors.py

class OxqError(Exception):
    """框架基础异常"""

class SymbolNotFoundError(OxqError):
    """本地数据文件不存在"""

class DownloadError(OxqError):
    """数据下载失败"""
```

## Data Directory Resolution

```python
# src/oxq/data/loaders.py

def resolve_data_dir(dest_dir: Path | None = None) -> Path:
    if dest_dir is not None:
        return dest_dir
    env = os.environ.get("OXQ_DATA_DIR")
    if env:
        return Path(env) / "market"
    return Path.home() / ".oxq" / "data" / "market"
```

`LocalMarketDataProvider` 和所有 Downloader 共用此函数。

## File Manifest

| File | Content |
|------|---------|
| `src/oxq/core/errors.py` | `OxqError`, `SymbolNotFoundError`, `DownloadError` |
| `src/oxq/data/providers.py` | `MarketDataProvider` Protocol |
| `src/oxq/data/market.py` | `LocalMarketDataProvider` |
| `src/oxq/data/loaders.py` | `resolve_data_dir()`, `Downloader` Protocol, `YFinanceDownloader`, `AkShareDownloader` |
| `src/oxq/data/__init__.py` | Public API exports |

Not modified: `factors.py` — factor layer out of scope.

## Dependencies (pyproject.toml)

- Core: `pyarrow` (Parquet engine)
- Optional `[yfinance]`: `yfinance`
- Optional `[akshare]`: `akshare`

Install: `pip install open-xquant[yfinance]` or `pip install open-xquant[akshare]`

## Tests

- `tests/data/test_market.py` — LocalMarketDataProvider
- `tests/data/test_loaders.py` — Downloader (mock API calls)
