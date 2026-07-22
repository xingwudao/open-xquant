# open-xquant × eQuant-Py 集成开发者文档

## 概述

本次变更将 eQuant-Py 集成为 open-xquant 的**底层量化工具层**。oxq 保留自己的公共 API（Strategy Spec 编译、Engine 管道、CLI、Protocol 体系），但指标计算、因子工程、数据获取等底层实现委托给 eQuant-Py 子包。

### 架构变更

```
Before (oxq 自实现)                After (eQuant-Py 底层)
─────────────────────              ──────────────────────
indicator.compute()                indicator.compute()
  ├─ 手写 rolling/ewm               ├─ to_panel(mktdata)
  └─ numpy 直接计算                  ├─ ettr.sma(df, n=20)
                                    └─ from_panel(result)

数据加载: YFinance/AkShare           数据加载: efactorcraft.get_data()
因子评估: 纯 pandas/scipy            因子评估: efactorcraft 函数
```

**关键设计决策：Protocol 桥接模式。** eQuant-Py 使用函数式面板 DataFrame 范式（`func(df) -> df`），而 oxq 使用面向对象的 Protocol 范式（`class Indicator: def compute(self, mktdata) -> Series`）。适配器层（`src/oxq/adapters/equant.py`）通过 `to_panel` / `from_panel` 转换桥接两种范式，无需修改 Engine 核心逻辑。

---

## 新增文件

### 1. `src/oxq/adapters/equant.py` — 核心适配器

| 导出 | 类型 | 说明 |
|------|------|------|
| `to_panel(mktdata, code)` | 函数 | per-symbol DataFrame → eQuant 长格式面板 |
| `from_panel(result, col, index, code)` | 函数 | eQuant 面板列 → per-symbol Series（对齐原索引） |
| `compute_panel_batch(fn, mktdata_dict, col)` | 函数 | 多 symbol 批量计算：一次 eQuant 调用处理全部 |
| `EQuantAdapter` | 类 | 函数引用解析（`"ettr::sma"` → `ettr.sma`）+ 执行 |

**数据转换细节：**

```
oxq per-symbol mktdata                    eQuant panel
────────────────────                      ────────────
DateTimeIndex(tz=Asia/Shanghai)           "date" column (tz-naive)
columns: [open,high,low,close,volume]     columns: [date,code,open,high,low,close,volume]
                                          one row per (symbol, date) pair
```

`from_panel` 通过 `dt.tz_localize(original_index.tz)` 而非 `tz_localize("UTC") + tz_convert()` 对齐时区，避免日期偏移。

### 2. `src/oxq/data/equant_provider.py` — eQuant 数据源

实现 `MarketDataProvider` 协议，通过 `efactorcraft.get_data()` 下载数据并缓存为 Parquet：

```python
provider = EQuantMarketDataProvider(
    source="akshare",      # yahoo | akshare | tushare | baostock
    timezone="Asia/Shanghai",
    currency="CNY",
)
engine.run(strategy, market=provider, ...)
```

支持两种模式：
- **本地缓存**：读取 `{data_dir}/{SYMBOL}.parquet`（兼容 LocalMarketDataProvider）
- **在线获取**：`efactorcraft.get_data()` 下载 → 缓存 → 返回

### 3. `src/oxq/factor_eval/equant.py` — eFactorCraft 因子评估

提供便捷函数，接受 oxq 的 `FactorBundle` 或长格式面板 DataFrame：

| 函数 | eFactorCraft 对应 | 说明 |
|------|------------------|------|
| `compute_ic_equant(bundle)` | `ic_analysis` | Pearson/Spearman IC |
| `factor_preprocess(df, factor_col)` | winsorize→standardize→neutralize | 一键预处理 |
| `synthesize_factors(df, factor_cols, method)` | equal/ic/icir/pca 合成 | 因子合成 |
| `screen_factors(df, factor_cols)` | IC+corr+stability 筛选 | 因子筛选报告 |
| `add_next_return_equant(df)` | `add_next_return` | 添加 forward return 列 |
| `detect_regime(df)` | `regime_detect` | 牛/熊/震荡市检测 |

### 4. `src/oxq/market_calendar_equant.py` — A 股交易日历

包装 `edatatools.cn_calendar`，提供 oxq 风格的接口：

```python
from oxq.market_calendar_equant import get_calendar, date_range, is_trading_day

cal = get_calendar("CN")
cal.is_trading_day("2024-01-01")   # → False (元旦)
cal.trading_days("2024-01", "2024-01")  # → DatetimeIndex of trading days
```

### 5. 外部修改：eQuant-Py `equant/pyproject.toml`

eQuant-Py 的 `equant/` 目录原本没有构建配置，导致 `pip install -e equant` 失败。新增了最小 `pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "equant"
version = "0.1.0"
requires-python = ">=3.9"

[tool.setuptools]
packages = ["equant", "equant.utils"]
[tool.setuptools.package-dir]
"equant" = "."
```

该文件位于 `/Users/tool/Desktop/eQuant-Py/equant/pyproject.toml`，需要在 eQuant-Py 仓库中提交。

---

## 修改的文件

### `pyproject.toml`

```toml
dependencies = [
    # ...原有依赖...
    "eTTR>=0.1.0",           # 技术指标（55+）
    "eClassic>=0.1.0",       # 经典因子
    "eFactorCraft>=0.1.0",   # 因子工程
    "eBacktestCraft>=0.1.0", # 回测引擎
    "edatatools>=0.1.0",     # 交易日历/数据工具
    "equant>=0.1.0",         # 统一入口/共享工具
]

[project.optional-dependencies]
equant = [
    "eAlpha101>=0.1.0",     # 101 Alpha 因子（可选）
    "eCandleSticks>=0.1.0", # K线形态（需要 TA-Lib C 库）
    "eFinCharts>=0.1.0",    # 金融图表
]
```

### 指标文件

**由 eQuant 支持的（重写为封装器，26 个）：**

| 文件 | 类 | eQuant 函数 | 说明 |
|------|----|-----------|------|
| `sma.py` | SMA | `ettr.sma` | 简单移动平均 |
| `builtin.py` | EMA, WMA, DEMA, TEMA | `ettr.ema/wma/dema` | TEMA 通过 EMA×3 合成 |
| `builtin.py` | RSI, ROC, PPO | `ettr.rsi/roc/po_` | PPO 使用 `ettr.po_` |
| `builtin.py` | MACDLine/Signal/Histogram | `ettr.macd` | 拆分为 3 个独立类 |
| `builtin.py` | BollingerUpper/Lower | `ettr.bollinger` | 分别提取上下轨 |
| `builtin.py` | ATR | `ettr.atr` | Wilder 平滑 |
| `builtin.py` | OBV, VWAP, MFI | `ettr.obv/vwap/mfi` | VWAP 支持滚动窗口 |
| `builtin.py` | ADX, AROON, CCI, StochK | `ettr.adx/aroon/cci/stoch` | 提取单列 |
| `momentum.py` | Momentum | `eclassic.momentum(type="log")` | 对数收益率动量 |
| `nday_return.py` | NdayReturn | `eclassic.return_(type="log")` | N 日对数收益 |
| `log_return.py` | LogReturn | `eclassic.return_(type="log", n=1)` | 单日对数收益 |
| `simple_momentum.py` | SimpleMomentum | `eclassic.momentum(type="continuous")` | 简单收益率动量 |
| `rps.py` | RPS | `eclassic.rps` | 截面排名（保留 `compute_cross_section`） |
| `rolling_volatility.py` | RollingVolatility | `eclassic.volatility(type="sd")` | 滚动标准差 |
| `annualized_volatility.py` | AnnualizedVolatility | `eclassic.volatility` × sqrt(252) | 年化波动率 |

**保留原实现（无 eQuant 等价物，15 个）：**

Ichimoku（5 类）、GarchVolatility、HurstExponent、RollingMDD、PE、PB、EP、BP、MarketCap、AccrualRatio、CashFlowRatio、NetProfitMargin、ROEChange、Ratio、PowerRatio、TurnoverRate

### 兼容性修复

| 文件 | 修改 | 原因 |
|------|------|------|
| `data/manifest.py` 等 8 个文件 | `from datetime import UTC, datetime` → `from datetime import datetime, timezone; UTC = timezone.utc` | `datetime.UTC` 仅在 Python 3.11+ 可用；`timezone.utc` 自 3.2 起可用 |
| `core/registry.py` | `entry_points(group=group)` → try/except 兼容 3.9-3.11 的 `entry_points()` 调用 | `importlib.metadata.entry_points(group=...)` 关键字参数仅在 3.12+ 可用 |

---

## 封装器模式详解

每个由 eQuant 支持的指标遵循统一模式：

```python
class SMA:
    name = "SMA"                    # 保持不变
    formula = r"SMA_t = ..."        # 保持不变

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        import ettr                                    # 延迟导入（仅在计算时加载）
        panel = to_panel(mktdata)                      # ① per-symbol → 迷你面板
        result = ettr.sma(panel, close_col=column,     # ② 调用 eQuant
                         n=period, append=True)
        return from_panel(result, f"SMA_{period}",     # ③ 提取列 + 对齐索引
                          mktdata.index)
```

**注意：**
- eQuant 导入是**延迟的**（在 `compute()` 内部 `import ettr`），而非在模块顶层。这使 eQuant 成为运行时依赖，而非硬性导入时依赖。
- eQuant 函数的列名约定各不相同：
  - `ettr.sma(n=20)` → 列名 `SMA_20`（包含后缀）
  - `ettr.bollinger(n=20)` → 列名 `BB_upper, BB_lower, BB_middle`（无 `_20` 后缀）
  - `ettr.stoch()` → 列名 `Stoch_fastK, Stoch_fastD, Stoch_slowD`（无后缀）
  - `eclassic.momentum(n=20)` → 列名 `mom_20`（包含后缀）
  - 每个封装器的 `from_panel` 调用使用正确的**确切列名**

---

## Engine 集成

Engine（`src/oxq/core/engine.py`）**无需修改**。现有指标循环：

```python
for ind_name, (indicator, params) in all_indicators.items():
    if callable(compute_cross_section):
        outputs = compute_cross_section(self._mktdata, **params)  # RPS 等
    else:
        for symbol in self._universe.symbols:
            self._mktdata[symbol][ind_name] = indicator.compute(
                self._mktdata[symbol], **params,
            )
```

每个封装器在每个 symbol 上独立调用 `compute()`，在内部执行 `to_panel → eQuant → from_panel`。这保证了正确性，但对每个 symbol 调用一次 eQuant 会带来开销。

**未来优化：** `compute_panel_batch()`（`adapters/equant.py` 内）可将所有 symbol 堆叠为单个面板，调用 eQuant 一次，然后拆分结果。这需要在 Engine 中添加对 `_equant_func` 属性或 `compute_panel` 方法的检测。

---

## 测试策略

### 测试内容

1. **单元：** 每个指标封装器——实例化、计算、验证输出形状和非空值
2. **集成：** 完整的 spec → compile → Engine.run 管道
3. **回归：** 现有 `tests/indicators/test_*.py` 套件（预期结果可能需要因 eQuant 与手写实现的数值差异而调整）

### 测试方式

```bash
# 安装开发依赖
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

# 运行指标测试
uv run pytest tests/indicators/ -v

# 运行完整回归
uv run pytest tests/ -v -m "not e2e and not integration"
```

### 已知差异

- eTTR 在滚动窗口起始处处理 NaN 的方式可能略有不同
- MACD 输出现在来自 eTTR 的 `macd()` 函数（产生 `MACD`、`MACD_signal`、`MACD_hist`），而非 3 个独立的逐段实现
- eclassic.rps 返回百分比排名 [0, 100]，可能需要缩放以匹配原有 oxq 行为

---

## Python 3.12 兼容性

所有 eQuant-Py 子包声明 `requires-python = ">=3.9"`，与 open-xquant 的 `>=3.12` 要求完全兼容。已验证的依赖链：

| 包 | 版本 | Python 3.12 支持 |
|---|---|---|
| eTTR | 0.1.0 | ✅（numba>=0.58 支持 3.12） |
| eClassic | 0.1.0 | ✅（仅 pandas/numpy） |
| eFactorCraft | 0.1.0 | ✅（scipy>=1.10, statsmodels>=0.14 支持 3.12） |
| eBacktestCraft | 0.1.0 | ✅（matplotlib>=3.7, seaborn>=0.12） |
| edatatools | 0.1.0 | ✅（仅 pandas/numpy） |

---

## 回滚

还原所有更改：

```bash
git checkout -- src/oxq/ pyproject.toml
```

或仅回滚指标层同时保留适配器：

```bash
git checkout -- src/oxq/indicators/
```
