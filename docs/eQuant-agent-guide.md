# open-xquant × eQuant-Py Agent 使用指南

## 快速开始

open-xquant 现在以 eQuant-Py 作为底层计算引擎。Agent 通过 open-xquant 的声明式 API（Strategy Spec、CLI、Python SDK）间接调用 eQuant-Py 的全部能力。

```
Agent 工作流
    │
    ▼
open-xquant (声明式, 可审计)     ← 你在这里操作
    │
    ▼
eQuant-Py (函数式, 高性能)       ← 自动调用
```

---

## 方式一：Python SDK（推荐用于 Agent）

### 直接调用 eQuant 函数

Agent 可以通过 `oxq.adapters.EQuantAdapter` 直接调用 eQuant-Py 的 100+ 函数：

```python
from oxq.adapters.equant import EQuantAdapter, to_panel, from_panel

# 1. 解析函数引用
fn = EQuantAdapter.resolve("ettr::sma")       # → ettr.sma
fn = EQuantAdapter.resolve("eclassic::momentum")  # → eclassic.momentum
fn = EQuantAdapter.resolve("efactorcraft::ic_analysis")

# 2. 在面板数据上直接调用
import pandas as pd
result = ettr.sma(df, close_col="close", n=20)       # 添加 SMA_20 列
result = ettr.rsi(df, n=14)                           # 添加 RSI_14 列
result = ettr.macd(df)                                # 添加 MACD/MACD_signal/MACD_hist
result = ettr.bollinger(df, n=20, sd=2.0)            # 添加 BB_upper/BB_lower/BB_middle

# 3. 在 oxq per-symbol 数据上调用
panel = to_panel(mktdata, code="000001")
result = ettr.sma(panel, n=20, append=True)
sma_series = from_panel(result, "SMA_20", mktdata.index, code="000001")
```

### 可用函数速查

#### eTTR — 技术指标（58 个函数）

```python
import ettr

# 趋势类（18 个）
ettr.sma(df, n=20)           # 简单移动平均
ettr.ema(df, n=20)           # 指数移动平均
ettr.wma(df, n=10)           # 加权移动平均
ettr.dema(df, n=20)          # 双指数移动平均
ettr.hma(df, n=20)           # Hull 移动平均
ettr.alma(df, n=9)           # Arnaud Legoux 移动平均
ettr.macd(df, n_fast=12, n_slow=26, n_signal=9)  # MACD
ettr.adx(df, n=14)           # 平均趋向指数
ettr.gmma(df)                 # Guppy 多重移动平均（12 条线）
ettr.trix(df, n=15)           # 三重指数平滑
ettr.kst(df)                  # 确然指标（KST）
ettr.vhf(df, n=28)            # 水平/垂直滤波器

# 动量类（14 个）
ettr.rsi(df, n=14)           # 相对强弱指数
ettr.stoch(df, n_fast_k=14)  # 随机指标
ettr.kdj(df, n=9)            # KDJ 指标
ettr.cci(df, n=20)           # 商品通道指数
ettr.cmo(df, n=14)           # Chande 动量振荡器
ettr.tsi(df)                  # 真实强度指数
ettr.smi(df)                  # 随机动量指数
ettr.wpr(df, n=14)           # Williams %R
ettr.roc(df, n=10)           # 变化率
ettr.ultimate_oscillator(df) # 终极振荡器

# 波动类（7 个）
ettr.atr(df, n=14)           # 平均真实波幅
ettr.bollinger(df, n=20, sd=2.0)  # 布林带
ettr.keltner(df, n_ema=20)   # 肯特纳通道
ettr.donchian(df, n=20)      # 唐奇安通道
ettr.volatility(df, n=10, calc="parkinson")  # 波动率（6 种算法）

# 成交量（9 个）
ettr.obv(df)                  # 能量潮
ettr.cmf(df, n=20)           # 蔡金资金流
ettr.vwap(df)                 # 成交量加权均价
ettr.mfi(df, n=14)           # 资金流量指数

# 形态（4 个）
ettr.zigzag(df, change=0.1)  # ZigZag 转折点
ettr.sar(df)                  # 抛物线 SAR
ettr.pivots(df)               # 枢轴点
ettr.snr(df, n=20)            # 支撑阻力
```

#### eClassic — 经典因子（13 个函数）

```python
import eclassic

eclassic.momentum(df, n=252)       # 动量因子（支持多周期）
eclassic.value(df, bv_col="bv", cap_col="cap")     # 价值因子
eclassic.size(df, cap_col="cap")                   # 规模因子
eclassic.volatility(df, n=60)                      # 波动率因子
eclassic.beta(df, close_col="close", benchmark_col="bm", n=60)  # Beta
eclassic.profitability(df, op_col="op", bv_col="bv")  # 盈利因子
eclassic.investment(df, assets_col="assets", n=252)   # 投资因子
eclassic.return_(df, n=1, type="log")                 # 收益因子
eclassic.rps(df, n=60)            # 相对价格强度（截面排名）
eclassic.ram(df, n=252, risk="vol")  # 风险调整动量
eclassic.sma(df, n=20)            # 均线偏离
eclassic.benchmark(df)            # 基准超额收益
```

#### eFactorCraft — 因子工程（30+ 函数）

```python
import efactorcraft

# 预处理
efactorcraft.winsorize(df, factor_col="mom_20", probs=(0.01, 0.99))
efactorcraft.standardize(df, factor_col="mom_20")
efactorcraft.industry_neutralize(df, factor_col="mom_20", industry_col="industry")
efactorcraft.factor_preprocess(df, factor_col="mom_20")  # 一键预处理

# IC 分析
efactorcraft.add_next_return(df, close_col="close", periods=(1, 5, 20))
ic = efactorcraft.ic_analysis(df, factor_cols=["mom_20", "vol_60"], forward_col="forward_20")

# 因子合成
efactorcraft.equal_weighted_composite(df, factor_cols=["mom_20", "vol_60"])
efactorcraft.icir_weighted_composite(df, factor_cols=..., forward_col="forward_20")
efactorcraft.pca_composite(df, factor_cols=["mom_20", "vol_60"])

# 因子筛选
efactorcraft.ic_screen(df, factor_cols=..., forward_col="forward_20", min_abs_ic=0.02)
efactorcraft.correlation_screen(df, factor_cols=..., max_corr=0.7)
efactorcraft.factor_report(df, factor_cols=..., forward_col="forward_20")

# 市场状态
efactorcraft.regime_detect(df, close_col="close", ma_period=60, vol_period=20)
efactorcraft.timing_weight(df, factor_col="mom_20", regime_col="regime")
```

#### eBacktestCraft — 回测引擎

```python
import ebacktestcraft as ebc

cfg = ebc.Config(init_capital=1_000_000, rebalance_cycle="monthly")
df = ebc.signal(df, indicator_cols=["mom_20"], signal_type="quantile", top_n=30)
df = ebc.equal_weight(df, signal_col="signal_quantile")
result = ebc.run(df, config=cfg, weight_col="weight_equal")
metrics = ebc.performance_analysis(result.equity_curve)
```

#### eAlpha101 — 101 Alpha 因子

```python
import ealpha101

# 直接使用
df = ealpha101.add_alpha001(df)
df = ealpha101.add_alpha020(df)

# 按编号获取
alpha_fn = ealpha101.get_alpha("alpha049")
df = alpha_fn(df)

# 查看目录
print(ealpha101.summary())          # 101 个因子的速查表
print(ealpha101.ALPHAS["alpha020"]) # 单个因子的元数据
```

---

## 方式二：oxq Indicator 类（兼容现有 Agent 工作流）

Agent 可以通过 oxq 的注册表和 Engine 使用指标，无需关心底层是 eQuant 还是自实现：

```python
from oxq.core.registry import _INDICATOR_REGISTRY

# 获取并计算指标
sma_cls = _INDICATOR_REGISTRY["SMA"]
sma = sma_cls()
result = sma.compute(mktdata, column="close", period=20)  # 返回 pd.Series
```

所有 48 个注册的指标均可通过 `oxq.list_indicators()` 发现：

```python
import oxq

for name, cls in oxq.list_indicators().items():
    print(f"{name}: {cls.formula}")
```

---

## 方式三：Strategy Spec（声明式，Agent 工作流推荐）

最强大的方式——Agent 编写 YAML spec，oxq 自动编译并执行：

```yaml
# strategy_spec.yaml
strategy_id: "momentum-value-rotation"
name: "动量价值轮动"
hypothesis: "买入高动量低估值的股票，月度调仓"

market:
  asset_class: equity

universe:
  type: static
  symbols: ["000001", "000002", "600000", "600036", "000858"]

indicators:
  - name: SMA
    column: close
    period: 20
  - name: Momentum
    column: close
    period: 252
  - name: RPS
    column: close
    period: 60

signals:
  - name: Threshold
    column: RPS
    threshold: 80
    relationship: gt

portfolio:
  type: EqualWeight

execution:
  initial_cash: 1000000
  rebalance_cycle: monthly
```

CLI 命令：

```bash
oxq spec init "动量价值轮动策略"    # 从自然语言生成 spec 模板
oxq spec validate strategy_spec.yaml  # 验证 spec
oxq backtest run strategy_spec.yaml   # 运行完整回测
```

---

## 数据准备

### 使用 eFactorCraft 下载数据

```python
import pandas as pd
from efactorcraft import get_data

# 构建股票池
universe = pd.DataFrame({
    "code": ["000001", "000002", "600000"],
    "name": ["平安银行", "万科A", "浦发银行"],
})

# 下载 A 股数据
df = get_data(universe, start_date="2024-01-01", end_date="2024-12-31",
              source="akshare")  # yahoo | akshare | tushare | baostock

# 数据格式：
# date       | code   | open | high | low | close | adjusted | volume
# 2024-01-02 | 000001 | 10.5 | 10.8 | 10.2| 10.6  | 10.6     | 123456
# 2024-01-02 | 000002 | 8.2  | 8.5  | 8.1 | 8.3   | 8.3      | 234567
```

### 使用 edatatools 构建交易日历

```python
from oxq.market_calendar_equant import get_calendar, date_range, is_trading_day

cal = get_calendar("CN")

# 判断是否交易日
cal.is_trading_day("2024-10-01")  # → False (国庆节)

# 获取交易日期范围
dates = cal.trading_days("2024-01-01", "2024-12-31")

# 距离下一个交易日
next_day = cal.next_trading_day("2024-01-01", shift=1)  # → 2024-01-02
```

### 使用 oxq EQuantMarketDataProvider

```python
from oxq.data.equant_provider import EQuantMarketDataProvider

provider = EQuantMarketDataProvider(
    source="akshare",
    timezone="Asia/Shanghai",
    currency="CNY",
)

# 获取单只股票数据
bars = provider.get_bars("000001", "2024-01-01", "2024-12-31")
# → DataFrame with tz-aware DatetimeIndex, cols: open/high/low/close/volume
```

---

## 完整 Agent 工作流示例

以下是一个完整的因子研究 → 回测 Agent 工作流：

```python
"""Agent 工作流：寻找有效的量价因子并回测"""

import pandas as pd
import ettr, eclassic, efactorcraft, ebacktestcraft as ebc

# ── Step 1: 获取数据 ──
universe = pd.DataFrame({
    "code": ["000001", "000002", "600000", "600036", "000858"],
    "name": ["平安银行", "万科A", "浦发银行", "招商银行", "五粮液"],
})
df = efactorcraft.get_data(universe, "2024-01-01", "2024-12-31", source="akshare")

# ── Step 2: 计算候选因子 ──
df = ettr.rsi(df, n=14)
df = eclassic.momentum(df, n=252)
df = eclassic.volatility(df, n=60)
df = eclassic.rps(df, n=120)
df = ettr.mfi(df, n=14)

# ── Step 3: 因子预处理 ──
factor_cols = ["mom_252", "vol_60", "rps_120"]
for col in factor_cols:
    df = efactorcraft.winsorize(df, factor_col=col, probs=(0.01, 0.99))
    df = efactorcraft.standardize(df, factor_col=f"win_{col}")

# ── Step 4: IC 分析（评估因子有效性）──
df = efactorcraft.add_next_return(df, close_col="close", periods=(5, 20))
ic_result = efactorcraft.ic_analysis(
    df, factor_cols=[f"std_win_{c}" for c in factor_cols],
    forward_col="forward_20", method="spearman"
)

# ── Step 5: 生成交易信号 ──
df = ebc.signal(df, indicator_cols=[f"std_win_{c}" for c in factor_cols],
                signal_type="quantile", top_n=10)

# ── Step 6: 分配权重 ──
df = ebc.equal_weight(df, signal_col="signal_quantile")

# ── Step 7: 回测 ──
cfg = ebc.Config(init_capital=1_000_000, rebalance_cycle="monthly",
                 fee_rate=0.0003, slippage_rate=0.001)
result = ebc.run(df, config=cfg, weight_col="weight_equal_signal_quantile")

# ── Step 8: 输出结果 ──
metrics = ebc.performance_analysis(result.equity_curve)
print(f"年化收益: {metrics.get('annual_return', 0):.2%}")
print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")

# ── Step 9: 反思与迭代 ──
# Agent 根据 IC 分析结果，调整因子组合、调仓周期，或尝试新的因子
```

---

## 函数引用语法

`EQuantAdapter.resolve()` 支持 `"pkg::func"` 格式：

```python
from oxq.adapters.equant import EQuantAdapter

# 别名映射
"indicator::sma"      → ettr.sma
"ettr::macd"          → ettr.macd
"classic::momentum"   → eclassic.momentum
"factor::volatility"  → eclassic.volatility
"alpha::add_alpha020" → ealpha101.add_alpha020
"engineering::ic_analysis" → efactorcraft.ic_analysis
"backtest::run"       → ebacktestcraft.run
"data::date_range"    → edatatools.date_range
```

---

## 数据约定

所有 eQuant 函数共享**统一的长格式面板约定**：

| 要求 | 说明 |
|------|------|
| 必需列 | `date` (datetime), `code` (str) |
| OHLCV 列 | `open`, `high`, `low`, `close`, `volume`（小写） |
| 返回 | 输入 DataFrame + 追加的指示器列 |
| 时区 | 日期为 tz-naive（无时区） |

**oxq ↔ eQuant 转换** 由 `EQuantAdapter` 自动处理。

---

## 与 INTEGRATION.md 的关系

[INTEGRATION.md](INTEGRATION.md) 是集成设计文档，描述了 eQuant-Py 各子包的完整 API。本指南是面向 Agent 开发者的操作手册。两者的关系：

- **INTEGRATION.md** → 了解 eQuant-Py 提供了什么
- **本指南** → 了解 Agent 如何使用这些能力
- **[eQuant-integration-dev.md](eQuant-integration-dev.md)** → 了解底层的代码变更
