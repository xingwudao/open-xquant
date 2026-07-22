# open-xquant 集成 eQuant-Py 指南

eQuant-Py 是 open-xquant 的**默认量化工具层**，提供数据→因子→回测全链路 API。open-xquant 作为 Agent/工作流层，通过 import 直接调用。

---

## 1. 安装

### 方式一：git clone + editable install（开发推荐）

```bash
git clone https://github.com/dengyishuo/eQuant-Py.git
cd eQuant-Py

# 安装所有子包（可编辑模式）
pip install -e eTTR-Py \
            -e eClassic-Py \
            -e eAlpha101-Py \
            -e eFactorCraft-Py \
            -e eBacktestCraft-Py \
            -e edatatools \
            -e eCandleSticks-Py \
            -e eFinCharts-Py \
            -e .

# 可选依赖
pip install akshare mplfinance
brew install ta-lib          # eCandleSticks 需要（macOS）
pip install TA-Lib
```

### 方式二：pip（发布后）

```bash
pip install eQuant
```

---

## 2. 架构关系

```
open-xquant（Agent / 工作流层）
    │
    │  import / 函数调用
    ▼
eQuant-Py（工具层）
    ├── equant            统一入口 + 共享工具
    ├── edatatools        交易日历、股票池、收益数据
    ├── eTTR              55+ 技术指标
    ├── eClassic          13 类经典因子
    ├── eAlpha101         101 个 WorldQuant Alpha
    ├── eCandleSticks     50+ K线形态识别
    ├── eFactorCraft      因子工程（预处理/合成/筛选/择时）
    ├── eBacktestCraft    事件驱动回测引擎
    ├── eFinCharts        金融图表可视化
    └── webapp            Streamlit 工作台
```

---

## 3. 统一数据规范

所有子包共享**长格式面板 DataFrame**：

| date | code | open | high | low | close | volume | factor_cols... |
|------|------|------|------|-----|-------|--------|---------------|
| 2024-01-02 | 000001 | 10.5 | 10.8 | 10.2 | 10.6 | 123456 | ... |
| 2024-01-02 | 000002 | 8.2  | 8.5  | 8.1  | 8.3  | 234567 | ... |
| 2024-01-03 | 000001 | 10.6 | 10.9 | 10.4 | 10.8 | 234567 | ... |

函数签名统一为 `func(df, ...) -> df`（输入输出都是 DataFrame，链式调用）。

---

## 4. Agent 调用模式

### 4.1 直接函数调用（推荐）

```python
from eTTR          import sma, ema, macd, rsi, atr, bollinger, kdj, stoch
from eClassic      import momentum, value, size, volatility, rps, ram
from eAlpha101     import add_alpha001, add_alpha020, add_alpha101
from eFactorCraft  import winsorize, standardize, ic_analysis, icir_weighted_composite
from eBacktestCraft import Config, run, signal, equal_weight, performance_analysis, plot_all
```

### 4.2 模块别名调用

```python
import ettr as indicators
import eclassic as factors
import ealpha101 as alpha101
import efactorcraft as eng
import ebacktestcraft as bt
```

### 4.3 通过 equant 统一入口（懒加载）

```python
import equant

# 子包按需加载
equant.ettr.sma(df, n=20)
equant.eclassic.momentum(df, n=252)
equant.ebacktestcraft.Config(init_capital=1_000_000)
```

---

## 5. Agent 工作流示例

### 5.1 典型分析链路

```python
import pandas as pd
from edatatools       import date_range, build_universe
from efactorcraft     import get_data
from ettr             import rsi, macd
from eclassic         import momentum, volatility
from efactorcraft     import winsorize, standardize, ic_analysis
from ebacktestcraft   import Config, run, signal, equal_weight

# ── Agent Step 1: 获取股票池 ──
dates = date_range("2024-01-01", "2024-12-31", region="CN")
universe = build_universe("2024-01-01", "2024-06-30", univ_type="csi300")

# ── Agent Step 2: 拉取数据 ──
df = get_data(codes=universe, start="20240101", end="20241231")

# ── Agent Step 3: 计算因子 ──
df = rsi(df, n=14)
df = macd(df)
df = momentum(df, n=252)
df = volatility(df, n=60)

# ── Agent Step 4: 因子工程 ──
factor_cols = ["RSI_14", "mom_252", "vol_60"]
for col in factor_cols:
    df = winsorize(df, factor_col=col, probs=(0.01, 0.99))
    df = standardize(df, factor_col=col)

# ── Agent Step 5: IC 分析（评估因子有效性）──
df = efactorcraft.add_next_return(df, close_col="close", periods=[5, 20])
ic = ic_analysis(df, factor_cols=factor_cols, forward_col="forward_20")

# ── Agent Step 6: 生成信号 ──
df = signal(df, indicator_cols=factor_cols, signal_type="quantile", top_n=30)

# ── Agent Step 7: 计算权重 ──
df = equal_weight(df, signal_col="signal_quantile")

# ── Agent Step 8: 回测 ──
config = Config(init_capital=1_000_000, rebalance_cycle="monthly")
result = run(df, config=config, weight_col="weight_equal_signal_quantile")

# ── Agent Step 9: 输出结果 ──
print(result.summary)
return result.equity_curve, result.summary
```

### 5.2 Agent 工具注册（供 LLM Function Calling）

```python
AGENT_TOOLS = [
    {
        "name": "calculate_technical_indicators",
        "description": "对面板数据计算技术指标",
        "parameters": {
            "df": "长格式面板 DataFrame（需含 open/high/low/close/volume）",
            "indicators": ["sma", "ema", "macd", "rsi", "atr", "bollinger", "kdj"],
        },
        "call": lambda df, indicators: _apply_indicators(df, indicators),
    },
    {
        "name": "run_backtest",
        "description": "对带有信号和权重的 DataFrame 执行回测",
        "parameters": {
            "df": "含信号列和权重列的 DataFrame",
            "init_capital": "初始资金（默认 100 万）",
            "rebalance_cycle": "调仓周期 daily/monthly/quarterly",
        },
        "call": lambda df, **kw: _run_backtest(df, **kw),
    },
    {
        "name": "evaluate_factor_quality",
        "description": "评估因子质量：IC 均值、IR、胜率",
        "parameters": {
            "df": "含因子列和 forward 收益列的 DataFrame",
            "factor_cols": ["mom_252", "vol_60"],
        },
        "call": lambda df, factor_cols: ic_analysis(df, factor_cols),
    },
]
```

---

## 6. 各子包可用类与方法速查

### edatatools — 数据基础设施

| 函数 | 说明 |
|------|------|
| `date_range(start, end, region)` | 交易日历（需 akshare 或 Tushare token） |
| `build_universe(start, end, univ_type, **kw)` | 构建股票池（csi300/csi500/fixed/all） |
| `TradingCalendar(region, dates)` | 自定义交易日历对象 |
| `get_cacs_return(codes, start, end)` | CACS 复权收益 |

### eTTR — 技术指标 (55+)

```python
from ettr import (
    # 趋势类：sma, ema, dema, wma, hma, zlema, alma, evwma, vwma,
    #         macd, adx, gmma, tdi, trix, dpo, vhf, kst, po_
    # 动量类：rsi, cci, cmo, tsi, smi, wpr, ultimate_oscillator,
    #         roc, momentum, cti, rvi, dvi, stoch, kdj
    # 波动类：atr, tr, bollinger, keltner, donchian, pbands, volatility
    # 成交量：obv, cmf, vwap, mfi, emv, clv, chaikin_ad, williams_ad
    # 形态类：zigzag, pivots, sar, snr
    # 杂项类：aroon, growth, td_setup, td_countdown, lags, roll_sfm
)
```

### eClassic — 经典因子

```python
from eclassic import (
    momentum,     # 动量因子 mom_N
    value,        # 价值因子（bv/cap）
    size,         # 规模因子 log(cap)
    volatility,   # 波动率因子 vol_N
    beta,         # 市场 Beta
    rps,          # 相对强度 rps_N（横截面排名 0-1）
    ram,          # 移动平均偏离 ram_N
    return_,      # 收益率因子 ret_N
    profitability,# 盈利因子
    investment,   # 投资因子
    sma,          # 均线偏离
    benchmark,    # 基准组合
    slope,        # 斜率因子
)
```

### eAlpha101 — 101 Alpha

```python
from ealpha101 import (
    add_alpha001, add_alpha002, ..., add_alpha101,  # 逐个导入
    ALPHAS,       # 全部 101 个的元数据目录
    summary,      # 因子摘要
    get_alpha,    # 按编号获取因子函数
)
# 或统一入口
from ealpha101.formulas import alpha001, alpha002, ..., alpha101
```

### eCandleSticks — K线形态

```python
from ecandlesticks import (
    scan_all,              # 扫描所有形态
    add_doji,              # 十字星
    add_hammer,            # 锤子线
    add_engulfing,         # 吞没形态
    add_morning_star,      # 晨星
    add_evening_star,      # 暮星
    add_harami,            # 孕线
    # ... 50+ 种形态
)
```

### eFactorCraft — 因子工程

```python
# 预处理
from efactorcraft import winsorize, standardize, industry_neutralize, size_neutralize, factor_preprocess

# 分析
from efactorcraft import ic_analysis, ir_analysis, add_next_return, quantile_analysis

# 排名
from efactorcraft import quantile_rank, quantile_flag, consecutive_days

# 数据获取
from efactorcraft import get_data  # 统一数据接口，聚合 Tushare/akshare/baostock

# 合成
from efactorcraft.synthesis import (
    equal_weighted_composite,   # 等权合成
    rank_weighted_composite,    # 排名加权
    ic_weighted_composite,      # IC 加权
    icir_weighted_composite,    # IC_IR 加权
    pca_composite,              # PCA 合成
    max_decay_composite,        # 最大衰减合成
)

# 筛选
from efactorcraft.selection import (
    ic_screen,                  # IC 筛选
    correlation_screen,         # 相关性筛选
    stability_screen,           # 稳定性筛选
    select_top,                 # Top-N 选择
    factor_report,              # 因子报告
)

# 择时
from efactorcraft.timing import (
    regime_detect,              # 市场状态检测（bull/bear/sideways）
    trend_filter,               # 趋势过滤器
    vol_filter,                 # 波动率过滤器
    timing_weight,              # 动态权重
    adaptive_composite,         # 自适应合成
)
```

### eBacktestCraft — 回测引擎

```python
from ebacktestcraft import (
    Config,           # 回测配置（init_capital, rebalance_cycle, lot_size...）
    BacktestConfig,   # 同 Config
    run,              # 执行回测 → BacktestResult
    BacktestResult,   # 结果对象（.equity_curve, .summary, .trades）
    signal,           # 信号生成（25 种类型）
    equal_weight,     # 等权
    fixed_weight,     # 固定权重
    norm_weight,      # 归一化权重
    # 增强权重
    vol_parity_weight,
    target_vol_weight,
    erp_weight,
    confidence_weight,
    risk_parity_weight,
    min_variance_weight,
    max_sharpe_weight,
    # 风险控制
    apply_vol_target,
    compute_turnover,
    # 分析
    performance_analysis,
    buy_and_hold_benchmark,
    compare_benchmarks,
    # 可视化
    plot_all, plot_equity_curve, plot_drawdown,
    plot_monthly_return, plot_return_dist, plot_benchmark_compare,
)
```

---

## 7. 继承与扩展

### 7.1 自定义因子

```python
from equant.utils.panel import validate_panel, slim_output
import numpy as np

def my_custom_factor(df, close_col=None, n=20):
    """自定义因子：价格偏离均线的标准差倍数"""
    from ettr._panel import _resolve_col
    col = _resolve_col(df, "close", close_col)
    result = df.copy()
    result["my_factor"] = np.nan
    for code, idx in result.groupby("code", sort=False).groups.items():
        vals = result.loc[idx, col].values.astype(np.float64)
        ma = pd.Series(vals).rolling(n, min_periods=n).mean().values
        std = pd.Series(vals).rolling(n, min_periods=n).std().values
        result.loc[idx, "my_factor"] = (vals - ma) / (std + 1e-15)
    return slim_output(result, "my_factor")
```

### 7.2 自定义 Agent 工作流

```python
class QuantAgent:
    """继承 eQuant-Py 的量化 Agent 基类"""

    def __init__(self, config: dict):
        self.universe = config.get("universe", [])
        self.start = config.get("start", "20240101")
        self.end = config.get("end", "20241231")
        self.factors = config.get("factors", ["mom_252", "vol_60"])

    def load_data(self) -> "pd.DataFrame":
        from efactorcraft import get_data
        return get_data(codes=self.universe, start=self.start, end=self.end)

    def compute_factors(self, df: "pd.DataFrame") -> "pd.DataFrame":
        from eclassic import momentum, volatility
        df = momentum(df, n=252)
        df = volatility(df, n=60)
        return df

    def preprocess(self, df: "pd.DataFrame") -> "pd.DataFrame":
        from efactorcraft import winsorize, standardize
        for col in self.factors:
            df = winsorize(df, factor_col=col)
            df = standardize(df, factor_col=col)
        return df

    def generate_signals(self, df: "pd.DataFrame") -> "pd.DataFrame":
        from ebacktestcraft import signal, equal_weight
        df = signal(df, indicator_cols=self.factors,
                     signal_type="quantile", top_n=30)
        sig_col = [c for c in df.columns if c.startswith("signal_")][0]
        df = equal_weight(df, signal_col=sig_col)
        return df

    def backtest(self, df: "pd.DataFrame") -> dict:
        from ebacktestcraft import Config, run
        weight_col = [c for c in df.columns if c.startswith("weight_")][0]
        config = Config(init_capital=1_000_000, rebalance_cycle="monthly")
        result = run(df, config=config, weight_col=weight_col)
        return {"equity": result.equity_curve, "summary": result.summary}

    def run(self) -> dict:
        df = self.load_data()
        df = self.compute_factors(df)
        df = self.preprocess(df)
        df = self.generate_signals(df)
        return self.backtest(df)
```

### 7.3 Open-Xquant Agent 注册模板

```python
# open_xquant/agents/quant_agent.py

from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class FactorConfig:
    name: str
    package: str      # "ettr" | "eclassic" | "ealpha101"
    func: str         # "sma" | "rsi" | "momentum" | "add_alpha001"
    params: Dict[str, Any]

class EQuantAdapter:
    """open-xquant 到 eQuant-Py 的适配层"""

    PACKAGE_MAP = {
        "ettr":           "ettr",
        "indicator":      "ettr",
        "classic":         "eclassic",
        "factor":          "eclassic",
        "alpha":           "ealpha101",
        "engineering":     "efactorcraft",
        "backtest":        "ebacktestcraft",
        "candlestick":     "ecandlesticks",
        "data":            "edatatools",
    }

    @staticmethod
    def resolve(func_ref: str):
        """解析 "ettr::sma" → (module, function)"""
        pkg, _, name = func_ref.partition("::")
        mod = __import__(EQuantAdapter.PACKAGE_MAP.get(pkg, pkg))
        return getattr(mod, name)

    @classmethod
    def execute(cls, func_ref: str, df, **params):
        """Agent 通用执行入口"""
        fn = cls.resolve(func_ref)
        return fn(df, **params)
```

---

## 8. 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥ 3.9 | |
| pandas | ≥ 1.5 | 核心数据结构 |
| numpy | ≥ 1.23 | 数值计算 |
| numba | ≥ 0.58 | eTTR 滚动计算加速 |
| scipy | ≥ 1.10 | eFactorCraft PCA/统计 |
| statsmodels | ≥ 0.14 | 因子中性化回归 |
| matplotlib | ≥ 3.7 | eBacktestCraft 图表 |
| yfinance | ≥ 0.2.30 | 境外数据 |
| akshare | latest | A 股数据（edatatools） |
| mplfinance | latest | K 线图（eFinCharts） |
| TA-Lib | C 库 | K 线形态（eCandleSticks） |

---

## 9. 相关链接

- eQuant-Py: https://github.com/dengyishuo/eQuant-Py
- 数据规范详见各子包 README
- 测试: `PYTHONPATH=<pkgs> python3 -m pytest */tests/ -v`
