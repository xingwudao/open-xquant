# open-xquant 架构文档

## 1. 设计哲学

open-xquant 是一个 **Agent-First** 的开源量化交易框架。底层是严谨的量化金融引擎，由 Universe（标的池）决定每个时间截面上参与计算的 symbol 集合，再经 Indicator → Signal → Rule 三阶段模型生成交易决策；核心资产是 **Python SDK + 协议无关的 Tool 定义**（名称、参数、语义），每个工作流编写 skill.md，指导 Agent 如何组合 tools 完成复杂任务。

**三种使用角色与入口**：

- **Coding Agent / 开发者** → `import oxq`（主要方式），直接调用 SDK 和 Tool 函数
- **非 Coding AI 客户端**（Claude Desktop、Windsurf 等）→ 通过 MCP Server 调用（可选分发层）
- **平台方** → 基于 SDK + Tool 定义自建接口（REST API、gRPC 等）

**四大设计原则**：

- **声明式**：策略定义与执行分离。策略是"做什么"的声明，引擎负责"怎么做"
- **确定性**：相同输入必须产生相同输出。不可变数据类型 + 纯函数计算 + 审计追踪
- **约束即自由**：统一的 Protocol 接口收窄 AI 的选择空间到只有正确的做法，消除幻觉温床
- **全流程**：从策略构建、回测、参数优化、统计检验到交易执行，端到端覆盖

---

## 2. 项目结构

```
open-xquant/
├── src/
│   └── oxq/                        # 主 Python 包（pip install open-xquant）
│       ├── __init__.py
│       ├── core/                    # 核心引擎
│       │   ├── types.py            # 基础数据类型
│       │   ├── strategy.py         # 策略定义与编译
│       │   ├── engine.py           # 策略执行引擎
│       │   ├── registry.py         # 组件注册中心
│       │   └── errors.py           # 异常体系
│       │
│       ├── indicators/             # 技术指标库
│       │   ├── base.py             # Indicator 基类
│       │   ├── builtin.py          # 内置指标 (SMA, EMA, RSI, MACD, BBands...)
│       │   └── custom.py           # 自定义指标支持
│       │
│       ├── signals/                # 信号生成器
│       │   ├── base.py             # Signal 基类
│       │   ├── crossover.py        # 交叉信号 (sigCrossover)
│       │   ├── threshold.py        # 阈值信号 (sigThreshold)
│       │   ├── comparison.py       # 比较信号 (sigComparison)
│       │   ├── formula.py          # 公式信号 (sigFormula)
│       │   └── composite.py        # 组合信号
│       │
│       ├── rules/                  # 交易规则
│       │   ├── base.py             # Rule 基类
│       │   ├── entry.py            # 入场规则
│       │   ├── exit.py             # 出场规则（止损/止盈/追踪止损）
│       │   ├── sizing.py           # 仓位管理（osMaxPos, 等权, 风险平价...）
│       │   ├── risk.py             # 风控规则（泡沫控制、因子风控）
│       │   └── rebalance.py        # 再平衡规则
│       │
│       ├── portfolio/              # 组合管理
│       │   ├── types.py            # Position, Portfolio, OrderBook
│       │   ├── portfolio.py        # 组合状态管理
│       │   ├── orderbook.py        # 订单簿（订单生命周期管理）
│       │   └── accounting.py       # 交易记账
│       │
│       ├── backtest/               # 回测引擎
│       │   ├── engine.py           # 事件驱动回测引擎
│       │   ├── simulator.py        # 模拟撮合
│       │   └── analytics.py        # 绩效分析（Sharpe, MaxDD, Calmar...）
│       │
│       ├── optimize/               # 参数优化
│       │   ├── paramset.py         # 参数集定义（分布 + 约束）
│       │   ├── search.py           # 搜索策略（grid, random, bayesian）
│       │   ├── walk_forward.py     # 滚动前推分析
│       │   └── validation.py       # 统计检验（Deflated Sharpe, Profit Hurdle）
│       │
│       ├── trade/                  # 交易执行
│       │   ├── types.py            # Order, TradingCost, OrderPlan
│       │   ├── generator.py        # 订单生成（权重 → 订单）
│       │   ├── fees.py             # 费率计算
│       │   ├── slippage.py         # 滑点模型
│       │   ├── exchanges.py        # 交易所配置
│       │   └── executor.py         # 执行接口（抽象，可对接券商 API）
│       │
│       ├── universe/               # Universe 构建
│       │   ├── base.py             # UniverseProvider Protocol
│       │   ├── static.py           # 静态标的池（手动指定）
│       │   ├── index.py            # 指数成分股（Point-in-Time）
│       │   └── filter.py           # 条件过滤（因子筛选、流动性门槛）
│       │
│       ├── data/                   # 数据层
│       │   ├── providers.py        # Provider 协议定义
│       │   ├── market.py           # 行情数据 provider
│       │   ├── factors.py          # 因子数据 provider
│       │   └── loaders.py          # 数据加载工具（CSV, parquet, API）
│       │
│       ├── observe/                # 可观测性
│       │   ├── tracer.py           # 执行追踪（每步输入/输出记录）
│       │   ├── logger.py           # 结构化日志
│       │   ├── events.py           # 事件总线（策略事件、交易事件）
│       │   └── audit.py            # 审计追踪（可复现性保证）
│       │
│       └── tools/                  # 协议无关的 Tool 定义（核心资产）
│           ├── __init__.py         # Tool 注册中心
│           ├── strategy_tools.py   # 策略管理 tool（名称、参数、描述、调用逻辑）
│           ├── data_tools.py       # 数据查询 tool
│           ├── backtest_tools.py   # 回测 tool
│           ├── optimize_tools.py   # 优化 tool
│           ├── analysis_tools.py   # 分析 tool
│           ├── universe_tools.py   # Universe tool
│           ├── trade_tools.py      # 交易 tool
│           └── observe_tools.py    # 监控 tool
│
├── mcp_server/                     # MCP 协议适配层（可选分发渠道）
│   ├── __init__.py
│   ├── server.py                   # MCP server 入口（从 oxq.tools 导入）
│   └── adapters.py                 # MCP 协议适配（参数映射、会话管理）
│
├── skills/                         # Agent Skill 定义
│   ├── strategy-builder.md         # 构建交易策略
│   ├── backtest-runner.md          # 运行回测
│   ├── parameter-tuner.md          # 参数优化
│   ├── risk-analyzer.md            # 风险分析
│   ├── performance-reviewer.md     # 绩效评估
│   ├── trade-executor.md           # 交易执行
│   └── strategy-monitor.md         # 策略监控
│
├── examples/                       # 示例策略
│   ├── ma_crossover.py             # 均线交叉
│   ├── momentum_rotation.py        # 动量轮动
│   ├── mean_reversion.py           # 均值回归
│   └── multi_strategy.py           # 多策略编排
│
├── tests/                         # 测试（镜像 src/oxq/ 结构）
│   ├── universe/                  # Universe 测试
│   └── ...
├── docs/
├── pyproject.toml
├── LICENSE                         # MIT
└── README.md
```

---

## 3. 分层架构

```
┌──────────────────────────────────────────────────────┐
│              Skill Layer (skill.md)                   │  ← Agent 工作流指导
│  strategy-builder / backtest-runner / tuner ...       │
├──────────────────────────────────────────────────────┤
│              SDK + Tool Layer                         │  ← 核心资产
│  oxq.universe / oxq.core / oxq.backtest / ...        │  Python SDK
│  oxq.tools (协议无关的 Tool 定义)                     │  Tool 定义
│          ┆                                            │
│     ┌────┴──────────────────────┐                    │
│     │ MCP Server (可选分发层)    │  ← 非 Coding AI    │
│     │ mcp_server/               │    客户端适配       │
│     └───────────────────────────┘                    │
├──────────────────────────────────────────────────────┤
│              Engine Layer                             │  ← 纯计算，无 I/O
│  Universe resolve → Indicator → Signal → Rule        │
├──────────────────────────────────────────────────────┤
│              Provider Layer                           │  ← 数据注入（Protocol）
│  Universe / MarketData / Factor / Portfolio           │
└──────────────────────────────────────────────────────┘
```

---

## 4. 核心引擎设计

### 4.1 Indicator → Signal → Rule 三阶段模型

策略由三层声明式组件构成：Indicator（指标计算）、Signal（信号生成）、Rule（交易决策）。采用 compile once → run many 模式，策略编译后可重复执行。

**SDK 方式定义策略**：

```python
strategy = Strategy("momentum_rotation")

# 0. Universe：确定参与计算的标的池
strategy.set_universe(IndexUniverse("000300.SS"))  # 沪深300成分股（Point-in-Time）

# 1. 指标层：路径无关的纯计算
strategy.add_indicator("sma_fast", SMA, params={"period": 10})
strategy.add_indicator("sma_slow", SMA, params={"period": 50})
strategy.add_indicator("rsi", RSI, params={"period": 14})

# 2. 信号层：基于指标生成逻辑信号
strategy.add_signal("golden_cross", Crossover,
    inputs={"fast": "@ind:sma_fast", "slow": "@ind:sma_slow"},
    params={"direction": "up"})
strategy.add_signal("rsi_oversold", Threshold,
    inputs={"value": "@ind:rsi"},
    params={"threshold": 30, "relationship": "lt"})

# 3. 规则层：路径相关，生成订单
strategy.add_rule("enter_long", EntryRule,
    inputs={"signal": "@sig:golden_cross"},
    params={"order_type": "market", "size_method": "equal_weight"})
strategy.add_rule("exit_stop", ExitRule,
    inputs={"signal": "@sig:rsi_oversold"},
    params={"stop_loss_pct": 0.05, "trailing": True})

# 编译 & 运行
compiled = strategy.compile(registry)
result = compiled.run(context, providers)
```

**等价的 Tool 调用（AI Agent 方式）**：

Tool 定义在 `oxq.tools` 中，协议无关——Coding Agent 直接 `import` 调用，MCP 客户端通过 MCP 协议调用，平台方也可通过 REST/gRPC 等任意方式触发。

```
→ strategy.create(name="momentum_rotation")
→ universe.set(strategy="momentum_rotation", type="index", code="000300.SS")  # 沪深300成分股
→ strategy.add_indicator(strategy="momentum_rotation", name="sma_fast", type="SMA", params={"period": 10})
→ strategy.add_indicator(strategy="momentum_rotation", name="sma_slow", type="SMA", params={"period": 50})
→ strategy.add_signal(strategy="momentum_rotation", name="golden_cross", type="Crossover",
    inputs={"fast": "@ind:sma_fast", "slow": "@ind:sma_slow"})
→ strategy.add_rule(strategy="momentum_rotation", name="enter_long", type="EntryRule",
    inputs={"signal": "@sig:golden_cross"}, params={"order_type": "market"})
→ backtest.run(strategy="momentum_rotation", start="2020-01-01", end="2024-12-31")
```

### 4.2 单一宽表数据模型

Indicator、Signal、Rule 三层共享同一个 pandas DataFrame `mktdata`，各层通过**追加列**逐步加宽，而非各自维护独立的输出。

```
原始行情             Indicator 后              Signal 后               Rule 阶段
+-----------+       +------------------+      +---------------------+
| Open      |       | Open             |      | Open                |
| High      |       | High             |      | High                |
| Low       | ───►  | Low              | ───► | Low                 | ───► 只读,
| Close     |       | Close            |      | Close               |      生成订单
| Volume    |       | Volume           |      | Volume              |
|           |       | ma50    (新增)   |      | ma50                |
|           |       | ma200   (新增)   |      | ma200               |
|           |       |                  |      | ma50_gt_ma200 (新增)|
+-----------+       +------------------+      +---------------------+
```

各层对 `mktdata` 的操作方式：

| 层 | 操作 | 模式 |
|---|---|---|
| Indicator | 追加列（计算结果写回 mktdata） | 读 + 写 |
| Signal | 追加列（信号值写回 mktdata） | 读 + 写 |
| Rule | 按行读取指标/信号列值，生成订单 | **只读** |

**设计动机**：单一宽表避免了层间数据传递的复杂性。Signal 无需知道 Indicator 的输出格式，只需按列名引用；Rule 同理。这也天然适合 AI Agent——所有中间结果在同一张表上可见可查。

> **Universe 与宽表的关系**：宽表中包含哪些 symbol，由 Universe 在每个时间截面上动态决定。Universe 变化时（如指数成分股调整），宽表随之增减 symbol 维度，确保后续三阶段计算不会引入 survivorship bias。

> **与 Binding 系统的关系**：`@ind:sma_fast` 等引用语法是 SDK 层的声明式抽象，编译后解析为对 mktdata 列名的直接引用。宽表是底层运行时模型，Binding 是上层定义时模型。

### 4.3 执行模型

#### 逐层推进，全 symbol 执行

采用逐层推进模式，每个阶段接收**全 universe 的数据**，天然支持截面操作（如跨 symbol 的动量归一化、行业轮动排序）：

```
┌─────────────────────────────────────────────────────┐
│  Phase 0: Universe Resolution                        │
│  确定当前时间截面参与计算的 symbol 集合                │
│  输入: UniverseProvider + as_of_date                 │
│  输出: list[str] (symbols)                           │
│  示例: 沪深300成分股(PIT), 流动性过滤, 静态列表        │
├─────────────────────────────────────────────────────┤
│  Phase 1: Indicator                                  │
│  对全 universe 计算指标/因子                          │
│  输入: mktdata (per symbol)                          │
│  输出: mktdata + 指标列 (per symbol)                 │
│  示例: SMA, RSI, 动量因子, 波动率因子                 │
├─────────────────────────────────────────────────────┤
│  Phase 2: Signal                                     │
│  基于全 universe 的指标结果生成信号                    │
│  输入: 全部 symbol 的指标结果                         │
│  输出: 信号值 / 目标权重 (cross-sectional)            │
│  示例: 截面排序, top_n 筛选, 归一化, 风控调整          │
├─────────────────────────────────────────────────────┤
│  Phase 3: Rule                                       │
│  基于信号和当前持仓状态生成交易决策                     │
│  输入: 目标权重 + 当前 portfolio 状态                  │
│  输出: 订单 (orders)                                  │
│  示例: 再平衡判断, 仓位计算, 止损触发                  │
└─────────────────────────────────────────────────────┘
```

外层循环按 phase 推进（`for phase in [universe, indicator, signal, rule]`）。Phase 0 先从 UniverseProvider 获取当前时间截面的 symbol 列表，后续各层输入该 universe 的数据。回测中 universe 可能随再平衡日变化（如指数成分股季度调整），引擎在每个再平衡日重新执行 Phase 0。当 universe 只有一个 symbol 时，执行流程退化为逐 symbol 模式，完全兼容单标的策略。

#### 向量化与逐 bar 状态机

三个阶段采用不同的计算模式：

| 阶段 | 计算模式 | 路径依赖 | 调用次数 |
|---|---|---|---|
| Indicator | **向量化** — 对全量时间序列一次调用 | 否 | 每个 symbol 1 次 |
| Signal | **向量化** — 对全量时间序列一次调用 | 否 | 1 次（跨 symbol） |
| Rule | **逐 bar 循环** — 状态机模式 | 是 | N 次（N = 触发点数） |

**向量化阶段**（Indicator / Signal）：

```python
# Phase 0：从 UniverseProvider 获取当前 universe
universe = universe_provider.get_universe(as_of_date=context.as_of_date)
# universe = ["600519.SS", "000858.SZ", ...]  随时间变化

# Indicator：per symbol，一次调用处理全部历史
for symbol in universe:
    result = indicator.compute(mktdata[symbol], **params)
    mktdata[symbol]["sma_fast"] = result

# Signal：cross-sectional，一次处理全 universe
scores = {}
for symbol in universe:
    scores[symbol] = mktdata[symbol]["momentum"]
# 截面归一化
weights = normalize(top_n(scores, n=5))
```

**状态机阶段**（Rule）：

```python
# 逐 bar 推进，依赖当前持仓、挂单等状态
while cur_index <= total_bars:
    # 按优先级执行规则：risk → order → exit → enter
    # 读取当前 bar 的信号/权重
    # 生成订单，更新持仓状态
    cur_index = next_index(cur_index)
```

**dindex 优化**：Rule 循环不需要遍历每一个 bar。利用 Signal 阶段已生成的布尔列，可以预先建立"信号触发索引"（dindex），仅在信号实际触发的时间点执行 Rule，大幅减少迭代次数。

**设计动机**：Indicator/Signal 的值仅取决于截至 T 时刻的行情数据，与历史交易无关，故可向量化一次算完。Rule 的决策取决于当前持仓、挂单、资金等状态，必须逐步推进。这种分离既保证了前两层的计算效率，又保证了 Rule 层的正确性。

### 4.4 核心数据类型

```python
SparseVector = Dict[str, float]           # 权重/分数向量

@dataclass(frozen=True)
class Position:                           # 单个持仓
    symbol: str
    shares: Decimal
    price: float
    cost_basis: Decimal                   # 成本基础

@dataclass
class Portfolio:                          # 组合状态
    positions: Dict[str, Position]
    cash: Decimal
    # total_value / get_actual_weights 等方法

@dataclass(frozen=True)
class RunContext:                          # 执行上下文
    as_of_date: str
    effective_date: str
    data_version: str
    portfolio: Portfolio
    trace_id: str                         # 追踪 ID（可观测性）

@dataclass(frozen=True)
class Order:                              # 订单
    id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["market", "limit", "stop", "stop_limit", "trailing_stop"]
    shares: int
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    trail_pct: Optional[float]
    status: Literal["open", "filled", "partial", "canceled", "expired"]
    created_at: str
    filled_at: Optional[str]
    filled_price: Optional[Decimal]
    filled_shares: Optional[int]

@dataclass(frozen=True)
class UniverseSnapshot:                   # 某时间截面的标的池快照
    as_of_date: str                       # 截面日期
    symbols: tuple[str, ...]              # 当前成分（不可变）
    source: str                           # 来源标识（"static", "000300.SS", "filter:liquidity"）
    metadata: Dict[str, Any]              # 附加信息（权重、行业等）

@dataclass(frozen=True)
class ParamDistribution:                  # 参数分布（用于参数优化）
    component: str                        # "sma_fast" (indicator name)
    param: str                            # "period"
    values: list                          # [5, 10, 15, 20, 25, 30]
    distribution: Optional[str]           # "uniform", "log_uniform"
    low: Optional[float]
    high: Optional[float]

@dataclass(frozen=True)
class ParamConstraint:                    # 参数约束
    expr: str                             # "sma_fast.period < sma_slow.period"
```

### 4.5 Binding 系统

Binding 系统提供声明式的组件间引用语法，编译时解析为具体的数据路径：

```
@ind:<name>      → 引用指标输出
@sig:<name>      → 引用信号输出
@rule:<name>     → 引用规则输出
@step:<alias>    → 引用管道步骤输出
@prev:<kind>     → 引用历史信号
@provider:<name> → 引用外部 provider
@universe        → 当前 universe 快照（UniverseSnapshot）
@portfolio       → 当前组合状态
@orderbook       → 当前订单簿
```

### 4.6 四接口架构：策略与执行分离

策略层（Indicator → Signal → Rule）只负责"在什么条件下下什么单"，通过四个可替换 Protocol 与执行环境解耦：

```
策略定义层（Indicator → Signal → Rule）
         │
         │  不关心下面是回测还是实盘
         │
    ┌────▼──────────────────────────────────────────┐
    │           四个可替换 Protocol                   │
    │                                                │
    │  1. UniverseProvider   ── 哪些标的参与计算？   │
    │  2. MarketDataProvider ── 数据从哪来？         │
    │  3. OrderRouter        ── 订单往哪送？         │
    │  4. FillReceiver       ── 成交怎么回来？       │
    └────────────────────────────────────────────────┘
```

```python
from typing import Protocol

class UniverseProvider(Protocol):
    """标的池接口：确定每个时间截面参与计算的 symbol 集合"""
    def get_universe(self, as_of_date: str) -> UniverseSnapshot: ...
    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]: ...

class MarketDataProvider(Protocol):
    """行情接口：策略获取行情数据的唯一入口"""
    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...
    def get_latest(self, symbol: str) -> pd.Series: ...

class OrderRouter(Protocol):
    """订单接口：策略提交订单的唯一出口"""
    def submit_order(self, order: Order) -> str: ...       # 返回 order_id
    def cancel_order(self, order_id: str) -> bool: ...
    def get_open_orders(self, symbol: str = None) -> list[Order]: ...

class FillReceiver(Protocol):
    """成交接口：成交回报回填到 portfolio 的唯一入口"""
    def get_fills(self, order_id: str) -> list[Fill]: ...
    def on_fill(self, callback: Callable[[Fill], None]) -> None: ...
```

三种运行模式通过注入不同实现切换，策略代码零修改：

| 模式 | UniverseProvider | MarketDataProvider | OrderRouter | FillReceiver |
|---|---|---|---|---|
| **回测** | `IndexUniverse` — PIT 成分股 | `HistoricalData` — 加载历史 OHLCV | `SimOrderBook` — 模拟订单簿 | `SimFillEngine` — 理想撮合 |
| **Paper Trade** | `IndexUniverse` — 最新成分股 | `RealtimeData` — 实时行情 | `SimOrderBook` — 模拟订单簿 | `SimFillEngine` — 模拟撮合（含延迟、部分成交） |
| **实盘** | `IndexUniverse` — 最新成分股 | `RealtimeData` — 实时行情 | `BrokerRouter` — 券商 API | `BrokerFill` — 实际成交回报 |

```python
# 回测模式
engine.run(strategy, providers={
    "universe": IndexUniverse("000300.SS", point_in_time=True),
    "market":   HistoricalData(start="2020-01-01", end="2024-12-31"),
    "order":    SimOrderBook(),
    "fill":     SimFillEngine(slippage=FixedSlippage(bps=5)),
})

# Paper trade：仅替换行情源，撮合引擎增加模拟延迟
engine.run(strategy, providers={
    "universe": IndexUniverse("000300.SS"),
    "market":   RealtimeData(source="websocket"),
    "order":    SimOrderBook(),
    "fill":     SimFillEngine(slippage=FixedSlippage(bps=5), latency_ms=100, partial_fill=True),
})

# 实盘：四个接口全部替换
engine.run(strategy, providers={
    "universe": IndexUniverse("000300.SS"),
    "market":   RealtimeData(source="websocket"),
    "order":    BrokerRouter(broker="interactive_brokers"),
    "fill":     BrokerFill(broker="interactive_brokers"),
})
```

> **当前阶段**：Phase 1 只实现 `StaticUniverse` + `HistoricalData` + `SimOrderBook` + `SimFillEngine`（回测）。`IndexUniverse`（Point-in-Time）和 `FilterUniverse` 留给 Phase 2。Paper trade 和实盘的实现留给后续 Phase，但四个 Protocol 从第一天就定义好，确保架构不需要重构。

---

## 5. 功能模块

### 5.1 指标库 (oxq.indicators)

**契约优先，而非约定优先**：

open-xquant 采用契约优先的设计——统一的 Protocol 接口取代松散的函数约定。原因是 Agent First 设计哲学——约定意味着"文档里写了但代码不强制"，正好是 AI 幻觉的温床。统一的 Protocol 让 AI 的选择空间收窄到只有正确的做法。

```python
from typing import Protocol

class Indicator(Protocol):
    """所有指标必须实现的契约"""
    name: str

    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series:
        """
        输入：mktdata 宽表（OHLCV + 已有指标列）
        输出：与 mktdata 等长的 Series，将追加为新列
        约束：纯函数，不得依赖外部状态，不得修改 mktdata
        """
        ...

# 内置指标实现示例
class SMA:
    name = "SMA"

    def compute(self, mktdata: pd.DataFrame, column: str = "Close", period: int = 20) -> pd.Series:
        return mktdata[column].rolling(period).mean()

# 接入第三方库（如 ta-lib）需要写 adapter
class TalibSMA:
    """ta-lib SMA 的适配器"""
    name = "SMA_talib"

    def compute(self, mktdata: pd.DataFrame, column: str = "Close", timeperiod: int = 20) -> pd.Series:
        import talib
        return pd.Series(talib.SMA(mktdata[column].values, timeperiod=timeperiod), index=mktdata.index)
```

> **设计取舍**：接入第三方库需要写 adapter 是有意为之的代价。adapter 本身是一次性工作，且内置指标库会覆盖最常用的指标。对 AI Agent 而言，统一接口带来的确定性远比"拿来即用"更重要。

**内置指标**（首批）：

| 类别 | 指标 | 说明 |
|------|------|------|
| 趋势 | SMA, EMA, WMA, DEMA | 移动平均系列 |
| 动量 | RSI, MACD, ROC, MOM | 动量指标 |
| 波动 | BBands, ATR, StdDev | 波动率指标 |
| 成交量 | OBV, VWAP | 量价指标 |
| 自定义 | Formula | 用户自定义公式 |

**Agent 体验**：Agent 不需要知道具体实现，只需说"给这个策略加一个 20 日 RSI 指标"，Tool 处理其余一切。

**Indicator 返回值的两条路径**：

| 路径 | 条件 | 去向 |
|---|---|---|
| A（主路径） | 返回与 mktdata 等长的时间序列 | 追加为 mktdata 的新列 |
| B（例外） | 返回不等长对象或非时间序列 | 存入独立的附加存储（`extras`） |

路径 A 是绝大多数场景。路径 B 极少使用（例如某些统计检验返回标量或不等长序列），此时 Signal/Rule 需从 `extras` 中显式读取，而非从 mktdata 列引用。框架优先引导用户走路径 A，路径 B 作为逃生舱口保留。

**预计算指标（因子）的接入**：

在实际量化研究中，指标/因子往往是预先计算好的（如离线因子库）。框架支持两种接入方式：

**方案一：直接注入宽表，不注册为 Indicator**

Signal/Rule 仅按列名引用数据，不关心列的来源。只要不注册同名 Indicator，预计算列不会被覆盖。

```python
# 1. 预计算结果直接附加到行情数据
mktdata["alpha_momentum"] = load_factor("momentum", symbols, dates)

# 2. 策略定义跳过 add_indicator，直接从 Signal 开始
strategy.add_signal("momentum_high", Threshold,
    inputs={"value": "alpha_momentum"},  # 直接引用列名
    params={"threshold": 0.8, "relationship": "gt"})
```

优点：最简单，零重算。缺点：策略定义不完整，无法对该指标参数做参数优化。

**方案二：注册为缓存读取 Indicator**

注册一个符合 Indicator Protocol 的类，其 `compute` 方法仅从缓存/因子库读取预计算结果：

```python
class CachedFactor:
    """从因子库加载预计算因子，符合 Indicator Protocol"""
    name = "CachedFactor"

    def compute(self, mktdata: pd.DataFrame, factor_name: str = "", **kwargs) -> pd.Series:
        return load_factor(factor_name, mktdata.index)

strategy.add_indicator("alpha_momentum", CachedFactor,
    params={"factor_name": "momentum"})
```

优点：策略定义完整，兼容参数优化（可优化 factor_name 的选择或后处理参数）。缺点：需自行管理缓存一致性。

> **推荐**：研究阶段用方案一快速迭代；生产阶段用方案二保证策略定义的完整性和可审计性。

### 5.2 信号生成器 (oxq.signals)

提供 7 种内置信号类型：

| 信号类型 | 说明 |
|----------|------|
| `Crossover` | 两条线交叉 |
| `Threshold` | 超过/低于阈值 |
| `Comparison` | 两个值比较 |
| `Formula` | 自定义布尔公式 |
| `Peak` | 峰值/谷值检测 |
| `Timestamp` | 时间条件触发 |
| `Composite` | 多信号 AND/OR 组合 |

### 5.3 交易规则 (oxq.rules)

每种规则有明确的执行优先级：

```
执行顺序（每个时间步）：
1. Risk Rules     → 熔断检查、全局风控     （最高优先级）
2. Order Rules    → 处理挂单（止损/止盈触发）
3. Rebalance      → 再平衡检查
4. Exit Rules     → 平仓信号
5. Entry Rules    → 建仓信号              （最低优先级）
```

**仓位管理函数**：
- `osMaxPos` - 最大仓位限制
- `osEqualWeight` - 等权分配
- `osRiskParity` - 风险平价
- `osPctEquity` - 固定比例
- `osKelly` - Kelly 公式

### 5.4 组合管理 (oxq.portfolio)

- **订单簿（OrderBook）**：支持 market/limit/stop/stop_limit/trailing_stop 订单类型，追踪订单全生命周期（open → filled/partial/canceled/expired）
- **Portfolio**：管理持仓状态、现金余额、组合总值、实际权重
- **交易记账（Accounting）**：记录每笔交易的成交价、手续费、滑点，维护成本基础

### 5.5 回测引擎 (oxq.backtest)

- 事件驱动回测引擎，基于三阶段模型执行
- 模拟撮合引擎，支持可配置的滑点模型和费率
- 绩效分析：Sharpe Ratio, Max Drawdown, Calmar Ratio, Win Rate, Profit Factor 等

### 5.6 参数优化 (oxq.optimize)

```python
# SDK 用法
paramset = ParamSet("sma_tuning")
paramset.add_distribution("sma_fast", "period", values=range(5, 30, 5))
paramset.add_distribution("sma_slow", "period", values=range(20, 100, 10))
paramset.add_constraint("sma_fast.period < sma_slow.period")

# 网格搜索
results = grid_search(strategy, paramset, data, metric="sharpe_ratio")

# 滚动前推分析
wfa_results = walk_forward(
    strategy, paramset, data,
    train_period="2Y",          # 训练窗口
    test_period="6M",           # 测试窗口
    step="3M",                  # 滚动步长
    optimize_metric="sharpe_ratio",
    anchored=False              # 滚动窗口 vs 扩展窗口
)

# 统计检验（防过拟合）
deflated_sr = deflated_sharpe(results, num_trials=len(paramset))
profit_hurdle = profit_hurdle_test(results, num_trials=len(paramset))
haircut_sr = haircut_sharpe(results, method="holm")
```

**Tool 调用**：
```
→ optimize.define_paramset(strategy="momentum", distributions=[...], constraints=[...])
→ optimize.run_walk_forward(strategy="momentum", paramset="sma_tuning", train="2Y", test="6M")
→ analysis.deflated_sharpe(results_id="wfa_001")
→ analysis.profit_hurdle(results_id="wfa_001")
```

### 5.7 统计检验 (oxq.optimize.validation)

| 方法 | 用途 |
|------|------|
| `deflated_sharpe()` | 校正多重比较后的 Sharpe Ratio |
| `haircut_sharpe()` | 对 Sharpe Ratio 施加 haircut |
| `profit_hurdle()` | 最低利润门槛检验 |
| `degrees_of_freedom()` | 自由度估计 |

### 5.8 多策略编排 (oxq.orchestrator)

```python
orchestrator = StrategyOrchestrator()
orchestrator.add("momentum", strategy_a, weight=0.4)
orchestrator.add("mean_reversion", strategy_b, weight=0.3)
orchestrator.add("trend_following", strategy_c, weight=0.3)

# 策略间约束
orchestrator.set_total_exposure(max=1.5, min=0.3)   # 总敞口限制
orchestrator.set_correlation_limit(max=0.7)          # 策略相关性限制
orchestrator.set_capital_allocation("risk_parity")   # 资金分配方式

result = orchestrator.run(context, providers)
```

### 5.9 可观测性 (oxq.observe)

```python
# 执行追踪：每步自动记录输入/输出/耗时
@dataclass
class TraceSpan:
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    component: str          # "indicator:sma_fast", "signal:golden_cross", "rule:enter"
    inputs: Dict[str, Any]  # 快照
    output: Any             # 快照
    params: Dict[str, Any]
    started_at: datetime
    ended_at: datetime
    status: Literal["ok", "error", "skipped"]

# 审计日志：保证可复现
@dataclass
class AuditRecord:
    run_id: str
    strategy_code: str
    strategy_version: str
    as_of_date: str
    data_version: str
    param_snapshot: Dict     # 完整参数快照
    result_hash: str         # 结果哈希（验证确定性）
```

**Tool 调用**：
```
→ observe.get_trace(run_id="run_20240101")        # 查看执行追踪
→ observe.replay(run_id="run_20240101")            # 重放某次执行
→ observe.compare_runs(run_ids=["run_a", "run_b"]) # 对比两次执行
```

### 5.10 交易执行 (oxq.trade)

- 完整订单簿：支持 limit/stop/trailing_stop，追踪订单生命周期
- 执行反馈闭环：实际成交 → 更新 portfolio → 记录 slippage
- 多交易所支持：SSE, SZSE, NYSE, NASDAQ, HKEX
- 券商 API 抽象：定义 Executor Protocol，社区可实现各券商适配器

策略定义与执行环境的分离详见 [4.6 四接口架构](#46-四接口架构策略与执行分离)。

### 5.11 数据层 (oxq.data)

**统一因子模型**：框架将所有外部数据统一视为 Factor（因子），无论其原始来源是行情、财务报表、宏观经济指标还是新闻舆情。从策略视角，PE ratio、GDP 增速、舆情分数与 RSI 本质相同——都是"某个时间点上的一个数值"。所有 factor 最终以列的形式汇入单一宽表 `mktdata`，参与 Indicator → Signal → Rule 管道。

**核心原则**：

1. **一切皆 Factor**：财务数据（PE、ROE、营收）、宏观数据（GDP、CPI、利率）、另类数据（新闻舆情、社交媒体情绪）在进入宽表前均完成预处理，以数值型 factor 的形式存在
2. **Point-in-Time 对齐**：factor 必须携带时间属性，按数据的实际可用日（announce_date）而非报告期（period_date）对齐进宽表，防止前视偏差
3. **频率打平**：低频 factor（季度财报、月度宏观）通过 forward-fill 对齐到日频宽表；事件型数据（新闻）需先聚合到日频再注入
4. **全局数据广播**：无 symbol 维度的数据（宏观指标）广播到全 universe 的每个 symbol，在宽表中作为同名列存在，Signal 层可自然地将其用作全局条件

### 5.12 Universe 构建 (oxq.universe)

Universe 决定"每个时间截面上，哪些 symbol 参与计算"。缺少显式 Universe 管理会导致 survivorship bias——用当前成分股回测历史数据，高估策略收益。

**UniverseProvider Protocol**：

```python
class UniverseProvider(Protocol):
    """标的池接口：确定参与计算的 symbol 集合"""
    def get_universe(self, as_of_date: str) -> UniverseSnapshot: ...
    def get_history(self, start: str, end: str) -> list[UniverseSnapshot]: ...
```

**三种内置实现**：

| 实现 | 说明 | 适用场景 |
|------|------|----------|
| `StaticUniverse` | 固定 symbol 列表，不随时间变化 | 单标的策略、手动指定标的池 |
| `IndexUniverse` | 指数成分股，支持 Point-in-Time 历史成分查询 | 沪深300轮动、行业指数策略 |
| `FilterUniverse` | 基于因子条件动态过滤（流动性、市值、ST 排除等） | 全市场因子策略 |

```python
# 静态列表
universe = StaticUniverse(["600519.SS", "000858.SZ", "000333.SZ"])

# 指数成分股（Point-in-Time，回测安全）
universe = IndexUniverse("000300.SS", point_in_time=True)

# 条件过滤：排除 ST、日均成交额 > 1000 万
universe = FilterUniverse(
    base=IndexUniverse("000985.SS"),  # 中证全指作为基础池
    filters=[
        ExcludeST(),
        MinTurnover(threshold=10_000_000),
    ]
)
```

**Tool 调用**：
```
→ universe.set(strategy="momentum", type="index", code="000300.SS")
→ universe.inspect(strategy="momentum", as_of_date="2023-06-30")  # 查看某日成分
→ universe.history(strategy="momentum", start="2020-01-01", end="2024-12-31")  # 成分变动历史
```

> **Point-in-Time 的重要性**：回测中使用当天实际生效的成分股（而非最新成分股）是消除 survivorship bias 的关键。`IndexUniverse` 的 `point_in_time=True` 确保在 2020 年的回测日使用 2020 年的成分股数据，而非 2024 年的。

---

## 6. Tool 定义与分发

### 6.1 Tool 定义（oxq.tools）

Tool 定义是框架的核心资产之一，与传输协议无关。每个 Tool 包含：名称、参数 schema、语义描述、调用逻辑。Tool 定义在 `src/oxq/tools/` 中实现，可通过多种方式触发：Coding Agent 直接 `import` 调用、MCP 客户端通过 MCP 协议调用、平台方通过 REST/gRPC 等自建接口调用。

**工具清单**：

| 工具组 | 工具名 | 说明 |
|--------|--------|------|
| **universe** | `universe.set` | 设置策略的 Universe（静态列表/指数/过滤条件） |
| | `universe.list_indexes` | 列出可用的指数代码 |
| | `universe.inspect` | 查看某日的 Universe 成分快照 |
| | `universe.history` | 查看 Universe 成分变动历史 |
| **strategy** | `strategy.create` | 创建策略 |
| | `strategy.add_indicator` | 添加指标 |
| | `strategy.add_signal` | 添加信号 |
| | `strategy.add_rule` | 添加规则 |
| | `strategy.list` | 列出所有策略 |
| | `strategy.inspect` | 查看策略详情 |
| | `strategy.validate` | 验证策略配置 |
| | `strategy.export` | 导出策略为 YAML/JSON |
| | `strategy.import` | 导入策略配置 |
| **data** | `data.load_symbols` | 加载标的行情数据 |
| | `data.list_symbols` | 列出可用标的 |
| | `data.inspect` | 查看数据摘要（时间范围、缺失值等） |
| | `data.query` | 查询特定数据（价格、因子等） |
| **backtest** | `backtest.run` | 运行回测 |
| | `backtest.results` | 获取回测结果 |
| | `backtest.compare` | 对比多个回测 |
| | `backtest.trade_list` | 查看交易记录 |
| **optimize** | `optimize.define_paramset` | 定义参数搜索空间 |
| | `optimize.run_search` | 运行参数搜索 |
| | `optimize.run_walk_forward` | 运行前推分析 |
| | `optimize.results` | 获取优化结果 |
| **analysis** | `analysis.performance` | 绩效指标（Sharpe, MaxDD, Calmar...） |
| | `analysis.deflated_sharpe` | Deflated Sharpe Ratio |
| | `analysis.profit_hurdle` | 利润门槛检验 |
| | `analysis.drawdown` | 回撤分析 |
| | `analysis.trade_stats` | 交易统计（胜率、盈亏比...） |
| **trade** | `trade.generate_orders` | 生成订单计划 |
| | `trade.estimate_costs` | 估算交易成本 |
| | `trade.execute` | 执行订单（需确认） |
| | `trade.status` | 查询订单状态 |
| **observe** | `observe.trace` | 查看执行追踪 |
| | `observe.audit_log` | 查看审计日志 |
| | `observe.replay` | 重放历史执行 |
| **orchestrator** | `orchestrator.create` | 创建多策略编排 |
| | `orchestrator.add_strategy` | 添加子策略 |
| | `orchestrator.set_constraints` | 设置全局约束 |
| | `orchestrator.run` | 运行编排 |

**设计原则**：

1. **原子性**：每个 tool 做一件事，返回结构化 JSON
2. **幂等性**：相同输入产生相同输出（除 execute 外）
3. **可组合**：tools 之间通过 ID 引用关联（strategy_id, backtest_id, paramset_id）
4. **错误友好**：返回清晰的错误信息 + 建议的修复动作
5. **渐进披露**：简单场景用少量参数，复杂场景可展开全部参数
6. **Thin Wrapper**：Tool 是 SDK 的薄封装，不得包含业务逻辑。每个 tool 函数体只做三件事：参数解析 → 调用 `oxq` SDK → 格式化返回。所有计算、状态管理、规则执行等逻辑必须实现在 `src/oxq/` 中，Tool 层只负责接口适配

### 6.2 MCP Server（可选分发层）

MCP Server 是 `oxq.tools` 的 MCP 协议适配，用于支持不能执行代码的 AI 客户端（Claude Desktop、Windsurf 等）。MCP Server 本身不包含业务逻辑，只做三件事：MCP 协议适配、会话状态管理、从 `oxq.tools` 导入 Tool 定义。

> **注意**：Coding Agent（如 Claude Code、Cursor）直接 `import oxq` 即可，不需要 MCP Server。MCP 的价值是**分发渠道**——让本地 SDK 能被非 Coding AI 客户端开箱即用。

```python
# mcp_server/server.py — 薄适配层，从 oxq.tools 导入
from mcp.server import Server
from oxq.tools import registry as tool_registry

server = Server("open-xquant")

# 会话状态管理（MCP server 维护跨 tool 调用的生命周期）
class SessionState:
    strategies: Dict[str, Strategy]
    universes: Dict[str, UniverseProvider]
    datasets: Dict[str, DataFrame]
    backtests: Dict[str, BacktestResult]
    paramsets: Dict[str, ParamSet]

# 自动从 oxq.tools 注册所有 tool 到 MCP server
for tool_def in tool_registry.all_tools():
    server.tool()(tool_def.as_mcp_handler(session_state))
```

---

## 7. Agent Skills

每个 skill.md 描述一个完整的 Agent 工作流，指导 AI Agent 如何组合 tools 完成任务。

> **Tool 引用是协议无关的**：skill 中引用的 tool 名称（如 `universe.*`、`strategy.*`）是 `oxq.tools` 中定义的协议无关 Tool。Coding Agent 通过 `import oxq.tools` 调用等价函数，MCP 客户端通过 MCP 协议调用——Tool 名称和语义完全一致，仅传输方式不同。

### 7.1 strategy-builder.md

```markdown
---
name: strategy-builder
description: 指导 Agent 构建量化交易策略
tools_required: [universe.*, strategy.*, data.*]
---

## 工作流

1. 理解用户意图：什么市场？什么风格（趋势/动量/均值回归）？
2. 设定 Universe：
   a. universe.list_indexes 查看可用指数
   b. universe.set 设定标的池（指数成分/静态列表/条件过滤）
   c. universe.inspect 确认成分合理
3. 加载数据：调用 data.load_symbols 加载行情
4. 探索数据：调用 data.inspect 了解数据特征
5. 构建策略：
   a. strategy.create 创建策略
   b. strategy.add_indicator 添加指标（根据风格选择）
   c. strategy.add_signal 添加信号
   d. strategy.add_rule 添加规则（入场 + 出场 + 风控）
6. 验证：strategy.validate 检查配置完整性
7. 快速回测：backtest.run 跑一个初始回测看结果是否合理

## 决策指南
- 趋势策略 → SMA/EMA + Crossover signal
- 动量策略 → RSI/MACD + Threshold signal
- 均值回归 → BBands + Threshold signal
- 始终添加止损规则（建议 ATR-based）
```

### 7.2 parameter-tuner.md

```markdown
---
name: parameter-tuner
description: 指导 Agent 进行参数优化并验证结果的统计显著性
tools_required: [optimize.*, analysis.*]
---

## 工作流

1. 定义参数空间：optimize.define_paramset
2. 运行前推分析：optimize.run_walk_forward（优先于 grid_search）
3. 统计检验：
   a. analysis.deflated_sharpe → SR 是否统计显著？
   b. analysis.profit_hurdle → 收益是否超过随机？
4. 如果不显著 → 建议简化策略（减少参数）或增加数据
5. 如果显著 → 输出最优参数 + 样本外表现

## 红线规则
- 永远不要只看 in-sample 结果
- 参数组合 > 100 时必须做 Deflated Sharpe 校正
- Walk-forward 窗口至少覆盖 2 个完整市场周期
```

### 7.3 其他 Skills

| Skill | 核心工作流 |
|-------|-----------|
| `backtest-runner.md` | 配置回测 → 运行 → 分析绩效 → 检查过拟合 → 给出改进建议 |
| `risk-analyzer.md` | 回撤分析 → 压力测试 → 尾部风险 → 建议风控规则 |
| `performance-reviewer.md` | 多维绩效分析 → 归因分析 → 与基准对比 → 生成报告 |
| `trade-executor.md` | 生成订单 → 估算成本 → 确认后执行 → 监控成交 → 记录结果 |
| `strategy-monitor.md` | 实盘监控 → 偏离检测 → 风控预警 → 建议调仓 |

---

## 8. 技术选型

| 决策 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.12+ | AI 生态最丰富 |
| 类型系统 | dataclass(frozen=True) + Protocol | 不可变 + 鸭子类型 |
| 金融精度 | Decimal | 避免浮点误差 |
| 时间序列 | pandas DataFrame/Series | Indicator Protocol 标准输入输出类型 |
| 核心依赖 | pandas, numpy | 向量化计算基础设施 |
| 可选依赖 | scipy (optimize), ta-lib (指标加速) | 仅在特定模块引入 |
| 可选依赖 | mcp (Python) | MCP Server 分发时需要，官方 Python SDK |
| 配置格式 | YAML + JSON Schema | Agent 友好（结构化），人也可读 |
| 构建工具 | hatch / uv | 现代 Python 项目管理 |
| 测试 | pytest | 标准选择 |
| 文档 | mkdocs-material | 适合开源项目 |

---

## 9. 实现路线

### Phase 1: 核心引擎 + SDK (MVP)
- `oxq.core`: Strategy, Engine, Registry, 基础类型
- `oxq.universe`: UniverseProvider Protocol, StaticUniverse
- `oxq.indicators`: 5 个内置指标 (SMA, EMA, RSI, MACD, BBands)
- `oxq.signals`: 3 个信号 (Crossover, Threshold, Comparison)
- `oxq.rules`: EntryRule, ExitRule, 基础 sizing
- `oxq.portfolio`: Portfolio, Position
- `oxq.backtest`: 基础回测引擎 + analytics
- `oxq.tools`: universe.* + strategy.* + backtest.* tool 定义（核心交付物）
- `skills/`: strategy-builder.md, backtest-runner.md
- **目标**: Coding Agent / 开发者可以通过 SDK 构建简单策略并回测

### Phase 2: 参数优化 + 统计检验 + Universe 扩展 + MCP 分发
- `oxq.universe`: IndexUniverse（Point-in-Time）, FilterUniverse
- `oxq.optimize`: ParamSet, GridSearch, WalkForward
- `oxq.optimize.validation`: DeflatedSharpe, ProfitHurdle
- `oxq.tools`: optimize.* + analysis.* tool 定义
- `mcp_server`: MCP 协议适配层（从 oxq.tools 导入，支持非 Coding AI 客户端）
- `skills/`: parameter-tuner.md, performance-reviewer.md
- **目标**: Agent 可以优化参数并验证统计显著性，Universe 支持指数成分和动态过滤，MCP 客户端可通过 MCP Server 使用全部功能

### Phase 3: 交易执行 + 可观测性
- `oxq.trade`: 完整订单簿 + 费率 + 滑点 + executor protocol
- `oxq.observe`: Tracer, AuditLog, EventBus
- `oxq.tools`: trade.* + observe.* tool 定义
- `skills/`: trade-executor.md, strategy-monitor.md
- **目标**: 端到端全链路，从构建到执行到监控

### Phase 4: 多策略 + 高级特性
- `oxq.orchestrator`: 多策略编排 + 资金分配
- 高级规则: 追踪止损、风险平价、因子风控
- 更多指标/信号
- `oxq.tools`: orchestrator.* tool 定义
- `skills/`: risk-analyzer.md
- **目标**: 机构级多策略管理能力

---

## 10. 验证方案

### 冒烟测试（SDK 主路径）
```bash
# 安装
pip install -e ".[dev]"

# 单元测试
pytest tests/ -v

# SDK 集成测试：直接 import oxq 构建策略 → 回测 → 查看结果
python examples/ma_crossover.py
```

### MCP Server 补充测试（可选分发路径）
```bash
# 需要安装 MCP 依赖
pip install -e ".[mcp]"

# MCP server 启动
python -m mcp_server.server

# 验证 MCP 协议适配正确
```

### Agent 端到端测试
```
主路径（Coding Agent）：
1. Coding Agent 直接 import oxq
2. 用户: "帮我构建一个 A 股 ETF 动量轮动策略，回测 2020-2024 年"
3. Agent 调用 strategy-builder skill，通过 SDK 依次调用 oxq.tools
4. 验证：策略创建成功 → 回测完成 → 绩效报告生成

补充路径（MCP 客户端）：
1. 启动 MCP server
2. Claude Desktop 等非 Coding 客户端连接 MCP server
3. 同样的用户请求，通过 MCP 协议调用相同的 Tool
4. 验证：结果与 SDK 直接调用一致
```

### 可复现性测试
```
1. 运行策略，记录 trace_id + result_hash
2. 使用 observe.replay(trace_id) 重放
3. 验证 result_hash 完全一致
```

---

## 参考

- **quantstrat (R)**: indicator → signal → rule 分层模型、paramset 参数优化、walk-forward analysis、Deflated Sharpe Ratio 统计检验、order book 管理
- **xquant.shop**: agent pipeline 架构、compile once → run many 模式、immutable specs、provider injection、SparseVector 权重表示
