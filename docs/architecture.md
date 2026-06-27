# open-xquant 架构文档

## 1. 设计哲学

open-xquant 是一个 **Agentic Quant Research Kernel**——面向 AI Coding Agent 和人类量化研究者的确定性量化研究内核。

**一句话定位**：

```text
OpenCode = 通用执行器，负责读写文件、调用命令、运行代码
open-xquant = 量化研究内核，负责约束、计算、审计和产物标准
```

核心工作流：**spec → validate → compile → backtest → audit → robustness → report**

底层是严谨的量化金融引擎，经 Universe → Indicator → Signal → Portfolio → Rule → Broker
管道生成交易决策；核心资产是 **Python SDK + 协议无关的 Tool 定义 + 声明式 Strategy Spec**。
在 SDK 层，`Strategy` 表达可复用的策略逻辑，不包含 Universe；Universe 是运行时输入，
同一策略可以在不同 Universe 上运行并得到不同组合。

**两种使用角色与入口**：

- **Coding Agent / 开发者** → `import oxq` 或 `oxq` CLI（主要方式）
- **平台方** → 基于 SDK + Tool 定义自建接口（REST API、gRPC 等）

**五大设计原则**：

- **声明式**：策略通过 `strategy_spec.yaml` 声明，spec → compiler → 可执行策略。策略是"做什么"的声明，引擎负责"怎么做"
- **确定性**：相同 spec + 相同数据 = 相同回测结果。不可变数据类型 + 纯函数计算 + hash 审计追踪
- **约束即安全**：spec validation + research bias audit + robustness tests 三道防线，自动检测常见回测陷阱
- **可审计**：每次研究留下结构化 artifacts——metrics、trades、equity curve、audit、report——可版本化、可 diff、可复现
- **全流程**：从 spec 创建、回测、审计、稳健性测试到报告生成，端到端覆盖

---

## 2. 总体架构（升级后）

```
┌────────────────────────────────────────────────────────────┐
│ Agent Runtime                                                │
│ OpenCode / Claude Code / Codex / Local CLI                   │
│ 负责读写文件、调用命令、运行代码                              │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│ open-xquant: Agentic Quant Research Kernel                  │
│ spec / validate / compile / backtest / audit / report / log  │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│ Data Providers                                               │
│ Local CSV / YFinance / AkShare / PIT Data Gateway / Custom   │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 项目结构（升级后）

```
open-xquant/
├── src/oxq/                        # 主 Python 包（pip install open-xquant）
│   ├── core/                       # 核心引擎（Strategy, Engine, types, registry）
│   ├── spec/                       # Strategy Spec (schema, parser, validator, compiler)
│   ├── data/                       # 数据层（Provider 协议、行情/因子数据）
│   ├── universe/                   # Universe 构建（静态、过滤、指数成分）
│   ├── indicators/                 # 技术指标库（30+ 内置指标）
│   ├── signals/                    # 信号生成器（8 种信号类型）
│   ├── portfolio/                  # 组合管理（优化器、持仓、订单簿、绩效分析）
│   ├── trade/                      # 交易执行（SimBroker、费率、滑点、OrderGenerator）
│   ├── rules/                      # 交易规则（止损、止盈、风控熔断）
│   ├── audit/                      # 审计系统（reproducibility + research bias）
│   ├── robustness/                 # 稳健性测试（成本扰动、参数扰动、IS/OOS 对比）
│   ├── report/                     # 研究报告生成器（Markdown、HTML、report_assets）
│   ├── observe/                    # 可观测性（追踪、审计记录、实验日志）
│   ├── optimize/                   # 参数优化（网格搜索、滚动前推、交叉验证）
│   ├── factor_eval/                # 因子评估（IC、ICIR、衰减、Tearsheet）
│   ├── tools/                      # 协议无关的 Tool 定义
│   ├── cli/                        # oxq 命令行接口
│   └── contrib/                    # 第三方券商/数据源集成
│       └── alpaca/                 # Alpaca 集成
│
├── agent/                          # Agent 层
│   ├── skills/                     # Agent Skill 定义（markdown 工作流）
│   ├── roles/                      # Multi-agent role 模板
│   └── opencode/                   # OpenCode 本地 skills.path 配置
│
├── examples/                       # 示例
│   ├── modules/                     # 模块 SDK 使用示例（可执行 Python 脚本）
│   ├── strategies/                 # 完整 E2E 策略示例 + spec 校验演示
│
├── tests/                          # 测试（镜像 src/oxq/ 结构）
├── docs/                           # 文档
│   ├── architecture.md             # 本文档
│   ├── design/                     # 设计文档和 RFC
│   └── schemas/                    # spec schema 文档
├── pyproject.toml
├── LICENSE                         # MIT
└── README.md
```

---

## 4. 核心引擎设计

### 4.1 一切皆组合

本框架的核心建模假设：**量化策略输出的一切皆组合**。即使策略只交易一个标的物，它产出的也是该标的物与现金（CASH）的组合——全仓买入是 `{AAPL: 1.0, CASH: 0.0}`，空仓是 `{CASH: 1.0}`，半仓是 `{AAPL: 0.5, CASH: 0.5}`。

这意味着策略管道的终点始终是一组目标权重，而非单个买卖指令。交易算法负责将当前持仓调整到目标组合，Broker 负责执行。

### 4.2 Strategy = Universe + Signal + Portfolio

Strategy 由三个核心组件构成：

- **Universe** — 确定标的池。可以是固定列表（StaticUniverse）、指数成分（IndexUniverse）或基于条件的动态过滤（FilterUniverse）
- **Signal** — 逐 symbol 产出交易意图。输出布尔或分类标签
  （`BUY` / `SELL` / `HOLD`），描述"交易的欲望"而非订单
- **Portfolio** — 跨 symbol 组合优化。接收 Signal 输出，通过 PortfolioOptimizer 计算目标权重

Strategy 是纯声明式容器——直接传给 Engine 执行。它始于假设和目标：hypothesis 定义了策略试图捕捉的市场现象，objectives 量化了成功标准，benchmarks 提供了比较的参照系。

**Indicator** 服务于上述三个组件以及 Rule，各模块通过 `required_indicators` 属性声明自己依赖的指标，Engine 负责统一收集和计算。

**Rule** 不属于 Strategy。Rule 的职责是对持仓组合的准入约束和持仓监控，通过 `Engine.run(rules=[...])` 传入。

Engine 驱动完整管道：**Indicator → Universe → Signal → Portfolio → Pre-trade Rule → Trading Algorithm → Broker → Post-trade Rule**。
分类信号不会直接下单；例如 `ROCTiming` 输出 `BUY`、`SELL`、`HOLD` 后，
由 `SignalToPosition` 将其转换为 `{symbol: weight}` 或 `{CASH: 1.0}` 目标组合，
再由交易算法生成订单。
`EqualWeight` 只消费布尔过滤信号；`BUY`、`SELL`、`HOLD` 这类分类交易意图
必须经由 `SignalToPosition` 或同等语义的 PortfolioOptimizer。自定义分类
Signal 在 spec 中通过 `signal.rules.<name>.output_domain` 声明输出域，
不要把该元数据放进 `params`。

```python
from oxq.core import Engine, Strategy
from oxq.indicators import SMA
from oxq.signals import Crossover
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.rules import ExitRule, StopLossRule
from oxq.universe import StaticUniverse

crossover = Crossover()
crossover.required_indicators = {
    "sma_fast": (SMA(), {"column": "close", "period": 10}),
    "sma_slow": (SMA(), {"column": "close", "period": 50}),
}

strategy = Strategy(
    name="sma_crossover",
    hypothesis="短期均线上穿长期均线的标的在后续持有期内有正超额收益",
    objectives={
        "total_return": {"min": 0.05},
        "sharpe_ratio": {"min": 0.5, "target": 1.5},
        "max_drawdown": {"max": -0.25, "target": -0.15},
    },
    benchmarks=["SPY"],
    signals={
        "golden_cross": (crossover, {"fast": "sma_fast", "slow": "sma_slow"}),
    },
    portfolio=EqualWeightOptimizer(),
)

engine = Engine()
result = engine.run(strategy,
    universe=StaticUniverse(("AAPL",)),
    market=LocalMarketDataProvider(),
    broker=sim_broker,
    rules=[ExitRule(fast="sma_fast", slow="sma_slow"),
           StopLossRule(threshold=0.05)],
    start="2023-01-01", end="2024-12-31")
```

### 4.3 组件 Protocol

```python
@runtime_checkable
class Indicator(Protocol):
    """逐 symbol 向量化计算，输出数值列。"""
    name: str
    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series: ...

@runtime_checkable
class Signal(Protocol):
    """逐 symbol 向量化计算，输出布尔/分类标签。"""
    name: str
    def compute(self, mktdata: pd.DataFrame, **params) -> pd.Series: ...

@runtime_checkable
class PortfolioOptimizer(Protocol):
    """跨 symbol 组合优化，输出目标权重。"""
    name: str
    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]: ...

@runtime_checkable
class Rule(Protocol):
    """逐 bar 有状态评估，输出 RuleResult。"""
    name: str
    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult: ...
```

### 4.4 宽表数据模型

`mktdata` 是按 symbol 索引的宽表集合（`dict[str, pd.DataFrame]`）。Indicator、Signal 阶段通过追加列逐步加宽各 symbol 的宽表。

```
原始行情             Indicator 后              Signal 后
+-----------+       +------------------+      +---------------------+
| open      |       | open             |      | open                |
| high      |       | high             |      | high                |
| low       | ───►  | low              | ───► | low                 |
| close     |       | close            |      | close               |
| volume    |       | volume           |      | volume              |
|           |       | sma_fast  (新增) |      | sma_fast            |
|           |       | sma_slow  (新增) |      | sma_slow            |
|           |       |                  |      | golden_cross (新增) |
+-----------+       +------------------+      +---------------------+
```

### 4.5 执行模型

Engine 按阶段逐层推进，回测时逐 bar 驱动管道：

```
Engine.setup() — 向量化阶段:
  Phase 1: Indicator   → 统一收集并计算所有依赖指标，追加为宽表列
  Phase 2: Signal      → 逐 symbol 计算信号，追加为宽表列

Engine.step(date) — 逐 bar 阶段:
  Phase 3: Portfolio    → PortfolioOptimizer 产出目标权重
  Phase 4: Pre-trade Rule  → 检查约束，调整权重或冻结交易
  Phase 5: Trading Algorithm → 目标权重 + 当前持仓 → 生成订单
  Phase 6: Broker       → 提交订单、撮合成交、更新持仓
  Phase 7: Post-trade Rule → 监控持仓（止损、止盈等）
  Phase 8: Broker       → 执行减仓订单
```

| 阶段 | 计算模式 | 路径依赖 |
|------|----------|----------|
| Indicator | 向量化 — 全量时间序列一次计算 | 否 |
| Signal | 向量化 — 全量时间序列一次计算 | 否 |
| Portfolio | 截面 — 当前 bar 的全 universe 优化 | 否 |
| Rule | 逐 bar 循环 — 状态机模式 | 是 |

### 4.6 Broker Protocol：策略与执行分离

三种运行模式通过注入不同实现切换，策略代码零修改：

| 模式 | MarketDataProvider | Broker |
|------|--------------------|--------|
| 回测 | `LocalMarketDataProvider` | `SimBroker` |
| Paper Trade | `AlpacaMarketDataProvider` | `SimBroker` |
| 实盘 | `AlpacaMarketDataProvider` | `LiveBroker` |

> 接入新券商？参考 [自定义 Broker 实现指南](custom-broker-guide.md)。

---

## 5. Strategy Spec 系统（新增）

### 5.1 设计动机

没有 spec，Agent 生成的是一次性代码。一次性代码难以审计、难以复现、难以比较、难以沉淀。

有 spec，策略才变成研究资产：可版本化、可 diff、可 hash、可编译、可验证、可由不同 Agent 和不同运行时重复执行。

### 5.2 Spec 文件

标准文件名：`strategy_spec.yaml`

```yaml
schema_version: "0.1"
strategy_id: "momentum_topn_weekly"
name: "20-day Momentum Top-N Weekly Rotation"

research:
  hypothesis: "Past 20-day relative strength has short-term continuation."
  rationale: "Momentum may persist due to delayed information diffusion."

market:
  asset_class: "equity"
  region: "us"
  currency: "USD"

universe:
  type: "static"
  symbols: ["SPY", "QQQ", "IWM"]
  point_in_time: false
  survivorship_bias_policy: "warn"

data:
  provider: "local"
  price_adjustment: "adjusted"
  required_columns: ["open", "high", "low", "close", "volume"]

signal:
  signal_time: "close_t"
  indicators:
    momentum_20:
      type: "NdayReturn"
      params:
        column: "close"
        period: 20
  rules:
    positive_momentum:
      type: "Threshold"
      params:
        column: "momentum_20"
        threshold: 0
        relationship: "gt"

portfolio:
  type: "TopNRanking"
  params:
    score_col: "momentum_20"
    n: 1
    filter_negative: true

execution:
  rebalance:
    frequency: "weekly"
  trade_time: "next_open"
  fill_price_mode: "next_open"
  order_timing: "next_session_open"
  price_bar: "next_session"
  price_type: "open"
  initial_cash: 100000
  cash_annual_return: 0.0
  lot_size_config:
    default: 1
    by_symbol: {}

cost:
  fee_rate: 0.001
  slippage_rate: 0.001

metrics:
  profile: open_xquant_default
  risk_free_rate: 0.0
  return_type: simple
  annualization_days: 252
  calmar_denominator: max_drawdown
  evaluation_window: full

benchmark:
  symbols: ["SPY"]

validation:
  train_period: ["2018-01-01", "2021-12-31"]
  test_period: ["2022-01-01", "2025-12-31"]
  required_oos: true

robustness:
  cost_multiplier: [1.0, 2.0]
  parameter_perturbation:
    momentum_20.period: [15, 20, 25]

decision_policy:
  reject_if:
    fatal_audit_findings: true
    oos_sharpe_lt: 0.5
    max_drawdown_lt: -0.30
  promote_if:
    oos_sharpe_gte: 1.0
    max_drawdown_gte: -0.20
```

### 5.3 Spec Validator

命令：`oxq spec validate strategy_spec.yaml`

P0 校验规则：

| 检查项 | 严重级别 | 说明 |
|--------|----------|------|
| hypothesis 为空 | fatal | 没有可测试假设 |
| universe 缺失 | fatal | 无法定义研究范围 |
| signal_time 缺失 | fatal | 无法判断是否未来函数 |
| trade_time 缺失 | fatal | 无法判断成交时点 |
| signal_time=close_t 且 trade_time=close_t | fatal | 同根 K 线生成并成交 |
| execution 语义冲突 | fatal | legacy 与显式执行字段不一致 |
| market calendar 不支持 | fatal | 非受支持交易日历 |
| lot_size_config 非法 | fatal | 交易单位不可执行 |
| metrics profile 非法 | fatal | 指标口径不可解释 |
| cost 缺失 | fatal | 默认零成本不可接受 |
| slippage 缺失 | fatal | 默认零滑点不可接受 |
| validation.test_period 缺失 | fatal | 无样本外验证 |
| benchmark 缺失 | warning | 难以判断超额收益 |
| static universe + point_in_time=false | warning | 可能有幸存者偏差 |

### 5.4 Spec Compiler

将 `strategy_spec.yaml` 编译为 open-xquant 可执行对象。

两种模式：
1. **Direct Runtime Mode**：直接从 spec 构造 Strategy 对象并运行（MVP 优先）
2. **Compiled Plan Artifact**：写出 `compiled_plan.json`，记录 spec 到运行时对象、
   执行语义和自动规则的确定性映射，并纳入 `artifact_hashes.json`
3. **Human Strategy Projection**：写出 `strategy.py`，以 Python 语法展示从
   Universe、Indicators、Signals、Portfolio、Rules 到模拟交易流程的可读投影，
   供用户 review

`strategy_spec.yaml` 仍是策略本体。`strategy.py` 是生成产物，不作为回测执行入口；
复现性审计会静态解析其中的 `STRATEGY_SPEC`、`COMPILED_PLAN` 和 hash anchor，
确认它们与 `strategy_spec.yaml`、`compiled_plan.json` 一致。`strategy.py` 可以
包含面向人类的流程函数和详细注释，但这些函数不能替代正式 runtime。

---

## 6. Audit System（新增）

### 6.1 两层审计

```
Reproducibility Audit — 验证同输入是否产生同输出
Research Bias Audit   — 判断回测研究是否可信
```

### 6.2 Reproducibility Audit

检查 spec hash、compiled plan hash、strategy.py 一致性、data manifest hash、
trades hash、equity curve hash、metrics hash、environment hash。

命令：`oxq audit reproducibility runs/<run_id>/`

### 6.3 Research Bias Audit

P0 检查项：

| ID | 级别 | 检查内容 |
|----|------|----------|
| execution_lag | fatal | 信号时间与成交时间是否冲突 |
| cost_model | fatal | 是否使用零手续费或零滑点 |
| oos_required | fatal | 是否存在样本外区间 |
| benchmark_present | warning | 是否有基准 |
| static_universe_survivorship | warning | 静态股票池是否可能幸存者偏差 |
| parameter_count | warning | 参数数量是否过多 |
| trade_count | warning | 交易次数是否过少 |
| concentration | warning | 收益是否依赖少数交易 |
| drawdown_tail | warning | 最大回撤是否不可接受 |
| missing_data | warning | 数据缺失是否严重 |

命令：`oxq audit research runs/<run_id>/`

---

## 7. Robustness Runner（新增）

P0 稳健性测试四类：

1. 样本内 / 样本外表现对比
2. 手续费与滑点加倍
3. 核心参数轻微扰动
4. 市场状态分段分析

输出 `robustness.json`，用于报告和实验比较。报告不应只复述
baseline Sharpe，而应保留 fragile、warn 和 error 状态。

命令：`oxq robustness run runs/<run_id>/`

---

## 8. Report Assets And Agent Report Writing（新增）

程序负责生成 run artifacts、审计结果、稳健性结果、指标和图表资产
manifest。最终 `research_report.md` 必须由 Agent 调用
`write-research-report` skill 写作，默认语言是中文；`research_report.html`
从最终 Markdown 渲染，不重新生成报告叙事。

```text
1. 执行结论 / Executive Decision
2. 研究假设 / Hypothesis
3. 策略配置摘要 / Strategy Spec Summary
4. 数据与执行假设 / Data and Execution Assumptions
5. 回测指标 / Backtest Metrics
6. 基准比较 / Benchmark Comparison
7. 图表资产 / Report Assets
8. 复现性审计 / Reproducibility Audit
9. 研究偏差审计 / Research Bias Audit
10. 稳健性测试 / Robustness Tests
11. 失败模式 / Failure Modes
12. 下一步 / Next Actions
```

决策规则：存在 fatal audit finding → Reject；无 fatal 但 OOS 显著退化 → Watchlist；通过 audit 且稳健性尚可 → Paper Trading Candidate。

图表和附件通过 manifest 登记：

```text
runs/<run_id>/
  report_assets/
    manifest.json
    figures/
    scripts/
    attachments/
```

报告资产命令：

```bash
oxq report asset add runs/<run_id>/ chart.png --id chart_id --title "Chart"
oxq report asset add-batch runs/<run_id>/ runs/<run_id>/report_assets/assets.json
oxq report asset list runs/<run_id>/
```

---

## 9. Experiment Registry（新增）

每次研究进入实验登记册，防止选择性记忆。

本地实现：`experiments/experiments.jsonl`

记录：experiment_id, strategy_id, spec_hash, run_id, metrics, audit_status, decision, created_at。

命令：`oxq experiment add runs/<run_id>/`

---

## 10. CLI 设计

```
oxq spec init "策略想法"
oxq spec validate strategy_spec.yaml
oxq spec-audit validate spec_audit.json
oxq strategy compile strategy_spec.yaml
oxq runtime-audit validate runtime_audit.json
oxq backtest run strategy_spec.yaml --spec-audit spec_audit.json --runtime-audit runtime_audit.json --component-catalog component_catalog.json --out runs/auto --json
oxq audit reproducibility runs/<run_id>/
oxq audit research runs/<run_id>/
oxq robustness run runs/<run_id>/
oxq experiment add runs/<run_id>/
```

CLI 是 SDK 的薄封装。业务逻辑在 SDK 中实现；最终研究报告文本由
Agent skill 完成。

---

## 11. 功能模块

### 11.1 数据层 (oxq.data)

一切外部数据统一视为 indicator——PE ratio、GDP 增速与 RSI 本质相同。最终以列的形式汇入宽表 `mktdata`。

核心原则：
1. **一切皆 indicator**
2. **Point-in-Time 对齐**
3. **频率打平**：低频数据通过 forward-fill 对齐到日频
4. **全局数据广播**：无 symbol 维度的数据广播到全 universe

### 11.2 Universe 构建 (oxq.universe)

| 实现 | 说明 | 适用场景 |
|------|------|----------|
| `StaticUniverse` | 固定 symbol 列表 | 单标的策略 |
| `IndexUniverse` | 指数成分股 | 指数轮动策略 |
| `FilterUniverse` | 基于 indicator 条件动态过滤 | 全市场因子策略 |

### 11.3 指标库 (oxq.indicators)

30+ 内置指标，按类别分：趋势 (SMA, EMA, WMA, DEMA, TEMA)、动量 (RSI, ROC, Momentum, NdayReturn)、MACD (MACDLine, MACDSignal, MACDHistogram)、波动 (Bollinger, ATR, RollingVolatility)、成交量 (OBV, VWAP, MFI)、方向 (ADX, AROON, CCI)。

### 11.4 信号生成器 (oxq.signals)

8 种信号类型：Crossover, Threshold, Comparison, Formula, Peak, Timestamp, Composite, ROCTiming。
其中 `ROCTiming` 是内置分类信号，输出 `BUY`、`SELL`、`HOLD`。
自定义分类信号用于 spec 时，应在 rule 顶层声明
`output_domain: [BUY, SELL, HOLD]`。

### 11.5 组合优化器 (oxq.portfolio.optimizers)

| 优化器 | 逻辑 |
|--------|------|
| `EqualWeightOptimizer` | 等权分配 |
| `RiskParityOptimizer` | 按波动率倒数加权 |
| `KellyOptimizer` | Kelly 公式计算最优仓位 |
| `TopNRankingOptimizer` | 按评分排名取 Top N 归一化 |
| `PctEquityOptimizer` | 每个信号标的固定权益比例 |
| `SignalToPositionOptimizer` | 将 `BUY`、`SELL`、`HOLD` 信号映射为目标仓位 |

`SignalToPositionOptimizer` 是有状态优化器：每个独立 run 开始时重置状态；
`BUY` 更新目标仓位，`SELL` 清空或降到 `sell_weight`，`HOLD` 维持上一目标仓位。
当 pre-trade Rule 只 override 部分标的时，其他 `HOLD` 标的不参与再平衡。

### 11.6 交易规则 (oxq.rules)

| 规则 | 时机 | 功能 |
|------|------|------|
| `MaxDrawdownRisk` | Pre-trade | 回撤熔断 |
| `DailyLossLimitRisk` | Pre-trade | 日亏损熔断 |
| `StopLossRule` | Post-trade | 止损 |
| `TakeProfitRule` | Post-trade | 止盈 |
| `TrailingStopRule` | Post-trade | 追踪止损 |
| `ExitRule` | Post-trade | 条件退出 |

### 11.7 交易执行 (oxq.trade)

- **OrderGenerator**：目标权重 + 当前持仓 → 订单列表
- **SimBroker**：模拟撮合，支持 market/limit/stop/trailing_stop
- **FeeModel / SlippageModel**：PercentageFee、PercentageSlippage

### 11.8 参数优化 (oxq.optimize)

GridSearch、WalkForward、TimeSeriesCV、过拟合分析。

### 11.9 因子评估 (oxq.factor_eval)

截面评估（IC、ICIR、RankIC、衰减、换手率）+ 时序评估（命中率、衰减曲线、盈亏比、Tearsheet）。

### 11.10 可观测性 (oxq.observe)

Execution tracing、AuditRecord（四维 hash）、StrategyMonitor、MarketStateDetector、ExperimentLog。

---

## 12. Tool 定义与分发

### 12.1 Tool 定义（oxq.tools）

Tool 定义与传输协议无关。每个 Tool 是 SDK 的薄封装。

| 工具组 | 工具名 | 说明 |
|--------|--------|------|
| **spec** | `spec_init`, `spec_validate` | Spec 创建与校验 |
| **strategy** | `strategy_create`, `strategy_add_indicator`, `strategy_add_signal` | 策略构建 |
| **data** | `data_load_symbols`, `data_list_symbols`, `data_inspect` | 数据管理 |
| **universe** | `universe_set`, `universe_inspect`, `universe_history` | Universe 管理 |
| **engine** | `engine_run`, `engine_results`, `engine_trade_list` | 回测执行 |
| **audit** | `audit_reproducibility`, `audit_research` | 审计 |
| **robustness** | `robustness_run` | 稳健性测试 |
| **report** | report asset CLI only | 图表与附件资产登记 |
| **experiment** | `experiment_add` | 实验登记 |
| **optimize** | `grid_search`, `walk_forward`, `cross_validate` | 参数优化 |
| **factor_eval** | `factor_evaluate`, `factor_evaluate_ts` | 因子评估 |
| **observe** | `observe_trace`, `observe_audit_*`, `observe_experiment_*` | 可观测性 |

---

## 13. Agent Layer（agent/）

### 13.1 Agent Skills（agent/skills/）

每个 skill.md 描述一个可编排的阶段能力，避免把构建、审计、运行和报告混成一个端到端流程。

| Skill | 说明 |
|-------|------|
| `build-strategy-spec` | 构建/编辑 SPEC，输出 builder phase result |
| `author-component` | 创建 workspace-local custom components、测试、manifest 和 catalog |
| `audit-strategy-spec` | 审核用户来源、默认值、组件 provenance 和 recipe canonicality |
| `audit-runtime-semantics` | 编译 preview 并审核 SPEC 到 compiled_plan 的执行语义一致性 |
| `run-authorized-backtest` | 读取授权 artifact，运行 gated backtest |
| `monitor-strategy-run` | 跑后 reproducibility/research audit、robustness 和 experiment 记录 |
| `explore-data` | 检查数据 → 下载行情/因子 → 质量检查 |
| `tune-parameters` | 参数优化 + 统计检验 |
| `review-performance` | 绩效分析 + 归因 |
| `evaluate-factor` | 因子评估路由 |
| `create-component` | 组件创建路由 |
| ... | ... |

### 13.2 Agent Roles（agent/roles/）

`agent/roles/*.md` 是 OpenXQuant multi-agent 预制角色的单一来源。
安装器会把这些角色渲染成各 Agent 的官方格式：

- Codex: `${CODEX_HOME:-~/.codex}/agents/*.toml`
- OpenCode: `~/.config/opencode/agents/*.md`
- Claude Code: `~/.claude/agents/*.md`
- Cursor: `~/.cursor/agents/*.md`

当前预制角色：

- `oxq-coordinator`: 面向用户的主控 Agent，只负责阶段路由和确认。
- `oxq-strategy-builder-worker`: 构建和验证 `strategy_spec.yaml`。
- `oxq-data-inspection-worker`: 检查数据可用性、provider readiness、
  parquet 质量和覆盖区间。
- `oxq-component-author-worker`: 创建 workspace-local Indicator、Signal、
  PortfolioOptimizer custom components；workspace-local Rule 默认阻塞。
- `oxq-spec-auditor-worker`: 审用户确认、字段来源和组件 provenance。
- `oxq-runtime-auditor-worker`: 编译并审核 runtime semantics。
- `oxq-runner-worker`: 授权后运行 backtest 和确定性跑后检查。
- `oxq-report-writer-worker`: 写图表资产和研究报告。
- `oxq-report-reviewer-worker`: 审核报告并输出 `report_review.json`。

没有官方确认 subagent 角色目录的 target 只安装 skills，不安装这些角色。

Workspace-local custom components 通过 `component_manifest.json` 加载，不修改
installed SDK bundle。确定性命令可以使用 `--component-manifest` 临时注册
extension 组件，并校验 `bundle_hash` 后再 validate、compile、export catalog
或 run backtest。

### 13.3 OpenCode 集成

OpenCode 不再保留 `agent/opencode/` 源码包。OpenCode skills 由
`agent/skills/<name>/SKILL.md` 单一来源安装到
`~/.config/opencode/skills/`；agent roles 由 `agent/roles/*.md` 单一来源
渲染到 `~/.config/opencode/agents/*.md`。

源码工作区直读 OpenCode skills 只作为开发者本地配置，不作为仓库内
target-specific 包维护。

---

## 14. 技术选型

| 决策 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.12+ | AI 生态最丰富 |
| 类型系统 | dataclass(frozen=True) + Protocol | 不可变 + 鸭子类型 |
| 金融精度 | Decimal | 避免浮点误差 |
| 时间序列 | pandas DataFrame/Series | 向量化计算基础设施 |
| 核心依赖 | pandas, numpy, pyyaml | spec 解析 + 向量化计算 |
| CLI | click | 标准 CLI 框架 |
| 构建工具 | uv | 现代 Python 项目管理 |
| 测试 | pytest | 标准选择 |

---

## 15. 实现路线

### Phase 0: 定位收敛与文档 ✅ 已完成
- README 定位更新
- 架构文档更新
- 明确 `spec → backtest → audit → report` 是 MVP 主线

### Phase 1: Spec 与 Validator ✅ 已完成
- `src/oxq/spec/schema.py`, `validator.py`
- `src/oxq/cli/` — CLI 入口（`oxq spec init`, `oxq spec validate`）
- `examples/strategies/spec_validation_demo.py`（5 个 pass/fail/warn 演示）

### Phase 2: Spec Compiler 与 Backtest Artifacts ✅ 已完成
- `src/oxq/spec/compiler.py`
- 标准化 run directory 结构，包括 `compiled_plan.json`、`strategy.py`、
  `target_weights.csv` 和
  `artifact_hashes.json` hash 覆盖
- CLI: `oxq strategy compile`, `oxq backtest run`

### Phase 3: Audit 与 Report ✅ 已完成
- `src/oxq/audit/reproducibility.py`, `research_bias.py`
- `src/oxq/report/assets.py`, `html.py`
- CLI: `oxq audit reproducibility`, `oxq audit research`, `oxq report asset *`, `oxq experiment add`

### Phase 4: Robustness 与 Experiment Registry ✅ 已完成
- `src/oxq/robustness/runner.py`
- CLI: `oxq robustness run`, `oxq experiment add`

### Phase 5: OpenCode 安装集成 ✅ 已完成
- 不再保留 `agent/opencode/` 源码包；OpenCode 通过安装器使用
  `agent/skills/<name>/SKILL.md` 和 `agent/roles/*.md`

### 已完成
- Phase 1 (原): 核心引擎 + SDK ✅
- Phase 2 (原): 参数优化 + 统计检验 + 因子评估 ✅
- Phase 3 (原): 交易执行 + 可观测性 🔄 部分完成
- Phase 0 (新): 定位收敛与文档 ✅
- Phase 1 (新): Spec 与 Validator ✅
- Phase 2 (新): Spec Compiler 与 Backtest Artifacts ✅
- Phase 3 (新): Audit 与 Report ✅
- Phase 4 (新): Robustness 与 Experiment Registry ✅
- Phase 5 (新): OpenCode 本地集成 ✅

---

## 16. 反模式（Avoid These）

| 反模式 | 说明 |
|--------|------|
| 把 open-xquant 做成新 Agent | open-xquant 是内核，不是 Agent。通用执行交给 OpenCode |
| 追求自动发现赚钱策略 | 会把系统带向过拟合机器。第一目标是**杀死坏策略** |
| 把所有壁垒都藏起来 | 开源版不可完整使用则用户不会信任 |
| 只做回测，不做审计 | 量化最危险的是"不报错但赚钱的假策略" |
| 让 Auditor 修改策略 | 审计者不能同时当优化者，否则审计变成收益包装 |
| 过早接实盘 | 研究 Agent 和交易执行系统必须隔离 |

---

## 17. 验收标准

用以下命令验证系统端到端可工作：

```bash
oxq spec init "20日动量轮动" --out strategy_spec.yaml
oxq spec validate strategy_spec.yaml
oxq spec-audit validate spec_audit.json
oxq runtime-audit validate runtime_audit.json
oxq backtest run strategy_spec.yaml --spec-audit spec_audit.json --runtime-audit runtime_audit.json --component-catalog component_catalog.json --out runs/auto --json
oxq audit research runs/<run_id>/
oxq robustness run runs/<run_id>/
oxq experiment add runs/<run_id>/
```

验证断言：

1. 缺少成本模型的 spec 会 fail
2. close_t 信号 close_t 成交会 fail
3. 缺少样本外验证会 fail
4. static universe 会产生 warning
5. 相同 spec + 相同数据重复运行，核心 hash 一致
6. 报告中不会把 fatal audit 策略标记为 Candidate

---

## 参考

- **quantstrat (R)**: indicator → signal → rule 分层模型、paramset 优化、walk-forward analysis
- **xquant.shop**: agent pipeline 架构、immutable specs、provider injection
- **Peterson, Brian G. (2017)**: *"Developing & Backtesting Systematic Trading Strategies"* — 假设驱动开发、统计检验方法论
