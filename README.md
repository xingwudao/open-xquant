# open-xquant

**Agent First 量化投资交易框架**

覆盖数据、因子、信号、回测、模拟交易、实盘交易全流程。

[English](#english) | [中文](#为什么需要-open-xquant)

---

## 为什么需要 open-xquant？

### 传统量化框架的困境

现有的量化回测框架（如 Backtrader、vnpy、Zipline 等）是为**程序员**设计的。它们假设使用者能精确编写每一行代码，能手动管理状态与数据流，能在复杂的 API 文档中找到正确的调用方式。

这在过去没有问题——因为人就是唯一的编程主体。

### AI 时代的新矛盾

大语言模型（LLM）正在重塑软件开发方式。越来越多的人开始通过 AI 编程（Vibe Coding）来构建量化策略。但这带来了一个根本性矛盾：

**AI 擅长理解意图、生成代码——但它会产生幻觉。**

当前主流 AI 基于 Transformer 架构，其生成过程本质上是概率性的。同一个提示词，两次生成的代码可能存在细微差异。这种不确定性在大多数软件领域可以接受，但在金融交易中是致命的：

> **不可重复 = 不可信 = 不可交易**

一个回测结果如果无法精确复现，那它就没有任何决策价值。

### 问题的根源

问题不在于 AI 不够聪明，而在于**现有框架从未考虑过 AI 作为使用者**。当 AI 被迫使用为人类设计的框架时：

- **过多的自由度** → AI 每次可能选择不同的实现路径
- **隐式的约定** → AI 无法可靠地遵守文档中未明确表述的规则
- **状态管理的复杂性** → AI 容易在多步操作中引入不一致

## open-xquant 的解法

open-xquant 采用 **Agent First** 的设计哲学——框架的首要使用者是 AI Agent，而非人类程序员。

这不意味着人不能使用它，而是意味着：

### 1. 声明式优先，而非命令式

用户（或 AI）描述**"做什么"**，框架负责**"怎么做"**。减少实现路径的分歧，从源头降低不确定性。

### 2. 确定性执行保证

相同的输入，必须产生相同的输出。框架层面强制保证回测结果的可重复性，不依赖使用者（无论是人还是 AI）的自律。

### 3. 约束即自由

通过精心设计的约束，收窄 AI 的选择空间。当正确的做法只有一种时，幻觉就无处可幻。

### 4. 全流程一体化

从数据获取到实盘交易，统一的数据模型和执行引擎。消除流程衔接中的不一致性——这正是 AI 最容易出错的地方。

## 设计目标

```
数据 → 因子 → 信号 → 回测 → 模拟交易 → 实盘交易
 ↑                                              |
 └──────────── 统一数据模型 & 执行语义 ──────────┘
```

| 目标 | 说明 |
|------|------|
| **可重复性** | 同一策略 + 同一数据 = 同一结果，无例外 |
| **AI 友好** | 最小化 AI 出错的可能性，最大化 AI 的生产力 |
| **人类可审计** | AI 生成的策略，人类能完整理解和验证 |
| **全流程覆盖** | 一个框架走完从研究到交易的全部流程 |
| **渐进式使用** | 从单因子回测到多策略实盘，按需渐进 |

## 谁适合使用 open-xquant？

- **AI 时代的量化学习者**：通过 Vibe Coding 学习量化投资，无需成为资深程序员
- **量化策略研究者**：专注于策略逻辑本身，而非框架的使用方式
- **AI 应用开发者**：构建基于 LLM 的自动化量化交易 Agent

## 通过示例学习

`examples/` 目录提供了由浅入深的学习路径，帮助你快速上手 open-xquant。

### 推荐学习顺序

**第一步：模块教程（`examples/tutorials/`）**

交互式 Jupyter Notebook，逐步讲解各核心模块的用法：

| Notebook | 内容 |
|----------|------|
| `data_module.ipynb` | 数据下载与读取——学习如何使用 YFinance/AkShare 下载美股和 A 股行情数据，并通过 `LocalMarketDataProvider` 统一读取 |
| `universe_module.ipynb` | 标的池构建——学习如何使用 `StaticUniverse` 定义固定标的池，以及使用 `FilterUniverse` 基于量价规则动态筛选 |
| `backtest_module.ipynb` | 回测引擎——学习 Indicator → Signal → Rule 三阶段模型，使用 `BacktestEngine` 运行 SMA 均线交叉策略回测 |

```bash
# 启动 Jupyter 运行教程
pip install open-xquant[yfinance,akshare]
jupyter notebook examples/tutorials/
```

**第二步：策略示例（`examples/strategies/`）**

完整的策略代码示例，展示 Indicator → Signal → Rule 三阶段模型的实际应用：

| 文件 | 策略类型 |
|------|----------|
| `sma_crossover.py` | SMA 均线交叉策略（完整回测示例） |
| `ma_crossover.py` | 均线交叉策略 |
| `momentum_rotation.py` | 动量轮动策略 |
| `mean_reversion.py` | 均值回归策略 |
| `multi_strategy.py` | 多策略组合 |

**第三步：Agent 应用（`examples/app/`）**

| 文件 | 说明 |
|------|------|
| `agent_demo.py` | 基于 Streamlit 的 AI Agent 演示，通过 MCP 协议调用 open-xquant 工具，体验 Agent First 的交互方式 |

```bash
# 运行 Agent Demo
pip install streamlit openai mcp nest_asyncio
streamlit run examples/app/agent_demo.py
```

## 项目状态

open-xquant 正处于早期设计阶段。我们正在：

- 定义核心数据模型和接口规范
- 设计声明式策略描述语言
- 构建确定性执行引擎的原型

欢迎关注项目进展，参与讨论。

## License

[MIT](LICENSE)

---

<a id="english"></a>

# open-xquant

**Agent First Quantitative Trading Framework**

End-to-end coverage: data, factors, signals, backtesting, paper trading, and live trading.

---

## Why open-xquant?

### The Problem with Traditional Quant Frameworks

Existing backtesting frameworks (Backtrader, vnpy, Zipline, etc.) are designed for **programmers**. They assume users can write every line of code precisely, manage state and data flow manually, and navigate complex API documentation to find the right calls.

This was fine in the past — humans were the only ones writing code.

### A New Contradiction in the AI Era

Large Language Models (LLMs) are reshaping software development. More and more people are building quantitative strategies through AI programming (Vibe Coding). But this introduces a fundamental contradiction:

**AI is great at understanding intent and generating code — but it hallucinates.**

Current mainstream AI is based on the Transformer architecture, where generation is inherently probabilistic. The same prompt can produce subtly different code on two runs. This uncertainty is acceptable in most software domains, but in financial trading it's fatal:

> **Not reproducible = not trustworthy = not tradable**

A backtest result that cannot be exactly reproduced has zero decision-making value.

### Root Cause

The problem isn't that AI isn't smart enough — it's that **existing frameworks were never designed with AI as a user**. When AI is forced to use frameworks built for humans:

- **Too many degrees of freedom** → AI may choose different implementation paths each time
- **Implicit conventions** → AI cannot reliably follow rules not explicitly stated in code
- **Complex state management** → AI easily introduces inconsistencies across multi-step operations

## The open-xquant Approach

open-xquant adopts an **Agent First** design philosophy — the framework's primary user is an AI Agent, not a human programmer.

This doesn't mean humans can't use it. It means:

### 1. Declarative First, Not Imperative

Users (or AI) describe **"what to do"**; the framework handles **"how to do it"**. Fewer divergent implementation paths, less uncertainty at the source.

### 2. Deterministic Execution Guarantee

Same input must produce same output. Reproducibility of backtest results is enforced at the framework level, not dependent on the discipline of the user — human or AI.

### 3. Constraints as Freedom

Carefully designed constraints narrow AI's choice space. When there's only one correct way to do something, hallucination has nowhere to go.

### 4. End-to-End Integration

From data acquisition to live trading — unified data models and execution engine. Eliminates inconsistencies at process boundaries, exactly where AI is most prone to errors.

## Design Goals

```
Data → Factors → Signals → Backtest → Paper Trading → Live Trading
 ↑                                                        |
 └──────────── Unified Data Model & Execution Semantics ──┘
```

| Goal | Description |
|------|-------------|
| **Reproducibility** | Same strategy + same data = same result, no exceptions |
| **AI-Friendly** | Minimize AI error surface, maximize AI productivity |
| **Human-Auditable** | AI-generated strategies that humans can fully understand and verify |
| **Full Pipeline** | One framework from research to trading |
| **Progressive** | From single-factor backtest to multi-strategy live trading, adopt incrementally |

## Who Is This For?

- **Quant learners in the AI era**: Learn quantitative investing through Vibe Coding — no need to be a senior programmer
- **Quant strategy researchers**: Focus on strategy logic, not framework mechanics
- **AI application developers**: Build LLM-powered automated trading agents

## Learn by Examples

The `examples/` directory provides a progressive learning path to help you get started with open-xquant.

### Recommended Learning Order

**Step 1: Module Tutorials (`examples/tutorials/`)**

Interactive Jupyter Notebooks that walk you through each core module:

| Notebook | Content |
|----------|---------|
| `data_module.ipynb` | Data download & reading — learn to fetch US and China A-share market data via YFinance/AkShare, and read it through `LocalMarketDataProvider` |
| `universe_module.ipynb` | Universe construction — learn to define a fixed symbol pool with `StaticUniverse` and dynamically filter with `FilterUniverse` based on price/volume rules |
| `backtest_module.ipynb` | Backtest engine — learn the Indicator → Signal → Rule three-phase model, run an SMA crossover strategy backtest with `BacktestEngine` |

```bash
# Launch Jupyter to run tutorials
pip install open-xquant[yfinance,akshare]
jupyter notebook examples/tutorials/
```

**Step 2: Strategy Examples (`examples/strategies/`)**

Complete strategy code demonstrating the Indicator → Signal → Rule three-phase model:

| File | Strategy Type |
|------|---------------|
| `sma_crossover.py` | SMA Crossover (complete backtest example) |
| `ma_crossover.py` | Moving Average Crossover |
| `momentum_rotation.py` | Momentum Rotation |
| `mean_reversion.py` | Mean Reversion |
| `multi_strategy.py` | Multi-Strategy Portfolio |

**Step 3: Agent Application (`examples/app/`)**

| File | Description |
|------|-------------|
| `agent_demo.py` | Streamlit-based AI Agent demo that calls open-xquant tools via MCP protocol — experience the Agent First interaction model |

```bash
# Run the Agent Demo
pip install streamlit openai mcp nest_asyncio
streamlit run examples/app/agent_demo.py
```

## Project Status

open-xquant is in early design phase. We are currently:

- Defining core data models and interface specifications
- Designing the declarative strategy description language
- Building a prototype of the deterministic execution engine

Follow along and join the discussion.

## License

[MIT](LICENSE)
