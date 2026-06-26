# open-xquant

open-xquant 是面向 AI Coding Agent 和人类量化研究者的 **Agentic Quant Research Kernel**。

> 本框架源于 [xquant.shop](https://xquant.shop) 量化研究平台的实践沉淀。

它把交易想法转化为可声明、可复现、可审计、可沉淀的研究产物：

```
spec → validate → compile → backtest → audit → robustness → report
```

open-xquant 不是一个 Coding Agent，而是 Coding Agent 应该调用的确定性量化研究内核。
CLI / SDK 只负责可复现的底层执行；报告撰写、图表选择、实验对比和
spec 来源追溯这类需要综合判断的任务，交给已安装的 Agent skills 驱动。

它的目标不是更快生成更多策略，而是更快识别和拒绝假的回测结果。

→ **[Human Guide](docs/human-guide.md)**（用 Agent 使用 open-xquant）
→ **[Agent Guide](docs/agent-guide.md)**（给 AI Agent 看的安装指南）

[English](#english) | [中文](#为什么需要-open-xquant)

---

## 为什么需要 open-xquant？

### 传统量化框架的困境

现有的量化回测框架（如 Backtrader、vnpy、Zipline 等）是为**程序员**设计的。它们假设使用者能精确编写每一行代码，能手动管理状态与数据流，能在复杂的 API 文档中找到正确的调用方式。

这在过去没有问题——因为人就是唯一的编程主体。

### AI 时代的新矛盾

大语言模型（LLM）正在重塑软件开发方式。越来越多的人开始通过 AI 编程来构建量化策略。但这带来了一个根本性矛盾：

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

open-xquant 采用 **Agentic Quant Research Kernel** 的设计哲学——它是一个给 AI Agent 使用的确定性量化研究内核，提供声明式策略规格、确定性回测、偏差审计、稳健性测试和研究报告产物标准。

### 1. 声明式优先

用户（或 AI）描述**"做什么"**，框架负责**"怎么做"**。策略通过 `strategy_spec.yaml` 声明，减少实现路径的分歧，从源头降低不确定性。

### 2. 确定性执行保证

相同 spec + 相同数据 = 相同回测结果，无例外。框架层面强制保证可重复性，不依赖使用者（无论是人还是 AI）的自律。

### 3. 约束即安全

通过 spec validation、research bias audit、robustness tests 三道防线，收窄 AI 的选择空间。当错误的做法会被自动检测时，幻觉就无处可存。

### 4. 结构化研究产物

每次研究都留下固定结构的 artifacts——metrics、trades、target weights、
equity curve、artifact hashes、audit、report——可版本化、可 diff、可沉淀。

## 核心流程

```
Agent loads open-xquant skill
  → build-strategy-spec writes and validates strategy_spec.yaml
  → audit-strategy-spec checks assumption provenance
  → oxq spec-audit validate spec_audit.json
  → audit-runtime-semantics compiles and checks compiled_plan.json semantics
  → oxq runtime-audit validate runtime_audit.json
  → run-authorized-backtest runs gated backtest
  → oxq audit reproducibility runs/<run_id>/
  → oxq audit research runs/<run_id>/
  → oxq robustness run runs/<run_id>/
  → build-report-charts registers chart assets when needed
  → write-research-report writes research_report.md/html
  → oxq report qa runs/<run_id>/
  → review-research-report reviews the final report
  → oxq experiment add runs/<run_id>/
```

这里的 `oxq` CLI 步骤是确定性 primitives：验证、编译、回测、审计、
稳健性、报告文件与资产完整性 QA，以及实验登记。报告数值叙事是否合理、
图表是否足以支撑结论、是否接受某个 run 为最终版本，都需要由 skill
结合上下文判断。

## 谁适合使用 open-xquant？

- **AI 时代的量化学习者**：通过声明式 spec 学习量化投资，无需成为资深程序员
- **量化策略研究者**：专注于策略逻辑本身，框架负责验证、审计、报告
- **AI 应用开发者**：构建基于 LLM 的自动化量化研究 Agent

## 通过示例学习

`examples/` 目录提供了由浅入深的学习路径。

### 推荐学习顺序

**第一步：模块示例（`examples/modules/`）**

可执行的 Python 脚本，逐个演示核心模块的 SDK 和等价 CLI 用法：

| 文件 | 内容 |
|------|------|
| `01_spec_and_validate.py` | Spec 创建与 P0 校验 |
| `02_data_and_universe.py` | 数据下载、读取、Universe 构建 |
| `03_backtest_and_artifacts.py` | Spec 编译、回测执行、artifact 读取 |
| `04_audit_and_robustness.py` | 可复现审计、偏差审计、稳健性测试 |
| `05_report_and_experiment.py` | 报告 artifact、QA 与实验登记 |
| `06_signals_and_rules.py` | Signal、Rule、ROCTiming 与 BUY/SELL/HOLD 语义 |

```bash
uv run python examples/modules/01_spec_and_validate.py
```

**第二步：Spec 校验（`examples/strategies/spec_validation_demo.py`）**

展示 validator 对 5 种 spec 的判断（pass / fail / warn）：

```bash
uv run python examples/strategies/spec_validation_demo.py
```

**第三步：策略示例（`examples/strategies/`）**

完整的端到端策略管线示例（spec → backtest → audit → report）：

| 文件 | 策略类型 |
|------|----------|
| `sma_crossover_spec.py` | SMA 均线交叉 — 完整 E2E 管线 |
| `momentum_rotation_spec.py` | 动量轮动 — 完整 E2E 管线 |
| `roc_timing_spec.py` | ROC 择时 — fixed threshold 与 rolling quantile spec |
| `factor_screen.py` | 多因子筛选示例 |

## 项目边界

open-xquant 是完整可用的开源研究内核，聚焦确定性计算、
声明式 spec、审计、artifact QA 和 Agent 可调用的 CLI / SDK / Tools。
需要语义判断的报告、图表和实验对比由 Agent skills 编排。

不属于 open-xquant 核心边界的能力：

- 托管式云端状态机。
- 多用户协作和计费。
- 私有研究记忆图谱。
- 私有 Eval Corpus。
- 托管式 PIT 数据服务。

原则：**开源版必须能独立完成可复现的量化研究闭环。**

## 项目状态

open-xquant 正在从 Agent First 量化交易框架升级为 Agentic Quant Research Kernel。

已完成：
- 核心引擎 (Engine, Strategy, types, registry)
- 30+ 指标库、8 种信号、9 种规则、6 种组合优化器
- 因子评估 (IC, ICIR, decay, turnover, tearsheet)
- 参数优化 (grid search, walk-forward, cross-validation)
- 可观测性 (tracing, audit, monitoring, experiment log)
- Strategy Spec (schema, validator, compiler)
- Audit System (reproducibility + research bias)
- Runtime execution assumptions (calendar, fill price, lot size, cash return)
- Metrics profiles (`open_xquant_default`, `xquant_production`)
- Robustness Runner (cost stress, IS/OOS diff, parameter perturbation, regimes)
- Report asset manifest and deterministic report QA
- Agent skills for report writing, chart building, spec auditing, and
  experiment comparison
- Workspace-local custom component manifests and deterministic extension
  loading
- Multi-Agent role presets for Codex, OpenCode, Claude Code, and Cursor,
  including component authoring
- OpenCode 集成

## License

[MIT](LICENSE)

---

<a id="english"></a>

# open-xquant

open-xquant is an **Agentic Quant Research Kernel** for AI coding agents and human quant researchers.

> This framework emerged from building the [xquant.shop](https://xquant.shop) quant research platform.

It turns trading ideas into declarative, reproducible, auditable, and persistent research artifacts:

```
spec → validate → compile → backtest → audit → robustness → report
```

open-xquant is not a coding agent. It is the deterministic quant research runtime that coding agents should use.
The CLI / SDK provide reproducible execution primitives; tasks that require
contextual judgment, such as report writing, chart selection, experiment
comparison, and spec provenance review, are driven by installed Agent skills.

Its goal is not to generate more strategies faster, but to make false backtests easier to detect and reject.

→ **[Human Guide](docs/human-guide.md)** (use open-xquant through an Agent)
→ **[Agent Guide](docs/agent-guide.md)** (installation guide for AI agents)

---

## Why open-xquant?

### The Problem with Traditional Quant Frameworks

Existing backtesting frameworks (Backtrader, vnpy, Zipline, etc.) are designed for **programmers**. They assume users can write every line of code precisely, manage state and data flow manually, and navigate complex API documentation.

This was fine in the past — humans were the only ones writing code.

### A New Contradiction in the AI Era

LLMs are reshaping software development. More people are building quant strategies through AI programming. But this introduces a fundamental contradiction:

**AI is great at understanding intent and generating code — but it hallucinates.**

Current mainstream AI generation is inherently probabilistic. The same prompt can produce subtly different code on two runs. This uncertainty is acceptable in most software domains, but in financial trading it's fatal:

> **Not reproducible = not trustworthy = not tradable**

### Root Cause

The problem isn't that AI isn't smart enough — it's that **existing frameworks were never designed with AI as a user**. When AI is forced to use frameworks built for humans:

- **Too many degrees of freedom** → AI may choose different implementation paths each time
- **Implicit conventions** → AI cannot reliably follow rules not explicitly stated
- **Complex state management** → AI easily introduces inconsistencies across multi-step operations

## The open-xquant Approach

open-xquant is an **Agentic Quant Research Kernel** — a deterministic runtime that provides declarative strategy specs, deterministic backtests, bias audits, robustness tests, and research report standards.

### 1. Declarative First

Users (or AI) describe **"what to do"**; the framework handles **"how to do it"**. Strategies are declared via `strategy_spec.yaml`, reducing divergent implementation paths at the source.

### 2. Deterministic Execution Guarantee

Same spec + same data = same backtest result, no exceptions. Reproducibility is enforced at the framework level.

### 3. Constraints as Safety

Three defense lines — spec validation, research bias audit, and robustness tests — narrow AI's choice space. When wrong approaches are automatically detected, hallucination has nowhere to go.

### 4. Structured Research Artifacts

Every research run produces fixed-structure artifacts — metrics, trades,
target weights, equity curve, artifact hashes, audit, report — versionable,
diffable, and persistent.

## Core Workflow

```
Agent loads the open-xquant skill
  → build-strategy-spec writes and validates strategy_spec.yaml
  → audit-strategy-spec checks assumption provenance
  → oxq spec-audit validate spec_audit.json
  → audit-runtime-semantics compiles and checks compiled_plan.json semantics
  → oxq runtime-audit validate runtime_audit.json
  → run-authorized-backtest runs gated backtest
  → oxq audit reproducibility runs/<run_id>/
  → oxq audit research runs/<run_id>/
  → oxq robustness run runs/<run_id>/
  → build-report-charts registers chart assets when needed
  → write-research-report writes research_report.md/html
  → oxq report qa runs/<run_id>/
  → review-research-report reviews the final report
  → oxq experiment add runs/<run_id>/
```

The `oxq` CLI steps are deterministic primitives: validation, compilation,
backtesting, audits, robustness, report file and asset-integrity QA, and
experiment registration. Skills handle contextual judgment, including whether
numeric narratives are justified, whether charts support the conclusion, and
whether a run should be accepted as final.

## Who Is This For?

- **Quant learners in the AI era**: Learn quant investing through declarative specs
- **Quant strategy researchers**: Focus on strategy logic; the framework handles validation, audit, and reporting
- **AI application developers**: Build LLM-powered automated quant research agents

## Learn by Examples

### Step 1: Module Examples (`examples/modules/`)

Runnable Python scripts demonstrating each core module with SDK and equivalent CLI:

| File | Content |
|------|---------|
| `01_spec_and_validate.py` | Spec creation & P0 validation |
| `02_data_and_universe.py` | Data download, inspect, universe construction |
| `03_backtest_and_artifacts.py` | Spec compile, backtest run, artifact inspection |
| `04_audit_and_robustness.py` | Reproducibility audit, bias audit, robustness tests |
| `05_report_and_experiment.py` | Report artifacts, QA, and experiment registry |
| `06_signals_and_rules.py` | Signals, rules, ROCTiming, and BUY/SELL/HOLD semantics |

```bash
uv run python examples/modules/01_spec_and_validate.py
```

### Step 2: Spec Validation (`examples/strategies/spec_validation_demo.py`)

Demonstrates 5 validator outcomes (pass / fail / warn):

```bash
uv run python examples/strategies/spec_validation_demo.py
```

### Step 3: Strategy Examples (`examples/strategies/`)

Complete E2E pipeline examples (spec → backtest → audit → report):

| File | Strategy Type |
|------|---------------|
| `sma_crossover_spec.py` | SMA Crossover — complete E2E pipeline |
| `momentum_rotation_spec.py` | Momentum Rotation — complete E2E pipeline |
| `roc_timing_spec.py` | ROC Timing — fixed threshold and rolling quantile specs |
| `factor_screen.py` | Multi-factor screening example |

## Project Boundaries

open-xquant is a complete open-source research kernel focused on deterministic
computation, declarative specs, audits, artifact QA, and Agent-callable
CLI / SDK / Tools. Semantic report writing, chart selection, and experiment
comparison are orchestrated by Agent skills.

Capabilities outside the core open-xquant boundary:

- Hosted cloud state machines.
- Multi-user collaboration and billing.
- Private research memory graphs.
- Private eval corpora.
- Hosted PIT data services.

Principle: **The open-source package must independently complete a
reproducible quant research loop.**

## Project Status

open-xquant is upgrading from an Agent First trading framework to an Agentic Quant Research Kernel.

Completed:
- Core engine (Engine, Strategy, types, registry)
- 30+ indicators, 8 signals, 9 rules, 6 portfolio optimizers
- Factor evaluation (IC, ICIR, decay, turnover, tearsheet)
- Parameter optimization (grid search, walk-forward, cross-validation)
- Observability (tracing, audit, monitoring, experiment log)
- Strategy Spec (schema, validator, compiler)
- Audit System (reproducibility + research bias)
- Runtime execution assumptions (calendar, fill price, lot size, cash return)
- Metrics profiles (`open_xquant_default`, `xquant_production`)
- Robustness Runner (cost stress, IS/OOS diff, parameter perturbation, regimes)
- Report asset manifest and deterministic report QA
- Agent skills for report writing, chart building, spec auditing, and
  experiment comparison
- Workspace-local custom component manifests and deterministic extension
  loading
- Multi-Agent role presets for Codex, OpenCode, Claude Code, and Cursor,
  including component authoring
- OpenCode integration

## License

[MIT](LICENSE)
