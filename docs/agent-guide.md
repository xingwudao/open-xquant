# Agent Guide — open-xquant 安装与使用

你是一个 AI Coding Agent。本文档指导你完成 open-xquant 的安装、验证和使用。

完成本文档所有步骤后，你就可以使用 `oxq` CLI 命令和 `oxq` Python SDK 为用户执行量化研究任务。

---

## 1. 安装

```bash
git clone https://github.com/xingwudao/open-xquant /tmp/open-xquant
cd /tmp/open-xquant
pip install -e ".[yfinance]"
```

如果项目已经在本地，跳过 clone，直接在本项目目录下执行后续命令。

> **重要：** 以下所有 `uv run oxq` 命令需要在项目根目录（包含 `pyproject.toml` 的目录）下执行。

**验证安装：**

```bash
uv run oxq --help
```

预期输出以 `Usage: oxq [OPTIONS] COMMAND [ARGS]...` 开头。

---

## 2. 验证核心工作流

运行一条完整的 spec → backtest → audit → report 管线：

```bash
# 创建策略 spec
uv run oxq spec init "SMA crossover: buy SPY when 10-day SMA crosses above 50-day SMA" --out /tmp/oxq_agent_test.yaml

# 验证 spec
uv run oxq spec validate /tmp/oxq_agent_test.yaml
# 预期: Status: PASS（可能有一条 survivorship warning）

# 运行回测
uv run oxq backtest run /tmp/oxq_agent_test.yaml --out /tmp/oxq_agent_runs
# 预期: Run complete. Artifacts written to ...

# 审计
RUN_DIR=$(ls -d /tmp/oxq_agent_runs/*/ | head -1)
uv run oxq audit research "$RUN_DIR"
# 预期: Status: PASS 或列出 warning

# 生成报告
uv run oxq report write "$RUN_DIR"
# 预期: Report written to .../research_report.md
```

全部通过则环境就绪。

---

## 3. 可用 Skill

项目在 `agent/skills/` 目录下提供了 20 个 Skill 文件。每个 Skill 指导你如何完成一项量化研究任务。

**使用 Skill 的方法：** 读取对应的 `.md` 文件，按其中的步骤执行。

**Skill 速查表：**

| Skill | 用途 | 何时加载 |
|-------|------|----------|
| `strategy-builder.md` | 从假设出发构建 strategy_spec.yaml | 用户提出策略想法时 |
| `rule-builder.md` | 配置风控和止损止盈规则 | 需要添加风控规则时 |
| `strategy-monitor.md` | 审计回测结果、监控策略健康 | 回测完成后 |
| `trade-executor.md` | 配置交易成本（手续费/滑点） | 用户关心执行成本时 |
| `data-explorer.md` | 下载行情、检查数据质量 | 需要准备数据时 |
| `factor-evaluator.md` | 评估因子的预测能力 | 用户询问因子有效性时 |
| `parameter-tuner.md` | 参数优化和过拟合分析 | 需要调参时 |
| `universe-builder.md` | 定义标的池 | 需要设定交易品种时 |
| `live-trader.md` | Alpaca 模拟/实盘交易 | 需要连接券商时 |
| `performance-reviewer.md` | 审查回测绩效、对比多次运行 | 需要解读结果时 |
| `chart-indicator.md` | 渲染 K 线图和指标叠加 | 需要可视化时 |
| `factor-screening.md` | 多因子筛选 | 需要选股时 |
| `component-creator.md` | 创建新的 Indicator/Signal/Rule 组件 | 需要扩展框架时 |
| `create-indicator.md` | 创建新指标 | component-creator 的子 skill |
| `create-signal.md` | 创建新信号 | component-creator 的子 skill |
| `create-rule.md` | 创建新规则 | component-creator 的子 skill |
| `create-portfolio-optimizer.md` | 创建新组合优化器 | component-creator 的子 skill |
| `evaluate-cross-sectional.md` | 截面因子评估（IC, RankIC） | factor-evaluator 的子 skill |
| `evaluate-time-series.md` | 时序因子评估（Hit Rate, Tearsheet） | factor-evaluator 的子 skill |
| `backtest-runner.md` | 重定向到 strategy-builder | 历史兼容 |

---

## 4. CLI 命令速查

```
uv run oxq spec init "<想法>"                         创建策略 spec 模板
uv run oxq spec validate <file>                       验证 spec 文件
uv run oxq strategy compile <file>                    编译 spec 为可执行策略
uv run oxq backtest run <file> --out runs/auto        运行回测
uv run oxq audit reproducibility <run_dir>            可复现性审计
uv run oxq audit research <run_dir>                   研究偏差审计
uv run oxq robustness run <run_dir>                   稳健性测试
uv run oxq report write <run_dir>                     生成研究报告
uv run oxq experiment add <run_dir>                   登记实验
```

---

## 5. SDK 使用

当需要自定义计算逻辑时，使用 Python SDK：

```python
from oxq.core import Engine, Strategy
from oxq.indicators import SMA
from oxq.signals import Crossover
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.universe import StaticUniverse
from oxq.data.market import LocalMarketDataProvider
from oxq.trade.sim_broker import SimBroker
from oxq.trade.fees import PercentageFee
from oxq.trade.slippage import PercentageSlippage

# 构建策略
crossover = Crossover()
crossover.required_indicators = {
    "sma_10": (SMA(), {"column": "close", "period": 10}),
    "sma_50": (SMA(), {"column": "close", "period": 50}),
}

strategy = Strategy(
    name="my_strategy",
    hypothesis="短期均线上穿长期均线产生正超额收益",
    benchmarks=["SPY"],
    universe=StaticUniverse(("SPY",)),
    signals={"golden_cross": (crossover, {"fast": "sma_10", "slow": "sma_50"})},
    portfolio=EqualWeightOptimizer(),
)

# 运行回测
engine = Engine()
result = engine.run(
    strategy=strategy,
    market=LocalMarketDataProvider(),
    broker=SimBroker(
        fee_model=PercentageFee(),
        slippage_model=PercentageSlippage(),
    ),
    start="2020-01-01", end="2024-12-31",
)

print(f"Sharpe: {result.sharpe_ratio():.2f}")
print(f"Max DD: {result.max_drawdown():.2%}")
print(f"Trades: {len(result.trades)}")
```

**原则：** 即使使用 SDK 自定义逻辑，也必须遵循 open-xquant 框架——通过 `oxq spec validate` 校验、留下结构化 artifacts、运行 audit 和 report。

---

## 6. 参考

- 完整架构：`docs/architecture.md`
- 人类快速入门：`docs/quickstart.md`
- 模块示例：`examples/modules/01_spec_and_validate.py` 等 10 个
- E2E 示例：`examples/strategies/sma_crossover_spec.py`

---

## 7. 红线

- 不跳过 spec validate 直接回测
- 不允许零手续费或零滑点
- 不让 Builder 同时做 Auditor
- 不修改已验证的 spec 来美化结果
- 不回测完成后不运行 audit
