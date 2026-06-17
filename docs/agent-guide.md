# Agent Guide - open-xquant 安装与使用

你是一个 AI Coding Agent。本文档指导你在用户环境中安装、
验证和使用 `open-xquant`。

`open-xquant` 不是 Agent。它是 Agent 应该调用的确定性量化研究内核。
你的职责是把用户的交易想法转成可验证的研究产物，而不是临场手写一套
不可复现的回测代码。

核心流程：

```text
idea -> strategy_spec.yaml -> validate -> backtest
     -> audit -> robustness -> report -> experiment
```

完成本文档后，你应该能够：

- 使用 `oxq` CLI 完成最小闭环。
- 根据用户意图选择合适的 `agent/skills/*.md`。
- 在缺数据、校验失败、审计失败时给出明确处理路径。
- 只在确有必要时进入 SDK 或组件扩展。

---

## 0. Agent 执行协议

每次使用 `open-xquant` 前，先遵守这些规则。

1. 先确认用户目标。
   - 用户要的是策略构建、回测、因子评估、调参、审计、可视化，
     还是实盘连接。
   - 不要把模糊想法直接当成可交易策略。

2. 优先走声明式 spec。
   - 策略唯一真源是 `strategy_spec.yaml`。
   - 不要为了快速出结果临时写一个独立回测脚本。

3. 每个研究 run 必须留下 artifacts。
   - 必须能找到 `metrics.json`、`equity_curve.csv`、`trades.csv`、
     `strategy_spec.yaml`、`artifact_hashes.json`。

4. 不通过验证就停止。
   - `oxq spec validate` 失败时，不允许回测。
   - `oxq audit research` 出现 fatal 时，不允许包装成好结果。

5. 区分研究结论和工程验证。
   - 下载样例数据并跑通流程，只说明环境可用。
   - 真实策略结论必须基于用户确认的数据、时间段、成本和假设。

---

## 1. 当前稳定可用范围

这部分是当前 `oxq` CLI 的安全路径。Agent 应先使用这些能力。

稳定可用：

- Python `>=3.12`。
- `uv run oxq ...` 或已安装环境中的 `oxq ...`。
- `static` universe。
- `local` data provider，也就是本地 parquet 行情文件。
- `XNYS` 市场日历。
- `signal_time: close_t`。
- `execution.trade_time: next_open`。
- `execution.fill_price_mode: next_open`。
- `EqualWeight` 组合。
- `SMA`、`Crossover` 等注册表中的内置指标和信号。
- 回测后的 reproducibility audit、research audit、robustness、report。

不要在默认 spec 工作流中承诺这些能力已完整可编译：

- `index` universe。
- `filter` universe。
- 非 `local` provider 直接参与 `oxq backtest run`。
- 同一 spec 中多个 `Crossover` 规则。
- `Peak` 信号作为因果回测信号。
- `Timestamp` 的 `month_end` 或 `quarter_end` 规则。
- 带 signal rules 的非 `EqualWeight` portfolio。

如果用户需要上面的扩展能力，先说明当前 CLI 约束，再改用 SDK、
组件开发或后续框架扩展流程。

---

## 2. 环境探测

在执行安装前，先在目标目录运行：

```bash
pwd
python --version
uv --version
```

判断当前目录是否已经是 `open-xquant` 仓库：

```bash
test -f pyproject.toml && test -d src/oxq && echo "open-xquant repo"
```

如果当前目录是用户自己的项目，不要把研究输出散落在源码目录。
优先使用用户指定目录；未指定时使用 `/tmp/oxq_agent_*`。

---

## 3. 安装方式

推荐 Agent 在仓库根目录使用 `uv`。根目录是包含 `pyproject.toml`
的目录。

### 3.1 已经在 open-xquant 仓库内

```bash
uv sync --extra yfinance
uv run oxq --help
```

预期输出以这一行开头：

```text
Usage: oxq [OPTIONS] COMMAND [ARGS]...
```

### 3.2 需要临时安装 open-xquant

```bash
git clone https://github.com/xingwudao/open-xquant /tmp/open-xquant
cd /tmp/open-xquant
uv sync --extra yfinance
uv run oxq --help
```

### 3.3 不使用 uv 的环境

```bash
python -m pip install -e ".[yfinance]"
oxq --help
```

注意：

- 如果使用 `pip install -e`，后续命令通常写成 `oxq ...`。
- 如果使用 `uv sync`，后续命令通常写成 `uv run oxq ...`。
- 不要混用虚拟环境后假设依赖一定存在。

---

## 4. 准备行情数据

`oxq backtest run` 默认读取本地 parquet 数据，不会自动下载行情。
干净环境下，缺少数据会看到类似错误：

```text
No data for 'SPY'. Run downloader first.
```

用 `yfinance` 准备一个最小验证数据集：

```bash
uv run python - <<'PY'
from pathlib import Path

from oxq.data.loaders import YFinanceDownloader

data_dir = Path("/tmp/oxq_agent_data")
YFinanceDownloader().download_many(
    symbols=["SPY"],
    start="2018-01-01",
    end="2026-01-01",
    dest_dir=data_dir,
)
print(data_dir)
PY
```

如果网络不可用，不要伪造真实研究结论。你可以：

- 询问用户是否已有本地 parquet 行情目录。
- 使用用户提供的目录作为 `--data-dir`。
- 只做 CLI 安装验证，不做研究结论。

本地 parquet 要求：

- 文件名是 `<SYMBOL>.parquet`，例如 `SPY.parquet`。
- index 是 tz-aware `DatetimeIndex`。
- 至少包含列 `open`、`high`、`low`、`close`、`volume`。

---

## 5. 最小工作流验证

以下命令只验证环境和主流程。不要把结果当作投资建议。

```bash
uv run oxq spec init \
  "SMA crossover: buy SPY when 10-day SMA crosses above 50-day SMA" \
  --out /tmp/oxq_agent_test.yaml
```

验证 spec：

```bash
uv run oxq spec validate /tmp/oxq_agent_test.yaml
```

预期：

```text
Status: PASS
```

可能出现 survivorship warning。这不阻止环境验证，但真实研究中必须向用户说明。

运行回测：

```bash
uv run oxq backtest run \
  /tmp/oxq_agent_test.yaml \
  --data-dir /tmp/oxq_agent_data \
  --out /tmp/oxq_agent_runs
```

记录 run 目录：

```bash
RUN_DIR=$(find /tmp/oxq_agent_runs -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
echo "$RUN_DIR"
```

执行审计和报告：

```bash
uv run oxq audit reproducibility "$RUN_DIR"
uv run oxq audit research "$RUN_DIR"
uv run oxq robustness run "$RUN_DIR"
uv run oxq report write "$RUN_DIR"
uv run oxq experiment add "$RUN_DIR" \
  --registry /tmp/oxq_agent_runs/experiments.jsonl
```

环境就绪的判断标准：

- `spec validate` 返回 `Status: PASS`。
- `backtest run` 输出 `Run complete`。
- `audit reproducibility` 返回 `Status: PASS`。
- `report write` 生成 `research_report.md`。
- run 目录中存在标准 artifacts。

如果 `robustness run` 返回 `WARN`，不一定代表环境失败。
它可能只是说明未配置参数扰动或 regime analysis。
真实研究报告中必须保留这些 warning。

---

## 6. 用户任务路由

当用户提出请求时，先选择对应 skill，再执行 CLI 或 SDK。
Skill 文件位于 `agent/skills/`。

策略想法或新策略：

- 加载 `agent/skills/strategy-builder.md`。
- 产物是 `strategy_spec.yaml`。
- 必须让用户确认假设、品种、成本、时间段和目标。

标的池：

- 加载 `agent/skills/universe-builder.md`。
- 当前 CLI 稳定路径优先使用 `static` universe。
- 如果用户要求指数成分或动态过滤，先说明当前 spec 编译限制。

数据准备：

- 加载 `agent/skills/data-explorer.md`。
- 先检查本地 parquet，再决定是否下载。
- 不要在用户未确认数据源时做正式结论。

交易成本、成交价、滑点：

- 加载 `agent/skills/trade-executor.md`。
- `fee_rate` 和 `slippage_rate` 必须大于 `0`。
- 默认安全组合是 `close_t` 信号和 `next_open` 成交。

风控、止损、止盈、调仓约束：

- 加载 `agent/skills/rule-builder.md`。
- 先判断当前需求能否通过 spec/CLI 覆盖。
- 不能覆盖时，再使用 SDK 组合规则。

回测后监控和审计：

- 加载 `agent/skills/strategy-monitor.md`。
- 必须运行 reproducibility audit 和 research audit。

绩效解读：

- 加载 `agent/skills/performance-reviewer.md`。
- 读取 `metrics.json`、`trades.csv`、`equity_curve.csv` 和报告。
- 先讲风险和限制，再讲收益指标。

因子评估：

- 加载 `agent/skills/factor-evaluator.md`。
- 由它路由到 `evaluate-cross-sectional.md` 或
  `evaluate-time-series.md`。

参数优化：

- 加载 `agent/skills/parameter-tuner.md`。
- 必须区分 IS 和 OOS。
- 不要把最优 IS 参数直接宣传为可交易参数。

可视化：

- 加载 `agent/skills/chart-indicator.md`。
- 用于图表检查，不替代 audit。

组件扩展：

- 先加载 `agent/skills/component-creator.md`。
- 再路由到 `create-indicator.md`、`create-signal.md`、
  `create-rule.md` 或 `create-portfolio-optimizer.md`。
- 扩展组件时必须读现有实现和测试模式。

实盘或模拟盘：

- 加载 `agent/skills/live-trader.md`。
- 必须确认 API key、paper/live 模式、权限和风险边界。
- 不要把回测结果直接转成实盘指令。

---

## 7. CLI 速查

查看入口：

```bash
uv run oxq --help
```

创建 spec：

```bash
uv run oxq spec init "<strategy idea>" --out strategy_spec.yaml
```

验证 spec：

```bash
uv run oxq spec validate strategy_spec.yaml
uv run oxq spec validate strategy_spec.yaml --json
```

编译检查：

```bash
uv run oxq strategy compile strategy_spec.yaml
```

运行回测：

```bash
uv run oxq backtest run strategy_spec.yaml --out runs/auto
uv run oxq backtest run strategy_spec.yaml --data-dir /path/to/parquet --out runs/auto
```

审计：

```bash
uv run oxq audit reproducibility runs/<run_id>/
uv run oxq audit research runs/<run_id>/
```

稳健性：

```bash
uv run oxq robustness run runs/<run_id>/
uv run oxq robustness run runs/<run_id>/ --json
```

报告：

```bash
uv run oxq report write runs/<run_id>/
uv run oxq report write runs/<run_id>/ --out report.md
```

实验登记：

```bash
uv run oxq experiment add runs/<run_id>/
uv run oxq experiment add runs/<run_id>/ --registry experiments.jsonl
```

---

## 8. Spec 最小模板

如果 `oxq spec init` 生成模板后需要人工补齐，优先保持这个安全形态：

```yaml
schema_version: "0.1"
strategy_id: sma_crossover
name: SMA Crossover

research:
  hypothesis: "短期均线上穿长期均线后，SPY 在后续持有期内有正收益。"

market:
  asset_class: equity
  region: us
  currency: USD
  calendar: XNYS

universe:
  type: static
  symbols: ["SPY"]
  point_in_time: false
  survivorship_bias_policy: warn

data:
  provider: local
  price_adjustment: adjusted
  required_columns: ["open", "high", "low", "close", "volume"]

signal:
  signal_time: close_t
  indicators:
    sma_fast:
      type: SMA
      params: { column: close, period: 10 }
    sma_slow:
      type: SMA
      params: { column: close, period: 50 }
  rules:
    golden_cross:
      type: Crossover
      params: { fast: sma_fast, slow: sma_slow }

portfolio:
  type: EqualWeight
  params: {}

execution:
  trade_time: next_open
  fill_price_mode: next_open
  rebalance:
    frequency: daily
    interval_days: 1
  lot_size: 1
  initial_cash: 100000

cost:
  fee_rate: 0.001
  fee_min: 0.0
  slippage_rate: 0.001

benchmark:
  symbols: ["SPY"]

validation:
  train_period: ["2018-01-01", "2021-12-31"]
  test_period: ["2022-01-01", "2025-12-31"]
  required_oos: true
```

常见 fatal 原因：

- `research.hypothesis` 为空。
- `universe.type` 不是 `static`。
- `data.provider` 不是 `local`。
- `market.calendar` 不是 `XNYS`。
- `signal.signal_time` 不是 `close_t`。
- `execution.trade_time` 和 `fill_price_mode` 不匹配。
- `close_t` 信号使用同根 K 线成交价。
- `fee_rate <= 0` 或 `slippage_rate <= 0`。
- 缺少 `validation.test_period`。
- `train_period` 和 `test_period` 重叠。
- 信号参数引用了不存在的指标列。

---

## 9. SDK 使用原则

优先使用 CLI。只有在这些场景进入 SDK：

- 用户需要自定义数据源。
- 用户需要自定义指标、信号、组合优化器或规则。
- 用户需要组合多个 run、读取 artifacts 或做二次分析。
- 当前 CLI 尚未覆盖用户明确需要的能力。

最小 SDK 回测形态如下。它仍然读取 spec，并写出标准 artifacts：

```python
from oxq.spec import StrategySpec, compile_run
from oxq.spec.validator import validate

spec = StrategySpec.from_yaml("strategy_spec.yaml")
result = validate(spec)
if result.status == "fail":
    raise SystemExit(result.errors)

run_result, run_dir = compile_run(
    spec,
    data_dir="/path/to/parquet",
    out_dir="runs/auto",
)

print(run_dir)
print(run_result.sharpe_ratio())
```

读取 artifacts：

```python
import json
from pathlib import Path

run_dir = Path("runs/<run_id>")
metrics = json.loads((run_dir / "metrics.json").read_text())
print(metrics["sharpe_ratio"])
```

组件开发不要从本文档复制临时代码。应加载对应 skill，阅读现有模块，
补测试，并注册到 `src/oxq/core/registry.py`。

---

## 10. 失败处理

`uv run oxq --help` 失败：

- 确认当前目录包含 `pyproject.toml`。
- 运行 `uv sync --extra yfinance`。
- 如果不用 `uv`，确认当前 Python 环境已安装 `open-xquant`。

`ModuleNotFoundError: yfinance`：

- 使用 `uv sync --extra yfinance`。
- 或使用 `python -m pip install -e ".[yfinance]"`。

`No data for '<SYMBOL>'`：

- 先下载行情。
- 或用 `--data-dir /path/to/parquet` 指向用户数据。
- 不要跳过数据问题继续解释收益。

`Status: FAIL`：

- 读取 fatal errors。
- 修改 spec。
- 重新运行 `oxq spec validate`。
- 不要直接运行 backtest。

`audit research` 有 warning：

- 报告 warning 的含义。
- 判断是否影响当前阶段。
- 不要删除 warning 或美化结论。

`audit research` 有 fatal：

- 标记研究失败。
- 回到 spec 或数据源修正。
- 不要给出推进到 paper/live trading 的建议。

`robustness run` 返回 `WARN`：

- 检查是否未配置参数扰动或 regime analysis。
- 在报告中保留 fragile/warn 信息。
- 不要只引用 baseline Sharpe。

---

## 11. 给用户的最低门槛话术

当用户只给一句策略想法时，你可以这样推进：

```text
我会先把你的想法转成 strategy_spec.yaml，然后做校验、
回测、审计和报告。开始前需要确认 5 个约束：
交易标的、训练/测试时间段、初始资金、手续费/滑点、
以及你想用什么指标判断成功。
```

当用户只想快速试用：

```text
我会用 SPY 均线交叉跑一个环境验证样例。
这个结果只证明 open-xquant 可以在当前机器上运行，
不代表策略有效。
```

当用户要求实盘：

```text
实盘前必须先有通过审计的研究报告，并确认 paper/live 模式、
券商权限、最大亏损限制和订单权限。我不会把未审计回测
直接转成实盘指令。
```

---

## 12. 参考入口

- 架构说明：`docs/architecture.md`
- 人类快速入门：`docs/quickstart.md`
- 自定义 Broker：`docs/custom-broker-guide.md`
- 模块示例：`examples/modules/`
- 策略示例：`examples/strategies/`
- Agent skills：`agent/skills/`
- OpenCode 集成：`agent/opencode/`

---

## 13. 红线

- 不跳过 `oxq spec validate`。
- 不用零手续费或零滑点做正式结论。
- 不让构建者同时美化审计结论。
- 不修改已验证 spec 来美化回测结果。
- 不删除 warning、fatal 或失败 artifacts。
- 不把 mock data 或环境验证样例包装成真实研究。
- 不把未审计回测推进到 paper trading 或 live trading。
- 不承诺当前 CLI 尚未稳定支持的 universe/provider 能力。
