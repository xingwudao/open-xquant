# Human Guide - 用 Agent 使用 open-xquant

这份文档给真实用户看。

你不需要学习 `oxq` 的每个命令，也不需要手写
`strategy_spec.yaml`。open-xquant 的推荐使用方式是：

```text
你提出研究想法 -> Agent 阅读 open-xquant 指南和 skills
              -> Agent 调用确定性研究内核
              -> 产出可复现、可审计的研究报告
```

`docs/agent-guide.md` 是给 Agent 读的执行手册。你第一次使用时把它交给
Agent；之后在新目录里通常只需要直接说策略想法。

---

## 1. 第一次使用

先把项目放到本机：

```bash
git clone https://github.com/xingwudao/open-xquant
cd open-xquant
```

然后打开你常用的 Agent，例如 Codex、OpenCode、Claude Code、Cursor 或 OpenClaw，
确保你的 Agent 在刚刚下载的 open-xquant 文件夹中，然后对它说：

```text
请阅读 docs/agent-guide.md。
完成 open-xquant 的长期 Agent 能力安装。
安装后做一次最小环境验证研究，确认以后我可以在新目录里直接说策略想法。
```

Agent 应该完成这些事：

- 检查 Python、`uv` 和项目环境。
- 安装 open-xquant skills 到你的长期 Agent 能力目录。
- 写入 `~/.config/open-xquant/agent.yaml`。
- 跑通一次最小研究闭环。
- 告诉你验证结果和报告路径。

这一步通常只需要做一次。

---

## 2. 以后在新目录使用

此后，你就不必再在open-xquant 下从事量化研究了。

假如你新建一个研究目录：

```bash
mkdir my-research
cd my-research
```

打开 Agent，直接说你的策略想法：

```text
我想研究 SPY 的 20/60 日均线趋势策略。
用 2018-2021 作为训练期，2022-2025 作为测试期。
初始资金 100000，手续费 0.1%，滑点 0.1%。
请完成研究并输出报告。
```

如果长期能力安装成功，Agent 会自动：

- 识别 open-xquant 相关任务。
- 读取已安装的 `strategy-builder` 等 skills。
- 在当前目录初始化研究 workspace。
- 生成并验证 `strategy_spec.yaml`。
- 准备或请求行情数据。
- 运行回测、审计、稳健性检查和报告。
- 把结论写到 `runs/<run_id>/research_report.md`。

你不需要再次粘贴 `docs/agent-guide.md`。

---

## 3. 你应该告诉 Agent 什么

策略研究至少需要这些信息：

- 标的或股票池。
- 策略想法。
- 训练期和测试期。
- 初始资金。
- 手续费和滑点。
- benchmark。
- 你关心的目标指标，例如 Sharpe、回撤或收益。
- 风险约束，例如最大回撤、仓位上限或止损规则。

信息不完整时，Agent 应该先追问，而不是直接编造假设。

如果策略输出的是 `BUY`、`SELL`、`HOLD` 这类交易意图，Agent 应该把它建模为
`Signal`，再用 `SignalToPosition` 转成目标仓位。`EqualWeight` 只适合布尔过滤
信号，不适合直接消费分类交易意图。自定义分类信号用于 spec 时，应在 signal
rule 顶层声明 `output_domain: [BUY, SELL, HOLD]`。

可以这样说：

```text
研究一个 AAPL、MSFT、NVDA 的动量轮动策略。
每月调仓，选择过去 6 个月收益最高的 1 只。
训练期 2018-2021，测试期 2022-2025。
初始资金 100000，手续费 0.1%，滑点 0.1%。
benchmark 用 SPY。
重点看 OOS Sharpe 和最大回撤。
```

---

## 4. 你应该期待什么产物

一次完整研究至少应该留下：

- `strategy_spec.yaml`
- `runs/<run_id>/metrics.json`
- `runs/<run_id>/execution_assumptions.json`
- `runs/<run_id>/equity_curve.csv`
- `runs/<run_id>/benchmark_curve.csv`，如果配置了 benchmark。
- `runs/<run_id>/trades.csv`
- `runs/<run_id>/target_weights.csv`，用于目标仓位和 baseline 对齐比较。
  `trades.csv` 用于执行成交比较，不要把两者混用。
- `runs/<run_id>/orders.csv`
- `runs/<run_id>/positions.csv`
- `runs/<run_id>/artifact_hashes.json`
- `runs/<run_id>/robustness.json`
- `runs/<run_id>/research_report.md`
- 可复现性审计结果。
- 研究偏差审计结果。
- 稳健性检查结果。

Agent 的最终回答应该明确区分：

- 环境验证是否成功。
- spec 校验是否通过。
- 使用了什么 metrics profile 和指标口径。
- 使用了什么成交价格、交易日历、交易单位和现金收益假设。
- 审计是否有 fatal 或 warning。
- 稳健性检查是否为 `PASS`、`WARN` 或 `FAIL`。
- IS/OOS、参数扰动和 regime 分段是否暴露了脆弱性。
- 策略是候选、观察、拒绝，还是仅用于演示。

不要接受只有收益率、没有审计和报告路径的结论。

---

## 5. 升级

如果你想让 Agent 更新长期能力，对它说：

```text
请使用 open-xquant 更新已安装到长期能力中的 skills。
如果当前目录是 open-xquant 仓库，请从当前本地分支更新。
更新后运行 agent status，并告诉我结果。
```

如果你希望从 GitHub 更新，对 Agent 说：

```text
请从 GitHub 更新 open-xquant 的长期 Agent 能力。
更新后运行 agent status，并告诉我哪些 Agent 已更新。
```

---

## 6. 卸载

如果你不想继续让 Agent 自动使用 open-xquant，对它说：

```text
请卸载 open-xquant 安装到所有长期 Agent 能力中的内容。
不要删除我的研究目录、runs、reports、experiments.jsonl 或行情数据。
卸载后运行 agent status，并告诉我结果。
```

卸载只应该删除 open-xquant 管理的 skills 和说明块，不应该删除你的研究产物。

---

## 7. 常见问题

### 我换目录后还要重新安装吗？

通常不需要。第一次安装完成后，Agent 会在新目录中读取
`~/.config/open-xquant/agent.yaml`，用里面的 runner 调用 open-xquant。

### Agent 说找不到 `oxq` 怎么办？

让 Agent 读取：

```text
~/.config/open-xquant/agent.yaml
~/.config/open-xquant/agent-install.json
```

然后使用 `preferred_runner` 或 `source.path`，不要在你的 home 目录里随机搜索
其他 open-xquant 副本。

### 我需要自己准备数据吗？

不一定。Agent 可以用 open-xquant 的数据工具准备样例数据。正式研究时，你应该确认
数据来源、时间范围和复权方式。缺数据时，Agent 应该停下来说明缺口。

### 这是投资建议吗？

不是。open-xquant 产出的是可复现研究材料。即使审计通过，也不等于策略可以实盘。
实盘前还需要独立验证、风控、权限确认和纸面交易。

### 我想手动使用 CLI 怎么办？

可以让 Agent 打开 `docs/agent-guide.md` 或直接运行：

```bash
uv run oxq --help
```

但对多数用户来说，更低门槛的方式是直接把研究目标交给 Agent。
