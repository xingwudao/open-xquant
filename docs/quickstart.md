# Quick Start

5 分钟跑通第一个回测。

## 1. 安装

```bash
git clone https://github.com/xingwudao/open-xquant
cd open-xquant
pip install -e ".[yfinance]"
```

使用 `uv` 则无需显式安装，直接在仓库目录下运行：
```bash
uv run oxq --help
```

## 2. 创建策略 Spec

```bash
oxq spec init "SMA crossover: buy SPY when 10-day SMA crosses above 50-day SMA" --out strategy_spec.yaml
```

打开 `strategy_spec.yaml`，填入关键字段：

```yaml
strategy_id: sma_crossover

research:
  hypothesis: "短期均线上穿长期均线产生正超额收益"

universe:
  symbols: ["SPY"]
  point_in_time: false

signal:
  signal_time: close_t
  indicators:
    sma_fast:  { type: SMA, params: { column: close, period: 10 } }
    sma_slow:  { type: SMA, params: { column: close, period: 50 } }
  rules:
    golden_cross: { type: Crossover, params: { fast: sma_fast, slow: sma_slow } }

portfolio:
  type: EqualWeight

execution:
  trade_time: next_open
  fill_price_mode: next_open
  initial_cash: 100000

cost:
  fee_rate: 0.001
  slippage_rate: 0.001

benchmark:
  symbols: ["SPY"]

validation:
  train_period: ["2018-01-01", "2021-12-31"]
  test_period:  ["2022-01-01", "2025-12-31"]
  required_oos: true
```

## 3. 验证

```bash
oxq spec validate strategy_spec.yaml
# Status: PASS
```

## 4. 回测

```bash
oxq backtest run strategy_spec.yaml --out runs/auto
# Run complete. Artifacts written to runs/20260616_120000_sma_crossover/
#   Total Return: 161.57%
#   Sharpe Ratio: 0.83
#   Max Drawdown: -30.80%
```

## 5. 审计

```bash
oxq audit research runs/<run_id>/
# Status: PASS (0 fatal, 0 warnings)
```

## 6. 报告

```bash
oxq report write runs/<run_id>/
# research_report.md written
# Decision: PAPER TRADING CANDIDATE
```

## 下一步

- 查看所有可用命令：`oxq --help`
- 模块示例：`uv run python examples/modules/01_spec_and_validate.py`
- 完整架构：`docs/architecture.md`
- Spec 校验演示：`uv run python examples/strategies/spec_validation_demo.py`
