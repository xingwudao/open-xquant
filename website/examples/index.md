---
title: 量化研究示例 | open-xquant
description: 汇总 open-xquant 仓库中的回测、审计、因子和实盘示例，说明依赖、命令、产物和限制。
---

# 量化研究示例

这些示例来自 open-xquant 仓库，适合用来理解 SDK、CLI 等价入口和标准产物。示例输出不能直接代表可交易结论；正式研究仍要进入审计、稳健性检验、报告和最终选择。

## Spec 与校验

[examples/modules/01_spec_and_validate.py](https://github.com/xingwudao/open-xquant/blob/main/examples/modules/01_spec_and_validate.py)

- Data dependency: 示例 spec 和本地临时输出，不需要外部行情。
- Command entry: `uv run python examples/modules/01_spec_and_validate.py`
- Expected artifact type: `/tmp/oxq_examples/01_spec/strategy_spec.yaml` 和校验结果。
- Limitation: 只展示 spec 创建和 P0 校验，不代表完成回测研究。

## 回测与产物

[examples/modules/03_backtest_and_artifacts.py](https://github.com/xingwudao/open-xquant/blob/main/examples/modules/03_backtest_and_artifacts.py)

- Data dependency: 依赖先运行 `01_spec_and_validate.py` 生成 spec；示例使用 SDK 编译并写入 `/tmp/oxq_examples/03_backtest`。
- Command entry: `uv run python examples/modules/03_backtest_and_artifacts.py`
- Expected artifact type: metrics、trades、target weights、equity curve 和 artifact hashes。
- Limitation: 示例包含 `--allow-unaudited` 的 CLI 等价说明，正式研究应走 [AI 量化回测](/workflows/strategy-backtest) 的授权门槛。

## 审计与稳健性

[examples/modules/04_audit_and_robustness.py](https://github.com/xingwudao/open-xquant/blob/main/examples/modules/04_audit_and_robustness.py)

- Data dependency: 依赖 `03_backtest_and_artifacts.py` 已生成 completed run。
- Command entry: `uv run python examples/modules/04_audit_and_robustness.py`
- Expected artifact type: `reproducibility_audit.json`、`research_bias_audit.json` 和 `robustness.json`。
- Limitation: 示例展示审计调用方式；报告解释仍需要结合 [量化回测审计](/workflows/research-audit) 与 [量化策略稳健性检验](/workflows/robustness-testing)。

## 因子评估

[examples/modules/08_factor_eval.py](https://github.com/xingwudao/open-xquant/blob/main/examples/modules/08_factor_eval.py)

- Data dependency: 使用 YFinance 下载 `SPY`、`QQQ`、`IWM` 历史数据并从本地 provider 读取。
- Command entry: `uv run python examples/modules/08_factor_eval.py`
- Expected artifact type: IC、Rank IC、ICIR、decay、turnover 和 tearsheet 相关输出。
- Limitation: 示例 symbols 较少，横截面结论需要谨慎；正式因子研究应先确认样本、horizon 和 forward return 对齐。

## 策略管线

[examples/strategies/sma_crossover_spec.py](https://github.com/xingwudao/open-xquant/blob/main/examples/strategies/sma_crossover_spec.py)

- Data dependency: 示例策略数据和本地运行目录，具体依赖以脚本内配置为准。
- Command entry: `uv run python examples/strategies/sma_crossover_spec.py`
- Expected artifact type: 端到端策略 run、审计和报告相关文件。
- Limitation: 示例用于学习完整管线，不应跳过用户确认、执行假设和成本审计。

## Alpaca paper trading

[examples/app/live_trading_demo.py](https://github.com/xingwudao/open-xquant/blob/main/examples/app/live_trading_demo.py)

- Data dependency: 需要安装 live extra，并通过环境变量提供 Alpaca paper API key。
- Command entry: `uv run python examples/app/live_trading_demo.py`
- Expected artifact type: broker 连接、订单提交回执、fills 和 open orders 控制台输出。
- Limitation: 示例默认 paper trading；live order submission 必须按 [AI 量化实盘交易](/workflows/live-trading) 重新确认风险。
