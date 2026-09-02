---
title: 量化策略稳健性检验 | open-xquant
description: 说明 open-xquant 稳健性检验的成本压力、参数扰动、样本外验证、行情状态和解释边界。
---

# 量化策略稳健性检验

稳健性检验用于识别回测结果是否过度依赖单一成本、单一参数或单一市场阶段。open-xquant 的 P0 robustness runtime 来自 `src/oxq/robustness/runner.py`，会读取已完成 run 的 spec 与 metrics，并写入 `robustness.json`。

## 为什么需要稳健性检验

单次回测只说明某组输入下的结果。即使审计通过，研究者仍需要检查交易成本上升、样本外区间、参数变化和行情状态变化下的表现是否暴露脆弱性。

## 成本压力

runtime 会构造 cost x2 压力测试，比较 baseline Sharpe 与成本翻倍后的 perturbed Sharpe。若成本变化让表现明显恶化，结果需要在报告中解释，不能把原始回测当成稳健结论。

## 参数扰动

runner 会根据 spec 中的敏感性线索执行参数扰动，观察指标是否对小范围参数变化高度敏感。参数调优场景也可以参考 [量化参数调优](/skills/tune-parameters)。

## 样本外验证

IS/OOS comparison 用于对比样本内与样本外表现。样本外结果弱化并不自动否定全部研究，但会降低结论强度，并要求回到假设、数据和执行边界重新解释。

## 结果解释

robustness 状态可能是 robust、warn、fragile 或 error。它不是收益保证，也不是风险消除证明。后续应结合 [量化回测审计](/workflows/research-audit)、[量化研究报告撰写](/skills/write-research-report) 和最终候选选择一起判断。
