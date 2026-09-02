---
title: AI 量化框架 | open-xquant
description: 面向中文 AI 量化研究的 open-xquant 入口，说明框架边界、工作流、Agent Skills 与入门路径。
---

# AI 量化研究框架

open-xquant 是中文友好的 AI 量化研究框架，面向 AI Coding Agent 和人类量化研究者，提供策略回测、因子研究、稳健性检验、审计报告与实盘交易工作流。

open-xquant 不是一个交易机器人，也不是替代研究判断的自动收益机器。它是面向 AI Agent 的 Agentic Quant Research Kernel：用声明式规格、确定性执行和可审计产物，把交易想法转化为可复现的量化研究流程。

## open-xquant 定义与边界

open-xquant 的核心边界是把不稳定的策略描述收敛为结构化证据链。Agent 可以帮助提问、整理、审计和写报告；CLI / SDK 负责可重复的 validate、compile、backtest、audit、robustness、report QA 和实验登记。

框架目标不是更快生成更多策略，而是更快识别不可复现、未授权、数据泄漏或证据链断裂的回测结果。实盘相关能力需要明确凭据、账户、订单和风险确认，不会把回测信号直接变成无人值守订单。

## 从想法到最终候选

典型研究路径如下：

1. 交易想法进入 [AI Agent 量化研究](/guide/agentic-quant-research)。
2. 想法和规格接受 [量化回测审计](/workflows/research-audit)。
3. 用户确认完整 `strategy_spec.yaml` 后进入 [AI 量化回测](/workflows/strategy-backtest)。
4. 回测产物继续接受可复现审计、偏差审计和 [量化策略稳健性检验](/workflows/robustness-testing)。
5. 报告、对比和最终候选选择保留版本、运行、哈希和审稿证据。

## 真实架构图

<img
  src="/images/open-xquant-subagent-collaboration.png"
  width="1920"
  height="1080"
  alt="open-xquant 中协调 Agent、专业 SubAgent 与确定性量化研究内核的协作关系"
>

## 策略回测、因子研究、审计、稳健性与实盘工作流

- [AI 量化回测](/workflows/strategy-backtest)：从授权过的 spec、spec audit 和 runtime audit 执行正式回测。
- [AI 因子研究](/workflows/factor-research)：区分横截面选股因子和时间序列择时因子，检查 forward return 对齐。
- [量化回测审计](/workflows/research-audit)：检查想法、规格、运行语义、偏差和产物血缘。
- [量化策略稳健性检验](/workflows/robustness-testing)：查看成本压力、参数扰动、样本外和行情状态分析。
- [AI 量化实盘交易](/workflows/live-trading)：说明 Alpaca paper / live 连接、订单确认和停止边界。

## Agent Skills 分组入口

[Agent Skills 索引](/skills/) 按研究治理、策略与审计、数据与因子、组件开发、执行与报告组织页面。常用入口包括 [open-xquant AI 量化研究路由](/skills/open-xquant)、[授权量化回测执行](/skills/run-authorized-backtest)、[量化因子评估](/skills/evaluate-factor) 和 [实盘交易管理](/skills/manage-live-trading)。

## 快速开始链接

- [Human Guide](https://github.com/xingwudao/open-xquant/blob/main/docs/human-guide.md)：给人类用户看的安装入口。
- [Agent Guide](https://github.com/xingwudao/open-xquant/blob/main/docs/agent-guide.md)：给 AI Agent 执行的长期能力安装指南。
- [GitHub 仓库](https://github.com/xingwudao/open-xquant)：查看源码、示例和 issue。
