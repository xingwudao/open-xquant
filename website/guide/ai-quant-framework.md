---
title: 什么是 AI 量化框架 | open-xquant
description: 解释 AI 量化框架在 open-xquant 中的含义、适用对象、边界和下一步学习路径。
---

# AI 量化框架

AI 量化框架不是让模型自由编写交易代码后直接下单，而是把 AI 擅长的意图理解和人类确认，接到确定性量化内核的可验证执行上。open-xquant 的定位是 Agentic Quant Research Kernel：声明式规格、确定性回测、审计、稳健性检验和结构化报告。

## 适合谁

- 想用中文描述策略想法，并希望 Agent 帮助整理成可审计研究流程的学习者。
- 需要把策略版本、运行产物、报告和最终选择分开治理的量化研究者。
- 正在构建 AI 量化应用，需要底层确定性执行与产物标准的开发者。

## 解决什么问题

传统回测框架通常假设人类开发者手写代码并管理状态。AI 参与后，过多自由度会带来实现路径漂移、隐式假设和不可复现结果。open-xquant 用 `strategy_spec.yaml`、编译、审计、artifact hash 和报告 QA，把研究结论绑定到具体数据、参数、执行假设和产物。

底层引擎围绕 Universe、Indicator、Signal、Portfolio、Rule 和 Broker 组合执行，策略表达“做什么”，引擎负责“怎么做”。这使同一策略可以在不同 Universe 上运行，也让回测结果有明确输入边界。

## 工作边界

open-xquant 不保证策略盈利，不消除市场风险，也不替代研究判断。Agent 层负责提问、来源追溯、报告解释和最终选择；CLI / SDK 负责 validate、compile、backtest、audit、robustness 和 report QA 等确定性动作。

如果你正在比较策略候选，仍需要看数据覆盖、交易成本、样本外表现、偏差审计和产物血缘，而不是只看单次收益率。

## 下一步

先阅读 [AI Agent 量化研究](/guide/agentic-quant-research) 理解 Agent 与 Skills 的分工，再进入 [可复现量化研究](/guide/reproducible-quant-research) 学习版本、运行和报告治理。准备跑示例时，可以从 [量化研究示例](/examples/) 和 [AI 量化回测](/workflows/strategy-backtest) 开始。
