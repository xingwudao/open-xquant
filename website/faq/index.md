---
title: AI 量化常见问题 | open-xquant
description: 回答 open-xquant 在 AI 量化、策略审计、可复现、实盘边界、Skills 与 CLI 分工上的问题。
---

# AI 量化常见问题

## open-xquant 是 AI 交易机器人吗？

不是。open-xquant 是面向 AI Agent 和人类研究者的确定性量化研究内核。Agent 负责理解意图、组织流程和审计上下文；open-xquant CLI / SDK 负责可复现执行和产物标准。

## AI 生成的量化策略为什么还需要审计？

AI 生成内容可能包含未确认假设、实现漂移、数据泄漏、同 bar 成交误用或不一致的成本设置。审计把想法、规格、运行语义和产物血缘拆开检查，避免把概率生成结果直接当成研究证据。

## 相同策略为什么必须能够复现？

如果相同 spec、相同数据和相同执行假设不能复现结果，就无法判断差异来自市场、代码、数据还是 Agent 的改写。可复现性让报告、比较和最终选择有可追溯证据。

## open-xquant 能直接用于实盘交易吗？

open-xquant 支持 Alpaca paper 和 live 连接能力，但默认使用 paper mode。任何 live endpoint、订单列表和风险暴露都需要用户明确确认；回测信号不能直接转成无人值守订单。

## Agent Skills 和 CLI 分别负责什么？

Agent Skills 负责路由、提问、来源追溯、阶段边界、报告语义和最终判断。CLI / SDK 负责 validate、compile、backtest、audit、robustness、report QA、workspace init 和 agent install/status 这类确定性动作。

## 如何开始第一次量化研究？

先按 Human Guide 克隆仓库并让 Agent 阅读 Agent Guide 安装长期能力。安装后，在研究目录描述策略想法，让 Agent 从 [open-xquant AI 量化研究路由](/skills/open-xquant) 开始；想先学习产物结构，可以运行 [量化研究示例](/examples/)。
