---
title: AI Agent 量化研究 | open-xquant
description: 说明 open-xquant 如何让 AI Agent 通过路由 Skill、窄阶段和确定性内核完成量化研究。
---

# AI Agent 量化研究

在 open-xquant 中，AI Agent 不直接扮演量化框架。Agent 先加载 `open-xquant` router skill，再由它把任务交给更具体的 Skill，例如想法梳理、规格审计、授权回测、因子评估、报告撰写或实盘连接。

## 适合谁

- 希望用自然语言启动量化研究，但不希望 Agent 任意改写执行规则的用户。
- 维护多阶段研究流程，需要把每个阶段的职责、输入和输出写清楚的团队。
- 给 Agent 安装长期量化能力，并希望在其他研究目录复用的开发者。

## 解决什么问题

Agentic 研究的关键不是让 Agent 一次性完成所有事情，而是让它按阶段收集证据、请求确认、调用确定性工具并保留产物。`docs/agent-guide.md` 要求 Agent 遇到量化研究、回测、因子、调参、审计、报告、图表资产、SDK 开发或实盘连接任务时，先加载 `open-xquant` skill，再路由到更具体的 skill。

常见路径包括 [open-xquant AI 量化研究路由](/skills/open-xquant)、[量化策略想法梳理](/skills/brainstorm-strategy-idea)、[量化策略规格审计](/skills/audit-strategy-spec) 和 [授权量化回测执行](/skills/run-authorized-backtest)。

## 工作边界

Agent 可以解释上下文、检查来源和组织报告，但不能把未确认字段当成事实，也不能绕过 spec audit、runtime audit 或 backtest authorization。CLI / SDK 的职责是确定性 primitives；报告数值叙事、图表选择和最终候选接受仍要由 Skill 结合证据判断。

你也不应该要求 Agent 把回测结果解释成收益承诺。open-xquant 只帮助建立更可审计的研究证据。

## 下一步

如果你还没有安装长期能力，按 GitHub 中的 Human Guide 和 Agent Guide 操作。已安装后，从 [AI 量化框架](/guide/ai-quant-framework) 理解整体边界，再选择 [AI 量化回测](/workflows/strategy-backtest)、[AI 因子研究](/workflows/factor-research) 或 [AI 量化实盘交易](/workflows/live-trading)。
