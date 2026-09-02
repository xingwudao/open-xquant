---
title: 量化策略最终版本选择
description: 只从满足审计、报告、血缘和用户确认策略门槛的候选中发布最终研究候选。
outline: deep
---

# 量化策略最终版本选择

只从满足审计、报告、血缘和用户确认策略门槛的候选中发布最终研究候选。

## 适用场景

- 用户要求选择、确认、提升或标记最终 open-xquant 策略版本。

## 输入

- 候选版本和运行、报告与审阅修订、血缘审计、比较证据、选择策略和协调器确认事件。

## 输出

- 选择策略、候选集、比较引用账本、`final_decision.json` 和 `current_final.json`。

## 约束

- 不运行回测，不修改运行、报告、指标、审计或比较产物，也不把结果称为可投资或实盘就绪。
- 必须验证协调器拥有的确认日志事件；不能接受复制、陈旧、跨版本或路径推断的证据。

## 关联工作流

- [AI Agent 量化研究](/guide/agentic-quant-research)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/select-final-version/SKILL.md)
