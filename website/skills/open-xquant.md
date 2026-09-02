---
title: open-xquant AI 量化研究路由
description: 根据量化研究请求选择正确的工作流，并在运行确定性工具前完成边界检查。
outline: deep
---

# open-xquant AI 量化研究路由

根据量化研究请求选择正确的工作流，并在运行确定性工具前完成边界检查。

## 适用场景

- 用户提出策略、回测、因子、审计、稳健性、报告或实盘研究请求。

## 输入

- 用户目标、研究目录和当前版本状态。

## 输出

- 明确的下一阶段 Skill 路由和前置条件。

## 约束

- 路由完成前不运行 CLI、SDK 或报告脚本。

## 关联工作流

- [AI Agent 量化研究](/guide/agentic-quant-research)
- [AI 量化回测](/workflows/strategy-backtest)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/open-xquant/SKILL.md)
