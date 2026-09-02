---
title: 量化实验结果比较
description: 在非版本化 legacy 工作区中比较两个已完成实验运行的 SPEC、审计和指标差异。
outline: deep
---

# 量化实验结果比较

在非版本化 legacy 工作区中比较两个已完成实验运行的 SPEC、审计和指标差异。

## 适用场景

- 用户在 legacy 工作区要求比较两个已完成 open-xquant 实验运行。

## 输入

- 两个运行目录中的策略 SPEC、指标、净值曲线、执行假设、研究偏差审计和可复现性审计。

## 输出

- `<comparisons_dir>/<comparison_id>/` 下的 SPEC 差异、指标比较、比较报告、图形和登记摘要。

## 约束

- 不在版本化工作区使用；版本化工作区必须路由到 `compare-strategy-versions`。
- 不修改任一运行产物，不把未审计假设当作可比，也不脱离上下文命名单一赢家。

## 关联工作流

- [AI Agent 量化研究](/guide/agentic-quant-research)
- [AI 量化回测](/workflows/strategy-backtest)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/compare-experiments/SKILL.md)
