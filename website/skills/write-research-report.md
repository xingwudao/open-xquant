---
title: 量化研究报告撰写
description: 基于指标、审计、稳健性和已登记图表资产撰写最终人类可读研究报告并渲染 HTML。
outline: deep
---

# 量化研究报告撰写

基于指标、审计、稳健性和已登记图表资产撰写最终人类可读研究报告并渲染 HTML。

## 适用场景

- 需要写入或编辑 `research_report.md`、`research_report.html` 或最终实验报告。

## 输入

- 策略 SPEC、审计、metrics、执行假设、compiled plan、data manifest、监控产物、报告资产和 facts API 输出。

## 输出

- revision-scoped `research_report.md`、`research_report.html`、`writer_result.json` 和 sealed candidate manifest。

## 约束

- 不发明证据，不修改指标、审计、稳健性或回测产物，不隐藏负面证据。
- 默认语言为中文；没有已登记的必需图表资产时写 blocked 结果并交给 `build-report-charts`。

## 关联工作流

- [AI 量化回测](/workflows/strategy-backtest)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/write-research-report/SKILL.md)
