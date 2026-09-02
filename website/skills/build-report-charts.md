---
title: 量化报告图表构建
description: 为研究报告构建默认专业图表包，登记图形、脚本和 manifest 哈希后交给报告写作。
outline: deep
---

# 量化报告图表构建

为研究报告构建默认专业图表包，登记图形、脚本和 manifest 哈希后交给报告写作。

## 适用场景

- 报告需要 chart assets、figures、visual evidence、绘图脚本或 notebook-like 资产。

## 输入

- 运行包、报告修订 ID、图表决策、metrics、曲线、交易、订单、持仓、target weights 和稳健性产物。

## 输出

- revision-scoped `report_assets/manifest.json`、figures、scripts 和 `chart_build_result.json`。

## 约束

- 不发明图表数据，不编辑回测产物，不把图表当作盈利证明。
- 默认报告模式自动构建默认专业图表包；缺失单张图只能用封闭 skip reason 记录，不能让整份报告无图。

## 关联工作流

- [AI 量化回测](/workflows/strategy-backtest)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/build-report-charts/SKILL.md)
