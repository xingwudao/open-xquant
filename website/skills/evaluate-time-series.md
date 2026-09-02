---
title: 时间序列因子评估
description: 用命中率、盈亏比、decay 曲线、空仓行为和 tearsheet 评估择时或轮动因子。
outline: deep
---

# 时间序列因子评估

用命中率、盈亏比、decay 曲线、空仓行为和 tearsheet 评估择时或轮动因子。

## 适用场景

- 因子预测单资产或小规模轮动集合的时间方向。

## 输入

- factor series、价格数据、forward periods、输出目录和可用图表依赖。

## 输出

- 时间序列 tearsheet、命中率、盈亏比、decay 和空仓期解释。

## 约束

- 不只报告命中率，不跳过 T+1 对齐，也不把 tearsheet 图片当作审计。

## 关联工作流

- [AI 因子评估](/workflows/factor-evaluation)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/evaluate-time-series/SKILL.md)
