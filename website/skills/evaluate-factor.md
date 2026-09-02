---
title: 量化因子评估
description: 根据因子问题类型路由到横截面或时间序列评估，并先确认定义、样本和对齐方式。
outline: deep
---

# 量化因子评估

根据因子问题类型路由到横截面或时间序列评估，并先确认定义、样本和对齐方式。

## 适用场景

- 用户询问一个因子是否预测未来收益，或需要判断应采用 IC 还是择时评估。

## 输入

- 因子定义、symbols、日期范围、forward return horizons、研究目标、数据源和缺失值处理。

## 输出

- 横截面或时间序列评估路由，以及必要的数据对齐和样本量前置要求。

## 约束

- 不在缺少未来收益对齐时评估因子，不只跑单一 horizon 支撑研究结论。
- 低样本量和高换手不能被隐藏。

## 关联工作流

- [AI 因子评估](/workflows/factor-evaluation)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/evaluate-factor/SKILL.md)
