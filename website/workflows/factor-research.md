---
title: AI 因子研究 | open-xquant
description: 说明 open-xquant AI 因子研究如何确认假设、数据要求、横截面和时间序列评估边界。
---

# AI 因子研究

open-xquant 的因子研究先确认因子定义、symbols、日期范围、forward return horizon、问题类型、数据源和缺失值处理，再决定进入横截面评估、时间序列评估或候选筛选。

## 因子假设

因子假设必须说明它预测什么：同日多资产排序、单资产方向择时，还是多因子候选列表。只给出一个公式但没有 forward return 对齐方式，不能构成可审计研究问题。

## 数据要求

[量化因子评估](/skills/evaluate-factor) 要求 factor values 与 forward returns 对齐索引，不能让同日执行泄漏进未来收益。正式报告需要说明 horizon、日期对齐和排除行。财务因子还要检查字段来源，不能假设 OHLCV 一定能计算所有财务指标。

## 横截面评估

当目标是在同一日期对多资产排序时，使用 IC、Rank IC、ICIR、decay 和 turnover 更合适。样本过少时不应把横截面 IC 当成主要证据；symbols 数量不足时需要谨慎解释。

## 时间序列评估

当问题是一只资产或小型轮动集合的方向择时时，命中率、盈亏比、decay curve 和 tearsheet 往往比横截面 IC 更直接。择时结论仍需要多个 horizon 和样本区间支持。

## 筛选与调优

[量化因子筛选](/skills/screen-factors) 可以基于价格、财务或自定义因子生成候选列表，但筛选不是回测。候选列表进入策略研究前，还要经过 [AI 量化回测](/workflows/strategy-backtest)、偏差审计和稳健性检验。
