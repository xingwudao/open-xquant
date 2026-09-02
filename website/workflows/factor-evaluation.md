---
title: AI 因子评估 | open-xquant
description: 汇总 open-xquant 因子研究中的横截面、时间序列、筛选和数据对齐评估入口。
---

# AI 因子评估

open-xquant 的 AI 因子评估先确认因子定义、样本、forward return
对齐和评估目标，再选择横截面或时间序列路径。它不是用单次相关性或命中率替代研究结论，而是把样本量、换手、decay、缺失值和数据来源写入可复核证据。

## 评估入口

- [AI 因子研究](/workflows/factor-research)：理解因子研究的整体工作流和边界。
- [量化因子评估](/skills/evaluate-factor)：判断因子应进入横截面还是时间序列评估。
- [横截面因子评估](/skills/evaluate-cross-sectional)：评估多资产同日排序因子。
- [时间序列因子评估](/skills/evaluate-time-series)：评估择时或轮动因子。
- [量化因子筛选](/skills/screen-factors)：生成候选列表但不替代回测验证。

## 约束

因子评估必须保留数据窗口、symbols、horizon、对齐方式和缺失值处理记录。缺少未来收益对齐、样本量不足或只覆盖单一 horizon 时，结果只能作为探索信号，不能作为最终策略证据。
