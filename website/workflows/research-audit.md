---
title: 量化回测审计 | open-xquant
description: 介绍 open-xquant 回测审计如何覆盖想法、规格、运行语义、偏差、报告资格和产物血缘。
---

# 量化回测审计

量化回测审计的目标是回答“这个结果从哪里来、是否按确认的规则运行、证据是否仍然有效”。open-xquant 把审计拆在研究前、中、后多个阶段，而不是在报告末尾补一句风险提示。

## 想法审计

想法审计检查策略想法是否来自用户确认的描述，是否按阶段解释、询问、拉回和确认。未确认的假设不能进入 `strategy_spec.yaml`。

## 规格审计

规格审计检查 spec 字段来源、默认值、组件选择和用户确认状态。相关 Skill 包括 [量化策略规格审计](/skills/audit-strategy-spec) 与 [量化策略规格构建](/skills/build-strategy-spec)。

## 运行语义审计

运行语义审计编译 `compiled_plan.json` 并检查执行语义是否保留 spec 的关键含义，例如信号时间、成交时间、价格选择、成本、Universe 和 Rule。通过后才可能进入授权回测。

## 偏差审计

跑后审计检查可复现性和研究偏差，包括数据、时间对齐、选择性记忆和过拟合线索。它不会把失败结论修饰成可交易结论，而是把 fatal、warning 和 evidence 明确写入产物。

## 产物血缘

[量化产物血缘审计](/skills/audit-artifact-lineage) 重新计算 version、run、report、comparison 和 final selection 之间的路径与 hash 绑定。最终候选只有在 spec audit、runtime audit、reproducibility audit、research audit、report review 和 lineage 都满足资格时才可被考虑。
