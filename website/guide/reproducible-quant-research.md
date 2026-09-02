---
title: 可复现量化研究 | open-xquant
description: 介绍 open-xquant 的可复现研究治理，覆盖版本、运行、产物血缘、确认和报告证据。
---

# 可复现量化研究

可复现量化研究要求同一 spec、同一数据和同一执行假设能够得到可验证的结果，并能说明每个结论来自哪个版本、哪个 run、哪些输入和哪些报告证据。open-xquant 用 strategy family -> strategy version -> run attempt 的目录模型治理这条证据链。

## 适合谁

- 不想让 `strategy_spec.yaml`、审计 JSON、回测结果和报告混在根目录的研究者。
- 需要反复修改策略，并保留旧版本证据链的团队。
- 希望在最终选择前检查报告、审计、鲁棒性和 lineage 的用户。

## 解决什么问题

一次漂亮的回测曲线如果缺少 spec hash、数据 manifest、交易成本、目标权重、审计结果和报告绑定，就很难复查。open-xquant 的治理文档要求根目录保留索引、实验登记、对比和最终选择；每个 version 内按 brainstorm、idea audit、spec build、data inspection、spec audit、compile preview、runtime audit、backtests 和 reports 分区。

这让 [量化产物血缘审计](/skills/audit-artifact-lineage) 可以检查 run、report、comparison 和 final selection 是否引用了当前有效证据。

## 工作边界

可复现不等于可盈利。它只说明研究过程可复查、输入输出可绑定、旧证据不会被覆盖。样本外失败、成本敏感或偏差审计失败，仍然是需要正视的研究结论。

open-xquant 也不鼓励复用旧选择结果来修补历史证据；历史刷新需要新报告、新审稿、新 lineage、新 selection 和新 comparison scope。

## 下一步

先看 [量化回测审计](/workflows/research-audit) 了解审计层级，再看 [量化策略稳健性检验](/workflows/robustness-testing) 理解压力测试。想快速感受产物结构，可以运行 [量化研究示例](/examples/) 中的回测与审计示例。
