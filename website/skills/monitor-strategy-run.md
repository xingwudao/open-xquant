---
title: 量化策略运行监控
description: 对已完成运行执行可复现性、研究偏差、稳健性和实验登记，并保持运行包完整性。
outline: deep
---

# 量化策略运行监控

对已完成运行执行可复现性、研究偏差、稳健性和实验登记，并保持运行包完整性。

## 适用场景

- 正式回测完成后，需要发布 post-run 审计、稳健性结果、实验日志并移交报告写作。

## 输入

- 受控运行目录、canonical SPEC 审计、runtime audit、运行标准产物、实验登记路径和 robustness 输出。

## 输出

- 运行包内的 `reproducibility_audit.json`、`research_bias_audit.json`、`robustness.json`、实验登记和监控结果。

## 约束

- 不编辑产物来让审计通过，不把失败审计概括为“基本可以”，不只登记成功实验。
- 监控发布顺序固定；不得用 shell 重定向生成 governed audit 产物。

## 关联工作流

- [AI 量化回测](/workflows/strategy-backtest)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/monitor-strategy-run/SKILL.md)
