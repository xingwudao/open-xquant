---
title: 量化交易执行配置
description: 配置信号时间、成交时间、价格模式、费用、滑点、手数、现金收益和交易日历假设。
outline: deep
---

# 量化交易执行配置

配置信号时间、成交时间、价格模式、费用、滑点、手数、现金收益和交易日历假设。

## 适用场景

- 用户讨论交易成本、订单执行、成交价格、手数、手续费、滑点或 broker 假设。

## 输入

- 信号时间、交易时间、fill price mode、order timing、费用税率、滑点、lot size、现金和日历。

## 输出

- 可用于受审计 CLI 回测的 execution/cost 配置或 SDK broker 配置。

## 约束

- 不允许同 bar close 信号和 close 成交进入受审计研究。
- 不在用户未确认时静默选择零费用或零滑点；比较不同执行、成本、日历、手数或现金假设时必须突出差异。

## 关联工作流

- [AI 量化回测](/workflows/strategy-backtest)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/configure-trade-execution/SKILL.md)
