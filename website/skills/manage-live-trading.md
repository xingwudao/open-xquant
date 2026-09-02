---
title: 量化实盘交易管理
description: 在严格安全门槛下连接 Alpaca 纸面或实盘交易，执行账户检查、行情读取或订单提交。
outline: deep
---

# 量化实盘交易管理

在严格安全门槛下连接 Alpaca 纸面或实盘交易，执行账户检查、行情读取或订单提交。

## 适用场景

- 用户明确要求 broker 连接、账户检查、实时数据、纸面交易、实盘交易或订单提交。

## 输入

- live 依赖、Alpaca 凭据、paper/live 模式、账户、行情、订单参数和风险确认。

## 输出

- 账户或持仓检查、行情样本、经确认的订单提交结果，或 LiveBroker 集成状态。

## 约束

- 默认 `paper=True`；实盘端点和任何 live 订单都需要用户明确要求并确认风险。
- 不在文件中保存 API keys，不批量下单除非用户审阅完整列表，不把回测信号直接变成订单。

## 关联工作流

- [AI 量化实盘交易](/workflows/live-trading)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/manage-live-trading/SKILL.md)
