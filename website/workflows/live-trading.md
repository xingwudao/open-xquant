---
title: AI 量化实盘交易 | open-xquant
description: 说明 open-xquant 实盘交易能力的研究边界、认证要求、Broker 配置、监控和停止条件。
---

# AI 量化实盘交易

open-xquant 支持连接 Alpaca paper 或 live trading，但实盘页面必须先说明边界：回测研究、paper trading 和 live order submission 是不同风险层级，不能把回测信号直接转换为无人值守实盘订单。

## 研究与实盘边界

研究阶段关注 spec、数据、回测、审计、报告和版本选择。实盘连接只在用户明确要求 broker connectivity、账户检查、实时数据或订单提交时进入 [实盘交易管理](/skills/manage-live-trading)。

## 认证要求

实盘依赖需要额外安装，凭据通过环境变量提供。默认 `paper=True`；连接 live endpoint 前必须由用户明确请求 live trading 并确认风险。API key 不应写入文件。

## Broker 配置

交易执行假设由 [交易执行配置](/skills/configure-trade-execution) 管理，包括 signal time、trade time、fill price mode、order timing、fee、slippage、lot size、cash return 和 rebalance schedule。审计研究中不允许 same-bar close signal 与 close fill。

## 监控

实盘前应先检查账户、positions、market data 和 open orders。使用 `LiveBroker` 时会启动 trade-update stream，完成后需要关闭连接。任何订单提交前都要展示 paper/live mode、symbol、side、quantity、order type、time in force 和 estimated risk。

## 停止条件

停止条件包括用户未确认 live 风险、订单列表未完整审阅、凭据缺失、账户状态异常、成本或 lot size 未确认、策略审计证据不足，或运行结果与研究假设不一致。open-xquant 不提供无人值守收益承诺。
