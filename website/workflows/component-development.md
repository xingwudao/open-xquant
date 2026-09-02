---
title: AI 量化组件开发 | open-xquant
description: 说明 open-xquant 指标、信号、规则和组合优化器组件开发时的路由、测试和注册边界。
---

# AI 量化组件开发

open-xquant 的组件开发用于在现有组件无法满足研究需要时，补充确定性的 Indicator、Signal、Rule 或 PortfolioOptimizer。组件开发必须先检查注册表和现有能力，再进入具体创建或工作区本地实现路径。

## 组件入口

- [量化组件创建](/skills/create-component)：确认是否需要新增框架内置组件。
- [量化组件实现](/skills/author-component)：实现工作区本地组件并生成 manifest。
- [量化指标创建](/skills/create-indicator)：创建纯数值时间序列指标。
- [量化信号创建](/skills/create-signal)：创建确定性的交易意图信号。
- [量化规则创建](/skills/create-rule)：创建 bar-by-bar 风控、持有或退出规则。
- [量化组合优化器创建](/skills/create-portfolio-optimizer)：创建目标权重生成组件。

## 约束

新增组件必须先有手算期望测试，测试通过前不注册。工作区本地组件不能修改安装的 SDK bundle；框架内置组件变更必须明确用户要求、注册表影响和回测语义边界。
