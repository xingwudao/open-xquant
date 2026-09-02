---
title: 量化组件创建
description: 在确认需要修改 open-xquant 框架内置组件时，检查注册表并路由到具体组件创建 Skill。
outline: deep
---

# 量化组件创建

在确认需要修改 open-xquant 框架内置组件时，检查注册表并路由到具体组件创建 Skill。

## 适用场景

- 用户明确要求修改 open-xquant 源码或新增内置 Indicator、Signal、Rule 或 PortfolioOptimizer。

## 输入

- 组件行为描述、期望名称、用户约束和当前组件注册表。

## 输出

- 一个具体组件类型路由，或发现现有组件已满足需求后的停止说明。

## 约束

- 不在导航 Skill 中写组件代码，不跳过注册表检查，不因同义词差异创建新组件。
- 请求可能属于多种组件类型时必须澄清，不能同时路由多个创建 Skill。

## 关联工作流

- [AI 量化组件开发](/workflows/component-development)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/create-component/SKILL.md)
