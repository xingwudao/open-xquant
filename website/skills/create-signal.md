---
title: 量化信号创建
description: 创建确定性的 boolean 或 categorical 交易意图 Signal，并验证输出域和因果行为。
outline: deep
---

# 量化信号创建

创建确定性的 boolean 或 categorical 交易意图 Signal，并验证输出域和因果行为。

## 适用场景

- `create-component` 已确认没有现有 Signal 满足请求的交易意图逻辑。

## 输入

- 触发条件、参数和默认值、输出类型、每个输出值含义、NaN 与边界行为、因果性说明。

## 输出

- `src/oxq/signals/{snake_name}.py`、对应测试、包导出和注册表更新。

## 约束

- 不返回未声明的浮点信号，不引入未说明的未来数据偏差。
- categorical trading intent 必须使用如 `BUY`、`SELL`、`HOLD` 的大写标签，测试通过前不注册。

## 关联工作流

- [AI 量化组件开发](/workflows/component-development)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/create-signal/SKILL.md)
