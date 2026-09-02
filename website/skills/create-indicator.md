---
title: 量化指标创建
description: 为 open-xquant 创建纯数值时间序列 Indicator，并补齐测试、导出和注册表接线。
outline: deep
---

# 量化指标创建

为 open-xquant 创建纯数值时间序列 Indicator，并补齐测试、导出和注册表接线。

## 适用场景

- `create-component` 已确认没有现有 Indicator 满足用户请求。

## 输入

- 公式、输入列、参数默认值、输出单位和符号、NaN 行为、常价行为和依赖要求。

## 输出

- `src/oxq/indicators/{snake_name}.py`、对应测试、包导出和注册表更新。

## 约束

- 先写手算期望的测试并确认缺失实现失败；测试通过前不注册。
- 不从实现复制期望值，不加入 I/O、随机性或可变全局状态。

## 关联工作流

- [AI 量化组件开发](/workflows/component-development)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/create-indicator/SKILL.md)
