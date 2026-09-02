---
title: 量化组合优化器创建
description: 创建返回目标权重的 PortfolioOptimizer，并用权重和 fallback 不变量测试验证。
outline: deep
---

# 量化组合优化器创建

创建返回目标权重的 PortfolioOptimizer，并用权重和 fallback 不变量测试验证。

## 适用场景

- `create-component` 已确认没有现有 PortfolioOptimizer 满足分配逻辑。

## 输入

- 分配公式、构造参数、signals 或 indicators 输入、所需列、CASH fallback、权重限制和状态行为。

## 输出

- `src/oxq/portfolio/{snake_name}.py` 或维护者要求的模块、对应测试、导出和注册表更新。

## 约束

- 每条路径必须返回非空权重字典且权重和为 `1.0`。
- 无效输入要回落到 `{"CASH": 1.0}` 或明确排除；测试通过前不注册。

## 关联工作流

- [AI 量化组件开发](/workflows/component-development)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/create-portfolio-optimizer/SKILL.md)
