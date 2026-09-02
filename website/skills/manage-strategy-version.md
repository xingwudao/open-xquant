---
title: 量化策略版本管理
description: 判断当前研究应继续现有版本、创建新策略版本，还是追加同一策略的运行尝试。
outline: deep
---

# 量化策略版本管理

判断当前研究应继续现有版本、创建新策略版本，还是追加同一策略的运行尝试。

## 适用场景

- 量化研究对话开始、恢复、语义发生变化，或阶段产物通过后需要更新版本治理状态。

## 输入

- 用户请求、`workflow_manifest.json`、`current.json`、`lineage.json`、版本清单和当前阶段产物。

## 输出

- 版本清单、阶段状态、`lineage.json`、`current.json` 和治理事务日志。

## 约束

- 不运行 `oxq`，不写策略 SPEC、审计或报告文件，也不选择最终版本。
- 新版本发布必须先创建受控阶段目录和匹配清单，最后发布 `current.json`。

## 关联工作流

- [AI Agent 量化研究](/guide/agentic-quant-research)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/manage-strategy-version/SKILL.md)
