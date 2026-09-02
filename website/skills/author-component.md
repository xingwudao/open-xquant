---
title: 量化组件实现
description: 在工作区本地实现自定义组件、测试、清理缓存并发布带哈希的组件清单和目录。
outline: deep
---

# 量化组件实现

在工作区本地实现自定义组件、测试、清理缓存并发布带哈希的组件清单和目录。

## 适用场景

- 组件目录和配方目录无法满足研究需要，需要工作区本地自定义 Indicator、Signal 或 PortfolioOptimizer。

## 输入

- `component_request.json`、`component_catalog.json`、对话上下文、确认记录和组件根路径。

## 输出

- 工作区组件 bundle、`component_manifest.json`、`component_catalog.json`、测试结果和阶段 `result.json`。

## 约束

- 不修改安装的 SDK bundle 或 open-xquant 源码，除非用户明确要求框架开发。
- 工作区本地 Rule 默认阻塞；不能跳过测试、导入、注册可见性或 manifest hash 失败。

## 关联工作流

- [AI 量化组件开发](/workflows/component-development)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/author-component/SKILL.md)
