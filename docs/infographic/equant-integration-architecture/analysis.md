---
title: "open-xquant 与 equant-py 整合架构设计"
topic: "AI 原生量化研究系统与官方量化计算层整合"
data_type: "分层架构、职责边界、迁移路线"
complexity: "complex"
point_count: 14
source_language: "zh"
user_language: "zh"
---

## Main Topic

open-xquant 是 AI 原生的实用量化研究系统；
equant-py 是其首个官方认证量化计算层。

该方案在两个仓库继续独立演进的前提下，保持 open-xquant 对研究语义、
因果执行、审计、机器学习和 Agent 工作流的控制权。

## Learning Objectives

After viewing this infographic, the viewer should understand:

1. open-xquant、equant-py 与 eBacktestCraft 的最终职责边界。
2. Agent、研究治理、编译、量化计算、Engine/执行和 ML 六层如何连接。
3. 本地需要新增、修改、删除什么，以及下一次 PR 如何分阶段整合。

## Target Audience

- **Knowledge Level**: 熟悉 open-xquant 代码结构、量化系统或跨仓库集成的
  中高级工程师和维护者。
- **Context**: 在 equant-py 完整满足 Quant Operator Contract v1 后，
  评审下一次集成 PR 的架构与范围。
- **Expectations**: 快速看清控制权、执行路径、代码改动、删减项和迁移顺序。

## Content Type Analysis

- **Data Structure**: 六层纵向架构连接 Agent 到执行与审计；equant-py 作为
  外部计算层通过官方集成边界横向接入；底部用新增、修改、删除和 PR 序列
  表达落地路线。
- **Key Relationships**: Strategy Spec 和 Model Spec 经过 Compiler 与 Runtime
  Audit 生成 Compiled Plan；Certified Operator Binding 调用 eQuant；
  open-xquant Engine 继续拥有组合、订单、Broker 和成交语义；运行产物回流到
  Audit、Robustness、Comparison 和 Final Selection。
- **Visual Opportunities**: 用六层剖面表达职责；用外接计算模块表达仓库独立；
  用锁、清单和摘要图标表达 operator catalog/lock/artifacts；用七步路线表达
  下一次 PR 的合理范围。

## Key Data Points (Verbatim)

- "open-xquant 是 AI 原生的实用量化研究系统；"
- "equant-py 是其首个官方认证量化计算层。"
- "eBacktestCraft 保持独立，但不成为 open-xquant 的正式执行引擎。"
- "它不等于“任意插件系统”。"
- "它只执行通过认证的 operator binding。"
- "equant-py 不参与订单和成交语义。"
- "Agent 不能直接改变正式运行语义。"
- "模型不能绕过训练边界、数据版本和推理时点。"
- "Agent -> Catalog -> Spec -> Compile -> Certified Executor"
- "核心 dependencies 不加入任何 eQuant 包。"
- "不能一次删除所有重复实现。"
- "native-only"
- "shadow-equant"
- "equant-default"
- "native-removed"
- "两个仓库可以独立发布和独立升级。"

## Layout × Style Signals

- Content type: 分层系统架构与迁移概览 → suggests `bento-grid` 或
  `structural-breakdown`。
- Tone: 工程架构、控制权、执行治理 → suggests `technical-schematic`。
- Audience: 仓库维护者和 PR 评审者 → suggests 明确的层级、边界和状态色。
- Complexity: complex → suggests landscape 画布与模块化密度。
- Keyword shortcut: 用户要求“详细的信息图” → leading layout 为
  `bento-grid`，leading style 为 `craft-handmade`。

## Design Instructions (from user input)

- 偏好全部采用推荐值。
- 语言使用中文。
- 布局、风格和比例自动选择。
- 图片后端使用 `auto`，当前环境应解析为 Codex 原生 `imagegen`。
- 图中需要同时表达未来 AI、Machine Learning 和 Agent 的位置。
- 图中要区分保留、增加、修改、删除和分阶段迁移，不能只画抽象架构。

## Recommended Combinations

1. **bento-grid + craft-handmade** (Recommended): 符合“信息图”关键词
   快捷规则；能够并列展示六层架构、边界、代码改动和七步 PR 路线。
2. **bento-grid + technical-schematic**: 保留全景信息密度，使用工程图视觉
   强化执行链、控制权和跨仓库接口。
3. **structural-breakdown + technical-schematic**: 最适合表现六层系统剖面和
   eQuant 外接模块，但需要压缩本地改动与 PR 路线信息。
