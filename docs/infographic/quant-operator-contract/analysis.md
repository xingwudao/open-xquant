---
title: "Quant Operator Contract v1 设计规范"
topic: "技术架构与跨仓库契约"
data_type: "系统结构、分类体系、认证流程"
complexity: "complex"
point_count: 12
source_language: "zh"
user_language: "zh"
---

## Main Topic

本规范定义 open-xquant 与外部量化计算层之间的稳定契约。

它为经过认证的高价值量化计算能力提供一条受治理的接入路径，
并明确 open-xquant、算子提供方和跨仓库契约各自负责的内容。

## Learning Objectives

After viewing this infographic, the viewer should understand:

1. open-xquant、算子提供方与 Quant Operator Contract 的职责边界。
2. `QuantPanel`、`OperatorManifest`、`OperatorRequest` 和
   `OperatorResult` 如何形成稳定执行接口。
3. execution scope、lifecycle、causality 和 certification level
   如何共同限制算子进入研究或交易路径。

## Target Audience

- **Knowledge Level**: 熟悉 Python、量化研究或框架集成的中高级工程师。
- **Context**: 评审独立算子仓库是否能够长期、可认证地接入 open-xquant。
- **Expectations**: 快速理解职责边界、核心对象、风险门禁和发布认证流程。

## Content Type Analysis

- **Data Structure**: 中央契约连接两侧责任主体，四类稳定对象形成执行接口，
  多组枚举语义形成门禁，异步发布流程形成闭环。
- **Key Relationships**: `OperatorManifest` 描述语义，`OperatorRequest`
  携带输入和上下文，独立算子仓库执行计算，`OperatorResult` 返回标准输出，
  open-xquant 负责运行、审计和产物治理。
- **Visual Opportunities**: 用中央桥接结构表达跨仓库边界；用四块对象卡片表达
  稳定接口；用分层门禁表达 scope、lifecycle、causality 和认证等级；
  用闭环箭头表达异步发布。

## Key Data Points (Verbatim)

- "本规范定义 open-xquant 与外部量化计算层之间的稳定契约。"
- "为经过认证的高价值量化计算能力提供一条受治理的接入路径。"
- "open-xquant 负责研究真实性："
- "算子提供方负责计算实现质量："
- "本契约负责二者之间的语义一致性。"
- "QuantPanel：标准数据交换对象。"
- "OperatorManifest：机器可读的算子语义。"
- "OperatorRequest：标准执行请求。"
- "OperatorResult：标准执行结果和诊断。"
- "execution_scope MUST 是以下值之一："
- "time_series"
- "cross_section"
- "panel"
- "research_only"
- "只有 past_only 可以进入正式交易信号路径。"
- "两个仓库不需要同时发布。"
- "只有该等级可以进入交易信号路径。"

## Layout × Style Signals

- Content type: 系统结构与多模块概览 → suggests `bento-grid`。
- Tone: 技术规范、工程门禁、跨仓库协作 → suggests
  `technical-schematic`。
- Audience: 工程师和架构评审者 → suggests 高可读标签和明确连接线。
- Complexity: complex → suggests 分区明确、密度受控的 landscape 画布。
- Keyword shortcut: 用户要求“详细的信息图” → leading layout 为
  `bento-grid`，leading style 为 `craft-handmade`。

## Design Instructions (from user input)

- 偏好全部采用推荐值。
- 语言使用中文。
- 布局、风格和比例自动选择。
- 图片后端使用 `auto`，当前环境应解析为 Codex 原生 `imagegen`。
- 图中只保留关键术语和短句，避免将 956 行规范压缩为不可读小字。
- 不新增源文档没有定义的承诺、角色或兼容结论。

## Recommended Combinations

1. **bento-grid + craft-handmade** (Recommended): 符合“信息图”关键词
   快捷规则；横向分区可同时承载职责、对象、门禁和认证流程。
2. **bento-grid + technical-schematic**: 保留同样的信息结构，强化工程规范、
   数据契约和认证门禁的技术感。
3. **structural-breakdown + technical-schematic**: 适合突出契约桥接两侧仓库的
   结构关系，但对认证流程和枚举分类的承载空间较少。
