# Quant Operator Contract v1

## Overview

本规范定义 open-xquant 与外部量化计算层之间的稳定契约。

为经过认证的高价值量化计算能力提供一条受治理的接入路径。

## Learning Objectives

The viewer will understand:

1. open-xquant、算子提供方和契约的职责边界。
2. 四类稳定对象如何形成标准执行路径。
3. 算子如何经过语义门禁和认证流程进入研究或交易路径。

---

## Section 1: 三方职责边界

**Key Concept**: 本契约负责二者之间的语义一致性。

**Content**:

- open-xquant 负责研究真实性：
- 策略语义。
- 因果执行。
- 交易时点。
- 回测、审计、稳健性和实验比较。
- 算子提供方负责计算实现质量：
- 数学公式和算法实现。
- 输入输出契约。
- 数值精度。
- 性能和资源消耗。

**Visual Element**:

- Type: 中央桥接架构图。
- Subject: 左侧为 open-xquant，右侧为独立算子仓库，中间为契约桥。
- Treatment: 两侧使用不同颜色，中央契约使用高对比描边；箭头双向连接。

**Text Labels**:

- Headline: "职责边界"
- Left: "open-xquant"
- Center: "Quant Operator Contract"
- Right: "独立算子仓库"
- Labels: "研究真实性", "语义一致性", "计算实现质量"

---

## Section 2: 四类稳定对象

**Key Concept**: 两者通过以下四类稳定对象协作。

**Content**:

- QuantPanel：标准数据交换对象。
- OperatorManifest：机器可读的算子语义。
- OperatorRequest：标准执行请求。
- OperatorResult：标准执行结果和诊断。

**Visual Element**:

- Type: 四块相连的对象模块。
- Subject: 数据面板、语义清单、执行请求、结果与诊断。
- Treatment: 从左到右形成执行链，`OperatorManifest` 位于链路上方提供约束。

**Text Labels**:

- Headline: "四类稳定对象"
- Labels: "QuantPanel", "OperatorManifest", "OperatorRequest",
  "OperatorResult"

---

## Section 3: 执行语义门禁

**Key Concept**: manifest 必须显式声明执行范围、生命周期和因果性。

**Content**:

- execution_scope MUST 是以下值之一：
- time_series
- cross_section
- panel
- research_only
- lifecycle MUST 是以下值之一：
- stateless
- fit_transform
- evaluation
- data_access
- visualization
- causality MUST 是以下值之一：
- past_only
- label_dependent
- future_using
- 只有 past_only 可以进入正式交易信号路径。

**Visual Element**:

- Type: 三列筛选门禁。
- Subject: Scope、Lifecycle、Causality 三组枚举。
- Treatment: `past_only` 以绿色通行标识连接交易信号，
  `label_dependent` 和 `future_using` 进入隔离研究区域。

**Text Labels**:

- Headline: "执行语义门禁"
- Labels: "Scope", "Lifecycle", "Causality", "交易信号路径",
  "隔离研究"

---

## Section 4: 四级认证

**Key Concept**: 认证等级决定算子能够进入的路径。

**Content**:

- contract-valid
- research-certified
- runtime-certified
- ml-certified
- 只有该等级可以进入交易信号路径。
- fit/transform 分离。
- 训练边界可验证。
- fitted state 可序列化。
- 推理不会隐式拟合。

**Visual Element**:

- Type: 四级阶梯或四层通行证。
- Subject: 从基础契约合法到研究、运行时和机器学习认证。
- Treatment: `runtime-certified` 连接正式交易信号；`ml-certified`
  连接训练与推理边界。

**Text Labels**:

- Headline: "认证等级"
- Labels: "contract-valid", "research-certified", "runtime-certified",
  "ml-certified"

---

## Section 5: 异步发布闭环

**Key Concept**: 两个仓库不需要同时发布。

**Content**:

- 算子仓库开发并发布 release candidate。
- 提供方完成单元测试和 contract test。
- open-xquant 对指定版本执行认证。
- 认证通过后更新兼容矩阵。
- open-xquant 在后续版本中启用绑定。
- 两个仓库不需要同时发布。

**Visual Element**:

- Type: 五步循环流程。
- Subject: 开发、测试、候选发布、认证、兼容绑定。
- Treatment: 认证失败以支路返回开发；认证通过进入兼容矩阵。

**Text Labels**:

- Headline: "异步发布"
- Labels: "开发", "Contract Test", "Release Candidate",
  "open-xquant Certification", "Certified Binding"

---

## Section 6: 最终原则

**Key Concept**: 两个仓库独立演进，通过认证绑定协作。

**Content**:

- 算子仓库不接管 open-xquant 的工作流。
- open-xquant 也不复制算子仓库的所有内部模块。
- 未认证能力不能进入正式运行路径。

**Visual Element**:

- Type: 底部结论带。
- Subject: 两个独立仓库通过受治理桥梁协作。
- Treatment: 使用短句和锁定图标作为全图结论。

**Text Labels**:

- Headline: "独立演进，认证协作"
- Labels: "独立仓库", "稳定契约", "受治理接入"

---

## Data Points (Verbatim)

### Quotes

- "为经过认证的高价值量化计算能力提供一条受治理的接入路径。"
- "本契约负责二者之间的语义一致性。"
- "只有 past_only 可以进入正式交易信号路径。"
- "两个仓库不需要同时发布。"

### Key Terms

- **QuantPanel**: 标准数据交换对象。
- **OperatorManifest**: 机器可读的算子语义。
- **OperatorRequest**: 标准执行请求。
- **OperatorResult**: 标准执行结果和诊断。

---

## Design Instructions

### Style Preferences

- 中文技术信息图。
- 简洁画布、留白充分，不使用复杂背景纹理。
- 主要使用清晰模块、连线、锁、清单、数据面板和认证徽章等视觉元素。
- 不在图中放置代码段、长句或大段正文。

### Layout Preferences

- 横版 `16:9`。
- 顶部为标题和定位，中部为三方边界与四类对象，
  下部为语义门禁、认证等级和异步发布。
- 采用 `bento-grid` 模块布局，保持明确阅读顺序。

### Other Requirements

- 图内技术名称保持英文原文，解释性标签使用中文。
- 不新增源文档没有定义的对象和结论。
- 重点标签必须可读，避免小字号密集列表。
