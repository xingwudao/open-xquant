# open-xquant × eQuant-Py 目标架构

## Overview

open-xquant 是 AI 原生的实用量化研究系统；
eQuant-Py 是其首个官方认证量化计算层。

两个仓库继续独立演进，open-xquant 保持对研究语义、因果执行、审计、
机器学习和 Agent 工作流的控制权。

## Learning Objectives

The viewer will understand:

1. 三个项目的职责边界。
2. 六层目标架构和正式执行路径。
3. 本地改动、删除内容和七个 PR 的迁移顺序。

---

## Section 1: 最终关系

**Key Concept**: open-xquant 负责研究真实性，eQuant-Py 负责计算能力。

**Content**:

- open-xquant 是 AI 原生的实用量化研究系统；
- eQuant-Py 是其首个官方认证量化计算层。
- eBacktestCraft 保持独立，但不成为 open-xquant 的正式执行引擎。
- 两个仓库可以独立发布和独立升级。

**Visual Element**:

- Type: 三方边界图。
- Subject: open-xquant 为主系统，eQuant-Py 为外部认证计算层，
  eBacktestCraft 为独立旁路项目。
- Treatment: open-xquant 使用主色，eQuant-Py 使用互补色，
  eBacktestCraft 使用灰色虚线并标注“不成为正式执行引擎”。

**Text Labels**:

- Headline: "方案 B：独立仓库，认证整合"
- Labels: "open-xquant", "eQuant-Py", "eBacktestCraft",
  "官方认证量化计算层", "正式执行引擎"

---

## Section 2: 六层目标架构

**Key Concept**: Agent 和模型都由确定性研究内核约束。

**Content**:

- Agent 层
- 研究治理层
- 编译层
- 官方量化计算层
- Engine 和交易执行层
- ML 研究层
- Agent 不能直接改变正式运行语义。
- 模型不能绕过训练边界、数据版本和推理时点。

**Visual Element**:

- Type: 六层系统剖面。
- Subject: 从 Researcher/Agent 到 Spec、Compiler、Certified Operators、
  Engine/Broker、Artifacts/Audit，以及横向 ML 研究层。
- Treatment: 使用清晰的垂直执行箭头；ML 层横跨数据、计算和 Engine；
  Audit/Robustness 形成反馈回路。

**Text Labels**:

- Headline: "六层目标架构"
- Labels: "Agent", "Research Governance", "Compiler",
  "Quant Compute Integration", "Engine & Execution", "ML Research"

---

## Section 3: 正式执行路径

**Key Concept**: Agent 只能通过受治理路径使用算子。

**Content**:

- Agent -> Catalog -> Spec -> Compile -> Certified Executor
- 它不等于“任意插件系统”。
- 它只执行通过认证的 operator binding。
- Strategy Spec 不包含 eQuant 实现细节。
- compiled plan 完整记录 operator binding。
- run artifacts 完整记录版本和摘要。

**Visual Element**:

- Type: 五步水平流水线。
- Subject: Catalog、Spec、Compile、Certified Executor、Run Artifacts。
- Treatment: 每一步通过锁定连接；任意函数动态导入以红色阻断符号表示。

**Text Labels**:

- Headline: "唯一正式路径"
- Labels: "Catalog", "Spec", "Compile", "Certified Executor",
  "Run Artifacts"

---

## Section 4: 谁拥有交易语义

**Key Concept**: eQuant-Py 不参与订单和成交语义。

**Content**:

- Universe。
- PortfolioOptimizer。
- ExposurePolicy 和 Rule。
- Order generation。
- Fee、tax 和 slippage。
- Broker lifecycle。
- Fill 和 Position。
- paper/live 一致性。
- eQuant-Py 不参与订单和成交语义。

**Visual Element**:

- Type: 责任边界清单。
- Subject: Engine 内部保留的交易语义，eQuant 仅提供计算输出。
- Treatment: Engine 侧使用盾牌边界；算子输出箭头止于指标或特征计算阶段。

**Text Labels**:

- Headline: "交易语义归 open-xquant"
- Labels: "Portfolio", "Policy & Rule", "Orders", "Broker",
  "Fees & Slippage", "Fill & Position"

---

## Section 5: 本地新增与修改

**Key Concept**: 新增受治理算子边界，只修改指标或特征计算阶段。

**Content**:

- contracts/quant-operators/
- src/oxq/operators/
- src/oxq/integrations/equant/
- tests/operators/
- tests/integrations/equant/
- 未来 ML 阶段新增：
- src/oxq/ml/
- tests/ml/
- 仅修改指标或特征计算阶段：
- 支持 compute_panel()。
- 批量调用 executor。
- 输出对齐和诊断。
- tracer 记录 operator provenance。
- 不重写逐 bar 交易阶段。
- 核心 dependencies 不加入任何 eQuant 包。

**Visual Element**:

- Type: 新增/修改双栏模块。
- Subject: 左侧显示新目录，右侧显示 Engine、registry、catalog、spec、compiler、
  validator、audit 和 artifacts 的受控修改。
- Treatment: 新增使用绿色加号，修改使用蓝色扳手。

**Text Labels**:

- Headline: "本地改动"
- Labels: "新增", "修改", "Operators", "eQuant Integration",
  "Compiler", "Audit", "Artifacts"

---

## Section 6: 必须删除或拒绝

**Key Concept**: 下一次 PR 应基于跨仓库契约重新组织提交。

**Content**:

- 删除当前通用动态适配器
- 删除当前数据 provider
- 删除当前 factor convenience wrapper
- 删除平行 calendar API
- 撤回直接指标替换
- 删除核心硬依赖
- 不合并无关兼容修改
- 重写 Agent 文档

**Visual Element**:

- Type: 红色删除清单。
- Subject: PR #61 中需要删除、拒绝或重写的八类内容。
- Treatment: 使用清晰删除线图标，不展示长文件路径。

**Text Labels**:

- Headline: "PR #61：删除、拒绝、重写"
- Labels: "动态适配器", "数据 Provider", "Factor Wrapper",
  "平行 Calendar", "直接替换", "核心硬依赖", "无关兼容", "Agent 文档"

---

## Section 7: 原生指标迁移状态

**Key Concept**: 不能一次删除所有重复实现。

**Content**:

- native-only
- shadow-equant
- equant-default
- native-removed
- operator runtime-certified。
- 数值差异被接受。
- 策略回归通过。
- 性能不退化。

**Visual Element**:

- Type: 四阶段迁移箭头。
- Subject: 原生实现从唯一实现到影子验证、默认切换和最终删除。
- Treatment: `shadow-equant` 阶段突出双轨对比；切换门前放置四个检查标识。

**Text Labels**:

- Headline: "四阶段指标迁移"
- Labels: "native-only", "shadow-equant", "equant-default",
  "native-removed"

---

## Section 8: 七个 PR 的整合顺序

**Key Concept**: 不再次提交同时修改依赖、指标、数据、因子和日历的大 PR。

**Content**:

- PR 1：契约和 manifest 基础
- PR 2：eQuant optional integration
- PR 3：首批时间序列指标
- PR 4：横截面和面板算子
- PR 5：拟合型因子工程
- PR 6：数据和日历适配
- PR 7：Agent catalog 和文档

**Visual Element**:

- Type: 七步路线。
- Subject: 从无 eQuant 调用的契约基础逐步推进到 Agent catalog 和文档。
- Treatment: 每一步使用短标签和编号，依赖顺序用单向箭头连接。

**Text Labels**:

- Headline: "下一次 PR：七步落地"
- Labels: "Contract", "Integration", "Time Series", "Panel",
  "Fit/Transform", "Data & Calendar", "Agent"

---

## Section 9: 长期能力

**Key Concept**: 这次整合建立三个长期能力。

**Content**:

- 可认证的量化算法资产供应链。
- 面向 Agent 和机器学习的结构化计算语义。
- 从研究到回测再到实盘的一致性和可追溯性。

**Visual Element**:

- Type: 底部三项结论带。
- Subject: 认证供应链、结构化语义、研究到实盘的连续链路。
- Treatment: 使用三个高对比图标和简短结论。

**Text Labels**:

- Headline: "三个长期能力"
- Labels: "认证供应链", "Agent + ML 语义", "Research-to-Live"

---

## Data Points (Verbatim)

### Quotes

- "open-xquant 是 AI 原生的实用量化研究系统；"
- "eQuant-Py 是其首个官方认证量化计算层。"
- "它只执行通过认证的 operator binding。"
- "eQuant-Py 不参与订单和成交语义。"
- "Agent 不能直接改变正式运行语义。"
- "模型不能绕过训练边界、数据版本和推理时点。"
- "两个仓库可以独立发布和独立升级。"

### Key Terms

- **Agent Layer**: 收集研究想法、查询 catalog、构建候选 Spec、解释证据。
- **Research Governance**: version、audit、artifact lineage、comparison、selection。
- **Quant Compute Integration**: QuantPanel、manifest、executor、diagnostics、provenance。
- **ML Research**: FeatureSet、LabelSpec、PIT、Purged CV、Walk-forward、FittedModel。

---

## Design Instructions

### Style Preferences

- 中文技术信息图。
- 简洁画布、留白充分，不使用复杂背景纹理。
- open-xquant、eQuant-Py、ML 和删除项使用稳定且可区分的颜色编码。
- 不在图中放置代码段或大段正文。

### Layout Preferences

- 横版 `16:9`。
- 中央展示六层目标架构和正式执行路径。
- 两侧展示职责边界与本地改动。
- 底部展示四阶段指标迁移、七个 PR 和三个长期能力。
- 采用 `bento-grid` 模块布局，保持从架构到落地的阅读顺序。

### Other Requirements

- 图内技术名称保持英文原文，解释性标签使用中文。
- 清晰区分“归 open-xquant 所有”“由 eQuant-Py 提供”和“必须删除”。
- eBacktestCraft 必须显示为独立项目，且不是正式执行引擎。
- 重点标签必须可读，避免将全部目录和验收条目塞入图片。
