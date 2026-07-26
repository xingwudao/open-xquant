# Image Generation Prompt

Use case: infographic-diagram
Asset type: project architecture and migration documentation infographic
Primary request: Create a professional Chinese infographic explaining the target
architecture for integrating open-xquant with eQuant-Py under方案 B while the two
repositories remain independent.
Canvas: landscape 16:9, wide composition, publication-ready, high resolution.
Language: Simplified Chinese for explanatory labels; preserve English technical
terms and project names exactly as provided.

## Layout

Use a clean, information-dense bento-grid layout with a 12-column modular structure.

- A full-width title strip at the top.
- A large central hero cell for the six-layer target architecture.
- A medium upper-side cell for project responsibility boundaries.
- A medium side cell for the governed execution path.
- Two medium lower cells for local changes and rejected PR #61 content.
- A wide lower cell for the four-stage native-indicator migration.
- A full-width seven-step PR roadmap near the bottom.
- A concise conclusion strip with the three long-term capabilities.
- Clear reading order from architecture to implementation and migration.
- Mixed rectangular cell sizes, consistent padding, ample whitespace.

## Style

Strict craft-handmade paper-cutout infographic style.

- Light cream paper background, subtle natural paper grain.
- Layered construction-paper shapes with restrained soft shadows.
- Hand-drawn outlines with consistent line weight.
- Professional multi-color palette with stable role coding:
  - open-xquant: muted teal
  - eQuant-Py: mustard yellow
  - ML: forest green
  - delete or reject: brick red
  - neutral governance and artifacts: soft blue and charcoal
- No single-hue palette and no dominant purple, beige, brown, or dark-blue theme.
- Simple handmade engineering icons: researcher, Agent, document, compiler gear,
  catalog, lock, calculator, data panel, engine, broker, audit clipboard, model,
  package, delete mark, migration arrows, and release milestones.
- Friendly engineering-document tone, not childish.
- No realistic or photographic elements.
- No complex decorative background, gradients, bokeh, glowing effects, or 3D UI.

## Exact Content And Composition

### Title strip

Render exactly:

- Main title: "open-xquant × eQuant-Py 目标架构"
- Subtitle: "方案 B：独立仓库，认证整合"

### Responsibility boundary cell

Headline: "最终关系"

Show three project modules:

- "open-xquant"
- "AI 原生的实用量化研究系统"
- "eQuant-Py"
- "官方认证量化计算层"
- "eBacktestCraft"
- "独立，但不是正式执行引擎"

Connect eQuant-Py to open-xquant through a locked certified binding.
Place eBacktestCraft to the side with a gray dashed line, not inside the engine.

### Central six-layer architecture cell

Headline: "六层目标架构"

Show six clearly separated paper layers with the following exact labels:

- "Agent Layer"
- "Research Governance"
- "Compiler"
- "Quant Compute Integration"
- "Engine & Execution"
- "ML Research"

Inside or beside the layers, render these exact supporting labels:

- "Idea & Evidence"
- "Strategy Spec / Model Spec"
- "Compiled Plan"
- "Certified Operators"
- "Broker & Fill"
- "Run Artifacts"
- "Audit · Robustness · Comparison"

The main flow is:

Researcher and Agent -> Strategy Spec / Model Spec -> Compiler -> Compiled Plan ->
Quant Compute Integration -> Engine & Execution -> Broker & Fill and Run Artifacts.

Show eQuant-Py as an external yellow module connected only to
Quant Compute Integration through "Certified Operators".

Show ML Research as a green cross-cutting layer connected to Spec, Quant Compute
Integration, and Engine & Execution.

Add two small guardrail callouts:

- "Agent 不能直接改变正式运行语义"
- "模型不能绕过训练边界、数据版本和推理时点"

### Governed execution path cell

Headline: "唯一正式路径"

Show a five-step locked pipeline with these exact labels:

- "Catalog"
- "Spec"
- "Compile"
- "Certified Executor"
- "Run Artifacts"

Under the pipeline render exactly:

- "只执行通过认证的 operator binding"

Show a red blocked shortcut labeled "任意函数调用" that cannot bypass the pipeline.

### Trading semantics cell

Headline: "交易语义归 open-xquant"

Use a shielded group containing these exact labels:

- "Portfolio"
- "ExposurePolicy & Rule"
- "Orders"
- "Broker"
- "Fees & Slippage"
- "Fill & Position"

Add the exact statement:

- "eQuant-Py 不参与订单和成交语义"

### Local changes cell

Headline: "本地改动"

Split the cell into green add and blue modify areas.

Green area title: "新增"

- "contracts/quant-operators/"
- "src/oxq/operators/"
- "src/oxq/integrations/equant/"
- "src/oxq/ml/"

Blue area title: "修改"

- "Registry & Catalog"
- "Engine compute_panel()"
- "Spec & Compiler"
- "Audit & Artifacts"

Add a lock icon beside this exact statement:

- "核心 dependencies 不加入任何 eQuant 包"

### Rejected PR content cell

Headline: "PR #61：删除、拒绝、重写"

Use a restrained brick-red checklist with these exact labels:

- "动态适配器"
- "数据 Provider"
- "Factor Wrapper"
- "平行 Calendar"
- "直接指标替换"
- "核心硬依赖"
- "无关兼容修改"
- "Agent 直调文档"

### Native indicator migration cell

Headline: "四阶段指标迁移"

Show a left-to-right four-stage migration with exact labels:

- "native-only"
- "shadow-equant"
- "equant-default"
- "native-removed"

Above the transition into equant-default, show four small approval stamps:

- "runtime-certified"
- "数值差异接受"
- "策略回归通过"
- "性能不退化"

Emphasize shadow-equant as a dual-track comparison stage.

### Seven-PR roadmap cell

Headline: "下一次 PR：七步落地"

Show seven numbered milestones with these exact labels:

- "1 契约和 Manifest"
- "2 Optional Integration"
- "3 时间序列指标"
- "4 横截面和 Panel"
- "5 Fit / Transform"
- "6 数据和日历"
- "7 Agent Catalog"

Use one-way arrows and show that each milestone builds on the previous one.

### Bottom conclusion strip

Headline: "三个长期能力"

Render exactly:

- "可认证的量化算法资产供应链"
- "面向 Agent 和机器学习的结构化计算语义"
- "从研究到回测再到实盘的一致性和可追溯性"

## Typography

- Main title is the largest text and fully readable.
- Cell headlines are bold, dark charcoal, and consistent.
- English technical terms use a clean monospaced or technical sans-serif style.
- Chinese labels use a clear hand-lettered sans-serif style, not cursive calligraphy.
- Keep all labels horizontal.
- Preserve every supplied label verbatim.
- Do not invent extra headings, captions, placeholder text, or pseudo-characters.
- Do not render code blocks.

## Constraints

- open-xquant must be visually presented as the owner of research truth,
  governance, compiler, engine, execution, audit, ML boundaries, and Agent workflow.
- eQuant-Py must remain an external independently released certified compute layer.
- eBacktestCraft must remain separate and must not appear as the formal engine.
- Do not present this as an arbitrary plugin platform.
- All important text must remain legible at normal document width.
- Use only the exact labels listed above.
- No logos, watermarks, signatures, QR codes, or unrelated illustrations.
