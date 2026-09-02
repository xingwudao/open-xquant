# open-xquant AI 量化 SEO 优化设计

## 1. 背景

open-xquant 当前在 GitHub 上公开可见，但围绕中文查询
`AI 量化`、`AI 量化框架` 和 `AI Agent 量化研究` 的搜索信号较弱。

现状检查结果：

- GitHub 仓库的 `description` 为空。
- GitHub 仓库的 `homepage` 为空。
- GitHub 仓库没有 topics。
- README 已经中文前置，但首句没有直接使用 `AI 量化框架`。
- `pyproject.toml` 的描述和关键词主要是英文，且关键词较宽泛。
- 仓库没有独立文档站、站点地图或搜索引擎验证配置。
- `agent/skills/` 当前包含 35 个 `SKILL.md`，不是 14 个。
- “70 个工具”尚无稳定、可自动验证的唯一统计口径。

本设计不承诺特定 Google 排名。目标是建立准确、可抓取、可持续维护的
中文主题信号，并用搜索数据验证效果。

## 2. 目标与非目标

### 2.1 目标

1. 让搜索引擎和首次访问者在首屏明确识别 open-xquant 是
   `AI 量化研究框架`。
2. 建立覆盖品牌、品类、问题和能力意图的中文可索引页面。
3. 让 Agent Skills 和核心工作流形成可维护的主题集群。
4. 通过 GitHub Pages 提供稳定站点、站点地图和规范链接。
5. 建立可重复的 SEO 验收和后续观测流程。

### 2.2 非目标

- 不使用关键词堆砌、批量低质量页面或自动改写内容。
- 不为追求流量编造功能、数量、用户案例或性能结论。
- 不在第一阶段建设完整英文文档站。
- 不重构量化运行时、CLI、SDK 或 Agent Skills 的行为。
- 不承诺进入某个固定排名或在固定时间内获得自然流量。

## 3. 受众与搜索意图

### 3.1 核心受众

- 想用 AI 辅助完成量化研究的中文开发者和研究者。
- 正在选择回测、因子研究和策略审计工具的量化用户。
- 构建量化研究 Agent 的 AI 应用开发者。
- 需要可复现、可审计研究流程的团队。

### 3.2 关键词集群

核心品类词：

- `AI 量化`
- `AI 量化框架`
- `AI Agent 量化研究`
- `智能量化研究`

能力词：

- `AI 量化回测`
- `AI 因子研究`
- `量化策略稳健性检验`
- `量化回测审计`
- `AI 量化实盘交易`
- `可复现量化研究`

问题词：

- `AI 生成量化策略可靠吗`
- `如何避免量化回测偏差`
- `如何验证量化策略稳健性`
- `如何让 AI 做量化回测`

品牌词：

- `open-xquant`
- `open-xquant 教程`
- `open-xquant Agent Skills`

页面应服务明确搜索意图。一个页面只设一个主要意图，相关词自然出现在
标题、首段、章节标题和正文中，不要求固定密度。

## 4. 方案选择

### 4.1 采用方案：GitHub 元数据 + README + VitePress 中文站

该方案分三层建立信号：

1. GitHub 元数据负责搜索结果和仓库发现入口。
2. README 首屏负责 GitHub 页面语义和访问者快速理解。
3. VitePress 中文站负责主题集群、独立页面和技术 SEO。

选择 VitePress 的原因：

- 构建结果是静态 HTML，核心正文无需客户端执行即可读取。
- 每页可配置独立的 `title`、`description`、canonical 和社交元数据。
- 适合分层导航、交叉链接和大量 Markdown 内容。
- 可以在构建阶段从仓库真实来源生成 Skills 清单，避免手工数量漂移。
- GitHub Pages 部署路径和构建流程清晰。

代价是仓库会增加 Node.js 工具链。Node 仅用于文档构建，不进入 Python
包或 open-xquant 运行时依赖。

### 4.2 未采用方案：仅修改 GitHub 和 README

- Pros: 改动少，发布快，不增加构建工具。
- Cons: 只有一个主要可索引页面，无法覆盖多个搜索意图。
- Best for: 第一批快速收益和独立站上线前的过渡阶段。
- Risk: 排名信号受 GitHub 页面结构和单页内容容量限制。

### 4.3 未采用方案：MkDocs Material 中文站

- Pros: `pyproject.toml` 已声明 `mkdocs-material`，与 Python 工具链一致。
- Cons: 当前仍没有站点配置；页面元数据和前端定制约束更强。
- Best for: 极度重视单一 Python 工具链的维护模式。
- Risk: 为 SEO 展示和后续内容组件做定制时，配置可能逐步复杂化。

## 5. 信息架构

站点发布地址默认为：

`https://xingwudao.github.io/open-xquant/`

首期页面结构：

```text
/
guide/ai-quant-framework
guide/agentic-quant-research
guide/reproducible-quant-research
workflows/strategy-backtest
workflows/factor-research
workflows/research-audit
workflows/robustness-testing
workflows/live-trading
skills/index
skills/<canonical-skill-slug>
tools/index
examples/index
faq/index
```

页面职责：

- 首页：定义 open-xquant 和 `AI 量化框架` 品类，展示核心价值、流程、
  真实能力入口和快速开始。
- 品类页：解释 AI 量化研究框架解决的问题、边界和适用对象。
- 工作流页：围绕回测、因子、审计、稳健性和实盘分别回答一个搜索意图。
- Skills 索引：从真实清单展示当前数量、分类和入口。
- Skill 详情页：说明用途、触发场景、输入、输出、约束、产物和关联工作流。
- 工具索引：先定义稳定统计口径，再展示可验证的 CLI 或 SDK 能力。
- 示例页：链接可运行示例，并说明数据、命令、产物和限制。
- FAQ：回答真实的安装、复现、回测可信度和边界问题。

首期不为每个工具生成薄页面。只有能提供独立用途、输入输出、示例和限制的
实体才拥有详情页。

## 6. 文案与元数据设计

### 6.1 GitHub 仓库元数据

建议 `description`：

```text
AI 量化研究框架：AI Agent 驱动策略回测、因子研究、稳健性检验、审计报告与实盘交易 | Agentic Quant Research Kernel
```

建议 `homepage`：

```text
https://xingwudao.github.io/open-xquant/
```

建议 topics：

```text
ai-quant
quantitative-finance
quant-research
ai-agents
agentic-ai
backtesting
factor-research
algorithmic-trading
trading-strategy
python
```

GitHub 设置属于仓库外部状态。实施时通过 `gh repo edit` 更新，并在更新前后
读取实际值验证。站点尚未成功部署时，不提前设置无效 homepage。

### 6.2 README 首屏

保留 `# open-xquant` 作为唯一 H1。第一段改为：

```text
open-xquant 是中文友好的 AI 量化研究框架，面向 AI Coding Agent 和人类量化研究者，提供策略回测、因子研究、稳健性检验、审计报告与实盘交易工作流。
```

第二段保留 `Agentic Quant Research Kernel` 的技术定位，并解释确定性、
可复现和可审计的差异。首屏增加中文文档站入口，但不增加营销式标语堆叠。

### 6.3 站点首页元数据

```text
title: AI 量化框架 | open-xquant
description: open-xquant 是面向 AI Agent 和量化研究者的中文友好 AI 量化研究框架，覆盖策略回测、因子研究、稳健性检验、审计报告与实盘交易工作流。
```

所有详情页必须有唯一标题和描述。标题格式：

```text
<页面主要意图> | open-xquant
```

不在标题中加入未经验证的“最佳”“第一”或固定数量。

### 6.4 Python 项目元数据

`pyproject.toml` 的 description 改为：

```text
AI 量化研究框架 | Agentic Quant Research Kernel for reproducible and auditable research
```

keywords 改为：

```text
ai-quant
quantitative-finance
quant-research
ai-agents
backtesting
factor-research
algorithmic-trading
```

Python 项目元数据不承担中文长尾内容职责，避免加入更长描述。

## 7. 内容来源与生成策略

### 7.1 单一事实来源

- Skills 名称和说明来源于 `agent/skills/*/SKILL.md`。
- CLI 命令来源于 Click 命令树或现有命令定义。
- 工作流和产物边界来源于现有 guide、contracts 和运行时实现。
- 数量由构建脚本计算，不手工写入多个页面。

### 7.2 Skills 页面生成

构建前执行生成器：

1. 扫描 `agent/skills/*/SKILL.md`。
2. 解析 front matter 中必需的 name 和 description。
3. 读取 `website/data/skills.zh.yaml` 中经过人工审阅的中文扩展字段。
4. 按 name 合并用途、触发场景、输入、输出、约束和关联工作流。
5. 生成 Skills 索引数据和详情页 Markdown。
6. 对缺失扩展、孤立扩展、重复 slug 或无法解析的文件直接失败。

生成文件带有“由源文件生成”的注释，不允许在生成文件中手工维护事实。
英文事实变化时改真实 Skill 文档；中文搜索说明变化时改 SEO 扩展数据文件。

### 7.3 工具统计口径

工具索引只收录 Click 命令树中的公开叶子命令，group 节点不计为工具。
生成器通过公开 CLI 入口遍历命令树，并把命令路径作为唯一标识。

首期外部文案不展示工具总数，也不使用“70 个工具”。将来若展示数量，页面
必须说明统计对象是“公开 CLI 叶子命令”，并由同一个生成器产出。

### 7.4 内容质量规则

- 每页先回答用户问题，再介绍框架能力。
- 每个结论链接到代码、契约、示例或现有文档中的证据。
- 对研究结果、实盘能力和安全边界使用准确限定语。
- 避免多个页面重复同一段内容。
- 相关页面之间建立上下文链接，不建立无语义的全量链接。
- 页面更新日期来自 Git 历史或构建元数据，不手工伪造。

## 8. 技术 SEO 与部署

### 8.1 VitePress 配置

新增独立 `website/` 目录，避免把内部 specs、plans 和本地研究文档发布出去。

核心文件职责：

```text
website/package.json                 文档构建依赖和脚本
website/.vitepress/config.mts        导航、base、sitemap、head 和主题配置
website/.vitepress/theme/index.ts    轻量主题扩展
website/.vitepress/theme/custom.css  可访问性和响应式样式
website/index.md                     SEO 首页
website/guide/                       品类与方法页面
website/workflows/                   核心工作流页面
website/skills/                      Skills 索引和生成页面
website/tools/                       工具索引
website/data/skills.zh.yaml          Skills 中文扩展字段
website/public/robots.txt            抓取规则
scripts/generate_seo_docs.py         从仓库事实源生成站点内容
```

站点 `base` 固定为 `/open-xquant/`。canonical、sitemap 和 Open Graph URL
使用完整生产地址，不使用构建机路径或预览地址。

### 8.2 每页技术要求

- 服务端可读取的静态 HTML 正文。
- 唯一的 `title` 和 meta description。
- 唯一 canonical URL。
- 一个语义明确的 H1。
- 合理的 H2/H3 层级。
- Open Graph 和基础社交分享元数据。
- 首页使用 `WebSite` 和 `SoftwareSourceCode` JSON-LD。
- 内页使用 `BreadcrumbList` JSON-LD。
- 图片有稳定尺寸、描述性文件名和准确 alt 文本。
- 内部链接不依赖 JavaScript 点击事件。
- 404 页面能返回用户入口，但不进入 sitemap。

结构化数据只描述页面可见且可验证的事实，不用于制造搜索展示效果。

### 8.3 GitHub Actions

新增 Pages 工作流：

1. 在 pull request 中安装锁定依赖并执行站点构建、生成器校验和链接检查。
2. 在 `main` 更新时重新构建并部署 GitHub Pages。
3. 使用最小权限，只授予 Pages 部署所需权限。
4. 通过并发组取消旧部署，避免过期构建覆盖新版本。
5. 上传构建产物前检查 sitemap、canonical 和关键页面。

实施后需要在 GitHub 仓库设置中将 Pages source 设为 GitHub Actions。

### 8.4 索引接入

- 部署成功后验证首页、核心页、robots 和 sitemap 可公开访问。
- 在 Google Search Console 验证站点所有权。
- 提交 `sitemap.xml`。
- 对首页和核心品类页请求首次抓取。
- 不使用批量索引服务或非官方提交接口。

Search Console 所有权操作需要仓库拥有者账户完成；代码侧可预留验证文件或
meta 标签插槽。

## 9. 数据流与失败处理

构建数据流：

```text
Skill/CLI/guide sources
  -> content generator
  -> generated Markdown and inventory data
  -> VitePress static build
  -> SEO validation and link check
  -> GitHub Pages artifact
  -> production smoke check
```

失败策略：

- 源文件无法解析：构建失败并报告具体文件。
- slug 重复：构建失败，不自动覆盖。
- 数量与索引不一致：构建失败。
- 内部链接失效：pull request 检查失败。
- production base 或 canonical 配置错误：部署前检查失败。
- GitHub Pages 部署失败：保留上一版本，不更新 homepage。
- 外部搜索服务不可用：不阻塞站点部署，只记录观测缺口。

## 10. 验收标准

### 10.1 仓库和文案

- 当前工作分支为 `seo-ai-quant-optimization`。
- GitHub description、homepage 和 topics 与批准文案一致。
- README 首段包含准确、自然的 `AI 量化研究框架` 定位。
- README 和站点只使用 canonical 项目命名。
- 不出现未经自动验证的 Skills 或工具数量。

### 10.2 站点

- `website/` 可在干净环境完成依赖安装和生产构建。
- 首页及所有首期核心页面生成非空静态 HTML。
- 每个可索引页面有唯一 title、description、canonical 和 H1。
- sitemap 只包含应公开的页面，且 URL 使用生产域名。
- robots 允许抓取公开内容并声明 sitemap。
- GitHub Pages 子路径下的 CSS、JS、图片和内部链接全部可用。
- 生成器输出的 Skill 数量等于 `agent/skills/*/SKILL.md` 实际数量。
- 链接检查、命名检查和 SEO 元数据检查通过。

### 10.3 发布后

- 生产首页、至少三个核心页、robots 和 sitemap 返回成功状态。
- Search Console 所有权验证完成并提交 sitemap。
- 记录上线日期和上线前查询基线。

## 11. 效果衡量

上线前记录：

- 品牌词和核心品类词当前是否出现、对应落地页和大致位置。
- GitHub 页面当前 title 和 snippet。
- 已被索引的页面数量。

上线后按周记录 Search Console 数据：

- 目标查询的 impressions、clicks、CTR 和 average position。
- 各落地页的 impressions 和 clicks。
- 已发现、已抓取、已索引页面数量。
- title 或 description 被 Google 改写的页面。
- 无效 canonical、重复页面和抓取错误。

首个评估窗口为上线后 6 至 8 周。判断依据是曝光、索引覆盖和目标查询趋势，
不以单次无痕搜索截图作为唯一结论。

## 12. 实施顺序

1. 建立基线记录和可验证文案。
2. 修改 README 与 Python 项目元数据。
3. 搭建 VitePress、生成器和首期内容。
4. 增加构建、链接和 SEO 验证。
5. 配置 GitHub Actions 并部署 Pages。
6. 部署成功后更新 GitHub description、homepage 和 topics。
7. 完成 Search Console 验证和 sitemap 提交。
8. 进入每周观测与按数据调整阶段。

## 13. 约束与风险

- GitHub description 可能被搜索引擎采用，但搜索引擎会自行生成 title 和
  snippet，不能假设该文本必然原样展示。
- 新站点初期缺少外部链接和历史权重，不能只依赖页面数量。
- 自动生成页面容易变薄，必须保证每页有真实用途、边界和证据。
- Agent Skills 数量会变化，所有数量展示必须自动生成。
- GitHub Pages 使用项目子路径，错误的 `base` 会导致资源和链接失效。
- GitHub 设置和 Search Console 属于外部状态，需要在实施计划中单独验证。
- 文档与运行时事实可能漂移，CI 必须阻止无法解析或失效的生成内容发布。

## 14. 官方参考

- Google title links:
  `https://developers.google.com/search/docs/appearance/title-link`
- Google snippets:
  `https://developers.google.com/search/docs/appearance/snippet`
- Google structured data:
  `https://developers.google.com/search/docs/appearance/structured-data/intro-structured-data`
- GitHub repository topics:
  `https://docs.github.com/articles/classifying-your-repository-with-topics`
- GitHub Pages:
  `https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages`
- VitePress deployment:
  `https://vitepress.dev/guide/deploy`
