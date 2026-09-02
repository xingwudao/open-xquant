---
title: open-xquant Agent Skills
description: open-xquant 中文 Agent Skill 主题集，覆盖研究治理、策略审计、数据因子、组件开发、执行与报告。
outline: deep
---

# Agent Skills 索引

这些页面由 `agent/skills/*/SKILL.md` 和 `website/data/skills.zh.yaml` 生成。
当前覆盖 35 个 canonical Skill。


## 研究治理

### [open-xquant AI 量化研究路由](/skills/open-xquant)

根据量化研究请求选择正确的工作流，并在运行确定性工具前完成边界检查。

- Canonical Skill: `open-xquant`
- Source: `agent/skills/open-xquant/SKILL.md`

### [量化实验结果比较](/skills/compare-experiments)

在非版本化 legacy 工作区中比较两个已完成实验运行的 SPEC、审计和指标差异。

- Canonical Skill: `compare-experiments`
- Source: `agent/skills/compare-experiments/SKILL.md`

### [量化研究工作区治理](/skills/govern-research-workspace)

检查版本化研究工作区的路径边界、阶段产物位置和跨产物血缘一致性。

- Canonical Skill: `govern-research-workspace`
- Source: `agent/skills/govern-research-workspace/SKILL.md`

### [量化策略最终版本选择](/skills/select-final-version)

只从满足审计、报告、血缘和用户确认策略门槛的候选中发布最终研究候选。

- Canonical Skill: `select-final-version`
- Source: `agent/skills/select-final-version/SKILL.md`

### [量化策略版本比较](/skills/compare-strategy-versions)

在版本化工作区中比较运行或策略版本，并区分同版本可复现性与跨版本策略证据。

- Canonical Skill: `compare-strategy-versions`
- Source: `agent/skills/compare-strategy-versions/SKILL.md`

### [量化策略版本管理](/skills/manage-strategy-version)

判断当前研究应继续现有版本、创建新策略版本，还是追加同一策略的运行尝试。

- Canonical Skill: `manage-strategy-version`
- Source: `agent/skills/manage-strategy-version/SKILL.md`


## 策略与审计

### [量化产物血缘审计](/skills/audit-artifact-lineage)

重新计算版本、运行、报告、比较和最终选择之间的路径与哈希绑定，验证候选证据当前有效。

- Canonical Skill: `audit-artifact-lineage`
- Source: `agent/skills/audit-artifact-lineage/SKILL.md`

### [量化策略想法审计](/skills/audit-strategy-idea)

审计策略想法简报和原始对话，确认每个已确认值都有用户证据后才允许构建 SPEC。

- Canonical Skill: `audit-strategy-idea`
- Source: `agent/skills/audit-strategy-idea/SKILL.md`

### [量化策略想法梳理](/skills/brainstorm-strategy-idea)

按固定阶段引导用户补全策略想法，并在 SPEC 编写前产出有用户证据的想法简报。

- Canonical Skill: `brainstorm-strategy-idea`
- Source: `agent/skills/brainstorm-strategy-idea/SKILL.md`

### [量化策略规格审计](/skills/audit-strategy-spec)

审计 SPEC 字段来源、默认值、组件选择和用户确认状态，阻止未批准假设进入回测。

- Canonical Skill: `audit-strategy-spec`
- Source: `agent/skills/audit-strategy-spec/SKILL.md`

### [量化策略规格构建](/skills/build-strategy-spec)

从已通过想法审计的简报构建并验证 `strategy_spec.yaml`，同时记录字段映射和构建结果。

- Canonical Skill: `build-strategy-spec`
- Source: `agent/skills/build-strategy-spec/SKILL.md`

### [量化运行语义审计](/skills/audit-runtime-semantics)

编译策略预览并审计 `compiled_plan.json` 是否保留 SPEC 的关键执行语义。

- Canonical Skill: `audit-runtime-semantics`
- Source: `agent/skills/audit-runtime-semantics/SKILL.md`


## 数据与因子

### [时间序列因子评估](/skills/evaluate-time-series)

用命中率、盈亏比、decay 曲线、空仓行为和 tearsheet 评估择时或轮动因子。

- Canonical Skill: `evaluate-time-series`
- Source: `agent/skills/evaluate-time-series/SKILL.md`

### [横截面因子评估](/skills/evaluate-cross-sectional)

用 IC、Rank IC、ICIR、decay 和 turnover 评估同日多资产排序因子。

- Canonical Skill: `evaluate-cross-sectional`
- Source: `agent/skills/evaluate-cross-sectional/SKILL.md`

### [量化参数调优](/skills/tune-parameters)

用网格搜索、walk-forward、时间序列交叉验证和过拟合信号检查调参结果。

- Canonical Skill: `tune-parameters`
- Source: `agent/skills/tune-parameters/SKILL.md`

### [量化因子筛选](/skills/screen-factors)

基于价格、财务和自定义因子筛选候选标的，但不把筛选结果等同于回测验证。

- Canonical Skill: `screen-factors`
- Source: `agent/skills/screen-factors/SKILL.md`

### [量化因子评估](/skills/evaluate-factor)

根据因子问题类型路由到横截面或时间序列评估，并先确认定义、样本和对齐方式。

- Canonical Skill: `evaluate-factor`
- Source: `agent/skills/evaluate-factor/SKILL.md`

### [量化数据探索](/skills/explore-data)

检查、下载和验证本地市场、宏观与财务数据，为回测或因子研究确认数据可用性。

- Canonical Skill: `explore-data`
- Source: `agent/skills/explore-data/SKILL.md`

### [量化标的池构建](/skills/build-universe)

定义策略可交易标的池，并说明静态、指数快照和存活偏差约束。

- Canonical Skill: `build-universe`
- Source: `agent/skills/build-universe/SKILL.md`


## 组件开发

### [量化信号创建](/skills/create-signal)

创建确定性的 boolean 或 categorical 交易意图 Signal，并验证输出域和因果行为。

- Canonical Skill: `create-signal`
- Source: `agent/skills/create-signal/SKILL.md`

### [量化指标创建](/skills/create-indicator)

为 open-xquant 创建纯数值时间序列 Indicator，并补齐测试、导出和注册表接线。

- Canonical Skill: `create-indicator`
- Source: `agent/skills/create-indicator/SKILL.md`

### [量化指标可视化](/skills/plot-indicators)

绘制 open-xquant 运行图表和指标叠加，用于视觉检查而不是替代验证或审计。

- Canonical Skill: `plot-indicators`
- Source: `agent/skills/plot-indicators/SKILL.md`

### [量化组件创建](/skills/create-component)

在确认需要修改 open-xquant 框架内置组件时，检查注册表并路由到具体组件创建 Skill。

- Canonical Skill: `create-component`
- Source: `agent/skills/create-component/SKILL.md`

### [量化组件实现](/skills/author-component)

在工作区本地实现自定义组件、测试、清理缓存并发布带哈希的组件清单和目录。

- Canonical Skill: `author-component`
- Source: `agent/skills/author-component/SKILL.md`

### [量化组合优化器创建](/skills/create-portfolio-optimizer)

创建返回目标权重的 PortfolioOptimizer，并用权重和 fallback 不变量测试验证。

- Canonical Skill: `create-portfolio-optimizer`
- Source: `agent/skills/create-portfolio-optimizer/SKILL.md`

### [量化规则创建](/skills/create-rule)

为 open-xquant 创建 bar-by-bar 风控、持有、权重覆盖或退出 Rule，并用组合状态测试验证。

- Canonical Skill: `create-rule`
- Source: `agent/skills/create-rule/SKILL.md`

### [量化规则实现](/skills/build-rule)

配置和解释交易规则、退出、风控 hold 与再平衡限制，并明确当前 SPEC 支持边界。

- Canonical Skill: `build-rule`
- Source: `agent/skills/build-rule/SKILL.md`


## 执行与报告

### [授权量化回测执行](/skills/run-authorized-backtest)

在 SPEC、SPEC 审计、运行语义审计和授权文件都通过后执行正式回测。

- Canonical Skill: `run-authorized-backtest`
- Source: `agent/skills/run-authorized-backtest/SKILL.md`

### [量化交易执行配置](/skills/configure-trade-execution)

配置信号时间、成交时间、价格模式、费用、滑点、手数、现金收益和交易日历假设。

- Canonical Skill: `configure-trade-execution`
- Source: `agent/skills/configure-trade-execution/SKILL.md`

### [量化实盘交易管理](/skills/manage-live-trading)

在严格安全门槛下连接 Alpaca 纸面或实盘交易，执行账户检查、行情读取或订单提交。

- Canonical Skill: `manage-live-trading`
- Source: `agent/skills/manage-live-trading/SKILL.md`

### [量化报告图表构建](/skills/build-report-charts)

为研究报告构建默认专业图表包，登记图形、脚本和 manifest 哈希后交给报告写作。

- Canonical Skill: `build-report-charts`
- Source: `agent/skills/build-report-charts/SKILL.md`

### [量化研究报告审阅](/skills/review-research-report)

审阅已完成报告的决策一致性、证据忠实度、审计解释、图表叙事和语义问题。

- Canonical Skill: `review-research-report`
- Source: `agent/skills/review-research-report/SKILL.md`

### [量化研究报告撰写](/skills/write-research-report)

基于指标、审计、稳健性和已登记图表资产撰写最终人类可读研究报告并渲染 HTML。

- Canonical Skill: `write-research-report`
- Source: `agent/skills/write-research-report/SKILL.md`

### [量化策略运行监控](/skills/monitor-strategy-run)

对已完成运行执行可复现性、研究偏差、稳健性和实验登记，并保持运行包完整性。

- Canonical Skill: `monitor-strategy-run`
- Source: `agent/skills/monitor-strategy-run/SKILL.md`

### [量化绩效复核](/skills/review-performance)

按审计、执行假设、稳健性、OOS 覆盖和指标顺序解释回测表现与限制。

- Canonical Skill: `review-performance`
- Source: `agent/skills/review-performance/SKILL.md`
