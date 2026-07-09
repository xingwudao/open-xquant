# WorkBuddy 技能体系与 OpenXQuant 对比学习报告

生成日期：2026-07-02

工作目录：`/Users/daodao/Documents/2-coding-space/git/github.com/open-xquant`

## 结论摘要

WorkBuddy 值得学习的核心不是某一个金融算法，而是“技能路由 +
数据工具 + 方法论资料 + 可执行脚本 + 插件/MCP/Hook 扩展”的整体产品化形态。
它把金融任务拆成了一个强制入口 `wb-finance-skill`，再分流到
`neodata-financial-search`、`westock-data`、`westock-tool` 和本地
Python helper。这个结构对用户体验非常有效。

OpenXQuant 的核心强项更偏工程可靠性：`strategy_spec.yaml`、
`spec_audit.json`、`compiled_plan.json`、`runtime_audit.json`、
`artifact_hashes.json`、`report_assets/manifest.json` 和 worker role
边界形成了可审计链路。WorkBuddy 生成的本地回测脚本很灵活，但同一轮动策略在
两个本地工作目录中出现了明显结果差异，说明它缺少 OpenXQuant 这种规格锁定、
运行审计和复现门禁。

建议 OpenXQuant 直接照搬 WorkBuddy 的“场景方法论 reference 包”、
“数据源路由/边界红线”、“动态清单命令优先于文档样例”和“技能包元数据扩展”。
插件市场、Hook、MCP app、Ask/Craft/Expert 模式和产物呈现机制适合借鉴思路。
不建议照搬 WorkBuddy 的混淆 JS 工具、自动安装 Python 包、未固定口径的
Agent 自写回测脚本。

## 扫描范围与方法

本次扫描只做读取和分析，未修改 WorkBuddy 安装目录和用户数据。
误落到 OpenXQuant 仓库根目录的临时 asar 抽取文件已经清理。

WorkBuddy 证据源：

- 应用包：`/Applications/WorkBuddy.app`
- Electron 主资源：`/Applications/WorkBuddy.app/Contents/Resources/app.asar`
- 未打包资源：
  `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked`
- 内置技能：
  `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills`
- 内置插件：
  `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins`
- 用户生成工作区：`/Users/daodao/Workbuddy`
- 日志：`/Users/daodao/Library/Logs/WorkBuddy`
- 认证信息：
  `/Users/daodao/Library/Application Support/CodeBuddyExtension/Data/Public/auth/workbuddy-desktop.info`

OpenXQuant 证据源：

- 技能：`agent/skills/*/SKILL.md`
- Worker roles：`agent/roles/*.md`
- Agent 安装器：`src/oxq/cli/agent.py`
- 目标适配器：`src/oxq/cli/agent_targets.py`
- Agent manifest：`src/oxq/cli/agent_manifest.py`
- 组件 manifest/catalog：`src/oxq/core/component_manifest.py`、
  `src/oxq/core/component_catalog.py`
- Runtime audit：`src/oxq/spec/runtime_audit_schema.py`
- 复现审计：`src/oxq/audit/reproducibility.py`
- Report QA：`src/oxq/report/qa.py`

限制说明：

- WorkBuddy 的 `westock-data/scripts/index.js` 和
  `westock-tool/scripts/index.js` 是大型混淆 JS 文件；本报告只记录其
  CLI 能力、包形态和运行边界，不逐行反编译。
- 认证文件含账户/授权字段，只确认存在和用途，不记录任何敏感值。
- `app.asar` 可列清单和读取文本资源；完整解包会碰到若干未打包二进制占位路径，
  但不影响技能、包、文档和插件证据判断。

## WorkBuddy 安装形态

应用基础信息：

- 应用路径：`/Applications/WorkBuddy.app`
- 应用版本：`5.1.7`
- Bundle id：`com.workbuddy.workbuddy`
- 类型：Electron 桌面应用
- 主入口：`Contents/Resources/app.asar`
- `app.asar` 大小：约 `234M`
- `app.asar.unpacked` 包含内置技能、插件、MCP app、CLI、原生模块和资源。

资源分层：

- `resources/builtin-skills`：应用内置技能，共 10 个技能目录。
- `resources/builtin-plugins`：应用内置插件；本机发现 `weixinpay` 插件。
- `resources/builtin-mcp-apps`：内置 MCP app；本机发现 `ardot-mcp-app`。
- `cli`：内置 CodeBuddy CLI，包含产品配置、web-ui 文档、vendor 和运行时。
- `node_modules`：未打包原生/运行时模块。
- `vendor/node.tar.gz` 与 `vendor/python.dat`：内置运行时资源。

宿主侧提示模板：

- `resources/templates` 下有 12 个 `.tpl` 模板。
- 模板覆盖 Ask、Craft、Expert、Design、Coding、system reminder 和 style。
- 只记录结构，不复述内部提示正文。

## WorkBuddy 技能总清单

内置技能共 10 个：

- `ardot-design-assistant`
  - 文件数：26
  - 重点：Ardot 画布设计、设计到代码、幻灯片设计稿、移动端/网页/海报规范。
  - 资料：9 个 references、4 个 workflows、3 个 rules、2 个 tool-usage。
- `buddy-multimodal-generation`
  - 文件数：2
  - 重点：文生视频、图片视频特效、文生/图生 3D。
  - 脚本：`scripts/buddy-cloud.py`
- `cloudstudio-deploy`
  - 文件数：3
  - 重点：静态站点部署到 CloudStudio sandbox。
  - 脚本：`scripts/deploy.js`
- `expert-manager`
  - 文件数：11
  - 重点：WorkBuddy 专家包创建、注册、打包、校验。
  - 脚本：5 个 Python 脚本。
  - 资料：4 个专家包规范 reference。
- `marketplace-skill-installer`
  - 文件数：1
  - 重点：市场技能安装。
  - `allowed-tools` 指向 `workbuddy_marketplace_skill`。
- `neodata-financial-search`
  - 文件数：4
  - 重点：自然语言金融数据查询。
  - 版本：`1.0.1`
  - `allowed-tools`：`Read,Bash`
- `skill-creator`
  - 文件数：5
  - 重点：创建和打包 CodeBuddy 技能。
  - 脚本：`init_skill.py`、`package_skill.py`、`quick_validate.py`
- `wb-finance-skill`
  - 文件数：63
  - 重点：金融服务总入口。
  - 版本：`1.6.0`
  - 资料：46 个金融方法论 reference。
  - 脚本：16 个 Python 文件。
- `westock-data`
  - 文件数：8
  - 重点：结构化行情、财务、事件、资讯、宏观、ETF、期货、外汇数据。
  - `package.json`：`westock-data@1.0.4`
- `westock-tool`
  - 文件数：8
  - 重点：选股、选基、筛选、策略、标签、事件、排行。
  - `package.json`：`westock-tool@1.0.0`

插件提供技能共 3 个，均来自 `weixinpay`：

- `weixinpay-register`
  - 重点：微信 AI 支付/AI 专属卡开通、绑定、状态查询。
- `weixinpay-feedback`
  - 重点：支付/绑定异常反馈。
- `weixinpay-required-filter`
  - 重点：过滤 `WeixinPay-Required:` 原始支付授权内容，避免直接暴露给用户。

## WorkBuddy 金融技能架构

金融入口：`wb-finance-skill`

- 覆盖金融、投资、股票、基金、ETF、板块、指数、宏观、外汇、大宗商品、
  财报、估值、持仓、交易、仓位、量化、因子、回测、选股、期权、衍生品、
  投行建模、技术指标、行情监控和预警。
- 明确要求金融任务先加载总入口，再调用数据 skill。
- 明确禁止裸答、编造数据、混淆财务概念、忽略时间口径。
- 要求根据具体场景读取 1-3 个 reference。
- 要求复杂分析优先生成 HTML 研报，并做 JavaScript 语法自检。

数据层分工：

- `neodata-financial-search`
  - 自然语言通用金融数据搜索。
  - 代理接口：`https://copilot.tencent.com/agenttool/v1/neodata`
  - Token 缓存：技能目录 `.neodata_token` 或 `~/.workbuddy/.neodata_token`
  - `requests` 为可选依赖，缺失时脚本会尝试安装，再退化到 `urllib`。
- `westock-data`
  - 单只或批量标的结构化数据查询。
  - CLI 支持 quote、minute、kline、technical、finance、report、notice、
    fund、shareholder、dividend、buyback、profile、search、hot、
    market-overview、index、connect、ipo、changedist、etf、macro、
    sector、calendar、events、risk、futures、forex、bond 等。
  - 支持 A 股、中国香港股票、美股、日韩股、ETF、指数、板块、期货、
    外汇、可转债等。
- `westock-tool`
  - 从全市场找“哪些股票/ETF”。
  - CLI 支持 `filter`、`strategy`、`label`、`event`、`ranking`。
  - 文档强制要求清单类问题执行 `--list`，不能凭文档样例回答。
  - 明确不要用 `westock-data` 拉一堆数据再手搓筛选。

方法论层：`wb-finance-skill/references`

46 个金融 reference 覆盖如下：

- `abnormal-detection.md`
- `announcement-impact.md`
- `breakout-patterns.md`
- `business-model.md`
- `crisis-event.md`
- `crypto-derivatives.md`
- `daily-briefing.md`
- `dividend-buyback.md`
- `earnings-preview.md`
- `earnings-review.md`
- `event-catalyst.md`
- `fixed-income.md`
- `forex-commodity.md`
- `fund-flow.md`
- `going-global.md`
- `html-report-style.md`
- `ib-deal-prep.md`
- `ib-models.md`
- `industry-chain.md`
- `institutional-holding.md`
- `leader-game.md`
- `macro-transmission.md`
- `management-assessment.md`
- `market-mainline.md`
- `market-state.md`
- `moat-quality.md`
- `monitor-alert.md`
- `options-strategies.md`
- `peer-comparison.md`
- `policy-impact.md`
- `portfolio-checkup.md`
- `portfolio-optimization.md`
- `position-sizing.md`
- `price-action-tools.md`
- `quality-growth.md`
- `quant-factor-research.md`
- `risk-stress.md`
- `sector-comparison.md`
- `stock-deep-research.md`
- `stock-first-look.md`
- `stop-discipline.md`
- `systematic-strategies.md`
- `tdx-mcp-quick-reference.md`
- `theme-lifecycle.md`
- `trade-plan.md`
- `valuation-pricing.md`

脚本层：`wb-finance-skill/scripts`

16 个 Python 文件：

- `run_signal.py`
- `quant/factor_fundamental.py`
- `quant/factor_multi.py`
- `quant/minute_data.py`
- `quant/pair_trading.py`
- `quant/seasonality.py`
- `quant/volatility.py`
- `price-action/basic_indicators.py`
- `price-action/candlestick_patterns.py`
- `price-action/chan_theory.py`
- `price-action/elliott_wave.py`
- `price-action/harmonic_patterns.py`
- `price-action/ichimoku.py`
- `price-action/smart_money.py`
- `ib/extract_ib_numbers.py`
- `ib/validate_dcf.py`

`run_signal.py` 统一支持 12 类 engine：

- `basic`
- `candlestick`
- `chan`
- `elliott`
- `factor_fundamental`
- `factor_multi`
- `harmonic`
- `ichimoku`
- `pair`
- `seasonality`
- `smc`
- `volatility`

另外 `run_signal.py` 内置 `vcp` 轻量检测器，不是单独脚本文件。

## WorkBuddy 插件、MCP 与 Hook 架构

插件结构：

- 插件根目录包含 `.codebuddy-plugin/plugin.json`。
- 技能放在插件根的 `skills/<skill>/SKILL.md`。
- MCP 配置放在插件根的 `.mcp.json`。
- Hooks 放在插件根的 `hooks/hooks.json`。
- 可执行文件放在插件根的 `bin/`。

本机内置插件：`weixinpay`

- Manifest：
  `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins/weixinpay/.codebuddy-plugin/plugin.json`
- MCP：
  `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins/weixinpay/.mcp.json`
- Hooks：
  `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins/weixinpay/hooks/hooks.json`
- MCP server：`weixinpay`
- Hook 事件：`UserPromptSubmit` 和 `PreToolUse`
- PreToolUse matcher：`DeferExecuteTool`

MCP app：

- `ardot-mcp-app/cli.cjs`
- `ardot-mcp-app/webview/app.html`
- `_workbuddy-runtime/mcp-app-bootstrap.cjs`

宿主文档显示的扩展能力：

- 技能支持 `allowed-tools`、`disable-model-invocation`、`user-invocable`、
  `context: fork`、`agent`、`model`、`hooks` 等 frontmatter。
- 插件支持 commands、agents、skills、hooks、`.mcp.json`、`.lsp.json`、
  `bin/`、`settings.json`。
- 插件市场支持 GitHub/Git/本地路径/HTTP URL。
- MCP 有 user/project/local 作用域和 allow/ask/deny 权限规则。
- 子代理支持项目级、用户级、插件级和 CLI 动态 JSON 定义。

## WorkBuddy 包与依赖扫描

技能级包：

- `westock-data/package.json`
  - `name`: `westock-data`
  - `version`: `1.0.4`
  - `type`: `module`
  - `bin`: `westock-data -> ./scripts/index.js`
  - declared dependencies：无
- `westock-tool/package.json`
  - `name`: `westock-tool`
  - `version`: `1.0.0`
  - `type`: `module`
  - `bin`: `westock-tool -> ./scripts/index.js`
  - declared dependencies：无

技能 Python 可见依赖：

- `pandas`
- `numpy`
- `requests`
- `czsc`
- `smartmoneyconcepts`
- `openpyxl`，仅在 DCF Excel 校验时动态检查。
- 标准库：`argparse`、`json`、`subprocess`、`pathlib`、`zipfile`、
  `re`、`datetime`、`hmac`、`hashlib`、`urllib` 等。

用户生成工作区可见依赖：

- `/Users/daodao/Workbuddy/2026-07-02-08-13-13/fetch_data.py`
  使用 `akshare` 和 `pandas`。
- `/Users/daodao/Workbuddy/2026-07-02-08-09-46/backtest.py`
  使用 `pandas` 和 `numpy`。

宿主顶层依赖，来自 WorkBuddy `package.json`：

- `@opentelemetry/api`
- `@opentelemetry/core`
- `@opentelemetry/exporter-trace-otlp-http`
- `@opentelemetry/exporter-trace-otlp-proto`
- `@opentelemetry/resources`
- `@opentelemetry/sdk-trace-base`
- `@modelcontextprotocol/sdk`
- `@tencent/aegis-electron-sdk-v2`
- `@tencent/aegis-web-sdk-v2`
- `@tencent/docs-engine`
- `@tencent/universal-report`
- `@wecom/aibot-node-sdk`
- `@larksuiteoapi/node-sdk`
- `dingtalk-stream`
- `axios`
- `better-sqlite3`
- `cos-nodejs-sdk-v5`
- `@lydell/node-pty`
- `@openai/agents`
- `adm-zip`
- `chokidar`
- `nunjucks`
- `reflect-metadata`
- `semver`
- `tar`
- `zod`
- `electron-log`
- `undici`
- `ws`
- `@slack/socket-mode`
- `@slack/web-api`
- `@tencent-connect/qqbot-connector`

宿主 optional dependencies：

- `@lydell/node-pty-darwin-arm64`
- `@lydell/node-pty-darwin-x64`
- `@lydell/node-pty-linux-arm64`
- `@lydell/node-pty-linux-x64`
- `@lydell/node-pty-win32-arm64`
- `@lydell/node-pty-win32-x64`
- `node-pty`

ASAR 内转置依赖扫描：

- `/tmp/workbuddy-asar-min/node_modules` 识别到 548 个 `package.json`。
- 关键包族包括 `@openai/agents`、`@modelcontextprotocol/sdk`、`openai`、
  `zod`、`express`、`axios`、OpenTelemetry、Slack/飞书/钉钉/企微/QQ
  连接器、`cos-nodejs-sdk-v5`、`better-sqlite3`、`node-pty`。
- 未打包原生/运行时模块包括 `better-sqlite3@12.8.0`、
  `node-pty@1.1.0`、`@lydell/node-pty-darwin-arm64@1.2.0-beta.12`、
  `@tencent/docs-engine@0.0.1-beta.9`、`fsevents@1.2.13`。

外部服务端点：

- `https://copilot.tencent.com/agenttool/v1/neodata`
- `https://copilot.tencent.com`
- `https://www.okx.com/api/v5/market/candles`
- CloudStudio sandbox 相关 API，位于 `cloudstudio-deploy/scripts/deploy.js`。

## WorkBuddy 本地金融回测产物

用户目录：`/Users/daodao/Workbuddy`

发现 3 个本地项目目录：

- `2026-07-02-08-09-46`
  - `backtest.py`
  - `gen_report.py`
  - `data/backtest_results.json`
  - `global_rotation_backtest_report.html`
  - 结果：策略总收益约 `26.37%`，基准约 `34.59%`，夏普约 `0.95`。
- `2026-07-02-08-13-13`
  - `fetch_data.py`
  - `data_510300.csv`
  - `data_513100.csv`
  - `data_518880.csv`
  - 使用 `akshare` 与 `pandas`。
- `2026-07-02-08-22-20`
  - `strategy_backtest.py`
  - `generate_report.py`
  - `backtest_result.json`
  - `backtest_report.html`
  - `nav_curve.csv`
  - `trades.csv`
  - 结果：策略总收益约 `41.57%`，年化约 `26.51%`，最大回撤约 `14.94%`。

关键观察：

- 同样是全球轮动策略，两个 WorkBuddy 产物的收益差异显著。
- 差异来自脚本自由生成后的口径变化，例如信号公式、回测结束日期、成交价、
  权重上限、现金处理、数据源格式和调仓生效规则。
- 这说明 WorkBuddy 的体验和速度强，但正式研究需要 OpenXQuant 式 spec、
  audit、compiled plan、artifact hash 和 reproducibility gate。

## OpenXQuant 当前实现基线

OpenXQuant 技能与角色：

- 技能数量：35 个。
- Worker role 数量：17 个。
- 入口技能：`agent/skills/open-xquant/SKILL.md`
- 角色目录：`agent/roles`
- 安装目标：`codex`、`opencode`、`claude-code`、`cursor`、`openclaw`、
  `trae`。

关键技能分组：

- Router：`open-xquant`
- 构建/审计/运行：`build-strategy-spec`、`audit-strategy-spec`、
  `audit-runtime-semantics`、`run-authorized-backtest`、
  `monitor-strategy-run`
- 数据/因子/筛选：`explore-data`、`evaluate-factor`、
  `evaluate-cross-sectional`、`evaluate-time-series`、`screen-factors`
- 参数与比较：`tune-parameters`、`compare-experiments`
- 报告：`build-report-charts`、`write-research-report`、
  `review-research-report`、`review-performance`、`plot-indicators`
- 组件：`create-component`、`author-component`、`create-indicator`、
  `create-signal`、`create-rule`、`create-portfolio-optimizer`
- 交易与规则：`build-universe`、`build-rule`、
  `configure-trade-execution`、`manage-live-trading`

强项：

- Router 明确禁止越过 leaf skill 直接跑 `oxq` 或写报告。
- Coordinator role 只协调，不产出 gated artifacts。
- Runner role 明确禁止改 `strategy_spec.yaml`、`spec_audit.json`、
  `runtime_audit.json`。
- Report writer role 只能读 gated run artifacts，不改 run/spec/audit。
- Agent 安装器写 `.open-xquant-managed.json` 和
  `~/.config/open-xquant/agent-install.json`。
- 每个安装文件记录 `source_sha256` 和 `dest_sha256`。
- 组件扩展有 `component_manifest.json`、source/test hash、
  `bundle_hash` 和 run-local archived manifest 验证。
- Report QA 检查 Markdown/HTML、图片路径、asset manifest、sha256、
  source script path 和数字声明。

当前短板：

- 技能 frontmatter 只有 `name` 和 `description`，缺少版本、工具权限、
  context、agent、model、tags、examples、compatibility。
- 组件 manifest 很强，但只覆盖 OpenXQuant 组件，不覆盖通用技能包、
  数据包、工具包、MCP、Hook。
- 没有 WorkBuddy 式插件市场、命名空间技能、项目/用户/本地 scope
  管理。
- UI/产物呈现弱于 WorkBuddy；OpenXQuant 主要靠文件、CLI 和报告。
- 数据源路由不如 WorkBuddy 的金融总入口细；OpenXQuant 强在本地数据和
  可审计，但对实时金融查询、选股/排行、宏观/研报/事件数据的产品化入口较少。

## 关键对比

[技能入口路由]
- Pros:
  WorkBuddy 的 `wb-finance-skill` 把金融任务统一入口、数据源顺序、
  时间口径、红线和 reference 选择写得很完整；OpenXQuant 的
  `open-xquant` router 把运行权限和 leaf skill 边界写得更硬。
- Cons:
  WorkBuddy 的规则更多靠提示执行；OpenXQuant 的方法论覆盖面不如
  WorkBuddy 宽。
- Best for:
  WorkBuddy 适合交互式金融问答和快速研究；OpenXQuant 适合正式、
  可复现、可审计回测。
- Risk:
  WorkBuddy 容易生成不同口径脚本；OpenXQuant 容易显得重、慢、
  需要用户接受更多门禁。

[数据工具分层]
- Pros:
  WorkBuddy 把自然语言数据、结构化详情、全市场筛选拆成三层；
  OpenXQuant 把本地数据、SPEC、运行和审计拆成确定链路。
- Cons:
  WorkBuddy 外部数据源和混淆工具不可完全审计；OpenXQuant 缺少
  即问即查的实时金融数据产品层。
- Best for:
  WorkBuddy 适合“今天哪些股票”“这家公司现价/新闻/公告”的问题；
  OpenXQuant 适合“这条策略能否复现、是否有偏差”的问题。
- Risk:
  WorkBuddy 数据口径依赖远端服务；OpenXQuant 若接外部源，需要保持
  PIT、survivorship、hash 和数据 manifest。

[方法论 reference]
- Pros:
  WorkBuddy 的 46 个金融 reference 形成了可复用研究手册；
  OpenXQuant 的技能更偏流程门禁和产物生成。
- Cons:
  WorkBuddy reference 不能替代确定性测试；OpenXQuant 需要补更多
  投研场景方法论。
- Best for:
  把 WorkBuddy 的“按场景读取 reference”模式移植到 OpenXQuant 的
  report/review/performance/factor 解释阶段。
- Risk:
  如果 reference 太多且无检索索引，Agent 可能读错、漏读或上下文膨胀。

[可执行 helper 脚本]
- Pros:
  WorkBuddy 的 `run_signal.py` 统一封装 12 类技术/量化 engine，
  对 Agent 很友好；OpenXQuant 的组件注册和 component catalog 更规范。
- Cons:
  WorkBuddy helper 有动态依赖和启发式逻辑；OpenXQuant 组件作者门槛较高。
- Best for:
  OpenXQuant 可增加轻量 `oxq signal run` 或 `oxq component demo`
  入口，但底层仍使用组件 manifest 和测试。
- Risk:
  helper 若绕开 SPEC，就会回到不可复现脚本研究。

[插件与市场]
- Pros:
  WorkBuddy 有 `.codebuddy-plugin/plugin.json`、命名空间技能、市场、
  hooks、MCP 和 bin；OpenXQuant 有多 Agent 目标安装和 hash 管理。
- Cons:
  WorkBuddy 插件机制通用但攻击面更大；OpenXQuant 的安装器更安全但生态化弱。
- Best for:
  OpenXQuant 可以借鉴插件 manifest 与 marketplace，但保留
  source/hash/managed marker 和目标适配器。
- Risk:
  引入插件市场后，供应链、权限和运行沙箱必须同步设计。

[Hook 与权限]
- Pros:
  WorkBuddy 的 Hook 能在 UserPromptSubmit、PreToolUse 等生命周期
  插入校验；OpenXQuant 当前依赖技能/角色文本和 CLI gate。
- Cons:
  Hook 是 Beta 风格，且可能引入隐式行为；OpenXQuant 的 gate 更透明。
- Best for:
  OpenXQuant 可用 Hook 做额外提醒或阻断，例如禁止未授权 backtest、
  禁止修改 gated artifacts。
- Risk:
  Hook 若不记录到 artifact，会削弱可复现性。

[报告与产物]
- Pros:
  WorkBuddy 强调 HTML 研报、图表和 presentable artifact；
  OpenXQuant 有 `report_assets/manifest.json`、hash 和 Report QA。
- Cons:
  WorkBuddy 的 HTML 质量检查主要靠提示要求；OpenXQuant 的报告
  交互体验不如 WorkBuddy。
- Best for:
  两者结合：OpenXQuant 保持报告 QA 和 asset hash，同时学习 WorkBuddy
  的首屏结论、ECharts 风格和产物呈现体验。
- Risk:
  只追求漂亮 HTML 会掩盖数据证据不足。

[本地回测生成]
- Pros:
  WorkBuddy 快速生成脚本、数据和 HTML，用户马上能看到结果；
  OpenXQuant 先审计再跑，慢但可信。
- Cons:
  WorkBuddy 同策略两次产物结果差异大；OpenXQuant 门禁多，探索阶段不够轻。
- Best for:
  OpenXQuant 可增加“草稿/探索模式”，但必须清晰标记不可作为正式回测。
- Risk:
  如果草稿结果被当作正式研究，会造成误导。

## 直接照搬清单

以下建议可以直接进入 OpenXQuant 的需求池，不需要大改架构。

1. 金融场景数据源路由矩阵

- 将“自然语言搜索、结构化详情、全市场筛选、公开信息兜底”的路由表做成
  OpenXQuant 数据/研究技能的前置说明。
- 明确“查单只标的详情”和“找哪些标的”是不同入口。
- 对每个数据源写已知限制、市场覆盖、字段口径、时间口径。

2. 动态清单优先

- WorkBuddy 要求“有哪些标签/策略/指标”时必须执行 `--list`。
- OpenXQuant 应对组件、recipes、indicators、signals、rules、
  portfolios、data providers 采用同样规则。
- 不要让 Agent 凭文档样例列组件清单。

3. 场景 reference 包

- 复制 WorkBuddy 的结构，而不是复制文本。
- 在 OpenXQuant 增加面向投研解释的 reference：
  - 因子解释
  - 组合归因
  - 回撤诊断
  - 策略失效解释
  - 执行成本解释
  - 风险预算
  - 报告叙事规范
- 每个 reference 保持“核心目标、分析步骤、避坑、可执行工具”四段式。

4. 技能 frontmatter 扩展

- 在 `agent/skills/*/SKILL.md` 增加标准字段：
  - `version`
  - `allowed-tools`
  - `user-invocable`
  - `context`
  - `agent`
  - `tags`
  - `inputs`
  - `outputs`
  - `compatibility`
- 保持向后兼容；缺字段时按当前行为处理。

5. 统一 helper CLI 入口

- 学习 `run_signal.py` 的统一入口形式。
- OpenXQuant 可提供：
  - `oxq component list`
  - `oxq component inspect <name>`
  - `oxq signal run <signal> --input ...`
  - `oxq indicator run <indicator> --input ...`
- 但所有正式运行仍必须回到 SPEC、compiled plan 和 audit。

6. HTML 报告语法自检

- WorkBuddy 在金融报告中要求 HTML 内联 JS 自检。
- OpenXQuant 已有 Report QA，可补一个更明确的 HTML/JS 检查：
  - 检查 `<script>` 语法。
  - 检查 ECharts 容器、非空数据、图表尺寸。
  - 检查图片和 HTML 引用都来自 `report_assets/...`。

7. 技能创建/校验工具

- WorkBuddy 有 `skill-creator` 和 quick validate。
- OpenXQuant 可以增加 `oxq agent skill validate`，检查：
  - frontmatter schema
  - 目录名与 `name` 一致
  - required skills 是否存在
  - allowed tools 是否有效
  - 路径是否安全
  - 文档是否引用不存在文件

## 借鉴思路清单

以下更适合做成 OpenXQuant 的长期生态能力。

1. 插件市场与命名空间

- WorkBuddy 的插件技能用 `/plugin-name:skill` 避免冲突。
- OpenXQuant 可做 `oxq-plugin-name/component-name` 或
  `plugin:component` 命名空间。
- 组件、数据源、报告模板、Agent 技能可以共享同一个 plugin manifest。

2. Plugin manifest

- 借鉴 `.codebuddy-plugin/plugin.json`，但加入 OpenXQuant 必需字段：
  - component bundle hash
  - source hash
  - test hash
  - supported oxq versions
  - data access permissions
  - live trading permissions
  - sandbox profile

3. Hook 生命周期

- 可用于 pre-backtest、pre-live-order、post-report、stop-review。
- 所有 Hook 决策必须落 artifact，例如 `hook_decisions.json`。
- 不应只依赖隐式 Hook 阻断。

4. MCP app

- WorkBuddy 将 Ardot 做成 MCP app + webview。
- OpenXQuant 可把数据浏览、回测监控、报告图表检查做成 MCP/app panel。
- 首个候选：run directory inspector。

5. Ask/Craft/Expert 模式

- OpenXQuant 可借鉴三模式：
  - Ask：只解释，不写文件，不跑回测。
  - Plan：只做策略/实验方案。
  - Craft：执行已授权流程。
- 这可以降低用户误触正式回测或 live trading 的风险。

6. 产物呈现体验

- WorkBuddy 强调最终文件必须呈现给用户。
- OpenXQuant 可以在 CLI/Agent 输出中加入更清楚的 artifact index：
  - Markdown report
  - HTML report
  - chart manifest
  - audit summary
  - run directory

7. 多连接器生态

- WorkBuddy 宿主依赖覆盖飞书、钉钉、Slack、企微、QQ、COS、OpenTelemetry。
- OpenXQuant 不是通用办公助手，不必照搬。
- 但可以借鉴“通知/报告分发连接器”思路，例如把研究报告发到企业微信、
  Slack 或邮件。

## WorkBuddy 不足与风险

1. 混淆工具降低可审计性

- `westock-data/scripts/index.js` 约 4.5MB。
- `westock-tool/scripts/index.js` 约 3.1MB。
- 两者可通过 CLI 验证能力，但无法像 OpenXQuant Python 源码一样审计
  细节和数据口径。

2. 技能包依赖声明不足

- `westock-data` 和 `westock-tool` 的 `package.json` 没声明依赖。
- Python helper 直接依赖 `pandas/numpy/czsc/smartmoneyconcepts/openpyxl`
  等，但没有技能级 lockfile 或 requirements。
- `neodata` 脚本会尝试自动 `pip install requests`，这对可控性和复现不友好。

3. 回测规格未锁定

- 本地两个 WorkBuddy 目录对同一全球轮动策略生成不同脚本和不同结果。
- 一版结果总收益约 `26.37%`，另一版约 `41.57%`。
- 差异说明脚本生成过程缺少：
  - spec hash
  - 参数审计
  - 数据 manifest
  - 交易假设 manifest
  - compiled plan
  - artifact hash
  - runtime audit

4. 数据源外部且口径不透明

- `neodata` 和 `westock` 很适合快速查询。
- 但正式策略研究必须明确：
  - 数据来源
  - PIT 约束
  - 复权口径
  - 生存者偏差
  - 交易日历
  - 字段更新时间
  - 缺失值处理

5. Prompt 规则多，确定性 gate 少

- WorkBuddy 金融技能写了很多“必须/禁止/避坑”。
- 这些规则能改善 Agent 行为，但不等于运行时强校验。
- OpenXQuant 应学习其规则覆盖面，同时用 schema、hash、QA、audit
  实现确定性门禁。

6. 插件/Hook/MCP 攻击面更大

- 插件可带 bin、hooks、MCP server、settings。
- 这很强，也意味着供应链和权限风险高。
- OpenXQuant 若引入，需要先设计权限、签名、hash、沙箱和审计日志。

7. 产物漂亮但不一定可信

- WorkBuddy 可快速生成 HTML 报告。
- 但如果数据、脚本、交易假设没有锁定，漂亮报告可能放大错误结论。
- OpenXQuant 应保持“证据先于叙事”。

## 对 OpenXQuant 的改进路线

P0：直接改进，风险低

- 为 `open-xquant` router 增加更细的数据源/研究场景路由矩阵。
- 为 `review-performance`、`write-research-report`、
  `review-research-report` 增加场景 reference 包。
- 增加 `oxq component list/inspect` 和 Agent 动态清单规则。
- 增加 `oxq agent skill validate`。
- 在 Report QA 中补 HTML/JS 语法和图表非空检查。
- 为 live trading 和 broker 工具增加 tool-level confirmation 证据落盘。

P1：生态化改进

- 设计 OpenXQuant plugin manifest。
- 给技能 frontmatter 增加版本、工具权限、上下文、输入输出、tags。
- 建立用户/项目/本地 scope 的 skill/component 安装规范。
- 增加 namespaced components 和 namespaced skills。
- 增加 hook decision artifact。

P2：产品体验改进

- 做 run directory inspector 或 report viewer MCP app。
- 增加 Ask/Plan/Craft 模式映射。
- 增加“草稿回测”模式，但强制标记为不可用于正式结论。
- 增加报告分发连接器。

P3：数据能力改进

- 接入实时行情、宏观、新闻、公告、研报、筛选/排行数据源。
- 每个数据源必须有 provider manifest。
- 每次调用必须写 source、timestamp、query、field mapping、coverage、
  latency、raw hash。
- 正式策略仍只允许进入已审计 data manifest。

## 不建议照搬的部分

- 不照搬混淆 JS 作为核心策略/回测逻辑。
- 不照搬自动安装 Python 包的方式。
- 不照搬“Agent 临场写 backtest.py 后直接报告”的正式研究流程。
- 不照搬只靠提示词约束关键金融安全边界。
- 不照搬未落盘的 Hook 决策。
- 不照搬外部数据源结果而不记录 raw hash 和字段口径。

## 建议优先实现的 10 个任务

1. 创建 `agent/references/finance/`，先放 OpenXQuant 自有的策略复盘、
   回撤诊断、报告叙事、因子解释 reference。
2. 在 `agent/skills/open-xquant/SKILL.md` 中增加“动态清单优先”规则。
3. 新增 `oxq component list --kind indicator|signal|portfolio|rule|recipe`。
4. 新增 `oxq component inspect <name>`，输出参数、公式、输出域、依赖。
5. 扩展 skill frontmatter schema 和测试。
6. 新增 `oxq agent skill validate`。
7. 在 `src/oxq/report/qa.py` 增加 HTML script 检查。
8. 为 report charts 增加图像尺寸/非空/像素检查。
9. 设计 `open-xquant-plugin.json` 草案。
10. 设计 `hook_decisions.json` 草案，用于未来 Hook gate 落盘。

## 附录 A：WorkBuddy 精确路径

应用与资源：

- `/Applications/WorkBuddy.app`
- `/Applications/WorkBuddy.app/Contents/Info.plist`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked`
- `/Applications/WorkBuddy.app/Contents/Resources/vendor/node.tar.gz`
- `/Applications/WorkBuddy.app/Contents/Resources/vendor/python.dat`

内置技能：

- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/ardot-design-assistant`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/buddy-multimodal-generation`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/cloudstudio-deploy`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/expert-manager`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/marketplace-skill-installer`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/neodata-financial-search`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/skill-creator`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/wb-finance-skill`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/westock-data`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-skills/westock-tool`

插件：

- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins/weixinpay`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins/weixinpay/.codebuddy-plugin/plugin.json`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins/weixinpay/.mcp.json`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-plugins/weixinpay/hooks/hooks.json`

MCP app：

- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-mcp-apps/ardot-mcp-app/cli.cjs`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-mcp-apps/ardot-mcp-app/webview/app.html`
- `/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/resources/builtin-mcp-apps/_workbuddy-runtime/mcp-app-bootstrap.cjs`

用户数据：

- `/Users/daodao/Workbuddy`
- `/Users/daodao/Library/Application Support/@genie/workbuddy-desktop`
- `/Users/daodao/Library/Application Support/CodeBuddyExtension/Data/Public/auth/workbuddy-desktop.info`
- `/Users/daodao/Library/Logs/WorkBuddy`
- `/Users/daodao/Library/HTTPStorages/com.workbuddy.workbuddy`

## 附录 B：OpenXQuant 对照路径

技能与角色：

- `agent/skills/open-xquant/SKILL.md`
- `agent/skills/build-strategy-spec/SKILL.md`
- `agent/skills/audit-strategy-spec/SKILL.md`
- `agent/skills/audit-runtime-semantics/SKILL.md`
- `agent/skills/run-authorized-backtest/SKILL.md`
- `agent/skills/monitor-strategy-run/SKILL.md`
- `agent/skills/build-report-charts/SKILL.md`
- `agent/skills/write-research-report/SKILL.md`
- `agent/skills/review-research-report/SKILL.md`
- `agent/roles/oxq-coordinator.md`
- `agent/roles/oxq-runner-worker.md`
- `agent/roles/oxq-report-writer-worker.md`

代码：

- `src/oxq/cli/agent.py`
- `src/oxq/cli/agent_targets.py`
- `src/oxq/cli/agent_manifest.py`
- `src/oxq/core/registry.py`
- `src/oxq/core/component_manifest.py`
- `src/oxq/core/component_catalog.py`
- `src/oxq/spec/runtime_audit_schema.py`
- `src/oxq/audit/reproducibility.py`
- `src/oxq/report/qa.py`
- `src/oxq/tools/registry.py`
- `src/oxq/tools/live.py`

测试：

- `tests/cli/test_agent_manifest.py`
- `tests/cli/test_agent_install.py`
- `tests/agent/test_report_chart_builder_skill.py`
- `tests/report/test_qa.py`

## 最后三行

- Decision: OpenXQuant 应吸收 WorkBuddy 的产品化技能生态，但保留自身可审计内核。
- Why: WorkBuddy 强在场景覆盖和交互效率，OpenXQuant 强在复现、审计和门禁。
- Next step: 先实现 P0 的 reference 包、动态清单、skill validate 和报告 QA 增强。
