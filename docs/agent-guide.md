# Agent Guide - open-xquant 安装

Round 26 current schema registry: decision schema `5`, candidate/policy/comparison schema `3`,
and lineage `3`; pointer validation rejects decision schema `4`. Confirmation
events use closed user/coordinator provenance and raw-line hashes. Historical
refresh is `write -> review -> lineage -> prepare new selection -> comparison ->
resume` with fresh selection and candidate hashes. The default chart pack uses
canonical requested equality, closed skip reasons, and unsealed chart retries.

本文档只说明如何把 open-xquant 的长期能力安装到 Agent 环境中。
研究流程、任务路由和写报告规则由已安装的 `open-xquant router skill`
以及更具体的 `agent/skills/*/SKILL.md` 负责。

安装完成后，Agent 遇到量化研究、回测、因子、调参、审计、报告、
图表资产、SDK 开发或实盘连接任务时，应先加载 `open-xquant` skill，
再由它路由到更具体的 skill。不要把本文档当成研究 workflow。

版本化研究路径必须先从 `.open-xquant/workspace.yaml` 的
`paths.versions_dir` 解析 `version_root`；仅当该键缺失时才默认使用
`versions`。该值必须是解析后仍位于 workspace 内的安全相对路径。
随后读取 `<version_root>/<version_id>/version_manifest.json`，所有阶段
输入输出都使用其中对应的 `phase_paths`，而不是自行拼接默认目录。
例如自定义 `research_versions` 根时，spec build 必须落到
`research_versions/v003/04_spec_build`。

workspace-local 组件路径必须独立从 `paths.components_dir` 解析
`components_dir`；only when that key is absent 才默认使用 `components`。
该值必须是 safe relative path，且解析后的目标 stays inside the workspace。
后续阶段统一使用 `<components_dir>`，不得重新拼接默认组件根。

边界原则：

- `oxq` CLI / SDK / Tools 只做确定性 primitives，例如 validate、
  compile、backtest、audit、robustness、报告文件与资产完整性 QA、
  asset manifest、workspace init、agent install/status。
- 需要上下文综合判断的任务必须留在 skill / Agent 层，例如 spec 字段
  来源追溯、是否继续未完成实验、图表选择、报告写作、实验差异解释、
  报告数值叙事是否成立、是否标记某个 run 为 final。
- 不要把报告写作、实验结论、图表叙事或来源判断下沉成 CLI 模板。
  `oxq report qa` 不是语义审稿器，不负责证明所有数值结论都合理。

---

## 1. 前置检查

在 open-xquant 源码根目录运行：

```bash
pwd
python --version
uv --version
test -f pyproject.toml && test -d src/oxq && echo "open-xquant repo"
```

根目录必须包含 `pyproject.toml`、`src/oxq/` 和 `agent/skills/`。

---

## 2. 长期能力安装

安装到所有支持的 Agent：

```bash
uv run oxq agent install --all-targets
uv run oxq agent status
```

安装器会询问本机使用哪种 Agent profile：

- `multi-agent`: 推荐给支持 multi-Agent / subagent 的 Agent。安装窄
  skill 和预制 open-xquant worker 角色，不安装
  `strategy-builder-standalone`、`quant-research` 等端到端入口。
- `standalone-agent`: 给单 Agent 顺序编排同一组窄 skill 的环境使用；不安装
  预制 worker 角色，也不安装端到端 workflow skill。

非交互安装可以显式指定：

```bash
uv run oxq agent install --all-targets --profile multi-agent --yes
uv run oxq agent install --all-targets --profile standalone-agent --yes
```

如果只传 `--yes` 且未指定 `--profile`，安装器会按目标 Agent 能力使用推荐
profile；对 Codex、OpenCode、Claude Code、Cursor 默认推荐
`multi-agent`。对当前未确认官方 subagent 角色目录的目标，默认使用
`standalone-agent`。

只安装某个目标：

```bash
uv run oxq agent install --target codex --profile multi-agent --yes
uv run oxq agent install --target opencode --profile multi-agent --yes
uv run oxq agent install --target claude-code --profile multi-agent --yes
uv run oxq agent install --target cursor --profile multi-agent --yes
uv run oxq agent install --target openclaw --profile standalone-agent --yes
uv run oxq agent install --target trae --profile standalone-agent --yes
```

安装位置：

- Codex: `${CODEX_HOME:-~/.codex}/skills/` 和
  `${CODEX_HOME:-~/.codex}/AGENTS.md`；multi-agent profile 还会安装
  `${CODEX_HOME:-~/.codex}/agents/*.toml`
- OpenCode: `~/.config/opencode/skills/` 和
  `~/.config/opencode/AGENTS.md`；multi-agent profile 还会安装
  `~/.config/opencode/agents/*.md`
- Claude Code: `~/.claude/skills/` 和 `~/.claude/CLAUDE.md`；
  multi-agent profile 还会安装 `~/.claude/agents/*.md`
- Cursor: `~/.cursor/skills/`；multi-agent profile 还会安装
  `~/.cursor/agents/*.md`
- OpenClaw: `~/.openclaw/skills/`
- TRAE: `~/.trae/skills/`

预制 multi-agent 角色：

- `oxq-coordinator`: 面向用户的主控 Agent，只路由阶段和管理确认。
- `oxq-version-manager-worker`: 使用 `manage-strategy-version`，只判断用户变更
  是否延续当前版本或创建新版本，并维护 version lineage。
- `oxq-artifact-governor-worker`: 使用 `govern-research-workspace`，只审查
  workspace 布局、root-level 污染和 phase handoff。
- `oxq-strategy-brainstorm-worker`: 使用 `brainstorm-strategy-idea`，引导
  用户按阶段说清楚策略描述，输出 `strategy_idea_brief.json`。
- `oxq-strategy-idea-auditor-worker`: 使用 `audit-strategy-idea`，检查
  brainstorm 是否按阶段解释、询问、拉回和确认，输出
  `strategy_idea_audit.json`。
- `oxq-strategy-builder-worker`: 使用 `build-strategy-spec`，只从通过审核的
  idea artifacts 构建和验证 `strategy_spec.yaml`。
- `oxq-data-inspection-worker`: 使用 `explore-data`，只检查数据可用性、
  provider readiness、parquet 质量和覆盖区间。
- `oxq-component-author-worker`: 使用 `author-component`，只创建
  workspace-local Indicator、Signal、PortfolioOptimizer components、测试、
  manifest 和 catalog；workspace-local Rule 默认阻塞。
- `oxq-spec-auditor-worker`: 使用 `audit-strategy-spec`，只审用户确认、字段来源
  和组件 provenance，并校准 SPEC 是否忠实映射已审核 idea。
- `oxq-runtime-auditor-worker`: 使用 `audit-runtime-semantics`，只编译并审核
  runtime semantics。
- `oxq-runner-worker`: 使用 `run-authorized-backtest`，只在授权后运行
  formal backtest 并写 `runner_result.json`。
- `oxq-monitor-worker`: 使用 `monitor-strategy-run`，只做跑后
  reproducibility、research audit、robustness 和 experiment registry。
- `oxq-lineage-auditor-worker`: 使用 `audit-artifact-lineage`，只审计
  version/run/final 引用、hash 和最终候选资格。
- `oxq-experiment-comparator-worker`: 使用 `compare-strategy-versions` 和
  `compare-experiments`，只做可比性审计和版本/run 对比。
- `oxq-final-selector-worker`: 使用 `select-final-version`，只在用户确认后
  标记最终版本。
- `oxq-report-writer-worker`: 使用 `build-report-charts` 和
  `write-research-report`，只写 immutable report revision、图表和报告。
- `oxq-report-reviewer-worker`: 使用 `review-research-report`，只输出
  immutable `reviews/<review_revision_id>/report_review.json`。

Report publication contract: chart assets, scripts, report manifests,
Markdown, HTML, writer results, and review results use
`publish_report_artifacts(report_dir, artifacts, *, lock_subject=None)`. The
mapping contains safe relative keys and complete `bytes`; `None` deletes a
target. A callable builder executes under the final-selection lock, performs a
baseline check, and commits one atomic all-or-rollback batch. Direct path
writes, shell redirection, and report asset CLI publication paths are invalid.
For exports outside the governed workspace, pass
`lock_subject=source_run_dir`. If report work needs coherent run locking, wrap
the publication with `run_digest_transaction(source_run_dir)`; runtime acquires
the run lock first and the final-selection lock second. Agent code must not
pre-acquire either publisher lock.

Other governed Agent publishers continue to use the shared final-selection
lock protocol. Selector pointer publication performs only direct byte snapshots,
the unchanged-byte sweep, and atomic pointer replacement while holding that
lock; it does not invoke a run-locking validator.

Final-selection comparison evidence is immutable and selection-scoped at
`<comparisons_dir>/<selection_id>/<comparison_id>/`. Existing output
directories are collisions, never update targets. A remediable retry keeps the
same selection and uses a fresh `comparison_id`; `restart_selection` allocates
a fresh selection directory and fresh comparison scope. Evidence reachable
from a prior `current_final.json` is never overwritten. Current manifests use
schema version 3; schema version 2 is historical recognition only.

Current report evidence is revision-scoped: sealed candidates live below
`10_reports/<run_id>/candidates/<report_revision_id>/`, and semantic reviews
live below `10_reports/<run_id>/reviews/<review_revision_id>/`. Final selection
binds both exact paths and hashes; historical repair creates fresh revisions
without changing the active version.

Version/bootstrap and governance batches use one recovery journal at
`<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json`.
The journal contains baselines, staged hashes, durable backup hashes,
replacement order, and `prepared -> committing -> committed`. The publisher
acquires `workspace-governance.lock`, then `final-selection.lock` last, performs
unchanged-byte checks, and holds both through recovery and fsync. Before the
first replacement recovery discards staging; after a non-pointer replacement
it must roll back; after `current.json` replacement it may roll forward only
when all exact staged bytes are present, otherwise it must roll back the whole
transaction.

Pointer publication must `fsync(<final_dir>)` after atomic replacement. A
post-rename directory-sync failure means publication outcome is indeterminate
and must not claim that the prior pointer is unchanged. Recover under
`final-selection.lock`: exact new pointer bytes are revalidated and synced,
exact prior pointer bytes are retried, and any other bytes block as corruption.
Parent fsync is required. `final_decision.json` is the sole canonical decision
artifact.

这些角色的单一来源是 `agent/roles/*.md`。安装器会按目标 Agent 的官方
格式渲染：Codex 使用 TOML custom agents；OpenCode、Claude Code、
Cursor 使用 Markdown agent files。没有官方确认 subagent 角色目录的目标
只安装 skills，不安装这些角色。

OpenCode 角色显式使用 `edit: ask` 和 `bash: ask`。安装时无法知道当前
workspace 解析后的配置根和 active version，因此不会用 basename glob
静默放行同名文件。custom `paths.conversations_dir` 仍可使用，但每个具体
编辑或命令都必须经过用户批准；该批准 does not expand 角色在
`agent/roles/*.md` 中声明的 ownership。

OpenCode 角色也显式设置 `permission.task`。worker 使用 `"*": deny`，
不能继续委派；coordinator 先 deny `*`，再 allow exact managed worker names，
不会放行 `general`、`explore` 或 worker wildcard。OpenCode 的 `--auto`
automatically approves permission requests that are not explicitly denied；
因此它会自动批准 `edit: ask` 和 `bash: ask`，但 explicit `deny` remains
enforced。只有用户明确接受该 session-wide 授权边界时才应启用；默认交互
模式会对具体 `ask` 操作逐次请求批准。

各目标安装的 skill 是平级目录，例如：

```text
<agent-skill-root>/
  open-xquant/
    SKILL.md
  brainstorm-strategy-idea/
    SKILL.md
  audit-strategy-idea/
    SKILL.md
  build-strategy-spec/
    SKILL.md
  write-research-report/
    SKILL.md
```

不要把具体 skill 嵌套到 `open-xquant/` 下。Agent Skills 标准和各
Agent 实现都以每个 `SKILL.md` 的 frontmatter `name` 和 `description`
作为发现与触发信息；层级关系由 `open-xquant` router skill 在正文中
表达。

多 skill 包也按这种方式组织：一个安装动作可以放置多个平级 skill
目录，但每个目录仍是独立发现单元。类似 superpowers 这类能力包也是由
一组平级 skill 组成，而不是把子 skill 嵌进一个父 skill 目录。
`open-xquant` 采用同样模式：`open-xquant` 是入口 router skill，
`build-strategy-spec`、`write-research-report` 等是平级 leaf skills。

---

## 3. SDK bundle 和 runner

具体目标安装会构建 cached SDK bundle：

```text
~/.config/open-xquant/sdk-bundles/<bundle_id>/
```

该 SDK bundle 包含：

- open-xquant wheel。
- `full-research` dependency lock。
- 安装过 open-xquant 和依赖的 runner venv。
- 供后续研究项目复用的 uv cache。

默认 `full-research` profile 会安装 `pyproject.toml` 中除 `dev`、
`docs`、`talib` 以外的 optional extras。当前版本包含 `agent`、
`akshare`、`chart`、`live`、`mcp`、`scipy`、`tushare`、`yfinance`。

### A 股数据源

Agent 可使用 AkShare 或 Tushare 下载 A 股数据。使用 Tushare 前先安装
可选依赖并配置凭据：

```bash
uv sync --extra tushare
export TUSHARE_TOKEN="your-token"
```

```python
from oxq.data import TushareDownloader

downloader = TushareDownloader()  # 首次使用时读取 TUSHARE_TOKEN
path = downloader.download("600519.SH", "2024-01-01", "2024-12-31")

# 显式构造参数 token 的优先级高于环境变量。
explicit = TushareDownloader(token="your-token")
```

Tushare 首版仅支持 A 股日线，代码必须完全匹配
`^[0-9]{6}\.(SH|SZ|BJ)$`：六位数字加大写交易所后缀。`end` 包含端点。
前复权（qfq）公式为
`raw_price * row_adj_factor / reference_adj_factor`；参考因子取包含端点的
`end` 当日或之前的最新有效复权因子，它可以独立于最后一条日线交易记录。
`volume` 单位为股。权限、积分和限流由 Tushare 平台决定。
Tushare `daily` 单次最多返回 6000 行。下载器会把长区间自动切成每块最多
3650 个包含端点的日历日，并让 `daily` 与 `adj_factor` 使用完全相同、无
重叠且无缺口的边界；短区间仍只请求一次。任一分块响应达到 6000 行时会在
写入前失败，避免接受可能被截断的数据；成功 manifest 仍记录用户请求的
完整范围。Agent 绝不能
打印或存储 token；open-xquant 不会将 token 持久化到本地状态、日志、异常
或产物。凭据传输由上游 Tushare SDK 控制，其当前官方客户端使用 HTTP，
Agent 应将其视为需要用户自行评估的上游安全边界。

下载后应把标准 Parquet 作为审计运行的输入，并继续使用
`data.provider: local`，不能把 provider 设置为 `tushare`。

安装完成后，`~/.config/open-xquant/agent.yaml` 会记录：

```yaml
agent_profile: multi-agent
preferred_runner: /path/to/sdk-bundles/<bundle>/runner/.venv/bin/oxq
preferred_runner_argv:
  - /path/to/sdk-bundles/<bundle>/runner/.venv/bin/oxq
```

`preferred_runner` 默认指向 cached runner，而不是最初安装时的源码目录。
用户删除最初下载的 open-xquant 源码后，Agent 仍应使用 cached runner。

如果当前目录是 open-xquant 源码 worktree，或用户明确要求在当前
worktree 中复现/开发，优先使用 current worktree runner，例如
`uv run oxq` 或 `uv run --project . oxq`。不要无脑读取全局
`preferred_runner` 去调用另一个 checkout。

如果 `agent.yaml` 缺失或 runner 失败，读取
`~/.config/open-xquant/agent-install.json`，使用其中的
`sdk_bundle.runner.argv` 或 `sdk_bundle.runner.oxq`。
不要默认回到最初安装时的源码路径；用户可能已经删除那个目录。

---

## 4. skill 单一来源

仓库内的 skill 单一来源是 `agent/skills/*/SKILL.md`。`oxq agent install`、
`oxq agent upgrade` 和各 Agent 的长期安装都从这个目录读取 skill。

不要维护 `agent/opencode/` 这样的 target-specific 源码包，也不要在仓库中
保留 OpenCode 专用 skills、agents 或 commands 副本。OpenCode 长期安装由
`oxq agent install --target opencode` 从 `agent/skills/<name>/SKILL.md` 和
`agent/roles/*.md` 渲染到 `~/.config/opencode/`。

如果开发者临时需要让 OpenCode 直接读取当前源码工作区，可以在自己的
OpenCode 配置中指向 `agent/skills`，但这个配置不作为仓库内运行包维护。

---

## 5. 升级、修复和卸载

修复当前安装：

```bash
uv run oxq agent install --repair --yes
```

从 GitHub 更新长期能力：

```bash
uv run oxq agent upgrade --all-targets --yes
```

从本地开发 checkout 更新：

```bash
uv run oxq agent upgrade --all-targets --from-local . --yes
```

卸载长期能力：

```bash
uv run oxq agent uninstall --all-targets --yes
```

同时删除 open-xquant Agent 配置和 managed SDK bundle：

```bash
uv run oxq agent uninstall --all-targets --purge-config --yes
```

安全边界：

- `uninstall` 只删除 manifest 记录且带 managed marker 的 skill 目录。
- `uninstall` 只删除 manifest 记录且 hash 未被用户修改的 managed role
  files。
- 不删除 `~/.oxq/data`。
- 不删除任何研究目录、`runs/`、`reports/` 或 `experiments.jsonl`。
- 只有 `--purge-config` 才会删除
  `~/.config/open-xquant/agent-install.json`、`agent.yaml` 和 manifest
  记录的 managed SDK bundle。

---

## 6. 安装状态检查

```bash
uv run oxq agent status
uv run oxq agent status --json
```

安装成功时应能看到：

- 目标 Agent 的 skill 目录中存在 `open-xquant/SKILL.md`。
- multi-agent profile 下，支持 subagent 的目标会显示并安装
  `agent_roles`。
- `agent-install.json` 记录安装 target、skill manifest 和 SDK bundle。
- `agent.yaml` 记录 cached `preferred_runner`。
- 对支持 instructions 文件的 Agent，managed block 只负责引导 Agent
  先使用 `open-xquant` skill。

如果需要检查当前研究目录环境，使用：

```bash
<preferred_runner> doctor --json
```

`doctor` 输出 workspace missing 时，由 `open-xquant` skill 和对应 leaf
skill 决定是否需要运行 `research init` 或 `research init --sdk`。

---

## 7. Workspace-local custom components

当 builder 输出 `needs_custom_component` 时，multi-agent 编排应调用
`oxq-component-author-worker`，而不是让 builder 写组件代码。
workspace-local custom Rule 当前不属于普通 authoring 能力；如果需要 Rule，
应阻塞并要求用户明确是否进入 open-xquant 框架开发。

组件 authoring 阶段写入已解析的组件根：

```text
<components_dir>/bundles/<bundle_id>/custom_components/
<components_dir>/bundles/<bundle_id>/component_manifest.json
<components_dir>/bundles/<bundle_id>/component_catalog.json
<phase_paths.03_component_authoring>/result.json
```

后续确定性命令通过 manifest 加载组件：

```bash
uv run oxq component-manifest validate <components_dir>/bundles/<bundle_id>/component_manifest.json
uv run oxq registry export \
  --component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json \
  --out <phase_paths.04_spec_build>/component_catalog.json
uv run oxq spec validate <phase_paths.04_spec_build>/strategy_spec.yaml \
  --component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json
uv run oxq strategy compile <phase_paths.04_spec_build>/strategy_spec.yaml \
  --component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json \
  --out <phase_paths.07_compile_preview>
# oxq-coordinator must first write:
# <phase_paths.08_runtime_audit>/backtest_authorization.json
# oxq-runner-worker then uses run-authorized-backtest.
uv run oxq backtest run <phase_paths.04_spec_build>/strategy_spec.yaml \
  --component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json \
  --spec-audit <phase_paths.06_spec_audit>/spec_audit.json \
  --runtime-audit <phase_paths.08_runtime_audit>/runtime_audit.json \
  --component-catalog <phase_paths.04_spec_build>/component_catalog.json \
  --out <phase_paths.09_backtests> \
  --json
```

`component_manifest.json` 的 `bundle_hash` 覆盖 manifest 内容
（排除 `bundle_hash` 字段本身）、component source、tests 和 extension
metadata。后续阶段必须使用同一个 manifest hash，不能静默加载不同的组件
bundle。
