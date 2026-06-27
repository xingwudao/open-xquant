# Agent Guide - open-xquant 安装

本文档只说明如何把 open-xquant 的长期能力安装到 Agent 环境中。
研究流程、任务路由和写报告规则由已安装的 `open-xquant router skill`
以及更具体的 `agent/skills/*/SKILL.md` 负责。

安装完成后，Agent 遇到量化研究、回测、因子、调参、审计、报告、
图表资产、SDK 开发或实盘连接任务时，应先加载 `open-xquant` skill，
再由它路由到更具体的 skill。不要把本文档当成研究 workflow。

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
  skill 和预制 OpenXQuant worker 角色，不安装
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
- `oxq-strategy-builder-worker`: 使用 `build-strategy-spec`，只构建和验证
  `strategy_spec.yaml`。
- `oxq-data-inspection-worker`: 使用 `explore-data`，只检查数据可用性、
  provider readiness、parquet 质量和覆盖区间。
- `oxq-component-author-worker`: 使用 `author-component`，只创建
  workspace-local Indicator、Signal、PortfolioOptimizer components、测试、
  manifest 和 catalog；workspace-local Rule 默认阻塞。
- `oxq-spec-auditor-worker`: 使用 `audit-strategy-spec`，只审用户确认、字段来源
  和组件 provenance。
- `oxq-runtime-auditor-worker`: 使用 `audit-runtime-semantics`，只编译并审核
  runtime semantics。
- `oxq-runner-worker`: 使用 `run-authorized-backtest`，只在授权后运行
  backtest 和跑后确定性检查。
- `oxq-report-writer-worker`: 使用 `build-report-charts` 和
  `write-research-report`，只写图表和报告。
- `oxq-report-reviewer-worker`: 使用 `review-research-report`，只输出
  `report_review.json`。

这些角色的单一来源是 `agent/roles/*.md`。安装器会按目标 Agent 的官方
格式渲染：Codex 使用 TOML custom agents；OpenCode、Claude Code、
Cursor 使用 Markdown agent files。没有官方确认 subagent 角色目录的目标
只安装 skills，不安装这些角色。

各目标安装的 skill 是平级目录，例如：

```text
<agent-skill-root>/
  open-xquant/
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
`akshare`、`chart`、`live`、`mcp`、`scipy`、`yfinance`。

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
应阻塞并要求用户明确是否进入 OpenXQuant 框架开发。

组件 authoring 阶段默认写入：

```text
custom_components/
component_manifest.json
component_catalog.json
result.json
```

后续确定性命令通过 manifest 加载组件：

```bash
uv run oxq component-manifest validate component_manifest.json
uv run oxq registry export \
  --component-manifest component_manifest.json \
  --out component_catalog.json
uv run oxq spec validate strategy_spec.yaml \
  --component-manifest component_manifest.json
uv run oxq strategy compile strategy_spec.yaml \
  --component-manifest component_manifest.json \
  --out compile_preview
uv run oxq backtest run strategy_spec.yaml \
  --component-manifest component_manifest.json \
  --spec-audit spec_audit.json \
  --runtime-audit runtime_audit.json \
  --component-catalog component_catalog.json
```

`component_manifest.json` 的 `bundle_hash` 覆盖 manifest 内容
（排除 `bundle_hash` 字段本身）、component source、tests 和 extension
metadata。后续阶段必须使用同一个 manifest hash，不能静默加载不同的组件
bundle。
