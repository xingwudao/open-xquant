# Agent Guide - open-xquant 安装

本文档只说明如何把 open-xquant 的长期能力安装到 Agent 环境中。
研究流程、任务路由和写报告规则由已安装的 `open-xquant router skill`
以及更具体的 `agent/skills/*.md` 负责。

安装完成后，Agent 遇到量化研究、回测、因子、调参、审计、报告、
图表资产、SDK 开发或实盘连接任务时，应先加载 `open-xquant` skill，
再由它路由到更具体的 skill。不要把本文档当成研究 workflow。

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
uv run oxq agent install --all-targets --yes
uv run oxq agent status
```

只安装某个目标：

```bash
uv run oxq agent install --target codex --yes
uv run oxq agent install --target opencode --yes
uv run oxq agent install --target claude-code --yes
uv run oxq agent install --target cursor --yes
uv run oxq agent install --target openclaw --yes
uv run oxq agent install --target trae --yes
```

安装位置：

- Codex: `${CODEX_HOME:-~/.codex}/skills/` 和
  `${CODEX_HOME:-~/.codex}/AGENTS.md`
- OpenCode: `~/.config/opencode/skills/` 和
  `~/.config/opencode/AGENTS.md`
- Claude Code: `~/.claude/skills/` 和 `~/.claude/CLAUDE.md`
- Cursor: `~/.cursor/skills/`
- OpenClaw: `~/.openclaw/skills/`
- TRAE: `~/.trae/skills/`

各目标安装的 skill 是平级目录，例如：

```text
<agent-skill-root>/
  open-xquant/
    SKILL.md
  strategy-builder/
    SKILL.md
  research-report-writer/
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
`strategy-builder`、`research-report-writer` 等是平级 leaf skills。

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
不要默认回到 `source.path`；用户可能已经删除最初的源码目录。

---

## 4. skill 单一来源

仓库内的 skill 单一来源是 `agent/skills/*.md`。`oxq agent install`、
`oxq agent upgrade` 和各 Agent 的长期安装都从这个目录读取 skill。

不要维护 `agent/opencode/skills/` 这样的第二份 skill 副本。

OpenCode 集成通过 `agent/opencode/opencode.json` 的 `skills.paths`
从 workspace root 加载 `agent/skills/`。运行 OpenCode 时，当前工作目录
应是包含 `agent/` 的工作区根目录。为了符合 OpenCode 的
`<name>/SKILL.md` 发现规则，`agent/skills/<name>/SKILL.md` 是真实
wrapper 适配器，包含发现用 frontmatter 和读取 canonical
`../<name>.md` 的指令，但不能复制出第二份完整 skill 内容。

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
