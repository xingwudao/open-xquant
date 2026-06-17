# Agent Lifecycle Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox syntax for tracking.

**Goal:** Make first-time `open-xquant` setup persistent so users can later
open any new research directory and type a strategy idea directly.

**Architecture:** Add an agent lifecycle management layer with manifest-backed
install, uninstall, upgrade, status, workspace init, and doctor checks. Global
Agent skills live in the user's Agent home, while each research directory keeps
only lightweight workspace config and a small `AGENTS.md` marker block.

**Tech Stack:** Python 3.12+, Click, stdlib `json`, `yaml`, `pathlib`,
`hashlib`, `tempfile`, `shutil`, pytest, uv.

---

## Scope

Implement these CLI entries:

```bash
oxq agent install
oxq agent uninstall
oxq agent upgrade
oxq agent status
oxq research init
oxq doctor
```

MVP targets:

- `codex`
- `opencode`
- `claude-code`
- `cursor`
- `openclaw`
- `generic`

Non-MVP targets:

- GUI management screens
- automatic data deletion
- complex three-way merge
- hosted team policy rollout

## Design Principles

- All global Agent writes must be reversible.
- `uninstall` must only delete paths recorded in the manifest.
- Every installed skill directory must contain a managed marker file.
- Managed instruction text must live inside marker blocks.
- No command should delete user research workspaces or market data.
- `doctor --json` should be suitable for an Agent to parse and act on.
- Re-running install, uninstall, upgrade, or research init must be idempotent.

## File Structure

Create or modify these files:

- Create: `src/oxq/cli/agent.py`
  - Agent lifecycle commands and helper orchestration.
- Create: `src/oxq/cli/research.py`
  - Research workspace initialization command.
- Create: `src/oxq/cli/doctor.py`
  - Environment, Agent install, workspace, data, and dependency checks.
- Create: `src/oxq/cli/agent_manifest.py`
  - Manifest/config schemas, path normalization, atomic JSON/YAML writes.
- Create: `src/oxq/cli/agent_targets.py`
  - Target adapters for `codex`, `opencode`, `claude-code`,
    `cursor`, `openclaw`, and `generic`.
- Modify: `src/oxq/cli/main.py`
  - Register `agent`, `research`, and `doctor` commands.
- Test: `tests/cli/test_agent_manifest.py`
- Test: `tests/cli/test_agent_install.py`
- Test: `tests/cli/test_agent_uninstall.py`
- Test: `tests/cli/test_agent_upgrade.py`
- Test: `tests/cli/test_research_init.py`
- Test: `tests/cli/test_doctor.py`
- Docs after implementation:
  - Modify: `docs/agent-guide.md`
  - Mention one-time `oxq agent install`, repeat-use `oxq research init`,
    and `oxq doctor`.

Do not create new top-level repo directories.

## Global Paths

Global config directory:

```text
~/.config/open-xquant/
  agent.yaml
  agent-install.json
  cache/
    open-xquant/
```

Default data directory:

```text
~/.oxq/data/
```

Codex target paths:

```text
~/.agents/skills/
${CODEX_HOME:-~/.codex}/AGENTS.md
```

OpenCode target paths:

```text
~/.config/opencode/skills/
~/.config/opencode/AGENTS.md
```

Claude Code target paths:

```text
~/.claude/skills/
~/.claude/CLAUDE.md
```

Cursor target paths:

```text
~/.cursor/skills/
~/.cursor/rules/
```

OpenClaw target paths:

```text
~/.openclaw/skills/
~/.openclaw/openclaw.json
```

Research workspace paths:

```text
<research-dir>/
  .open-xquant/
    workspace.yaml
  AGENTS.md
  strategy_specs/
  runs/
  reports/
  experiments.jsonl
```

## Official Agent Support Matrix

The target adapters must follow the official Agent surfaces below.

Codex:

- Official source:
  - `https://developers.openai.com/codex/skills`
  - `https://developers.openai.com/codex/guides/agents-md`
- Skill surface:
  - Codex reads skill directories containing `SKILL.md`.
  - User-scope skills live under `~/.agents/skills`.
  - Repo-scope skills live under `.agents/skills`.
  - Skills require `name` and `description` frontmatter.
- Instruction surface:
  - Global instructions live in `${CODEX_HOME:-~/.codex}/AGENTS.md`.
  - Project instructions live in `AGENTS.md`.
  - Codex reads global guidance before project guidance.
- Installer decision:
  - Install managed skills to `~/.agents/skills/<name>/`.
  - Install global bootstrap text to `${CODEX_HOME:-~/.codex}/AGENTS.md`.

OpenCode:

- Official source:
  - `https://opencode.ai/docs/skills/`
  - `https://opencode.ai/docs/rules/`
- Skill surface:
  - Native global skills live under `~/.config/opencode/skills`.
  - Native project skills live under `.opencode/skills`.
  - OpenCode also scans `.claude/skills`, `~/.claude/skills`,
    `.agents/skills`, and `~/.agents/skills`.
  - `SKILL.md` frontmatter recognizes `name`, `description`,
    `license`, `compatibility`, and `metadata`.
- Instruction surface:
  - Global rules live in `~/.config/opencode/AGENTS.md`.
  - Project rules live in `AGENTS.md`.
  - `CLAUDE.md` is only a compatibility fallback when `AGENTS.md`
    is absent.
- Installer decision:
  - Install managed skills to
    `~/.config/opencode/skills/<name>/`.
  - Install global bootstrap text to `~/.config/opencode/AGENTS.md`.

Claude Code:

- Official source:
  - `https://code.claude.com/docs/en/skills`
  - `https://code.claude.com/docs/en/memory`
  - `https://code.claude.com/docs/en/claude-directory`
- Skill surface:
  - Personal skills live under `~/.claude/skills/<skill-name>/SKILL.md`.
  - Project skills live under `.claude/skills/<skill-name>/SKILL.md`.
  - Plugin skills live under `<plugin>/skills/<skill-name>/SKILL.md`.
  - Claude can use skills automatically or via `/skill-name`.
- Instruction surface:
  - User instructions live in `~/.claude/CLAUDE.md`.
  - Project instructions live in `./CLAUDE.md` or `./.claude/CLAUDE.md`.
  - Claude Code reads `CLAUDE.md`, not `AGENTS.md`.
  - To share project `AGENTS.md`, write `@AGENTS.md` inside `CLAUDE.md`.
- Installer decision:
  - Install managed skills to `~/.claude/skills/<name>/`.
  - Install global bootstrap text to `~/.claude/CLAUDE.md`.

Cursor:

- Official source:
  - `https://cursor.com/docs/skills`
  - `https://cursor.com/docs/rules`
- Skill surface:
  - Project skills live under `.cursor/skills`.
  - User skills live under `~/.cursor/skills`.
  - Cursor also documents `.agents/skills` skill layout.
  - Skills are automatically applied when the agent decides they are
    relevant.
  - `disable-model-invocation: true` makes a skill behave like an
    explicit slash-command style skill.
- Instruction surface:
  - Project rules live in `.cursor/rules` as `.mdc` files.
  - `AGENTS.md` is supported in the project root and subdirectories.
  - User Rules are configured in Cursor Settings and apply to Agent Chat.
  - Remote GitHub rules are imported through Cursor Settings.
- Installer decision:
  - Install managed skills to `~/.cursor/skills/<name>/`.
  - Do not write Cursor User Rules directly, because the official user-rule
    surface is UI-managed.
  - For global bootstrap, install a managed user skill whose description
    triggers on quant strategy research.

OpenClaw:

- Official source:
  - `https://docs.openclaw.ai/tools/skills`
  - `https://docs.openclaw.ai/cli/agent`
  - `https://docs.openclaw.ai/`
- Skill surface:
  - Workspace skills live under `<workspace>/skills`.
  - Project agent skills live under `<workspace>/.agents/skills`.
  - Personal agent skills live under `~/.agents/skills`.
  - Shared managed skills live under `~/.openclaw/skills`.
  - Extra directories are configured with `skills.load.extraDirs`.
  - Every skill needs `name` and `description` frontmatter.
  - OpenClaw snapshots eligible skills when a session starts.
- Instruction surface:
  - OpenClaw builds an agent prompt from eligible skills.
  - Global behavior should be carried by managed skills and
    `skills.entries` config, not a Codex-style `AGENTS.md`.
- Installer decision:
  - Install managed skills to `~/.openclaw/skills/<name>/`.
  - If `~/.openclaw/openclaw.json` exists, merge a non-destructive
    `skills.entries` block for open-xquant skills.

## Manifest Schema

Path:

```text
~/.config/open-xquant/agent-install.json
```

Schema version:

```json
{
  "schema_version": 1,
  "installed_at": "2026-06-17T00:00:00Z",
  "updated_at": "2026-06-17T00:00:00Z",
  "source": {
    "type": "local",
    "repo": "xingwudao/open-xquant",
    "ref": "main",
    "commit": "abc123",
    "path": "/path/to/open-xquant"
  },
  "targets": {
    "codex": {
      "installed": true,
      "installed_at": "2026-06-17T00:00:00Z",
      "updated_at": "2026-06-17T00:00:00Z",
      "skills_dir": "/Users/alice/.agents/skills",
      "instruction_file": "/Users/alice/.codex/AGENTS.md",
      "installed_paths": [
        "/Users/alice/.agents/skills/strategy-builder"
      ],
      "managed_blocks": [
        {
          "file": "/Users/alice/.codex/AGENTS.md",
          "marker": "open-xquant"
        }
      ],
      "skills": [
        {
          "name": "strategy-builder",
          "source": "agent/skills/strategy-builder.md",
          "dest": "/Users/alice/.agents/skills/strategy-builder/SKILL.md",
          "source_sha256": "...",
          "dest_sha256": "..."
        }
      ]
    },
    "claude-code": {
      "installed": true,
      "skills_dir": "/Users/alice/.claude/skills",
      "instruction_file": "/Users/alice/.claude/CLAUDE.md",
      "installed_paths": [
        "/Users/alice/.claude/skills/strategy-builder"
      ],
      "managed_blocks": [
        {
          "file": "/Users/alice/.claude/CLAUDE.md",
          "marker": "open-xquant"
        }
      ],
      "skills": []
    }
  }
}
```

Rules:

- Store absolute expanded paths.
- Accept `~` in CLI inputs but normalize before writing.
- Manifest is the source of truth for uninstall.
- Missing manifest means `uninstall` fails with a repair message.
- MVP does not implement path discovery uninstall without a manifest.
- `source_sha256` tracks the canonical repo skill file.
- `dest_sha256` tracks the installed target-rendered `SKILL.md`.

## Global Agent Config

Path:

```text
~/.config/open-xquant/agent.yaml
```

Schema version:

```yaml
schema_version: 1
default_target: auto
installed_targets: []
default_data_dir: ~/.oxq/data
auto_init_workspace: true
allow_auto_download: ask
preferred_runner: uv run oxq
```

Rules:

- `install` creates this file if missing.
- `upgrade` preserves user values.
- Future migrations should merge missing defaults only.
- `uninstall` keeps this file unless `--purge-config`.

## Managed Marker Schema

Each installed skill directory contains:

```text
.open-xquant-managed.json
```

Schema:

```json
{
  "schema_version": 1,
  "managed_by": "open-xquant",
  "target": "codex",
  "name": "strategy-builder",
  "installed_at": "2026-06-17T00:00:00Z",
  "source_commit": "abc123",
  "source_sha256": "...",
  "dest_sha256": "..."
}
```

Safety rules:

- Delete a skill directory only if this marker exists.
- Marker must have `managed_by == "open-xquant"`.
- If the marker is missing, skip and warn.
- A future `--force` may override this, but MVP should not.

## Instruction Marker Blocks

Global Agent marker:

```text
<!-- open-xquant:begin -->
## open-xquant

When the user asks about quant strategy, backtest, factor evaluation,
parameter tuning, audit, robustness, report, broker connectivity, or live
trading, use the installed open-xquant skills.

If the current directory has no `.open-xquant/workspace.yaml`, run
`oxq research init` before creating strategy artifacts.

Default workflow:
`strategy_spec.yaml` -> validate -> backtest -> audit -> robustness -> report.
<!-- open-xquant:end -->
```

Claude Code global marker:

```text
<!-- open-xquant:begin -->
## open-xquant

When the user asks about quant strategy, backtest, factor evaluation,
parameter tuning, audit, robustness, report, broker connectivity, or live
trading, use the installed open-xquant skills.

If the current directory has no `.open-xquant/workspace.yaml`, run
`oxq research init` before creating strategy artifacts.

If this project has an `AGENTS.md`, also read it when it is relevant to
open-xquant work.
<!-- open-xquant:end -->
```

OpenClaw config merge block:

```json5
{
  skills: {
    entries: {
      "strategy-builder": { enabled: true },
      "backtest-runner": { enabled: true }
    }
  }
}
```

Workspace marker:

```text
<!-- open-xquant-workspace:begin -->
This is an open-xquant research workspace.

For quant strategy, factor, backtest, audit, robustness, report,
and live trading tasks, use the installed open-xquant skills.

Use `.open-xquant/workspace.yaml` for local paths.
<!-- open-xquant-workspace:end -->
```

Rules:

- Insert block if absent.
- Replace block if both markers exist.
- Remove only the block on uninstall.
- Fail on partial marker.
- Never modify content outside marker blocks.

## Skill Installation Mapping

Source:

```text
agent/skills/<name>.md
```

Target destinations:

```text
codex:      ~/.agents/skills/<name>/SKILL.md
opencode:   ~/.config/opencode/skills/<name>/SKILL.md
claude-code: ~/.claude/skills/<name>/SKILL.md
cursor:     ~/.cursor/skills/<name>/SKILL.md
openclaw:   ~/.openclaw/skills/<name>/SKILL.md
```

Example:

```text
agent/skills/strategy-builder.md
-> ~/.agents/skills/strategy-builder/SKILL.md
```

Rules:

- Treat `agent/skills/*.md` as the canonical source.
- Preserve markdown body exactly.
- Preserve frontmatter semantics, but allow target adapters to render
  compatible frontmatter.
- Use the source skill `name` as the destination directory.
- This is required for OpenCode, and it keeps Claude slash commands natural.
- Write `.open-xquant-managed.json`.
- Record `SKILL.md` SHA-256 in manifest.
- Record both source SHA-256 and rendered destination SHA-256.
- If a target requires stricter frontmatter than can be rendered safely,
  fail validation before writing.
- Do not install one shared directory and symlink it into every target in MVP.
- Symlinks complicate uninstall safety and official target refresh behavior.

Frontmatter rendering:

- `codex`, `opencode`, `claude-code`, and `cursor`:
  - Preserve source frontmatter format where valid.
  - Ensure `name` equals the destination directory name.
- `openclaw`:
  - Render `name` and `description` as single-line YAML scalars.
  - Preserve no unsupported custom fields in MVP.
  - Render `metadata` only if it can be converted to single-line JSON.
  - Fail with a clear message if conversion is lossy.

Target-specific skill validation:

- `codex`:
  - Require `name` and `description`.
  - Destination directory may differ from `name`.
- `opencode`:
  - Require lowercase hyphenated `name`.
  - Destination directory must match `name`.
- `claude-code`:
  - Require `description`.
  - Preserve optional invocation keys if present.
  - Destination directory becomes the slash command name.
- `cursor`:
  - Require `name` and `description`.
  - Preserve `paths` and `disable-model-invocation` if present.
  - Do not write `.cursor/rules` for generated skills.
- `openclaw`:
  - Require `name` and `description`.
  - Ensure `metadata` is single-line JSON if present.
  - Normalize folded multi-line `description` to a single-line scalar.

## Source Resolution

Implement `resolve_agent_source()` with this order:

1. Explicit path from `--from-local`.
2. Current working repo if `agent/skills` exists.
3. Installed package resource if package includes assets.
4. GitHub cache path for `upgrade`.
5. Fail with actionable error.

Packaging note:

- Current wheel target may only include `src/oxq`.
- If pip installs should support `agent install` without GitHub, package
  `agent/skills` as data.
- MVP can require local repo for `install` and use GitHub for `upgrade`.

## Command Specs

### `oxq agent install`

Supported options:

```bash
oxq agent install
oxq agent install --target codex
oxq agent install --target opencode
oxq agent install --target claude-code
oxq agent install --target cursor
oxq agent install --target openclaw
oxq agent install --target generic
oxq agent install --all-targets
oxq agent install --dry-run
oxq agent install --repair
oxq agent install --yes
```

Target selection:

1. If `--target` is provided, install only that target.
2. If `--all-targets` is provided, install every concrete target.
3. Else use `agent.yaml.default_target` if it is not `auto`.
4. Else detect installed Agent homes and install all detected targets.
5. Else install `generic` and print exact manual paths.

Detection rules:

- `codex` is detected when `codex` is on `PATH`, `CODEX_HOME` is set,
  or `~/.codex` exists.
- `opencode` is detected when `opencode` is on `PATH` or
  `~/.config/opencode` exists.
- `claude-code` is detected when `claude` is on `PATH` or `~/.claude` exists.
- `cursor` is detected when `cursor` is on `PATH`, `~/.cursor` exists,
  or the user passes `--target cursor`.
- `openclaw` is detected when `openclaw` is on `PATH` or `~/.openclaw`
  exists.

Target install behavior:

1. Resolve source skill directory.
2. Resolve target paths.
3. Validate source skills exist.
4. Validate target-specific skill requirements.
5. Create global config if missing.
6. Create target skill dirs.
7. Copy or render all target-compatible skills.
8. Write managed markers.
9. Insert or replace target instruction marker when the target has one.
10. Merge target config when the target uses config instead of a marker file.
11. Write manifest.
12. Print summary.

Dry run behavior:

- Print planned writes.
- Do not create directories.
- Do not edit files.

Repair behavior:

- Reinstall missing managed paths.
- Replace marker block.
- Recompute hashes.
- Preserve user config.

Conflicts:

- Existing destination without marker: skip and warn.
- Partial instruction marker: fail.
- Unknown target: fail and list supported targets.

Target-specific behavior:

- `codex`:
  - Write skills under `~/.agents/skills`.
  - Write marker block to `${CODEX_HOME:-~/.codex}/AGENTS.md`.
- `opencode`:
  - Write skills under `~/.config/opencode/skills`.
  - Write marker block to `~/.config/opencode/AGENTS.md`.
- `claude-code`:
  - Write skills under `~/.claude/skills`.
  - Write marker block to `~/.claude/CLAUDE.md`.
- `cursor`:
  - Write skills under `~/.cursor/skills`.
  - Do not write `~/.cursor/rules` unless official docs add a file-backed
    User Rules surface.
  - Ensure at least one installed skill description clearly triggers on
    `open-xquant`, quant strategy, backtest, and research workflow.
- `openclaw`:
  - Write skills under `~/.openclaw/skills`.
  - If `~/.openclaw/openclaw.json` exists, merge managed `skills.entries`.
  - If it does not exist, do not create a full OpenClaw config in MVP;
    skills are still discoverable from the managed directory.
- `generic`:
  - Print skill source paths and instruction block.
  - Do not write unknown Agent files.
  - May write only `agent.yaml` with `--yes`.

### `oxq agent uninstall`

Supported options:

```bash
oxq agent uninstall
oxq agent uninstall --target codex
oxq agent uninstall --target opencode
oxq agent uninstall --target claude-code
oxq agent uninstall --target cursor
oxq agent uninstall --target openclaw
oxq agent uninstall --all-targets
oxq agent uninstall --dry-run
oxq agent uninstall --purge-config
oxq agent uninstall --yes
```

Behavior:

1. Read manifest.
2. Select target or targets.
3. Verify each installed path is managed.
4. Delete managed skill directories.
5. Remove managed instruction block.
6. Update manifest target state.
7. Delete global config only with `--purge-config`.

Never delete:

```text
~/.oxq/data
.open-xquant/
strategy_specs/
runs/
reports/
experiments.jsonl
```

Failure behavior:

- Missing installed path: warn and continue.
- Missing marker: skip and warn.
- Partial instruction marker: fail.
- Missing manifest: fail with `oxq agent install` suggestion.

### `oxq agent upgrade`

Supported options:

```bash
oxq agent upgrade
oxq agent upgrade --target codex
oxq agent upgrade --target opencode
oxq agent upgrade --target claude-code
oxq agent upgrade --target cursor
oxq agent upgrade --target openclaw
oxq agent upgrade --all-targets
oxq agent upgrade --from-local /path/to/open-xquant
oxq agent upgrade --repo https://github.com/xingwudao/open-xquant
oxq agent upgrade --ref main
oxq agent upgrade --dry-run
oxq agent upgrade --yes
```

Source resolution:

1. `--from-local` if provided.
2. Current package source if it has `agent/skills`.
3. GitHub repo/ref into cache.
4. Fail.

GitHub cache path:

```text
~/.config/open-xquant/cache/open-xquant/<ref-or-commit>/
```

MVP GitHub fetch:

```bash
git clone --depth 1 --branch <ref> <repo> <cache-dir>
```

Security:

- Never execute code from the downloaded repo.
- Only copy markdown from `agent/skills`.
- Reject symlinked source skill files in MVP.

Upgrade behavior:

1. Read manifest.
2. Resolve new source.
3. Compute new source commit if possible.
4. Verify installed paths are managed.
5. Detect local modifications by comparing manifest `dest_sha256`
   to installed target `SKILL.md` SHA.
6. Skip modified installed skills by default.
7. Overwrite unmodified managed `SKILL.md`.
8. Update managed marker JSON.
9. Replace instruction marker block.
10. Preserve `agent.yaml`.
11. Update manifest source and hashes.

Dry run behavior:

- Show old commit.
- Show new commit.
- Show files to update.
- Show conflicts.
- Write nothing.

### `oxq agent status`

Supported options:

```bash
oxq agent status
oxq agent status --json
```

Behavior:

- Read global config.
- Read manifest.
- For each target, report:
  - installed state
  - installed skill count
  - missing paths
  - instruction block presence
  - source commit
  - config path
  - manifest path

Text output:

```text
open-xquant agent status

Config:   ~/.config/open-xquant/agent.yaml
Manifest: ~/.config/open-xquant/agent-install.json

Target: codex
Installed: yes
Skills: 20/20
Instruction block: present
Commit: abc123
```

### `oxq research init`

Supported options:

```bash
oxq research init
oxq research init --name my-study
oxq research init --data-dir ~/.oxq/data/market
oxq research init --minimal
oxq research init --force
```

Behavior:

1. Resolve current directory.
2. If `.open-xquant/workspace.yaml` exists, report already initialized.
3. Create `.open-xquant`.
4. Create `strategy_specs`, `runs`, `reports`.
5. Create `experiments.jsonl` if missing.
6. Write `.open-xquant/workspace.yaml`.
7. Insert or replace local `AGENTS.md` workspace marker.

Workspace YAML:

```yaml
schema_version: 1
name: my-study
created_at: "2026-06-17T00:00:00Z"

paths:
  specs_dir: strategy_specs
  runs_dir: runs
  reports_dir: reports
  experiment_registry: experiments.jsonl

data:
  market_data_dir: ~/.oxq/data/market
  provider: local

workflow:
  require_validate_before_backtest: true
  require_audit_before_report: true
  default_output_dir: runs/auto
```

Rules:

- Do not copy global skills into the workspace.
- Do not overwrite existing `AGENTS.md` content outside marker block.
- `--force` replaces the workspace marker block and workspace config.

### `oxq doctor`

Supported options:

```bash
oxq doctor
oxq doctor --json
oxq doctor --fix
```

MVP checks:

- CLI:
  - Python version `>=3.12`
  - `oxq` import works
  - `oxq --help` command available
- Agent:
  - `agent.yaml` exists
  - `agent-install.json` exists
  - selected target or all manifest targets installed
  - installed skill dirs exist
  - managed markers valid
  - instruction block present when target has an instruction file
  - OpenClaw managed `skills.entries` present when config exists
- Workspace:
  - `.open-xquant/workspace.yaml` exists
  - dirs from workspace config exist
  - `experiments.jsonl` exists
- Data:
  - configured market data dir exists
  - if specs exist, cheap symbol parquet check
- Deps:
  - `yfinance` import available
  - `mplfinance` import available for chart readiness
  - `httpx` and `websockets` import available for live readiness

Text output:

```text
open-xquant doctor

CLI:        OK
Agent:      OK
Workspace:  MISSING
Data:       OK
Deps:       WARN

Suggested fixes:
- oxq research init
- uv sync --extra chart
```

JSON output:

```json
{
  "status": "warn",
  "checks": {
    "cli": {
      "status": "ok"
    },
    "agent": {
      "status": "ok",
      "target": "codex",
      "skills": {
        "installed": 20,
        "expected": 20
      }
    },
    "workspace": {
      "status": "missing",
      "fixes": ["oxq research init"]
    },
    "deps": {
      "status": "warn",
      "missing": ["mplfinance"],
      "fixes": ["uv sync --extra chart"]
    }
  },
  "fixes": [
    "oxq research init",
    "uv sync --extra chart"
  ]
}
```

`--fix` MVP:

- May run `research init` if workspace is missing and config allows it.
- Must not install dependencies automatically.
- Must not run global `agent install` automatically.

## Atomic Writes

For JSON, YAML, and markdown:

1. Write a temp file in the destination directory.
2. Flush and close.
3. Rename over destination.

For installed skill update:

1. Write `SKILL.md.tmp`.
2. Rename to `SKILL.md`.
3. Write marker temp file.
4. Rename marker file.

For uninstall:

- Verify all safety rules for a path before deleting it.
- Skip unsafe paths; do not abort the entire uninstall unless marker parsing
  or instruction marker state is ambiguous.

## Security Boundaries

- Never execute code from a downloaded GitHub repo during upgrade.
- Only copy markdown files from `agent/skills`.
- Refuse source files that are symlinks in MVP.
- Validate source filenames end with `.md`.
- Refuse path traversal.
- Refuse to write outside resolved target directories.
- Do not follow symlinks during delete in MVP.

## Idempotency Requirements

`agent install` twice:

- no duplicate instruction block
- managed skill files refreshed
- manifest updated

`research init` twice:

- no duplicate local `AGENTS.md` block
- no directory errors

`agent uninstall` twice:

- second run reports target not installed or paths already missing
- no destructive failure

`agent upgrade` when up to date:

- reports no changes or refreshes managed files safely

## Task 1: Manifest And Marker Utilities

**Files:**

- Create: `src/oxq/cli/agent_manifest.py`
- Test: `tests/cli/test_agent_manifest.py`

- [ ] Define constants for config paths.
- [ ] Implement `expand_path(path: str | Path) -> Path`.
- [ ] Implement `sha256_file(path: Path) -> str`.
- [ ] Implement atomic text, JSON, and YAML writes.
- [ ] Implement manifest load and save.
- [ ] Implement global config load and save.
- [ ] Implement managed marker read and write.
- [ ] Add tests for round trip, missing files, and path expansion.

Verification:

```bash
uv run pytest tests/cli/test_agent_manifest.py -v
```

Expected:

```text
passed
```

## Task 2: Marker Block Editing

**Files:**

- Modify: `src/oxq/cli/agent_manifest.py`
- Test: `tests/cli/test_agent_manifest.py`

- [ ] Implement `upsert_marker_block(path, marker, content)`.
- [ ] Implement `remove_marker_block(path, marker)`.
- [ ] Raise a clear error on partial markers.
- [ ] Preserve user content outside marker blocks.
- [ ] Test insert into empty file.
- [ ] Test append to existing file.
- [ ] Test replace existing block.
- [ ] Test remove block.
- [ ] Test partial marker failure.

Verification:

```bash
uv run pytest tests/cli/test_agent_manifest.py -v
```

Expected:

```text
passed
```

## Task 3: Target Adapters

**Files:**

- Create: `src/oxq/cli/agent_targets.py`
- Test: `tests/cli/test_agent_install.py`

- [ ] Define `AgentTarget` dataclass.
- [ ] Implement `resolve_codex_target()`.
- [ ] Implement `resolve_opencode_target()`.
- [ ] Implement `resolve_claude_code_target()`.
- [ ] Implement `resolve_cursor_target()`.
- [ ] Implement `resolve_openclaw_target()`.
- [ ] Implement `generic` target behavior.
- [ ] Implement target detection from binary and home paths.
- [ ] Implement source skill discovery from local repo.
- [ ] Implement skill source validation.
- [ ] Implement target-specific destination mapping.
- [ ] Implement OpenCode directory-name validation.
- [ ] Implement OpenClaw metadata validation.
- [ ] Test `CODEX_HOME` override.
- [ ] Test Codex `~/.agents/skills` destination.
- [ ] Test OpenCode `~/.config/opencode/skills` destination.
- [ ] Test Claude Code `~/.claude/skills` destination.
- [ ] Test Cursor `~/.cursor/skills` destination.
- [ ] Test OpenClaw `~/.openclaw/skills` destination.
- [ ] Test default auto-detection order.
- [ ] Test skill mapping names.

Verification:

```bash
uv run pytest tests/cli/test_agent_install.py -v
```

Expected:

```text
passed
```

## Task 4: Agent Install

**Files:**

- Create: `src/oxq/cli/agent.py`
- Modify: `src/oxq/cli/main.py`
- Test: `tests/cli/test_agent_install.py`

- [ ] Add `agent` command group.
- [ ] Add `agent install` command.
- [ ] Implement `--target`, `--dry-run`, `--repair`, and `--yes`.
- [ ] Implement `--all-targets`.
- [ ] Copy all source skills to target skill directories.
- [ ] Write managed marker in each skill directory.
- [ ] Insert target instruction marker blocks where supported.
- [ ] Merge OpenClaw `skills.entries` when config exists.
- [ ] Write global config if missing.
- [ ] Write manifest.
- [ ] Test dry run writes nothing.
- [ ] Test Codex install writes all skills.
- [ ] Test OpenCode install writes all skills.
- [ ] Test Claude Code install writes all skills.
- [ ] Test Cursor install writes all skills.
- [ ] Test OpenClaw install writes all skills.
- [ ] Test managed markers exist.
- [ ] Test second install is idempotent.
- [ ] Test unmarked destination conflict is skipped.
- [ ] Test `--all-targets` records every target in manifest.

Verification:

```bash
uv run pytest tests/cli/test_agent_install.py -v
uv run oxq agent install --target generic --dry-run
```

Expected:

```text
passed
```

## Task 5: Agent Uninstall

**Files:**

- Modify: `src/oxq/cli/agent.py`
- Test: `tests/cli/test_agent_uninstall.py`

- [ ] Add `agent uninstall` command.
- [ ] Implement `--target`, `--all-targets`, `--dry-run`,
  `--purge-config`, and `--yes`.
- [ ] Delete only managed skill directories.
- [ ] Remove global instruction marker block.
- [ ] Remove target-specific instruction marker blocks.
- [ ] Remove only managed OpenClaw `skills.entries` keys.
- [ ] Update manifest state.
- [ ] Keep global config unless `--purge-config`.
- [ ] Test dry run writes nothing.
- [ ] Test managed dirs removed.
- [ ] Test unmarked dirs skipped.
- [ ] Test marker block removed.
- [ ] Test research dirs and data dirs are never removed.

Verification:

```bash
uv run pytest tests/cli/test_agent_uninstall.py -v
```

Expected:

```text
passed
```

## Task 6: Agent Status

**Files:**

- Modify: `src/oxq/cli/agent.py`
- Test: `tests/cli/test_agent_install.py`

- [ ] Add `agent status` command.
- [ ] Implement text output.
- [ ] Implement `--json`.
- [ ] Report config path, manifest path, target state, skill count,
  missing paths, instruction block state, and commit.
- [ ] Report all installed targets by default.
- [ ] Test status before install.
- [ ] Test status after install.
- [ ] Test multi-target status.
- [ ] Test JSON shape.

Verification:

```bash
uv run pytest tests/cli/test_agent_install.py -v
```

Expected:

```text
passed
```

## Task 7: Research Init

**Files:**

- Create: `src/oxq/cli/research.py`
- Modify: `src/oxq/cli/main.py`
- Test: `tests/cli/test_research_init.py`

- [ ] Add `research` command group.
- [ ] Add `research init` command.
- [ ] Implement `--name`, `--data-dir`, `--minimal`, and `--force`.
- [ ] Create `.open-xquant/workspace.yaml`.
- [ ] Create `strategy_specs`, `runs`, `reports`.
- [ ] Create `experiments.jsonl`.
- [ ] Insert local `AGENTS.md` workspace marker block.
- [ ] Preserve existing `AGENTS.md` user content.
- [ ] Test idempotent second run.
- [ ] Test `--force` replaces marker block.

Verification:

```bash
uv run pytest tests/cli/test_research_init.py -v
```

Expected:

```text
passed
```

## Task 8: Doctor

**Files:**

- Create: `src/oxq/cli/doctor.py`
- Modify: `src/oxq/cli/main.py`
- Test: `tests/cli/test_doctor.py`

- [ ] Add top-level `doctor` command.
- [ ] Implement `--json`.
- [ ] Implement `--fix` for workspace init only.
- [ ] Check CLI readiness.
- [ ] Check Agent install readiness.
- [ ] Check workspace readiness.
- [ ] Check data directory existence.
- [ ] Check optional dependencies.
- [ ] Return aggregate `ok`, `warn`, or `fail`.
- [ ] Test missing workspace suggests `oxq research init`.
- [ ] Test installed Agent reports skill count.
- [ ] Test missing optional chart dependency reports warning.

Verification:

```bash
uv run pytest tests/cli/test_doctor.py -v
uv run oxq doctor --json
```

Expected:

```text
passed
```

## Task 9: Agent Upgrade From Local

**Files:**

- Modify: `src/oxq/cli/agent.py`
- Test: `tests/cli/test_agent_upgrade.py`

- [ ] Add `agent upgrade` command.
- [ ] Implement `--target`, `--all-targets`, `--from-local`, `--dry-run`,
  and `--yes`.
- [ ] Resolve source skill directory from `--from-local`.
- [ ] Compare installed SHA against manifest `dest_sha256`.
- [ ] Skip locally modified installed skills by default.
- [ ] Replace unmodified managed skills.
- [ ] Update marker JSON.
- [ ] Replace target instruction marker blocks.
- [ ] Refresh managed OpenClaw config keys.
- [ ] Preserve `agent.yaml`.
- [ ] Update manifest source and skill hashes.
- [ ] Test dry run writes nothing.
- [ ] Test updated skill content changes.
- [ ] Test modified installed skill is skipped.
- [ ] Test upgrade across all concrete targets.

Verification:

```bash
uv run pytest tests/cli/test_agent_upgrade.py -v
```

Expected:

```text
passed
```

## Task 10: Agent Upgrade From GitHub

**Files:**

- Modify: `src/oxq/cli/agent.py`
- Test: `tests/cli/test_agent_upgrade.py`

- [ ] Implement `--repo`.
- [ ] Implement `--ref`.
- [ ] Clone into `~/.config/open-xquant/cache/open-xquant/<ref>/`.
- [ ] Do not execute downloaded code.
- [ ] Copy only markdown files from `agent/skills`.
- [ ] Reject symlinked source skill files.
- [ ] Record repo, ref, and commit in manifest.
- [ ] Test with a local git fixture instead of network.

Verification:

```bash
uv run pytest tests/cli/test_agent_upgrade.py -v
```

Expected:

```text
passed
```

## Task 11: Docs Update

**Files:**

- Modify: `docs/agent-guide.md`

- [ ] Reframe `docs/agent-guide.md` as a one-time bootstrap guide.
- [ ] Add `oxq agent install`.
- [ ] Add `oxq doctor`.
- [ ] Add `oxq research init`.
- [ ] Add repeat-use flow where user directly provides a strategy idea.
- [ ] Add uninstall and upgrade commands.
- [ ] Keep command examples terminal-safe.

Verification:

```bash
git diff --check -- docs/agent-guide.md
```

Expected:

```text
no output
```

## Task 12: Full Verification

**Files:**

- No new files.

- [ ] Run focused CLI tests.
- [ ] Run full test suite.
- [ ] Run formatting/lint checks if configured.
- [ ] Manually smoke test install and uninstall with temp `CODEX_HOME`.
- [ ] Manually smoke test research init in a temp directory.

Commands:

```bash
uv run pytest tests/cli -v
uv run pytest
home=$(mktemp -d)
HOME="$home" CODEX_HOME="$home/.codex" uv run oxq agent install --all-targets
HOME="$home" CODEX_HOME="$home/.codex" uv run oxq agent status
HOME="$home" CODEX_HOME="$home/.codex" uv run oxq agent uninstall \
  --all-targets --yes
tmpdir=$(mktemp -d)
cd "$tmpdir"
uv run --project /path/to/open-xquant oxq research init
```

Expected:

```text
tests pass
agent install creates managed skills
agent uninstall removes only managed skills
research init creates workspace files
```

## Acceptance Criteria

First-time setup:

```bash
home=$(mktemp -d)
HOME="$home" CODEX_HOME="$home/.codex" uv run oxq agent install --all-targets
```

Creates:

```text
~/.agents/skills/strategy-builder/SKILL.md
~/.agents/skills/strategy-builder/.open-xquant-managed.json
$CODEX_HOME/AGENTS.md
~/.config/opencode/skills/strategy-builder/SKILL.md
~/.config/opencode/AGENTS.md
~/.claude/skills/strategy-builder/SKILL.md
~/.claude/CLAUDE.md
~/.cursor/skills/strategy-builder/SKILL.md
~/.openclaw/skills/strategy-builder/SKILL.md
~/.config/open-xquant/agent-install.json
```

New research directory:

```bash
mkdir /tmp/study
cd /tmp/study
uv run --project /path/to/open-xquant oxq research init
```

Creates:

```text
.open-xquant/workspace.yaml
AGENTS.md
strategy_specs/
runs/
reports/
experiments.jsonl
```

Doctor:

```bash
uv run oxq doctor --json
```

Returns a machine-readable status with suggested fixes.

Uninstall:

```bash
HOME="$home" CODEX_HOME="$home/.codex" uv run oxq agent uninstall \
  --all-targets --yes
```

Removes managed skill directories and target marker blocks, while keeping:

```text
~/.config/open-xquant/agent.yaml
~/.oxq/data
research workspaces
```

Upgrade:

```bash
HOME="$home" CODEX_HOME="$home/.codex" uv run oxq agent upgrade \
  --all-targets \
  --from-local /path/to/open-xquant
```

Updates managed skills and manifest while preserving user config.

## Self-Review Notes

Spec coverage:

- Install, uninstall, upgrade, status, research init, and doctor are all mapped
  to tasks.
- Manifest, global config, workspace config, skill markers, and instruction
  marker blocks are specified.
- Codex, OpenCode, Claude Code, Cursor, and OpenClaw official support
  surfaces are mapped to target adapters.
- Target-specific frontmatter and path constraints are specified.
- Safety and idempotency rules are specified.
- Tests are specified for every command family.

Known implementation decision:

- The plan uses `src/oxq/cli/*` modules instead of a new `src/oxq/agent/`
  package to keep the MVP close to the existing Click CLI.

Potential follow-up:

- If maintainers want these capabilities available from PyPI without GitHub,
  packaging must include `agent/skills` as package data.

- Decision: Implement lifecycle with manifest-backed Agent commands and
  workspace initialization.
- Why: This lets users reuse open-xquant in new directories without rereading
  `docs/agent-guide.md`.
- Next step: Execute Task 1 with tests, then proceed task-by-task.
