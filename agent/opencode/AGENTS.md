# open-xquant OpenCode Integration

This directory contains OpenCode agent definitions and commands for the
open-xquant Agentic Quant Research Kernel.

OpenCode skill definitions are single-sourced from `agent/skills/`. Do not add
or edit duplicate skills under `agent/opencode/skills/`.

## Usage

Copy or symlink the repository `agent/` directory to your OpenCode workspace so
`agent/opencode/opencode.json` can load the shared `agent/skills/` directory:

```bash
cp -r agent /path/to/opencode/workspace/
```

The canonical skill text lives in `agent/skills/*.md`. OpenCode discovers
skills through `agent/skills/<name>/SKILL.md` symlink adapters that point back
to those canonical files. Keep those adapters inside `agent/skills/`; never
create a second skill copy under `agent/opencode/skills/`.

## Agent Roles

| Agent | Role | Permissions |
|-------|------|-------------|
| **quant-planner** | Draft strategy specs from ideas | Read docs, write spec files |
| **quant-builder** | Compile and backtest strategies | Read spec, write code, run backtest |
| **quant-auditor** | Audit backtest results | Read all artifacts, write audit reports |
| **quant-reporter** | Generate research reports | Read reports/metrics, write report.md |

**Critical rule**: Builder and Auditor must be separate agents.
The auditor must never modify strategy code.
