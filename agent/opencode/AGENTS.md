# open-xquant OpenCode Integration

This directory contains OpenCode agent definitions and commands for the
open-xquant Agentic Quant Research Kernel.

OpenCode skill definitions are single-sourced from `agent/skills/`. Do not add
or edit duplicate skills under `agent/opencode/skills/`.

## Usage

Copy or symlink the repository `agent/` directory to your OpenCode workspace
root, and load OpenCode from that same workspace root, so
`agent/opencode/opencode.json` can resolve the shared `agent/skills/`
directory:

```bash
cp -r agent /path/to/opencode/workspace/
```

The canonical skill text lives in `agent/skills/*.md`. OpenCode discovers
skills through `agent/skills/<name>/SKILL.md` wrapper adapters. Those adapters
contain only discovery frontmatter plus instructions to read the canonical
`../<name>.md` file; they must not copy full skill bodies. Keep those adapters
inside `agent/skills/`; never create a second skill copy under
`agent/opencode/skills/`.

## Agent Roles

| Agent | Role | Permissions |
|-------|------|-------------|
| **quant-planner** | Draft strategy specs from ideas | Read docs, write spec files |
| **quant-builder** | Compile and backtest strategies | Read spec, write code, run backtest |
| **quant-auditor** | Audit backtest results | Read all artifacts, write audit reports |
| **quant-reporter** | Generate research reports | Read reports/metrics, write report.md |

**Critical rule**: Builder and Auditor must be separate agents.
The auditor must never modify strategy code.
