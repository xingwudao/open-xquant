# open-xquant OpenCode Integration

This directory contains OpenCode agent definitions, commands, and skills
for the open-xquant Agentic Quant Research Kernel.

## Usage

Copy or symlink this directory to your OpenCode workspace:

```bash
cp -r agent/opencode /path/to/opencode/workspace/
```

## Agent Roles

| Agent | Role | Permissions |
|-------|------|-------------|
| **quant-planner** | Draft strategy specs from ideas | Read docs, write spec files |
| **quant-builder** | Compile and backtest strategies | Read spec, write code, run backtest |
| **quant-auditor** | Audit backtest results | Read all artifacts, write audit reports |
| **quant-reporter** | Generate research reports | Read reports/metrics, write report.md |

**Critical rule**: Builder and Auditor must be separate agents.
The auditor must never modify strategy code.
