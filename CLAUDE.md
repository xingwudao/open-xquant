# CLAUDE.md

## Project Overview

open-xquant is an Agentic Quant Research Kernel. See `README.md` for motivation and `docs/architecture.md` for full design.

## Project Structure

- `src/oxq/` — main Python package (pip install open-xquant)
- `agent/` — Agent layer (skills, bootstrap configs, OpenCode integration)
  - `agent/skills/` — Agent skill definitions (markdown workflows)
  - `agent/openclaw/bootstrap/` — Agent startup files (OpenClaw etc.)
  - `agent/opencode/` — OpenCode integration (agents, commands, skills)
- `examples/` — example strategies, research cases, module demos
- `tests/` — mirrors src/oxq/ structure
- `docs/` — documentation

## Bug Fixing

Follow this strict TDD protocol for every bug fix:

1. **Write a failing test first** — describe the expected behavior with hand-calculated values, not values copied from buggy code.
2. **Run the test, confirm it fails for the right reason** — if it passes, your understanding of the bug is wrong.
3. **Implement the smallest possible fix** — no refactoring, no drive-by improvements.
4. **Run the new test** — confirm it passes.
5. **Run the full test suite** (`uv run pytest`) — confirm no regressions.
6. **Grep for the same pattern across the entire codebase** — fix ALL instances before declaring done. E.g., if `prices = {symbol: price}` is wrong in `risk.py`, check `entry.py`, `rebalance.py`, etc.

Never guess root causes — provide concrete evidence (specific line numbers, variable values) before proposing a fix.

## Cross-File Sync

When fixing a bug or updating logic in one module, always check and update all related modules that share the same pattern. Use `grep` to find all affected locations before editing any of them.

## Conventions

- Code and comments in English
- User-facing docs may be in Chinese
- Follow ruff defaults (E, F, I, N, W, UP rules)
- Prefer Protocol over ABC for interfaces
- Keep Indicator/Signal compute functions pure (no side effects)
- **Do not add new top-level directories** — this is an open-source project; keep the root structure stable. New apps, demos, and examples go under `examples/`.

## Component Creation

When a user needs a component that doesn't exist in oxq (Indicator, Signal, PortfolioOptimizer, or Rule),
read and follow `agent/skills/component-creator.md` to check the registry and route to the appropriate creation sub-skill.

**In this project**, the sub-skills use placeholder `{target_dir}` — replace with these concrete paths:
- Component code: `src/oxq/{component_type}/{snake_name}.py`
- Test code: `tests/{component_type}/test_{snake_name}.py`
- Module import path: `oxq.{component_type}.{snake_name}`

After validation passes, also update these files to make it a built-in:
1. `src/oxq/{component_type}/__init__.py` — add import + `__all__` entry
2. `src/oxq/core/registry.py` `_load_builtins()` — add to the appropriate registration block
