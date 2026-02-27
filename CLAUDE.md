# CLAUDE.md

## Project Overview

open-xquant is an Agent First quantitative trading framework. See `README.md` for motivation and `docs/architecture.md` for full design.

## Tech Stack

- Python 3.12+, pandas, numpy
- Build: hatch / uv
- Test: pytest
- Lint: ruff
- Type check: mypy
- MCP SDK: mcp (Python)

## Project Structure

- `src/oxq/` — main Python package (pip install open-xquant)
- `mcp_server/` — MCP protocol server exposing oxq as tools
- `skills/` — Agent skill definitions (markdown)
- `examples/` — example strategies, demo apps, tutorials
- `tests/` — mirrors src/oxq/ structure
- `docs/` — documentation

## Key Architecture

- **Indicator → Signal → Rule** three-phase model
- Single wide DataFrame (mktdata) shared across phases
- Three Protocol interfaces decouple strategy from execution: MarketDataProvider, OrderRouter, FillReceiver
- `dataclass(frozen=True)` for immutable data types, `Protocol` for interfaces
- Use `Decimal` for financial precision

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Conventions

- Code and comments in English
- User-facing docs may be in Chinese
- Follow ruff defaults (E, F, I, N, W, UP rules)
- Prefer Protocol over ABC for interfaces
- Keep Indicator/Signal compute functions pure (no side effects)
- **Do not add new top-level directories** — this is an open-source project; keep the root structure stable. New apps, demos, and tutorials go under `examples/`.
