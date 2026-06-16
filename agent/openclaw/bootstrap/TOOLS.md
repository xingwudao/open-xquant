# TOOLS.md - Environment & Tool Reference

Skills define _how_ tools work. This file documents _this specific setup_ —
what's available, where things live, and how to reach them.

---

## open-xquant CLI / SDK

open-xquant provides two primary interaction surfaces:

- **CLI**: `oxq <command>` — the primary way for AI agents to run research workflows
- **SDK**: `import oxq` — direct Python API for programmatic use

- **Repo**: github.com/xingwudao/open-xquant
- **PyPI package**: `open-xquant` (import as `oxq`)
- **Python requirement**: >= 3.12
- **Package manager**: `uv` (preferred)

Verify the CLI is installed before starting any research session:
```bash
oxq --help
```

---

## Core CLI Workflow

The standard research pipeline from idea to audited report:

```bash
# 1. Create a strategy spec from an idea
oxq spec init "SMA crossover with weekly rebalance"

# 2. Edit the spec file, then validate
oxq spec validate strategy_spec.yaml

# 3. Compile and run backtest
oxq strategy compile strategy_spec.yaml
oxq backtest run strategy_spec.yaml --out runs/auto

# 4. Audit the results
oxq audit reproducibility runs/<run_id>/
oxq audit research runs/<run_id>/

# 5. Robustness tests
oxq robustness run runs/<run_id>/

# 6. Generate report
oxq report write runs/<run_id>/

# 7. Register experiment
oxq experiment add runs/<run_id>/
```

---

## oxq Core Pipeline

完整执行管道：

```
Indicator → Universe → Signal → Portfolio → Pre-trade Rule
    → Trading Algorithm → Broker → Post-trade Rule
```

两个阶段：
- **向量化阶段** (setup): Indicator + Signal 对全量时间序列一次计算
- **逐 bar 阶段** (step): Portfolio → Rule → Trading → Broker 逐步推进

Key interfaces (all use Protocol over ABC — prefer structural typing):
- `Indicator.compute(df) → Series` — 纯函数，输出连续数值
- `Signal.compute(df) → Series` — 纯函数，输出离散标签（buy/hold/sell）
- `PortfolioOptimizer.optimize(signals, indicators) → dict[str, float]` — 截面优化，输出目标权重
- `Rule.evaluate(symbol, row, portfolio) → RuleResult` — 逐 bar 有状态，输出约束/减仓意图

Universe: `StaticUniverse` (fixed pool), `FilterUniverse` (dynamic screening)
Data providers: `LocalMarketDataProvider` (unified read interface)
Data sources: YFinance (US equities), AkShare (A-shares), WorldBank (macro factors)

---

## CLI Command Reference

### spec (2 commands)

| Command | Description |
|---------|-------------|
| `oxq spec init "<idea>"` | Create a strategy_spec.yaml template from a natural language idea |
| `oxq spec validate <file>` | Validate a spec against P0 rules (hypothesis, universe, signal_time, cost, OOS, etc.) |

### strategy (1 command)

| Command | Description |
|----------|-------------|
| `oxq strategy compile <file>` | Compile a spec into an executable Strategy object and report the result |

### backtest (1 command)

| Command | Description |
|----------|-------------|
| `oxq backtest run <file> --out runs/auto` | Run a backtest from a spec and write standardized artifacts |

### audit (2 commands)

| Command | Description |
|----------|-------------|
| `oxq audit reproducibility <run_dir>` | Verify spec hash, equity hash, trades hash, metrics hash consistency |
| `oxq audit research <run_dir>` | Check execution lag, cost model, OOS, survivorship bias, parameter count, trade count, etc. |

### robustness (1 command)

| Command | Description |
|----------|-------------|
| `oxq robustness run <run_dir>` | Run cost x2 perturbation, IS/OOS comparison, parameter perturbation check |

### report (1 command)

| Command | Description |
|----------|-------------|
| `oxq report write <run_dir>` | Generate research_report.md with executive decision (REJECT/WATCHLIST/CANDIDATE) |

### experiment (1 command)

| Command | Description |
|----------|-------------|
| `oxq experiment add <run_dir>` | Register a backtest run in experiments.jsonl to prevent selective memory |

---

## Python SDK (programmatic use)

For strategies that need custom logic beyond what specs can express:

```python
from oxq.core import Engine, Strategy
from oxq.indicators import SMA
from oxq.signals import Crossover
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.rules import ExitRule
from oxq.universe import StaticUniverse

crossover = Crossover()
crossover.required_indicators = {
    "sma_fast": (SMA(), {"column": "close", "period": 10}),
    "sma_slow": (SMA(), {"column": "close", "period": 50}),
}

strategy = Strategy(
    name="sma_crossover",
    hypothesis="短期均线上穿长期均线的标的在后续持有期内有正超额收益",
    benchmarks=["SPY"],
    universe=StaticUniverse(("AAPL",)),
    signals={"golden_cross": (crossover, {"fast": "sma_fast", "slow": "sma_slow"})},
    portfolio=EqualWeightOptimizer(),
)

engine = Engine()
result = engine.run(strategy,
    market=LocalMarketDataProvider(),
    broker=sim_broker,
    rules=[ExitRule(fast="sma_fast", slow="sma_slow")],
    start="2023-01-01", end="2024-12-31")

print(f"Sharpe: {result.sharpe_ratio():.2f}")
```

---

## Skills Directory

Skills live in `agent/skills/` within the repo.
Installed skills for this agent live in `~/.openclaw/workspace/skills/`.

Before using any skill, read its content to understand:
- what tools it calls
- what inputs it expects
- what outputs it produces

---

## Python Environment

- Preferred runner: `uv run python` or `uv run pytest`
- Linter/formatter: `ruff` (rules: E, F, I, N, W, UP)
- Type checker: `mypy` (strict mode)
- Test runner: `uv run pytest` (unit tests by default; e2e and integration tests excluded unless flagged)

Install variants:
```bash
pip install open-xquant                          # core only
pip install open-xquant[yfinance,akshare]        # with data sources
pip install open-xquant[live]                    # Alpaca live trading
pip install open-xquant[chart]                   # chart visualization
```

---

## Examples

Located in `examples/` within the repo:

| Path | Content |
|------|---------|
| **Module Examples** | |
| `modules/01_spec_and_validate.py` | Spec creation & P0 validation |
| `modules/02_data_and_universe.py` | Data download, inspect, universe |
| `modules/03_backtest_and_artifacts.py` | Backtest run & artifact inspection |
| `modules/04_audit_and_robustness.py` | Reproducibility + bias audit, robustness |
| `modules/05_report_and_experiment.py` | Report generation & experiment registry |
| **Research Cases** | |
| `research_cases/sma_crossover_valid/` | SMA crossover golden case |
| `research_cases/momentum_topn_valid/` | Momentum rotation golden case |
| `research_cases/same_bar_execution_invalid/` | Same-bar execution (audit fail) |
| `research_cases/zero_cost_invalid/` | Zero-cost model (audit fail) |
| `research_cases/static_universe_warning/` | Static universe (audit warn) |
| **Strategies** | |
| `strategies/sma_crossover_spec.py` | SMA crossover — full E2E pipeline |
| `strategies/momentum_rotation_spec.py` | Momentum rotation — full E2E pipeline |
| `strategies/sma_crossover.py` | SMA crossover (SDK only) |
| `strategies/momentum_rotation.py` | Momentum rotation (SDK only) |

---

## Research Output Conventions

- Research specs: `strategy_spec.yaml` (version-controlled)
- Backtest results: `runs/<timestamp>_<strategy_id>/` (10 fixed artifacts)
- Audit results: run `oxq audit research` and `oxq audit reproducibility`
- Experiment registry: `experiments.jsonl`
- Reproducibility check: re-run with identical spec + data before recording any result

---

## Framework Feedback Log

Running log of friction points and improvement suggestions:
`~/.openclaw/workspace/memory/framework-feedback.md`

Format per entry:
```
## [YYYY-MM-DD] <topic>
**Friction**: what was hard or missing
**Suggestion**: what would fix it
**Priority**: P0 / P1 / P2
```

---

## This Server

- Host: {{hostname}}
- OS: {{os_version}}
- OpenClaw: {{openclaw_version}}
- Gateway: {{gateway_url}}
- Channels: {{connected_channels}}
