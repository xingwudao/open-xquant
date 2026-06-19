# Metrics Profile Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit metric profiles and report transparency without changing existing specs by default.

**Architecture:** Add a small `metrics` spec section, a focused portfolio metrics-profile module, and wire it into compiler artifact generation and report rendering. Existing `RunResult` methods stay backward-compatible; profile-aware calculations live in the new module and are used only by compiler artifacts.

**Tech Stack:** Python dataclasses, pandas/numpy, pytest, existing open-xquant spec/compiler/report modules.

---

### Task 1: Spec Metrics Section

**Files:**
- Modify: `src/oxq/spec/schema.py`
- Modify: `src/oxq/spec/validator.py`
- Test: `tests/spec/test_validator.py`
- Test: `tests/spec/test_execution_semantics.py`

- [ ] **Step 1: Write failing schema/validator tests**

Add tests that load a spec with:

```yaml
metrics:
  profile: xquant_production
  risk_free_rate: 0.02
  return_type: log
  annualization_days: 252
  calmar_denominator: max_drawdown
  evaluation_window: oos
```

Assert `StrategySpec.from_yaml()` parses those fields and `validate()` passes.
Add a second test asserting `metrics.profile: unknown` fails validation with
`metrics_profile_unsupported`.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/spec/test_validator.py::test_validate_accepts_supported_metrics_profile tests/spec/test_validator.py::test_validate_rejects_unknown_metrics_profile -q
```

Expected: failure because `StrategySpec` has no `metrics` section or validator has no metrics checks.

- [ ] **Step 3: Implement minimal schema/validator support**

Add `MetricsSection` with:

```python
profile: str = "open_xquant_default"
risk_free_rate: float = 0.0
return_type: str = "simple"
annualization_days: int = 252
calmar_denominator: str = "max_drawdown"
evaluation_window: str = "full"
```

Parse it from YAML, include it in `StrategySpec`, and validate supported values:
`profile in {"open_xquant_default", "xquant_production"}`,
`return_type in {"simple", "log"}`,
`annualization_days > 0`,
`calmar_denominator == "max_drawdown"`,
`evaluation_window in {"full", "oos"}`,
and finite `risk_free_rate`.

- [ ] **Step 4: Verify GREEN**

Run the same two tests and then:

```bash
.venv/bin/python -m pytest tests/spec/test_validator.py tests/spec/test_execution_semantics.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/oxq/spec/schema.py src/oxq/spec/validator.py tests/spec/test_validator.py tests/spec/test_execution_semantics.py
git commit -m "feat(spec): add metrics profile section"
```

### Task 2: Profile-Aware Metrics Computation

**Files:**
- Create: `src/oxq/portfolio/metrics.py`
- Test: `tests/portfolio/test_metrics_profile.py`

- [ ] **Step 1: Write failing profile computation tests**

Add tests for:
- default profile matches existing `RunResult` simple-return annualized return, volatility, Sharpe, Sortino, and Calmar.
- `xquant_production` uses log returns and subtracts annual risk-free rate in Sharpe.
- changing `annualization_days` changes annualized return and volatility.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/portfolio/test_metrics_profile.py -q
```

Expected: failure because `oxq.portfolio.metrics` does not exist.

- [ ] **Step 3: Implement metrics profile module**

Implement:

```python
compute_profile_metrics(result: RunResult, config: MetricsSection, *, run_id: str) -> dict[str, Any]
```

Return existing metric keys plus:

```json
"metrics_profile": "...",
"metric_assumptions": {
  "return_type": "...",
  "risk_free_rate": 0.0,
  "annualization_days": 252,
  "calmar_denominator": "max_drawdown",
  "evaluation_window": "full"
}
```

Keep `open_xquant_default` values aligned with existing `RunResult` methods.
Use log returns for `return_type=log`, annualize mean return by
`annualization_days`, and subtract `risk_free_rate / annualization_days` from
period returns for Sharpe.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/portfolio/test_metrics_profile.py tests/portfolio/test_analytics.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/oxq/portfolio/metrics.py tests/portfolio/test_metrics_profile.py
git commit -m "feat(portfolio): compute metrics profiles"
```

### Task 3: Compiler Artifact Wiring

**Files:**
- Modify: `src/oxq/spec/compiler.py`
- Test: `tests/spec/test_compiler.py`

- [ ] **Step 1: Write failing compiler tests**

Add tests that compile/write artifacts with `metrics.profile: xquant_production`
and assert `metrics.json` contains:
- `metrics_profile: xquant_production`
- matching `metric_assumptions`
- `annualized_return`, `annualized_volatility`, `sharpe_ratio`, and
  `calmar_ratio` computed from the selected profile.

Add a compatibility test that a default spec writes `metrics_profile:
open_xquant_default` while preserving old metric values.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/spec/test_compiler.py::test_metrics_json_records_profile_assumptions tests/spec/test_compiler.py::test_default_metrics_profile_preserves_existing_values -q
```

Expected: failure because compiler still calls `RunResult` metric methods directly and does not write profile assumptions.

- [ ] **Step 3: Wire compiler to profile module**

Update `_build_metrics()` to call `compute_profile_metrics()`, then add
compiler-owned fields:
`strategy_id`, `run_id`, `trade_count`, `cost_paid`, `slippage_paid`, and OOS
metrics. OOS Sharpe should use the same `annualization_days` and return type
as the configured profile.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/spec/test_compiler.py tests/audit/test_research_bias.py tests/audit/test_reproducibility.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/oxq/spec/compiler.py tests/spec/test_compiler.py
git commit -m "feat(spec): write metrics profile artifacts"
```

### Task 4: Report Transparency

**Files:**
- Modify: `src/oxq/report/generator.py`
- Test: `tests/report/test_generator.py`

- [ ] **Step 1: Write failing report tests**

Add tests that generate a report from `metrics.json` containing
`metrics_profile` and `metric_assumptions`. Assert report includes:
- `### Metrics Profile`
- profile name
- return type
- risk-free rate
- annualization days
- Calmar denominator
- evaluation window
- a non-default profile note for `xquant_production`.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/report/test_generator.py::test_report_includes_metric_assumptions -q
```

Expected: failure because report does not render metric assumptions yet.

- [ ] **Step 3: Implement report rendering**

Add a metrics-profile subsection in section 5 before the metrics table. Use
`metrics["metric_assumptions"]` when present; otherwise infer the legacy
`open_xquant_default` assumptions for old run directories.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/report/test_generator.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/oxq/report/generator.py tests/report/test_generator.py
git commit -m "feat(report): show metric assumptions"
```

### Final Verification

- [ ] Run focused suite:

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy .venv/bin/python -m pytest tests/spec/test_validator.py tests/spec/test_execution_semantics.py tests/portfolio/test_metrics_profile.py tests/portfolio/test_analytics.py tests/spec/test_compiler.py tests/report/test_generator.py tests/audit/test_research_bias.py tests/audit/test_reproducibility.py -q
```

- [ ] Run lint on changed Python files:

```bash
git diff --name-only main...HEAD | rg '\.py$' | xargs .venv/bin/ruff check
```

- [ ] Run full suite:

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy .venv/bin/python -m pytest -q
```

- [ ] Push branch and open PR.
