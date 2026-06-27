"""E2E Strategy: SMA Crossover with spec workflow.

Complete end-to-end pipeline using the spec→backtest→audit→report workflow.
Demonstrates both SDK (programmatic) and CLI (shell command) interaction.

Strategy: Buy SPY/QQQ when 10-day SMA crosses above 50-day SMA.
Exit when 10-day SMA crosses below 50-day SMA.

Run: uv run python examples/strategies/sma_crossover_spec.py
"""

from pathlib import Path

import yaml

from oxq.audit import audit_reproducibility, audit_research
from oxq.observe.experiment_registry import add_experiment
from oxq.report import generate_report
from oxq.robustness import run_robustness
from oxq.spec import StrategySpec, compile_run, validate
from oxq.spec.schema import IndicatorDef, SignalRuleDef

OUT_DIR = Path("/tmp/oxq_examples/e2e_sma_crossover")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Phase 1: Create and validate the strategy spec
# ===========================================================================
print("=" * 60)
print("PHASE 1: Spec Creation & Validation")
print("=" * 60)

spec = StrategySpec.template(
    strategy_id="sma_crossover_e2e",
    hypothesis="Short-term SMA (10-day) crossing above long-term SMA (50-day) "
    "signals upward momentum. The reverse crossover signals exit.",
)

spec.name = "SMA Crossover — E2E Example"
spec.universe.symbols = ["SPY", "QQQ"]
spec.data.price_adjustment = "adjusted"

# Signal configuration
spec.signal.signal_time = "close_t"
spec.signal.indicators = {
    "sma_fast": IndicatorDef(type="SMA", params={"column": "close", "period": 10}),
    "sma_slow": IndicatorDef(type="SMA", params={"column": "close", "period": 50}),
}
spec.signal.rules = {
    "golden_cross": SignalRuleDef(type="Crossover", params={"fast": "sma_fast", "slow": "sma_slow"}),
}

# Execution
spec.execution.trade_time = "next_open"
spec.execution.fill_price_mode = "next_open"
spec.execution.order_timing = "next_session_open"
spec.execution.price_bar = "next_session"
spec.execution.price_type = "open"
spec.execution.initial_cash = 100_000
spec.execution.cash_annual_return = 0.0
spec.execution.lot_size_config.default = 1

# Costs (non-zero required)
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001

# Metrics profile
spec.metrics.profile = "open_xquant_default"
spec.metrics.risk_free_rate = 0.0
spec.metrics.return_type = "simple"
spec.metrics.annualization_days = 252
spec.metrics.calmar_denominator = "max_drawdown"
spec.metrics.evaluation_window = "full"

# Benchmark and validation
spec.benchmark.symbols = ["SPY"]
spec.validation.train_period = ["2018-01-01", "2021-12-31"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
spec.validation.required_oos = True

# Robustness configuration
spec.robustness.cost_multiplier = [1.0, 2.0]
spec.robustness.parameter_perturbation = {
    "sma_fast.period": [5, 10, 20],
    "sma_slow.period": [30, 50, 80],
}
spec.robustness.regime_analysis = True

# Decision policy
spec.decision_policy.reject_if = {"fatal_audit_findings": True, "oos_sharpe_lt": 0.3}
spec.decision_policy.promote_if = {"oos_sharpe_gte": 1.0}

# Write spec
spec_path = OUT_DIR / "strategy_spec.yaml"
spec_yaml = spec.to_dict()
spec_yaml.setdefault("execution", {}).update(
    {
        "cash_annual_return": spec.execution.cash_annual_return,
        "lot_size_config": {
            "default": spec.execution.lot_size_config.default,
            "by_symbol": dict(spec.execution.lot_size_config.by_symbol),
        },
    }
)
spec_yaml["metrics"] = {
    "profile": spec.metrics.profile,
    "risk_free_rate": spec.metrics.risk_free_rate,
    "return_type": spec.metrics.return_type,
    "annualization_days": spec.metrics.annualization_days,
    "calmar_denominator": spec.metrics.calmar_denominator,
    "evaluation_window": spec.metrics.evaluation_window,
}
spec_path.write_text(
    yaml.dump(spec_yaml, sort_keys=False, allow_unicode=True, default_flow_style=False),
    encoding="utf-8",
)
print(f"Spec:    {spec_path}")
print(f"Hash:    {spec.compute_hash()}")

# Validate
result = validate(spec)
print(f"Valid:   {result.status.upper()} ({len(result.errors)} errors, {len(result.warnings)} warnings)")
if result.status == "fail":
    for e in result.errors:
        print(f"  FATAL: {e['check']}: {e['message']}")
    raise SystemExit(1)
for w in result.warnings:
    print(f"  WARN:  {w['check']}")

# ===========================================================================
# Phase 2: Backtest
# ===========================================================================
print(f"\n{'='*60}")
print("PHASE 2: Backtest")
print("=" * 60)

print(f"Running backtest for '{spec.strategy_id}'...")
result, run_dir = compile_run(spec, out_dir=str(OUT_DIR / "runs"))

print(f"\nRun:     {run_dir.name}")
print(f"Return:  {result.total_return():.2%}")
print(f"Sharpe:  {result.sharpe_ratio():.2f}")
print(f"Max DD:  {result.max_drawdown():.2%}")
print(f"Trades:  {len(result.trades)}")

# ===========================================================================
# Phase 3: Audit
# ===========================================================================
print(f"\n{'='*60}")
print("PHASE 3: Audit")
print("=" * 60)

repro = audit_reproducibility(run_dir)
print(f"Reproducibility: {repro['status'].upper()} ({repro['fatal_count']} fatal)")

bias = audit_research(run_dir)
print(f"Research Bias:   {bias['status'].upper()} ({bias['fatal_count']} fatal, {bias['warning_count']} warnings)")
for c in bias["checks"]:
    if c["status"] == "fail":
        print(f"  [{c['severity']:7s}] FAIL {c['id']}: {c['message'][:80]}")

# ===========================================================================
# Phase 4: Robustness
# ===========================================================================
print(f"\n{'='*60}")
print("PHASE 4: Robustness")
print("=" * 60)

robust = run_robustness(run_dir)
print(f"Status:  {robust['status'].upper()}")
for t in robust["tests"]:
    if t["status"] != "pass":
        print(f"  [{t['status'].upper():5s}] {t['name']}: {t.get('message', '')[:80]}")

# ===========================================================================
# Phase 5: Report & Experiment
# ===========================================================================
print(f"\n{'='*60}")
print("PHASE 5: Report & Experiment")
print("=" * 60)

report_md = generate_report(run_dir)
report_path = run_dir / "research_report.md"
report_path.write_text(report_md)

# Extract decision
decision = "UNKNOWN"
for line in report_md.split("\n"):
    if line.startswith("**") and ("REJECT" in line or "WATCHLIST" in line or "CANDIDATE" in line):
        decision = line.strip("*").strip()
        break
print(f"Decision: {decision}")

# Register experiment
entry = add_experiment(run_dir, registry_path=OUT_DIR / "experiments.jsonl")
print(f"Registry: {entry['experiment_id']}")

# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Spec:       {spec_path}")
print(f"  Run:        {run_dir}/")
print(f"  Report:     {report_path}")
print(f"  Registry:   {OUT_DIR / 'experiments.jsonl'}")
print(f"  Decision:   {decision}")

# ---------------------------------------------------------------------------
# CLI equivalent (one-shot)
# ---------------------------------------------------------------------------
print(f"""
{'='*60}
CLI + Agent pipeline:
  oxq spec validate {spec_path}
  oxq backtest run {spec_path} --out {OUT_DIR}/runs/auto --allow-unaudited --json
  oxq audit research {run_dir}/
  oxq robustness run {run_dir}/
  /quant-report {run_dir}/
  oxq experiment add {run_dir}/
{'='*60}
""")
