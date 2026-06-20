"""E2E Strategy: Momentum Top-N Rotation with spec workflow.

Complete end-to-end pipeline using the spec→backtest→audit→report workflow.
Demonstrates both SDK (programmatic) and CLI interaction.

Strategy: Every week, buy the ETF (from SPY, QQQ, IWM, GLD) with the highest
20-day return. Hold until next rebalance.

Run: uv run python examples/strategies/momentum_rotation_spec.py
"""

import json
from pathlib import Path

import yaml

from oxq.audit import audit_reproducibility, audit_research
from oxq.data.loaders import YFinanceDownloader
from oxq.observe.experiment_registry import add_experiment
from oxq.report import generate_report
from oxq.robustness import run_robustness
from oxq.spec import StrategySpec, compile_run, validate
from oxq.spec.schema import IndicatorDef

OUT_DIR = Path("/tmp/oxq_examples/e2e_momentum")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLS = ["SPY", "QQQ", "IWM", "GLD"]

# Download data for all symbols
print("Downloading data...")
downloader = YFinanceDownloader()
for sym in SYMBOLS:
    downloader.download(symbol=sym, start="2018-01-01", end="2025-12-31")
print("Done.\n")

# ===========================================================================
# Phase 1: Spec
# ===========================================================================
print("=" * 60)
print("PHASE 1: Strategy Spec")
print("=" * 60)

spec = StrategySpec.template(
    strategy_id="momentum_rotation_e2e",
    hypothesis="Past 20-day relative strength among liquid ETFs has short-term "
    "continuation. Weekly rotation to the strongest ETF captures momentum.",
)

spec.name = "Momentum Top-1 Rotation — E2E Example"
spec.universe.symbols = list(SYMBOLS)
spec.data.price_adjustment = "adjusted"

# Signal: 20-day return. The TopNRanking portfolio consumes the indicator
# directly and filters negative scores.
spec.signal.signal_time = "close_t"
spec.signal.indicators = {
    "momentum_20": IndicatorDef(type="NdayReturn", params={"column": "close", "period": 20}),
}

# Portfolio: Top-1 by momentum score
spec.portfolio.type = "TopNRanking"
spec.portfolio.params = {"score_col": "momentum_20", "n": 1, "filter_negative": True}

# Execution: weekly, next_open fill
spec.execution.trade_time = "next_open"
spec.execution.fill_price_mode = "next_open"
spec.execution.order_timing = "next_session_open"
spec.execution.price_bar = "next_session"
spec.execution.price_type = "open"
spec.execution.initial_cash = 100_000
spec.execution.rebalance.interval_days = 5
spec.execution.cash_annual_return = 0.0
spec.execution.lot_size_config.default = 1

# Costs
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001

# Metrics profile
spec.metrics.profile = "open_xquant_default"
spec.metrics.risk_free_rate = 0.0
spec.metrics.return_type = "simple"
spec.metrics.annualization_days = 252
spec.metrics.calmar_denominator = "max_drawdown"
spec.metrics.evaluation_window = "full"

# Validation
spec.benchmark.symbols = ["SPY"]
spec.validation.train_period = ["2018-01-01", "2021-12-31"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
spec.validation.required_oos = True

# Robustness
spec.robustness.cost_multiplier = [1.0, 2.0, 3.0]
spec.robustness.parameter_perturbation = {"momentum_20.period": [10, 20, 30, 60]}
spec.robustness.regime_analysis = True

# Decision policy
spec.decision_policy.reject_if = {
    "fatal_audit_findings": True,
    "oos_sharpe_lt": 0.3,
    "max_drawdown_lt": -0.40,
}
spec.decision_policy.promote_if = {"oos_sharpe_gte": 1.0, "max_drawdown_gte": -0.20}

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
print(f"Spec:    {spec_path.name}")
print(f"Hash:    {spec.compute_hash()}")

result = validate(spec)
print(f"Valid:   {result.status.upper()} ({len(result.errors)} errors, {len(result.warnings)} warnings)")
if result.status == "fail":
    for e in result.errors:
        print(f"  FATAL: {e['check']}: {e['message']}")
    raise SystemExit(1)

# ===========================================================================
# Phase 2: Backtest
# ===========================================================================
print(f"\n{'='*60}")
print("PHASE 2: Backtest")
print("=" * 60)

result, run_dir = compile_run(spec, out_dir=str(OUT_DIR / "runs"))

metrics = json.loads((run_dir / "metrics.json").read_text())
print(f"Run:     {run_dir.name}")
print(f"Return:  {metrics['total_return']:.2%}")
print(f"Sharpe:  {metrics['sharpe_ratio']:.2f}")
print(f"Max DD:  {metrics['max_drawdown']:.2%}")
print(f"Trades:  {metrics['trade_count']}")

# Show top 5 trades
import pandas as pd  # noqa: E402

trades_df = pd.read_csv(run_dir / "trades.csv")
print(f"\nTrade summary ({len(trades_df)} total):")
if len(trades_df) > 0:
    print(trades_df.head(5).to_string(index=False))

# ===========================================================================
# Phase 3: Audit
# ===========================================================================
print(f"\n{'='*60}")
print("PHASE 3: Audit")
print("=" * 60)

repro = audit_reproducibility(run_dir)
bias = audit_research(run_dir)
print(f"Reproducibility: {repro['status'].upper()}")
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
    if "baseline_sharpe" in t:
        print(f"         Sharpe: {t['baseline_sharpe']:.3f} → {t['perturbed_sharpe']:.3f}")

# ===========================================================================
# Phase 5: Report
# ===========================================================================
print(f"\n{'='*60}")
print("PHASE 5: Report & Experiment")
print("=" * 60)

report_md = generate_report(run_dir)
report_path = run_dir / "research_report.md"
report_path.write_text(report_md)

decision = "UNKNOWN"
for line in report_md.split("\n"):
    if line.startswith("**") and any(kw in line for kw in ["REJECT", "WATCHLIST", "CANDIDATE"]):
        decision = line.strip("*").strip()
        break
print(f"Decision: {decision}")

entry = add_experiment(run_dir, registry_path=OUT_DIR / "experiments.jsonl")
print(f"Registry: {entry['experiment_id']}")

print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Strategy:   {spec.strategy_id}")
print(f"  Universe:   {spec.universe.symbols}")
print(f"  Portfolio:  {spec.portfolio.type}")
print(f"  Decision:   {decision}")
print(f"  Report:     {report_path}")
print(f"  Registry:   {OUT_DIR / 'experiments.jsonl'}")
