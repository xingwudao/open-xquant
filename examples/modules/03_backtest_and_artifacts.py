"""Module example: Backtest execution and artifact inspection.

Demonstrates compiling a strategy_spec.yaml, running a backtest,
and reading the standardized output artifacts.

Run: uv run python examples/modules/03_backtest_and_artifacts.py
"""

import json
from pathlib import Path

import pandas as pd

from oxq.spec import StrategySpec, compile_run

OUT_DIR = Path("/tmp/oxq_examples/03_backtest")
SPEC_FILE = Path("/tmp/oxq_examples/01_spec/strategy_spec.yaml")

# ---------------------------------------------------------------------------
# SDK: Load and compile the spec from file
# ---------------------------------------------------------------------------

if not SPEC_FILE.exists():
    print("Run 01_spec_and_validate.py first to create the spec file.")
    print(f"  Expected: {SPEC_FILE}")
    raise SystemExit(1)

spec = StrategySpec.from_yaml(str(SPEC_FILE))
print(f"Loaded spec: {spec.strategy_id}")
print(f"  Hash:      {spec.compute_hash()}")

# ---------------------------------------------------------------------------
# SDK: Compile + Run backtest
# ---------------------------------------------------------------------------

print(f"\nRunning backtest for '{spec.strategy_id}'...")
result, run_dir = compile_run(spec, out_dir=str(OUT_DIR))

print(f"\nRun complete: {run_dir}/")
print(f"  Total Return:     {result.total_return():.2%}")
print(f"  Annualized Ret:   {result.annualized_return():.2%}")
print(f"  Sharpe Ratio:     {result.sharpe_ratio():.2f}")
print(f"  Max Drawdown:     {result.max_drawdown():.2%}")
print(f"  Trades:           {len(result.trades)}")
final_value = float(
    sum(p.shares * float(p.avg_cost) for p in result.portfolio.positions.values())
) + float(result.portfolio.cash)
print(f"  Final Portfolio:  ${final_value:,.2f}")

# ---------------------------------------------------------------------------
# SDK: Read standardized artifacts
# ---------------------------------------------------------------------------

print(f"\nArtifacts in {run_dir}/:")
for fname in sorted(p.name for p in run_dir.iterdir()):
    print(f"  {fname}")

# Read metrics
metrics = json.loads((run_dir / "metrics.json").read_text())
print("\nMetrics from metrics.json:")
print(f"  sharpe_ratio:  {metrics['sharpe_ratio']:.2f}")
print(f"  max_drawdown:  {metrics['max_drawdown']:.2%}")
print(f"  trade_count:   {metrics['trade_count']}")

# Read target weights for allocation-level baseline comparisons
target_weights = pd.read_csv(run_dir / "target_weights.csv")
print(f"\nTarget weights ({len(target_weights)} rows):")
print(target_weights.head(5).to_string(index=False))

# Read artifact hashes used by reproducibility audit
artifact_hashes = json.loads((run_dir / "artifact_hashes.json").read_text())
print("\nArtifact hash keys:")
for key in sorted(artifact_hashes):
    print(f"  {key}")

eq = pd.read_csv(run_dir / "equity_curve.csv")
print(f"\nEquity curve ({len(eq)} bars):")
print(eq.head(3).to_string(index=False))
print("  ...")
print(eq.tail(3).to_string(index=False))

# ---------------------------------------------------------------------------
# CLI equivalents
# ---------------------------------------------------------------------------
print(f"""
{'='*60}
CLI equivalents:
  oxq strategy compile {SPEC_FILE}
  oxq backtest run {SPEC_FILE} --out runs/auto --allow-unaudited --json
{'='*60}
""")
