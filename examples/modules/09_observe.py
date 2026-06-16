"""Module example: Observability — Tracing, Audit, and Monitoring.

Demonstrates how to trace engine execution, build audit records,
detect market states, and monitor strategy health.

Run: uv run python examples/modules/09_observe.py
"""

from oxq.core.engine import Engine
from oxq.core.strategy import Strategy
from oxq.data.loaders import YFinanceDownloader
from oxq.data.market import LocalMarketDataProvider
from oxq.indicators import SMA
from oxq.observe.audit import AuditRecord
from oxq.observe.detector import MarketStateDetector
from oxq.observe.monitor import StrategyMonitor
from oxq.observe.tracer import DefaultTracer
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.signals.crossover import Crossover
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse

print("Downloading SPY data...")
YFinanceDownloader().download(symbol="SPY", start="2020-01-01", end="2024-12-31")
market = LocalMarketDataProvider()

# Build strategy
crossover = Crossover()
crossover.required_indicators = {
    "sma_10": (SMA(), {"column": "close", "period": 10}),
    "sma_50": (SMA(), {"column": "close", "period": 50}),
}
strategy = Strategy(
    name="observe_demo",
    hypothesis="SMA crossover with tracing and monitoring",
    benchmarks=["SPY"],
    universe=StaticUniverse(("SPY",)),
    signals={"golden_cross": (crossover, {"fast": "sma_10", "slow": "sma_50"})},
    portfolio=EqualWeightOptimizer(),
)
broker = SimBroker(
    fee_model=PercentageFee(),
    slippage_model=PercentageSlippage(),
    fill_price_mode=FillPriceMode.NEXT_OPEN,
)

# ===========================================================================
# Part 1: Execution Tracing
# ===========================================================================
print("=" * 50)
print("EXECUTION TRACING")
print("=" * 50)

tracer = DefaultTracer()
engine = Engine()
result = engine.run(
    strategy=strategy,
    market=market,
    broker=broker,
    start="2020-01-01",
    end="2024-12-31",
    tracer=tracer,
)

print(f"Trace ID:    {tracer.trace_id}")
print(f"Run status:  {tracer.run_status}")
print(f"Total spans: {len(tracer.spans)}")

# Count spans by type
indicator_count = sum(1 for s in tracer.spans if "indicator" in s.component)
signal_count = sum(1 for s in tracer.spans if "signal" in s.component)
print(f"  Indicators: {indicator_count}")
print(f"  Signals:    {signal_count}")

# Show a sample span
if tracer.spans:
    sample = tracer.spans[0]
    print(f"\nSample span: {sample.component} ({sample.duration_ms:.1f}ms, status={sample.status})")

# ===========================================================================
# Part 2: Audit Record
# ===========================================================================
print(f"\n{'='*50}")
print("AUDIT RECORD (4-dimension hash)")
print("=" * 50)

record = AuditRecord.build(
    tracer=tracer,
    result=result,
    strategy_name=strategy.name,
    strategy_config={"universe": list(strategy.universe.get_universe("2024-01-01").symbols)},
    start_date="2020-01-01",
    end_date="2024-12-31",
    initial_cash=100_000.0,
)

print(f"  Mktdata hash:  {record.mktdata_hash}")
print(f"  Trades hash:   {record.trades_hash}")
print(f"  Equity hash:   {record.equity_hash}")
print(f"  Result hash:   {record.result_hash}")

# Save and reload
from pathlib import Path  # noqa: E402

audit_dir = Path("/tmp/oxq_examples")
audit_dir.mkdir(parents=True, exist_ok=True)
record.to_json(str(audit_dir / "audit_record.json"))
reloaded = AuditRecord.from_json(str(audit_dir / "audit_record.json"))
print(f"  Roundtrip OK:  {reloaded.mktdata_hash == record.mktdata_hash}")

# ===========================================================================
# Part 3: Market State Detection
# ===========================================================================
print(f"\n{'='*50}")
print("MARKET STATE DETECTION")
print("=" * 50)

detector = MarketStateDetector(result=result, symbols=("SPY",))
states = detector.states
if states is not None:
    state_counts = states.value_counts()
    print(f"  Vol median:    {detector.vol_median:.4f}")
    print(f"  High vol line: {detector.high_vol_line:.4f}")
    print(f"  Low vol line:  {detector.low_vol_line:.4f}")
    print(f"  State counts:  {dict(state_counts)}")

    # Performance by state
    try:
        perf = detector.performance_by_state(result)
        for state, metrics in perf.items():
            print(f"  {state:6s}: sharpe={metrics.get('sharpe_ratio', 0):.2f}  return={metrics.get('total_return', 0):.2%}")
    except Exception:
        pass

# ===========================================================================
# Part 4: Strategy Monitoring
# ===========================================================================
print(f"\n{'='*50}")
print("STRATEGY MONITORING")
print("=" * 50)

monitor = StrategyMonitor(
    result=result,
    benchmark="SPY",
    roll_window=63,
    min_bad_days=20,
)

summary = monitor.summary()
print(f"  Rolling Sharpe:    {summary.get('current_sharpe', 0):.3f} (63-day window)")
print(f"  Rolling Drawdown:  {summary.get('current_drawdown', 0):.2%}")
print(f"  Bad periods:       {len(monitor.bad_periods)}")
print(f"  Status:            {summary.get('status', 'unknown')}")

# Show worst bad period
if monitor.bad_periods:
    worst = sorted(monitor.bad_periods, key=lambda p: p.avg_sharpe)[0]
    print(f"  Worst period:      {worst.start} → {worst.end}, Sharpe={worst.avg_sharpe:.2f}")

print("\nObservability demo complete.")
