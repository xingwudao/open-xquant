"""Module example: Signals and Rules.

Demonstrates 8 signal types and 9 rule types via SDK.
Signals are pure functions that output boolean or BUY/SELL/HOLD intent.
Each rule is a state machine that outputs constraints on the portfolio.

Run: uv run python examples/modules/06_signals_and_rules.py
"""

from oxq.core.engine import Engine
from oxq.core.strategy import Strategy
from oxq.data.loaders import YFinanceDownloader
from oxq.data.market import LocalMarketDataProvider
from oxq.indicators import ROC, SMA
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.rules.constraint import MaxHoldingsRule, RebalanceFrequencyRule
from oxq.rules.exit import ExitRule
from oxq.rules.order import StopLossRule, TakeProfitRule, TrailingStopRule
from oxq.rules.risk import DailyLossLimitRisk, MaxDrawdownRisk
from oxq.signals.comparison import Comparison
from oxq.signals.composite import Composite
from oxq.signals.crossover import Crossover
from oxq.signals.formula import Formula
from oxq.signals.peak import Peak
from oxq.signals.roc_timing import ROCTiming
from oxq.signals.threshold import Threshold
from oxq.signals.timestamp import Timestamp
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse

# Download data once
print("Downloading SPY data...")
YFinanceDownloader().download(symbol="SPY", start="2020-01-01", end="2024-12-31")
market = LocalMarketDataProvider()
bars = market.get_bars("SPY", "2020-01-01", "2024-12-31")
print(f"  {len(bars)} bars loaded\n")

# ===========================================================================
# Part 1: Signal Types
# ===========================================================================
print("=" * 50)
print("SIGNAL TYPES (8 total)")
print("=" * 50)

# Compute indicators first
bars["sma_10"] = SMA().compute(bars, column="close", period=10)
bars["sma_50"] = SMA().compute(bars, column="close", period=50)

# 1. Crossover — detect when fast crosses above slow
print("\n1. Crossover (sma_10 crosses above sma_50):")
crossover = Crossover().compute(bars, fast="sma_10", slow="sma_50")
print(f"   Cross-up days: {crossover.sum()}")

# 2. Threshold — detect when column exceeds threshold
print("\n2. Threshold (close > 400):")
threshold = Threshold().compute(bars, column="close", threshold=400, relationship="gt")
print(f"   Days close > 400: {threshold.sum()}")

# 3. Comparison — compare two columns
print("\n3. Comparison (sma_10 > sma_50):")
comparison = Comparison().compute(bars, left="sma_10", right="sma_50", relationship="gt")
print(f"   Days sma_10 > sma_50: {comparison.sum()}")

# 4. Formula — evaluate a pandas expression
print("\n4. Formula (close > sma_50 and volume > 50_000_000):")
formula = Formula().compute(bars, expr="(close > sma_50) & (volume > 50_000_000)")
print(f"   Days matching: {formula.sum()}")

# 5. Peak — detect local peaks/troughs
print("\n5. Peak (close peak detection, order=5):")
peaks = Peak().compute(bars, column="close", kind="peak", order=5)
troughs = Peak().compute(bars, column="close", kind="trough", order=5)
print(f"   Peaks: {peaks.sum()}, Troughs: {troughs.sum()}")

# 6. Composite — AND/OR combination of multiple signals
print("\n6. Composite (crossover AND close > 400):")
bars["_crossover"] = crossover
bars["_threshold"] = threshold
composite = Composite().compute(bars, signals=["_crossover", "_threshold"], logic="and")
print(f"   Both true: {composite.sum()}")

# 7. Timestamp — time-based conditions
print("\n7. Timestamp (month_start):")
ts_signal = Timestamp().compute(bars, rule="month_start")
print(f"   Month-start days: {ts_signal.sum()}")
ts_qend = Timestamp().compute(bars, rule="quarter_end")
print(f"   Quarter-end days: {ts_qend.sum()}")

# 8. ROCTiming — categorical BUY/SELL/HOLD timing intent
print("\n8. ROCTiming (ROC fixed thresholds):")
bars["roc_20"] = ROC().compute(bars, column="close", period=20)
roc_timing = ROCTiming().compute(
    bars,
    column="roc_20",
    mode="fixed",
    bottom=-5.0,
    top=5.0,
)
print(f"   Values: {sorted(set(roc_timing.dropna()))}")
print("   BUY means target long, SELL means target cash, HOLD preserves position.")

# ===========================================================================
# Part 2: Rule Types
# ===========================================================================
print(f"\n{'='*50}")
print("RULE TYPES (9 total)")
print("=" * 50)

# Build a simple strategy for backtesting
crossover_sig = Crossover()
crossover_sig.required_indicators = {
    "sma_10": (SMA(), {"column": "close", "period": 10}),
    "sma_50": (SMA(), {"column": "close", "period": 50}),
}
strategy = Strategy(
    name="rule_demo",
    hypothesis="Demo strategy for rule testing",
    benchmarks=["SPY"],
    universe=StaticUniverse(("SPY",)),
    signals={"golden_cross": (crossover_sig, {"fast": "sma_10", "slow": "sma_50"})},
    portfolio=EqualWeightOptimizer(),
)
broker = SimBroker(
    fee_model=PercentageFee(),
    slippage_model=PercentageSlippage(),
    fill_price_mode=FillPriceMode.NEXT_OPEN,
)

# Run with all 9 rules
all_rules = [
    ExitRule(fast="sma_10", slow="sma_50"),
    StopLossRule(threshold=0.05),
    TakeProfitRule(threshold=0.20),
    TrailingStopRule(trail_pct=0.05),
    MaxDrawdownRisk(max_drawdown=0.15),
    DailyLossLimitRisk(max_daily_loss=0.03),
    MaxHoldingsRule(max_holdings=5),
    RebalanceFrequencyRule(interval_days=5),
]

engine = Engine()
result = engine.run(
    strategy=strategy,
    market=market,
    broker=broker,
    rules=all_rules,
    start="2020-01-01",
    end="2024-12-31",
)

print("\nBacktest with all rules applied:")
print(f"  Trades:     {len(result.trades)}")
print(f"  Max DD:     {result.max_drawdown():.2%}")
print(f"  Sharpe:     {result.sharpe_ratio():.2f}")

# Show rule names and when they fire (summary)
print("\nActive rules in the pipeline:")
for rule in all_rules:
    rule_type = type(rule).__name__
    print(f"  {rule_type:25s} — {rule.__doc__ and rule.__doc__.split(chr(10))[0].strip()}")
