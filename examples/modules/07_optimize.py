"""Module example: Parameter Optimization.

Demonstrates GridSearch, WalkForward, and Time-Series Cross-Validation.
Finds optimal SMA parameters and detects overfitting via IS/OOS comparison.

Run: uv run python examples/modules/07_optimize.py
"""

from oxq.core.strategy import Strategy
from oxq.data.loaders import YFinanceDownloader
from oxq.data.market import LocalMarketDataProvider
from oxq.indicators import SMA
from oxq.optimize.paramset import ParameterSet
from oxq.optimize.search import GridSearch
from oxq.optimize.validation import TimeSeriesCV
from oxq.optimize.walk_forward import WalkForward
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.signals.crossover import Crossover
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse

print("Downloading SPY data...")
YFinanceDownloader().download(symbol="SPY", start="2018-01-01", end="2024-12-31")
market = LocalMarketDataProvider()


def _make_strategy() -> Strategy:
    crossover = Crossover()
    crossover.required_indicators = {
        "sma_10": (SMA(), {"column": "close", "period": 10}),
        "sma_50": (SMA(), {"column": "close", "period": 50}),
    }
    return Strategy(
        name="sma_opt",
        hypothesis="SMA crossover parameter tuning",
        benchmarks=["SPY"],
        universe=StaticUniverse(("SPY",)),
        signals={"golden_cross": (crossover, {"fast": "sma_10", "slow": "sma_50"})},
        portfolio=EqualWeightOptimizer(),
    )


def _broker_factory() -> SimBroker:
    return SimBroker(
        fee_model=PercentageFee(),
        slippage_model=PercentageSlippage(),
        fill_price_mode=FillPriceMode.NEXT_OPEN,
    )


# ===========================================================================
# Part 1: Grid Search
# ===========================================================================
print("=" * 50)
print("GRID SEARCH: SMA Crossover Period Optimization")
print("=" * 50)

paramset = ParameterSet(name="sma_tuning")
paramset.add("sma_10", "period", list(range(5, 30, 5)))
paramset.add("sma_50", "period", list(range(30, 100, 20)))
paramset.add_constraint("sma_10.period < sma_50.period")

combos = paramset.grid()
print(f"Parameter combinations: {len(combos)}")
print("Training period:        2018-01-01 → 2021-12-31")

result = GridSearch(paramset).run(
    strategy=_make_strategy(),
    market=market,
    broker_factory=_broker_factory,
    start="2018-01-01",
    end="2021-12-31",
    metric="sharpe_ratio",
)

print("\nTop 5 by Sharpe:")
print(f"{'Rank':<5} {'sma_10':<10} {'sma_50':<10} {'Sharpe':<10} {'Return':<10} {'MaxDD':<10}")
for i, trial in enumerate(result.top_n(5)):
    p = trial.params
    r = trial.run_result
    print(f"{i+1:<5} {p.get('sma_10',{}).get('period','?'):<10} "
          f"{p.get('sma_50',{}).get('period','?'):<10} "
          f"{r.sharpe_ratio():<10.3f} {r.total_return():<10.2%} {r.max_drawdown():<10.2%}")

best = result.best
print(f"\nBest: sma_10={best.params.get('sma_10',{}).get('period')} "
      f"sma_50={best.params.get('sma_50',{}).get('period')} "
      f"sharpe={best.metric_value:.3f}")

# ===========================================================================
# Part 2: Walk-Forward Analysis
# ===========================================================================
print(f"\n{'='*50}")
print("WALK-FORWARD: IS vs OOS Performance")
print("=" * 50)

wf = WalkForward(
    paramset=paramset,
    train_period="2Y",
    test_period="1Y",
    step="1Y",
    anchored=False,
)
wf_result = wf.run(
    strategy=_make_strategy(),
    market=market,
    broker_factory=_broker_factory,
    start="2018-01-01",
    end="2024-12-31",
    metric="sharpe_ratio",
)

print(f"Windows: {len(wf_result.windows)}")
print(f"{'Train':<22} {'IS Sharpe':<12} {'OOS Sharpe':<12}")
for w in wf_result.windows:
    train_range = f"{w.train_start}→{w.train_end}"
    is_s = w.in_sample_metric
    oos_s = w.oos_result.sharpe_ratio()
    print(f"{train_range:<22} {is_s:<12.3f} {oos_s:<12.3f}")

if hasattr(wf_result, "deterioration"):
    det = wf_result.deterioration()
    if det:
        print("\nOOS deterioration per metric:")
        for metric, value in det.items():
            direction = "worse" if value < 0 else "better"
            print(f"  {metric}: {value:+.1%} ({direction})")
    else:
        print("\nOOS stable")

# ===========================================================================
# Part 3: Time-Series Cross-Validation
# ===========================================================================
print(f"\n{'='*50}")
print("TIME-SERIES CROSS-VALIDATION")
print("=" * 50)

cv = TimeSeriesCV(n_splits=4, expanding=True)
cv_result = cv.cross_validate(
    strategy=_make_strategy(),
    market=market,
    broker_factory=_broker_factory,
    start="2018-01-01",
    end="2024-12-31",
    paramset=paramset,
    metric="sharpe_ratio",
)

print(f"Splits: {len(cv_result.splits)}")
for i, split in enumerate(cv_result.splits):
    s = split.split
    oos_s = split.oos_result.sharpe_ratio()
    print(f"  Split {i+1}: train={s.train_start}→{s.train_end}  "
          f"test={s.test_start}→{s.test_end}  "
          f"train_sharpe={split.in_sample_metric or 0:.3f}  test_sharpe={oos_s:.3f}")
