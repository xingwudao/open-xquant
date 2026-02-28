"""SMA Crossover Strategy — end-to-end backtest example.

Strategy: Buy when SMA10 crosses above SMA50 (golden cross),
          sell when SMA10 drops below SMA50 (death cross).

Usage:
    # First download data
    python -c "from oxq.data import YFinanceDownloader; YFinanceDownloader().download('AAPL', '2023-01-01', '2024-12-31')"

    # Then run the strategy
    python examples/strategies/sma_crossover.py
"""

from oxq.backtest import BacktestEngine, SimBroker
from oxq.core import Strategy
from oxq.data import LocalMarketDataProvider
from oxq.indicators import SMA
from oxq.rules import EntryRule, ExitRule
from oxq.signals import Crossover
from oxq.universe import StaticUniverse

# ── Strategy Definition ──────────────────────────────────────────────────

strategy = Strategy(
    name="sma_crossover",
    hypothesis="短期均线上穿长期均线的标的在后续持有期内有正超额收益",
    universe=StaticUniverse(("AAPL",)),
    indicators={
        "sma_10": (SMA(), {"period": 10}),
        "sma_50": (SMA(), {"period": 50}),
    },
    signals={
        "sma_10_x_sma_50": (Crossover(), {"fast": "sma_10", "slow": "sma_50"}),
    },
    entry_rules=[EntryRule(signal="sma_10_x_sma_50", shares=100)],
    exit_rules=[ExitRule(fast="sma_10", slow="sma_50")],
)

# ── Run Backtest ─────────────────────────────────────────────────────────

engine = BacktestEngine()
result = engine.run(
    strategy,
    market=LocalMarketDataProvider(),
    broker=SimBroker(),
    start="2023-01-01",
    end="2024-12-31",
)

# ── Results ──────────────────────────────────────────────────────────────

print("=" * 60)
print(f"Strategy: {strategy.name}")
print(f"Hypothesis: {strategy.hypothesis}")
print("=" * 60)
print(f"Total Return:  {result.total_return():>8.2%}")
print(f"Sharpe Ratio:  {result.sharpe_ratio():>8.2f}")
print(f"Max Drawdown:  {result.max_drawdown():>8.2%}")
print(f"Total Trades:  {len(result.trades):>8d}")
print(f"Final Cash:    {result.portfolio.cash:>12,.2f}")
print()

# Show trades
if result.trades:
    print("Trades:")
    print("-" * 60)
    for fill in result.trades:
        print(
            f"  {fill.filled_at}  {fill.order.side:>4}  "
            f"{fill.order.shares:>4} {fill.order.symbol}  "
            f"@ {fill.filled_price:.2f}"
        )
