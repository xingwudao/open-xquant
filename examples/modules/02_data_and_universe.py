"""Module example: Data download and universe construction.

Demonstrates downloading market data via YFinance/AkShare, inspecting data,
and constructing a universe for backtesting.

Run: uv run python examples/modules/02_data_and_universe.py
"""

from oxq.data.loaders import YFinanceDownloader
from oxq.data.market import LocalMarketDataProvider
from oxq.universe.static import StaticUniverse

SYMBOLS = ["SPY", "QQQ"]

# ---------------------------------------------------------------------------
# SDK: Download market data
# ---------------------------------------------------------------------------

print(f"Downloading data for {SYMBOLS}...")
downloader = YFinanceDownloader()
for symbol in SYMBOLS:
    downloader.download(
        symbol=symbol,
        start="2020-01-01",
        end="2025-12-31",
    )

# ---------------------------------------------------------------------------
# SDK: Read data and inspect
# ---------------------------------------------------------------------------

market = LocalMarketDataProvider()
for symbol in SYMBOLS:
    bars = market.get_bars(symbol, "2020-01-01", "2025-12-31")
    print(f"\n{symbol}:")
    print(f"  Rows:       {len(bars)}")
    print(f"  Date range: {bars.index[0].date()} → {bars.index[-1].date()}")
    print(f"  Columns:    {list(bars.columns)}")
    print(f"  Close:      {bars['close'].iloc[0]:.2f} → {bars['close'].iloc[-1]:.2f}")
    if bars["close"].isna().any():
        print(f"  Missing:    {bars['close'].isna().sum()} NaN values")

# ---------------------------------------------------------------------------
# SDK: Build a universe
# ---------------------------------------------------------------------------

universe = StaticUniverse(tuple(SYMBOLS))
snapshot = universe.get_universe(as_of_date="2025-01-01")
print("\nUniverse snapshot (2025-01-01):")
print(f"  Source:  {snapshot.source}")
print(f"  Symbols: {snapshot.symbols}")

# Universe history — list all snapshots across date range
history = universe.get_history("2024-01-01", "2025-01-01")
print(f"\nUniverse history (2024 → 2025): {len(history)} snapshots")
print(f"  First: {history[0].as_of_date}, Last: {history[-1].as_of_date}")
