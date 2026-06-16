"""Module example: Alpaca Paper Trading (Optional — requires API keys).

Demonstrates connecting to Alpaca Markets for paper trading:
- Account info
- Fetching live market data
- Submitting and canceling orders

Prerequisites:
  1. Sign up at https://alpaca.markets
  2. Set environment variables:
     export ALPACA_API_KEY="your_paper_api_key"
     export ALPACA_SECRET_KEY="your_paper_secret_key"
  3. Install: pip install open-xquant[live]

Run: uv run python examples/modules/10_alpaca_live.py
"""

import os
import sys

API_KEY = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

if not API_KEY or not SECRET_KEY:
    print("Alpaca API keys not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    print("  export ALPACA_API_KEY=PK...")
    print("  export ALPACA_SECRET_KEY=...")
    print("\nSign up for a free paper trading account at https://alpaca.markets")
    sys.exit(0)

# ===========================================================================
# Part 1: Connect and check account
# ===========================================================================
print("=" * 50)
print("ALPACA PAPER TRADING")
print("=" * 50)

from oxq.contrib.alpaca.client import AlpacaClient  # noqa: E402
from oxq.contrib.alpaca.market_data import AlpacaMarketDataProvider  # noqa: E402

# Connect (paper=True for paper trading)
client = AlpacaClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
account = client.get_account()
print("\nAccount status:")
print(f"  ID:          {account.get('id', '?')[:8]}...")
print(f"  Status:      {account.get('status', '?')}")
print(f"  Equity:      ${float(account.get('equity', 0)):,.2f}")
print(f"  Cash:        ${float(account.get('cash', 0)):,.2f}")
print(f"  Buying power: ${float(account.get('buying_power', 0)):,.2f}")
print(f"  PDT:         {account.get('pattern_day_trader', '?')}")

# ===========================================================================
# Part 2: Fetch live market data
# ===========================================================================
print(f"\n{'='*50}")
print("LIVE MARKET DATA")
print("=" * 50)

provider = AlpacaMarketDataProvider(api_key=API_KEY, secret_key=SECRET_KEY, feed="iex")

# Latest bar
try:
    latest = provider.get_latest("SPY")
    print(f"SPY latest:  O={latest['open']:.2f} H={latest['high']:.2f} "
          f"L={latest['low']:.2f} C={latest['close']:.2f} V={latest['volume']:.0f}")
except Exception as e:
    print(f"  Latest bar failed (market may be closed): {e}")

# Historical bars
try:
    bars = provider.get_bars("SPY", "2024-01-01", "2024-06-01")
    print(f"SPY 2024 H1: {len(bars)} bars, close range {bars['close'].iloc[0]:.2f} → {bars['close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"  Historical bars failed: {e}")

# ===========================================================================
# Part 3: List positions and open orders
# ===========================================================================
print(f"\n{'='*50}")
print("POSITIONS & ORDERS")
print("=" * 50)

positions = client.get_positions()
print(f"Open positions: {len(positions)}")
for pos in positions[:5]:
    print(f"  {pos['symbol']:6s} Qty={pos['qty']:>8s}  MktVal=${float(pos['market_value']):,.2f}  "
          f"P/L=${float(pos['unrealized_pl']):,.2f}")

open_orders = client.list_open_orders()
print(f"\nOpen orders: {len(open_orders)}")
for order in open_orders[:5]:
    print(f"  {order['id'][:8]}...  {order['symbol']} {order['side']} "
          f"Qty={order['qty']}  {order['type']}  Status={order['status']}")

# ===========================================================================
# Part 4: Place and cancel a test order (PAPER ONLY)
# ===========================================================================
print(f"\n{'='*50}")
print("TEST ORDER (paper only)")
print("=" * 50)

try:
    test_order = client.submit_order({
        "symbol": "SPY",
        "qty": "1",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    })
    order_id = test_order.get("id", "")
    print(f"  Submitted:  {order_id[:8]}...  {test_order.get('symbol')} {test_order.get('side')} "
          f"Qty={test_order.get('qty')}  Status={test_order.get('status')}")

    # Check status
    status = client.get_order(order_id)
    print(f"  Status:     {status.get('status')}")

    # Cancel if still open
    if status.get("status") == "accepted":
        client.cancel_order(order_id)
        print(f"  Canceled order {order_id[:8]}...")
except Exception as e:
    print(f"  Test order failed (expected if market closed): {e}")

print("\nClose connection...")
client.close()
provider.close()
print("Done.")
