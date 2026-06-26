---
name: manage-live-trading
description: >-
  Connect open-xquant to Alpaca paper or live trading; use only when users
  explicitly ask for broker connectivity, account checks, live data, or order
  submission.
---

# Live Trader

You handle broker connectivity with strict safety gates.

## Preconditions

Install live dependencies:

```bash
uv sync --extra live
```

Set credentials:

```bash
export ALPACA_API_KEY="..."
export ALPACA_SECRET_KEY="..."
```

Default to `paper=True`. Never connect to live endpoints unless the user
explicitly asks for live trading and confirms the risk.

## Account Check

```python
from oxq.contrib.alpaca.client import AlpacaClient

client = AlpacaClient(paper=True)
account = client.get_account()
positions = client.get_positions()
print(account)
print(positions)
client.close()
```

## Market Data

```python
from oxq.contrib.alpaca.market_data import AlpacaMarketDataProvider

provider = AlpacaMarketDataProvider(feed="iex")
bars = provider.get_bars("SPY", "2024-01-01", "2024-06-01")
print(bars.tail())
provider.close()
```

## Order Submission

Before submitting any order, show:

- paper or live mode
- symbol
- side
- quantity
- order type
- time in force
- estimated risk

Then require explicit user confirmation.

```python
from oxq.contrib.alpaca.client import AlpacaClient

client = AlpacaClient(paper=True)
order = client.submit_order({
    "symbol": "SPY",
    "qty": "1",
    "side": "buy",
    "type": "market",
    "time_in_force": "day",
})
print(order)
client.close()
```

## LiveBroker

Use `LiveBroker` only when integrating with the engine-level broker protocol.
It starts a trade-update stream, so close it when finished:

```python
from oxq.trade.live_broker import LiveBroker

broker = LiveBroker(paper=True)
print(broker.get_account())
broker.close()
```

## Red Lines

- Do not submit live orders without explicit user confirmation.
- Do not batch orders unless the user explicitly reviewed the full list.
- Do not store API keys in files.
- Do not turn a backtest signal directly into an order without risk checks.
- Do not default to `paper=False`.
