---
name: trade-executor
description: >-
  Configure open-xquant execution timing, fees, slippage, lot size, fill price
  mode, and broker assumptions; use when users discuss trading costs or order
  execution.
---

# Trade Executor

You configure execution assumptions in `strategy_spec.yaml` or SDK brokers.

## Stable Spec Configuration

Use this for audited CLI backtests:

```yaml
signal:
  signal_time: close_t

execution:
  trade_time: next_open
  fill_price_mode: next_open
  initial_cash: 100000
  lot_size: 1
  rebalance:
    frequency: daily
    interval_days: 1

cost:
  fee_rate: 0.001
  fee_min: 0.0
  slippage_rate: 0.001
```

Validator constraints:

- `signal_time=close_t` with `trade_time=close_t` is fatal.
- `fill_price_mode=close` or `mid` with `signal_time=close_t` is fatal.
- `fee_rate <= 0` is fatal.
- `slippage_rate <= 0` is fatal.
- `lot_size` must be a positive integer.
- `initial_cash` must be positive and finite.

## Cost Defaults Are Not Conclusions

If the user does not specify costs, use conservative placeholders only after
stating the assumption:

- US equity demo: `fee_rate=0.001`, `slippage_rate=0.001`, `lot_size=1`
- A-share demo: ask first; lot size is often `100`
- crypto demo: ask exchange and fee tier first

Do not silently choose zero fees or zero slippage.

## SDK Broker Pattern

```python
from decimal import Decimal

from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage

broker = SimBroker(
    fee_model=PercentageFee(rate=Decimal("0.001")),
    slippage_model=PercentageSlippage(rate=Decimal("0.001")),
    fill_price_mode=FillPriceMode.NEXT_OPEN,
)
```

## Red Lines

- Do not allow same-bar close signal and close fill in audited research.
- Do not call a result tradable unless execution costs were explicit.
- Do not compare two strategies with different cost assumptions without
  highlighting the difference.
