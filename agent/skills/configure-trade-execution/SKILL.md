---
name: configure-trade-execution
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
  order_timing: next_session_open
  price_bar: next_session
  price_type: open
  initial_cash: 100000
  lot_size: 1
  lot_size_config:
    default: 1
    by_symbol: {}
  cash_annual_return: 0.0
  rebalance:
    frequency: daily
    interval_days: 1
    schedule: ""

cost:
  fee_rate: 0.001
  fee_min: 0.0
  slippage_rate: 0.001
```

Use side-specific costs only when confirmed:

```yaml
cost:
  fee_rate: 0.0003
  fee_min: 0.0
  buy_fee_rate: 0.0003
  sell_fee_rate: 0.0003
  sell_tax_rate: 0.001
  slippage_rate: 0.001
```

Validator constraints:

- `signal_time=close_t` with `trade_time=close_t` is fatal.
- `fill_price_mode=close` or `mid` with `signal_time=close_t` is fatal.
- Legacy `trade_time` / `fill_price_mode` must agree with explicit
  `order_timing` / `price_bar` / `price_type` when both are present.
- `fee_rate` and `slippage_rate` must each be positive.
- `buy_fee_rate` and `sell_fee_rate` may override `fee_rate` when the user
  explicitly confirms side-specific fees.
- `sell_tax_rate` and `stamp_tax` model sell-side tax. If both are set, they
  must match; otherwise use one confirmed field and leave the other empty or
  zero.
- The only zero-cost exception is when both values are exactly `0.0` and the
  spec declares explicit execution semantics for a replay-style validation run;
  preserve the warning.
- `lot_size` must be a positive integer.
- `lot_size_config.default` must be a positive integer.
- `lot_size_config.by_symbol` is parsed but not executable yet; leave it empty
  until by-symbol sizing is supported.
- `initial_cash` must be positive and finite.
- `cash_annual_return` must be non-negative and finite.
- `execution.rebalance.frequency=daily` uses `interval_days`.
- `frequency=weekly` requires `schedule: week_start`.
- `frequency=monthly` requires `schedule: month_start`.
- Do not use month-end or week-end calendar schedules in audited specs; those
  depend on future-row knowledge unless the runtime has explicit support.

Supported audited calendars are `XNYS`, `ARCX`, `XSHG`, and `XSHE`.
Use `lot_size_config.default: 100` for A-share examples.

## Cost Defaults Are Not Conclusions

If the user does not specify costs, use conservative placeholders only after
stating the assumption:

- US equity demo: `fee_rate=0.001`, `slippage_rate=0.001`, `lot_size=1`
- A-share demo: ask first; lot size is often `100`; commission, stamp tax, and
  slippage must be confirmed instead of inferred
- crypto demo: ask exchange and fee tier first

Do not silently choose zero fees or zero slippage.

## SDK Broker Pattern

```python
from decimal import Decimal

from oxq.trade.fees import PercentageFee, SideAwarePercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage

broker = SimBroker(
    fee_model=SideAwarePercentageFee(
        buy_rate=Decimal("0.0003"),
        sell_rate=Decimal("0.0003"),
        sell_tax_rate=Decimal("0.001"),
    ),
    slippage_model=PercentageSlippage(rate=Decimal("0.001")),
    fill_price_mode=FillPriceMode.NEXT_OPEN,
    market_calendar="XNYS",
)
```

## Red Lines

- Do not allow same-bar close signal and close fill in audited research.
- Do not call a result tradable unless execution costs were explicit.
- Do not compare two strategies with different cost assumptions without
  highlighting the difference.
- Do not compare two strategies with different fill price, calendar, lot size,
  or cash return assumptions without highlighting the difference.
