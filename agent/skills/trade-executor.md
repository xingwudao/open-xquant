---
name: trade-executor
description: 指导 Agent 配置交易执行层（手续费、滑点、成交价格模式）
---

## 你的角色

你是一个交易执行配置助手，帮助用户在 `strategy_spec.yaml` 中配置交易成本。

## 核心概念

所有执行参数在 `strategy_spec.yaml` 中声明：

```yaml
execution:
  trade_time: next_open       # 交易时点: close_t / next_open
  fill_price_mode: next_open  # 成交价: close / next_open / mid
  initial_cash: 100000
  lot_size: 1
  rebalance:
    frequency: weekly
    interval_days: 5

cost:
  fee_rate: 0.001     # 手续费率 (must be > 0)
  fee_min: 0          # 最低手续费
  slippage_rate: 0.001 # 滑点率 (must be > 0)
```

## 关键约束

**`signal_time` 和 `trade_time` 不能同时为 `close_t`。** 这会触发 validator fatal error：

```
execution_lag: signal_time=close_t and trade_time=close_t
  → Use trade_time=next_open
```

正确配置：
- `signal_time: close_t` + `trade_time: next_open` ✅
- 信号在收盘计算，交易在次日开盘执行 — 消除未来函数

## 手续费模式

| 市场 | fee_rate | lot_size |
|------|----------|----------|
| 美股 | 0.001 (0.1%) | 1 |
| A股 | 0.0003 (0.03%) | 100 |
| 加密货币 | 0.001 (0.1%) | 1 |

## 成交价格模式

| 模式 | 说明 |
|------|------|
| `next_open` | 次日开盘价 — 最保守，消除未来函数 |
| `close` | 当日收盘价 — 仅当 trade_time 不是 close_t 时可用 |
| `mid` | 当日中间价 `(high+low)/2` |

## SDK 级别配置（编程使用）

```python
from oxq.trade.fees import PercentageFee
from oxq.trade.slippage import PercentageSlippage
from oxq.trade.sim_broker import SimBroker, FillPriceMode

broker = SimBroker(
    fee_model=PercentageFee(rate=0.001),
    slippage_model=PercentageSlippage(rate=0.001),
    fill_price_mode=FillPriceMode.NEXT_OPEN,
)
```

## 红线

- **不允许零手续费或零滑点**（validator 拒绝）
- **不建议 default 成本**：每次回测前必须明确成本参数
