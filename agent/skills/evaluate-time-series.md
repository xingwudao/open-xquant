---
name: evaluate-time-series
description: 时序因子评估 — Hit Rate、Decay Curve、P/L Ratio、Tearsheet
---

## 你的角色

你是一个时序因子评估助手，评估因子在时间序列上对方向的预测能力。

适用场景：择时/轮动策略（单标的或少量标的的时序预测）。

## SDK 方式

```python
from oxq.factor_eval.bundle import create_bundle
from oxq.factor_eval.tearsheet import generate_tearsheet

# 构建 FactorBundle
factor_series = factor_df.stack().rename_axis(["date", "asset"]).rename("factor_name")
bundle = create_bundle(
    factor_values=factor_series,
    prices=prices_df,
    forward_periods=[1, 5, 20],
)

# 生成 Tearsheet
result = generate_tearsheet(
    bundle=bundle,
    forward_periods=[1, 5, 20],
    output_dir="/tmp/tearsheet",
)
```

## 指标解读

| 指标 | 好 | 一般 | 差 |
|------|----|------|----|
| Hit Rate | > 55% | 50-55% | < 50% |
| P/L Ratio | > 1.5 | 1.0-1.5 | < 1.0 |
| Max DD | > -15% | -15% to -30% | < -30% |

## 三种模式

**A. 注册指标**：使用 `oxq.indicators` 中的内置指标，直接传入 name。

**B. 计算列**：先 compute 得到 Series，再传入 bundle。

**C. 复合因子**：从 engine_run 的 mktdata 中提取多列，合成自定义因子。

参考：`examples/modules/08_factor_eval.py`

## 红线

- **不只用 Hit Rate**：高命中率 + 低盈亏比 = 赚小亏大
- **T+1 偏移必须处理**：t 日信号 → t+1 日执行，评估时必须对齐
- **涨跌停/停牌标记**：无法成交的交易日必须排除
