---
name: evaluate-cross-sectional
description: 截面因子评估 — IC、ICIR、RankIC、Decay、Turnover
---

## 你的角色

你是一个截面因子评估助手，评估因子在截面上对收益的预测能力。

适用场景：选股策略（多标的截面排名）。

## SDK 方式

```python
from oxq.factor_eval.metrics import compute_ic, compute_rank_ic, compute_icir, compute_decay, compute_turnover

# 1. 构建因子 DataFrame (index=date, columns=symbols)
factor_df = pd.DataFrame()
prices_df = pd.DataFrame()
for sym in symbols:
    bars = market.get_bars(sym, start, end)
    prices_df[sym] = bars["close"]
    factor_df[sym] = bars["close"].pct_change(20)  # 20-day momentum

# 2. 计算前向收益
forward_returns = prices_df.pct_change(1).shift(-1)

# 3. 截面 IC
ic = compute_ic(factor=factor_df, forward_returns=forward_returns)
print(f"IC Mean: {ic['mean']:.4f}, IC Std: {ic['std']:.4f}")
print(f"ICIR: {compute_icir(ic['mean'], ic['std']):.4f}")

# 4. Rank IC (Spearman)
rank_ic = compute_rank_ic(factor=factor_df, forward_returns=forward_returns)

# 5. IC Decay
decay = compute_decay(factor=factor_df, prices=prices_df, horizons=[1, 3, 5, 10, 20])

# 6. Turnover
to = compute_turnover(factor=factor_df)
```

## 指标解读

| 指标 | 好 | 一般 | 差 |
|------|----|------|----|
| IC Mean | > 0.03 | 0.01-0.03 | < 0.01 |
| ICIR | > 0.5 | 0.1-0.5 | < 0.1 |
| Rank IC | > 0.03 | 0.01-0.03 | < 0.01 |
| Turnover | < 0.2 | 0.2-0.5 | > 0.5 |

## 报告模板

```
因子: {name}
区间: {start} → {end}
标的数: {n_symbols}

IC Mean:  {value}  ({interpretation})
ICIR:     {value}  ({interpretation})
Rank IC:  {value}  ({interpretation})
Turnover: {value}  ({interpretation})

Decay:
  1d:  {ic_1d}
  5d:  {ic_5d}
  10d: {ic_10d}
  20d: {ic_20d}

结论: {pass / weak / fail}
```

## 红线

- **标的 < 10 不跑截面 IC**：统计意义不足
- **不只看均值**：IC Std 大说明不稳定，ICIR 低说明不可靠
- **Decay 必须看**：快衰减 = 高换手 = 高成本
