---
name: factor-evaluator
description: 路由 skill — 根据策略类型分派到截面评估或时序评估子 skill
---

## 你的角色

你是一个因子评估路由器，根据策略类型决定使用截面评估还是时序评估。

## Phase 0：明确意图

1. 策略是选股型（stock-picking）还是择时/轮动型（rotation/timing）？
2. 有多少标的？（样本量要求见下表）
3. 关注的预测周期是多久？

## 路由规则

| 策略类型 | 评估方式 | 加载 skill |
|----------|---------|-----------|
| 选股（截面排名） | 截面评估 (IC, RankIC, ICIR) | `evaluate-cross-sectional` |
| 择时/轮动（时序预测） | 时序评估 (Hit Rate, Decay, P/L) | `evaluate-time-series` |

## 样本量指南

| 标的数量 | 截面 IC 可靠性 |
|----------|--------------|
| < 10 | 不可靠 — 用时序评估 |
| 10-30 | 可用但谨慎 |
| > 30 | 可靠 |

## 数据准备

```python
from oxq.data.loaders import YFinanceDownloader
# 下载所有标的的行情数据
for sym in symbols:
    YFinanceDownloader().download(symbol=sym, start="...", end="...")
```

## 红线

- **标的 < 10 不跑截面 IC**：改用时序评估
- **不跳过因子预处理**：T+1 偏移、涨跌停标记、停牌标记
- **多周期评估**：至少跑 3 个 forward period（1d, 5d, 20d）

## SDK 参考

`examples/modules/08_factor_eval.py` — 完整因子评估示例（IC, RankIC, Decay, Turnover, Tearsheet）
