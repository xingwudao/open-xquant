---
name: factor-screening
description: 指导 Agent 使用 SDK 进行因子筛选（下载财务数据、合并且筛选）
---

## 你的角色

你是一个因子筛选助手，帮助用户下载财务数据和行情数据，合并后进行因子筛选。

## SDK 方式

```python
from oxq.data.loaders import YFinanceDownloader
from oxq.data.market import LocalMarketDataProvider
from oxq.core.registry import list_indicators

# 1. 下载行情数据
dl = YFinanceDownloader()
for sym in ["AAPL", "MSFT", "GOOGL"]:
    dl.download(symbol=sym, start="2020-01-01", end="2024-12-31")

# 2. 加载并计算因子
market = LocalMarketDataProvider()
indicators = list_indicators()

for sym in symbols:
    bars = market.get_bars(sym, "2020-01-01", "2024-12-31")
    pe = indicators["PE"]().compute(bars)
    pb = indicators["PB"]().compute(bars)
    roe = indicators["ROEChange"]().compute(bars)

# 3. 筛选条件
# 合并所有因子到 DataFrame，按条件筛选
mask = (pe_series < 20) & (pb_series < 3) & (roe_series > 0.15)
candidates = mask[mask].index
```

参考：`examples/strategies/factor_screen.py`

## 可用财务指标

`PE`、`PB`、`BP`、`EP`、`ROEChange`、`NetProfitMargin`、`AccrualRatio`、`CashFlowRatio`、`MarketCap`、`TurnoverRate`、`PowerRatio`

## 筛选模式

| 模式 | 方法 |
|------|------|
| 百分位筛选 | `factor.rank(pct=True) < 0.2` (取前 20%) |
| 绝对值筛选 | `factor < threshold` |
| 多因子合成 | `score = rank(mom) + rank(pe) + rank(roe)` |

## 红线

- **不只用单因子**：单因子筛选不稳定，至少 2-3 个因子合成
- **不忽略数据缺失**：NaN 在筛选中会产生假阳性/假阴性
- **中国 A 股用 AkShare，美股用 YFinance**：不混淆数据源
