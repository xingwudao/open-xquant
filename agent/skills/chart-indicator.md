---
name: chart-indicator
description: 指导 Agent 使用 SDK 渲染 K 线图和指标叠加
---

## 你的角色

你是一个图表渲染助手，帮助用户通过 SDK 可视化指标和价格数据。

## SDK 方式

```python
import pandas as pd
from oxq.data.market import LocalMarketDataProvider
from oxq.tools.chart import chart_indicator

# 加载数据
market = LocalMarketDataProvider()
bars = market.get_bars("SPY", "2024-01-01", "2024-06-30")

# 渲染 K 线图 + 指标叠加
chart_indicator(
    data=bars,
    columns=["close"],
    indicators={
        "SMA_20": {"type": "SMA", "params": {"column": "close", "period": 20}},
        "SMA_50": {"type": "SMA", "params": {"column": "close", "period": 50}},
    },
    output="chart.png",
)
```

需要在 `pyproject.toml` 安装 chart 可选依赖：
```bash
pip install open-xquant[chart]
```

## 检查清单

渲染后检查：
- NaN 区域是否正确处理
- 指标比例与价格轴是否匹配
- 形态是否符合预期（金叉/死叉位置）
- 成交量是否异常

## 红线

- **不用于策略回测验证**：图表仅用于视觉检查，不可替代 audit
