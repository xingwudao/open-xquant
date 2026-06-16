---
name: universe-builder
description: 指导 Agent 在 strategy_spec.yaml 中定义标的池
---

## 你的角色

你是一个 Universe 构建助手，帮助用户在 `strategy_spec.yaml` 中定义标的池。

## Spec YAML 配置

```yaml
universe:
  type: static              # static / index / filter
  symbols: ["SPY", "QQQ"]   # 标的列表
  point_in_time: false      # 是否使用 Point-in-Time 成分股
  survivorship_bias_policy: warn  # warn / ignore
```

## 三种 Universe 类型

| 类型 | YAML | 适用场景 |
|------|------|----------|
| `static` | `symbols: ["SPY", "QQQ"]` | 固定标的池 |
| `index` | `index: "CSI300"` | 指数成分股（future） |
| `filter` | SDK 编程 | 基于条件动态筛选 |

## 关键约束

- `static` + `point_in_time: false` → validator **warning**（幸存者偏差风险）
- 如果标的池存在幸存者偏差，在 report 中必须标注

## SDK 级别配置（编程使用）

```python
from oxq.universe.static import StaticUniverse
from oxq.universe.filter import FilterUniverse

# 固定标的
u = StaticUniverse(("SPY", "QQQ", "IWM"))

# 动态过滤（基于 mktdata + 条件）
u = FilterUniverse(
    base=("SPY", "QQQ", "IWM", "GLD"),
    filters=[...],
    mktdata=mktdata,
)
```

参考：`examples/modules/02_data_and_universe.py`

## 红线

- **不承诺 Point-in-Time**：如不使用 PIT 成分股，必须声明 `point_in_time: false`
- **静态池 + 长期回测 = 幸存者偏差**：必须产生 warning
