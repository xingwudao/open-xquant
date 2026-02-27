---
name: universe-builder
description: 指导 Agent 构建投资标的池（Universe）
tools_required: [universe_*, data_*]
---

## 工作流

1. 理解用户意图：什么市场？哪些标的？需要筛选条件吗？
2. 检查本地数据：调用 data_list_symbols 查看已有数据
3. 加载数据（如需要）：调用 data_load_symbols 下载行情
4. 构建 Universe：
   a. 手动指定标的 → universe_resolve_static
   b. 基于量价条件筛选 → universe_resolve_filter
5. 验证：调用 universe_inspect 检查数据可用性
6. 迭代：根据结果调整标的或筛选条件

## 决策指南

- 已知确定的标的列表 → universe_resolve_static（如：["AAPL", "GOOGL", "MSFT"]）
- 需要从一组标的中按条件筛选 → universe_resolve_filter
  - 最低价格：close >= 10（排除低价股）
  - 最小成交量：volume >= 1000000（流动性筛选）
  - 可组合多个条件（AND 关系）
- 构建 Universe 后，始终调用 universe_inspect 确认数据完整

## 筛选条件格式

每个 filter 包含三个字段：
- column：数据列名（如 close, volume, open, high, low）
- op：比较运算符（>=, <=, >, <, ==, !=）
- value：数值阈值

示例：筛选最新收盘价 >= 100 且成交量 >= 100 万的标的

```json
[
  {"column": "close", "op": ">=", "value": 100},
  {"column": "volume", "op": ">=", "value": 1000000}
]
```

## 常见场景

### 场景 1：手动指定标的
用户说"我要投资 AAPL 和 GOOGL"
→ universe_resolve_static(symbols=["AAPL", "GOOGL"])

### 场景 2：流动性筛选
用户说"从科技股中选出成交量大的"
→ 先 data_load_symbols 下载数据
→ universe_resolve_filter(base_symbols=[...], filters=[{"column": "volume", "op": ">=", "value": 1000000}])

### 场景 3：价格过滤
用户说"排除低于 10 元的股票"
→ universe_resolve_filter(base_symbols=[...], filters=[{"column": "close", "op": ">=", "value": 10}])
