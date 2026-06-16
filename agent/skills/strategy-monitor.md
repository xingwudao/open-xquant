---
name: strategy-monitor
description: 指导 Agent 进行策略健康监控、审计验证、市场状态诊断和实验日志记录
---

## 你的角色

你是一个策略监控和审计助手，帮助用户通过 CLI 审计回测结果、监控策略健康、诊断市场状态。

## Phase 1：审计

```bash
# 可复现性审计 — 验证 hash 一致性
oxq audit reproducibility runs/<run_id>/

# 研究偏差审计 — 检测常见回测陷阱
oxq audit research runs/<run_id>/
```

偏差审计检查项：

| 检查项 | 级别 | 说明 |
|--------|------|------|
| execution_lag | fatal | 信号时间与成交时间冲突 |
| cost_model | fatal | 零手续费/零滑点 |
| oos_required | fatal | 无样本外验证 |
| benchmark_present | warning | 无基准 |
| static_universe_survivorship | warning | 幸存者偏差 |
| parameter_count | warning | 参数过多 |
| trade_count | warning | 交易次数不足 |
| drawdown_tail | warning | 回撤过大 |
| missing_data | warning | 数据缺失 |

## Phase 2：稳健性测试

```bash
oxq robustness run runs/<run_id>/
```

测试内容：
- 成本加倍 (cost x2)
- IS/OOS 对比
- 参数扰动配置检查

## Phase 3：报告

```bash
oxq report write runs/<run_id>/
```

报告包含 Executive Decision：REJECT / WATCHLIST / PAPER TRADING CANDIDATE。

## Phase 4：实验登记

```bash
oxq experiment add runs/<run_id>/
```

防止选择性记忆，每次研究都记录到 `experiments.jsonl`。

## SDK 级别监控（编程使用）

```python
from oxq.observe.monitor import StrategyMonitor
from oxq.observe.detector import MarketStateDetector

# 滚动指标监控
monitor = StrategyMonitor(result, benchmark="SPY", roll_window=63)
print(monitor.summary())  # current_sharpe, current_drawdown, bad_periods

# 市场状态检测
detector = MarketStateDetector(result, symbols=("SPY",))
print(detector.states.value_counts())  # high / normal / low

# 按市场状态分组表现
perf = detector.performance_by_state(result)
```

## 红线

- **审计失败不能美化**：fatal audit → REJECT，不允许绕过
- **Bad period 必须记录**：检测到策略恶化期必须写入实验日志
- **不允许修改策略代码来通过审计**
