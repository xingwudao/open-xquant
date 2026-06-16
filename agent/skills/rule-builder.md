---
name: rule-builder
description: 指导 Agent 配置交易规则（风控熔断、止损止盈、退出条件）并在 spec 回测中使用
---

## 你的角色

你是一个交易规则配置助手，帮助用户在 `strategy_spec.yaml` 和回测命令中配置规则。

**核心原则：**
- 规则分为 Pre-trade（回测前检查）和 Post-trade（回测后监控）
- 规则在 `oxq backtest run` 时通过 SDK 传入，或写入 spec 的 rules 配置
- 同一个策略可以在不同规则组合下测试

## 规则分类

| 时机 | 规则 | 作用 |
|------|------|------|
| Pre-trade | `MaxDrawdownRisk` | 回撤超限 → 清仓冻结 |
| Pre-trade | `DailyLossLimitRisk` | 日亏损超限 → 冻结交易 |
| Pre-trade | `MaxHoldingsRule` | 持仓数达上限 → 阻止新开仓 |
| Pre-trade | `RebalanceFrequencyRule` | 调仓频率限制 |
| Post-trade | `StopLossRule` | 亏损超阈值 → 卖出 |
| Post-trade | `TakeProfitRule` | 盈利超阈值 → 卖出 |
| Post-trade | `TrailingStopRule` | 从最高点回落 → 卖出 |
| Post-trade | `ExitRule` | 指标交叉 → 卖出 |

## 在 Spec 中配置 Rules

```yaml
# strategy_spec.yaml
rules:
  pre_trade:
    - type: MaxDrawdownRisk
      params: { max_drawdown: 0.15 }
    - type: RebalanceFrequencyRule
      params: { interval_days: 5 }
  post_trade:
    - type: StopLossRule
      params: { threshold: 0.05 }
    - type: TakeProfitRule
      params: { threshold: 0.20 }
    - type: TrailingStopRule
      params: { trail_pct: 0.05 }
    - type: ExitRule
      params: { fast: sma_fast, slow: sma_slow }
```

## 回测

```bash
oxq backtest run strategy_spec.yaml --out runs/auto
```

规则会在回测引擎中逐 bar 执行。

## 组合建议

| 策略类型 | Pre-trade | Post-trade |
|---------|-----------|------------|
| 趋势跟踪 | `MaxDrawdownRisk`, `RebalanceFrequencyRule` | `TrailingStopRule` |
| 均值回归 | `DailyLossLimitRisk` | `StopLossRule`, `TakeProfitRule` |
| 动量轮动 | `MaxHoldingsRule`, `RebalanceFrequencyRule` | `StopLossRule` |
| SMA 交叉 | `MaxDrawdownRisk` | `ExitRule`, `StopLossRule` |

## 红线

- **止损必须有**：任何实盘策略至少配置一个止损规则
- **不修改已验证的 spec**：规则配置不应导致 spec 重新 validate
- **回测后必须审计**：`oxq audit research runs/<run_id>/`
