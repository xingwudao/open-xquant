---
title: AI 量化回测 | open-xquant
description: 面向中文搜索意图说明 open-xquant 授权回测流程、输入门槛、执行产物和常见阻断原因。
---

# AI 量化回测

open-xquant 的 AI 量化回测不是让 Agent 直接运行任意策略代码，而是在 spec、spec audit、runtime audit 和用户授权都通过后，才由确定性 runner 执行正式回测。

## 输入

正式回测读取 `strategy_spec.yaml`、`spec_audit.json`、`runtime_audit.json`、`backtest_authorization.json`，在需要 workspace-local 组件时还读取 component catalog 和 component manifest。数据目录或数据 manifest 必须来自授权文件。

## 授权门槛

[授权量化回测执行](/skills/run-authorized-backtest) 要求 spec audit 为 pass、用户确认状态为 confirmed，并且授权文件中的 spec、audit、runtime audit 哈希与实际文件一致。运行前还要验证完整策略源码展示事件，避免 runtime worker 的路径式交接替代用户可见确认。

## 确定性执行

回测阶段只执行被授权的输入，不修改 spec、不修复审计、不下载未批准数据，也不写报告。正式输出写入版本清单指定的 `<phase_paths.09_backtests>/<run_id>/`，而不是根目录 `runs/`。

## 回测产物

典型产物包括 `metrics.json`、`equity_curve.csv`、`trades.csv`、`positions.csv`、`orders.csv`、`target_weights.csv`、`execution_assumptions.json`、`data_manifest.json`、`artifact_hashes.json` 和 `runner_result.json`。后续可进入 [量化回测审计](/workflows/research-audit) 与 [量化策略稳健性检验](/workflows/robustness-testing)。

## 常见失败

常见阻断包括缺少 `backtest_authorization.json`、用户确认事件不完整、hash 过期、component bundle 绑定不一致、phase path 逃逸版本目录、运行语义审计未通过，或同 bar 收盘信号与收盘成交这类执行语义错误。
