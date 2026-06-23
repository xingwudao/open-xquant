---
name: spec-auditor
description: Audit strategy_spec.yaml field provenance before backtests.
---

# Spec Auditor

Use this skill after `oxq spec validate strategy_spec.yaml` passes and before
any `oxq backtest run`. Its job is to prevent unapproved strategy assumptions
from entering a formal experiment.

## Inputs

- `strategy_spec.yaml`
- the current conversation context
- `runs/` metadata when present

## Source Boundary

Use the last successful `oxq spec validate` as the experiment boundary. When a
prior run contains `spec_hash.txt` and `environment.json`, use that run's
recorded timestamp as the boundary. If no prior validated run exists, trace
from the start of the current conversation.

## Field Classification

Classify material fields as:

- `confirmed`: the user explicitly gave the value or an equivalent meaning.
- `default`: the value matches a documented strategy-builder template default.
- `unconfirmed`: the value is not a template default and no user source exists.

Material fields include:

- train/test periods
- symbols and benchmark
- execution timing and fill price fields
- initial cash and cash return
- fee rate and slippage rate
- risk-free rate and metrics profile
- exit, risk, and rebalance constraints

## Gate

Any `unconfirmed` field blocks backtest. Always group related fields instead of
asking one question per YAML key:

- execution assumptions
- cost assumptions
- train/test split
- cash and risk-free assumptions
- benchmark and success metric

Ask the user to either confirm the group or provide replacement values. After
any change, re-run `oxq spec validate strategy_spec.yaml` and repeat this
auditor gate.

## Output

Report a compact summary:

- confirmed fields
- default fields that will be used
- unconfirmed fields that block progress

Do not run or approve a backtest while blocking fields remain.
