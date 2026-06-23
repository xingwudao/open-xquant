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

Use the start of the current experiment as the source boundary. A just-finished
`oxq spec validate` is only a checkpoint proving the current YAML is
well-formed; it is not the boundary for tracing user confirmations made while
building that spec.

When resuming after a prior run, use the latest relevant prior run containing
`spec_hash.txt` and `environment.json` to identify the previous experiment's
timestamp, then trace user confirmations made after that point. If no prior
validated run exists, trace from the start of the current conversation.

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
