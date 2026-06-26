---
name: evaluate-factor
description: >-
  Route open-xquant factor evaluation tasks to cross-sectional or time-series
  workflows; use when users ask whether a factor predicts returns.
---

# Factor Evaluator

You decide which factor evaluation workflow to use.

## Ask First

Confirm:

- factor definition
- symbols
- date range
- forward return horizons
- whether the question is stock selection or timing
- data source and missing-data treatment

## Route

Use `agent/skills/evaluate-cross-sectional/SKILL.md` when:

- the user ranks many assets on each date
- the goal is IC, Rank IC, ICIR, decay, or turnover
- there are enough symbols for cross-sectional statistics

Use `agent/skills/evaluate-time-series/SKILL.md` when:

- the user evaluates one asset or a small rotation set
- the question is directional timing
- hit rate, P/L ratio, decay curve, or tearsheet is more relevant

Rule of thumb:

- fewer than 10 symbols: avoid cross-sectional IC as primary evidence
- 10 to 30 symbols: use IC cautiously
- more than 30 symbols: cross-sectional IC is more defensible

## Data Requirements

Build factor values and forward returns with aligned indexes. Do not let
same-day execution leak into forward returns. For formal reports, state the
horizon, date alignment, and excluded rows.

## Red Lines

- Do not evaluate a factor without forward-return alignment.
- Do not run only one horizon when the user is making a research claim.
- Do not hide low sample size or high turnover.
