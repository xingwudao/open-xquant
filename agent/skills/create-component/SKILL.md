---
name: create-component
description: >-
  Route open-xquant component creation requests after checking the registry;
  use when users ask for a new Indicator, Signal, Rule, or PortfolioOptimizer.
---

# Component Creator

Use this skill when the user explicitly asks to modify OpenXQuant source code
or add a built-in component to the framework.

For workspace-local custom components in a research task, use
`author-component` instead. That worker writes `custom_components/`,
`component_manifest.json`, and a refreshed `component_catalog.json` without
mutating the installed SDK bundle.

When this built-in/framework development path applies, decide whether a new
component is actually needed, then route to the correct creation skill.

Do not write component code in this navigator skill.

## Step 1: Classify The Request

Choose one component type:

- Indicator: numeric time-series computation from market or factor data
- Signal: boolean or categorical trading intent
- PortfolioOptimizer: target portfolio weights from signals or indicators
- Rule: bar-by-bar trading constraint, hold, or exit action

If the request could fit more than one type, ask the user to clarify. For
example, "momentum" can be an Indicator, while "buy when momentum is positive"
is a Signal.

For ROC timing strategies, prefer the existing split:

- `ROC` is the numeric Indicator.
- `ROCTiming` is the Signal that emits `BUY`, `SELL`, or `HOLD`.
- `SignalToPosition` is the PortfolioOptimizer that turns those labels into
  target weights while `HOLD` maintains the prior target position.

For other categorical timing signals, keep the same boundary: the Signal emits
labels, the spec declares `output_domain`, and `SignalToPosition` maps labels
to target weights. Do not route that behavior to an Indicator only to make a
spec compile.

## Step 2: Check Existing Registry

```bash
uv run python - <<'PY'
import oxq

print("Indicators:", sorted(oxq.list_indicators()))
print("Signals:", sorted(oxq.list_signals()))
print("Portfolios:", sorted(oxq.list_portfolio_optimizers()))
print("Rules:", sorted(oxq.list_rules()))
PY
```

Search for exact and near matches. If a component already exists, report the
existing name and stop unless the user explicitly needs different behavior.

## Step 3: Route

Load exactly one sub-skill:

- Indicator: `agent/skills/create-indicator/SKILL.md`
- Signal: `agent/skills/create-signal/SKILL.md`
- PortfolioOptimizer: `agent/skills/create-portfolio-optimizer/SKILL.md`
- Rule: `agent/skills/create-rule/SKILL.md`

Tell the sub-skill the component type, requested behavior, desired name if
given, and any user constraints.

## Red Lines

- Do not skip the registry check.
- Do not create a component because the user used a different synonym.
- Do not guess formulas, thresholds, or risk logic.
- Do not route to multiple component creation skills at once.
