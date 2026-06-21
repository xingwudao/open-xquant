---
name: quant-research
description: Complete quantitative research workflow from idea to audited report.
---

# Quant Research Skill

Execute a complete quantitative research workflow: idea -> spec -> backtest ->
audit -> report.

## Workflow

### Step 1: Create Spec

```bash
oxq spec init "<strategy idea>" --out strategy_spec.yaml
```

Edit the spec file with proper parameters, then validate:

```bash
oxq spec validate strategy_spec.yaml
```

Fix any errors. Spec MUST pass validation before proceeding.

### Step 2: Backtest

```bash
oxq backtest run strategy_spec.yaml --out runs/auto --json > backtest.json
```

Read `run_dir` and artifact paths from `backtest.json`. Use
`target_weights.csv` for target allocation comparisons and `trades.csv` for
execution comparisons.

When a strategy produces `BUY`, `SELL`, or `HOLD` labels, model that as a
categorical Signal and map it with `SignalToPosition`. Do not wire categorical
labels directly to `EqualWeight`. For custom categorical signals in spec, place
`output_domain: [BUY, SELL, HOLD]` at the signal rule top level, not under
`params`.

### Step 3: Audit

```bash
oxq audit reproducibility runs/<run_id>/
oxq audit research runs/<run_id>/
oxq robustness run runs/<run_id>/
```

### Step 4: Report

```bash
oxq experiment add runs/<run_id>/
```

Use `research-report-writer` to write `research_report.md` from run artifacts,
audits, robustness output, metrics, and registered chart assets. Render
`research_report.html` from that final Markdown.

Run deterministic Final report QA before presenting results:

```bash
oxq report qa runs/<run_id>/
```

The deterministic QA checklist covers Markdown/HTML image counts, manifest
order and hash, HTML image paths, configured end date, and
effective last trading day.

Then use `research-report-reviewer` to check decision_policy consistency,
audit/robustness fidelity, numeric warning triage, chart narrative quality,
CJK/font risk, and whether the report structure supports the stated decision.

## Quality Gates

- Spec validation: all fatal checks pass; fix spec and re-validate on failure.
- Assumptions: metrics profile, execution assumptions, calendar, lot size,
  costs, and cash return are explicit; do not compare runs without matching
  assumptions.
- Backtest: run completes without error; debug and re-run on failure.
- Reproducibility audit: hashes match; investigate inconsistency.
- Research bias audit: no fatal findings; reject or fix spec on fatal findings.
- Robustness: review cost stress, IS/OOS diff, parameter perturbation, and
  regime analysis; flag fragile, warning, or error statuses.
- Report: executive decision issued; document decision and limitations.
- Final report QA: `oxq report qa` has no fatal deterministic findings, and
  `research-report-reviewer` finds no blocking semantic issues.

## Critical Rules

1. Builder and auditor must be separate agents.
2. Never skip audit; unaudited backtests have no decision value.
3. Never beautify failures; report the truth, not what sounds good.
4. Every run gets an experiment entry to prevent selective memory.
