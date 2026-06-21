---
name: quant-research
description: Complete quantitative research workflow — from idea to audited report.
---

# Quant Research Skill

Execute a complete quantitative research workflow: idea → spec → backtest → audit → report.

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
Read `run_dir` and artifact paths from `backtest.json`. Use `target_weights.csv`
for target allocation comparisons and `trades.csv` for execution comparisons.

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

## Critical Rules

1. **Builder ≠ Auditor** — separate agents must handle backtest and audit.
2. **Never skip audit** — unaudited backtests have no decision value.
3. **Never beautify failures** — report the truth, not what sounds good.
4. **Every run gets an experiment entry** — prevent selective memory.
