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
oxq backtest run strategy_spec.yaml --out runs/auto
```
Note the run directory path printed by the command.

### Step 3: Audit
```bash
oxq audit reproducibility runs/<run_id>/
oxq audit research runs/<run_id>/
oxq robustness run runs/<run_id>/
```

### Step 4: Report
```bash
oxq report write runs/<run_id>/
oxq experiment add runs/<run_id>/
```

## Quality Gates

| Gate | Check | Action on Failure |
|------|-------|-------------------|
| Spec Validation | All P0 checks pass | Fix spec and re-validate |
| Backtest | Run completes without error | Debug and re-run |
| Reproducibility Audit | All hashes match | Investigate inconsistency |
| Research Bias Audit | No fatal findings | REJECT or fix spec |
| Robustness | Cost x2 doesn't destroy Sharpe | Flag as fragile |
| Report | Executive decision issued | Document decision |

## Critical Rules

1. **Builder ≠ Auditor** — separate agents must handle backtest and audit.
2. **Never skip audit** — unaudited backtests have no decision value.
3. **Never beautify failures** — report the truth, not what sounds good.
4. **Every run gets an experiment entry** — prevent selective memory.
