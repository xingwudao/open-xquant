---
name: quant-research
description: Complete quantitative research workflow from idea to audited report.
---

# Quant Research Skill

Execute a complete quantitative research workflow: idea -> spec -> backtest ->
audit -> report.

## Workflow

### Step 0: Experiment Lifecycle Check

Before starting a new run, inspect `runs/` and classify existing experiments by
the presence of:

- `metrics.json`
- `research_report.md`
- `research_bias_audit.json`
- `robustness.json`

Skip robustness sub-runs such as `<run_id>_cost_x2` and parameter-perturbation
siblings when deciding whether a user experiment is unfinished. Treat those
directories as child artifacts of the parent run, not resumable experiments.

If any run has `metrics.json` but no `research_report.md`, ask whether to
continue that unfinished experiment or abandon it and start a new one. Do not
delete abandoned runs; record abandonment in `experiments.jsonl` when the
registry exists.

### Step 1: Create Spec

```bash
oxq spec init "<strategy idea>" --out strategy_spec.yaml
```

Edit the spec file with proper parameters, then validate:

```bash
oxq spec validate strategy_spec.yaml
```

Fix any errors. Spec MUST pass validation before proceeding.

After validation passes, use `spec-auditor` to trace material field sources.
Any unconfirmed field blocks the backtest until the user confirms or changes
the grouped assumptions and the spec passes validation again.

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

Before final report writing, ask the user whether chart assets are needed. If
the answer is yes, use `report-chart-builder` to discuss requirements, generate
figures, and register them before handing off to `research-report-writer`.
If the user wants charts, do not start the final narrative until that handoff is
complete. If the user declines charts, continue with the report and disclose
that no chart assets were requested.

This is a mandatory handoff. Do not write the final report directly after
recognizing that `research-report-writer` applies, even if all artifacts are
already in context.

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
If the reviewer requires Markdown report edits, render `research_report.html`
again from the updated Markdown before rerunning `oxq report qa`.

When the user accepts the completed experiment, ask whether to mark it as final.
If yes, read `.open-xquant/workspace.yaml` and resolve `paths.final_dir`; fall
back to `runs/final` when the config value is absent. Write the lightweight
final pointer there:

- `<final_dir>/strategy_spec.yaml`
- `<final_dir>/selected.json`
- `<final_dir>/README.md`

Mark the accepted run as final in `experiments.jsonl` when present. The final
pointer is a copy of the selected spec plus metadata, not a copy of the full
run directory.

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
5. Applicable open-xquant skills are routing gates, not suggestions; use the
   matching skill before running tools or writing outputs.
