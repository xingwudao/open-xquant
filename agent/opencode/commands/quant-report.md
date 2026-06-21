# /quant-report

Generate a research report with executive decision.

## Usage

```
/quant-report runs/<run_id>/
```

## Steps

1. Confirm audit artifacts exist or run `/quant-audit runs/<run_id>/`.
2. Confirm `robustness.json` exists and is covered by `artifact_hashes.json`.
3. If robustness is missing or untrusted, run `oxq robustness run runs/<run_id>/`.
4. If the user wants charts, discuss chart requirements first, write plotting
   Python, save figures under `report_assets/figures`, save scripts under
   `report_assets/scripts`, and register them with `oxq report asset add`.
5. `oxq report write runs/<run_id>/ --lang zh --format all` — generate
   `research_report.md` and `research_report.html`
6. `oxq experiment add runs/<run_id>/` — register in experiment log
7. Present the executive decision, key findings, and report paths

## Decision Scale

- **REJECT**: fatal audit findings, invalid artifacts, or missing OOS evidence
- **WATCHLIST**: warnings, weak robustness, or incomplete robustness artifacts
- **PAPER TRADING CANDIDATE**: audits pass, robustness acceptable, and the
  metrics/execution assumptions are explicit

Include metrics profile, execution assumptions, and robustness highlights when
present.

If `research_report.md` says **PAPER TRADING CANDIDATE** but robustness is still
missing, untrusted, `warn`, `fragile`, or `error`, present the final command
decision as **WATCHLIST** and state the robustness reason. Charts are report
assets only; they do not override audit or robustness findings.
