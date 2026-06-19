# /quant-report

Generate a research report with executive decision.

## Usage

```
/quant-report runs/<run_id>/
```

## Steps

1. `oxq report write runs/<run_id>/` — generate research_report.md
2. `oxq experiment add runs/<run_id>/` — register in experiment log
3. Present the executive decision and key findings

## Decision Scale

- **REJECT**: fatal audit findings, invalid artifacts, or missing OOS evidence
- **WATCHLIST**: warnings, weak robustness, or incomplete robustness artifacts
- **PAPER TRADING CANDIDATE**: audits pass, robustness acceptable, and the
  metrics/execution assumptions are explicit

Include metrics profile, execution assumptions, and robustness highlights when
present.
