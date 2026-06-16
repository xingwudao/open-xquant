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

| Decision | Criteria |
|----------|----------|
| **REJECT** | Fatal audit findings |
| **WATCHLIST** | Warnings but no fatal issues |
| **PAPER TRADING CANDIDATE** | All audits pass, robustness acceptable |
