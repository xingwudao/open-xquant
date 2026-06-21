# /quant-report

Generate a human-written research report with executive decision.

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
   If one script regenerates multiple figures, register them with
   `oxq report asset add-batch`.
5. Use `research-report-writer` to read audits, robustness, metrics, execution
   assumptions, strategy spec, and registered chart assets, then write the final
   `research_report.md` for human researchers and potential investors.
6. Render `research_report.html` from final Markdown with
   `render_markdown_html_report`.
7. Run Final report QA:
   `oxq report qa runs/<run_id>/`.
8. `oxq experiment add runs/<run_id>/` — register in experiment log.
9. Present the executive decision, key findings, and report paths.

## Final Report QA

Before presenting the report, `oxq report qa` must pass without fatal findings.
Review warnings explicitly:

- Markdown/HTML image counts match.
- HTML images only use `report_assets/...` paths.
- Manifest order and hash checks pass.
- CJK font risk is reviewed for Chinese charts.
- Key numbers in prose trace back to artifacts or facts.
- The report discloses configured end date and effective last trading day.

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
