---
name: research-report-writer
description: >-
  Use when writing the final human-readable open-xquant experiment report from
  generated evidence, audit results, metrics, robustness output, and registered
  chart assets.
---

# Research Report Writer

Use this skill after the run artifacts, audits, robustness checks, and optional
chart assets exist. The program may generate metrics, audit findings,
robustness outputs, chart files, and asset manifests, but the Agent must write
the final report for human decision-making.

## Inputs

Read:

- `strategy_spec.yaml`
- `metrics.json`
- `execution_assumptions.json`
- reproducibility audit output
- research-bias audit output
- `robustness.json` when present
- `report_assets/manifest.json` and registered figures
- `equity_curve.csv`, `benchmark_curve.csv`, `trades.csv`, `positions.csv`,
  and `target_weights.csv` only when needed to verify a claim

Do not call a report-writing CLI or tool. The final report narrative must be
written by the Agent using this skill, then saved to `research_report.md`.

## Writing Goal

Write for two audiences:

- The human researcher who may allocate their own capital based on the report.
- The potential investor who needs a professional, visually credible,
  evidence-rich explanation before funding the strategy.

The report must make the decision easy to audit:

1. State the conclusion first.
2. Explain the logic that supports the conclusion.
3. Tie every claim to specific evidence.
4. Interpret charts in plain language.
5. Separate strengths from blocking risks.
6. Explain what must happen before capital allocation.

Default language is Chinese unless the user asks otherwise.

## Required Structure

Write `research_report.md` with:

1. Executive decision.
   - Use `REJECT`, `NO EVIDENCE`, `WATCHLIST`, or
     `PAPER TRADING CANDIDATE`.
   - Include one concise decision paragraph before details.

2. Evidence chain.
   - Cover reproducibility, research-bias audit, robustness, IS/OOS behavior,
     benchmark-relative performance, costs, drawdown, and trade count.
   - Explain why each item supports or weakens the decision.

3. Strategy and experiment context.
   - Summarize strategy, universe, signal, execution assumptions, cost model,
     benchmark, and metrics profile.

4. Chart narrative.
   - Embed registered figures using their manifest paths.
   - Add a caption and an interpretation paragraph for each important figure.
   - Do not include unregistered images.

5. Investor-readable presentation.
   - Use clear section headings, short paragraphs, and compact tables.
   - Redundant explanations are acceptable when they improve comprehension.
   - Avoid notebook-style code dumps and raw artifact listings.

6. Risks and next actions.
   - Include what would invalidate the thesis.
   - Include concrete next experiments or monitoring gates.

## HTML Output

After writing final Markdown, render HTML from that same Markdown:

```bash
uv run python - <<'PY'
from pathlib import Path
from oxq.report.html import render_markdown_html_report

run_dir = Path("runs/<run_id>")
markdown = (run_dir / "research_report.md").read_text(encoding="utf-8")
html = render_markdown_html_report(markdown, lang="zh")
(run_dir / "research_report.html").write_text(html, encoding="utf-8")
PY
```

The HTML renderer may format the final Markdown, but it must not regenerate the
report narrative from templates.

## Red Lines

- Do not invent evidence.
- Do not modify metrics, audit files, robustness output, or backtest artifacts.
- Do not hide adverse evidence behind generic positive language.
- Do not promote a run with fatal audit findings, failed reproducibility,
  fragile robustness, or no usable OOS evidence.
- Do not call the strategy investable; describe research evidence and limits.
- Do not treat charts as proof; charts illustrate evidence that must already be
  supported by artifacts.
