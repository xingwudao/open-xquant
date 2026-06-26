---
name: review-research-report
description: >-
  Use when reviewing a completed open-xquant research report for decision
  consistency, artifact fidelity, audit/robustness interpretation, chart
  narrative quality, and semantic issues that deterministic report QA should not
  judge.
---

# Research Report Reviewer

Use this skill after `write-research-report` has written `research_report.md`,
HTML has been rendered from the same Markdown, and deterministic artifact QA has
run with `oxq report qa runs/<run_id>/`.

`oxq report qa` checks files, dates, registered image references, manifest
integrity, hashes, and dimensions. This skill reviews the research meaning:
whether the report's decision, explanation, risk language, and chart narrative
are faithful to the artifacts.

## Inputs

Read:

- `research_report.md` and `research_report.html`
- `strategy_spec.yaml`, especially `decision_policy`
- `spec_audit.json`, including field provenance, recipe matches,
  component audits, missing requirements, agent-added fields, contradictions,
  and blocking findings
- `metrics.json` and `execution_assumptions.json`
- `research_bias_audit.json` and `reproducibility_audit.json`
- `robustness.json`, including perturbation and regime analysis sections
- `report_assets/manifest.json`, captions, source scripts, and figure files
- `equity_curve.csv`, `benchmark_curve.csv`, `trades.csv`, `positions.csv`,
  and `target_weights.csv` when needed to verify a claim
- the deterministic `oxq report qa` output
- optional/advisory numeric QA output when available, especially any
  `numeric_claim_unverified` warnings

Do not rewrite metrics, audits, robustness output, backtest artifacts, Markdown,
or HTML. This skill reviews the report and writes `report_review.json`; it does
not edit the report. If a report edit is needed, return a blocking finding for
the upstream writer phase.

## Review Checklist

1. Decision consistency.
   - Extract the executive decision: `REJECT`, `NO EVIDENCE`, `WATCHLIST`, or
     `PAPER TRADING CANDIDATE`.
   - Evaluate `decision_policy` against `metrics.json`; for example, if
     `reject_if.max_drawdown_lt` is breached, the report must not conclude
     `WATCHLIST` or `PAPER TRADING CANDIDATE`.
   - Treat fatal audits, failed reproducibility, missing OOS evidence, or
     fragile/error robustness as blockers unless the report explicitly rejects
     or withholds promotion.

2. Audit and robustness fidelity.
   - Confirm reproducibility status, research-bias fatal/warning counts, and
     robustness status are stated accurately.
   - Confirm material warnings are explained, not summarized away as
     "风险可控" or "审计通过".
   - Check that perturbation and regime analysis findings are represented when
     they affect the decision.
   - Confirm `spec_audit.json` conclusions are represented, including
     unconfirmed defaults, component provenance warnings, selected recipes, and
     any unresolved blocking questions.

3. Numeric warning triage.
   - Do not treat every `numeric_claim_unverified` as a report error.
   - Classify each important warning as one of:
     `likely report error`, `facts registry gap`, `precision/format issue`, or
     `low-impact appendix claim`.
   - Escalate executive-decision, core metric table, drawdown, Sharpe, return,
     benchmark, and OOS trade-count mismatches above appendix parameter values.

4. Evidence chain and structure.
   - The report must put conclusion first, then evidence, then risks and next
     actions.
   - It must include strategy context, configured end date, effective last
     trading day, costs, benchmark-relative behavior, drawdown, trade count,
     audit status, and robustness interpretation.
   - Claims like "OOS 优于 IS 表明没有过拟合", "方向正确", "风险可控", or
     "适合 paper trading" require explicit artifact-backed reasoning.

5. Chart review.
   - Inspect registered figures when they support key conclusions.
   - Check that charts are not blank, nearly blank, unreadable, mislabeled, or
     purely decorative.
   - Prefer English chart labels unless the final rendered figure already
     proves local-language labels are readable.
   - Captions must identify source artifacts and avoid unsupported conclusions.

6. Markdown/HTML semantic consistency.
   - Confirm HTML appears to represent the same report as Markdown.
   - Check that section order, figure order, captions, and key paragraphs are
     not missing or reordered in a way that changes meaning.

## Output

Write `report_review.json` and return a concise summary. The JSON shape is:

```json
{
  "status": "pass | blocked | fail",
  "verdict": "consistent | inconsistent | needs_revision",
  "findings": [],
  "blocking_findings": [],
  "required_report_edits": [],
  "reviewed_artifacts": [],
  "errors": []
}
```

Include:

- Final decision verdict: `consistent`, `inconsistent`, or `needs revision`.
- Blocking issues first, each tied to artifact evidence.
- Warning triage summary, separating report defects from facts registry gaps.
- Required report edits, if any.
- Follow-up framework improvements, such as facts registry fields that should
  later move into deterministic QA.

## Red Lines

- Do not approve a report whose decision conflicts with `decision_policy`.
- Do not approve positive risk language that omits fatal or material warnings.
- Do not approve a report that omits unresolved `spec_audit.json` blockers or
  presents non-canonical component choices as confirmed.
- Do not let many low-value numeric warnings hide a decision-critical mismatch.
- Do not use chart aesthetics as evidence of strategy quality.
- Do not claim chart text is readable solely because plotting code configured
  styling; inspect the final rendered figure when readability matters.
- Do not edit `research_report.md` or `research_report.html`; write
  `report_review.json` instead.
