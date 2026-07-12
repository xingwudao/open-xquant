---
name: write-research-report
description: >-
  Use when writing the final human-readable open-xquant experiment report from
  generated evidence, audit results, metrics, robustness output, and registered
  chart assets.
---

## Phase Path Containment Preflight

Before any phase artifact read, write, directory creation, command, or handoff,
run this preflight completely and block on any failure:

1. Read `.open-xquant/workspace.yaml`. Resolve `version_root` from
   `paths.versions_dir`, using `versions` only when that key is absent.
   Require `version_root` to be a safe workspace-relative path: reject an
   absolute path or any `..` path segment, resolve it canonically with
   symlinks, and require the result to stay inside the workspace.
2. Read `current.json`. For normal phase work, set `expected_version_id` to
   `current.json.active_version`; it must exist. Only a contract that
   explicitly owns cross-version inspection may instead set
   `expected_version_id` from the referenced version id for that historical
   read. This exception never permits active-version work to consume another
   version.
3. Set the intended version directory to
   `<version_root>/<expected_version_id>/` and resolve it canonically. Require
   the intended version directory to remain inside the canonical version root
   and workspace; otherwise treat it as a symlink escape. Read
   `version_manifest.json` only from that exact directory. The manifest
   `version_id` must equal `expected_version_id`; for normal phase work it
   therefore must also equal `current.json.active_version`.
4. Before using each required `phase_paths` value, require a non-empty
   workspace-relative string. Reject an absolute path and any `..` path
   segment. Resolve `<workspace>/<phase_path>` canonically, including existing
   symlink ancestors even when the leaf will be created, and require the target
   to be the intended version directory or a descendant of it. A symlink escape
   outside that directory is invalid.
5. On any identity or path failure, stop before phase artifact reads, writes,
   directory creation, commands, or handoffs. Do not normalize an unsafe path
   into acceptance and do not fall back to a default phase path.

Block examples when `expected_version_id` is `v001` include
`strategy_store/v001/../v002/04_spec_build`,
`strategy_store/v002/04_spec_build`, `/tmp/04_spec_build`, and
`strategy_store/v001/escape/04_spec_build` when `escape` is a symlink
whose target is outside the intended version directory. An allowed custom
nested phase path is
`strategy_store/v001/custom/phases/04_spec_build` when its canonical
target remains under the intended version directory.

For a new-version bootstrap, only `manage-strategy-version` may proceed before
the new manifest exists or the new id becomes active. It must apply the same
workspace-relative, traversal, canonical-containment, and symlink checks to
every constructed phase path before directory creation, then write a matching
manifest before publishing `current.json` last.

## Version Path Resolution

Before using any `<phase_paths.*>` or `<version_root>` placeholder, read
`.open-xquant/workspace.yaml`. Resolve `version_root` from
`paths.versions_dir`; use `versions` only when that key is absent. Require a
safe relative path whose resolved target stays inside the workspace. Then read
`<version_root>/<version_id>/version_manifest.json` and use its exact
`phase_paths` entry for each phase. For example, a configured root of
`research_versions` must resolve the spec-build phase to
`research_versions/v003/04_spec_build`; never redirect it to a default-root phase path.

# Research Report Writer

Round 26 chart retry: keep a blocked report as an unsealed revision attempt,
allocate a fresh `report_revision_id`, and seal only after chart completion.
The default professional pack uses its canonical requested set and cannot be
narrowed by the caller; omitted charts receive closed skip reasons. Historical
refresh routes `write -> review -> lineage -> prepare new selection -> comparison
-> resume` with fresh selection and candidate hashes.

Use this skill after the run artifacts, audits, robustness checks, and report
chart assets exist. Final research reports require registered chart assets by
default. The program may generate metrics, audit findings, robustness outputs,
chart files, and asset manifests, but the Agent must write the final report for
human decision-making.

## Mandatory Routing

This skill is mandatory whenever the Agent writes or edits
`research_report.md`, `research_report.html`, or any final human-readable
experiment report. Having already loaded metrics, audits, robustness output,
facts, or chart assets is not a reason to bypass this skill.

If the Agent recognizes that this skill applies and starts to think "I can just
write the report directly", stop. Continue from this skill and follow its input,
structure, fact, and red-line rules before writing the report.

## Version-Governed Report Package

Before writing any report artifact, read `current.json` and use
`active_version` as `version_id`. The source run package is:

```text
<phase_paths.09_backtests>/<run_id>/
```

For current production, write final report artifacts only under the supplied
immutable candidate revision:

```text
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.md
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.html
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/writer_result.json
<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/report_assets/
```

Do not write root-level `research_report.md`, `research_report.html`,
`writer_result.json`, or `report_assets/`.

The old working-head path
`<phase_paths.10_reports>/<run_id>/research_report.md` is retained only for
historical layout recognition; it is not a current selectable write target.

## Language Parameter Gate

Before drafting or deciding whether to block for chart generation, set a
`report_language` parameter from the user request, coordinator input, or report
task metadata. Resolve this gate before the Chart Decision Gate so every
blocked `writer_result.json` can hand the intended language to the chart phase.

- If the user does not explicitly request another language, set
  `report_language` to `中文`.
- Write the report in `report_language`, including headings, the executive
  decision explanation, evidence interpretation, risks, captions, and next
  actions.
- Do not localize the canonical executive decision token. The token itself must
  remain exactly one of `REJECT`, `NO EVIDENCE`, `WATCHLIST`, or
  `PAPER TRADING CANDIDATE`; localize only the surrounding explanation.
- Do not switch the whole report to English because artifact keys, chart labels,
  or prior reports are in English.
- Pass the same language to chart generation and HTML rendering. Derive the
  renderer `lang` from `report_language` instead of hardcoding one locale.
- Record the resolved value in `writer_result.json` as `"language": "中文"` when
  the default is used.

## Chart Decision Gate

After resolving `report_language`, inspect `report_assets/manifest.json` if it
exists and inspect the invoking task inputs for a chart decision. Final report
charts are required by default.

- If the invoking task omits a chart decision, set
  `chart_decision: default_professional_chart_pack`; do not ask the user
  whether to build charts.
- If registered chart assets for the required report pack already exist and the
  task does not request a refresh, continue with the registered assets.
- If registered chart assets are missing, stale, unreadable, unregistered, or
  insufficient for the required pack, stop report writing and write
  `writer_result.json` with `status: blocked`, `language`: `report_language`,
  `chart_decision: default_professional_chart_pack`,
  `blocking_reason: missing_required_report_charts`,
  `next_required_phase: chart_building`, and
  `next_skill: build-report-charts`.
- If the user requested a richer or custom chart pack, use that chart decision
  as the required pack and apply the same missing/stale block.
- Do not ask the user whether to build charts, and do not continue to a
  successful final report without registered chart assets.

## Inputs

Read:

- `strategy_spec.yaml`
- `spec_audit.json`
- `metrics.json`
- `execution_assumptions.json`
- `compiled_plan.json`
- `data_manifest.json`
- reproducibility audit output
- research-bias audit output
- `robustness.json` when present
- `report_assets/manifest.json` and registered figures
- `equity_curve.csv`, `benchmark_curve.csv`, `trades.csv`, `positions.csv`,
  and `target_weights.csv` only when needed to verify a claim
- facts from `build_report_facts(RunArtifacts.load(run_dir))` when monthly
  returns, trade contribution, positive/negative month counts, OOS trade counts,
  configured end date, or effective last trading day are discussed. Use the
  installed package API exactly:

  ```python
  from oxq.report.artifacts import RunArtifacts
  from oxq.report.facts import build_report_facts

  facts = build_report_facts(RunArtifacts.load(run_dir))
  ```

  Do not import `RunArtifacts` from `oxq.api`. Do not import `RunArtifacts`
  from `oxq.run.artifacts`; those are not the report artifact API.

Do not call a report-writing CLI or tool. The final report narrative must be
written by the Agent using this skill, then saved to `research_report.md`.

Evidence is generated by the framework; the narrative is authored by the Agent.
Do not write "this report was generated by the framework" or equivalent wording.

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

Default language is controlled by `report_language`; the fallback value is
`中文`.

## Institutional Report Standard

Write a layered institutional research report: professional enough for a
potential investor to trust at a glance, and rigorous enough for the researcher
to audit before risking capital.

- 30-second investor view: start with an Executive Snapshot containing the
  decision, one-line thesis, metric scorecard, trust and audit status, top
  risks near the decision, and the next gate before any capital allocation.
- 3-minute research view: follow with the evidence chain, chart narrative,
  robustness interpretation, and a clear explanation of what supports or
  weakens the thesis.
- professional appendix: keep methodology, assumptions, artifact sources,
  audit details, limitations, and reproducibility notes available without
  interrupting the main reading flow.
- Use message-first section openings. Each major section should begin with the
  point the reader should take away, then present the supporting evidence.
- Use compact tables and short paragraphs. Redundant explanation is acceptable
  only when it helps a non-specialist understand the decision without weakening
  professional precision.
- Put risks near the decision, not only at the end. A report that looks
  attractive but hides blocking risks is unacceptable.

## Required Structure

Write `research_report.md` with:

1. Executive decision.
   - Use `REJECT`, `NO EVIDENCE`, `WATCHLIST`, or
     `PAPER TRADING CANDIDATE`.
   - Include one concise decision paragraph before details.
   - Include an Executive Snapshot table with metric scorecard, trust and
     audit status, top risks, and next gate.

2. Evidence chain.
   - Cover reproducibility, research-bias audit, robustness, IS/OOS behavior,
     benchmark-relative performance, costs, drawdown, and trade count.
   - Explain why each item supports or weakens the decision.

3. Strategy and experiment context.
   - Summarize strategy, universe, signal, execution assumptions, cost model,
     benchmark, and metrics profile.
   - Disclose configured end date and effective last trading day.
     Use deterministic QA-recognized labels exactly:
     `配置结束日：YYYY-MM-DD` and `有效数据最后交易日：YYYY-MM-DD` for
     Chinese reports, or `Configured end date: YYYY-MM-DD` and
     `Effective last trading day: YYYY-MM-DD` for English reports.
     Do not use label variants such as `配置的回测结束日期`,
     `有效最后交易日`, or English fallback labels inside a Chinese report.
   - Cite `spec_audit.json` for field provenance, unconfirmed defaults,
     selected canonical recipes, catalog hash status, and component provenance
     warnings.
   - Cite `compiled_plan.json` or `execution_assumptions.json` for actual
     runtime execution semantics. Do not describe rebalance frequency,
     execution timing, fees, slippage, validation split, or runtime rules from
     the raw YAML alone.
   - Disclose data warmup policy from `data_manifest.json` and
     `compiled_plan.json`, including `data.min_start_date`, effective data
     directory, and whether early exposure may be affected by missing warmup.
   - State that different execution, cost, validation, or data warmup settings
     can make performance comparisons non-comparable unless
     `oxq backtest compare-runs` passes.

4. Chart narrative.
   - Resolve each package-relative manifest path `figures/<name>.png` beneath
     `report_assets/`, then embed the report image URL
     `report_assets/figures/<name>.png`. Do not embed the package-relative
     manifest path directly and do not add a second `report_assets/` prefix.
   - Add a caption and an interpretation paragraph for each important figure.
   - Do not include unregistered images.

5. Investor-readable presentation.
   - Use clear section headings, short paragraphs, and compact tables.
   - Redundant explanations are acceptable when they improve comprehension.
   - Avoid notebook-style code dumps and raw artifact listings.

6. Risks and next actions.
   - Include what would invalidate the thesis.
   - Include concrete next experiments or monitoring gates.

## Fact Rules

- monthly returns must come from artifact or facts API calculations.
- positive/negative month counts must come from artifact or facts API
  calculations.
- trade contribution must come from artifact or facts API calculations. If the
  current artifacts do not support round-trip PnL attribution, say it is not
  available instead of estimating it.
- maximum drawdown peak/trough dates and equity values must be computed from
  `equity_curve.csv` or an artifact-backed facts calculation. If the exact
  peak/trough period cannot be verified, omit the peak/trough dates and equity values
  and state only the artifact-backed `max_drawdown` metric.
- Do not hand-write or eyeball any of the above values from charts.
- When using facts API values, keep the wording human-readable and cite the
  source artifact or fact group in nearby text.
- When discussing why the SPEC was built a certain way, cite `spec_audit.json`
  and the selected recipe or catalog component. Do not infer provenance from
  the final YAML alone.
- When discussing what actually ran, cite `compiled_plan.json` or
  `execution_assumptions.json`. If those artifacts contradict
  `strategy_spec.yaml` or `spec_audit.json`, stop and return the run to
  `monitor-strategy-run` instead of writing a successful report.
- Do not write raw `<` or `>` comparison text in Markdown prose or tables.
  Markdown/HTML report QA strips HTML tags before checking dates and numeric
  claims, so a sentence like "trade count < threshold" can accidentally swallow
  nearby labels. Use words such as "低于" / "高于" or code formatting for literal
  operators.

## HTML Output

After drafting final Markdown in memory, render HTML from that same string and
publish all coupled outputs together:

Do not fall back to `zh` for an explicitly requested non-Chinese language. Map
known language names to valid HTML language tags, preserve explicit language
codes, and use `und` only when the requested language cannot be expressed as a
safe language tag.

Use the resolved runner's virtualenv Python from
`~/.config/open-xquant/agent.yaml` or `~/.config/open-xquant/agent-install.json`
when the workspace is using an installed SDK bundle. `oxq python` does not
exist. Do not run `uv run python` in an installed research workspace unless the
current directory is the open-xquant source worktree being developed.

```bash
<runner-python> - <<'PY'
from pathlib import Path
from oxq.report.assets import publish_report_artifacts
from oxq.report.html import render_markdown_html_report

report_dir = Path("<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>")
source_run_dir = Path("<phase_paths.09_backtests>/<run_id>")
report_language = "中文"  # Use the resolved report_language value from this phase.


def language_to_html_lang(language: str) -> str:
    normalized = language.strip().lower().replace("_", "-")
    html_lang_map = {
        "中文": "zh",
        "chinese": "zh",
        "zh": "zh",
        "zh-cn": "zh",
        "english": "en",
        "en": "en",
        "en-us": "en",
        "japanese": "ja",
        "日本語": "ja",
        "ja": "ja",
        "français": "fr",
        "french": "fr",
        "fr": "fr",
        "spanish": "es",
        "español": "es",
        "es": "es",
    }
    if normalized in html_lang_map:
        return html_lang_map[normalized]
    if normalized and all(ch.isalnum() or ch == "-" for ch in normalized):
        return normalized
    return "und"


def publish_rendered_report(markdown: str, writer_result_bytes: bytes) -> None:
    html_lang = language_to_html_lang(report_language)
    html = render_markdown_html_report(markdown, lang=html_lang)
    publish_report_artifacts(
        report_dir,
        {
            "research_report.md": markdown.encode("utf-8"),
            "research_report.html": html.encode("utf-8"),
            "writer_result.json": writer_result_bytes,
        },
        lock_subject=source_run_dir,
    )
PY
```

The HTML renderer may format the final Markdown, but it must not regenerate the
report narrative from templates.

## Governed Publication Lock

Route Markdown, HTML, `writer_result.json`, manifest changes, and any coupled
asset changes through
`publish_report_artifacts(report_dir, artifacts, *, lock_subject=None)`. The
mapping accepts safe relative keys and complete `bytes`; `None` deletes a
target. A callable builder executes under the final-selection lock so it can
re-read the locked baseline, run the baseline check, and commit one atomic
all-or-rollback batch. This applies to blocked results too. Direct
`Path.write_text`, `Path.write_bytes`, open-file writes, shell redirection, and
asset CLI publication paths are forbidden for these targets.

For an export whose `report_dir` is outside the governed workspace, pass
`lock_subject=source_run_dir`. If the writer needs coherent run evidence, wrap
the publisher with `run_digest_transaction(source_run_dir)`; the runtime owns
the order, acquiring the run lock first and the final-selection lock second.
Do not pre-acquire either publisher lock or nest another lock inside the
publisher callback.

## Red Lines

- Do not invent evidence.
- Do not modify metrics, audit files, robustness output, or backtest artifacts.
- Do not hide adverse evidence behind generic positive language.
- Do not omit blocking or unresolved `spec_audit.json` findings.
- Do not claim that a material execution rule ran unless `compiled_plan.json`
  or a runtime artifact shows it was preserved.
- Do not compare returns across runs unless their execution, cost, validation,
  data warmup, and compiled runtime assumptions are comparable.
- Do not promote a run with fatal audit findings, failed reproducibility,
  fragile robustness, or no usable OOS evidence.
- Do not call the strategy investable; describe research evidence and limits.
- Do not treat charts as proof; charts illustrate evidence that must already be
  supported by artifacts.
- Do not claim "the framework generated this report"; use "Evidence is
  generated by the framework; the narrative is authored by the Agent."
- Do not bypass this skill after recognizing it applies, even when all data is
  already in context.
- Do not ask the user questions directly from this skill. When required inputs
  are missing, write `writer_result.json` with `status: blocked` for the
  upstream orchestrator.

## Phase Result

Write `writer_result.json` after this phase:

```json
{
  "status": "pass | blocked | fail",
  "version_id": "<version_id>",
  "run_id": "<run_id>",
  "report_revision_id": "<report_revision_id>",
  "strategy_id": "<strategy_id>",
  "source_run_dir": "<phase_paths.09_backtests>/<run_id>",
  "language": "中文",
  "report_markdown": "<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.md",
  "report_html": "<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/research_report.html",
  "blocking_reason": "",
  "next_required_phase": "report_review",
  "errors": []
}
```

## Current Immutable Candidate Publication

The direct-path phase-result example above is a legacy working-head shape, not
a selectable producer target. Current report writing consumes a validated
`chart_build_result.json` and publishes one immutable report revision at
`<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/`. The
coordinator supplies a fresh `report_revision_id` and a fresh
`review_revision_id`; create the candidate directory exclusively and publish
`candidate_manifest.json` last. Its appearance seals the revision. Never
overwrite, delete, rename, merge, or repair a sealed revision or any evidence
reachable from any prior selection.

Before drafting, validate the chart handoff's
requested/applicable/generated/skipped set invariants, closed skip reason codes,
every generated asset hash, and exact `{path, sha256}` manifest reference.
Within the manifest require each safe package-relative `source.script`, its full
lowercase `source.script_sha256`, and exact current regular-file bytes. Recompute
the script SHA-256 and block on any script mutation. The same checks run again
inside the publication callable before the candidate manifest is committed.

The sealed candidate manifest uses this exact shape. Every artifact reference
is a safe workspace-relative current regular file below the same revision, and
every digest is full SHA-256 over exact bytes:

```json
{
  "schema_version": 1,
  "version_id": "v001",
  "run_id": "runA",
  "report_revision_id": "report_20260712_181000",
  "hash_algorithm": "sha256-file-bytes-v1",
  "source_run": {
    "path": "<phase_paths.09_backtests>/runA",
    "digest": "sha256:1111111111111111"
  },
  "artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/research_report.md",
      "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/research_report.html",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/writer_result.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "chart_build_result.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/chart_build_result.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    "report_assets/manifest.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/report_assets/manifest.json",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    }
  }
}
```

For `candidate_scoped_historical_report_revision`, use only the explicit
inactive version/run and current-state guard. Stage and publish under a fresh
`report_revision_id`, reserve the handoff's fresh `review_revision_id`, and
must not reactivate the inactive version or mutate `current.json`, phase state,
lineage state, or active run. The mode must not overwrite a prior candidate or
review revision. Return the sealed candidate reference to the coordinator for
the guarded `write -> review -> lineage -> comparison -> reselection` route;
prior revision bytes remain reachable.

Accept only the coordinator's closed schema-version-1 historical handoff; reject
aliases, extra keys, missing base/guard fields, and unknown reason codes.
