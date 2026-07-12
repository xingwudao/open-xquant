---
name: audit-strategy-spec
description: Audit strategy_spec.yaml field provenance before backtests.
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

# Spec Auditor

Use this skill after `build-strategy-spec` validates `strategy_spec.yaml` and
before any `oxq backtest run`. Its job is to prevent unapproved strategy
assumptions from entering a formal experiment.

This is a SPEC calibration audit. The separate `audit-strategy-idea` skill owns
the brainstorm workflow audit.

## Reference Loading

Read these one-level references only when the matching audit area is needed:

- `references/field-classification.md`: effective field classification,
  default handling, material fields, data warmup, material categories, and
  source-boundary rules.
- `references/component-and-mapping-audit.md`: component provenance, catalog
  hash checks, unsupported mapping disclosure, mapping-contract validation,
  recipe examples, and cross-sectional feasibility.
- `references/confirmation-and-output.md`: two-step audit confirmation, default
  confirmation checklist, `spec_audit.json`, confirmation-event rules, hashes,
  strict validation, and output details.

## Inputs

- `strategy_spec.yaml`
- `strategy_idea_brief.json`
- `strategy_idea_audit.json`
- `builder_phase_result.json`
- `spec_build_notes.md` when available
- `spec_mapping_notes.md` when available
- `spec_mapping_contract.json` when available
- `data_inspection_result.json` when the builder cites data coverage, column
  coverage, common date range, price adjustment, or `latest_available`
  resolution evidence
- Agent-provided raw conversation history. Do not assume a filename or path.
  The invoking Agent must supply the source text or Studio-provided object in a
  task-local variable such as:

  ```text
  CONVERSATION_HISTORY_RAW:
  <paste or load the exact user/agent conversation text for this experiment>
  ```

  If Studio provides a `conversation.json` object or path variable, use that
  provided value. Do not hardcode `conversation.json` as a required path.
- `component_catalog.json` produced by the builder after the audited idea gate
  passes

## Version-Governed Artifact Gate

Before auditing, read `current.json` and use `active_version` as `version_id`.
If no active version exists, block and return to `manage-strategy-version`.

Read phase artifacts from these paths:

```text
<phase_paths.01_brainstorm>/strategy_idea_brief.json
<phase_paths.02_idea_audit>/strategy_idea_audit.json
<phase_paths.04_spec_build>/strategy_spec.yaml
<phase_paths.04_spec_build>/builder_phase_result.json
<phase_paths.04_spec_build>/component_catalog.json
<phase_paths.04_spec_build>/spec_build_notes.md
<phase_paths.04_spec_build>/spec_mapping_notes.md
<phase_paths.04_spec_build>/spec_mapping_contract.json
<phase_paths.05_data_inspection>/data_inspection_result.json
```

Write SPEC audit artifacts only under:

```text
<phase_paths.06_spec_audit>/spec_audit.json
<phase_paths.06_spec_audit>/audit_notes.md
<phase_paths.06_spec_audit>/spec_confirmation_table.md  # only for all_pass / user confirmation pending or confirmed
```

The coordinator must show
`<phase_paths.06_spec_audit>/spec_confirmation_table.md` to the user
when the audit reaches all pass but confirmation is pending.
Do not create a placeholder `spec_confirmation_table.md` for
`audit_conclusion: blocked`; blocked audits may omit `spec_confirmation_table`
from `spec_audit.json` or set it to `null`.

Do not write root-level `spec_audit.json`, `audit_notes.md`, or
`spec_confirmation_table.md`.
Do not write `<phase_paths.04_spec_build>/component_catalog.json` or
any other builder phase artifact.

## Read-Only SPEC Boundary

The SPEC auditor is read-only for builder outputs.
Do not write, patch, rewrite, normalize, or repair
`<phase_paths.04_spec_build>/strategy_spec.yaml` or any other
`<phase_paths.04_spec_build>/` artifact during this skill.

If the audit discovers that a user-confirmed value was misplaced, ignored, dropped, or mistranslated
in YAML, block with `next_required_phase: build`.
Examples include a confirmed `execution.initial_cash` value placed under an
ignored section, a confirmed lot-size value represented only by a non-operative
field, or a user-confirmed rebalance value hidden behind an effective default.
Do not convert that finding into a confirmation-table row and do not ask the
user to confirm the wrong effective value. The builder must make the YAML change,
rerun `oxq spec validate`, refresh build notes or mapping artifacts when
needed, and return to `build-strategy-spec` before this audit resumes.

Before completing the audit, compare every user-confirmed source value from the
audited idea to the effective StrategySpec value that will actually run. Use
`StrategySpec.from_yaml(...).to_effective_dict()` as the field-path source for
effective SPEC auditing. Do not treat a YAML-only key that is absent from the
effective dictionary as a valid confirmed field.

Read `references/field-classification.md` for the full effective-field audit
rules, including the `portfolio.initial_cash` blocker example.

## Runner Resolution

In a new research directory, `uv run oxq` may fail because open-xquant is
installed as a long-lived Agent capability, not as a package in that directory.
Before running deterministic validation commands:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq` or bare `oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

## Audited Idea Gate

First fast-fail if the brainstorm audit is missing or blocked. Before catalog
work, field provenance, or grouped confirmation questions, check:

- `strategy_idea_brief.json` exists and its hash matches the builder result
  when a hash is available.
- `strategy_idea_audit.json` exists.
- `strategy_idea_audit.json` has `status: pass`,
  `idea_workflow_pass: true`, and `next_required_phase: build`.

If any check fails, do not continue the SPEC audit. Write or return a blocked
result with:

- `status: block`
- `spec_provenance_pass: false`
- `next_required_phase: brainstorm`
- a blocking finding that names the missing or failed idea artifact

Only continue when the audited idea gate passes.

## Spec Calibration Audit

After the audited idea gate passes, verify the spec faithfully maps the audited idea.
This audit answers whether `strategy_spec.yaml` is a correct, confirmed, and
catalog-backed implementation of the already audited strategy description.

Check that:

- every material `strategy_spec.yaml` value is confirmed by
  `strategy_idea_brief.json` or by later explicit user evidence
- no value depends only on a template, parser, runtime, or Agent-chosen default
- every confirmed phase in the idea brief appears in the SPEC field group it
  controls
- user-stated requirements are not missing from the SPEC
- SPEC values do not contradict the audited idea or raw conversation
- Indicator definitions from the idea brief map to `signal.indicators.*`
- signal rules map to `signal.rules.*`
- portfolio, execution, cost, metrics, robustness, and decision policy fields
  map to their corresponding audited idea sections

When the SPEC faithfully maps the idea but effective defaults still need user
approval, block with grouped confirmation questions. When the SPEC
mistranslates the audited idea, block with `next_required_phase: build`. When
the audited idea itself is incomplete or unconfirmed, block with
`next_required_phase: brainstorm`.

A SPEC mistranslation includes any effective StrategySpec value that contradicts
the audited idea because the YAML field was misplaced, ignored, dropped, or
mapped to a non-operative field. This is a builder error, not a user
confirmation problem.

Use these next-phase labels in findings:

- `next_required_phase: brainstorm`
- `next_required_phase: build`
- `next_required_phase: data_inspection`
- `next_required_phase: user_spec_confirmation`
- `next_required_phase: runtime_audit`

## Provenance And Data Boundaries

Audit `required_oxq_version` as a material provenance field.
OpenXQuant version provenance is not a user strategy assumption, so the user
does not choose a default version value during brainstorm. The builder must
still record it from the resolved runner/package version, and the coordinator
must show it in the full SPEC confirmation table.

A missing, empty, contradictory, or stale `required_oxq_version` blocks formal
backtest and must return to `next_required_phase: build`.

Data Boundary Audit: data coverage, column coverage, common date ranges, price
adjustment evidence, provider readiness, and `latest_available` resolution must
come from a prior
`<phase_paths.05_data_inspection>/data_inspection_result.json` or from
explicit user-confirmed data snapshot evidence.
Do not accept builder-authored notes saying it inspected parquet files, listed
data directories, or inferred the latest date as sufficient evidence.

Block with `next_required_phase: data_inspection` when a material SPEC value
depends on data facts and no data-inspection artifact or user-confirmed
snapshot supports it. Block with `next_required_phase: build` when the builder
removed a confirmed data or tradability requirement because local data was
missing or unverified.

Read `references/field-classification.md` for source-boundary and field
classification details.

## Component, Mapping, And Cross-Sectional Audit

Before approving a spec for backtest, audit component provenance against the
same catalog used while building the spec. Also audit
`spec_mapping_notes.md`, `builder_phase_result.json`, and
`spec_mapping_contract.json` so unsupported strategy semantics cannot pass
silently.

Read `references/component-and-mapping-audit.md` before approving catalog
hashes, component selections, recipes, unsupported mappings, mapping-contract
rows, or cross-sectional runtime feasibility.

## Gate

Any `default`, `unconfirmed`, `contradiction`, or `agent_added` effective field
blocks formal backtest until the user confirms the grouped assumption or
provides a replacement value. Any blocking component provenance issue blocks
backtest. Train/test splits and required OOS settings are material and must not
pass silently.

Always group related fields instead of asking one question per YAML key:

- data warmup and local data coverage
- execution assumptions
- cost assumptions
- train/test split
- cash and risk-free assumptions
- benchmark and success metric
- component/catalog provenance

Ask the user to either confirm the group or provide replacement values. If the
user provides replacement values, do not edit `strategy_spec.yaml` in this
audit. Write a blocked audit with `next_required_phase: build`; the builder
must apply the change, rerun deterministic validation, and then return for
another audit pass.

## Two-Step Spec Audit Gate

The SPEC audit has two semantic states before downstream runtime work:

- `audit_conclusion: all_pass`: the auditor found no remaining provenance,
  calibration, default, component, recipe, or mapping blockers.
- `user_confirmation_status: confirmed`: the user explicitly confirmed the
  complete SPEC table shown by the coordinator.

These are not the same. When the auditor reaches `audit_conclusion: all_pass`
but the user has not confirmed the table, write `spec_audit.json` with
`status: block`, `user_confirmation_status: pending`, and
`next_required_phase: user_spec_confirmation`.

Only after the user explicitly confirms the full Markdown table may the audit
become confirmed. Then update or write `spec_audit.json` with `status: pass`,
`user_confirmation_status: confirmed`, a valid `confirmation_event` reference,
and `next_required_phase: runtime_audit`.

Read `references/confirmation-and-output.md` before writing confirmation
tables, default confirmation checklists, `spec_audit.json`, hashes, or strict
validation commands.

## Output

Report a compact summary:

- audited idea artifact hashes
- confirmed fields
- default fields awaiting checklist confirmation
- unconfirmed fields that block progress
- selected catalog components and recipes
- spec calibration findings
- component provenance issues, including catalog hash mismatch, missing
  components, non-canonical recipe decomposition, or missing user-requested
  semantics
- unsupported_mappings disclosure and whether no unsupported source fields were
  found
- path to `spec_audit.json`
- Full Spec Confirmation Table when `audit_conclusion: all_pass` is reached
  but `user_confirmation_status` is not confirmed
- Default Confirmation Checklist when defaults need user approval
- blocking confirmation questions

Do not run or approve a backtest while blocking fields remain. After this skill
is confirmed by the user, the next formal gate is `audit-runtime-semantics`.
