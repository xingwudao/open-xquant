---
name: build-strategy-spec
description: >-
  Build open-xquant strategy_spec.yaml files from audited strategy idea briefs
  for multi-Agent systems; stops after deterministic validation and writes a
  builder phase result for downstream orchestration.
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

# Strategy Builder

Build or edit `strategy_spec.yaml` from an audited strategy idea brief. This
skill owns only the SPEC build phase in a multi-Agent research workflow.

## Read-Only Validation-Only Mode

When the router asks only to validate an existing SPEC, use this mode instead
of the builder workflow below. Run `<resolved_runner> spec validate
<existing_strategy_spec.yaml>` and report the validator result without writing
files. This mode does not require `strategy_idea_brief.json` and does not
require `strategy_idea_audit.json`.

- Do not run `oxq registry export` or any equivalent catalog export.
- Do not create or modify `builder_phase_result.json`.
- Do not create or modify `spec_mapping_contract.json`.
- Do not create or modify `spec_mapping_notes.md`, `component_catalog.json`,
  or the existing SPEC.
- Do not apply the builder input gates, output ownership rules, or phase-result
  requirements below. They apply only to normal builder mode.

## Reference Loading

Read these one-level references only when the matching work is needed:

- `references/spec-shape-and-candidates.md`: SPEC YAML shape, stable candidate
  values, tradability, lag, timing, warmup, and `latest_available` rules.
- `references/source-mapping-and-catalog.md`: source mapping, unsupported
  disclosure, mapping-contract fields, component catalog query patterns,
  canonical recipe selection, and cross-sectional feasibility.
- `references/builder-output-schema.md`: open-xquant version provenance,
  validation commands, `builder_phase_result.json`, and output schema.

## Version-Governed Artifact Gate

Before any catalog export, SPEC initialization, SPEC edit, or validation, read
`current.json` and use `active_version` as `version_id`. If no active version
exists, block and return to `manage-strategy-version`.

The required audited idea inputs are:

```text
<phase_paths.01_brainstorm>/strategy_idea_brief.json
<phase_paths.02_idea_audit>/strategy_idea_audit.json
```

This builder may write only inside:

```text
<phase_paths.04_spec_build>/
```

Required outputs are:

```text
<phase_paths.04_spec_build>/strategy_spec.yaml
<phase_paths.04_spec_build>/component_catalog.json
<phase_paths.04_spec_build>/spec_build_notes.md
<phase_paths.04_spec_build>/spec_mapping_notes.md
<phase_paths.04_spec_build>/spec_mapping_contract.json
<phase_paths.04_spec_build>/builder_phase_result.json
```

Do not write root-level `strategy_spec.yaml`, `component_catalog.json`,
`spec_build_notes.md`, `spec_mapping_notes.md`,
`spec_mapping_contract.json`, or `builder_phase_result.json`.

## Scope

Do:

- require `strategy_idea_brief.json` and passing `strategy_idea_audit.json`
  before any SPEC or catalog work
- convert the audited strategy idea into `strategy_spec.yaml`
- use the component catalog and canonical recipes before choosing components
- preserve explicit user requirements and user-confirmed candidate values
- write concise `spec_build_notes.md` when component or recipe choices matter
- write `spec_mapping_notes.md` and `spec_mapping_contract.json` that map
  audited idea fields to executable SPEC fields and disclose unsupported fields
- inventory every flattened source field in `source_fields` and require exactly
  one `field_mappings` row for each inventory entry before builder pass
- run deterministic spec validation
- write `builder_phase_result.json` for the downstream orchestrator

Do not:

- produce `spec_audit.json`
- approve assumptions on behalf of the user
- call audit skills
- ask the user to complete brainstorm phases
- inspect, list, or read market data files, parquet files, manifests, or
  provider directories
- download market data
- run `oxq strategy compile`
- run `oxq backtest run`
- attach provenance for runs, backtests, monitoring, or reports
- run monitoring, robustness, report writing, or report review
- describe an unaudited spec as ready for formal research

## Audited Idea Input Gate

Before doing any SPEC work, require both artifacts:

- `strategy_idea_brief.json`
- `strategy_idea_audit.json`

The idea audit must have `status: pass`, `idea_workflow_pass: true`, and
`next_required_phase: build`. If either artifact is missing, stale, blocked,
or failing, stop and return a blocked `builder_phase_result.json` with
`next_required_phase: brainstorm`.

Hard gate:

- Do not run `oxq spec init` before `strategy_idea_audit.json` passes.
- Do not run `oxq registry export` before `strategy_idea_audit.json` passes.
- Do not write or edit `strategy_spec.yaml` before `strategy_idea_audit.json`
  passes.

If the audited brief contains unresolved defaults, candidates, contradictions,
or missing phase evidence, do not choose values. Return the issue to
`brainstorm-strategy-idea` through the coordinator.

## Runner Resolution

In a new research directory, `uv run oxq` may fail because open-xquant is
installed as a long-lived Agent capability, not as a package in that directory.
Before running commands after the audited idea gate passes:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

When this skill needs a Python SDK snippet in an installed research workspace,
use the resolved runner's virtualenv Python from the same `bin` directory as
the runner. `oxq run python` does not exist, and `uv run python` is only
appropriate inside a source checkout where `oxq` is installed for that project.

## Provenance And SPEC Shape

Every generated `strategy_spec.yaml` must distinguish the SPEC schema version
from the open-xquant package version:

- `schema_version` is the SPEC schema version.
- `required_oxq_version` is the open-xquant package version used by the
  resolved runner for this builder phase.

Do not change `schema_version` to the package version. When writing or
materially editing `strategy_spec.yaml`, write `required_oxq_version` from the
resolved runner/package version. If the runner cannot report a version, block
instead of emitting a SPEC with an empty `required_oxq_version`.

Use `references/spec-shape-and-candidates.md` before writing or repairing SPEC
fields. It contains the current `StrategySpec` YAML shape and stable candidate
value rules.

## Data Inspection Boundary

This builder does not own data coverage or data quality investigation. Do not
list `data_dir`, read parquet files, inspect columns, call `list_symbols`,
call `inspect_symbol`, or resolve date coverage from local market data. The
`explore-data` skill and `oxq-data-inspection-worker` own those actions and
write:

```text
<phase_paths.05_data_inspection>/data_inspection_result.json
<phase_paths.05_data_inspection>/data_availability_report.md
```

The builder may use a data fact only when it is already present in the audited
idea, explicitly confirmed by the user, or recorded in a prior
`data_inspection_result.json` for the active version. Cite that artifact in
`spec_build_notes.md` and `spec_mapping_contract.json`.

If `latest_available`, required column coverage, common symbol date range,
price adjustment evidence, provider readiness, or suspension/tradability data
availability is unknown, do not infer it and do not inspect the files. Return
`status: blocked` with `next_required_phase: data_inspection`. If a prior data
inspection artifact says required data is unavailable, preserve the requested
strategy semantic in `spec_mapping_contract.json` and block instead of
removing the requirement from the SPEC.

## SPEC Audit Repair Handoff

When a blocked SPEC audit result returns `next_required_phase: build`, read
`<phase_paths.06_spec_audit>/spec_audit.json` and `audit_notes.md`
before editing the SPEC. The auditor is read-only and does not repair YAML.
The builder owns the repair.

For contradictions that include `effective_field_path`, `source_yaml_path`, and
`builder_required_fix`:

- move the value to the effective field path that open-xquant actually parses
- remove the non-operative YAML path when it is not part of
  `StrategySpec.from_yaml(...).to_effective_dict()`
- preserve the original user-confirmed value and cite the audit finding in
  `spec_build_notes.md`
- update `spec_mapping_notes.md`, `spec_mapping_contract.json`, and
  `builder_phase_result.json`
- rerun `oxq spec validate` on
  `<phase_paths.04_spec_build>/strategy_spec.yaml`

Example: if the SPEC audit says `source_yaml_path: portfolio.initial_cash`,
`effective_field_path: execution.initial_cash`, and
`builder_required_fix: move the value to execution.initial_cash and remove
portfolio.initial_cash`, write the confirmed value under
`execution.initial_cash` and remove `portfolio.initial_cash`. Do not preserve a
known non-operative field merely as documentation.

## Build Flow

1. Verify the audited idea gate.
2. Resolve the runner and `required_oxq_version`.
3. Read `references/spec-shape-and-candidates.md`.
4. Read `references/source-mapping-and-catalog.md` before catalog export,
   component choice, recipe choice, custom-component blocking, source mapping,
   or unsupported disclosure.
5. Initialize a SPEC only after the audited idea gate and component catalog gate
   pass:

   ```bash
   uv run oxq spec init "<audited strategy idea>" --out <phase_paths.04_spec_build>/strategy_spec.yaml
   ```

6. Replace any template values that are not backed by
   `strategy_idea_brief.json` and `strategy_idea_audit.json`.
7. Edit `strategy_spec.yaml` so it contains only audited strategy values and
   documented implementation choices.
8. Write `spec_build_notes.md`, `spec_mapping_notes.md`,
   `spec_mapping_contract.json`, and `builder_phase_result.json`.

Do not run the initializer with its default output path. The only acceptable
initializer target is `<phase_paths.04_spec_build>/strategy_spec.yaml`
or a temporary file inside the same `04_spec_build` directory.

## Validate And Output

Run deterministic validation:

```bash
uv run oxq spec validate <phase_paths.04_spec_build>/strategy_spec.yaml
```

Fix fatal validation errors before completing the builder phase only when they
are mechanical SPEC-shape or translation errors. If a fatal validation error
comes from an intentionally unresolved blocked dependency such as
`latest_available`, unconfirmed required data columns, or unavailable
tradability data, do not invent replacement values. Keep `validation.status:
fail`, return `status: blocked`, set `next_required_phase: data_inspection`
or the earliest owning phase, and explain the blocker in
`builder_phase_result.json`. Warnings are allowed, but record them for the
downstream orchestrator.

Read `references/builder-output-schema.md` before writing
`builder_phase_result.json`. The builder skill stops after writing the phase
artifacts. The downstream multi-Agent system decides which audit, compile,
execution, monitor, or report worker runs next.

## Red Lines

- Do not skip the Audited Idea Input Gate.
- Do not skip the component catalog gate.
- Do not run catalog export, `oxq spec init`, or SPEC writes before the idea
  audit passes.
- Never run `oxq spec init` without an explicit `--out` path under
  `<phase_paths.04_spec_build>/`, and do not create or consult
  root-level `strategy_spec.yaml`. The default command writes root-level
  `strategy_spec.yaml`, which is a workspace layout violation even if deleted
  later. If accidentally created, delete it,
  record `layout_violation` in `builder_phase_result.json`, and keep the
  builder result blocked.
- Do not invent component names when a catalog component or recipe exists.
- Do not omit `spec_mapping_notes.md`, `unmapped_source_fields`, or
  `unsupported_mappings` when the source idea contains fields beyond the
  executable SPEC surface.
- Do not omit `spec_mapping_contract.json` or hide strategy semantics as
  non-material Studio/report metadata.
- Do not convert a full backtest interval into required OOS without user
  confirmation in the audited idea.
- Do not treat template defaults as confirmed values.
- Do not write `spec_audit.json`.
- Do not list or read market data files, parquet files, data manifests, or
  provider directories; route data coverage and column questions to
  `explore-data`.
- Do not call audit, compile, backtest, monitor, robustness, experiment, or
  report skills from this builder skill.
- Do not call component creation skills from this builder skill. When a custom
  component is required, write `needs_custom_component` and stop.
- Do not run formal backtests from this skill.
