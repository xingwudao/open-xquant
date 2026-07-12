---
name: author-component
description: >-
  Author workspace-local OpenXQuant custom components from component_request.json,
  with tests, manifest hashing, catalog refresh, and no global SDK mutation.
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

Resolve `components_dir` from `paths.components_dir`; use `components` only
when that key is absent. Require a safe relative path whose resolved target
stays inside the workspace. Reject absolute paths, traversal outside the
workspace, and symlink escapes whose resolved target leaves the workspace. Use
the resolved `<components_dir>` for every reusable component bundle path below.

# Component Author

Create workspace-local custom OpenXQuant components only after the component
catalog and recipe catalog cannot satisfy the requested behavior.

This is code authoring, testing, registration, and provenance work. It is not
strategy building, spec auditing, runtime auditing, backtest execution, or
report writing.

## Inputs

Read:

- `component_request.json`
- `component_catalog.json`
- conversation context supplied by the coordinator
- `confirmations.json` when present
- workspace root or explicit extension root

## Version-Governed Output Gate

Before reading or writing component-authoring artifacts in a research
workspace, read `current.json` and use `active_version` as `version_id`. If no
active version exists, block and return to `manage-strategy-version`.

Formal component-authoring phase artifacts live only under:

```text
<phase_paths.03_component_authoring>/component_request.json
<phase_paths.03_component_authoring>/result.json
<phase_paths.03_component_authoring>/component_manifest.json
<phase_paths.03_component_authoring>/component_catalog.json
```

Do not write root-level `component_request.json`, root-level `result.json`,
root-level `component_manifest.json`, or root-level `component_catalog.json` as
formal outputs in a version-governed workspace.
Do not write root-level `result.json`.

Reusable workspace-local code bundles, when a component is actually authored,
live under:

```text
<components_dir>/bundles/<bundle_id>/
  component_manifest.json
  component_catalog.json
  custom_components/
```

The phase-local `result.json` must point to the reusable bundle paths. If the
request is blocked, still write the phase-local `component_request.json` and
`result.json` so the `03_component_authoring/` stage is auditable without
creating an empty directory.

If `component_request.json` does not identify exactly one component kind from
`Indicator`, `Signal`, `Rule`, or `PortfolioOptimizer`, stop with a blocked
result. Do not guess. Workspace-local `Rule` authoring is currently blocked:
the audited SPEC/runtime path only supports built-in runtime rules such as
`RebalanceFrequencyRule`, and a custom workspace `Rule` would not be consumed
by formal compile/backtest semantics.

## Output Layout

Default authored component code to a workspace-local reusable bundle:

```text
<components_dir>/bundles/<bundle_id>/
  component_manifest.json
  component_catalog.json
  custom_components/
    pyproject.toml
    oxq_components/
      __init__.py
      indicators/
      signals/
      portfolio/
    tests/

<phase_paths.03_component_authoring>/
  component_request.json
  component_manifest.json
  component_catalog.json
  result.json
```

Task-local scratch files may live inside the phase directory while authoring,
but formal reusable code belongs under `<components_dir>/bundles/<bundle_id>/`.
Do not write outside these boundaries unless the coordinator explicitly
provides another task-local root.

## Generated Cache Cleanup

Before returning a handoff result, remove generated test/build/cache directories
from `<components_dir>/bundles/<bundle_id>/` and from the phase-local
`03_component_authoring/` scratch area. These paths must not remain in authored
component bundles or phase artifacts:

- `__pycache__/`
- `.pytest_cache/`
- `*.egg-info/`
- `.mypy_cache/`
- `.ruff_cache/`
- `*.pyc`
- `*.pyo`

Do not list these generated cache paths as formal artifacts in `result.json`.
If cleanup fails, mark the result `failed` or `blocked` instead of handing off a
dirty component bundle.

## Workflow

1. Read `component_request.json`.
2. Re-check the current registry and recipe catalog. Use:

   ```bash
   uv run oxq registry export \
     --out <phase_paths.03_component_authoring>/component_catalog.json
   ```

3. If an existing component or recipe satisfies the request, write a blocked
   `result.json` and stop.
4. If the request asks for a workspace-local `Rule`, write a blocked
   `result.json` explaining that custom `Rule` components require explicit
   OpenXQuant framework development and runtime support before they can be
   used in audited backtests.
5. Block when behavior, formula, thresholds, output domain, state semantics, or
   causal suitability are ambiguous.
6. Create or update the local extension package.
7. Write targeted tests before implementation for new components.
8. Implement using OpenXQuant protocols and existing component patterns.
9. Register the component from the extension package, without modifying the
   installed SDK bundle.
   Use an extension module namespace such as `oxq_components.*`; do not declare
   workspace components under `oxq.*`.
10. Run targeted tests.
11. Write `<components_dir>/bundles/<bundle_id>/component_manifest.json` without
    `bundle_hash`, compute it with:

    ```bash
    uv run oxq component-manifest hash <components_dir>/bundles/<bundle_id>/component_manifest.json
    ```

12. Update `<components_dir>/bundles/<bundle_id>/component_manifest.json` with the
    returned `bundle_hash`.
13. Validate importability and hash:

    ```bash
    uv run oxq component-manifest validate <components_dir>/bundles/<bundle_id>/component_manifest.json
    ```

14. Refresh the catalog with:

    ```bash
    uv run oxq registry export \
      --component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json \
      --out <components_dir>/bundles/<bundle_id>/component_catalog.json
    ```

15. Copy or reference the bundle manifest and catalog from the phase-local
    `<phase_paths.03_component_authoring>/` artifacts.
16. Remove generated cache/build artifacts that must not remain.
17. Write `<phase_paths.03_component_authoring>/result.json`.

## Cross-Sectional Component Feasibility

When `component_request.json` describes cross-sectional logic, do not treat the
requested component kind as final until the runtime input requirements are
checked. Cross-sectional logic includes same-date winsorization, ranking,
z-score, percentile rank, neutralization, or clipping across multiple symbols.

Use `PortfolioOptimizer first` as a feasibility preference because current
`PortfolioOptimizer.optimize()` receives all-symbol same-date inputs through
`dict[str, DataFrame]`. This preference is not a guarantee:

- If the request says `Indicator` but the behavior requires all-symbol same-date input,
  do not implement it as an `Indicator`.
- reclassify the candidate kind to `PortfolioOptimizer` only when the behavior
  can be implemented inside allocation from existing columns, with documented
  output weights and no hidden factor column claim.
- If the optimizer cannot faithfully implement the requested behavior, do not force `PortfolioOptimizer`;
  write `status: blocked` with `blocked_reason`
  containing `framework_unsupported`.
- If the request requires a reusable cross-sectional factor column, a distinct
  pre-portfolio factor transform, or runtime artifacts that the current
  framework cannot represent, block and explain that first-class
  cross-sectional framework support is required.

Do not silently downgrade a cross-sectional transform into a per-symbol
`Indicator`, and do not claim that `TopNRanking.max_weight` is the same as
score winsorization.

## Component Requirements

Indicator:

- Compute a numeric `pd.Series`.
- Preserve input index alignment.
- Document units, sign, warmup behavior, and NaN handling.
- Include at least one hand-calculated test case.

Signal:

- Return a boolean or declared categorical `pd.Series`.
- Categorical trading intent must use exact uppercase labels such as `BUY`,
  `SELL`, and `HOLD`.
- Make causal behavior explicit.
- Block future-looking logic unless the user explicitly accepts that it is not
  suitable for audited causal backtests.

Rule:

- Block workspace-local `Rule` authoring by default. A custom `Rule` is only
  allowed when the user explicitly states this is OpenXQuant framework
  development and the implementation adds audited spec validation, compile,
  runtime, and backtest support in the source tree.
- Do not emit `component_ready` for a workspace-local custom `Rule`.

PortfolioOptimizer:

- Return a non-empty target-weight dictionary.
- Weights must sum to `1.0`.
- Invalid or empty inputs should fall back to `{"CASH": 1.0}` or another
  documented safe behavior.
- Make state handling explicit when consuming categorical labels such as
  `BUY`, `SELL`, and `HOLD`.

## Tests

Minimum targeted coverage:

- protocol compliance
- registered name
- deterministic output
- hand-calculated scenario
- no-trigger or neutral scenario when applicable
- invalid or insufficient data behavior
- categorical output domain when applicable
- stateful behavior when applicable

The result artifact must include commands and pass/fail status.

## Manifest

Write `<components_dir>/bundles/<bundle_id>/component_manifest.json` with:

```json
{
  "schema_version": 1,
  "extension_id": "custom_components",
  "extension_root": "custom_components",
  "bundle_hash": "sha256:...",
  "components": [
    {
      "name": "RiskAdjustedTiming",
      "kind": "Signal",
      "source": "workspace_extension",
      "module": "oxq_components.signals.risk_adjusted_timing",
      "class": "RiskAdjustedTiming",
      "protocol": "Signal",
      "output_domain": ["BUY", "SELL", "HOLD"],
      "parameters": {
        "return_period": 20,
        "volatility_period": 20
      },
      "tests": [
        "custom_components/tests/test_risk_adjusted_timing.py"
      ],
      "source_path": "oxq_components/signals/risk_adjusted_timing.py",
      "source_hash": "sha256:...",
      "test_hash": "sha256:..."
    }
  ]
}
```

The `bundle_hash` must be computed by OpenXQuant, not invented by the worker.

## Result Artifact

Write `<phase_paths.03_component_authoring>/result.json`:

```json
{
  "schema_version": 1,
  "role": "component_author",
  "phase": "component_authoring",
  "status": "component_ready",
  "component_kind": "Signal",
  "component_name": "RiskAdjustedTiming",
  "artifacts": {
    "component_manifest": "<components_dir>/bundles/<bundle_id>/component_manifest.json",
    "component_catalog": "<components_dir>/bundles/<bundle_id>/component_catalog.json",
    "source_root": "<components_dir>/bundles/<bundle_id>/custom_components/",
    "phase_component_manifest": "<phase_paths.03_component_authoring>/component_manifest.json",
    "phase_component_catalog": "<phase_paths.03_component_authoring>/component_catalog.json",
    "tests": [
      "<components_dir>/bundles/<bundle_id>/custom_components/tests/test_risk_adjusted_timing.py"
    ]
  },
  "hashes": {
    "component_bundle_hash": "sha256:...",
    "component_catalog_hash": "sha256:..."
  },
  "tests": [
    {
      "command": "pytest <components_dir>/bundles/<bundle_id>/custom_components/tests/test_risk_adjusted_timing.py -q",
      "status": "pass"
    }
  ],
  "blocked_reason": null
}
```

Allowed statuses are `component_ready`, `blocked`, and `failed`.

Blocked results must include a clear `blocked_reason` and a grouped question
for the user or supervising agent.

## Red Lines

- Do not build or edit `strategy_spec.yaml`.
- Do not write `spec_audit.json`.
- Do not write `runtime_audit.json`.
- Do not run formal backtests.
- Do not write reports.
- Do not modify generated run artifacts.
- Do not modify the installed SDK bundle.
- Do not modify OpenXQuant source code unless the user explicitly says the task
  is framework development.
- Do not install unapproved third-party packages.
- Do not download network data.
- Do not silently continue after failed tests, failed import, failed registry
  visibility, or failed manifest hashing.
