---
name: author-component
description: >-
  Author workspace-local OpenXQuant custom components from component_request.json,
  with tests, manifest hashing, catalog refresh, and no global SDK mutation.
---

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
versions/<version_id>/03_component_authoring/component_request.json
versions/<version_id>/03_component_authoring/result.json
versions/<version_id>/03_component_authoring/component_manifest.json
versions/<version_id>/03_component_authoring/component_catalog.json
```

Do not write root-level `component_request.json`, root-level `result.json`,
root-level `component_manifest.json`, or root-level `component_catalog.json` as
formal outputs in a version-governed workspace.
Do not write root-level `result.json`.

Reusable workspace-local code bundles, when a component is actually authored,
live under:

```text
components/bundles/<bundle_id>/
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
components/bundles/<bundle_id>/
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

versions/<version_id>/03_component_authoring/
  component_request.json
  component_manifest.json
  component_catalog.json
  result.json
```

Task-local scratch files may live inside the phase directory while authoring,
but formal reusable code belongs under `components/bundles/<bundle_id>/`.
Do not write outside these boundaries unless the coordinator explicitly
provides another task-local root.

## Generated Cache Cleanup

Before returning a handoff result, remove generated test/build/cache directories
from `components/bundles/<bundle_id>/` and from the phase-local
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
     --out versions/<version_id>/03_component_authoring/component_catalog.json
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
11. Write `components/bundles/<bundle_id>/component_manifest.json` without
    `bundle_hash`, compute it with:

    ```bash
    uv run oxq component-manifest hash components/bundles/<bundle_id>/component_manifest.json
    ```

12. Update `components/bundles/<bundle_id>/component_manifest.json` with the
    returned `bundle_hash`.
13. Validate importability and hash:

    ```bash
    uv run oxq component-manifest validate components/bundles/<bundle_id>/component_manifest.json
    ```

14. Refresh the catalog with:

    ```bash
    uv run oxq registry export \
      --component-manifest components/bundles/<bundle_id>/component_manifest.json \
      --out components/bundles/<bundle_id>/component_catalog.json
    ```

15. Copy or reference the bundle manifest and catalog from the phase-local
    `versions/<version_id>/03_component_authoring/` artifacts.
16. Remove generated cache/build artifacts that must not remain.
17. Write `versions/<version_id>/03_component_authoring/result.json`.

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

Write `components/bundles/<bundle_id>/component_manifest.json` with:

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

Write `versions/<version_id>/03_component_authoring/result.json`:

```json
{
  "schema_version": 1,
  "role": "component_author",
  "phase": "component_authoring",
  "status": "component_ready",
  "component_kind": "Signal",
  "component_name": "RiskAdjustedTiming",
  "artifacts": {
    "component_manifest": "components/bundles/<bundle_id>/component_manifest.json",
    "component_catalog": "components/bundles/<bundle_id>/component_catalog.json",
    "source_root": "components/bundles/<bundle_id>/custom_components/",
    "phase_component_manifest": "versions/<version_id>/03_component_authoring/component_manifest.json",
    "phase_component_catalog": "versions/<version_id>/03_component_authoring/component_catalog.json",
    "tests": [
      "components/bundles/<bundle_id>/custom_components/tests/test_risk_adjusted_timing.py"
    ]
  },
  "hashes": {
    "component_bundle_hash": "sha256:...",
    "component_catalog_hash": "sha256:..."
  },
  "tests": [
    {
      "command": "pytest components/bundles/<bundle_id>/custom_components/tests/test_risk_adjusted_timing.py -q",
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
