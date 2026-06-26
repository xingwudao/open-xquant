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

If `component_request.json` does not identify exactly one component kind from
`Indicator`, `Signal`, `Rule`, or `PortfolioOptimizer`, stop with a blocked
result. Do not guess. Workspace-local `Rule` authoring is currently blocked:
the audited SPEC/runtime path only supports built-in runtime rules such as
`RebalanceFrequencyRule`, and a custom workspace `Rule` would not be consumed
by formal compile/backtest semantics.

## Output Layout

Default to a workspace-local extension:

```text
custom_components/
  pyproject.toml
  oxq_components/
    __init__.py
    indicators/
    signals/
    rules/
    portfolio/
  tests/
component_manifest.json
component_catalog.json
result.json
```

Do not write outside this workspace-local extension root unless the coordinator
explicitly provides another task-local root.

## Workflow

1. Read `component_request.json`.
2. Re-check the current registry and recipe catalog. Use:

   ```bash
   uv run oxq registry export --out component_catalog.json
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
11. Write `component_manifest.json` without `bundle_hash`, compute it with:

    ```bash
    uv run oxq component-manifest hash component_manifest.json
    ```

12. Update `component_manifest.json` with the returned `bundle_hash`.
13. Validate importability and hash:

    ```bash
    uv run oxq component-manifest validate component_manifest.json
    ```

14. Refresh the catalog with:

    ```bash
    uv run oxq registry export \
      --component-manifest component_manifest.json \
      --out component_catalog.json
    ```

15. Write `result.json`.

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

Write `component_manifest.json` with:

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

Write `result.json`:

```json
{
  "schema_version": 1,
  "role": "component_author",
  "phase": "component_authoring",
  "status": "component_ready",
  "component_kind": "Signal",
  "component_name": "RiskAdjustedTiming",
  "artifacts": {
    "component_manifest": "component_manifest.json",
    "component_catalog": "component_catalog.json",
    "source_root": "custom_components/",
    "tests": [
      "custom_components/tests/test_risk_adjusted_timing.py"
    ]
  },
  "hashes": {
    "component_bundle_hash": "sha256:...",
    "component_catalog_hash": "sha256:..."
  },
  "tests": [
    {
      "command": "pytest custom_components/tests/test_risk_adjusted_timing.py -q",
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
