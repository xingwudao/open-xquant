---
name: audit-runtime-semantics
description: >-
  Compile an open-xquant strategy spec preview and audit that material SPEC
  execution semantics are preserved in compiled_plan.json before backtests.
---

# Runtime Auditor

Use this skill after `audit-strategy-spec` passes and before any formal
`oxq backtest run`. Its job is deterministic artifact consistency:
`strategy_spec.yaml` must compile into a `compiled_plan.json` that preserves
material execution semantics.

This skill does not audit conversation provenance, ask the user to confirm
assumptions, edit `strategy_spec.yaml`, run a backtest, monitor a run, or write
a report.

## Inputs

- `strategy_spec.yaml`
- `spec_audit.json`
- `component_catalog.json` when available
- `backtest_authorization.json` or the intended formal run inputs when
  available, including `data_dir` and `component_manifest` paths
- Optional prior `compile_preview/compiled_plan.json`

## Compile Preview

If no current compile preview exists for the current `spec_hash`, run:

```bash
uv run oxq strategy compile strategy_spec.yaml \
  --data-dir data \
  --component-manifest component_manifest.json \
  --out compile_preview
```

Use the same `data_dir` and every `component_manifest` path that the formal
`oxq backtest run` will use. Omit `--data-dir` only when the formal run will
also omit it, and omit `--component-manifest` only when no workspace-local
custom components are authorized. The preview `compiled_plan.json` includes the
resolved effective `data_dir`, so a preview made with different run inputs is
not a valid runtime gate.
When component manifests are used, record their authorized `bundle_hash` values
in `runtime_audit.json` as `component_bundle_hashes`.

Read `compile_preview/compiled_plan.json` and
`compile_preview/spec_hash.txt`. The `spec_hash` in the compile preview must
match `strategy_spec.yaml` and `spec_audit.json`.

## Material Runtime Fields

Compare material SPEC fields against `compiled_plan.json`. At minimum audit:

- rebalance interval and runtime rebalance source
- execution timing and fill price mode
- fee, minimum fee, and slippage
- validation train/test periods and `required_oos`
- rule components that affect orders or positions
- initial cash, cash return, benchmark, and metrics profile when represented in
  runtime artifacts

If the compiled plan omits a supported material field, or contradicts the
SPEC, block the backtest. For example, if `strategy_spec.yaml` says
`portfolio.rules.rebalance.params.interval_days: 10` but
`compiled_plan.json` shows daily or `interval_days: 1`, this is a blocking
runtime mismatch.

If the engine cannot represent a material field in `compiled_plan.json`, block
and report that the runtime artifact is insufficient for formal execution. Do
not infer success from a missing field.

## `runtime_audit.json`

Write `runtime_audit.json` before authorizing a formal backtest:

```json
{
  "schema_version": 1,
  "status": "pass | block | fail",
  "runtime_semantics_pass": true,
  "spec_hash": "sha256:<hash>",
  "spec_audit_hash": "sha256:<hash>",
  "compiled_plan_hash": "sha256:<hash>",
  "component_bundle_hashes": ["sha256:<hash>"],
  "compiled_plan_path": "compile_preview/compiled_plan.json",
  "material_field_audits": [
    {
      "field_path": "portfolio.rules.rebalance",
      "spec_value": {
        "type": "RebalanceFrequencyRule",
        "params": {"interval_days": 10}
      },
      "runtime_path": "execution.rebalance",
      "runtime_value": {
        "frequency": "every_n_sessions",
        "interval_days": 10,
        "source": "portfolio.rules.rebalance"
      },
      "status": "preserved | missing | mismatch | not_applicable",
      "evidence": ["compiled_plan.json preserves interval_days: 10"],
      "blocking": false
    }
  ],
  "blocking_findings": [{"message": "..."}]
}
```

After writing `runtime_audit.json`, run:

```bash
uv run oxq runtime-audit validate runtime_audit.json
```

Schema validation only proves artifact shape. The comparison still belongs to
this skill.

## Output

Report a compact summary:

- compile preview path
- `spec_hash`
- material fields preserved
- material fields missing or mismatched
- path to `runtime_audit.json`
- whether formal backtest remains blocked

Do not run a formal backtest while `runtime_audit.json` is missing, blocked,
failed, or mismatched.
