---
name: audit-runtime-semantics
description: >-
  Compile an open-xquant strategy spec preview and audit that material SPEC
  execution semantics are preserved in compiled_plan.json before backtests.
---

# Runtime Auditor

Use this skill after a confirmed `spec_audit.json` from `audit-strategy-spec`
and before any formal `oxq backtest run`. Its job is deterministic artifact
consistency:
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
- Optional prior `versions/<version_id>/07_compile_preview/compiled_plan.json`

## Version-Governed Runtime Gate

Before compiling, read `current.json` and use `active_version` as
`version_id`. If no active version exists, block and return to
`manage-strategy-version`.

Read the confirmed SPEC package from:

```text
versions/<version_id>/04_spec_build/strategy_spec.yaml
versions/<version_id>/06_spec_audit/spec_audit.json
versions/<version_id>/04_spec_build/component_catalog.json
```

Write compile preview artifacts only under:

```text
versions/<version_id>/07_compile_preview/compiled_plan.json
versions/<version_id>/07_compile_preview/strategy.py
versions/<version_id>/07_compile_preview/spec_hash.txt
```

Write the runtime audit only to:

```text
versions/<version_id>/08_runtime_audit/runtime_audit.json
```

Do not write root-level `compile_preview/`, `compiled_plan.json`, or
`runtime_audit.json`.

## Confirmed Spec Audit Gate

Before compiling, read `spec_audit.json` and block unless all of these are
true:

- `status: pass`
- `schema_version: 4`
- `spec_provenance_pass: true`
- `audit_conclusion: all_pass`
- `user_confirmation_status: confirmed`
- `confirmation_event` exists, points to
  `conversations/<conversation_id>/confirmations.jsonl`, records the full
  SPEC table path/hash, and records `spec_audit_path` plus `spec_audit_hash`
  for the pre-confirmation `spec_audit.json`. The JSONL line must match the
  audit reference for `path`, `event_id`, `line_number`, `event_hash`,
	  `artifact_path`, `artifact_hash`, `spec_audit_path`, and
	  `spec_audit_hash`.

Before compiling, run deterministic validation and block unless it exits 0:

```bash
<resolved_runner> spec-audit validate versions/<version_id>/06_spec_audit/spec_audit.json \
  --spec versions/<version_id>/04_spec_build/strategy_spec.yaml \
  --strict-confirmed \
  --json
```

An `audit_conclusion: all_pass` audit with
`user_confirmation_status: pending` is still blocked. Do not treat it as
authorization for runtime audit, compile preview, or backtest work.

## Runner Resolution

In a research workspace, do not assume `uv run oxq` is installed locally.
Before running deterministic validation or compile commands:

1. Read `~/.config/open-xquant/agent.yaml`.
2. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner` in place of `uv run oxq` or bare `oxq`.
3. If it is missing or fails, read `~/.config/open-xquant/agent-install.json`,
   take `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`, and use that
   cached runner.

Keep the shell in the user's research directory. Do not search unrelated home
directories for another open-xquant checkout.

## Compile Preview

If no current compile preview exists for the current `spec_hash`, run:

```bash
<resolved_runner> strategy compile versions/<version_id>/04_spec_build/strategy_spec.yaml \
  --data-dir data \
  --component-manifest components/bundles/<bundle_id>/component_manifest.json \
  --out versions/<version_id>/07_compile_preview
```

Use the same `data_dir` and every `component_manifest` path that the formal
`oxq backtest run` will use. Omit `--data-dir` only when the formal run will
also omit it. Omit `--component-manifest` when no workspace-local custom
components are authorized; repeat it for each authorized bundle manifest. The
preview `compiled_plan.json` includes the resolved effective `data_dir`, so a
preview made with different run inputs is not a valid runtime gate.
When component manifests are used, record their authorized `bundle_hash` values
in `runtime_audit.json` as `component_bundle_hashes`.

Read `versions/<version_id>/07_compile_preview/compiled_plan.json` and
`versions/<version_id>/07_compile_preview/spec_hash.txt`. The `spec_hash` in
the compile preview must match `strategy_spec.yaml` and `spec_audit.json`.

Read the compiled plan's `open_xquant_version` and compare it with
`strategy_spec.yaml` field `required_oxq_version`. When a prior or preview
runtime artifact includes `environment.json`, use it as additional evidence for
the same version comparison. A compile preview produced by a different
OpenXQuant package version than the SPEC requires is not a valid runtime gate.

The compile preview must also include
`versions/<version_id>/07_compile_preview/strategy.py`. After the
user has confirmed the SPEC table and after this preview is generated, print the complete `strategy.py` source
to the user in a fenced `python` block before summarizing runtime audit
findings. Do not print only excerpts.

## Material Runtime Fields

Compare material SPEC fields against `compiled_plan.json`. At minimum audit:

- rebalance interval and runtime rebalance source
- calendar rebalance frequency and schedule
- execution timing and fill price mode
- fee, side-specific fees, sell-side tax, minimum fee, and slippage
- universe type, index metadata, point-in-time policy, and constituent symbols
- data filters, `suspension_policy`, and required filter columns
- indicator component classes, params, and `lag_bars`
- cross-sectional indicator preservation, including `RPS` when present
- portfolio optimizer params that affect allocation, including
  `TopNRanking.pre_filter_signal`, `weighting`, and `ascending`
- validation train/test periods and `required_oos`
- rule components that affect orders or positions
- portfolio constraints that affect target weights
- initial cash, cash return, benchmark, and metrics profile when represented in
  runtime artifacts
- `required_oxq_version` against the OpenXQuant `open_xquant_version` recorded
  by compile/runtime artifacts such as `compiled_plan.json` and
  `environment.json` when available

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

Hash fields must use the same semantic/canonical hash boundaries as the
formal backtest gate:

- `spec_hash`: `StrategySpec.from_yaml(...).compute_hash()` or the `Spec Hash`
  printed by `oxq spec validate`. Do not use raw `shasum strategy_spec.yaml`.
- `spec_audit_hash`: canonical JSON hash of `spec_audit.json`, matching
  OpenXQuant's `_hash_json_file` helper. Do not use raw file bytes.
- `compiled_plan_hash`: canonical JSON hash of the compiled plan payload that
  will be used by the formal run, matching OpenXQuant's `_hash_json_payload`
  or `_hash_json_file` helper. Do not use raw file bytes.

When computing hashes from Python, use the installed OpenXQuant helpers exactly:

```python
from pathlib import Path

from oxq.spec.schema import StrategySpec
from oxq.spec.compiler import _hash_json_file

spec_path = Path("versions/<version_id>/04_spec_build/strategy_spec.yaml")
spec_audit_path = Path("versions/<version_id>/06_spec_audit/spec_audit.json")
compiled_plan_path = Path("versions/<version_id>/07_compile_preview/compiled_plan.json")

spec_hash = StrategySpec.from_yaml(spec_path).compute_hash()
spec_audit_hash = _hash_json_file(spec_audit_path)
compiled_plan_hash = _hash_json_file(compiled_plan_path)
```

Do not import non-existent helpers from `oxq.core.hashing`.

```json
{
  "schema_version": 1,
  "status": "pass | block | fail",
  "runtime_semantics_pass": true,
  "strategy_source_printed": true,
  "spec_hash": "sha256:<StrategySpec.compute_hash()>",
  "spec_audit_hash": "sha256:<canonical spec_audit JSON hash>",
  "compiled_plan_hash": "sha256:<canonical compiled_plan JSON hash>",
  "component_bundle_hashes": ["sha256:<hash>"],
  "compiled_plan_path": "versions/<version_id>/07_compile_preview/compiled_plan.json",
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
<resolved_runner> runtime-audit validate versions/<version_id>/08_runtime_audit/runtime_audit.json
```

Schema validation only proves artifact shape. The comparison still belongs to
this skill.

## Output

Report a compact summary:

- compile preview path
- path to `versions/<version_id>/07_compile_preview/strategy.py`
- confirmation that the complete `strategy.py` source was printed to the user
- `spec_hash`
- material fields preserved
- material fields missing or mismatched
- path to `versions/<version_id>/08_runtime_audit/runtime_audit.json`
- whether formal backtest remains blocked

Do not run a formal backtest while `runtime_audit.json` is missing, blocked,
failed, or mismatched.
