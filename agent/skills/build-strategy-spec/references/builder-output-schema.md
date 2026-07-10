# Builder Output Schema

Use this reference before writing `builder_phase_result.json` or recording
OpenXQuant package-version provenance.

## Contents

- OpenXQuant Version Provenance
- Validation
- `builder_phase_result.json`
- Phase Output

## OpenXQuant Version Provenance

Read the package version with the resolved runner's virtualenv Python. Do not
call `oxq --version`, `oxq version`, or `oxq spec show`; those are not the
builder version source. In an installed research workspace:

```bash
RUNNER="/path/to/resolved/oxq"
PYTHON="$(dirname "$RUNNER")/python"
"$PYTHON" - <<'PY'
import oxq

print(oxq.__version__)
PY
```

`required_oxq_version` is provenance, not a strategy default chosen for the
user. Do not ask the user to invent or confirm it before writing the SPEC, but
ensure the later SPEC confirmation table shows it because a package-version
change affects reproducibility. Record the resolved version in
`builder_phase_result.json`.

## Validation

Run deterministic validation:

```bash
uv run oxq spec validate versions/<version_id>/04_spec_build/strategy_spec.yaml
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

## `builder_phase_result.json`

Write `builder_phase_result.json` after validation:

```json
{
  "status": "pass | blocked | fail",
  "strategy_spec": "versions/<version_id>/04_spec_build/strategy_spec.yaml",
  "strategy_idea_brief": "versions/<version_id>/01_brainstorm/strategy_idea_brief.json",
  "strategy_idea_audit": "versions/<version_id>/02_idea_audit/strategy_idea_audit.json",
  "strategy_idea_brief_hash": "sha256:<hash>",
  "strategy_idea_audit_hash": "sha256:<hash>",
  "required_oxq_version": "0.1.0",
  "component_catalog": "versions/<version_id>/04_spec_build/component_catalog.json",
  "catalog_hash": "<component_catalog.catalog_hash>",
  "spec_build_notes": "versions/<version_id>/04_spec_build/spec_build_notes.md",
  "spec_mapping_notes": "versions/<version_id>/04_spec_build/spec_mapping_notes.md",
  "spec_mapping_contract": "versions/<version_id>/04_spec_build/spec_mapping_contract.json",
  "validation": {
    "status": "pass | fail",
    "spec_hash": "sha256:<hash>",
    "warnings": [],
    "errors": []
  },
  "selected_components": [],
  "selected_recipes": [],
  "unmapped_source_fields": [],
  "unsupported_mappings": [
    {
      "source_field": "portfolio.constraints.min_position_value",
      "requested_semantic": "minimum notional position size",
      "reason": "current SPEC parses this field but the audited runtime cannot execute it",
      "disposition": "blocked",
      "blocking": true
    }
  ],
  "data_warmup_policy": {
    "status": "confirmed | blocked",
    "min_start_date": "",
    "reason": ""
  },
  "needs_custom_component": [],
  "next_required_phase": "audit | brainstorm | build | data_inspection | component_authoring"
}
```

## Phase Output

The builder phase output is:

- `versions/<version_id>/04_spec_build/strategy_spec.yaml`
- `versions/<version_id>/04_spec_build/component_catalog.json`
- `versions/<version_id>/04_spec_build/spec_build_notes.md` or equivalent
  build notes
- `versions/<version_id>/04_spec_build/spec_mapping_notes.md`
- `versions/<version_id>/04_spec_build/spec_mapping_contract.json`
- `versions/<version_id>/04_spec_build/builder_phase_result.json`
