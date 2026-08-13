# Quant Operator Compatibility Policy

This policy defines how open-xquant evolves the machine-readable files in this
directory while operator providers remain independently versioned.

## Versioning

- The integer `schema_version` identifies a schema generation.
- Additive optional fields may be released within the same schema generation.
- Removing a field, changing a required field, narrowing accepted values, or
  changing QuantPanel key/time semantics requires a new schema generation.
- A provider must publish the exact contract version used to build its catalog.

## Publication

- JSON Schemas are the portable validation interface for external repositories.
- Markdown is normative where a semantic requirement cannot be expressed by
  JSON Schema.
- Serialized QuantPanel fixtures are conforming only after both JSON Schema
  validation and `validate_serialized_quant_panel()` semantic validation;
  JSON Schema cannot express uniqueness of the composite `(date, code)` key.
- A provider release must pin immutable source, tree, manifest, and build
  identities. Editable installs and local source paths are not certifiable.

## Certification

- `contract-valid` means schema and baseline contract tests pass.
- `research-certified`, `runtime-certified`, and `ml-certified` are separate
  open-xquant decisions and must not be inferred from provider CI alone.
- Certification applies to an exact distribution version and digest tuple.
- A changed manifest digest invalidates prior certification until re-evaluated.

## Deprecation

- A supported schema generation receives at least one documented migration
  path before removal.
- open-xquant may read archived locks for retired schema generations, but does
  not have to allow them in new Strategy Specs.
- Security, causality, or silent-data-corruption defects may cause immediate
  revocation of certification without waiting for a normal deprecation window.
