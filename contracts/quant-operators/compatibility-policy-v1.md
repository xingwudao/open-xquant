# Quant Operator Compatibility Policy v1

## 1. Ownership

OpenXQuant owns the normative schemas and certification results.
Providers own their releases and manifests.

## 2. Binding

Each enabled binding MUST pin all of the following:

- operator identity and operator version;
- distribution version;
- full algorithm-prefixed source commit;
- source-tree digest;
- manifest digest;
- implementation digest;
- contract surface release;
- exact-file releases and digests for the QuantPanel schema,
  OperatorManifest schema, OperatorBinding schema, and
  `reference_validator_v1.py`.

The v1 contract surface release is the combined acceptance set of that exact
pinned four-artifact tuple. JSON Schema is the structural layer and
`reference_validator_v1.py` is the non-bypassable semantic layer. Passing only
one layer is not contract validity.

Before enablement, every binding MUST pass `operator-binding-v1.schema.json`
and the published `validate_operator_binding()` semantic entry point with the
exact manifest path, provider source root, formal implementation artifact, and
all four contract-surface artifact paths. A schema-valid binding is not an
enabled binding. Duplicate legacy and contract-surface pins MUST agree; a
consumer MUST NOT silently choose one of conflicting provenance fields.

The manifest digest is SHA-256 of the exact UTF-8 manifest file bytes and
MUST be stored in the external binding/certification record. It MUST NOT be
stored in the manifest body. All digest scopes and encodings are normative in
`hash-profile-v1.md`. The `manifest_path` bytes are the authoritative manifest
artifact: they MUST decode as strict UTF-8 JSON without duplicate object keys
or non-standard numeric constants, and the decoded value MUST be identical to
the manifest object used by the Schema and semantic validation layers under
recursive JSON type semantics. One binding validation MUST read the manifest
path once and use that single in-memory byte snapshot for strict decoding,
object comparison, and manifest-digest calculation.

## 3. Certification states

The only certification states are `candidate`, `contract-valid`,
`research-certified`, `runtime-certified`, `ml-certified`, and `revoked`.

## 4. Runtime gate

Only an operator that is both `runtime-certified` and `past_only` MAY enter
formal signal execution.

## 5. Engine boundary

Provider backtests, reports, apps, and workflow artifacts are never governed
OpenXQuant runs.

## 6. Release flow

The release flow is: provider release candidate, provider contract tests,
OpenXQuant certification, compatibility record, and binding enablement.
Provider contract tests and OpenXQuant certification MUST both execute the
published JSON Schema structure layer and the non-bypassable
`reference_validator_v1.py` semantic layer. Binding enablement MUST additionally
execute `validate_operator_binding()` against the certified exact artifacts.

## 7. Change policy

This section governs only the contract/schema vocabulary and its meanings. It
does not govern provider formula, default, or output semantics.

Compatibility within v1 is whole-surface backward compatibility: every object
accepted by the entire older pinned tuple of three schemas and semantic
validator MUST continue to be accepted by the entire newer pinned tuple. A v1
release MAY add an optional field, but doing so MUST create a new surface
release and exact artifact digests. An older consumer MAY reject an instance
that uses the new optional field; therefore every binding MUST pin the exact
surface release and tuple that its consumer actually supports before
enablement.

Adding a required field, narrowing acceptance in either a JSON Schema or the
semantic validator, rejecting an object accepted by the older pinned tuple, or
changing the meaning of an existing contract field is a breaking contract
change and MUST create v2.

Provider formula, default, or output changes continue to follow section 6.3
of `operator-contract-v1.md`: they require an operator/package major version
and recertification. Such a provider semantic change does not automatically
create Contract v2.

## 8. Schema vendoring

Provider copies of all three schemas and the reference validator MUST record
their exact surface release and SHA-256 digests and MUST NOT be edited locally.
A provider MUST upgrade the vendored tuple explicitly; matching only the major
contract number is not sufficient evidence of compatibility.
