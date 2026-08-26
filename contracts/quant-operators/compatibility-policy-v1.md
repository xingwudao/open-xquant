# Quant Operator Compatibility Policy v1

## 1. Ownership

OpenXQuant owns the normative schemas and certification results.
Providers own their releases and manifests.

## 2. Binding

Each enabled binding MUST pin all of the following:

- distribution version;
- full algorithm-prefixed source commit;
- source-tree digest;
- manifest digest;
- implementation digest;
- contract schema release, schema `$id`, and exact schema-file digest.

The manifest digest is SHA-256 of the exact UTF-8 manifest file bytes and
MUST be stored in the external binding/certification record. It MUST NOT be
stored in the manifest body. All digest scopes and encodings are normative in
`hash-profile-v1.md`.

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
`reference_validator_v1.py` semantic layer.

## 7. Change policy

This section governs only the contract/schema vocabulary and its meanings. It
does not govern provider formula, default, or output semantics.

Compatibility within v1 is validator-backward compatibility: every instance
accepted by an older v1 schema release MUST continue to be accepted by each
newer v1 schema release. A v1 release MAY add an optional field, but doing so
MUST create a new schema release and exact schema digest. An older consumer
MAY reject an instance that uses the new optional field; therefore every
binding MUST pin the schema release and digest that its consumer actually
supports before enablement.

Adding a required field, rejecting an instance that was legal under an older
v1 schema release, or changing the meaning of an existing contract field is a
breaking contract change and MUST create v2.

Provider formula, default, or output changes continue to follow section 6.3
of `operator-contract-v1.md`: they require an operator/package major version
and recertification. Such a provider semantic change does not automatically
create Contract v2.

## 8. Schema vendoring

Provider schema and reference-validator copies MUST record their exact release
and SHA-256 digests and MUST NOT be edited locally. A provider MUST upgrade the
vendored pair explicitly; matching only the major contract number is not
sufficient evidence of compatibility.
