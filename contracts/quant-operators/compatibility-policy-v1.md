# Quant Operator Compatibility Policy v1

## 1. Ownership

OpenXQuant owns the normative schemas and certification results.
Providers own their releases and manifests.

## 2. Binding

Each enabled binding MUST pin the distribution version, source commit,
source-tree digest, manifest digest, and implementation digest.

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

## 7. Change policy

Only backward-compatible v1 changes are permitted. Any breaking schema or
semantic change creates v2.

## 8. Schema vendoring

Provider schema copies MUST record their exact SHA-256 digests and MUST NOT be
edited locally.
