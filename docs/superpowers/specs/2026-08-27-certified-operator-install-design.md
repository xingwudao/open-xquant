# Certified Operator One-Step Install Design

## Status

Approved in conversation on 2026-08-27 for implementation planning.

This document defines the first public distribution and research-runtime path
for certified `equant-py` operators. It is the shared design authority for
changes in both `open-xquant` and `equant-py`.

## Goal

Make the following command download, verify, install, register, and smoke-test
the certified `equant-py` 1.0.0 operator release without modifying the Python
environment that runs `oxq`:

```bash
oxq operator install equant-py==1.0.0
```

After installation, open-xquant research code can resolve and invoke all 60
`research-certified` operators through a managed provider runtime.

## User-facing scope

The first release provides:

- deterministic certification bundle export and import;
- a public `equant-py` GitHub Release containing the certified assets;
- exact-version, one-step installation from the official release;
- a managed, exact-wheel provider runtime;
- installed operator listing;
- a Python research-runtime invocation API;
- an end-to-end installation and invocation smoke test for all 60 operators.

The first published target is exactly:

- CPython 3.12;
- macOS 14 or newer on arm64;
- `equant-py` 1.0.0;
- `research-certified` state.

Other platforms must fail with a stable `operator_target_unavailable` error.
They must not fall back to an uncertified dependency resolution.

## Non-goals

This version does not provide:

- cryptographic publisher signatures or a signed remote registry;
- PyPI publication;
- arbitrary third-party provider repositories;
- version ranges, automatic upgrades, or `latest` resolution;
- Windows, Linux, CPython 3.13, or non-arm64 macOS targets;
- `runtime-certified`, backtest-certified, broker, or live-trading authority;
- in-process imports of provider code;
- high-throughput persistent workers or Arrow IPC;
- automatic integration into strategy compilation or live execution.

`research-certified` remains limited to research and offline analysis.

## Existing constraints

open-xquant currently publishes local certification records containing a
record, registry entry, and bindings. The publication is structurally strict
and digest-linked, but it does not contain the original operator manifests or
wheel bytes. Production code does not yet consume `CertificationRegistry`
bindings to execute operators.

The installed open-xquant agent runner is a cached SDK environment. Its package
versions can differ from a provider's certified dependency closure. Installing
provider wheels into that environment would mutate the SDK bundle and can
create dependency conflicts. Provider code must therefore remain isolated.

The existing `equant-py` certification fixes these key versions:

- `equant-core` 1.0.0;
- `equant-ttr` 1.0.0;
- `numpy` 2.2.6;
- `pandas` 2.2.3;
- `numba` 0.61.2;
- `llvmlite` 0.44.0.

## Repository responsibilities

### `equant-py`

`equant-py` owns provider source, operator manifests, first-party wheels, the
provider release index, and the public GitHub Release.

The 1.0.0 Release publishes only `equant-core` and `equant-ttr` as certified
first-party distributions. The other workspace distributions are not part of
this certified release.

### `open-xquant`

`open-xquant` owns certification bundle schemas, bundle validation, the
official-provider allowlist, remote release discovery, managed installation,
installed registry lookup, runtime isolation, invocation, and the user-facing
CLI.

## Trust model

Phase one is unsigned. It has two distinct trust layers:

1. The built-in official provider map binds `equant-py` to the exact GitHub
   repository `xingwudao/equant-py`.
2. HTTPS and the GitHub Release source establish transport and account-source
   trust. The release index then binds every downloaded asset by SHA-256.

The official one-step command does not require
`--trust-unsigned-bundle`, because the source is the built-in provider map.
Install output and records must still describe the trust state as
`github-source-trusted`, never `signed`.

Manual bundle import from an arbitrary local path remains possible, but it
requires an explicit `--trust-unsigned-bundle` acknowledgement.

Redirects during official downloads are checked at every hop, remain HTTPS,
and are limited to `api.github.com`, `github.com`,
`release-assets.githubusercontent.com`, `objects.githubusercontent.com`, and
`files.pythonhosted.org`. Credentials and URL query strings must never be
written to logs or install records.

## Official provider map

open-xquant packages a canonical JSON map with this initial entry:

```json
{
  "schema_version": 1,
  "providers": {
    "equant-py": {
      "repository": "xingwudao/equant-py",
      "release_asset": "operator-release-v1.json"
    }
  }
}
```

The installer accepts only the frozen exact requirement grammar
`<provider>==<semver>`, where provider releases use the existing open-xquant
SemVer identity rules. Version ranges, PEP 440 epochs or local versions, and
implicit versions are rejected. Third-party Python distribution versions in
the artifact closure continue to use PEP 440.

## Provider release index

The canonical `operator-release-v1.json` is the bootstrap document downloaded
from the exact GitHub tag `v<release>`. It includes:

- `schema_version` and release type;
- provider and release identity;
- submission and source commits;
- certification state;
- one or more explicit runtime targets;
- certification bundle filename, HTTPS URL, `size_bytes`, and SHA-256;
- every wheel's filename, distribution, version, role, wheel tags, HTTPS
  download URL, `size_bytes`, and SHA-256;
- the expected operator count.

The initial target declares:

```text
python_tag: cp312
abi_tag: cp312
platform_tag: macosx_14_0_arm64
```

Target selection uses `packaging.tags.sys_tags()`. There must be exactly one
compatible target, and every wheel tag in that target must be compatible with
the running interpreter. Zero matches is unsupported; more than one match is
an invalid release index.

The index lists the complete exact wheel closure needed to create the provider
runtime. First-party wheels are GitHub Release assets. Third-party wheels may
use exact PyPI file URLs, but their filenames and SHA-256 values must match the
certification record exactly.

The release index, bundle manifest, certification record v2, installed commit
marker, and child protocol each have a packaged JSON Schema. Schemas set
`additionalProperties: false`, define exact digest and identity grammars, and
set numeric upper bounds for `size_bytes`, collection counts, and protocol
payloads. CLI `--json` success and error envelopes and exit codes are covered by
stable tests.

## Certification bundle

The certification bundle is a deterministic ZIP named:

```text
equant-py-1.0.0-macos-arm64-py312.open-xquant-certification.zip
```

It contains:

```text
bundle-manifest.json
publication/
  certification-record.json
  registry-entry.json
  bindings/
    <operator-id>@<version>.binding.json
manifests/
  <operator-id>@<version>.operator.json
baselines/
  <baseline-set>.json
```

`bundle-manifest.json` binds:

- schema version and bundle type;
- provider and release;
- target tags;
- publication path and registry-entry digest;
- every original manifest path and SHA-256;
- every numerical baseline set path and SHA-256;
- operator count.

The existing v1 registry publication validator remains supported for local
records. Public install requires a newly issued certification record v2. The
v2 record adds explicit target tags plus the raw SHA-256 of every manifest and
baseline set and a defined digest for every certified baseline case. A
baseline-set digest hashes the original file bytes. A case digest hashes the
open-xquant canonical JSON encoding of the parsed individual case object and
is keyed by baseline path, JSON array index, and `case_id`; it never depends on
an ambiguous raw byte slice. The real 60-operator certification must be rerun
to issue this v2 record; the existing v1 record cannot be upgraded by copying
or editing JSON.

Bundle validation proves that every binding has exactly one manifest and at
least one certified numerical case, every raw manifest and baseline digest
matches the v2 record, and all operator, version, distribution,
implementation, target, and case identities agree.

ZIP creation is deterministic: generated bundle metadata uses canonical JSON;
source manifests and baseline sets are preserved byte-for-byte; members use
lexical order, fixed timestamps and permissions, no extra fields, and deflate
with a fixed level. Preserved source JSON is still parsed with duplicate-key
and non-standard-number rejection and validated against its schema.

## Bundle CLI

Export:

```bash
oxq operator export-certification \
  --provider equant-py \
  --release 1.0.0 \
  --registry-dir .open-xquant/certifications \
  --manifest-dir /path/to/equant-py/compat/open_xquant/manifests \
  --baseline-file /path/to/equant-py/compat/open_xquant/numerical_baselines/technical-v1.json \
  --target cp312-cp312-macosx_14_0_arm64 \
  --output /path/to/bundle.zip
```

Import:

```bash
oxq operator import-certification \
  --bundle /path/to/bundle.zip \
  --output-dir .open-xquant/certifications \
  --trust-unsigned-bundle
```

Both commands support `--json` with stable status and error codes.

Export output must be outside the source registry. Import input must be
outside the destination registry. Import validates into same-filesystem
staging, atomically publishes only the v1/v2 certification publication into
the destination registry, and writes the full bundle to a separate audit
bundle store when `--bundle-store` is supplied. Plain import does not create an
installed runtime and cannot make an operator invocable. Offline installation
uses:

```bash
oxq operator install \
  --release-index <local-path> \
  --artifact-dir <directory> \
  --trust-unsigned-release
```

Every offline asset is resolved strictly as a direct child of `artifact-dir`
using the filename declared by the index; remote URLs are not read and relative
paths are not accepted. The installed trust state is
`local-unsigned-user-trusted`, never `github-source-trusted`. Reimporting
byte-identical content is idempotent. A different
publication for an existing provider and release is a
`certification_conflict`.

The ZIP reader must not use `extractall`. It rejects absolute paths, `..`,
backslashes, NUL bytes, duplicate members, members outside the declared roots,
symlinks, special files, encrypted entries, unsupported compression, excessive
member counts, excessive per-file or total expanded size, and excessive
compression ratios.

## Managed installation store

The default store is:

```text
~/.config/open-xquant/operator-releases/
```

Tests and managed deployments can override it with:

```text
OPEN_XQUANT_OPERATOR_HOME
```

Each installed target has this immutable layout:

```text
<provider>/<release>/<target>/
  operator-release-v1.json
  certification-bundle.zip
  certification/
    publication/
    manifests/
    baselines/
  wheels/
  installed-release-v1.json
```

`installed-release-v1.json` is the commit marker. A directory without a valid
commit marker is never visible to lookup or invocation.

The commit marker is canonical JSON and binds the raw SHA-256 and size of the
stored release index, original bundle ZIP, every expanded publication,
manifest, baseline, and wheel file, plus a canonical tree digest over each
expanded directory's ordered relative-path/digest/size entries. It also records the interpreter implementation and version, operator
runtime protocol digest, and minimum and maximum compatible open-xquant
versions. Lookup revalidates the complete immutable file set before use and
rejects an installation whose protocol range does not include the running
open-xquant version.

Installation uses a provider/release/target process lock. Downloads,
validation, and all 60 canonical smoke cases run from a sibling staging
directory. The commit marker is written into staging only after every check
passes, then the complete immutable release directory is atomically renamed
into place. On failure, staging is removed and pre-existing valid
installations are never modified.

An identical installed release is idempotent. Different bytes under the same
provider/release/target produce `operator_install_conflict`.

## One-step install flow

`oxq operator install equant-py==1.0.0` performs:

1. Parse the exact provider requirement.
2. Resolve the provider through the built-in official provider map.
3. Fetch the exact GitHub Release and `operator-release-v1.json`.
4. Validate canonical JSON, schema, identity, HTTPS URLs, and target selection.
5. Download the certification bundle and exact wheel closure to staging.
6. Verify every filename, wheel tag, size, and SHA-256.
7. Validate the bundle, publication, all 60 manifests, and all 60 bindings.
8. Confirm release-index artifacts equal certification-record artifacts.
9. Use the new exact-closure child to extract the verified wheel closure into a
   fresh temporary runtime with no ambient site packages.
10. Smoke-import every manifest module and callable in the contained child.
11. Run one v2-bound canonical numerical case for each of the 60 operators.
12. Write the installed commit marker and atomically publish the immutable
    release directory.
13. Report provider, release, target, trust state, certification state, and
    operator count.

The installer never imports provider code in the `oxq` process and never runs
pip. Installation means storing an exact verified wheel closure, not mutating
the open-xquant SDK or any user Python environment.

Every wheel is treated as an untrusted ZIP before execution. Validation applies
member-count, per-member, expanded-total, compression-ratio, duplicate-name,
path, encryption, symlink, and special-file bounds across each wheel and the
complete closure. It also verifies `.dist-info/METADATA`, `WHEEL`, and `RECORD`:
distribution name, version, dependency requirements, wheel tags, archive paths,
file hashes, and sizes must agree with the release index and certification
record. Undeclared files or dependencies fail closed.

## Installed operator listing

`oxq operator list` scans only committed installed releases and prints:

- provider and release;
- target;
- trust state;
- certification state;
- operator count;
- installation path.

`--json` returns a stable array. Corrupt or partial installations produce a
structured error rather than being silently skipped. Each JSON release entry
contains the complete ordered operator identity list, so callers can verify all
60 identities while human output remains a compact release summary with
`operator_count: 60`.

## Research runtime

The public Python entry point is:

```python
from oxq.operators.runtime import CertifiedOperatorRuntime

runtime = CertifiedOperatorRuntime()
result = runtime.invoke(
    "equant.ttr.sma",
    operator_version="1.0.0",
    provider_release="equant-py==1.0.0",
    panel=quant_panel_literal,
    parameters={"n": 5},
)
```

`panel` is a `Mapping[str, object]` conforming exactly to QuantPanel v1,
including records, columns, primary key, and context. The return type is a
pandas `DataFrame` whose primary-key order and result columns have been
validated against the manifest. `provider_release` makes resolution
unambiguous when multiple provider releases contain the same operator and
operator version.

The 1.0.0 manifests must be corrected before v2 certification so every
parameter that changes output fields has `affects_output_fields: true` and a
truthful default output template. Phase one nevertheless accepts only a
parameter mapping that byte-for-byte matches the canonical parameter object of
a v2-bound certified case. More cases can expand the allowed parameter set;
schema-valid but uncertified parameter combinations fail with
`operator_invocation_invalid`. The example uses the currently certified `n=5`
case. Arbitrary schema-valid parameters are a later certification capability.

The parent process:

1. resolves a unique installed binding and manifest;
2. opens stored assets without following links, snapshots and verifies the
   exact bytes and installed commit marker, and executes only those snapshots;
3. validates operator version, input QuantPanel, and parameter names/types;
4. serializes the request using the existing QuantPanel literal model;
5. starts the current compatible CPython interpreter with the open-xquant
   exact-wheel research child entry script;
6. enforces a configurable timeout;
7. validates and reconstructs the returned aligned result.

For every invocation, the parent starts the child as
`sys.executable -I -S <child-script>`. The child uses only the standard library
until it has verified and safely extracted every wheel snapshot, constructed a
restricted `sys.path`, and hidden ambient modules. Only then may it first import
the certified NumPy, pandas, Numba, dependencies, and provider callable. Every
third-party module origin must be inside the extracted exact closure. The child
then reconstructs the pandas frame, invokes the callable, and writes an
authenticated response file. NaN values use JSON `null`; dates use ISO 8601;
row order, asset identity, and output columns are explicit.

The v1 child protocol reuses and extracts the existing baseline runner's
process containment, HMAC-authenticated request/response files, descriptor
handling, output validation, timeout, and descendant termination. Child stdout
and stderr go to `DEVNULL`; provider writes cannot corrupt the protocol. A
request contains protocol version, operator identity, module, callable,
validated parameters, QuantPanel records, columns, primary key, and context. A
success response contains protocol version, status, ordered output fields,
ordered primary-key values, and output values. An error response contains only
a stable error category and sanitized message. Parent and child both enforce
bounded request and response sizes. Generated protocol JSON is canonical. The
protocol schema is packaged with open-xquant and its digest is recorded in the
installed commit marker.

This requires refactoring the current baseline child: its present startup
imports SDK NumPy and pandas before wheel extraction and is not suitable for
installed execution. Certification and installed invocation must share the new
stdlib-first exact-closure bootstrap after the refactor.

The first implementation starts one fresh extraction and subprocess per
invocation. Persistent extracted runtimes, workers, and Arrow IPC are deferred
until correctness and trust boundaries are stable.

The managed exact-wheel runtime provides dependency and integrity isolation,
not an OS security sandbox. Phase one trusts official provider code and does
not claim filesystem or network confinement beyond the process controls
already enforced by the baseline runner.

The runtime supports only `research-certified` offline invocation. It does not
grant strategy compilation, backtest, broker, or live-trading authority.

## Registry precedence

Only committed releases in the managed installed store are invocable. A plain
workspace-local `.open-xquant/certifications` registry provides certification
lookup and audit only; it cannot add an invocable identity because it lacks the
complete wheel and runtime evidence closure. Offline invocation first requires
`operator install --release-index <local-path>`.

## Release construction

The `equant-py` 1.0.0 GitHub tag does not exist yet. It will point to the exact
final submission commit whose corrected manifests, baselines, release index
inputs, and release tooling are certified. The implementation source commit
remains `84b574582e77e66bcf8c1b6954a45bfa19669a4d`, and the first-party wheel
bytes remain unchanged.

The final submission commit never contains the generated v2 certification
record, certification bundle, or generated release index. Those files are
external Release assets derived from and naming the already frozen submission
commit. They are not committed back to that branch, which avoids recursively
changing `submission_commit` after certification.

The first Release is constructed after the open-xquant install implementation
is merged and an install-capable open-xquant package or cached SDK bundle is
available. It publishes:

- `equant_core-1.0.0-py3-none-any.whl`;
- `equant_ttr-1.0.0-py3-none-any.whl`;
- the platform-specific certification bundle;
- `operator-release-v1.json`;
- `SHA256SUMS`;
- candidate build and toolchain records;
- the operator catalog.

The certified first-party wheel digests must remain:

```text
equant-core: sha256:da6b9135e9dc6adea9d82b843662db150838530191fce7d659d7b833e1c6a30a
equant-ttr:  sha256:087f5f3b9ce1ff4071d31cf80e6b6015e64c677b330f31e1049dec26ba2ce8e5
```

Files that affect wheel metadata or bytes must not change before the Release.
Release tooling, corrected external manifests, baseline inputs, and
documentation live outside packaged wheel metadata inputs. The final tagged
submission is recertified as v2 exactly once before public publication. The v2
record retains its real `certified_at` time, so a later issuance is not expected
to reproduce the same record or bundle bytes. The validated release tool
freezes the generated external assets from that issuance. The workflow on the
tagged commit verifies those frozen asset bytes, independently reruns all 60
bound cases against them, and publishes or approves them; it does not issue a
second certification record.

## Error model

All new library errors use structured stable codes. At minimum:

- `operator_requirement_invalid`;
- `operator_provider_unknown`;
- `operator_release_unavailable`;
- `operator_release_invalid`;
- `operator_target_unavailable`;
- `operator_download_failed`;
- `operator_artifact_invalid`;
- `operator_bundle_invalid`;
- `operator_install_conflict`;
- `operator_install_failed`;
- `operator_not_installed`;
- `operator_runtime_invalid`;
- `operator_invocation_invalid`;
- `operator_invocation_failed`;
- `operator_invocation_timeout`.

Human output must not contain credentials, query strings, temporary paths, or
provider-controlled tracebacks. `--json` includes code, stage, message,
provider, release, and operator identity when known.

## Security requirements

- No provider import or pip mutation occurs in the control process.
- Every remote URL is HTTPS and subject to the official-host redirect policy.
- Every downloaded asset is size-limited and digest-verified before use.
- ZIP processing follows the bundle constraints above.
- Generated index, bundle metadata, records, markers, and protocol JSON is
  canonical. Preserved provider manifests and baselines remain byte-identical
  while still receiving strict duplicate-key, number, and schema validation.
- Filesystem ancestors are checked for symlinks before sensitive writes.
- Writes use restrictive permissions and same-filesystem atomic publication.
- Install operations are process-locked and crash-recoverable.
- Subprocess environments are allowlisted and exclude ambient `PYTHONPATH`,
  credentials, and proxy secrets unless explicitly needed for the download
  control plane.
- Runtime invocation has a timeout and bounded input/output sizes.
- Provider stdout, stderr, and exceptions cannot corrupt the JSON protocol.
- Partial corruption, link replacement, and check/use races are in scope:
  verified snapshot bytes, rather than reopened paths, are supplied to the
  child. Coordinated malicious same-user replacement of both assets and their
  unsigned commit marker is out of scope for phase one; phase-two signatures
  provide the external trust anchor needed to detect it.

## Testing strategy

open-xquant tests follow TDD and cover:

- release-index schema and exact requirement parsing;
- target selection and unsupported-platform behavior;
- official GitHub discovery with mocked HTTP responses;
- redirect, URL, size, digest, and download failures;
- deterministic bundle byte equality;
- bundle round-trip export/import;
- malicious ZIP cases and decompression limits;
- manifest/binding/publication cross-checks;
- install idempotence, conflicts, locks, interrupted staging, and rollback;
- proof that install never invokes pip or mutates a Python environment;
- proof that the cached SDK environment remains unchanged;
- partial store corruption, link replacement, and snapshot/use race rejection;
- wheel ZIP limits plus METADATA, WHEEL, RECORD, and closure validation;
- partial and corrupt install rejection;
- `operator list` human and JSON output;
- runtime request validation, timeout, protocol corruption, and provider errors;
- invocation of representative Series, single-frame, and multi-output operators;
- an opt-in end-to-end test that installs the actual 1.0.0 assets and runs all
  60 canonical numerical cases;
- installed open-xquant wheel smoke coverage for packaged schemas and CLI.

`equant-py` tests cover:

- exact certified wheel filenames and SHA-256 values;
- release-index generation from the frozen candidate build;
- inclusion and digest matching of all 60 manifests;
- complete artifact closure matching the certification record;
- `SHA256SUMS` generation;
- GitHub Release workflow validation;
- no publication of uncertified workspace distributions.

## Documentation

User documentation must distinguish:

- wheel code from certification metadata;
- `github-source-trusted` from cryptographically signed;
- `research-certified` from runtime/backtest/live authority;
- supported from unsupported runtime targets;
- one-step official install from explicit local bundle import.

The primary quick start uses only:

```bash
oxq operator install equant-py==1.0.0
oxq operator list
```

Manual export/import is documented as an audit, development, and offline path.

## Delivery sequence

1. Implement bundle schema, deterministic export/import, and security tests in
   `open-xquant`.
2. Implement release-index parsing, official discovery, target selection, and
   managed transactional installation in `open-xquant`.
3. Implement installed lookup, listing, child protocol, and research runtime in
   `open-xquant`.
4. Correct dynamic-output manifest metadata, implement Release asset
   validation and workflow in `equant-py`, and keep certified wheel bytes
   unchanged.
5. Generate provisional v2 fixtures, bundle, and release index for development.
6. Use the public offline interface with a local release index and artifact
   directory to run a clean-store one-step install and all-operator canonical
   case test before publication.
7. Push `equant-py`, open the `open-xquant` PR, and complete review.
8. After merge, publish an install-capable open-xquant package or cached SDK
   bundle.
9. Using that final published open-xquant version, rerun v2 certification
   against the final `equant-py` submission commit, rebuild the bundle and
   release index, and repeat the complete offline acceptance test.
10. Create the exact certified `equant-py` 1.0.0 tag and public GitHub Release.
11. In a fresh supported environment with no resolver override, run the exact
    public command and all-operator invocation test from public URLs.

## Acceptance criteria

The work is accepted only when:

- a clean supported machine can run exactly
  `oxq operator install equant-py==1.0.0`;
- the command installs no package into the `oxq` SDK environment;
- all release, bundle, manifest, binding, and wheel digests validate;
- `operator list --json` returns all 60 operator identities as
  `research-certified`, while human output reports `operator_count: 60`;
- all 60 canonical numerical cases pass through the installed child runtime;
- repeating install is idempotent;
- unsupported targets and tampered assets fail closed;
- an induced failure leaves no visible installed release;
- manual deterministic bundle export/import round-trips successfully;
- the public offline interface proves pre-publication installation from local
  assets, and the final acceptance run uses only the built-in official GitHub
  resolver;
- the public GitHub Release contains the documented assets and checksums;
- documentation makes the unsigned and research-only limitations explicit.
