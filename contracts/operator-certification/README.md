# Local operator certification profile v1

This directory defines the open-xquant-owned intake and publication profile
used to certify an external operator provider. The frozen operator contract
itself remains under `contracts/quant-operators/`.

## Provider repository layout

A submission commit contains committed metadata and an earlier committed
implementation source tree. Built wheels remain local artifacts and do not
need to be committed.

```text
<provider-repo>/
  provider-catalog-v1.json
  candidate-build-v1.json
  manifests/
    <operator-id>.operator.json
  numerical_baselines/
    <baseline>.json
  src/
    ... implementation source ...
  dist/
    ... exact implementation and runtime-dependency wheels ...
```

`provider-catalog-v1.json` keys every candidate as
`<operator_id>@<operator_version>` and references its standalone manifest and
baseline file. `candidate-build-v1.json` records every wheel filename, role,
version, build identifier, and SHA-256 digest. Exactly one implementation
artifact serves each operator; any additional imported provider distribution
must be declared as a `runtime-dependency` artifact.

## Two-commit provenance

Certification uses two exact commits:

- The implementation source commit contains the source files used for the
  declared source-tree digest and wheel build.
- The later submission commit contains the catalog, manifests, numerical
  baselines, and build record. The build record stores the implementation
  commit as `git-sha1:<40 lowercase hex characters>`.

The implementation commit must be an ancestor of the submission commit. The
CLI argument is the raw, full 40-character lowercase submission SHA. This
split avoids requiring a commit to contain its own hash.

## Manual command

Run from the open-xquant repository or another desired output workspace:

```bash
oxq operator certify-provider \
  --provider-repo ../equant-py \
  --provider-commit <full-40-character-lowercase-submission-sha> \
  --artifact-dir ../equant-py/dist \
  --output-dir .open-xquant/certifications \
  --trust-provider-code \
  --json
```

`--artifact-dir` defaults to `<provider-repo>/dist`. `--output-dir` defaults
to `<current-directory>/.open-xquant/certifications`.

The v1 command accepts only an existing local Git directory and an exact local
commit. It does not accept a GitHub URL, clone or fetch a repository, access
the network, build wheels, or install provider packages.

## Trust and execution boundary

Certification imports and executes the exact verified provider wheels in a
child Python process. `--trust-provider-code` is mandatory and must be supplied
before repository loading begins. The child process provides import, failure,
and timeout isolation. It is not an operating-system sandbox and does not make
malicious third-party code safe.

The certifier validates committed metadata, implementation source digests,
wheel digests, frozen manifests and bindings, QuantPanel inputs, invocation
parameters, input immutability, alignment, output fields, and all numerical
baseline values. Any failed operator rejects the whole provider release.

## Published result

A successful invocation atomically publishes:

```text
.open-xquant/certifications/<provider>/<release>/
  certification-record.json
  bindings/
    <operator-id>@<operator-version>.binding.json
  registry-entry.json
```

The publication records provenance and digests but does not copy provider
source or wheels. Repeating identical input is idempotent. A different
submission for an existing provider/release is rejected as a conflict and is
never allowed to overwrite the original release.

The resulting `research-certified` state permits research and offline analysis
only. It does not authorize strategy runtime or live trading. Those uses remain
gated on `runtime-certified` together with `past_only` causality.
