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
  compat/
    open_xquant/
      operator_catalog.json
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

`compat/open_xquant/operator_catalog.json` is the only submission entry point
and keys every candidate as
`<operator_id>@<operator_version>` and references its standalone manifest and
baseline file. Its `build_record` and every `manifest` and `baseline` value are
normalized paths relative to `compat/open_xquant/`; they cannot escape that
directory or resolve through links. Manifest `implementation.source_files`
remain relative to the implementation source commit's repository root.
`candidate-build-v1.json` records every wheel filename, role, version, build
identifier, and SHA-256 digest. Exactly one implementation artifact serves
each operator; any additional imported provider distribution must be declared
as a `runtime-dependency` artifact.

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
commit. It does not accept a GitHub URL. Certifier-owned code does not clone,
fetch, download, install, build, or proactively retrieve artifacts from the
network.

## Trust and execution boundary

Certification imports and executes the exact verified provider wheels in a
child Python process. `--trust-provider-code` is mandatory and must be supplied
before repository loading begins. The child process provides import, failure,
and timeout isolation. It is neither an operating-system nor a network
sandbox and does not make malicious third-party code safe. Trusted provider
code can still access local files and the network.

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

## Portable certification bundles

Export a targeted v2 certification with its manifest and baseline evidence:

```bash
oxq operator export-certification \
  --provider equant-py \
  --release 1.0.0 \
  --registry-dir .open-xquant/certifications \
  --manifest-dir /path/to/equant-py/compat/open_xquant/manifests \
  --baseline-file /path/to/equant-py/compat/open_xquant/numerical_baselines/technical-v1.json \
  --target cp312-cp312-macosx_14_0_arm64 \
  --output /path/to/certification.zip \
  --json
```

`--baseline-file` may be repeated. The output ZIP must be outside the source
registry. Import it only after explicitly acknowledging the local trust
decision:

```bash
oxq operator import-certification \
  --bundle /path/to/certification.zip \
  --output-dir .open-xquant/certifications \
  --trust-unsigned-bundle \
  --bundle-store .open-xquant/certification-bundles \
  --json
```

The import validates the entire ZIP before atomically publishing only its
certification publication. When supplied, `--bundle-store` receives the
original ZIP after registry publication succeeds. A plain bundle import does
not create an installed runtime or make an operator invocable.
