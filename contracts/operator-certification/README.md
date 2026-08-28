# Operator certification profile v1

This directory defines the open-xquant-owned intake profile used to certify
an external operator provider. The frozen operator contract itself remains
under `contracts/quant-operators/`.

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
  --trust-provider-code \
  --json
```

`--artifact-dir` defaults to `<provider-repo>/dist`.

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

## Certified package result

A certified provider is distributed as normal Python packages. The canonical
provider name remains `equant-py`, while the installed Python distributions are
the certified split packages such as `equant-core` and `equant-ttr`. The
implementation wheels carry their own open-xquant certification artifacts, for
example:

```text
<provider-wheel>/
  <runtime modules>
  open_xquant/
    manifests/
      <operator-id>.operator.json
    numerical_baselines/
      <baseline>.json
```

open-xquant does not import a ZIP bundle or maintain a local certification
registry for runtime use. Users install the provider with Python packaging and
then ask open-xquant to verify the installed package:

```bash
pip install equant-core==1.0.0 equant-ttr==1.0.0
oxq operator verify equant-py==1.0.0
oxq operator list --provider equant-py
```

The resulting `research-certified` state permits research and offline analysis
only. It does not authorize strategy runtime or live trading. Those uses remain
gated on `runtime-certified` together with `past_only` causality.
