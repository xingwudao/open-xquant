# Local Operator Certification Design

## Purpose

`open-xquant` is the sole certification authority for external quantitative
operators. A provider such as eQuant publishes immutable candidate artifacts;
`open-xquant` independently verifies them and, only after executing their
declared numerical baselines, issues `research-certified` bindings.

This first version is manually triggered from a local `open-xquant` checkout.
It accepts a local provider Git repository and an exact 40-character commit.
It does not clone remote repositories, contact GitHub, run a backtest, or make
an operator eligible for formal/live execution.

## User Interface

The entry point is:

```bash
oxq operator certify-provider \
  --provider-repo ../equant-py \
  --provider-commit <40-character-git-commit> \
  --trust-provider-code
```

Optional arguments are:

- `--artifact-dir`, defaulting to `<provider-repo>/dist`.
- `--output-dir`, defaulting to
  `<open-xquant-worktree>/.open-xquant/certifications`.
- `--json`, for a stable machine-readable result.

`--provider-repo` accepts only a local Git working-tree path in v1. A GitHub
repository must be cloned manually first. `--provider-commit` is the provider
release/submission commit and must be a full commit object ID, not a branch,
tag, abbreviated hash, or working-tree state. The build record separately pins
the earlier implementation source commit from which the wheels were built;
that implementation commit must be an ancestor of the submission commit.
`--trust-provider-code` is mandatory because research certification executes
code from provider wheels. The child process is isolated for imports and
failure containment, but it is not an operating-system security sandbox.

## Provider Submission Layout

The selected provider commit must contain:

```text
compat/open_xquant/
├── operator_catalog.json
├── candidate-build-v1.json
├── manifests/
│   ├── equant.ttr.sma.operator.json
│   └── ...
└── numerical_baselines/
    └── technical-v1.json
```

The local artifact directory contains the immutable wheels named by the build
record, including the primary implementation wheel and any provider runtime
wheel required to execute it. For the first eQuant submission these are
`equant-core` and `equant-ttr`.

The catalog is an index, not the authoritative manifest byte stream. Each
entry contains the operator ID/version and relative paths to one standalone
manifest and one baseline file. The standalone manifest file is what the
binding's `manifest_digest` authenticates.

The build record contains:

- schema version and provider release;
- the exact implementation source commit;
- every local wheel's distribution, version, filename, role, and SHA-256;
- the Python version and exact build command used by the provider.

The baseline file contains one or more cases. Each case identifies the exact
operator ID/version, invocation parameters, a QuantPanel-compatible literal
input, literal expected output values, and absolute/relative tolerances.

The catalog, baseline, and build-record formats form an independently
versioned local certification intake profile. Their schemas live under
`contracts/operator-certification/`; they do not modify the already frozen
quant-operator contract v1 surface or its four digests.

## Architecture

The implementation adds an `oxq.operators` package with four focused units:

- `submission.py` resolves the exact Git commit into a temporary archive,
  loads the catalog/build/baseline files, enforces path containment, and
  rejects referenced symlinks.
- `certification.py` performs schema and semantic validation, constructs the
  binding, coordinates numerical execution, and returns a typed result.
- `baseline_runner.py` runs the provider wheel in a separate Python process
  with explicit wheel paths and a timeout, then compares literal outputs.
- `registry.py` atomically publishes bindings and certification records and
  provides read-only lookup by provider, operator ID, and operator version.

The frozen quant-operator schemas and validator are included as exact package
data below `oxq.operators.contracts.v1` and accessed with
`importlib.resources`. Packaging tests prove that a built `open-xquant` wheel
can certify without access to the repository-root `contracts/` directory.
`jsonschema` becomes a normal runtime dependency because certification always
performs Draft 2020-12 structural validation before semantic validation.

## Certification Flow

One CLI invocation runs two internal stages.

### Stage 1: Contract validation

1. Resolve and archive the exact provider submission commit without using
   working-tree contents.
2. Validate the catalog, build record, and baseline against the certification
   intake profile.
3. Resolve the build record's exact implementation source commit, prove it is
   an ancestor of the submission commit, and archive it as the source root.
4. Load each standalone manifest from the submission archive as strict UTF-8 JSON.
5. Validate each manifest against the frozen Draft 2020-12 schema and the
   frozen standalone semantic validator.
6. Match manifest identity, package version, implementation source commit,
   build identifier, source-tree digest, and implementation digest to the
   implementation archive and local wheel bytes.
7. Construct an in-memory `contract-valid` binding and validate it against the
   binding schema and `validate_operator_binding()` using all four frozen
   contract-surface artifacts.

Any failure rejects the entire provider release.

### Stage 2: Research validation

1. Start a separate Python process with isolated interpreter flags, a bounded
   timeout, and only the declared provider wheels prepended to its import path.
2. Import the exact module and callable declared by the manifest.
3. Convert the literal QuantPanel records to a pandas DataFrame.
4. Execute the declared invocation while retaining a deep copy of the input.
5. Verify that the input was not mutated, row/key alignment is unchanged, the
   declared output field exists, and literal values match within the declared
   tolerance.
6. Repeat every baseline case for every catalog entry.
7. Change each in-memory binding state to `research-certified` and validate it
   again before publication.

The subprocess executes the exact wheel artifacts, not provider source files
or an installed development checkout. It inherits the supported pandas/numpy
runtime from `open-xquant`; provider wheels are loaded without network access
or dependency installation.

## Results and Registry

Success atomically publishes a release directory containing:

```text
.open-xquant/certifications/<provider>/<release>/
├── certification-record.json
├── bindings/
│   ├── equant.ttr.sma@1.0.0.binding.json
│   └── ...
└── registry-entry.json
```

Each binding remains valid frozen-contract JSON. The separate certification
record stores the certifier identity (`open-xquant-local`), timestamp,
provider submission commit, implementation source commit, artifact digests,
baseline case results, and binding digests.
The registry entry is an index over these immutable records; it does not copy
provider source code and does not modify provider files.

Publication uses a staging directory, file flush/fsync, and atomic rename.
Failure before the final rename leaves no partial release. Re-certifying the
same provider release is idempotent only when every input and generated digest
matches; conflicting bytes are rejected rather than overwritten.

`research-certified` permits research and offline analysis only. Strategy or
live execution remains gated on `runtime-certified` plus `past_only`; this
feature does not promote operators to that state.

## Error Model

The service exposes stable stage-specific errors, including:

- invalid repository or non-full/non-commit revision;
- malformed catalog, build record, baseline, manifest, or binding;
- archive path escape or referenced symlink;
- source, manifest, contract-surface, or wheel digest mismatch;
- missing runtime artifact or undeclared dependency artifact;
- provider import/call failure, timeout, input mutation, alignment failure, or
  numerical mismatch;
- conflicting existing certification output.

Human CLI output is concise. `--json` emits a stable object containing status,
stage, code, message, provider, release, and operator when known. Expected
validation failures exit with status 1 and never publish output.

## Testing

Implementation follows strict red-green-refactor TDD.

Service tests cover exact commit archival, strict JSON, containment, schema
then semantic ordering, digest binding, binding generation, all error codes,
and atomic/idempotent publication.

Baseline-runner tests build small local wheels, execute literal cases, and
prove rejection of mutation, wrong output, import failure, and timeout. They
do not access the network or global package installers.

CLI tests use Click's `CliRunner` and verify argument handling, JSON output,
exit codes, trust acknowledgement, success output, and no-output-on-failure.

Packaging tests build an `open-xquant` wheel, install it into a temporary
environment, and prove the frozen schemas and validator are available through
package resources.

The final cross-repository acceptance uses the real eQuant commit, catalog,
manifests, baseline, build record, `equant-core` wheel, and `equant-ttr` wheel.
It must issue five `research-certified` bindings for SMA, EMA, RSI, ATR, and
Momentum, and independently recompute every recorded digest.

## Explicit Non-Goals

- Remote GitHub cloning or authentication.
- Building provider wheels during certification.
- Network dependency installation.
- Backtesting or strategy-performance evaluation.
- Runtime/live certification.
- Copying provider source into `open-xquant`.
- Loading untrusted third-party code without an external security sandbox.
