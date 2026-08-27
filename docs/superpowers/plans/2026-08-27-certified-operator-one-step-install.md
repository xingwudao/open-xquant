# Certified Operator One-Step Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `oxq operator install equant-py==1.0.0` securely download, verify, install, register, smoke-test, list, and invoke the 60 certified `equant-py` operators on the first supported target.

**Architecture:** open-xquant signs certification record v2 directly from a committed provider submission, exports deterministic portable evidence, resolves an official GitHub Release, and stores a verified exact-wheel closure without running pip. Every research invocation snapshots that closure and runs it through a stdlib-first `-I -S` contained child; `equant-py` owns the frozen provider manifests, Release inputs, asset builder, and public Release workflow.

**Tech Stack:** Python 3.12, Click, jsonschema Draft 2020-12, packaging, urllib, ZIP/wheel formats, pytest, GitHub Actions, GitHub Releases.

**Spec:** `docs/superpowers/specs/2026-08-27-certified-operator-install-design.md`

## Global Constraints

- Use only canonical names `open-xquant` and `equant-py`.
- Public provider releases use exact SemVer; Python distribution versions use PEP 440.
- First target is CPython 3.12, ABI `cp312`, `macosx_14_0_arm64`.
- Certification state remains `research-certified`; do not grant backtest, broker, or live authority.
- Phase-one trust states are exactly `github-source-trusted` and `local-unsigned-user-trusted`; never report `signed`.
- Provider code must never import in the `oxq` control process.
- Installation must never run pip or mutate the open-xquant SDK or a user environment.
- Child startup is `sys.executable -I -S`; it imports third-party modules only after exact-wheel extraction and `sys.path` restriction.
- Official URLs are HTTPS and every redirect hop is limited to the hosts frozen in the spec.
- v2 records, bundles, and generated release indexes are external Release assets and are never committed back into the provider submission.
- Preserve certified wheel bytes: `equant-core` digest `sha256:da6b9135e9dc6adea9d82b843662db150838530191fce7d659d7b833e1c6a30a`; `equant-ttr` digest `sha256:087f5f3b9ce1ff4071d31cf80e6b6015e64c677b330f31e1049dec26ba2ce8e5`.
- Preserve `technical-v1.json` bytes at `sha256:3b928d48f6a0b2c0500cd1e97f1ad9b10d0cf7c91ab9b908da51d7b9b1b8d579`.
- Do not modify `equant-py` root README, `packages/equant-ttr/README.md`, package pyprojects, `src/equant/**`, or `packages/equant-ttr/**` before the 1.0.0 Release.
- Exclude any pre-existing or generated untracked `equant-py/uv.lock` from all commits.
- Follow TDD for every behavior: observe the focused test fail before implementation, then pass, then run the listed regression set.

---

## Phase A: Certification record v2 and portable bundle

### Task 1: Shared strict formats and packaged distribution schemas

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/formats.py`
- Create: `contracts/operator-certification/certification-record-v2.schema.json`
- Create: `contracts/operator-distribution/certification-bundle-manifest-v1.schema.json`
- Create: `contracts/operator-install/operator-release-v1.schema.json`
- Create: `contracts/operator-install/installed-release-v1.schema.json`
- Create: `contracts/operator-install/operator-runtime-protocol-v1.schema.json`
- Create: `contracts/operator-install/official-providers-v1.json`
- Create: `tests/contracts/test_operator_distribution_schemas.py`
- Modify: `src/oxq/operators/resources.py`
- Modify: `src/oxq/operators/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/operators/test_resources.py`
- Modify: `tests/operators/test_installed_certification.py`

**Interfaces:**
- Produces `strict_json_object(raw: bytes) -> dict[str, object]`, `canonical_json_bytes(value: Mapping[str, object]) -> bytes`, `sha256_bytes(value: bytes) -> str`, and `safe_relative_path(value: str) -> PurePosixPath`.
- Produces `materialize_operator_distribution_profile()` and `materialize_operator_install_profile()` resource context managers.

- [ ] **Step 1: Write failing schema and resource tests**

```python
def test_operator_distribution_resources_are_packaged_and_strict() -> None:
    with materialize_operator_distribution_profile() as paths:
        schema = json.loads(paths["certification_bundle_manifest"].read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
```

Also assert v2 requires `target`, `baseline_sets`, per-case `baseline_path`, `case_index`, and `case_digest`; install schemas require `size_bytes` upper bounds and reject extra fields.

- [ ] **Step 2: Run the focused tests and observe missing resources**

Run: `uv run pytest tests/contracts/test_operator_distribution_schemas.py tests/operators/test_resources.py -q`

Expected: FAIL because the new schemas and materializers do not exist.

- [ ] **Step 3: Add schemas, shared format functions, and wheel force-includes**

```python
def strict_json_object(raw: bytes) -> dict[str, object]:
    value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return cast(dict[str, object], value)

def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
```

Move behavior, not error policy, from registry private helpers. Keep compatibility wrappers in `registry.py` until later tasks migrate callers.

- [ ] **Step 4: Run focused and installed-wheel resource tests**

Run: `uv run pytest tests/contracts/test_operator_distribution_schemas.py tests/operators/test_resources.py tests/operators/test_installed_certification.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add contracts/operator-certification/certification-record-v2.schema.json contracts/operator-distribution contracts/operator-install src/oxq/operators/formats.py src/oxq/operators/resources.py src/oxq/operators/__init__.py pyproject.toml tests/contracts/test_operator_distribution_schemas.py tests/operators/test_resources.py tests/operators/test_installed_certification.py
git commit -m "feat: package operator distribution contracts"
```

### Task 2: Submission provenance and direct v2 certification issuance

**Repository:** `open-xquant`

**Files:**
- Modify: `src/oxq/operators/models.py`
- Modify: `src/oxq/operators/submission.py`
- Modify: `src/oxq/operators/certification.py`
- Modify: `src/oxq/operators/registry.py`
- Modify: `src/oxq/cli/main.py`
- Modify: `tests/operators/helpers.py`
- Modify: `tests/operators/test_submission.py`
- Create: `tests/operators/test_certification_v2.py`
- Modify: `tests/operators/test_certification_registry.py`
- Modify: `tests/cli/test_operator_certify.py`

**Interfaces:**
- Produces immutable `CertificationTarget.parse(value)` and `.key`.
- Changes `publish_certification(result, output_dir, *, target: CertificationTarget | None = None)`; `None` remains v1 and a target directly issues v2.
- Produces `read_certification_publication(release_dir)` and `import_certification_publication(publication_dir, output_dir)`.
- Adds `oxq operator certify-provider --target cp312-cp312-macosx_14_0_arm64`; omitting `--target` preserves v1 behavior.

- [ ] **Step 1: Write failing provenance and v2 issuance tests**

```python
def test_targeted_certification_issues_v2_from_committed_evidence(tmp_path: Path) -> None:
    result = certified_result_with_committed_baseline(tmp_path)
    published = publish_certification(
        result,
        tmp_path / "registry",
        target=CertificationTarget.parse("cp312-cp312-macosx_14_0_arm64"),
    )
    record = strict_json_object((published.release_dir / "certification-record.json").read_bytes())
    assert record["schema_version"] == 2
    assert record["target"]["platform_tag"] == "macosx_14_0_arm64"
    assert record["baseline_sets"][0]["digest"].startswith("sha256:")
```

Add tests that case digest equals SHA-256 of canonical parsed case JSON, v1 remains byte-compatible, target errors occur before provider execution, and v2 rejects synthetic cases without committed path/index provenance.

- [ ] **Step 2: Run focused tests and observe v2 failure**

Run: `uv run pytest tests/operators/test_submission.py tests/operators/test_certification_v2.py tests/operators/test_certification_registry.py tests/cli/test_operator_certify.py -q`

Expected: FAIL because provenance fields, target parsing, and v2 publication are absent.

- [ ] **Step 3: Preserve committed baseline provenance**

```python
@dataclass(frozen=True)
class CertificationTarget:
    python_tag: str
    abi_tag: str
    platform_tag: str

@dataclass(frozen=True)
class BaselineCase:
    operator_id: str
    operator_version: str
    case_id: str
    input: Mapping[str, object]
    parameters: Mapping[str, object]
    expected: Mapping[str, object]
    tolerance: Mapping[str, object]
    baseline_path: Path | None = None
    baseline_relative_path: str | None = None
    case_index: int | None = None
```

Populate the three new fields while enumerating committed baseline arrays in `submission.py`; preserve them through certification freezing.

- [ ] **Step 4: Implement v2 render/read dispatch without converting v1**

Render raw manifest digest, raw baseline-set digest, and canonical per-case digest directly from the certified submission snapshots. Dispatch record validation by `schema_version`; keep the registry entry schema stable because it already binds the record digest.

- [ ] **Step 5: Run v2, v1 regression, and CLI tests**

Run: `uv run pytest tests/operators/test_submission.py tests/operators/test_certification_v2.py tests/operators/test_certification_registry.py tests/cli/test_operator_certify.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/oxq/operators/models.py src/oxq/operators/submission.py src/oxq/operators/certification.py src/oxq/operators/registry.py src/oxq/cli/main.py tests/operators/helpers.py tests/operators/test_submission.py tests/operators/test_certification_v2.py tests/operators/test_certification_registry.py tests/cli/test_operator_certify.py
git commit -m "feat: issue targeted operator certifications"
```

### Task 3: Deterministic certification bundle validation and export

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/bundle.py`
- Create: `tests/operators/test_certification_bundle.py`

**Interfaces:**
- Produces `ValidatedCertificationBundle`.
- Produces `export_certification_bundle(*, provider: str, release: str, registry_dir: str | Path, manifest_dir: str | Path, baseline_files: Sequence[str | Path], target: CertificationTarget, output_path: str | Path) -> ValidatedCertificationBundle`.
- Produces `validate_certification_bundle(bundle_path: str | Path) -> ValidatedCertificationBundle`.
- Produces `materialize_validated_bundle(bundle: ValidatedCertificationBundle, destination: Path) -> None`.
- Freezes ZIP limits: 512 members, 16 MiB per expanded member, 64 MiB total expanded bytes, compression ratio 100, compressed bundle size 32 MiB.

```python
@dataclass(frozen=True)
class ValidatedCertificationBundle:
    bundle_path: Path
    provider: str
    release: str
    target: CertificationTarget
    operator_count: int
    digest: str
    members: Mapping[str, bytes]

```

- [ ] **Step 1: Write failing deterministic and malicious-ZIP tests**

```python
def test_export_is_byte_deterministic_and_preserves_source_json(tmp_path: Path) -> None:
    first = export_bundle_fixture(tmp_path / "one.zip")
    second = export_bundle_fixture(tmp_path / "two.zip")
    assert first.bundle_path.read_bytes() == second.bundle_path.read_bytes()
    with zipfile.ZipFile(first.bundle_path) as archive:
        assert archive.read("manifests/equant.ttr.sma@1.0.0.operator.json") == RAW_MANIFEST
```

Parameterize traversal, backslash, NUL, duplicate name, symlink, special file, encrypted entry, wrong compression, member-count, size, total-size, and ratio failures.

- [ ] **Step 2: Run focused tests and observe missing bundle API**

Run: `uv run pytest tests/operators/test_certification_bundle.py -q`

Expected: FAIL at import.

- [ ] **Step 3: Implement strict member streaming and deterministic ZIP writing**

Use lexical member order, timestamp `(1980, 1, 1, 0, 0, 0)`, Unix regular mode `0o100644`, no comments or extras, `ZIP_DEFLATED`, and `compresslevel=9`. Never call `extractall`; validate each `ZipInfo` before reading its bounded bytes.

- [ ] **Step 4: Cross-check publication, manifests, baseline sets, and v2 case evidence**

Require exactly one manifest and at least one v2-bound case per binding; verify raw file digests, canonical case digests, target, implementation, distribution, source commit, and operator count.

- [ ] **Step 5: Run bundle and registry tests**

Run: `uv run pytest tests/operators/test_certification_bundle.py tests/operators/test_certification_registry.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/oxq/operators/bundle.py tests/operators/test_certification_bundle.py
git commit -m "feat: export deterministic certification bundles"
```

### Task 4: Atomic bundle import and bundle CLI

**Repository:** `open-xquant`

**Files:**
- Modify: `src/oxq/operators/bundle.py`
- Create: `tests/cli/test_operator_bundle.py`
- Modify: `tests/operators/test_certification_bundle.py`
- Modify: `contracts/operator-certification/README.md`

**Interfaces:**
- Produces `import_certification_bundle(bundle_path, output_dir, *, trust_unsigned_bundle, bundle_store=None)`.
- Adds `operator export-certification` and `operator import-certification` commands with `--json`.

- [ ] **Step 1: Write failing import transaction and CLI tests**

```python
def test_import_requires_trust_and_is_idempotent(tmp_path: Path) -> None:
    bundle = export_bundle_fixture(tmp_path / "bundle.zip")
    with pytest.raises(OperatorCertificationError, match="trust"):
        import_certification_bundle(bundle.bundle_path, tmp_path / "registry", trust_unsigned_bundle=False)
    one = import_certification_bundle(bundle.bundle_path, tmp_path / "registry", trust_unsigned_bundle=True)
    two = import_certification_bundle(bundle.bundle_path, tmp_path / "registry", trust_unsigned_bundle=True)
    assert one.release_dir == two.release_dir
```

Add conflicting bytes, input-inside-output, staging failure, fsync failure, bundle-store atomicity, human output, JSON output, and stable exit-code tests.

- [ ] **Step 2: Run focused tests and observe missing import and commands**

Run: `uv run pytest tests/operators/test_certification_bundle.py tests/cli/test_operator_bundle.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement import through the registry atomic commit path**

Validate the full bundle first, materialize the publication into same-filesystem sibling staging, then reuse `import_certification_publication`. Publish the optional audit bundle only after registry success. Plain import must not create an installed runtime.

- [ ] **Step 4: Add CLI options exactly as frozen in the spec**

Support repeated `--baseline-file`; require export output outside the source registry; require `--trust-unsigned-bundle` for import; sanitize errors and emit sorted JSON.

- [ ] **Step 5: Run bundle, CLI, and installed-wheel tests**

Run: `uv run pytest tests/operators/test_certification_bundle.py tests/cli/test_operator_bundle.py tests/operators/test_installed_certification.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/oxq/operators/bundle.py src/oxq/cli contracts/operator-certification/README.md tests/operators/test_certification_bundle.py tests/cli/test_operator_bundle.py
git commit -m "feat: import certification bundles atomically"
```

## Phase B: Official release resolution, managed installation, and runtime

### Task 5: Release index, exact requirement, official provider, and target selection

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/release_index.py`
- Create: `src/oxq/operators/install_errors.py`
- Create: `tests/operators/test_release_index.py`

**Interfaces:**
- Produces `OfficialProvider`, `ReleaseAsset`, `ReleaseWheel`, `ReleaseTarget`, `OperatorReleaseIndex`, `parse_exact_requirement`, `load_official_provider`, `parse_release_index`, and `select_release_target`.
- Produces structured `OperatorInstallError(code, message, *, stage, provider=None, release=None, operator_id=None)` with sanitized `as_dict()`.

```python
@dataclass(frozen=True)
class ReleaseAsset:
    filename: str
    url: str
    size_bytes: int
    digest: str

@dataclass(frozen=True)
class ReleaseWheel(ReleaseAsset):
    distribution: str
    version: str
    role: str
    tags: tuple[str, ...]

@dataclass(frozen=True)
class ReleaseTarget:
    python_tag: str
    abi_tag: str
    platform_tag: str
    bundle: ReleaseAsset
    wheels: tuple[ReleaseWheel, ...]

@dataclass(frozen=True)
class OperatorReleaseIndex:
    raw_bytes: bytes
    provider: str
    release: str
    submission_commit: str
    source_commit: str
    certification_state: str
    operator_count: int
    targets: tuple[ReleaseTarget, ...]
```

- [ ] **Step 1: Write failing exact requirement and target tests**

```python
@pytest.mark.parametrize("value", ["equant-py", "equant-py>=1.0.0", "equant-py==1.0", "equant-py==1!1.0.0", "equant-py==1.0.0+local"])
def test_requirement_rejects_non_exact_semver(value: str) -> None:
    with pytest.raises(OperatorInstallError) as caught:
        parse_exact_requirement(value)
    assert caught.value.code == "operator_requirement_invalid"
```

Test unknown provider, zero compatible targets, multiple compatible targets, current-tag mismatch, extra fields, collection bounds, and compatibility of every wheel tag.

- [ ] **Step 2: Run tests and observe missing parser**

Run: `uv run pytest tests/operators/test_release_index.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement frozen dataclasses and schema-first parsing**

```python
def parse_exact_requirement(value: str) -> tuple[str, str]:
    match = EXACT_PROVIDER_REQUIREMENT.fullmatch(value)
    if match is None:
        raise install_error("operator_requirement_invalid", "operator requirement must be provider==semver")
    return match.group("provider"), match.group("release")
```

Parse only strict canonical JSON, validate the packaged schema, and require exactly one target whose declared tag and every wheel tag occur in `packaging.tags.sys_tags()`.

- [ ] **Step 4: Run release-index tests**

Run: `uv run pytest tests/operators/test_release_index.py tests/contracts/test_operator_distribution_schemas.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/oxq/operators/release_index.py src/oxq/operators/install_errors.py tests/operators/test_release_index.py
git commit -m "feat: parse certified operator releases"
```

### Task 6: Official GitHub discovery and bounded verified downloads

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/release_discovery.py`
- Create: `tests/operators/test_release_discovery.py`

**Interfaces:**
- Produces `OfficialReleaseResolver.resolve(provider, release)` and `download_verified_asset(asset, destination, *, opener=None)`.

- [ ] **Step 1: Write failing source-trust and redirect tests**

Test exact tag `v1.0.0`, missing asset, HTTP downgrade, disallowed redirect host, excessive redirects, interrupted stream, size mismatch, digest mismatch, and secret/query redaction.

- [ ] **Step 2: Run tests and observe missing resolver**

Run: `uv run pytest tests/operators/test_release_discovery.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement standard-library GitHub resolver and streaming downloader**

Use `urllib.request`, explicit redirect handling, HTTPS checks at every hop, small fixed index limit, chunked asset writes, simultaneous byte count/SHA-256, and atomic destination replace. Return trust state exactly `github-source-trusted`.

- [ ] **Step 4: Run discovery tests**

Run: `uv run pytest tests/operators/test_release_discovery.py tests/operators/test_release_index.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/oxq/operators/release_discovery.py tests/operators/test_release_discovery.py
git commit -m "feat: resolve official operator releases"
```

### Task 7: Exact wheel-closure validation

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/wheel_closure.py`
- Create: `tests/operators/wheel_helpers.py`
- Create: `tests/operators/test_wheel_closure.py`

**Interfaces:**
- Produces `VerifiedWheel`, `VerifiedWheelClosure`, and `verify_wheel_closure(target, wheel_paths, *, certified_artifacts)`.

- [ ] **Step 1: Write failing wheel format and closure tests**

Cover distribution/version/tag mismatches, unresolved and extra dependencies, METADATA errors, WHEEL errors, RECORD hash/size/missing/extra entries, path escape, duplicate names, backslashes, symlinks, encryption, special files, per-member/total/ratio bounds, and proof that `subprocess` never receives pip.

- [ ] **Step 2: Run tests and observe missing verifier**

Run: `uv run pytest tests/operators/test_wheel_closure.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement bounded ZIP validation before metadata parsing**

Reuse the requirement-closure semantics from `submission.py`, but validate every archive member and RECORD entry first. Cross-check release index, certification record, METADATA Name/Version/Requires-Dist, WHEEL Tag, filename tags, and complete closure identity.

- [ ] **Step 4: Run wheel and submission regressions**

Run: `uv run pytest tests/operators/test_wheel_closure.py tests/operators/test_submission.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```bash
git add src/oxq/operators/wheel_closure.py tests/operators/wheel_helpers.py tests/operators/test_wheel_closure.py
git commit -m "feat: verify exact operator wheel closures"
```

### Task 8: Extract the contained child-process controller

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/child_process.py`
- Create: `tests/operators/test_child_process.py`
- Modify: `src/oxq/operators/baseline_runner.py`
- Modify: `tests/operators/test_baseline_runner.py`

**Interfaces:**
- Produces `run_contained_child(command, *, timeout_seconds, response_secret, environment) -> int`.

- [ ] **Step 1: Move existing process-control tests to the new public seam**

Move or duplicate tests for DEVNULL stdout/stderr, timeout, Linux descendants and supervisor, macOS fail-closed containment, Windows job objects, and detached descendants. Add an assertion that commands missing `-I` or `-S` are rejected.

- [ ] **Step 2: Run the new tests and observe missing module**

Run: `uv run pytest tests/operators/test_child_process.py tests/operators/test_baseline_runner.py -q`

Expected: FAIL in the new tests while existing baseline tests remain green.

- [ ] **Step 3: Move process-control code without changing behavior**

Transfer `_run_child_process`, containment launchers, process-tree termination, subreaper, Windows job, and platform helpers. Build child environments from an allowlist and remove `PYTHONPATH`, credentials, and proxy variables.

- [ ] **Step 4: Make baseline runner consume the shared controller**

Delete duplicated implementations only after the baseline suite passes against the shared controller.

- [ ] **Step 5: Run full process-control regression**

Run: `uv run pytest tests/operators/test_child_process.py tests/operators/test_baseline_runner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 8**

```bash
git add src/oxq/operators/child_process.py src/oxq/operators/baseline_runner.py tests/operators/test_child_process.py tests/operators/test_baseline_runner.py
git commit -m "refactor: share contained operator child control"
```

### Task 9: Stdlib-first exact-closure child and authenticated protocol

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/_exact_wheel_child.py`
- Create: `src/oxq/operators/runtime_protocol.py`
- Create: `tests/operators/test_exact_wheel_child_security.py`
- Create: `tests/operators/test_runtime_protocol.py`
- Modify: `src/oxq/operators/baseline_runner.py`
- Modify or remove after migration: `src/oxq/operators/_baseline_child.py`
- Modify: `tests/operators/test_baseline_child_security.py`

**Interfaces:**
- Produces `canonical_protocol_bytes(value)` and `run_exact_wheel_request(request, wheel_snapshots, *, timeout_seconds)`.

- [ ] **Step 1: Write a regression that proves ambient NumPy/pandas are never imported**

```python
def test_child_imports_platform_runtime_only_from_exact_closure(tmp_path: Path) -> None:
    response = run_exact_fixture_with_poisoned_ambient_runtime(tmp_path)
    assert response["status"] == "passed"
    assert response["module_origins"]["numpy"].startswith(response["closure_root"])
    assert response["module_origins"]["pandas"].startswith(response["closure_root"])
```

Add tests for HMAC tampering, response bounds, import origin, provider stdout/stderr, provider exception sanitization, dynamic execution restrictions, wheel mutation after snapshot, and descendant termination.

- [ ] **Step 2: Run tests and observe the ambient-runtime regression fail**

Run: `uv run pytest tests/operators/test_exact_wheel_child_security.py tests/operators/test_runtime_protocol.py -q`

Expected: FAIL because the current child imports SDK NumPy/pandas before extraction.

- [ ] **Step 3: Implement stdlib-only bootstrap before third-party import**

Start with `sys.executable -I -S`. Use only stdlib to validate and extract wheel snapshot files, set restricted `sys.path`, hide ambient modules, then first import NumPy, pandas, Numba, dependencies, and provider. Reject every third-party module origin outside the closure.

- [ ] **Step 4: Preserve existing alignment, dtype, context, import-gate, and response security**

Move the validated output and frame helpers from `_baseline_child.py`; retain HMAC response files and DEVNULL streams. Certification baseline execution and installed runtime must call the same child protocol.

- [ ] **Step 5: Run child, baseline, and security regressions**

Run: `uv run pytest tests/operators/test_exact_wheel_child_security.py tests/operators/test_runtime_protocol.py tests/operators/test_baseline_child_security.py tests/operators/test_baseline_runner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 9**

```bash
git add src/oxq/operators/_exact_wheel_child.py src/oxq/operators/runtime_protocol.py src/oxq/operators/baseline_runner.py src/oxq/operators/_baseline_child.py tests/operators
git commit -m "feat: run operators from exact wheel closures"
```

### Task 10: Safe installed-release store and snapshots

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/safe_files.py`
- Create: `src/oxq/operators/installed_store.py`
- Create: `tests/operators/test_installed_store.py`
- Modify: `src/oxq/operators/registry.py`

**Interfaces:**
- Produces `InstalledRelease`, `InstalledOperator`, `InstalledReleaseSnapshot`, and `InstalledReleaseStore.publish/list/get/resolve_operator/snapshot`.

```python
@dataclass(frozen=True)
class InstalledRelease:
    provider: str
    release: str
    target: str
    path: Path
    trust_state: str
    certification_state: str
    operators: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class InstalledOperator:
    release: InstalledRelease
    binding: Mapping[str, object]
    manifest: Mapping[str, object]
    certified_cases: tuple[Mapping[str, object], ...]

@dataclass(frozen=True)
class InstalledReleaseSnapshot:
    release: InstalledRelease
    marker: Mapping[str, object]
    release_index: bytes
    bundle: bytes
    publication_files: Mapping[str, bytes]
    manifest_files: Mapping[str, bytes]
    baseline_files: Mapping[str, bytes]
    wheel_snapshots: Mapping[str, Path]
```

- `InstalledReleaseStore.publish(staging_dir: Path, marker: Mapping[str, object]) -> InstalledRelease`.
- `InstalledReleaseStore.list() -> tuple[InstalledRelease, ...]`.
- `InstalledReleaseStore.get(provider: str, release: str, target: str | None = None) -> InstalledRelease`.
- `InstalledReleaseStore.resolve_operator(operator_id: str, operator_version: str, provider: str, provider_release: str) -> InstalledOperator`.
- `InstalledReleaseStore.snapshot(release: InstalledRelease) -> ContextManager[InstalledReleaseSnapshot]`.

- [ ] **Step 1: Write failing marker, conflict, corruption, and race tests**

```python
def test_release_is_invisible_until_valid_marker_is_committed(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    create_partial_release(store.home)
    assert store.list() == ()
```

Test canonical tree digest, bundle ZIP retention, exact file set, protocol/version range, idempotence, conflict, concurrent lock, staging crash, symlink, partial mutation, and snapshot/check-use path replacement.

- [ ] **Step 2: Run focused tests and observe missing store**

Run: `uv run pytest tests/operators/test_installed_store.py -q`

Expected: FAIL.

- [ ] **Step 3: Extract no-follow and atomic filesystem helpers**

Move tested regular-file descriptor reads, real-directory checks, fsync, and atomic directory replace from registry into `safe_files.py`, retaining Windows and POSIX behavior.

- [ ] **Step 4: Implement store publication and snapshot semantics**

Default to `~/.config/open-xquant/operator-releases`, honor `OPEN_XQUANT_OPERATOR_HOME`, write marker last, expose no unmarked directory, validate all path/digest/size/tree entries, and snapshot verified bytes rather than passing reopened managed paths.

- [ ] **Step 5: Run store and registry regressions**

Run: `uv run pytest tests/operators/test_installed_store.py tests/operators/test_certification_registry.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 10**

```bash
git add src/oxq/operators/safe_files.py src/oxq/operators/installed_store.py src/oxq/operators/registry.py tests/operators/test_installed_store.py tests/operators/test_certification_registry.py
git commit -m "feat: store installed operator releases safely"
```

### Task 11: Offline and official transactional installer

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/installer.py`
- Create: `tests/operators/test_installer.py`

**Interfaces:**
- Produces `CertifiedOperatorInstaller.install_local(release_index: Path, artifact_dir: Path, *, trust_unsigned_release: bool, smoke_timeout_seconds: float = 30) -> InstalledRelease`.
- Produces `CertifiedOperatorInstaller.install_official(requirement: str, *, smoke_timeout_seconds: float = 30) -> InstalledRelease`.

- [ ] **Step 1: Write the failing offline happy path and transaction failures**

```python
def test_local_install_validates_smokes_and_commits_release(tmp_path: Path) -> None:
    installer = installer_with_fake_exact_child(tmp_path)
    installed = installer.install_local(
        tmp_path / "assets/operator-release-v1.json",
        tmp_path / "assets",
        trust_unsigned_release=True,
    )
    assert installed.trust_state == "local-unsigned-user-trusted"
    assert len(installed.operators) == 60
```

Test missing trust, filenames outside direct artifact children, remote URL access in local mode, target mismatch, bundle/index/record disagreement, wheel failure, smoke failure, marker failure, cleanup, idempotence, and conflict.

- [ ] **Step 2: Run installer tests and observe missing orchestrator**

Run: `uv run pytest tests/operators/test_installer.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement one shared staged install pipeline**

Freeze order: parse index, select target, acquire assets, verify size/digest, validate bundle, validate wheel closure, cross-check v2 evidence, snapshot assets, run all 60 v2 cases through exact child, write marker, atomically publish.

- [ ] **Step 4: Connect official resolution to the same pipeline**

Official mode resolves the built-in repository and downloads assets; it records `github-source-trusted`. Local mode resolves every declared filename directly under `artifact_dir` and records `local-unsigned-user-trusted`.

- [ ] **Step 5: Run installer and dependency suites**

Run: `uv run pytest tests/operators/test_installer.py tests/operators/test_release_discovery.py tests/operators/test_wheel_closure.py tests/operators/test_installed_store.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 11**

```bash
git add src/oxq/operators/installer.py tests/operators/test_installer.py
git commit -m "feat: install certified operator releases"
```

### Task 12: Operator CLI extraction, install, and list

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/cli/operator.py`
- Create: `tests/cli/test_operator_install.py`
- Create: `tests/cli/test_operator_list.py`
- Modify: `src/oxq/cli/main.py`

**Interfaces:**
- Adds the exact commands from the spec and preserves `certify-provider`, export, and import behavior.

- [ ] **Step 1: Write failing Click command tests**

Test official requirement mode, offline option grouping, missing trust, human summary, JSON identity list, corrupt installation error, sanitized structured failure, and stable exit code 1.

- [ ] **Step 2: Run CLI tests and observe missing commands**

Run: `uv run pytest tests/cli/test_operator_install.py tests/cli/test_operator_list.py tests/cli/test_operator_bundle.py tests/cli/test_operator_certify.py -q`

Expected: FAIL for install/list.

- [ ] **Step 3: Extract the operator group and implement install/list adapters**

```python
@operator_group.command(name="install")
@click.argument("requirement", required=False)
@click.option("--release-index", type=click.Path(path_type=Path))
@click.option("--artifact-dir", type=click.Path(path_type=Path))
@click.option("--trust-unsigned-release", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def install_command(requirement, release_index, artifact_dir, trust_unsigned_release, as_json):
    mode = validate_install_mode(requirement, release_index, artifact_dir, trust_unsigned_release)
    render_install_result(run_install(mode), as_json=as_json)
```

Human list output is one release summary with `operator_count`; JSON includes the complete ordered operator identity array.

- [ ] **Step 4: Run all operator CLI tests**

Run: `uv run pytest tests/cli/test_operator_install.py tests/cli/test_operator_list.py tests/cli/test_operator_bundle.py tests/cli/test_operator_certify.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 12**

```bash
git add src/oxq/cli/operator.py src/oxq/cli/main.py tests/cli
git commit -m "feat: add certified operator install commands"
```

### Task 13: Python research runtime

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/runtime.py`
- Create: `tests/operators/test_runtime.py`
- Modify: `src/oxq/operators/__init__.py`

**Interfaces:**
- Produces `CertifiedOperatorRuntime.__init__(operator_home=None, *, timeout_seconds=30)` and `.invoke(operator_id, *, operator_version, provider_release, panel, parameters) -> pandas.DataFrame`.

- [ ] **Step 1: Write failing resolution, case-bound parameter, and result tests**

```python
def test_invoke_returns_aligned_dataframe_from_installed_exact_closure(tmp_path: Path) -> None:
    runtime = runtime_with_installed_sma(tmp_path)
    result = runtime.invoke(
        "equant.ttr.sma",
        operator_version="1.0.0",
        provider_release="equant-py==1.0.0",
        panel=quant_panel_literal(),
        parameters={"n": 5},
    )
    assert list(result.columns) == ["sma_5"]
    assert result.index.names == ["date", "code"]
```

Test missing release, ambiguous identity, invalid QuantPanel, parameters not canonical-equal to a v2 case, Series/DataFrame/multi-output, timeout, HMAC failure, provider error redaction, and proof provider modules never enter parent `sys.modules`.

- [ ] **Step 2: Run focused tests and observe missing runtime**

Run: `uv run pytest tests/operators/test_runtime.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement resolve, validate, snapshot, invoke, and reconstruct**

Parse exact `provider_release`; resolve one installed binding/manifest/case set; validate QuantPanel and request parameters; require canonical parameter bytes equal a bound v2 case; snapshot exact bytes; run the exact child; verify response identity, HMAC, keys, column contract, dtype, order, and size; construct a pandas DataFrame with the declared primary-key MultiIndex.

- [ ] **Step 4: Prevent use as a runtime/backtest/live authority**

Keep this class under `oxq.operators.runtime`; do not register it in strategy compilation, portfolio execution, broker, or live modules. Add an import-isolation test proving no such integration occurs.

- [ ] **Step 5: Run runtime and full operator suites**

Run: `uv run pytest tests/operators/test_runtime.py tests/operators -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 13**

```bash
git add src/oxq/operators/runtime.py src/oxq/operators/__init__.py tests/operators/test_runtime.py
git commit -m "feat: invoke installed research operators"
```

## Phase C: Provider metadata, Release assets, and cross-repository acceptance

### Task 14: Correct dynamic output metadata without changing wheel bytes

**Repository:** `../equant-py`

**Files:**
- Modify: `../equant-py/tools/generate_open_xquant_contract.py`
- Modify: exactly the 31 affected files under `../equant-py/compat/open_xquant/manifests/`
- Modify: `../equant-py/tests/contracts/test_initial_operator_catalog.py`

**Interfaces:**
- Produces explicit `OUTPUT_FIELD_TEMPLATES: dict[str, tuple[str, ...]]` and parameter definitions with truthful `affects_output_fields`.

- [ ] **Step 1: Write failing template/default and changed-parameter tests**

Assert `name_template.format(**case["parameters"])` equals expected fields for all 60 cases. Assert the 31 `n` parameters listed in the spec review are `affects_output_fields: true`; all other declared parameters remain false.

- [ ] **Step 2: Run contract tests and observe dynamic-template failures**

Run from `../equant-py`: `PYTHONPATH=src:packages/equant-ttr uv run --with jsonschema --with pyyaml --with pytest --with numba==0.61.2 --with pandas==2.2.3 --with numpy==2.2.6 python -m pytest tests/contracts/test_initial_operator_catalog.py -q`

Expected: FAIL for current literal templates/flags.

- [ ] **Step 3: Implement explicit templates and regenerate manifests**

Use this fixed mapping, not string guessing:

```python
OUTPUT_FIELD_TEMPLATES = {
    "sma": ("sma_{n}",),
    "ema": ("ema_{n}",),
    "dema": ("DEMA_{n}",),
    "wma": ("WMA_{n}",),
    "hma": ("HMA_{n}",),
    "zlema": ("ZLEMA_{n}",),
    "alma": ("ALMA_{n}",),
    "evwma": ("EVWMA_{n}",),
    "vwma": ("VWMA_{n}",),
    "adx": ("ADX_{n}", "DIp_{n}", "DIn_{n}", "DX_{n}"),
    "tdi": ("TDI_{n}", "DI_{n}"),
    "trix": ("TRIX_{n}", "TRIX_{n}_signal"),
    "dpo": ("DPO_{n}",),
    "vhf": ("VHF_{n}",),
    "rsi": ("rsi_{n}",),
    "cci": ("CCI_{n}",),
    "cmo": ("CMO_{n}",),
    "wpr": ("WPR_{n}",),
    "roc": ("ROC_{n}",),
    "momentum": ("momentum_{n}",),
    "cti": ("CTI_{n}",),
    "rvi": ("RVI_{n}", "RVI_{n}_signal"),
    "dvi": ("DVI_{n}",),
    "atr": ("atr_{n}",),
    "volatility": ("Vol_close_{n}",),
    "cmf": ("CMF_{n}",),
    "mfi": ("MFI_{n}",),
    "emv": ("EMV_{n}",),
    "chaikin_volatility": ("ChaikinVol_{n}",),
    "growth": ("growth_{n}",),
    "lags": ("lag_{n}",),
}
```

Mark only these callables' `n` parameter as `affects_output_fields: true`. Regenerate and assert exactly 31 manifests changed. Keep `compat/open_xquant/numerical_baselines/technical-v1.json` raw digest `3b928d48f6a0b2c0500cd1e97f1ad9b10d0cf7c91ab9b908da51d7b9b1b8d579`.

- [ ] **Step 4: Verify contracts and frozen artifacts**

Run the focused contract command above, then:

```bash
shasum -a 256 dist/equant_core-1.0.0-py3-none-any.whl dist/equant_ttr-1.0.0-py3-none-any.whl compat/open_xquant/numerical_baselines/technical-v1.json
```

Expected: PASS. All 60 manifests pass; wheel and baseline digests equal the Global Constraints.

- [ ] **Step 5: Commit Task 14 in `equant-py`**

```bash
git -C ../equant-py add tools/generate_open_xquant_contract.py compat/open_xquant/manifests tests/contracts/test_initial_operator_catalog.py
git -C ../equant-py commit -m "fix: declare dynamic operator output fields"
```

### Task 15: Deterministic provider Release asset builder

**Repository:** `../equant-py`

**Files:**
- Create: `../equant-py/compat/open_xquant/operator-release-input-v1.json`
- Create: `../equant-py/tools/build_operator_release.py`
- Create: `../equant-py/tests/contracts/test_operator_release.py`
- Modify: `../equant-py/.gitignore`

**Interfaces:**
- Produces `build` and `verify` subcommands exactly as specified below.

- [ ] **Step 1: Write failing deterministic allowlist and cross-check tests**

```python
def test_release_builder_emits_exact_eight_asset_allowlist(tmp_path: Path) -> None:
    output = build_release_fixture(tmp_path)
    assert sorted(path.name for path in output.iterdir()) == sorted(EXPECTED_EIGHT_ASSETS)
```

Test provider/release/target/submission/source mismatch, bundle tamper, first- and third-party wheel digest mismatch, manifest/baseline/case mismatch, extra distribution, nonempty output, sorted non-self-referential `SHA256SUMS`, and byte determinism.

- [ ] **Step 2: Run focused tests and observe missing builder**

Run from `../equant-py`: `uv run --with pytest --with jsonschema --with packaging python -m pytest tests/contracts/test_operator_release.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement fixed Release input and builder**

```bash
release_work_dir=$(mktemp -d)
python tools/build_operator_release.py build --bundle "$release_work_dir/equant-py-1.0.0-macos-arm64-py312.open-xquant-certification.zip" --artifact-dir "$release_work_dir/closure" --output-dir "$release_work_dir/assets" --submission-commit "$(git rev-parse HEAD)"
python tools/build_operator_release.py verify --asset-dir "$release_work_dir/assets" --artifact-dir "$release_work_dir/closure" --submission-commit "$(git rev-parse HEAD)"
```

The input freezes official URLs, target, bundle filename, 60 operators, two first-party assets, and these eight exact third-party artifacts:

```json
[
  {
    "filename": "llvmlite-0.44.0-cp312-cp312-macosx_11_0_arm64.whl",
    "size_bytes": 26201105,
    "digest": "sha256:5f79a728e0435493611c9f405168682bb75ffd1fbe6fc360733b850c80a026db",
    "url": "https://files.pythonhosted.org/packages/d6/53/373b6b8be67b9221d12b24125fd0ec56b1078b660eeae266ec388a6ac9a0/llvmlite-0.44.0-cp312-cp312-macosx_11_0_arm64.whl"
  },
  {
    "filename": "numba-0.61.2-cp312-cp312-macosx_11_0_arm64.whl",
    "size_bytes": 2779287,
    "digest": "sha256:4ddce10009bc097b080fc96876d14c051cc0c7679e99de3e0af59014dab7dfe8",
    "url": "https://files.pythonhosted.org/packages/92/4a/fe4e3c2ecad72d88f5f8cd04e7f7cff49e718398a2fac02d2947480a00ca/numba-0.61.2-cp312-cp312-macosx_11_0_arm64.whl"
  },
  {
    "filename": "numpy-2.2.6-cp312-cp312-macosx_14_0_arm64.whl",
    "size_bytes": 5084103,
    "digest": "sha256:894b3a42502226a1cac872f840030665f33326fc3dac8e57c607905773cdcde3",
    "url": "https://files.pythonhosted.org/packages/3c/65/4baa99f1c53b30adf0acd9a5519078871ddde8d2339dc5a7fde80d9d87da/numpy-2.2.6-cp312-cp312-macosx_14_0_arm64.whl"
  },
  {
    "filename": "pandas-2.2.3-cp312-cp312-macosx_11_0_arm64.whl",
    "size_bytes": 11363475,
    "digest": "sha256:a5a1595fe639f5988ba6a8e5bc9649af3baf26df3998a0abe56c02609392e0a4",
    "url": "https://files.pythonhosted.org/packages/e1/0c/ad295fd74bfac85358fd579e271cded3ac969de81f62dd0142c426b9da91/pandas-2.2.3-cp312-cp312-macosx_11_0_arm64.whl"
  },
  {
    "filename": "python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
    "size_bytes": 229892,
    "digest": "sha256:a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427",
    "url": "https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl"
  },
  {
    "filename": "pytz-2026.3.post1-py2.py3-none-any.whl",
    "size_bytes": 508283,
    "digest": "sha256:dd95840dd199baea12d9cc096a1d452caa6596a1c1e4b5f3dbd1541855d5e815",
    "url": "https://files.pythonhosted.org/packages/0f/7b/39c34ca613b0b198cb866466651b26b045e2009864c5183c979a3b83f383/pytz-2026.3.post1-py2.py3-none-any.whl"
  },
  {
    "filename": "six-1.17.0-py2.py3-none-any.whl",
    "size_bytes": 11050,
    "digest": "sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
    "url": "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl"
  },
  {
    "filename": "tzdata-2026.3-py2.py3-none-any.whl",
    "size_bytes": 348168,
    "digest": "sha256:dc096730c87af6cab1b171c9d532be840741ff5d459015e7f6947bd7d7e54931",
    "url": "https://files.pythonhosted.org/packages/e5/6d/b53b99a9f2766d095985947a5782f1702cabb129a34f7a802d7197af832f/tzdata-2026.3-py2.py3-none-any.whl"
  }
]
```

Freeze the first-party sizes as 8401 bytes for `equant_core-1.0.0-py3-none-any.whl` and 33186 bytes for `equant_ttr-1.0.0-py3-none-any.whl`, with their GitHub Release URLs and Global Constraint digests. The builder validates all 10 local wheels but copies only the two first-party wheels. The output allowlist is exactly: two wheels, bundle, release index, `SHA256SUMS`, candidate build, toolchain, and catalog.

- [ ] **Step 4: Run Release and contract regressions**

Run: `uv run --with pytest --with jsonschema --with packaging python -m pytest tests/contracts/test_operator_release.py tests/contracts/test_initial_operator_catalog.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 15 in `equant-py`**

```bash
git -C ../equant-py add compat/open_xquant/operator-release-input-v1.json tools/build_operator_release.py tests/contracts/test_operator_release.py .gitignore
git -C ../equant-py commit -m "feat: build certified operator release assets"
```

### Task 16: Draft Release verification workflow and user documentation

**Repository:** `../equant-py`

**Files:**
- Create: `../equant-py/.github/workflows/release-certified-operators.yml`
- Modify: `../equant-py/compat/open_xquant/README.md`
- Modify: `../equant-py/docs/architecture/certified-operator-library.md`
- Modify: `../equant-py/tests/contracts/test_operator_release.py`
- Modify: `README.md` and `contracts/operator-certification/README.md` in open-xquant only after verifying open-xquant wheel expectations

**Interfaces:**
- Produces a manual-only workflow that verifies and publishes an existing draft Release; it never issues a second certification record.
- Pins `runs-on: macos-14`; the first shell step must assert `uname -m` is `arm64`, and the Python setup step must pin CPython 3.12.

- [ ] **Step 1: Write failing workflow-structure tests**

Parse YAML and assert `workflow_dispatch` only, `contents: write`, exact tag guard, `runs-on: macos-14`, an explicit `uname -m == arm64` guard, CPython 3.12 setup, eight-asset allowlist, eight third-party temporary downloads, frozen wheel rebuild, offline install and 60-case verification before `gh release edit --draft=false`, and no provider pip install.

- [ ] **Step 2: Run focused tests and observe missing workflow**

Run from `../equant-py`: `uv run --with pytest --with pyyaml python -m pytest tests/contracts/test_operator_release.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement workflow and documentation**

Document only the public quick start:

```bash
oxq operator install equant-py==1.0.0
oxq operator list
```

State `github-source-trusted != signed`, `research-certified != runtime/backtest/live`, and first-target limitations. Correct stale five-operator text. Do not edit README or package metadata files that alter certified wheels.

- [ ] **Step 4: Run Release, naming, and wheel-digest checks**

Run Release tests; run the repository legacy-name scan; recheck both wheel digests.

- [ ] **Step 5: Commit Task 16 in `equant-py`**

```bash
git -C ../equant-py add .github/workflows/release-certified-operators.yml compat/open_xquant/README.md docs/architecture/certified-operator-library.md tests/contracts/test_operator_release.py
git -C ../equant-py commit -m "ci: verify certified operator release"
```

### Task 17: Final v2 certification, offline acceptance, PR, and public Release

**Repositories:** `open-xquant` and `../equant-py`

**Files:**
- Generate outside Git: v2 certification registry, bundle, ten-wheel closure, eight draft Release assets, clean operator home
- Do not commit: generated v2 record, bundle, release index, wheels, `SHA256SUMS`, or local operator store

**Interfaces:**
- Consumes the final published install-capable open-xquant runner and the final frozen `equant-py` submission commit.
- Produces the public `v1.0.0` Release and a public-URL acceptance record in command output/PR evidence, not in the provider submission.

- [ ] **Step 1: Run full repository verification before publication**

Open-xquant:

```bash
uv run pytest tests/operators tests/cli/test_operator_*.py tests/contracts/test_operator_distribution_schemas.py -q
uv run ruff check src/oxq/operators src/oxq/cli tests/operators tests/cli
uv run mypy src/oxq/operators src/oxq/cli/operator.py
uv build --wheel
```

`equant-py`:

```bash
PYTHONPATH=src:packages/equant-ttr uv run --with jsonschema --with pyyaml --with pytest --with numba==0.61.2 --with pandas==2.2.3 --with numpy==2.2.6 python -m pytest tests/contracts -q
```

Expected: all commands exit 0 and frozen digests remain unchanged.

- [ ] **Step 2: Push `equant-py` and open the open-xquant PR**

Push the provider commits to `master` only if repository policy still allows it. Push `feat/operator-install-v1`, create/link an issue, open a PR to `main`, include the spec/plan and verification, and request `@codex review`. Do not merge without explicit user approval.

- [ ] **Step 3: After merge, publish or install the final open-xquant runner**

Build an install-capable open-xquant wheel or cached SDK bundle from merged `main`; verify `oxq operator install --help` exposes official and offline modes before final provider certification.

- [ ] **Step 4: Issue final v2 evidence exactly once**

Use final open-xquant against the exact final `equant-py` submission commit and target `cp312-cp312-macosx_14_0_arm64`. Export the bundle from that v2 publication. Do not edit or commit the resulting evidence.

- [ ] **Step 5: Build and verify the frozen eight Release assets**

Download the exact eight third-party wheels to a temporary ten-wheel closure, validate every digest, run `build_operator_release.py build`, then `verify`.

- [ ] **Step 6: Run clean offline one-step acceptance**

```bash
acceptance_work_dir=$(mktemp -d)
OPEN_XQUANT_OPERATOR_HOME="$acceptance_work_dir/operator-home" oxq operator install --release-index "$acceptance_work_dir/install-assets/operator-release-v1.json" --artifact-dir "$acceptance_work_dir/install-assets" --trust-unsigned-release
OPEN_XQUANT_OPERATOR_HOME="$acceptance_work_dir/operator-home" oxq operator list --json
```

Expected: trust state `local-unsigned-user-trusted`, 60 ordered identities, 60 canonical cases pass, repeat install is idempotent, and the SDK environment digest is unchanged.

- [ ] **Step 7: Create draft tag/Release and run the verification workflow**

Create `v1.0.0` pointing exactly at the certified submission commit, create a draft Release, upload exactly eight assets, dispatch `release-certified-operators.yml`, and publish only after it rebuilds frozen wheels and repeats offline installation/cases successfully.

- [ ] **Step 8: Run fresh public-URL acceptance**

```bash
public_acceptance_dir=$(mktemp -d)
OPEN_XQUANT_OPERATOR_HOME="$public_acceptance_dir/operator-home" oxq operator install equant-py==1.0.0
OPEN_XQUANT_OPERATOR_HOME="$public_acceptance_dir/operator-home" oxq operator list --json
```

Expected: trust state `github-source-trusted`, 60 identities, 60 cases pass, unsupported target fails closed, and no package in the open-xquant SDK changes.

- [ ] **Step 9: Record final evidence without changing the certified submission**

Update the open-xquant PR/issue or Release notes with exact test commands, public Release URL, tag commit, asset SHA-256 values, supported target, unsigned trust limitation, and research-only authority. Never commit generated evidence back to the tagged `equant-py` submission.

## Final verification matrix

- open-xquant v1 and v2 certification registry suites pass.
- Deterministic bundle export gives identical bytes for identical frozen inputs.
- Bundle import is secure, atomic, idempotent, and audit-only.
- Release index, network, and wheel-closure adversarial suites pass.
- Exact child proves NumPy, pandas, Numba, and provider origins are inside the certified closure.
- Installed-store corruption, crash, conflict, and check/use tests pass.
- Offline and official installers share the same staged transaction.
- Runtime accepts only v2-bound parameters and returns aligned pandas results.
- `equant-py` baseline and wheel digests remain frozen.
- Public Release contains exactly eight assets and third-party wheels are not uploaded.
- A fresh supported environment completes the exact public command and all 60 cases.
