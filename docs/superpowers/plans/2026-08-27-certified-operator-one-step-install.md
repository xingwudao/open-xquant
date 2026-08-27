# Certified Operator Environment Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the self-managed operator install store with verification and loading of certified provider packages installed in the current Python environment.

**Architecture:** `pip` or another environment manager installs `equant-py`; `open-xquant` discovers that installed distribution with `importlib.metadata`, validates packaged manifests and baselines against an official certification index, and exposes only verified certified operators. `open-xquant` no longer downloads, stores, or publishes provider wheels in its own install directory.

**Tech Stack:** Python 3.12, `importlib.metadata`, `importlib.resources`, Click, jsonschema Draft 2020-12, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-certified-operator-install-design.md`

## Global Constraints

- Use only canonical names `open-xquant` and `equant-py`.
- `open-xquant` must not act as a package manager in phase one.
- Do not use `OPEN_XQUANT_OPERATOR_HOME`.
- Do not create or use an installed-release store.
- Do not download GitHub Release wheels during provider use.
- Do not install provider packages into the open-xquant SDK bundle.
- Verification must require exact provider requirement syntax: `equant-py==1.0.0`.
- Verification must reject missing packages, wrong versions, changed manifest bytes, and changed baseline bytes.
- Certification state remains `research-certified`.
- Normal Python environment installation is explicitly allowed: `pip install equant-py==1.0.0`.
- Preserve existing certification/bundle code only where it supports evidence production or validation; do not use bundle import as the provider-code installation mechanism.
- Follow TDD for every behavior: focused failing test, implementation, focused passing test, regression suite, commit.

---

## Task 1: Freeze the new environment-install contract

**Repository:** `open-xquant`

**Files:**
- Modify: `docs/superpowers/specs/2026-08-27-certified-operator-install-design.md`
- Modify: `docs/superpowers/plans/2026-08-27-certified-operator-one-step-install.md`
- Modify: `.superpowers/sdd/2026-08-27-certified-operator-one-step-install/progress.md`

**Interfaces:**
- Produces the new authoritative phase-one contract: provider code is installed by Python tooling; `open-xquant` verifies and loads it.
- Marks the prior installed-store path as superseded, not partially accepted.

- [ ] **Step 1: Verify the old plan no longer claims store ownership**

Run:

```bash
rg -n "installed-release store|OPEN_XQUANT_OPERATOR_HOME|without running pip|exact-wheel closure|managed installation" docs/superpowers/specs/2026-08-27-certified-operator-install-design.md docs/superpowers/plans/2026-08-27-certified-operator-one-step-install.md
```

Expected: no matches except lines explicitly saying those features are removed or forbidden.

- [ ] **Step 2: Verify the new plan states the pip-based contract**

Run:

```bash
rg -n "pip install equant-py==1.0.0|operator verify equant-py==1.0.0|open-xquant.*not act as a package manager" docs/superpowers/specs/2026-08-27-certified-operator-install-design.md docs/superpowers/plans/2026-08-27-certified-operator-one-step-install.md
```

Expected: matches in both design and plan.

- [ ] **Step 3: Commit Task 1**

```bash
git add docs/superpowers/specs/2026-08-27-certified-operator-install-design.md docs/superpowers/plans/2026-08-27-certified-operator-one-step-install.md .superpowers/sdd/2026-08-27-certified-operator-one-step-install/progress.md
git commit -m "docs: pivot certified operators to environment installs"
```

## Task 2: Remove the self-managed installed-release store path

**Repository:** `open-xquant`

**Files:**
- Delete: `src/oxq/operators/installed_store.py`
- Delete: `tests/operators/test_installed_store.py`
- Modify: `src/oxq/operators/resources.py`
- Modify: `tests/operators/test_resources.py`
- Modify: `tests/contracts/test_operator_distribution_schemas.py`
- Modify: `pyproject.toml`
- Modify: `src/oxq/cli/main.py`

**Interfaces:**
- Removes all runtime dependencies on `InstalledReleaseStore`.
- Removes `installed-release-v1` as an active phase-one install contract.
- Keeps `export-certification` and `import-certification` only as evidence tools, not installation tools.

- [ ] **Step 1: Write failing deletion guard tests**

Add to `tests/operators/test_resources.py`:

```python
def test_environment_install_phase_does_not_package_installed_release_schema() -> None:
    from oxq.operators.resources import materialize_operator_install_profile

    with materialize_operator_install_profile() as paths:
        assert "installed_release" not in paths
```

Add to a CLI test:

```python
def test_operator_install_is_guidance_not_package_manager(cli_runner) -> None:
    result = cli_runner.invoke(app, ["operator", "install", "equant-py==1.0.0"])
    assert result.exit_code != 0
    assert "pip install equant-py==1.0.0" in result.output
    assert "oxq operator verify equant-py==1.0.0" in result.output
```

- [ ] **Step 2: Run focused tests and observe failure**

Run:

```bash
uv run pytest tests/operators/test_resources.py tests/cli/test_operator_bundle.py -q
```

Expected: FAIL while installed-release resources or self-managed install behavior still exist.

- [ ] **Step 3: Remove store module and command surface**

Delete `InstalledReleaseStore` and its tests. Remove any CLI path that publishes provider wheels into an `open-xquant` store. If `operator install` exists, make it a non-mutating guidance command that prints:

```text
Install provider package with:
pip install equant-py==1.0.0
Then run:
oxq operator verify equant-py==1.0.0
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/operators/test_resources.py tests/cli/test_operator_bundle.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git rm -f src/oxq/operators/installed_store.py tests/operators/test_installed_store.py
git add src/oxq/operators src/oxq/cli/main.py tests/operators tests/cli pyproject.toml
git commit -m "refactor: drop managed operator install store"
```

## Task 3: Add official environment provider index

**Repository:** `open-xquant`

**Files:**
- Create: `contracts/operator-install/official-environment-providers-v1.json`
- Create: `src/oxq/operators/environment_index.py`
- Create: `tests/operators/test_environment_index.py`
- Modify: `src/oxq/operators/resources.py`
- Modify: `tests/operators/test_resources.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces `CertifiedOperatorRef(operator_id: str, operator_version: str, manifest_path: str, baseline_paths: tuple[str, ...])`.
- Produces `EnvironmentProvider(provider: str, distribution: str, version: str, certification_state: str, operators: tuple[CertifiedOperatorRef, ...], manifest_digests: Mapping[str, str], baseline_digests: Mapping[str, str])`.
- Produces `parse_exact_provider_requirement(value: str) -> tuple[str, str]`.
- Produces `load_environment_provider(provider: str, version: str) -> EnvironmentProvider`.

- [ ] **Step 1: Write failing index tests**

```python
def test_parse_exact_provider_requirement_accepts_only_exact_version() -> None:
    assert parse_exact_provider_requirement("equant-py==1.0.0") == ("equant-py", "1.0.0")
    with pytest.raises(ValueError, match="exact"):
        parse_exact_provider_requirement("equant-py>=1.0.0")
```

```python
def test_official_index_contains_equant_py_100() -> None:
    provider = load_environment_provider("equant-py", "1.0.0")
    assert provider.distribution == "equant-py"
    assert provider.certification_state == "research-certified"
    assert provider.operators
```

- [ ] **Step 2: Run focused tests and observe missing module**

Run:

```bash
uv run pytest tests/operators/test_environment_index.py tests/operators/test_resources.py -q
```

Expected: FAIL because `environment_index.py` and packaged index do not exist.

- [ ] **Step 3: Implement packaged index loading**

Create `official-environment-providers-v1.json` with the initial `equant-py==1.0.0` entry. Use the current real certified operator list when available. If the full list is not yet packaged, include the minimal certified fixture entry `equant.ttr.sma@1.0.0` and make Task 7 replace it with the generated official list.

Implement strict parsing with no ranges, no implicit version, no whitespace-normalized alternate meaning, and canonical provider naming.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/operators/test_environment_index.py tests/operators/test_resources.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add contracts/operator-install/official-environment-providers-v1.json src/oxq/operators/environment_index.py src/oxq/operators/resources.py tests/operators/test_environment_index.py tests/operators/test_resources.py pyproject.toml
git commit -m "feat: add certified environment provider index"
```

## Task 4: Verify installed provider package metadata and bytes

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/environment_provider.py`
- Create: `tests/operators/test_environment_provider.py`
- Modify: `src/oxq/operators/environment_index.py`

**Interfaces:**
- Produces `InstalledEnvironmentProvider`.
- Produces `verify_installed_provider(requirement: str) -> InstalledEnvironmentProvider`.
- Uses `importlib.metadata.distribution(distribution_name)` to locate installed package files.
- Reads only files declared by the official index.

- [ ] **Step 1: Write failing verification tests with fake distributions**

```python
def test_verify_installed_provider_rejects_missing_distribution(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)),
    )
    with pytest.raises(OperatorCertificationError, match="not installed"):
        verify_installed_provider("equant-py==1.0.0")
```

```python
def test_verify_installed_provider_rejects_wrong_version(fake_distribution) -> None:
    fake_distribution.version = "1.0.1"
    with pytest.raises(OperatorCertificationError, match="version"):
        verify_installed_provider("equant-py==1.0.0")
```

```python
def test_verify_installed_provider_rejects_changed_manifest_bytes(fake_distribution) -> None:
    fake_distribution.add_file(
        "compat/open_xquant/manifests/equant.ttr.sma@1.0.0.operator.json",
        b"tampered",
    )
    with pytest.raises(OperatorCertificationError, match="digest"):
        verify_installed_provider("equant-py==1.0.0")
```

- [ ] **Step 2: Run focused tests and observe missing verifier**

Run:

```bash
uv run pytest tests/operators/test_environment_provider.py -q
```

Expected: FAIL because `verify_installed_provider` is missing.

- [ ] **Step 3: Implement distribution verification**

Use `distribution.files` and `distribution.locate_file(package_path)` to find declared manifest and baseline files. Reject missing files, directories, symlinks, and digest mismatches. Return parsed manifest and baseline bytes only after digest verification.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/operators/test_environment_provider.py tests/operators/test_environment_index.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/oxq/operators/environment_provider.py src/oxq/operators/environment_index.py tests/operators/test_environment_provider.py
git commit -m "feat: verify certified environment providers"
```

## Task 5: Add CLI verification and listing

**Repository:** `open-xquant`

**Files:**
- Modify: `src/oxq/cli/main.py`
- Create: `tests/cli/test_operator_environment.py`

**Interfaces:**
- Adds `oxq operator verify <provider>==<version>`.
- Adds or updates `oxq operator list --provider <provider>`.
- `--json` emits stable success and error envelopes.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_operator_verify_prints_verified_provider(cli_runner, fake_verified_provider) -> None:
    result = cli_runner.invoke(app, ["operator", "verify", "equant-py==1.0.0"])
    assert result.exit_code == 0
    assert "equant-py==1.0.0 verified" in result.output
```

```python
def test_operator_verify_json_reports_operator_count(cli_runner, fake_verified_provider) -> None:
    result = cli_runner.invoke(app, ["operator", "verify", "equant-py==1.0.0", "--json"])
    payload = json.loads(result.output)
    assert payload["provider"] == "equant-py"
    assert payload["version"] == "1.0.0"
    assert payload["operator_count"] > 0
```

- [ ] **Step 2: Run focused CLI tests and observe missing commands**

Run:

```bash
uv run pytest tests/cli/test_operator_environment.py -q
```

Expected: FAIL because CLI commands are missing.

- [ ] **Step 3: Implement CLI commands**

Wire commands to `verify_installed_provider()`. For `operator install`, do not mutate the environment; return guidance text with the exact `pip install` and `operator verify` commands.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run pytest tests/cli/test_operator_environment.py tests/cli/test_operator_bundle.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/oxq/cli/main.py tests/cli/test_operator_environment.py tests/cli/test_operator_bundle.py
git commit -m "feat: verify certified provider packages"
```

## Task 6: Resolve verified environment operators for research use

**Repository:** `open-xquant`

**Files:**
- Create: `src/oxq/operators/environment_runtime.py`
- Create: `tests/operators/test_environment_runtime.py`
- Modify: `src/oxq/operators/__init__.py`

**Interfaces:**
- Produces `resolve_environment_operator(operator_id: str, operator_version: str, provider_requirement: str)`.
- The resolver verifies the provider before returning an invocation binding.
- The first phase may import provider code from the installed Python environment.

- [ ] **Step 1: Write failing runtime resolver tests**

```python
def test_resolve_environment_operator_rejects_uncertified_operator(fake_verified_provider) -> None:
    with pytest.raises(OperatorCertificationError, match="not certified"):
        resolve_environment_operator("equant.ttr.not_real", "1.0.0", "equant-py==1.0.0")
```

```python
def test_resolve_environment_operator_returns_callable_binding(fake_verified_provider) -> None:
    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")
    assert binding.operator_id == "equant.ttr.sma"
    assert callable(binding.callable)
```

- [ ] **Step 2: Run focused tests and observe missing runtime**

Run:

```bash
uv run pytest tests/operators/test_environment_runtime.py -q
```

Expected: FAIL because `environment_runtime.py` is missing.

- [ ] **Step 3: Implement verified resolver**

Use the verified manifest implementation module and callable fields. Import only after provider verification succeeds. Reject any manifest whose certification state is not `research-certified`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/operators/test_environment_runtime.py tests/operators/test_environment_provider.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/oxq/operators/environment_runtime.py src/oxq/operators/__init__.py tests/operators/test_environment_runtime.py
git commit -m "feat: resolve verified environment operators"
```

## Task 7: Cross-repository certification acceptance with `equant-py`

**Repository:** `open-xquant` and `equant-py`

**Files:**
- Modify in `equant-py`: packaging config so certified manifests and baselines are included in the wheel.
- Modify in `open-xquant`: official environment provider index with the final certified operator/digest list.
- Create or update: cross-repository acceptance notes under each repository's docs.

**Interfaces:**
- Confirms an actual `equant-py` wheel can be installed into a temporary environment and verified by `open-xquant`.

- [ ] **Step 1: Build candidate `equant-py` wheel**

Run in `equant-py`:

```bash
uv build --wheel
```

Expected: wheel exists under `dist/`.

- [ ] **Step 2: Install into a temporary verification environment**

Run in a temp directory:

```bash
python -m venv .venv
.venv/bin/python -m pip install /path/to/open-xquant /path/to/equant-py/dist/equant_py-1.0.0-*.whl
.venv/bin/oxq operator verify equant-py==1.0.0 --json
```

Expected: JSON reports `research-certified` and the certified operator count.

- [ ] **Step 3: Run operator listing**

Run:

```bash
.venv/bin/oxq operator list --provider equant-py --json
```

Expected: JSON lists the certified operators and no uncertified operators.

- [ ] **Step 4: Commit cross-repository updates**

Commit `equant-py` packaging updates in `equant-py`.

Commit final index and acceptance notes in `open-xquant`.

## Self-review checklist

- The plan no longer asks `open-xquant` to store provider wheels.
- The plan no longer depends on `OPEN_XQUANT_OPERATOR_HOME`.
- The user-visible path starts with normal Python installation.
- Verification still checks certified bytes and exact versions.
- Runtime loading is allowed only after verification.
- Bundle export/import remains evidence tooling, not package installation.
