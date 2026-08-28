"""Installed environment provider verification tests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path

import pytest

import oxq.operators.environment_provider as environment_provider
from oxq.operators.environment_index import CertifiedOperatorRef, EnvironmentProvider
from oxq.operators.environment_provider import verify_installed_provider
from oxq.operators.errors import OperatorCertificationError

MANIFEST_PATH = "manifests/equant.ttr.sma.operator.json"
BASELINE_PATH = "numerical_baselines/equant.ttr.sma.json"
MANIFEST_BYTES = b'{"operator_id":"equant.ttr.sma","version":"1.0.0"}\n'
CONFLICTING_MANIFEST_BYTES = (
    b'{"certification_state":"contract-valid","operator_id":"equant.ttr.sma","version":"1.0.0"}\n'
)
BASELINE_BYTES = b'{"cases":[]}\n'
RUNTIME_PATH = "ettr/__init__.py"
RUNTIME_BYTES = b"def sma(frame, **parameters):\n    return frame\n"


class FakeDistribution:
    def __init__(self, root: Path, version: str = "1.0.0") -> None:
        self.root = root
        self.version = version
        self._files: list[str] = []

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self._files)

    def add_file(self, path: str, content: bytes) -> Path:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self._files.append(path)
        return target

    def add_directory(self, path: str) -> Path:
        target = self.root / path
        target.mkdir(parents=True, exist_ok=True)
        self._files.append(path)
        return target

    def add_symlink(self, path: str, target: str) -> Path:
        link = self.root / path
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)
        self._files.append(path)
        return link

    def locate_file(self, path: object) -> Path:
        return self.root / str(path)


@pytest.fixture
def official_provider(monkeypatch: pytest.MonkeyPatch) -> EnvironmentProvider:
    provider = EnvironmentProvider(
        provider="equant-py",
        distribution="equant-core",
        distributions=("equant-core", "equant-ttr"),
        version="1.0.0",
        certification_state="research-certified",
        operators=(
            CertifiedOperatorRef(
                operator_id="equant.ttr.sma",
                operator_version="1.0.0",
                manifest_path=MANIFEST_PATH,
                baseline_paths=(BASELINE_PATH,),
            ),
        ),
        manifest_digests={MANIFEST_PATH: _digest(MANIFEST_BYTES)},
        baseline_digests={BASELINE_PATH: _digest(BASELINE_BYTES)},
        runtime_digests={RUNTIME_PATH: _digest(RUNTIME_BYTES)},
    )
    monkeypatch.setattr(
        environment_provider,
        "load_environment_provider",
        lambda provider_name, version: provider,
    )
    return provider


@pytest.fixture
def fake_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_provider: EnvironmentProvider,
) -> FakeDistribution:
    del official_provider
    distribution = FakeDistribution(tmp_path)
    distribution.add_file(MANIFEST_PATH, MANIFEST_BYTES)
    distribution.add_file(BASELINE_PATH, BASELINE_BYTES)
    distribution.add_file(RUNTIME_PATH, RUNTIME_BYTES)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    return distribution


def test_verify_installed_provider_rejects_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
    official_provider: EnvironmentProvider,
) -> None:
    del official_provider
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)),
    )

    with pytest.raises(OperatorCertificationError, match="not installed"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_rejects_wrong_version(fake_distribution: FakeDistribution) -> None:
    fake_distribution.version = "1.0.1"

    with pytest.raises(OperatorCertificationError, match="version"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_rejects_changed_manifest_bytes(
    fake_distribution: FakeDistribution,
) -> None:
    fake_distribution.add_file(MANIFEST_PATH, b"tampered")

    with pytest.raises(OperatorCertificationError, match="digest"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_rejects_missing_declared_file(
    fake_distribution: FakeDistribution,
) -> None:
    fake_distribution._files.remove(BASELINE_PATH)

    with pytest.raises(OperatorCertificationError, match="missing"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_rejects_changed_runtime_bytes(
    fake_distribution: FakeDistribution,
) -> None:
    fake_distribution.add_file(RUNTIME_PATH, b"tampered")

    with pytest.raises(OperatorCertificationError, match="digest"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_rejects_missing_runtime_file(
    fake_distribution: FakeDistribution,
) -> None:
    fake_distribution._files.remove(RUNTIME_PATH)

    with pytest.raises(OperatorCertificationError, match="missing"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_rejects_symlinked_parent_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_provider: EnvironmentProvider,
) -> None:
    del official_provider
    external = tmp_path / "external"
    external.mkdir()
    (external / "__init__.py").write_bytes(RUNTIME_BYTES)
    distribution = FakeDistribution(tmp_path / "site-packages")
    distribution.add_file(MANIFEST_PATH, MANIFEST_BYTES)
    distribution.add_file(BASELINE_PATH, BASELINE_BYTES)
    runtime_parent = distribution.root / "ettr"
    runtime_parent.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(external, runtime_parent)
    distribution._files.append(RUNTIME_PATH)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)

    with pytest.raises(OperatorCertificationError, match="regular file"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_allows_symlink_above_distribution_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_provider: EnvironmentProvider,
) -> None:
    del official_provider
    real_root = tmp_path / "real-site-packages"
    link_root = tmp_path / "linked-site-packages"
    distribution = FakeDistribution(link_root)
    real_root.mkdir()
    os.symlink(real_root, link_root)
    distribution.add_file(MANIFEST_PATH, MANIFEST_BYTES)
    distribution.add_file(BASELINE_PATH, BASELINE_BYTES)
    distribution.add_file(RUNTIME_PATH, RUNTIME_BYTES)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)

    installed = verify_installed_provider("equant-py==1.0.0")

    assert installed.runtime_files[RUNTIME_PATH].path == real_root / RUNTIME_PATH


def test_verify_installed_provider_reads_declared_files_from_distribution_closure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_provider: EnvironmentProvider,
) -> None:
    object.__setattr__(official_provider, "distribution", "equant-core")
    object.__setattr__(official_provider, "distributions", ("equant-core", "equant-ttr"))
    core_distribution = FakeDistribution(tmp_path / "core")
    core_distribution.add_file(MANIFEST_PATH, MANIFEST_BYTES)
    core_distribution.add_file(BASELINE_PATH, BASELINE_BYTES)
    ttr_distribution = FakeDistribution(tmp_path / "ttr")
    ttr_distribution.add_file(RUNTIME_PATH, RUNTIME_BYTES)

    def distribution(name: str) -> FakeDistribution:
        if name == "equant-core":
            return core_distribution
        if name == "equant-ttr":
            return ttr_distribution
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)

    installed = verify_installed_provider("equant-py==1.0.0")

    assert installed.manifests[MANIFEST_PATH]["operator_id"] == "equant.ttr.sma"
    assert installed.baselines[BASELINE_PATH] == BASELINE_BYTES
    assert installed.runtime_files[RUNTIME_PATH].path == ttr_distribution.root / RUNTIME_PATH


def test_verify_installed_provider_reads_shared_baseline_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_provider: EnvironmentProvider,
) -> None:
    object.__setattr__(
        official_provider,
        "operators",
        (
            CertifiedOperatorRef(
                operator_id="equant.ttr.sma",
                operator_version="1.0.0",
                manifest_path=MANIFEST_PATH,
                baseline_paths=(BASELINE_PATH,),
            ),
            CertifiedOperatorRef(
                operator_id="equant.ttr.ema",
                operator_version="1.0.0",
                manifest_path=MANIFEST_PATH,
                baseline_paths=(BASELINE_PATH,),
            ),
        ),
    )
    core_distribution = FakeDistribution(tmp_path / "core")
    core_distribution.add_file(MANIFEST_PATH, MANIFEST_BYTES)
    baseline_path = core_distribution.add_file(BASELINE_PATH, BASELINE_BYTES)
    ttr_distribution = FakeDistribution(tmp_path / "ttr")
    ttr_distribution.add_file(RUNTIME_PATH, RUNTIME_BYTES)
    baseline_reads = 0
    real_read_bytes = Path.read_bytes

    def observed_read_bytes(path: Path) -> bytes:
        nonlocal baseline_reads
        if path == baseline_path:
            baseline_reads += 1
        return real_read_bytes(path)

    def distribution(name: str) -> FakeDistribution:
        if name == "equant-core":
            return core_distribution
        if name == "equant-ttr":
            return ttr_distribution
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(Path, "read_bytes", observed_read_bytes)

    installed = verify_installed_provider("equant-py==1.0.0")

    assert installed.baselines[BASELINE_PATH] == BASELINE_BYTES
    assert baseline_reads == 1


@pytest.mark.parametrize("artifact_kind", ["directory", "symlink"])
def test_verify_installed_provider_rejects_non_regular_declared_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    official_provider: EnvironmentProvider,
    artifact_kind: str,
) -> None:
    del official_provider
    distribution = FakeDistribution(tmp_path)
    distribution.add_file(MANIFEST_PATH, MANIFEST_BYTES)
    distribution.add_file(RUNTIME_PATH, RUNTIME_BYTES)
    if artifact_kind == "directory":
        distribution.add_directory(BASELINE_PATH)
    else:
        distribution.add_symlink(BASELINE_PATH, MANIFEST_PATH)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)

    with pytest.raises(OperatorCertificationError, match="regular file"):
        verify_installed_provider("equant-py==1.0.0")


def test_verify_installed_provider_returns_verified_artifacts(
    fake_distribution: FakeDistribution,
) -> None:
    fake_distribution.add_symlink("undeclared-link", MANIFEST_PATH)

    installed = verify_installed_provider("equant-py==1.0.0")

    assert installed.provider.provider == "equant-py"
    assert installed.manifests[MANIFEST_PATH] == {
        "certification_state": "research-certified",
        "operator_id": "equant.ttr.sma",
        "version": "1.0.0",
    }
    assert installed.baselines[BASELINE_PATH] == BASELINE_BYTES
    assert installed.runtime_files[RUNTIME_PATH].digest == _digest(RUNTIME_BYTES)
    assert installed.runtime_files[RUNTIME_PATH].path == fake_distribution.root / RUNTIME_PATH


def test_verify_installed_provider_rejects_conflicting_manifest_state(
    fake_distribution: FakeDistribution,
    official_provider: EnvironmentProvider,
) -> None:
    fake_distribution.add_file(MANIFEST_PATH, CONFLICTING_MANIFEST_BYTES)
    object.__setattr__(
        official_provider,
        "manifest_digests",
        {MANIFEST_PATH: _digest(CONFLICTING_MANIFEST_BYTES)},
    )

    with pytest.raises(OperatorCertificationError, match="certification state"):
        verify_installed_provider("equant-py==1.0.0")


def _digest(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"
