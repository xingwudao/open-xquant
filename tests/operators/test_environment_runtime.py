"""Runtime resolver coverage for verified environment operators."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import oxq.operators.environment_runtime as environment_runtime
from oxq.operators.environment_index import CertifiedOperatorRef, EnvironmentProvider
from oxq.operators.environment_provider import InstalledEnvironmentProvider, VerifiedRuntimeFile
from oxq.operators.environment_runtime import resolve_environment_operator
from oxq.operators.errors import OperatorCertificationError


@pytest.fixture
def fake_verified_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> InstalledEnvironmentProvider:
    module_root = tmp_path / "site-packages"
    module_root.mkdir()
    verified_ettr = module_root / "ettr.py"
    verified_ettr.write_text(
        "def sma(frame, **parameters):\n"
        "    return frame\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_root))
    sys.modules.pop("ettr", None)

    provider = EnvironmentProvider(
        provider="equant-py",
        distribution="equant-py",
        version="1.0.0",
        certification_state="research-certified",
        operators=(
            CertifiedOperatorRef(
                operator_id="equant.ttr.sma",
                operator_version="1.0.0",
                manifest_path="manifests/equant.ttr.sma.operator.json",
                baseline_paths=("numerical_baselines/equant.ttr.sma.json",),
            ),
        ),
        manifest_digests={
            "manifests/equant.ttr.sma.operator.json": "sha256:" + "a" * 64,
        },
        baseline_digests={
            "numerical_baselines/equant.ttr.sma.json": "sha256:" + "b" * 64,
        },
        runtime_digests={
            "ettr.py": _digest(verified_ettr.read_bytes()),
        },
    )
    installed = InstalledEnvironmentProvider(
        provider=provider,
        manifests={
            "manifests/equant.ttr.sma.operator.json": {
                "operator_id": "equant.ttr.sma",
                "operator_version": "1.0.0",
                "certification_state": "research-certified",
                "module": "ettr",
                "callable": "sma",
            },
        },
        baselines={"numerical_baselines/equant.ttr.sma.json": b'{"cases":[]}\n'},
        runtime_files={
            "ettr.py": VerifiedRuntimeFile(
                package_path="ettr.py",
                path=verified_ettr,
                digest=_digest(verified_ettr.read_bytes()),
            ),
        },
    )
    monkeypatch.setattr(
        environment_runtime,
        "verify_installed_provider",
        lambda requirement: installed,
    )
    return installed


def test_resolve_environment_operator_rejects_uncertified_operator(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    with pytest.raises(OperatorCertificationError, match="not certified"):
        resolve_environment_operator("equant.ttr.not_real", "1.0.0", "equant-py==1.0.0")


def test_resolve_environment_operator_rejects_missing_manifest_certification_state(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    manifest = fake_verified_provider.manifests["manifests/equant.ttr.sma.operator.json"]
    manifest.pop("certification_state")

    with pytest.raises(OperatorCertificationError, match="not research-certified"):
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")


def test_resolve_environment_operator_rejects_non_research_certified_manifest(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    manifest = fake_verified_provider.manifests["manifests/equant.ttr.sma.operator.json"]
    manifest["certification_state"] = "contract-valid"

    with pytest.raises(OperatorCertificationError, match="not research-certified"):
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")


def test_resolve_environment_operator_wraps_missing_implementation_module(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    manifest = fake_verified_provider.manifests["manifests/equant.ttr.sma.operator.json"]
    manifest["module"] = "missing_certified_provider_module"

    with pytest.raises(OperatorCertificationError) as caught:
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert caught.value.code == "environment_operator_module_unavailable"
    assert caught.value.stage == "environment_runtime"


def test_resolve_environment_operator_returns_callable_binding(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert binding.operator_id == "equant.ttr.sma"
    assert callable(binding.callable)


def test_resolve_environment_operator_rejects_shadowed_runtime_module_before_import(
    fake_verified_provider: InstalledEnvironmentProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del fake_verified_provider
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    (shadow_root / "ettr.py").write_text(
        "raise RuntimeError('shadow module executed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(shadow_root))
    sys.modules.pop("ettr", None)

    with pytest.raises(OperatorCertificationError) as caught:
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert caught.value.code == "environment_operator_module_unverified"
    assert "ettr" not in sys.modules


def test_resolve_environment_operator_rejects_preloaded_provider_module(
    fake_verified_provider: InstalledEnvironmentProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_verified_provider
    module = types.ModuleType("ettr")
    module.__file__ = str(Path(__file__))
    module.sma = lambda frame, **parameters: "not verified"
    monkeypatch.setitem(sys.modules, "ettr", module)

    with pytest.raises(OperatorCertificationError) as caught:
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert caught.value.code == "environment_operator_module_preloaded"


def test_resolve_environment_operator_wraps_provider_import_runtime_failure(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    verified = next(iter(fake_verified_provider.runtime_files.values())).path
    raw = b"raise RuntimeError('broken runtime dependency')\n"
    verified.write_bytes(raw)
    digest = _digest(raw)
    object.__setattr__(fake_verified_provider.provider, "runtime_digests", {"ettr.py": digest})
    fake_verified_provider.runtime_files["ettr.py"] = VerifiedRuntimeFile(
        package_path="ettr.py",
        path=verified,
        digest=digest,
    )
    sys.modules.pop("ettr", None)

    with pytest.raises(OperatorCertificationError) as caught:
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert caught.value.code == "environment_operator_module_unavailable"
    assert caught.value.stage == "environment_runtime"


def _digest(raw: bytes) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(raw).hexdigest()}"
