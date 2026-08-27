"""Runtime resolver coverage for verified environment operators."""

from __future__ import annotations

import sys
import threading
import types
from importlib import _bootstrap_external
from importlib.util import cache_from_source
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
    environment_runtime._TRUSTED_RUNTIME_MODULES.clear()

    provider = EnvironmentProvider(
        provider="equant-py",
        distribution="equant-py",
        distributions=("equant-py",),
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


def test_resolve_environment_operator_allows_repeated_verified_resolution(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")
    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert binding.callable({"verified": True}) == {"verified": True}


def test_resolve_environment_operator_reloads_mutated_cached_callable(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider
    resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")
    loaded = sys.modules["ettr"]
    exec(
        "def sma(frame, **parameters):\n"
        "    return 'mutated-trusted-module-callable'\n",
        loaded.__dict__,
    )

    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert binding.callable({"verified": True}) == {"verified": True}


def test_resolve_environment_operator_reloads_mutated_callable_code(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider
    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    def mutated(frame, **parameters):
        return "mutated-code-object"

    binding.callable.__code__ = mutated.__code__

    resolved = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert resolved.callable({"verified": True}) == {"verified": True}


def test_resolve_environment_operator_serializes_first_verified_import(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    verified = next(iter(fake_verified_provider.runtime_files.values())).path
    verified.write_text(
        "import time\n"
        "time.sleep(0.05)\n"
        "def sma(frame, **parameters):\n"
        "    return frame\n",
        encoding="utf-8",
    )
    digest = _digest(verified.read_bytes())
    object.__setattr__(fake_verified_provider.provider, "runtime_digests", {"ettr.py": digest})
    fake_verified_provider.runtime_files["ettr.py"] = VerifiedRuntimeFile(
        package_path="ettr.py",
        path=verified,
        digest=digest,
    )
    sys.modules.pop("ettr", None)
    environment_runtime._TRUSTED_RUNTIME_MODULES.clear()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def resolve() -> None:
        try:
            barrier.wait(timeout=1)
            binding = resolve_environment_operator(
                "equant.ttr.sma",
                "1.0.0",
                "equant-py==1.0.0",
            )
            assert binding.callable({"verified": True}) == {"verified": True}
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=resolve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []


def test_resolve_environment_operator_ignores_unverified_bytecode_cache(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    source = next(iter(fake_verified_provider.runtime_files.values())).path
    source.write_text(
        "def sma(frame, **parameters):\n"
        "    return 'verified-source'\n",
        encoding="utf-8",
    )
    raw = source.read_bytes()
    digest = _digest(raw)
    object.__setattr__(fake_verified_provider.provider, "runtime_digests", {"ettr.py": digest})
    fake_verified_provider.runtime_files["ettr.py"] = VerifiedRuntimeFile(
        package_path="ettr.py",
        path=source,
        digest=digest,
    )
    stat = source.stat()
    malicious_code = compile(
        "def sma(frame, **parameters):\n"
        "    return 'malicious-pyc'\n",
        str(source),
        "exec",
    )
    pyc_path = Path(cache_from_source(str(source)))
    pyc_path.parent.mkdir(parents=True, exist_ok=True)
    pyc_path.write_bytes(
        _bootstrap_external._code_to_timestamp_pyc(
            malicious_code,
            int(stat.st_mtime),
            stat.st_size,
        )
    )
    sys.modules.pop("ettr", None)

    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert binding.callable(None) == "verified-source"


def test_resolve_environment_operator_ignores_shadowed_runtime_module_before_import(
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

    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert binding.callable({"verified": True}) == {"verified": True}
    assert "ettr" in sys.modules


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


def test_resolve_environment_operator_rejects_preloaded_reexport_owner_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_root = tmp_path / "site-packages"
    package = module_root / "ettr"
    package.mkdir(parents=True)
    init_path = package / "__init__.py"
    trend_path = package / "trend.py"
    init_path.write_text("from .trend import sma\n", encoding="utf-8")
    trend_path.write_text(
        "def sma(frame, **parameters):\n"
        "    return 'verified-trend'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_root))
    sys.modules.pop("ettr", None)
    malicious = types.ModuleType("ettr.trend")
    malicious.__file__ = str(trend_path)
    malicious.sma = lambda frame, **parameters: "malicious-preload"
    monkeypatch.setitem(sys.modules, "ettr.trend", malicious)
    provider = EnvironmentProvider(
        provider="equant-py",
        distribution="equant-py",
        distributions=("equant-py",),
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
        manifest_digests={"manifests/equant.ttr.sma.operator.json": "sha256:" + "a" * 64},
        baseline_digests={"numerical_baselines/equant.ttr.sma.json": "sha256:" + "b" * 64},
        runtime_digests={
            "ettr/__init__.py": _digest(init_path.read_bytes()),
            "ettr/trend.py": _digest(trend_path.read_bytes()),
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
            "ettr/__init__.py": VerifiedRuntimeFile(
                package_path="ettr/__init__.py",
                path=init_path,
                digest=_digest(init_path.read_bytes()),
            ),
            "ettr/trend.py": VerifiedRuntimeFile(
                package_path="ettr/trend.py",
                path=trend_path,
                digest=_digest(trend_path.read_bytes()),
            ),
        },
    )
    monkeypatch.setattr(
        environment_runtime,
        "verify_installed_provider",
        lambda requirement: installed,
    )

    with pytest.raises(OperatorCertificationError) as caught:
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert caught.value.code == "environment_operator_module_preloaded"


def test_resolve_environment_operator_rejects_undeclared_provider_namespace_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_root = tmp_path / "site-packages"
    package = module_root / "ettr"
    package.mkdir(parents=True)
    init_path = package / "__init__.py"
    helper_path = package / "helper.py"
    init_path.write_text(
        "from .helper import value\n"
        "def sma(frame, **parameters):\n"
        "    return value\n",
        encoding="utf-8",
    )
    helper_path.write_text("value = 'undeclared-helper'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(module_root))
    sys.modules.pop("ettr", None)
    sys.modules.pop("ettr.helper", None)
    environment_runtime._TRUSTED_RUNTIME_MODULES.clear()
    provider = EnvironmentProvider(
        provider="equant-py",
        distribution="equant-ttr",
        distributions=("equant-ttr",),
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
        manifest_digests={"manifests/equant.ttr.sma.operator.json": "sha256:" + "a" * 64},
        baseline_digests={"numerical_baselines/equant.ttr.sma.json": "sha256:" + "b" * 64},
        runtime_digests={
            "ettr/__init__.py": _digest(init_path.read_bytes()),
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
            "ettr/__init__.py": VerifiedRuntimeFile(
                package_path="ettr/__init__.py",
                path=init_path,
                digest=_digest(init_path.read_bytes()),
            ),
        },
    )
    monkeypatch.setattr(
        environment_runtime,
        "verify_installed_provider",
        lambda requirement: installed,
    )

    with pytest.raises(OperatorCertificationError) as caught:
        resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert caught.value.code == "environment_operator_module_unavailable"


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
