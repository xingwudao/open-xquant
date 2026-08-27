"""Runtime resolver coverage for verified environment operators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import oxq.operators.environment_runtime as environment_runtime
from oxq.operators.environment_index import CertifiedOperatorRef, EnvironmentProvider
from oxq.operators.environment_provider import InstalledEnvironmentProvider
from oxq.operators.environment_runtime import resolve_environment_operator
from oxq.operators.errors import OperatorCertificationError


@pytest.fixture
def fake_verified_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> InstalledEnvironmentProvider:
    module_root = tmp_path / "site-packages"
    module_root.mkdir()
    (module_root / "ettr.py").write_text(
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


def test_resolve_environment_operator_returns_callable_binding(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    binding = resolve_environment_operator("equant.ttr.sma", "1.0.0", "equant-py==1.0.0")

    assert binding.operator_id == "equant.ttr.sma"
    assert callable(binding.callable)
