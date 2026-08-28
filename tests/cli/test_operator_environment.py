"""CLI coverage for certified operator environment providers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import oxq.operators.environment_provider as environment_provider
from oxq.cli.main import main
from oxq.operators.environment_index import CertifiedOperatorRef, EnvironmentProvider
from oxq.operators.environment_provider import InstalledEnvironmentProvider, VerifiedRuntimeFile


@pytest.fixture
def fake_verified_provider(monkeypatch: pytest.MonkeyPatch) -> InstalledEnvironmentProvider:
    runtime_file = Path(__file__)
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
            "ettr.py": "sha256:" + "c" * 64,
        },
    )
    installed = InstalledEnvironmentProvider(
        provider=provider,
        manifests={
            "manifests/equant.ttr.sma.operator.json": {
                "operator_id": "equant.ttr.sma",
                "operator_version": "1.0.0",
            },
        },
        baselines={"numerical_baselines/equant.ttr.sma.json": b'{"cases":[]}\n'},
        runtime_files={
            "ettr.py": VerifiedRuntimeFile(
                package_path="ettr.py",
                path=runtime_file,
                digest="sha256:" + "c" * 64,
            ),
        },
    )
    monkeypatch.setattr(
        environment_provider,
        "verify_installed_provider",
        lambda requirement: installed,
    )
    return installed


def test_operator_verify_prints_verified_provider(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    result = CliRunner().invoke(main, ["operator", "verify", "equant-py==1.0.0"])

    assert result.exit_code == 0, result.output
    assert "equant-py==1.0.0 verified" in result.output
    assert "Operators: 1" in result.output


def test_operator_verify_json_reports_operator_count(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    result = CliRunner().invoke(
        main,
        ["operator", "verify", "equant-py==1.0.0", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "equant-py"
    assert payload["version"] == "1.0.0"
    assert payload["operator_count"] > 0


def test_operator_list_provider_prints_verified_provider(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    result = CliRunner().invoke(main, ["operator", "list", "--provider", "equant-py"])

    assert result.exit_code == 0, result.output
    assert "Provider: equant-py" in result.output
    assert "Version: 1.0.0" in result.output
    assert "Operators: 1" in result.output


def test_operator_list_provider_json_is_stable(
    fake_verified_provider: InstalledEnvironmentProvider,
) -> None:
    del fake_verified_provider

    result = CliRunner().invoke(
        main,
        ["operator", "list", "--provider", "equant-py", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "operator_count": 1,
        "operators": [{"operator_id": "equant.ttr.sma", "operator_version": "1.0.0"}],
        "provider": "equant-py",
        "status": "research-certified",
        "version": "1.0.0",
    }


@pytest.mark.parametrize(
    "legacy_command",
    ["export-certification", "import-certification"],
)
def test_operator_legacy_bundle_commands_are_not_exposed(
    legacy_command: str,
) -> None:
    result = CliRunner().invoke(main, ["operator", legacy_command, "--help"])

    assert result.exit_code == 2
    assert "No such command" in result.output


def test_operator_install_prints_distribution_closure() -> None:
    result = CliRunner().invoke(main, ["operator", "install", "equant-py==1.0.0"])

    assert result.exit_code == 1
    assert "Install provider distributions with:" in result.output
    assert "pip install equant-core==1.0.0 equant-ttr==1.0.0" in result.output
    assert "oxq operator verify equant-py==1.0.0" in result.output
