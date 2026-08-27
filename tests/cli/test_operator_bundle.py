"""CLI coverage for portable certification bundle export and import."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from click.testing import CliRunner

import oxq.operators.baseline_runner as baseline_runner
from oxq.operators import runtime_protocol
from oxq.cli.main import main
from oxq.operators.bundle import export_certification_bundle
from oxq.operators.certification import certify_provider
from oxq.operators.models import CertificationTarget
from oxq.operators.registry import publish_certification
from oxq.operators.submission import load_provider_submission
from tests.operators.test_baseline_runner import _write_certifiable_provider

TARGET = "cp312-cp312-macosx_14_0_arm64"


def _patch_fixture_runtime() -> None:
    runtime_paths = [
        path for path in sys.path if "site-packages" in Path(path).parts
    ]

    def run_fixture_request(
        request: Mapping[str, object],
        wheel_snapshots: Sequence[str | Path],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        return runtime_protocol.run_exact_wheel_request(
            request,
            wheel_snapshots,
            timeout_seconds=timeout_seconds,
            _test_runtime_paths=runtime_paths,
        )

    baseline_runner.run_exact_wheel_request = run_fixture_request


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    _patch_fixture_runtime()
    fixture = _write_certifiable_provider(tmp_path / "fixture", expected=[None, None, 2.0])
    submission = load_provider_submission(fixture.path, fixture.submission_commit, fixture.artifact_dir)
    try:
        certified = certify_provider(submission)
        registry = tmp_path / "registry"
        publish_certification(certified, registry, target=CertificationTarget.parse(TARGET))
    finally:
        submission.__exit__(None, None, None)
    return registry, fixture.path / "compat/open_xquant", fixture.path


def test_export_emits_sorted_json_and_import_requires_trust(tmp_path: Path) -> None:
    registry, compat, _ = _fixture(tmp_path)
    output = tmp_path / "bundle.zip"
    runner = CliRunner()

    exported = runner.invoke(
        main,
        [
            "operator", "export-certification", "--provider", "equant-py", "--release", "1.0.0",
            "--registry-dir", str(registry), "--manifest-dir", str(compat / "manifests"),
            "--baseline-file", str(compat / "numerical_baselines/technical-v1.json"),
            "--target", TARGET, "--output", str(output), "--json",
        ],
    )

    assert exported.exit_code == 0, exported.output
    assert json.loads(exported.output) == {
        "bundle": str(output.resolve()), "operator_count": 1, "provider": "equant-py",
        "release": "1.0.0", "status": "research-certified", "target": TARGET,
    }

    untrusted = runner.invoke(
        main,
        ["operator", "import-certification", "--bundle", str(output), "--output-dir", str(tmp_path / "destination"), "--json"],
    )
    assert untrusted.exit_code == 1
    assert json.loads(untrusted.output) == {
        "code": "bundle_trust_required", "message": "--trust-unsigned-bundle is required to import a bundle",
        "stage": "trust", "status": "fail",
    }


def test_import_emits_human_output_and_stable_success_exit_code(tmp_path: Path) -> None:
    registry, compat, _ = _fixture(tmp_path)
    bundle = export_certification_bundle(
        provider="equant-py", release="1.0.0", registry_dir=registry,
        manifest_dir=compat / "manifests", baseline_files=[compat / "numerical_baselines/technical-v1.json"],
        target=CertificationTarget.parse(TARGET), output_path=tmp_path / "bundle.zip",
    )

    result = CliRunner().invoke(
        main,
        [
            "operator", "import-certification", "--bundle", str(bundle.bundle_path),
            "--output-dir", str(tmp_path / "destination"), "--trust-unsigned-bundle",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Status: research-certified" in result.output
    assert "Provider: equant-py" in result.output


def test_operator_install_is_guidance_not_package_manager() -> None:
    result = CliRunner().invoke(main, ["operator", "install", "equant-py==1.0.0"])

    assert result.exit_code != 0
    assert "pip install equant-py==1.0.0" in result.output
    assert "oxq operator verify equant-py==1.0.0" in result.output
