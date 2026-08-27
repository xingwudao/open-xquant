"""Installed-wheel smoke coverage for operator certification resources."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from tests.operators.test_baseline_runner import _write_certifiable_provider

_SMOKE_SCRIPT = r"""
import hashlib
import json
import sys
from pathlib import Path

import oxq
from click.testing import CliRunner
from oxq.cli.main import main
from oxq.operators.registry import CertificationRegistry
from oxq.operators.resources import (
    materialize_certification_profile,
    materialize_contract_surface,
    materialize_operator_distribution_profile,
    materialize_operator_install_profile,
)

EXPECTED_DIGESTS = {
    "quant_panel_schema": "fd6fcd7f3102cdd63913644f87a154a22713c0286a6e9e1cc16e84ca6b283a9c",
    "operator_manifest_schema": "b50b15c446d8940a358ff32c31e191407a7841ab57a30f2fcbafb9a15b9f8793",
    "operator_binding_schema": "1d0e3ed12acde2a2d0c1fe2309f9a090ea7b0f8193bc0f3f6fd659c178047de6",
    "reference_validator": "33528a97f6405809ead8a9f542c1c2dae9d89cb6c966943def6ca80097e8f67a",
    "provider_catalog": "701ff89a33dd71cb7c2f019904ce2ffcc203237734d1719a6188e386b920689c",
    "candidate_build": "a289245a7e67a77fdb597d0c74835bfdd622ddb2df4f97b58fe7c84ad6c6828e",
    "numerical_baseline": "36b524b5d9df67b7bfa78882f6606815778d0edeb1a12b6778fd9fd9f4c11219",
    "certification_record": "a696a76b0b1d902067b8735ba797a3962ccb369b0e3eb2104648e7234c9ea2cd",
    "certification_record_v2": "756e46b85a58832f7df3e2d258a05469fcee277b15e29043897d641b62031020",
    "certification_bundle_manifest": "705874b33e075c8cf26648a3eecb31bb221ac6ae3b18cfd2e7c1c786f05dce08",
    "operator_release": "3ce811095604ff16cd3fe7d8fb81a9056578197d88600fce73678be9a0848f8d",
    "installed_release": "73619350daff45e27db5d1bd9e546f8adb4e932ec5153b4e37bf803b63658cff",
    "runtime_protocol": "0b65c7adba6497463c4143cde5b0786ae001e316dec78a62cacad42d77acd239",
    "official_providers": "95494c6e56e7b6a611019cacdba2497fd511c731b194adf4ad1084000345c626",
}

provider_repo = Path(sys.argv[1])
provider_commit = sys.argv[2]
artifact_dir = Path(sys.argv[3])
installed_environment = Path(sys.argv[4]).resolve()
checkout = Path(sys.argv[5]).resolve()
output_dir = Path(sys.argv[6]).resolve()
module_path = Path(oxq.__file__).resolve()
assert module_path.is_relative_to(installed_environment), module_path
assert not module_path.is_relative_to(checkout), module_path

with materialize_contract_surface() as surface:
    with materialize_certification_profile() as profile:
        with materialize_operator_distribution_profile() as distribution:
            with materialize_operator_install_profile() as install:
                paths = {**surface, **profile, **distribution, **install}
    actual = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
assert actual == EXPECTED_DIGESTS

completed = CliRunner().invoke(
    main,
    [
        "operator",
        "certify-provider",
        "--provider-repo",
        str(provider_repo),
        "--provider-commit",
        provider_commit,
        "--artifact-dir",
        str(artifact_dir),
        "--output-dir",
        str(output_dir),
        "--trust-provider-code",
        "--json",
    ],
)
assert completed.exit_code == 0, (completed.output, completed.exception)
payload = json.loads(completed.output)
binding = CertificationRegistry(output_dir).get("equant.ttr.sma", "1.0.0")
assert binding is not None
result = {
    "cli_status": payload["status"],
    "module": str(module_path),
    "operator_count": payload["operator_count"],
    "output": payload["output"],
    "provider": payload["provider"],
    "release": payload["release"],
    "resource_count": len(actual),
    "state": binding["certification_state"],
}
print(json.dumps(result, sort_keys=True))
"""


def test_installed_wheel_certifies_without_source_checkout(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    fixture = _write_certifiable_provider(
        tmp_path / "provider-fixture",
        expected=[None, None, 2.0],
    )
    wheel_directory = tmp_path / "open-xquant-wheel"
    wheel_directory.mkdir()
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(wheel_directory),
            "--offline",
        ],
        cwd=repository_root,
        check=True,
        text=True,
        capture_output=True,
    )
    wheel = next(wheel_directory.glob("open_xquant-*.whl"))

    environment = tmp_path / "installed-environment"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(environment),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    python = environment / "bin" / "python"
    env = os.environ.copy()
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    # Nested venvs do not inherit their parent venv's packages through
    # --system-site-packages, so expose only the already verified test runtime.
    current_site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    installed_site_packages = subprocess.run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    ).stdout.strip()
    (Path(installed_site_packages) / "verified-test-runtime.pth").write_text(
        str(current_site_packages) + "\n",
        encoding="utf-8",
    )

    outside_checkout = tmp_path / "outside-checkout"
    outside_checkout.mkdir()
    output_dir = outside_checkout / "certifications"
    script = outside_checkout / "installed_smoke.py"
    script.write_text(_SMOKE_SCRIPT, encoding="utf-8")
    completed = subprocess.run(
        [
            str(python),
            str(script),
            str(fixture.path),
            fixture.submission_commit,
            str(fixture.artifact_dir),
            str(environment),
            str(repository_root),
            str(output_dir),
        ],
        cwd=outside_checkout,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert json.loads(completed.stdout) == {
        "cli_status": "research-certified",
        "module": str(
            (Path(installed_site_packages) / "oxq" / "__init__.py").resolve()
        ),
        "operator_count": 1,
        "output": str((output_dir / "equant-py" / "1.0.0").resolve()),
        "provider": "equant-py",
        "release": "1.0.0",
        "resource_count": 14,
        "state": "research-certified",
    }
