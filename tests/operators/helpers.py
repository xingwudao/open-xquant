"""Fixtures for exercising a local provider submission repository."""

import hashlib
import json
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def sha256(value: bytes) -> str:
    """Return the contract digest for literal fixture bytes."""
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


@dataclass(frozen=True)
class ProviderRepository:
    path: Path
    implementation_commit: str
    submission_commit: str
    artifact_dir: Path
    wheel_name: str


def write_provider_repository(
    root: Path,
    mutate: Callable[[Path], None] | None = None,
) -> ProviderRepository:
    """Create a committed provider source tree and ignored wheel artifact."""
    repository = root / "provider"
    repository.mkdir(parents=True)
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test User")
    source = repository / "src" / "equant_ttr" / "sma.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("def sma():\n    return 10.0\n", encoding="utf-8")
    (repository / ".gitignore").write_text("dist/\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "implementation source")
    implementation_commit = _git(repository, "rev-parse", "HEAD").stdout.strip()
    wheel_name = "equant_ttr-1.0.0-py3-none-any.whl"
    wheel_bytes = _wheel_bytes()
    wheel_digest = sha256(wheel_bytes)
    _write_json(repository / "provider-catalog-v1.json", _catalog())
    _write_json(repository / "candidate-build-v1.json", _build(implementation_commit, wheel_name, wheel_digest))
    _write_json(repository / "numerical_baselines" / "technical-v1.json", _baseline())
    _write_json(repository / "manifests" / "equant.ttr.sma.operator.json", _manifest())
    if mutate is not None:
        mutate(repository)
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "provider submission")
    submission_commit = _git(repository, "rev-parse", "HEAD").stdout.strip()
    artifact_dir = repository / "dist"
    artifact_dir.mkdir()
    (artifact_dir / wheel_name).write_bytes(wheel_bytes)
    return ProviderRepository(repository, implementation_commit, submission_commit, artifact_dir, wheel_name)


def rewrite_json(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)


def commit_mutation(repository: Path, message: str = "mutate submission") -> str:
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", message)
    return _git(repository, "rev-parse", "HEAD").stdout.strip()


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _wheel_bytes() -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("equant_ttr/__init__.py", "")
    return buffer.getvalue()


def _panel() -> dict[str, object]:
    return {
        "schema_version": 1,
        "primary_key": ["date", "code"],
        "columns": [{"name": "close", "dtype": "float64", "required": True}],
        "context": {
            "timezone": "Asia/Shanghai",
            "calendar": "XSHG",
            "frequency": "1d",
            "timestamp_semantics": "bar_close",
            "currency": "CNY",
            "price_adjustment": "raw",
            "data_version": "v1",
            "source": "test",
        },
        "alignment": "preserve_input_order",
        "records": [{"date": "2026-08-26", "code": "000001.SZ", "close": 10.0}],
    }


def _catalog() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": {"name": "equant-py", "release": "1.0.0"},
        "contract_version": "1.0.0",
        "status": "candidate",
        "build_record": "candidate-build-v1.json",
        "operators": {
            "equant.ttr.sma@1.0.0": {
                "manifest": "manifests/equant.ttr.sma.operator.json",
                "baseline": "numerical_baselines/technical-v1.json",
            }
        },
    }


def _build(source_commit: str, wheel_name: str, wheel_digest: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": "equant-py",
        "release": "1.0.0",
        "source_commit": f"git-sha1:{source_commit}",
        "python": "3.12.0",
        "build_command": "uv build",
        "artifacts": [{
            "distribution": "equant-ttr",
            "version": "1.0.0",
            "filename": wheel_name,
            "role": "implementation",
            "digest": wheel_digest,
        }],
    }


def _baseline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": "equant-py",
        "release": "1.0.0",
        "cases": [{
            "operator_id": "equant.ttr.sma",
            "operator_version": "1.0.0",
            "parameters": {"window": 3},
            "input": _panel(),
            "expected": {"sma_3": [None, 10.0]},
            "tolerance": {"absolute": 0.0, "relative": 0.0},
        }],
    }


def _manifest() -> dict[str, object]:
    return {
        "operator_id": "equant.ttr.sma",
        "operator_version": "1.0.0",
        "implementation": {"source_files": ["src/equant_ttr/sma.py"]},
    }
