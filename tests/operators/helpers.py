"""Fixtures for exercising a local provider submission repository."""

import hashlib
import json
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BUILD_IDENTIFIER = "build-20260826-equant-ttr"
COMPATIBILITY_ROOT = Path("compat/open_xquant")
CATALOG_NAME = "operator_catalog.json"


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
    implementation_mutate: Callable[[Path], None] | None = None,
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
    if implementation_mutate is not None:
        implementation_mutate(repository)
    source_tree_digest = _source_tree_digest(repository, ["src/equant_ttr/sma.py"])
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "implementation source")
    implementation_commit = _git(repository, "rev-parse", "HEAD").stdout.strip()
    wheel_name = "equant_ttr-1.0.0-py3-none-any.whl"
    wheel_bytes = _wheel_bytes()
    wheel_digest = sha256(wheel_bytes)
    compatibility_root = repository / COMPATIBILITY_ROOT
    _write_json(compatibility_root / CATALOG_NAME, _catalog())
    _write_json(
        compatibility_root / "candidate-build-v1.json",
        _build(
            implementation_commit,
            wheel_name,
            wheel_digest,
            BUILD_IDENTIFIER,
        ),
    )
    _write_json(
        compatibility_root / "numerical_baselines" / "technical-v1.json",
        _baseline(),
    )
    _write_json(
        compatibility_root / "manifests" / "equant.ttr.sma.operator.json",
        _manifest(
            implementation_commit,
            source_tree_digest,
            wheel_digest,
            BUILD_IDENTIFIER,
        ),
    )
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
        archive.writestr(
            "equant_ttr-1.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
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


def _build(
    source_commit: str,
    wheel_name: str,
    wheel_digest: str,
    build_identifier: str,
) -> dict[str, object]:
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
            "build_identifier": build_identifier,
            "digest": wheel_digest,
        }],
    }


def _baseline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": "equant-py",
        "release": "1.0.0",
        "cases": [{
            "case_id": "sma-window-3",
            "operator_id": "equant.ttr.sma",
            "operator_version": "1.0.0",
            "parameters": {"window": 3},
            "input": _panel(),
            "expected": {"sma_3": [None, 10.0]},
            "tolerance": {"absolute": 0.0, "relative": 0.0},
        }],
    }


def _source_tree_digest(root: Path, source_files: list[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(source_files):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256((root / relative_path).read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _manifest(
    source_commit: str,
    source_tree_digest: str,
    implementation_digest: str,
    build_identifier: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_version": 1,
        "operator_id": "equant.ttr.sma",
        "operator_version": "1.0.0",
        "semantic_name": "SMA",
        "distribution": "equant-ttr",
        "module": "equant_ttr",
        "callable": "sma",
        "execution_scope": "time_series",
        "lifecycle": "stateless",
        "causality": "past_only",
        "availability": "close_t",
        "mutates_input": False,
        "availability_depends_on_input": False,
        "input": {
            "required_columns": ["close"],
            "optional_columns": [],
            "supported_dtypes": ["float64"],
            "minimum_assets": 1,
            "minimum_time_length": 1,
            "requires_complete_cross_section": False,
            "requires_benchmark": False,
            "requires_industry_data": False,
            "requires_market_cap_data": False,
            "requires_fundamental_data": False,
            "requires_sorted_input": False,
            "required_context": [
                "timezone",
                "calendar",
                "frequency",
                "timestamp_semantics",
                "currency",
                "price_adjustment",
                "data_version",
                "source",
            ],
        },
        "parameters": {
            "window": {
                "type": "integer",
                "default": 3,
                "required": False,
                "constraints": {"minimum": 1},
                "unit": "sessions",
                "affects_warmup": True,
                "affects_output_fields": True,
                "affects_causality": False,
                "affects_availability": False,
            }
        },
        "output": {
            "fields": [{"name_template": "sma_{window}", "dtype": "float64", "value_range": "finite_or_nan"}],
            "alignment": "canonical_order",
            "warmup": "window - 1",
            "nan_policy": "propagate",
            "multiple_outputs": False,
        },
        "determinism": {
            "bitwise_deterministic": True,
            "random_seed_required": False,
            "tolerance": {"absolute": 0, "relative": 0},
            "tested_platforms": ["test-python3.12"],
        },
        "implementation": {
            "package_version": "1.0.0",
            "source_commit": f"git-sha1:{source_commit}",
            "source_files": ["src/equant_ttr/sma.py"],
            "source_tree_digest": source_tree_digest,
            "implementation_digest": implementation_digest,
            "build_identifier": build_identifier,
        },
    }
