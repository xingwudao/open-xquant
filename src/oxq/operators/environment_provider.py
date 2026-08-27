"""Verify installed certified operator environment providers."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from oxq.operators.environment_index import (
    EnvironmentProvider,
    load_environment_provider,
    parse_exact_provider_requirement,
)
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.formats import sha256_bytes, strict_json_object


@dataclass(frozen=True)
class VerifiedRuntimeFile:
    package_path: str
    path: Path
    digest: str


@dataclass(frozen=True)
class InstalledEnvironmentProvider:
    provider: EnvironmentProvider
    manifests: Mapping[str, dict[str, object]]
    baselines: Mapping[str, bytes]
    runtime_files: Mapping[str, VerifiedRuntimeFile]


def verify_installed_provider(requirement: str) -> InstalledEnvironmentProvider:
    """Verify installed package bytes against the official environment index."""
    try:
        provider_name, version = parse_exact_provider_requirement(requirement)
        provider = load_environment_provider(provider_name, version)
    except ValueError as exc:
        raise _error("environment_provider_invalid", str(exc)) from None

    try:
        distribution = importlib.metadata.distribution(provider.distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise _error(
            "environment_provider_not_installed",
            f"environment provider distribution is not installed: {provider.distribution}",
        ) from exc

    installed_version = getattr(distribution, "version", None)
    if installed_version != provider.version:
        raise _error(
            "environment_provider_version_mismatch",
            (
                "environment provider version mismatch: "
                f"expected {provider.version}, found {installed_version}"
            ),
        )

    available_files = getattr(distribution, "files", None)
    if available_files is None:
        raise _error(
            "environment_provider_files_unavailable",
            f"installed environment provider file metadata is unavailable: {provider.distribution}",
        )
    installed_paths = {str(path) for path in available_files}
    for package_path in provider.declared_artifact_paths:
        if package_path not in installed_paths:
            raise _error(
                "environment_provider_file_missing",
                f"installed environment provider file is missing: {package_path}",
            )

    manifests: dict[str, dict[str, object]] = {}
    baselines: dict[str, bytes] = {}
    runtime_files: dict[str, VerifiedRuntimeFile] = {}
    for operator in provider.operators:
        manifest_bytes = _read_declared_file(
            distribution,
            installed_paths,
            operator.manifest_path,
            provider.manifest_digests[operator.manifest_path],
        )
        manifest = strict_json_object(manifest_bytes)
        _verify_manifest_certification_state(manifest, provider.certification_state, operator.manifest_path)
        manifest["certification_state"] = provider.certification_state
        manifests[operator.manifest_path] = manifest
        for baseline_path in operator.baseline_paths:
            baselines[baseline_path] = _read_declared_file(
                distribution,
                installed_paths,
                baseline_path,
                provider.baseline_digests[baseline_path],
            )
    for runtime_path, expected_digest in provider.runtime_digests.items():
        raw, path = _read_declared_file_with_path(
            distribution,
            installed_paths,
            runtime_path,
            expected_digest,
        )
        del raw
        runtime_files[runtime_path] = VerifiedRuntimeFile(
            package_path=runtime_path,
            path=path,
            digest=expected_digest,
        )

    return InstalledEnvironmentProvider(
        provider=provider,
        manifests=manifests,
        baselines=baselines,
        runtime_files=runtime_files,
    )


def _read_declared_file(
    distribution: importlib.metadata.Distribution,
    installed_paths: set[str],
    package_path: str,
    expected_digest: str,
) -> bytes:
    raw, _ = _read_declared_file_with_path(
        distribution,
        installed_paths,
        package_path,
        expected_digest,
    )
    return raw


def _read_declared_file_with_path(
    distribution: importlib.metadata.Distribution,
    installed_paths: set[str],
    package_path: str,
    expected_digest: str,
) -> tuple[bytes, Path]:
    if package_path not in installed_paths:
        raise _error(
            "environment_provider_file_missing",
            f"installed environment provider file is missing: {package_path}",
        )
    path = Path(distribution.locate_file(package_path))
    if path.is_symlink() or not path.is_file():
        raise _error(
            "environment_provider_file_not_regular",
            f"installed environment provider file must be a regular file: {package_path}",
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _error(
            "environment_provider_file_unreadable",
            f"installed environment provider file is unreadable: {package_path}",
        ) from exc
    actual_digest = sha256_bytes(raw)
    if actual_digest != expected_digest:
        raise _error(
            "environment_provider_digest_mismatch",
            f"installed environment provider file digest mismatch: {package_path}",
        )
    return raw, path.resolve(strict=True)


def _verify_manifest_certification_state(
    manifest: dict[str, object],
    certification_state: str,
    package_path: str,
) -> None:
    manifest_state = manifest.get("certification_state")
    if manifest_state is not None and manifest_state != certification_state:
        raise _error(
            "environment_provider_manifest_state_mismatch",
            f"installed environment provider manifest certification state mismatch: {package_path}",
        )


def _error(code: str, message: str) -> OperatorCertificationError:
    return OperatorCertificationError(code, message, stage="environment_provider")
