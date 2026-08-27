"""Resolve verified environment operator callables for research use."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from oxq.operators.environment_index import CertifiedOperatorRef
from oxq.operators.environment_provider import InstalledEnvironmentProvider, verify_installed_provider
from oxq.operators.errors import OperatorCertificationError


@dataclass(frozen=True)
class EnvironmentOperatorBinding:
    operator_id: str
    operator_version: str
    provider_requirement: str
    manifest: Mapping[str, object]
    callable: Callable[..., object]


def resolve_environment_operator(
    operator_id: str,
    operator_version: str,
    provider_requirement: str,
) -> EnvironmentOperatorBinding:
    """Return a callable binding from a verified installed environment provider."""
    installed = verify_installed_provider(provider_requirement)
    if installed.provider.certification_state != "research-certified":
        raise _error(
            "environment_provider_not_research_certified",
            "environment provider is not research-certified",
            operator_id,
        )

    operator = _find_certified_operator(installed, operator_id, operator_version)
    manifest = installed.manifests.get(operator.manifest_path)
    if manifest is None:
        raise _error(
            "environment_operator_manifest_missing",
            "certified environment operator manifest is missing",
            operator_id,
        )
    _validate_manifest_identity(manifest, operator_id, operator_version)
    _validate_manifest_certification_state(manifest, operator_id)

    module_name = manifest.get("module")
    callable_name = manifest.get("callable")
    if not isinstance(module_name, str) or not isinstance(callable_name, str):
        raise _error(
            "environment_operator_implementation_invalid",
            "certified environment operator implementation is invalid",
            operator_id,
        )

    module = importlib.import_module(module_name)
    implementation = getattr(module, callable_name, None)
    if not callable(implementation):
        raise _error(
            "environment_operator_callable_missing",
            "certified environment operator callable is unavailable",
            operator_id,
        )

    return EnvironmentOperatorBinding(
        operator_id=operator_id,
        operator_version=operator_version,
        provider_requirement=provider_requirement,
        manifest=manifest,
        callable=implementation,
    )


def _find_certified_operator(
    installed: InstalledEnvironmentProvider,
    operator_id: str,
    operator_version: str,
) -> CertifiedOperatorRef:
    for operator in installed.provider.operators:
        if operator.operator_id == operator_id and operator.operator_version == operator_version:
            return operator
    raise _error(
        "environment_operator_not_certified",
        f"environment operator is not certified: {operator_id}@{operator_version}",
        operator_id,
    )


def _validate_manifest_identity(
    manifest: Mapping[str, object],
    operator_id: str,
    operator_version: str,
) -> None:
    if (
        manifest.get("operator_id") != operator_id
        or manifest.get("operator_version") != operator_version
    ):
        raise _error(
            "environment_operator_manifest_mismatch",
            "certified environment operator manifest identity does not match",
            operator_id,
        )


def _validate_manifest_certification_state(
    manifest: Mapping[str, object],
    operator_id: str,
) -> None:
    state = manifest.get("certification_state")
    if state is not None and state != "research-certified":
        raise _error(
            "environment_operator_not_research_certified",
            "environment operator manifest is not research-certified",
            operator_id,
        )


def _error(
    code: str,
    message: str,
    operator_id: str,
) -> OperatorCertificationError:
    return OperatorCertificationError(
        code,
        message,
        stage="environment_runtime",
        operator_id=operator_id,
    )
