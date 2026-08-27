"""Official certified operator environment provider index."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from oxq.operators.formats import safe_relative_path, strict_json_object
from oxq.operators.resources import materialize_operator_install_profile

_SEMVER = (
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:(?:0|[1-9][0-9]*)|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:(?:0|[1-9][0-9]*)|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
)
_EXACT_PROVIDER_REQUIREMENT = re.compile(
    rf"(?P<provider>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)==(?P<version>{_SEMVER})"
)
_OPERATOR_REF = re.compile(
    rf"(?P<operator_id>[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+)@(?P<operator_version>{_SEMVER})"
)


@dataclass(frozen=True)
class CertifiedOperatorRef:
    operator_id: str
    operator_version: str
    manifest_path: str
    baseline_paths: tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentProvider:
    provider: str
    distribution: str
    version: str
    certification_state: str
    operators: tuple[CertifiedOperatorRef, ...]
    manifest_digests: Mapping[str, str]
    baseline_digests: Mapping[str, str]


def parse_exact_provider_requirement(value: str) -> tuple[str, str]:
    """Parse the official environment provider requirement grammar."""
    match = _EXACT_PROVIDER_REQUIREMENT.fullmatch(value)
    if match is None:
        raise ValueError("environment provider requirement must be exact provider==semver")
    return match.group("provider"), match.group("version")


def load_environment_provider(provider: str, version: str) -> EnvironmentProvider:
    """Load one official certified environment provider entry."""
    payload = _load_index_payload()
    providers = payload.get("providers")
    if (
        set(payload) != {"schema_version", "providers"}
        or payload["schema_version"] != 1
        or not isinstance(providers, dict)
    ):
        raise ValueError("official environment provider index is invalid")

    provider_entries = providers.get(provider)
    if not isinstance(provider_entries, dict):
        raise ValueError("environment provider is not officially supported")

    entry = provider_entries.get(version)
    if not isinstance(entry, dict):
        raise ValueError("environment provider version is not officially supported")
    return _provider_from_payload(provider, version, entry)


def _load_index_payload() -> dict[str, object]:
    with materialize_operator_install_profile() as paths:
        return strict_json_object(paths["official_environment_providers"].read_bytes())


def _provider_from_payload(
    provider: str,
    version: str,
    payload: dict[str, object],
) -> EnvironmentProvider:
    if set(payload) != {
        "distribution",
        "certification_state",
        "operators",
        "manifest_digests",
        "baseline_digests",
    }:
        raise ValueError("official environment provider entry is invalid")
    distribution = payload["distribution"]
    certification_state = payload["certification_state"]
    operators = payload["operators"]
    manifest_digests = payload["manifest_digests"]
    baseline_digests = payload["baseline_digests"]
    if (
        distribution != provider
        or not isinstance(distribution, str)
        or not isinstance(certification_state, str)
        or not isinstance(operators, list)
        or not isinstance(manifest_digests, dict)
        or not isinstance(baseline_digests, dict)
    ):
        raise ValueError("official environment provider entry is invalid")

    typed_manifest_digests = _digest_map(manifest_digests)
    typed_baseline_digests = _digest_map(baseline_digests)
    typed_operators = tuple(_operator_from_payload(_operator_payload(item)) for item in operators)
    _validate_operator_artifacts(typed_operators, typed_manifest_digests, typed_baseline_digests)
    return EnvironmentProvider(
        provider=provider,
        distribution=distribution,
        version=version,
        certification_state=certification_state,
        operators=typed_operators,
        manifest_digests=typed_manifest_digests,
        baseline_digests=typed_baseline_digests,
    )


def _operator_from_payload(payload: dict[str, object]) -> CertifiedOperatorRef:
    if set(payload) != {"operator", "manifest_path", "baseline_paths"}:
        raise ValueError("official environment operator entry is invalid")
    operator = payload["operator"]
    manifest_path = payload["manifest_path"]
    baseline_paths = payload["baseline_paths"]
    if not isinstance(operator, str) or not isinstance(manifest_path, str) or not isinstance(baseline_paths, list):
        raise ValueError("official environment operator entry is invalid")
    match = _OPERATOR_REF.fullmatch(operator)
    if match is None:
        raise ValueError("official environment operator entry is invalid")
    safe_relative_path(manifest_path)
    typed_baselines = tuple(_baseline_path(value) for value in baseline_paths)
    return CertifiedOperatorRef(
        operator_id=match.group("operator_id"),
        operator_version=match.group("operator_version"),
        manifest_path=manifest_path,
        baseline_paths=typed_baselines,
    )


def _operator_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("official environment operator entry is invalid")
    return cast(dict[str, object], value)


def _baseline_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("official environment operator entry is invalid")
    safe_relative_path(value)
    return value


def _digest_map(payload: dict[object, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, digest in payload.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("official environment digest entry is invalid")
        safe_relative_path(path)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError("official environment digest entry is invalid")
        result[path] = digest
    return result


def _validate_operator_artifacts(
    operators: tuple[CertifiedOperatorRef, ...],
    manifest_digests: Mapping[str, str],
    baseline_digests: Mapping[str, str],
) -> None:
    if not operators:
        raise ValueError("official environment provider must contain operators")
    for operator in operators:
        if operator.manifest_path not in manifest_digests:
            raise ValueError("official environment operator manifest digest is missing")
        for baseline_path in operator.baseline_paths:
            if baseline_path not in baseline_digests:
                raise ValueError("official environment operator baseline digest is missing")
