"""Strict parsing of certified provider release indexes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from packaging import tags

from oxq.operators.formats import canonical_json_bytes, strict_json_object
from oxq.operators.install_errors import OperatorInstallError, install_error
from oxq.operators.resources import materialize_operator_install_profile

_SEMVER = (
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:(?:0|[1-9][0-9]*)|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:(?:0|[1-9][0-9]*)|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_PUBLIC_RELEASE_SEMVER = _SEMVER.removesuffix(r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?")
EXACT_PROVIDER_REQUIREMENT = re.compile(
    rf"(?P<provider>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)==(?P<release>{_PUBLIC_RELEASE_SEMVER})"
)


@dataclass(frozen=True)
class OfficialProvider:
    name: str
    repository: str
    release_asset: str


@dataclass(frozen=True)
class ReleaseAsset:
    filename: str
    url: str
    size_bytes: int
    digest: str


@dataclass(frozen=True)
class ReleaseWheel(ReleaseAsset):
    distribution: str
    version: str
    role: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseTarget:
    python_tag: str
    abi_tag: str
    platform_tag: str
    bundle: ReleaseAsset
    wheels: tuple[ReleaseWheel, ...]


@dataclass(frozen=True)
class OperatorReleaseIndex:
    raw_bytes: bytes
    provider: str
    release: str
    submission_commit: str
    source_commit: str
    certification_state: str
    operator_count: int
    targets: tuple[ReleaseTarget, ...]


def parse_exact_requirement(value: str) -> tuple[str, str]:
    """Parse the one accepted provider requirement grammar."""
    match = EXACT_PROVIDER_REQUIREMENT.fullmatch(value)
    if match is None:
        raise install_error(
            "operator_requirement_invalid",
            "operator requirement must be provider==semver",
            stage="requirement",
        )
    return match.group("provider"), match.group("release")


def load_official_provider(name: str) -> OfficialProvider:
    """Load one provider from the byte-frozen official provider map."""
    try:
        with materialize_operator_install_profile() as paths:
            payload = strict_json_object(paths["official_providers"].read_bytes())
        providers = payload["providers"]
        if (
            set(payload) != {"schema_version", "providers"}
            or payload["schema_version"] != 1
            or not isinstance(providers, dict)
        ):
            raise ValueError("official provider map is invalid")
        entry = providers.get(name)
        if entry is None:
            raise install_error(
                "operator_provider_unknown",
                "operator provider is not officially supported",
                stage="provider",
                provider=name,
            )
        if (
            not isinstance(entry, dict)
            or set(entry) != {"repository", "release_asset"}
            or not isinstance(entry.get("repository"), str)
            or not isinstance(entry.get("release_asset"), str)
        ):
            raise ValueError("official provider entry is invalid")
        return OfficialProvider(
            name=name,
            repository=entry["repository"],
            release_asset=entry["release_asset"],
        )
    except OperatorInstallError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, KeyError):
        raise install_error(
            "operator_release_invalid",
            "official provider map is invalid",
            stage="provider",
            provider=name,
        ) from None


def parse_release_index(raw_bytes: bytes) -> OperatorReleaseIndex:
    """Validate strict canonical index bytes before materializing records."""
    try:
        payload = strict_json_object(raw_bytes)
        if canonical_json_bytes(payload) != raw_bytes:
            raise ValueError("release index is not canonical")
        _release_validator().validate(payload)
        return _release_index_from_payload(raw_bytes, payload)
    except (OSError, ValueError, TypeError, UnicodeError, ValidationError):
        raise install_error(
            "operator_release_invalid",
            "operator release index is invalid",
            stage="release",
        ) from None


def select_release_target(index: OperatorReleaseIndex) -> ReleaseTarget:
    """Select exactly one target and require its entire wheel closure to fit."""
    supported_tags = {str(tag) for tag in tags.sys_tags()}
    matches = [
        target
        for target in index.targets
        if _target_tag(target) in supported_tags
    ]
    if not matches:
        raise install_error(
            "operator_target_unavailable",
            "operator release has no compatible target",
            stage="target",
            provider=index.provider,
            release=index.release,
        )
    if len(matches) != 1:
        raise install_error(
            "operator_release_invalid",
            "operator release has multiple compatible targets",
            stage="target",
            provider=index.provider,
            release=index.release,
        )
    target = matches[0]
    if any(tag not in supported_tags for wheel in target.wheels for tag in wheel.tags):
        raise install_error(
            "operator_release_invalid",
            "operator release target contains incompatible wheel tags",
            stage="target",
            provider=index.provider,
            release=index.release,
        )
    return target


def _release_validator() -> Draft202012Validator:
    with materialize_operator_install_profile() as paths:
        schema = json.loads(paths["operator_release"].read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _release_index_from_payload(raw_bytes: bytes, payload: dict[str, object]) -> OperatorReleaseIndex:
    targets = tuple(_target_from_payload(cast(dict[str, object], target)) for target in cast(list[object], payload["targets"]))
    return OperatorReleaseIndex(
        raw_bytes=raw_bytes,
        provider=cast(str, payload["provider"]),
        release=cast(str, payload["release"]),
        submission_commit=cast(str, payload["submission_commit"]),
        source_commit=cast(str, payload["source_commit"]),
        certification_state=cast(str, payload["certification_state"]),
        operator_count=cast(int, payload["operator_count"]),
        targets=targets,
    )


def _target_from_payload(payload: dict[str, object]) -> ReleaseTarget:
    return ReleaseTarget(
        python_tag=cast(str, payload["python_tag"]),
        abi_tag=cast(str, payload["abi_tag"]),
        platform_tag=cast(str, payload["platform_tag"]),
        bundle=_asset_from_payload(cast(dict[str, object], payload["bundle"])),
        wheels=tuple(_wheel_from_payload(cast(dict[str, object], wheel)) for wheel in cast(list[object], payload["wheels"])),
    )


def _asset_from_payload(payload: dict[str, object]) -> ReleaseAsset:
    return ReleaseAsset(
        filename=cast(str, payload["filename"]),
        url=cast(str, payload["url"]),
        size_bytes=cast(int, payload["size_bytes"]),
        digest=cast(str, payload["digest"]),
    )


def _wheel_from_payload(payload: dict[str, object]) -> ReleaseWheel:
    return ReleaseWheel(
        **_asset_from_payload(payload).__dict__,
        distribution=cast(str, payload["distribution"]),
        version=cast(str, payload["version"]),
        role=cast(str, payload["role"]),
        tags=tuple(cast(list[str], payload["tags"])),
    )


def _target_tag(target: ReleaseTarget) -> str:
    return f"{target.python_tag}-{target.abi_tag}-{target.platform_tag}"
