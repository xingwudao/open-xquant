"""Deterministic, import-free operator catalog ingestion."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml  # type: ignore[import-untyped]

from oxq.operators._version import is_semantic_version
from oxq.operators.errors import InvalidManifestError
from oxq.operators.manifest import OperatorManifest, load_operator_manifest

_CATALOG_KEYS = {"schema_version", "contract_version", "package", "operators", "catalog_digest"}
_PACKAGE_KEYS = {"distribution", "version", "source_commit", "source_tree_digest", "build_identifier"}
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _find_recursive_container(value: Any, path: str = "catalog") -> str | None:
    stack: list[tuple[Any, str, bool]] = [(value, path, False)]
    active: set[int] = set()
    while stack:
        current, current_path, exiting = stack.pop()
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        identity = id(current)
        if exiting:
            active.remove(identity)
            continue
        if identity in active:
            return current_path
        active.add(identity)
        stack.append((current, current_path, True))
        if isinstance(current, Mapping):
            children = [(item, f"{current_path}.{key}") for key, item in current.items()]
        else:
            children = [(item, f"{current_path}[{index}]") for index, item in enumerate(current)]
        for child, child_path in reversed(children):
            stack.append((child, child_path, False))
    return None


def _reject_recursive_containers(value: Any) -> None:
    recursive_path = _find_recursive_container(value)
    if recursive_path is not None:
        raise InvalidManifestError(f"catalog contains a recursive or cyclic container: {recursive_path}")


def _find_non_string_mapping_key(value: Any, path: str = "catalog") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path}[{key!r}]"
            found = _find_non_string_mapping_key(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_non_string_mapping_key(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class OperatorCatalog:
    schema_version: int
    contract_version: int
    package: Mapping[str, str]
    operators: tuple[OperatorManifest, ...]
    digest: str
    _raw: Mapping[str, Any]

    @property
    def operator_ids(self) -> tuple[str, ...]:
        return tuple(item.operator_id for item in self.operators)

    def get(self, operator_id: str) -> OperatorManifest:
        for item in self.operators:
            if item.operator_id == operator_id:
                return item
        raise KeyError(operator_id)

    def to_json(self) -> str:
        payload = copy.deepcopy(dict(self._raw))
        payload["catalog_digest"] = self.digest
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def load_operator_catalog(source: str | Path | Mapping[str, Any]) -> OperatorCatalog:
    payload = _read_catalog(source)
    non_string_key_path = _find_non_string_mapping_key(payload)
    if non_string_key_path is not None:
        raise InvalidManifestError(f"catalog mapping keys must be strings: {non_string_key_path}")
    unknown = sorted(set(payload) - _CATALOG_KEYS)
    if unknown:
        raise InvalidManifestError(f"catalog contains unknown fields: {', '.join(unknown)}")
    for key in ("schema_version", "contract_version", "package", "operators"):
        if key not in payload:
            raise InvalidManifestError(f"catalog requires {key}")
    for key in ("schema_version", "contract_version"):
        value = payload[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidManifestError(f"catalog {key} must be an integer")
    if payload["schema_version"] != 1 or payload["contract_version"] != 1:
        raise InvalidManifestError("catalog only supports schema_version=1 and contract_version=1")
    package = payload["package"]
    valid_package = (
        isinstance(package, dict) and set(package) == _PACKAGE_KEYS and all(isinstance(value, str) and value for value in package.values())
    )
    if not valid_package:
        raise InvalidManifestError(f"catalog package must contain exactly {sorted(_PACKAGE_KEYS)}")
    if not is_semantic_version(package["version"]):
        raise InvalidManifestError("catalog package.version must be semantic versioning")
    if not _COMMIT_RE.fullmatch(package["source_commit"]):
        raise InvalidManifestError("catalog package.source_commit must be a full hexadecimal commit")
    if not _DIGEST_RE.fullmatch(package["source_tree_digest"]):
        raise InvalidManifestError("catalog package.source_tree_digest must be a sha256 digest")
    raw_operators = payload["operators"]
    if not isinstance(raw_operators, list):
        raise InvalidManifestError("catalog operators must be an array")
    operators: list[OperatorManifest] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_operators):
        if not isinstance(item, dict):
            raise InvalidManifestError(f"catalog operators[{index}] must be an object")
        manifest = load_operator_manifest(item)
        if manifest.operator_id in seen:
            raise InvalidManifestError(f"duplicate operator_id: {manifest.operator_id}")
        if manifest.distribution != package["distribution"]:
            raise InvalidManifestError(f"operator distribution mismatch: {manifest.operator_id}")
        if "manifest_digest" not in item:
            raise InvalidManifestError(f"operator manifest_digest is required: {manifest.operator_id}")
        seen.add(manifest.operator_id)
        operators.append(manifest)
    normalized = copy.deepcopy(payload)
    normalized.pop("catalog_digest", None)
    normalized["operators"] = sorted(raw_operators, key=lambda item: item["operator_id"])
    digest = "sha256:" + hashlib.sha256(_canonical_json(normalized).encode()).hexdigest()
    declared = payload.get("catalog_digest")
    if declared is not None and declared != digest:
        raise InvalidManifestError(f"catalog_digest mismatch: declared={declared}, actual={digest}")
    return OperatorCatalog(
        schema_version=1,
        contract_version=1,
        package=MappingProxyType(dict(package)),
        operators=tuple(sorted(operators, key=lambda item: item.operator_id)),
        digest=digest,
        _raw=MappingProxyType(normalized),
    )


def _read_catalog(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        _reject_recursive_containers(source)
        payload = _thaw(source)
        assert isinstance(payload, dict)
        return payload
    path = Path(source)
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise InvalidManifestError(f"operator catalog is invalid: {path}: {exc}") from exc
    _reject_recursive_containers(payload)
    if not isinstance(payload, dict):
        raise InvalidManifestError(f"operator catalog must contain an object: {path}")
    return payload
