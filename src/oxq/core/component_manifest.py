"""Workspace-local component extension manifest loading and hashing."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from oxq.core.registry import (
    _INDICATOR_REGISTRY,
    _PORTFOLIO_OPTIMIZER_REGISTRY,
    _RULE_REGISTRY,
    _SIGNAL_REGISTRY,
    register_indicator,
    register_portfolio_optimizer,
    register_rule,
    register_signal,
)

MANIFEST_SCHEMA_VERSION = 1
_KIND_TO_REGISTER = {
    "Indicator": register_indicator,
    "Signal": register_signal,
    "Rule": register_rule,
    "PortfolioOptimizer": register_portfolio_optimizer,
}
_KIND_TO_REGISTRY = {
    "Indicator": _INDICATOR_REGISTRY,
    "Signal": _SIGNAL_REGISTRY,
    "Rule": _RULE_REGISTRY,
    "PortfolioOptimizer": _PORTFOLIO_OPTIMIZER_REGISTRY,
}
_COMPONENT_REGISTRIES = (
    _INDICATOR_REGISTRY,
    _SIGNAL_REGISTRY,
    _PORTFOLIO_OPTIMIZER_REGISTRY,
    _RULE_REGISTRY,
)
_LOADED_EXTENSION_ROOTS: set[Path] = set()


def snapshot_component_registries() -> Callable[[], None]:
    """Capture component registries and return a restore callback."""

    snapshots = [dict(registry) for registry in _COMPONENT_REGISTRIES]

    def restore() -> None:
        for registry, snapshot in zip(_COMPONENT_REGISTRIES, snapshots, strict=True):
            registry.clear()
            registry.update(snapshot)

    return restore


@contextmanager
def scoped_component_registries():
    """Temporarily allow workspace component registration inside a scope."""

    restore = snapshot_component_registries()
    try:
        yield
    finally:
        restore()


def load_component_manifest(path: str | Path, *, verify_hash: bool = True) -> dict[str, Any]:
    """Load and register workspace extension components from a manifest.

    The manifest is the deterministic contract between the component author
    worker and later strategy, audit, compile, and run commands.
    """

    manifest_path = Path(path).resolve()
    payload = _read_manifest(manifest_path)
    if verify_hash:
        expected = payload.get("bundle_hash")
        actual = compute_component_bundle_hash(manifest_path)
        if expected != actual:
            raise ValueError(f"component bundle hash mismatch: stored={expected}, actual={actual}")
    root = _extension_root(manifest_path, payload)
    if verify_hash:
        _validate_declared_file_hashes(payload, root, manifest_path.parent)
    _clear_extension_module_cache(payload, root)
    importlib.invalidate_caches()
    with _prepend_sys_path(root):
        for index, component in enumerate(_components(payload)):
            _register_manifest_component(component, index, root)
    _LOADED_EXTENSION_ROOTS.add(root.resolve())
    return payload


def load_component_manifests_from_run(run_dir: str | Path, *, verify_hash: bool = True) -> list[dict[str, Any]]:
    """Load component manifests recorded by a completed run, if any."""

    run_path = Path(run_dir)
    summary = _read_run_manifest_summary(run_path)

    manifests: list[dict[str, Any]] = []
    for index, item in enumerate(summary):
        if not isinstance(item, dict):
            raise ValueError(f"component_manifests.json[{index}] must be an object")
        manifest_path = item.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise ValueError(f"component_manifests.json[{index}].manifest_path is required")
        recorded_hash = item.get("bundle_hash")
        if not isinstance(recorded_hash, str) or not recorded_hash:
            raise ValueError(f"component_manifests.json[{index}].bundle_hash is required")
        resolved = _resolve_run_manifest_path(run_path, item, recorded_hash, len(summary))
        loaded = load_component_manifest(resolved, verify_hash=verify_hash)
        loaded_hash = loaded.get("bundle_hash")
        if loaded_hash != recorded_hash:
            raise ValueError(
                "recorded component bundle hash mismatch: "
                f"recorded={recorded_hash}, manifest={loaded_hash}, path={resolved}"
            )
        manifests.append(loaded)
    return manifests


def validate_component_manifest_records_from_run(run_dir: str | Path) -> list[str]:
    """Validate run component manifest records without importing component code."""

    run_path = Path(run_dir)
    summary = _read_run_manifest_summary(run_path)
    warnings: list[str] = []
    for index, item in enumerate(summary):
        if not isinstance(item, dict):
            raise ValueError(f"component_manifests.json[{index}] must be an object")
        manifest_path = item.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise ValueError(f"component_manifests.json[{index}].manifest_path is required")
        recorded_hash = item.get("bundle_hash")
        if not isinstance(recorded_hash, str) or not recorded_hash:
            raise ValueError(f"component_manifests.json[{index}].bundle_hash is required")
        try:
            resolved = _resolve_run_manifest_path(run_path, item, recorded_hash, len(summary))
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        payload = _read_manifest(resolved)
        loaded_hash = payload.get("bundle_hash")
        if loaded_hash != recorded_hash:
            raise ValueError(
                "recorded component bundle hash mismatch: "
                f"recorded={recorded_hash}, manifest={loaded_hash}, path={resolved}"
            )
    return warnings


def _read_run_manifest_summary(run_path: Path) -> list[Any]:
    summary_path = run_path / "component_manifests.json"
    if not summary_path.exists():
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"component_manifests.json is invalid: {exc}") from exc
    if not isinstance(summary, list):
        raise ValueError("component_manifests.json must be a list")
    return summary


def _resolve_run_manifest_path(run_path: Path, item: dict[str, Any], recorded_hash: str, summary_count: int) -> Path:
    archived_path = item.get("archived_manifest_path")
    if isinstance(archived_path, str) and archived_path:
        archived = _safe_run_relative_file(run_path, archived_path)
        if archived.exists():
            return archived
        raise ValueError(f"archived component manifest not found: {archived}")

    archived = run_path / "component_manifest.json"
    if summary_count == 1 and archived.exists():
        archived_payload, actual_hash = _read_legacy_archived_manifest(run_path, archived)
        if archived_payload.get("bundle_hash") == recorded_hash and actual_hash == recorded_hash:
            return archived
        raise ValueError(
            "legacy archived component manifest hash mismatch: "
            f"recorded={recorded_hash}, manifest={archived_payload.get('bundle_hash')}, actual={actual_hash}"
        )

    manifest_path = item.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError("component manifest path is required")
    resolved = Path(manifest_path)
    if not resolved.is_absolute():
        resolved = run_path / resolved
    if resolved.exists():
        return resolved

    raise ValueError(f"recorded component manifest not found: {resolved}")


def _read_legacy_archived_manifest(run_path: Path, archived: Path) -> tuple[dict[str, Any], str]:
    if archived.is_symlink():
        raise ValueError("legacy archived component manifest must not be a symlink: component_manifest.json")
    if not archived.resolve().is_relative_to(run_path.resolve()):
        raise ValueError("legacy archived component manifest escapes run directory: component_manifest.json")
    try:
        payload = _read_manifest(archived)
        actual_hash = compute_component_bundle_hash(archived)
    except ValueError as exc:
        raise ValueError(f"legacy archived component manifest is invalid: {exc}") from exc
    return payload, actual_hash


def _safe_run_relative_file(run_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archived component manifest path is unsafe: {raw_path}")
    candidate = run_path / path
    if candidate.is_symlink():
        raise ValueError(f"archived component manifest path must not be a symlink: {raw_path}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(run_path.resolve()):
        raise ValueError(f"archived component manifest path escapes run directory: {raw_path}")
    return resolved


def compute_component_bundle_hash(path: str | Path) -> str:
    """Compute the deterministic hash for a component extension bundle."""

    manifest_path = Path(path).resolve()
    payload = _read_manifest(manifest_path)
    root = _extension_root(manifest_path, payload)
    manifest_without_hash = dict(payload)
    manifest_without_hash.pop("bundle_hash", None)
    pieces_by_path: dict[str, dict[str, Any]] = {}
    _add_hash_piece(
        pieces_by_path,
        _relative_path(manifest_path, manifest_path.parent),
        _stable_payload_hash(manifest_without_hash),
    )
    for bundle_file in sorted(root.rglob("*")):
        if bundle_file.is_symlink():
            raise ValueError(f"component bundle file must not be a symlink: {_relative_path(bundle_file, manifest_path.parent)}")
        if bundle_file.resolve() == manifest_path:
            continue
        if _is_bundle_file(bundle_file):
            _add_hash_piece(pieces_by_path, _relative_path(bundle_file, manifest_path.parent), _sha256_file(bundle_file))
    for component in _components(payload):
        source_paths = []
        if isinstance(component.get("source_path"), str):
            source_paths.append(component["source_path"])
        if isinstance(component.get("module"), str):
            source_paths.append(component["module"].replace(".", "/") + ".py")
        for raw_source_path in source_paths:
            candidate = _safe_relative_file(root, raw_source_path)
            if candidate.exists():
                _add_hash_piece(pieces_by_path, _relative_path(candidate, manifest_path.parent), _sha256_file(candidate))
        tests = component.get("tests")
        if isinstance(tests, list):
            for raw in tests:
                if isinstance(raw, str):
                    test_file = _safe_relative_file(manifest_path.parent, raw)
                    if test_file.exists():
                        _add_hash_piece(pieces_by_path, _relative_path(test_file, manifest_path.parent), _sha256_file(test_file))
    pieces = sorted(pieces_by_path.values(), key=lambda item: item["path"])
    return _stable_payload_hash(sorted(pieces, key=lambda item: item["path"]))


def _is_bundle_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in path.parts):
        return False
    return path.suffix not in {".pyc", ".pyo"}


def component_manifest_summary(path: str | Path) -> dict[str, Any]:
    """Return hash and component metadata without importing extension code."""

    manifest_path = Path(path).resolve()
    payload = _read_manifest(manifest_path)
    actual_hash = compute_component_bundle_hash(manifest_path)
    return {
        "status": "pass" if payload.get("bundle_hash") == actual_hash else "fail",
        "manifest": str(manifest_path),
        "extension_id": payload.get("extension_id", ""),
        "bundle_hash": payload.get("bundle_hash", ""),
        "computed_bundle_hash": actual_hash,
        "components": _components(payload),
    }


def _register_manifest_component(component: dict[str, Any], index: int, root: Path) -> None:
    kind = component.get("kind")
    register = _KIND_TO_REGISTER.get(kind)
    if register is None:
        raise ValueError(f"components[{index}].kind is unsupported: {kind}")
    registry = _KIND_TO_REGISTRY[kind]
    declared_name = _require_str(component, "name", index)
    if declared_name in registry:
        raise ValueError(f"components[{index}].name already exists in the {kind} registry: {declared_name}")
    module_name = _require_str(component, "module", index)
    class_name = _require_str(component, "class", index)
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not Path(module_file).resolve().is_relative_to(root):
        raise ValueError(f"components[{index}].module must resolve inside the component extension root")
    cls = getattr(module, class_name)
    registry_name = getattr(cls, "name", cls.__name__)
    if registry_name != declared_name:
        raise ValueError(
            f"components[{index}].name must match registered class name: "
            f"declared={declared_name}, actual={registry_name}"
        )
    register(cls)


def _clear_extension_module_cache(payload: dict[str, Any], root: Path) -> None:
    root = root.resolve()
    top_level_packages = {
        module.split(".", 1)[0]
        for component in _components(payload)
        if isinstance((module := component.get("module")), str) and module
    }
    if "oxq" in top_level_packages:
        raise ValueError("workspace component modules must not be declared under the oxq package")
    loaded_roots = set(_LOADED_EXTENSION_ROOTS)
    loaded_roots.add(root)
    for module_name in list(sys.modules):
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        module_path = Path(module_file).resolve() if isinstance(module_file, str) else None
        if module_path is not None and any(module_path.is_relative_to(loaded_root) for loaded_root in loaded_roots):
            sys.modules.pop(module_name, None)
            continue
        for package in top_level_packages:
            if module_name == package or module_name.startswith(package + "."):
                if module_path is not None and (module_path.is_relative_to(root) or not _is_protected_runtime_module(module_path)):
                    sys.modules.pop(module_name, None)
                break


def _is_protected_runtime_module(path: Path) -> bool:
    protected_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    return any(path.is_relative_to(root) for root in protected_roots)


def _validate_declared_file_hashes(payload: dict[str, Any], root: Path, workspace_root: Path) -> None:
    for index, component in enumerate(_components(payload)):
        source_hash = component.get("source_hash")
        if isinstance(source_hash, str) and source_hash:
            source_path = component.get("source_path")
            if not isinstance(source_path, str) and isinstance(component.get("module"), str):
                source_path = component["module"].replace(".", "/") + ".py"
            if not isinstance(source_path, str):
                raise ValueError(f"components[{index}].source_hash requires source_path or module")
            actual = _sha256_file(_safe_relative_file(root, source_path))
            if actual != source_hash:
                raise ValueError(f"components[{index}].source_hash mismatch: stored={source_hash}, actual={actual}")
        test_hash = component.get("test_hash")
        tests = component.get("tests")
        if isinstance(test_hash, str) and test_hash and isinstance(tests, list) and len(tests) == 1 and isinstance(tests[0], str):
            actual = _sha256_file(_safe_relative_file(workspace_root, tests[0]))
            if actual != test_hash:
                raise ValueError(f"components[{index}].test_hash mismatch: stored={test_hash}, actual={actual}")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"component manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("component manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"component manifest schema_version must be {MANIFEST_SCHEMA_VERSION}")
    if not isinstance(payload.get("extension_id"), str) or not payload["extension_id"]:
        raise ValueError("component manifest extension_id must be a non-empty string")
    if not isinstance(payload.get("components"), list):
        raise ValueError("component manifest components must be a list")
    return payload


def _components(payload: dict[str, Any]) -> list[dict[str, Any]]:
    components = payload.get("components")
    assert isinstance(components, list)
    result = []
    for index, item in enumerate(components):
        if not isinstance(item, dict):
            raise ValueError(f"components[{index}] must be an object")
        result.append(item)
    return result


def _extension_root(manifest_path: Path, payload: dict[str, Any]) -> Path:
    raw_root = payload.get("extension_root") or payload.get("extension_id")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError("component manifest extension_root or extension_id must be a non-empty string")
    root = (manifest_path.parent / raw_root).resolve()
    if not root.is_relative_to(manifest_path.parent):
        raise ValueError("component extension root must stay inside the workspace")
    if not root.is_dir():
        raise ValueError(f"component extension root does not exist: {root}")
    return root


def _safe_relative_file(root: Path, raw_path: str) -> Path:
    candidate = root / raw_path
    if _relative_path_contains_symlink(root, raw_path):
        raise ValueError(f"path must not traverse symlinks: {raw_path}")
    path = candidate.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes component extension root: {raw_path}")
    return path


def _relative_path_contains_symlink(root: Path, raw_path: str) -> bool:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        return True
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _add_hash_piece(pieces: dict[str, dict[str, Any]], path: str, digest: str) -> None:
    pieces[path] = {"path": path, "sha256": digest}


def _require_str(component: dict[str, Any], key: str, index: int) -> str:
    value = component.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"components[{index}].{key} must be a non-empty string")
    return value


@contextmanager
def _prepend_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _stable_payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
