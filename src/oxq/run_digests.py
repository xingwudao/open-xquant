"""Canonical run digest replacement and run-artifact inventory validation.

Inventory contract schema 1 uses ``artifact_hashes.json.schema_version`` as
the profile selector. Required base bindings are cumulative:

* v0 legacy: data manifest, equity curve, trades, and metrics.
* v1: v0 plus strategy spec, environment, positions, and orders.
* v2: v1 plus execution assumptions; target weights are governed if present.
* v3: v2 plus target weights.
* v4: v3 plus the compiled plan.
* v5: v4 plus ``strategy.py``.

The closed extension set is declared in ``_EXTENSION_BINDINGS``. Extensions
are optional, but an extension file present in the run must have a current
binding. Later base bindings may appear in an older supported profile without
making files introduced after that profile mandatory. Digest rows pin the
resolved profile as ``artifact_inventory: {"schema_version": 1,
"profile": "artifact_hashes_vN"}`` (v0 uses ``artifact_hashes_v0_legacy``),
so refreshing a digest cannot silently downgrade an existing run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone; UTC = timezone.utc  # py3.9 compat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from oxq.process_lock import (
    ProcessFileLock,
    stable_filesystem_identity,
    stable_path_location_identity,
)
from oxq.selection_lock import final_selection_lock_path, hold_final_selection_lock

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{16,64}")
_WINDOWS_RESERVED_NAMES = frozenset({"CON", "PRN", "AUX", "NUL"})
_WINDOWS_RESERVED_NUMBERED_RE = re.compile(r"(?:COM|LPT)[1-9]")
_WINDOWS_ILLEGAL_CHARACTERS = frozenset('<>:"|?*')
_JOURNAL_SCHEMA_VERSION = 1
_ROLLBACK_GUARD_SCHEMA_VERSION = 1
RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION = 1
_CANONICAL_JSON_ARTIFACTS = frozenset(
    {
        "data_manifest.json",
        "execution_assumptions.json",
        "compiled_plan.json",
        "spec_audit.json",
        "runtime_audit.json",
        "component_manifest.json",
        "component_manifests.json",
    }
)
_BASE_BINDINGS_BY_SCHEMA = {
    0: frozenset({"data_manifest.json", "equity_curve.csv", "trades.csv", "metrics.json"}),
    1: frozenset(
        {
            "data_manifest.json",
            "equity_curve.csv",
            "trades.csv",
            "metrics.json",
            "strategy_spec.yaml",
            "environment.json",
            "positions.csv",
            "orders.csv",
        }
    ),
    2: frozenset(
        {
            "data_manifest.json",
            "equity_curve.csv",
            "trades.csv",
            "metrics.json",
            "strategy_spec.yaml",
            "environment.json",
            "execution_assumptions.json",
            "positions.csv",
            "orders.csv",
        }
    ),
    3: frozenset(
        {
            "data_manifest.json",
            "equity_curve.csv",
            "trades.csv",
            "metrics.json",
            "strategy_spec.yaml",
            "environment.json",
            "execution_assumptions.json",
            "positions.csv",
            "orders.csv",
            "target_weights.csv",
        }
    ),
    4: frozenset(
        {
            "data_manifest.json",
            "equity_curve.csv",
            "trades.csv",
            "metrics.json",
            "strategy_spec.yaml",
            "environment.json",
            "execution_assumptions.json",
            "positions.csv",
            "orders.csv",
            "target_weights.csv",
            "compiled_plan.json",
        }
    ),
    5: frozenset(
        {
            "data_manifest.json",
            "equity_curve.csv",
            "trades.csv",
            "metrics.json",
            "strategy_spec.yaml",
            "environment.json",
            "execution_assumptions.json",
            "positions.csv",
            "orders.csv",
            "target_weights.csv",
            "compiled_plan.json",
            "strategy.py",
        }
    ),
}
_EXTENSION_BINDINGS = frozenset(
    {
        "benchmark_curve.csv",
        "benchmark_equity_curve.csv",
        "benchmark_prices.csv",
        "robustness.json",
        "reproducibility_audit.json",
        "research_bias_audit.json",
        "spec_audit.json",
        "runtime_audit.json",
        "conversation_hash.txt",
        "component_catalog_hash.txt",
        "recipe_catalog_hash.txt",
        "component_manifest.json",
        "component_manifests.json",
        "component_bundle_hash.txt",
    }
)
_PROVENANCE_BINDING_GROUP = frozenset(
    {
        "spec_audit.json",
        "conversation_hash.txt",
        "component_catalog_hash.txt",
        "recipe_catalog_hash.txt",
    }
)
_COMPONENT_BINDING_GROUP = frozenset(
    {
        "component_manifest.json",
        "component_manifests.json",
        "component_bundle_hash.txt",
    }
)
_ALL_BASE_BINDINGS = frozenset().union(*_BASE_BINDINGS_BY_SCHEMA.values())
_ALL_REGISTERED_BINDINGS = _ALL_BASE_BINDINGS | _EXTENSION_BINDINGS
_active_digest_boundary_label: str | None = None


class RunDigestError(ValueError):
    """Raised when a run digest registry is malformed or inconsistent."""


class ArtifactHashesError(RunDigestError):
    """Raised when an artifact hash manifest cannot be updated safely."""


def _canonical_existing_path(path: str | Path) -> Path:
    supplied = Path(path)
    resolved = supplied.resolve(strict=True)
    canonical = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        requested = canonical / part
        requested_status = requested.stat()
        exact_match: Path | None = None
        portable_match: Path | None = None
        for child in canonical.iterdir():
            if child.name == part:
                exact_match = child
                break
            if _portable_component_key(child.name) != _portable_component_key(part):
                continue
            try:
                child_status = child.stat()
            except OSError:
                continue
            if os.path.samestat(child_status, requested_status):
                portable_match = child
        canonical = exact_match or portable_match or requested
    return canonical


def _canonical_run_path(run_dir: str | Path) -> Path:
    supplied = Path(run_dir)
    try:
        resolved = _canonical_existing_path(supplied)
    except (OSError, RuntimeError) as exc:
        raise RunDigestError(f"run directory could not be resolved: {supplied}: {exc}") from exc
    if not resolved.is_dir() or not _is_normal_run_id(resolved.name):
        raise RunDigestError(f"run directory must resolve to a real, normally named directory: {supplied}")
    return resolved


@dataclass(frozen=True)
class RunArtifactInventoryProfile:
    """Resolved schema for one versioned ``artifact_hashes.json`` manifest.

    ``contract_schema_version`` versions this executable inventory contract;
    ``artifact_hashes_schema_version`` is the authoritative profile selector
    read from the run manifest. ``required_bindings`` must be present exactly,
    ``optional_bindings`` are the only additional accepted hash keys, and
    ``bind_when_present`` lists governed optional files that may be absent but
    may not exist unbound in the run directory.
    """

    contract_schema_version: int
    name: str
    artifact_hashes_schema_version: int
    required_bindings: frozenset[str]
    optional_bindings: frozenset[str]
    bind_when_present: frozenset[str]


def _build_inventory_profiles() -> dict[int, RunArtifactInventoryProfile]:
    profiles: dict[int, RunArtifactInventoryProfile] = {}
    for schema_version, required in _BASE_BINDINGS_BY_SCHEMA.items():
        accepted = _ALL_REGISTERED_BINDINGS - required
        bind_when_present = set(_EXTENSION_BINDINGS)
        if schema_version >= 2 and "target_weights.csv" not in required:
            bind_when_present.add("target_weights.csv")
        profiles[schema_version] = RunArtifactInventoryProfile(
            contract_schema_version=RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION,
            name=("artifact_hashes_v0_legacy" if schema_version == 0 else f"artifact_hashes_v{schema_version}"),
            artifact_hashes_schema_version=schema_version,
            required_bindings=required,
            optional_bindings=accepted,
            bind_when_present=frozenset(bind_when_present),
        )
    return profiles


RUN_ARTIFACT_INVENTORY_PROFILES = _build_inventory_profiles()


@contextmanager
def run_digest_transaction(run_dir: str | Path) -> Iterator[None]:
    """Lock one run registry and recover any interrupted publication."""
    run_path = _canonical_run_path(run_dir)
    digest_path = run_path.parent / "run_digests.jsonl"
    lock_path = digest_path.with_suffix(digest_path.suffix + ".lock")
    selection_lock_path = final_selection_lock_path(run_path)
    with ProcessFileLock(lock_path):
        with hold_final_selection_lock(selection_lock_path):
            _recover_publication_locked(run_path.parent)
            yield


@contextmanager
def multi_run_digest_read_transaction(
    run_dirs: Collection[str | Path],
) -> Iterator[tuple[Path, ...]]:
    """Hold one coherent, validated read transaction across multiple runs.

    Canonical paths are yielded in input order, including repeated aliases so
    callers can preserve left/right argument identity. Lock acquisition is
    deduplicated and globally ordered: all unique run-digest locks sorted by
    canonical path, then all unique final-selection locks sorted by canonical
    path. Interrupted publications are recovered once per run parent and every
    unique run digest is validated before control is yielded to the caller.
    """
    canonical = tuple(_canonical_run_path(run_dir) for run_dir in run_dirs)
    unique_runs = sorted(_unique_paths_by_location(canonical), key=_path_order_key)
    digest_lock_paths = sorted(
        _unique_paths_by_location(
            [
                (run_path.parent / "run_digests.jsonl.lock").resolve(strict=False)
                for run_path in unique_runs
            ]
        ),
        key=_path_order_key,
    )
    selection_lock_paths = sorted(
        _unique_paths_by_location(
            [
                lock_path
                for run_path in unique_runs
                if (lock_path := final_selection_lock_path(run_path)) is not None
            ]
        ),
        key=_path_order_key,
    )
    with ExitStack() as stack:
        for lock_path in digest_lock_paths:
            stack.enter_context(ProcessFileLock(lock_path))
        for lock_path in selection_lock_paths:
            stack.enter_context(hold_final_selection_lock(lock_path))
        parents = _unique_paths_by_location([run_path.parent for run_path in unique_runs])
        for parent in sorted(parents, key=_path_order_key):
            _recover_publication_locked(parent)
        for run_path in unique_runs:
            _require_current_run_digest_locked(run_path)
        yield canonical


def _unique_paths_by_location(paths: Collection[Path]) -> list[Path]:
    unique: list[Path] = []
    for path in paths:
        if not any(_paths_share_location_or_file_identity(path, previous) for previous in unique):
            unique.append(path)
    return unique


def _paths_share_location_or_file_identity(left: Path, right: Path) -> bool:
    if stable_path_location_identity(left) == stable_path_location_identity(right):
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _path_order_key(path: Path) -> str:
    return os.path.normcase(str(path))


def replace_run_digest_entry(run_dir: str | Path, artifact_hashes_hash: str) -> None:
    """Atomically replace all entries for one run with one canonical entry."""
    run_path = _canonical_run_path(run_dir)
    if _DIGEST_RE.fullmatch(artifact_hashes_hash) is None:
        raise RunDigestError(f"invalid artifact_hashes digest: {artifact_hashes_hash}")

    with run_digest_transaction(run_path):
        digest_path = run_path.parent / "run_digests.jsonl"
        new_content = _render_run_digest_content(run_path, artifact_hashes_hash)
        _commit_journaled_targets(
            run_path,
            [_journal_target("digest", digest_path, new_content.encode())],
            lambda: _replace_run_digest_entry_locked(run_path, artifact_hashes_hash),
            expected_digest=artifact_hashes_hash,
            validate_completed=lambda: _validate_completed_run_state(run_path, artifact_hashes_hash),
        )


def _replace_run_digest_entry_locked(run_path: Path, artifact_hashes_hash: str) -> None:
    content = _render_run_digest_content(run_path, artifact_hashes_hash)
    digest_path = run_path.parent / "run_digests.jsonl"
    _atomic_write_text(digest_path, content, boundary=_active_digest_boundary_label)


def _render_run_digest_content(run_path: Path, artifact_hashes_hash: str) -> str:
    digest_path = run_path.parent / "run_digests.jsonl"
    entries = _read_entries(digest_path) if digest_path.exists() else []
    retained: list[dict[str, Any]] = []
    existing: dict[str, Any] | None = None
    for entry in entries:
        if entry["run_id"] == run_path.name:
            if existing is None:
                existing = entry
            continue
        retained.append(entry)

    created_at = existing.get("created_at") if existing is not None else None
    if not isinstance(created_at, str) or not created_at:
        created_at = datetime.now(UTC).isoformat()
    artifact_inventory = existing.get("artifact_inventory") if existing is not None else None
    if artifact_inventory is None:
        artifact_inventory = _derive_inventory_registry_metadata(run_path)
    inventory_fields = {"artifact_inventory": artifact_inventory} if artifact_inventory is not None else {}
    retained.append(
        {
            **(existing or {}),
            "run_id": run_path.name,
            "artifact_hashes": artifact_hashes_hash,
            "created_at": created_at,
            **inventory_fields,
        }
    )
    return "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in retained)


def publish_run_artifacts(
    run_dir: str | Path,
    artifacts: Mapping[str, bytes],
    *,
    canonical_json: Collection[str] = (),
    replacement_paths: Mapping[str, str | Path | None] | None = None,
    remove_artifacts: Collection[str] = (),
) -> str:
    """Publish run artifacts and replacement paths in one transaction."""
    run_path = _canonical_run_path(run_dir)
    artifact_contents = dict(artifacts)
    replacement_sources = dict(replacement_paths or {})
    removed_names = frozenset(remove_artifacts)
    canonical_names = frozenset(canonical_json)
    if not artifact_contents and not replacement_sources and not removed_names:
        raise ArtifactHashesError("at least one run artifact, replacement path, or artifact removal is required")
    if not canonical_names <= artifact_contents.keys():
        raise ArtifactHashesError("canonical_json names must refer to published artifacts")
    incompatible_canonical_names = canonical_names - _CANONICAL_JSON_ARTIFACTS
    if incompatible_canonical_names:
        raise ArtifactHashesError(
            "canonical_json cannot override artifact hash contracts: "
            f"{sorted(incompatible_canonical_names)}"
        )
    for name, content in artifact_contents.items():
        _validate_artifact_name(name)
        if name == "artifact_hashes.json":
            raise ArtifactHashesError("artifact_hashes.json is managed by the publication transaction")
        if not isinstance(content, bytes):
            raise ArtifactHashesError(f"published artifact content must be bytes: {name}")
    if removed_names & artifact_contents.keys():
        raise ArtifactHashesError("published and removed artifact names must not overlap")
    for name in removed_names:
        _validate_artifact_name(name)
        if name == "artifact_hashes.json":
            raise ArtifactHashesError("artifact_hashes.json is managed by the publication transaction")
    for name in replacement_sources:
        _validate_replacement_name(name)
    _require_no_portable_path_collisions(
        [*artifact_contents, *removed_names, *replacement_sources]
    )
    _validate_replacement_sources(run_path, replacement_sources, artifact_contents, removed_names)

    def update(manifest: dict[str, Any]) -> None:
        for name in removed_names:
            manifest.pop(name, None)
        for name, content in artifact_contents.items():
            manifest[name] = _hash_bound_artifact(name, content)

    return _publish_run_state(
        run_path,
        artifact_contents,
        update,
        replacement_sources=replacement_sources,
        removed_artifacts=removed_names,
    )


def update_artifact_hashes_and_run_digest(
    run_dir: str | Path,
    update: Callable[[dict[str, Any]], None],
) -> str:
    """Atomically publish an artifact-hash update and its registry digest."""
    run_path = _canonical_run_path(run_dir)
    return _publish_run_state(run_path, {}, update)


def _update_artifact_hashes_and_run_digest_locked(
    run_path: Path,
    update: Callable[[dict[str, Any]], None],
) -> str:
    """Publish a manifest/digest pair inside an existing run transaction."""
    manifest_path = run_path / "artifact_hashes.json"
    artifact_hashes = _read_artifact_hashes(manifest_path)
    _validate_inventory_binding_names(artifact_hashes)
    updated_hashes = dict(artifact_hashes)
    update(updated_hashes)
    _validate_inventory_binding_names(updated_hashes)
    _require_no_portable_path_collisions(
        [
            name
            for name in updated_hashes
            if isinstance(name, str) and name != "schema_version"
        ]
    )
    manifest_content = (json.dumps(updated_hashes, indent=2, sort_keys=True) + "\n").encode()
    digest = _hash_json_payload(updated_hashes)
    digest_content = _render_run_digest_content(run_path, digest)
    _atomic_write_bytes(manifest_path, manifest_content)
    _atomic_write_text(run_path.parent / "run_digests.jsonl", digest_content)
    _validate_completed_run_state(run_path, digest)
    return digest


def _publish_run_state(
    run_path: Path,
    artifact_contents: Mapping[str, bytes],
    update: Callable[[dict[str, Any]], None],
    *,
    replacement_sources: Mapping[str, str | Path | None] | None = None,
    removed_artifacts: Collection[str] = (),
) -> str:
    run_path = _canonical_run_path(run_path)
    manifest_path = run_path / "artifact_hashes.json"
    with run_digest_transaction(run_path):
        artifact_hashes = _read_artifact_hashes(manifest_path)
        _validate_inventory_binding_names(artifact_hashes)
        updated_hashes = dict(artifact_hashes)
        update(updated_hashes)
        _validate_inventory_binding_names(updated_hashes)
        _require_no_portable_path_collisions(
            [
                *(
                    name
                    for name in updated_hashes
                    if isinstance(name, str) and name != "schema_version"
                ),
                *(replacement_sources or {}),
            ]
        )
        manifest_content = (json.dumps(updated_hashes, indent=2, sort_keys=True) + "\n").encode()
        digest = _hash_json_payload(updated_hashes)
        digest_content = _render_run_digest_content(run_path, digest).encode()
        path_targets = [
            _journal_path_target(run_path, name, source)
            for name, source in (replacement_sources or {}).items()
        ]
        artifact_targets = [
            *(
                _journal_target("artifact", _safe_run_artifact_path(run_path, name), content, name=name)
                for name, content in artifact_contents.items()
            ),
            *(
                _journal_target("artifact", _safe_run_artifact_path(run_path, name), None, name=name)
                for name in sorted(removed_artifacts)
            ),
        ]
        targets = [
            *path_targets,
            *artifact_targets,
            _journal_target("manifest", manifest_path, manifest_content),
            _journal_target("digest", run_path.parent / "run_digests.jsonl", digest_content),
        ]

        def commit() -> None:
            global _active_digest_boundary_label
            for target in path_targets:
                _atomic_replace_path_snapshot(
                    _target_path(run_path, target),
                    _validated_path_snapshot(target.get("new_path")),
                    boundary=f"path:{target['name']}",
                )
            for target in artifact_targets:
                target_path = _target_path(run_path, target)
                content = _decode_content(target["new"])
                if content is None:
                    _atomic_remove_file(target_path, boundary=f"artifact:{target['name']}")
                else:
                    _atomic_write_bytes(
                        target_path,
                        content,
                        boundary=f"artifact:{target['name']}",
                    )
            _atomic_write_bytes(manifest_path, manifest_content, boundary="manifest")
            previous_label = _active_digest_boundary_label
            _active_digest_boundary_label = "digest"
            try:
                _replace_run_digest_entry_locked(run_path, digest)
            finally:
                _active_digest_boundary_label = previous_label

        _commit_journaled_targets(
            run_path,
            targets,
            commit,
            expected_digest=digest,
            validate_completed=lambda: _validate_completed_run_state(run_path, digest),
        )
        return digest


def _validate_completed_run_state(run_path: Path, expected_digest: str) -> None:
    artifact_hashes = _read_artifact_hashes(run_path / "artifact_hashes.json")
    actual_digest = _hash_json_payload(artifact_hashes)
    if actual_digest != expected_digest:
        raise RunDigestError(
            "completed publication artifact_hashes digest mismatch: "
            f"expected={expected_digest}, actual={actual_digest}"
        )

    digest_path = run_path.parent / "run_digests.jsonl"
    entries = _read_entries(digest_path)
    matches = [entry for entry in entries if entry["run_id"] == run_path.name]
    if len(matches) != 1:
        raise RunDigestError(
            f"completed publication must contain exactly one digest row for run_id {run_path.name}; found {len(matches)}"
        )
    entry = matches[0]
    if entry["artifact_hashes"] != expected_digest:
        raise RunDigestError(
            "completed publication digest row mismatch: "
            f"stored={entry['artifact_hashes']}, expected={expected_digest}"
        )

    artifact_inventory = entry.get("artifact_inventory")
    if artifact_inventory is not None or _uses_versioned_artifact_inventory(artifact_hashes):
        profile = _validate_run_artifact_inventory(run_path, artifact_hashes)
        if artifact_inventory is not None:
            _require_registry_inventory_profile(artifact_inventory, profile)
    else:
        _require_bound_artifacts_current(run_path, artifact_hashes)


def require_current_run_digest(run_dir: str | Path) -> None:
    """Require one current digest and a valid versioned artifact inventory."""
    run_path = _canonical_run_path(run_dir)
    with run_digest_transaction(run_path):
        _require_current_run_digest_locked(run_path)


def _require_current_run_digest_locked(run_path: Path) -> None:
    digest_path = run_path.parent / "run_digests.jsonl"
    entries = _read_entries(digest_path) if digest_path.exists() else []
    matches = [entry for entry in entries if entry["run_id"] == run_path.name]
    if len(matches) != 1:
        raise RunDigestError(
            f"run_digests.jsonl must contain exactly one valid entry for run_id {run_path.name}; found {len(matches)}: {digest_path}"
        )

    expected = matches[0]["artifact_hashes"]
    manifest_path = run_path / "artifact_hashes.json"
    artifact_hashes = _read_artifact_hashes(manifest_path)
    actual = _hash_json_payload(artifact_hashes)
    if actual != expected:
        raise RunDigestError(f"run digest mismatch for artifact_hashes.json: stored={expected}, actual={actual}")
    artifact_inventory = matches[0].get("artifact_inventory")
    if artifact_inventory is not None or _uses_versioned_artifact_inventory(artifact_hashes):
        profile = _validate_run_artifact_inventory(run_path, artifact_hashes)
        if artifact_inventory is not None:
            _require_registry_inventory_profile(artifact_inventory, profile)
    else:
        _require_bound_artifacts_current(run_path, artifact_hashes)


def validate_run_artifact_inventory(run_dir: str | Path) -> RunArtifactInventoryProfile:
    """Validate and return the authoritative run-artifact inventory profile.

    Contract schema ``1`` derives its profile only from
    ``artifact_hashes.json.schema_version``; callers cannot request a weaker
    profile. Supported artifact-hash schemas are ``0`` through ``5``. Schema
    ``0`` is the explicit legacy profile and is accepted only with an absent
    or zero ``data_manifest.json.schema_version``. A missing artifact-hash
    schema is accepted as legacy only under that same data-manifest rule.

    Every profile has exact ``required_bindings`` and a closed
    ``optional_bindings`` set. Unknown bindings, non-canonical relative POSIX
    paths (including ``./name`` aliases), incomplete provenance bundles, and
    governed optional files present without a binding are rejected. Every
    accepted binding must resolve to a current regular, non-symlink file
    inside the run directory. The return value exposes contract schema,
    profile name, source artifact-hash schema, required and optional sets, and
    the optional files governed by the bind-when-present rule.
    """
    run_path = _canonical_run_path(run_dir)
    artifact_hashes = _read_artifact_hashes(run_path / "artifact_hashes.json")
    return _validate_run_artifact_inventory(run_path, artifact_hashes)


def _validate_run_artifact_inventory(
    run_path: Path,
    artifact_hashes: Mapping[str, Any],
) -> RunArtifactInventoryProfile:
    profile = _resolve_inventory_profile(run_path, artifact_hashes)
    binding_names = _validate_inventory_binding_names(artifact_hashes)

    missing = sorted(profile.required_bindings - binding_names)
    if missing:
        raise RunDigestError(f"{profile.name} is missing required bindings: {missing}")

    unregistered = sorted(binding_names - profile.required_bindings - profile.optional_bindings)
    if unregistered:
        raise RunDigestError(f"artifact_hashes.json contains unregistered artifact bindings: {unregistered}")

    provenance_present = binding_names & _PROVENANCE_BINDING_GROUP
    if provenance_present and provenance_present != _PROVENANCE_BINDING_GROUP:
        missing_provenance = sorted(_PROVENANCE_BINDING_GROUP - provenance_present)
        raise RunDigestError(f"artifact_hashes.json has an incomplete provenance binding group: {missing_provenance}")

    unbound_governed = sorted(name for name in profile.bind_when_present - binding_names if _run_artifact_present(run_path, name))
    if unbound_governed:
        raise RunDigestError(f"artifact_hashes.json has unbound governed files: {unbound_governed}")

    _require_bound_artifacts_current(run_path, artifact_hashes)
    _validate_component_provenance(run_path, binding_names)
    return profile


def _validate_component_provenance(run_path: Path, binding_names: frozenset[str]) -> None:
    component_bindings = binding_names & _COMPONENT_BINDING_GROUP
    if not component_bindings:
        return
    if "component_manifests.json" not in component_bindings:
        raise RunDigestError("artifact_hashes.json has an incomplete component provenance group: component_manifests.json")

    summary = _read_component_summary(run_path)
    has_legacy = "component_manifest.json" in component_bindings
    has_text_hash = "component_bundle_hash.txt" in component_bindings
    if not summary:
        if has_legacy or has_text_hash:
            raise RunDigestError("bare component provenance cannot contain single-component bindings")
        return
    single_component = len(summary) == 1
    if not single_component and (has_legacy or has_text_hash):
        raise RunDigestError("multi-component provenance must use component_manifests.json without single-component bindings")

    verified_hashes = [
        _verify_component_summary_entry(run_path, item, index, len(summary), has_legacy)
        for index, item in enumerate(summary)
    ]
    if has_legacy:
        _verify_component_manifest_hash(run_path / "component_manifest.json", verified_hashes[0], "component_manifest.json")
    if has_text_hash:
        text_path = _safe_run_artifact_path(run_path, "component_bundle_hash.txt")
        content = _read_regular_file_bytes(text_path, artifact_name="component_bundle_hash.txt")
        try:
            text_hash = content.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise RunDigestError(f"component_bundle_hash.txt is invalid: {exc}") from exc
        if text_hash != verified_hashes[0]:
            raise RunDigestError(
                "component_bundle_hash.txt hash mismatch: "
                f"stored={text_hash}, verified_component_bundle={verified_hashes[0]}"
            )


def _read_component_summary(run_path: Path) -> list[Any]:
    path = _safe_run_artifact_path(run_path, "component_manifests.json")
    content = _read_regular_file_bytes(path, artifact_name="component_manifests.json")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunDigestError(f"component_manifests.json is invalid: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise RunDigestError("component_manifests.json must contain a JSON list")
    return payload


def _verify_component_summary_entry(
    run_path: Path,
    item: Any,
    index: int,
    summary_count: int,
    has_legacy: bool,
) -> str:
    if not isinstance(item, dict):
        raise RunDigestError(f"component_manifests.json[{index}] must be an object")
    manifest_reference = item.get("manifest_path")
    if not isinstance(manifest_reference, str) or not manifest_reference:
        raise RunDigestError(f"component_manifests.json[{index}].manifest_path is required")
    recorded = item.get("bundle_hash")
    if not isinstance(recorded, str) or _DIGEST_RE.fullmatch(recorded) is None:
        raise RunDigestError(f"component_manifests.json[{index}].bundle_hash is invalid")
    archived_path = item.get("archived_manifest_path")
    if isinstance(archived_path, str) and archived_path:
        manifest_path = _safe_run_artifact_path(run_path, archived_path)
        artifact_name = archived_path
    elif summary_count == 1 and has_legacy:
        manifest_path = _safe_run_artifact_path(run_path, "component_manifest.json")
        artifact_name = "component_manifest.json"
    else:
        raise RunDigestError(f"component_manifests.json[{index}] has no verifiable in-run component manifest")
    _verify_component_manifest_hash(manifest_path, recorded, artifact_name)
    return recorded


def _verify_component_manifest_hash(path: Path, recorded: str, artifact_name: str) -> None:
    from oxq.core.component_manifest import compute_component_bundle_hash

    content = _read_regular_file_bytes(path, artifact_name=artifact_name)
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunDigestError(f"component manifest is invalid: {artifact_name}: {exc}") from exc
    manifest_hash = payload.get("bundle_hash") if isinstance(payload, dict) else None
    try:
        actual = compute_component_bundle_hash(path)
    except (OSError, ValueError) as exc:
        raise RunDigestError(f"component manifest could not be verified: {artifact_name}: {exc}") from exc
    if manifest_hash != recorded or actual != recorded:
        raise RunDigestError(
            f"component manifest hash mismatch: {artifact_name}: "
            f"summary={recorded}, manifest={manifest_hash}, actual={actual}"
        )


def _resolve_inventory_profile(
    run_path: Path,
    artifact_hashes: Mapping[str, Any],
) -> RunArtifactInventoryProfile:
    raw_schema_version = artifact_hashes.get("schema_version")
    if raw_schema_version is None:
        schema_version = 0
    elif isinstance(raw_schema_version, bool) or not isinstance(raw_schema_version, int):
        raise RunDigestError(
            f"unsupported artifact_hashes.json schema_version: {raw_schema_version!r}; supported={sorted(RUN_ARTIFACT_INVENTORY_PROFILES)}"
        )
    else:
        schema_version = raw_schema_version
    try:
        profile = RUN_ARTIFACT_INVENTORY_PROFILES[schema_version]
    except KeyError as exc:
        raise RunDigestError(
            f"unsupported artifact_hashes.json schema_version: {schema_version!r}; supported={sorted(RUN_ARTIFACT_INVENTORY_PROFILES)}"
        ) from exc

    if schema_version == 0:
        data_manifest_version = _read_data_manifest_schema_version(run_path)
        if data_manifest_version > 0:
            raise RunDigestError(
                f"current data_manifest.json cannot use legacy artifact inventory: data_manifest schema_version={data_manifest_version}"
            )
    return profile


def _read_data_manifest_schema_version(run_path: Path) -> int:
    path = _safe_run_artifact_path(run_path, "data_manifest.json")
    content = _read_regular_file_bytes(path, artifact_name="data_manifest.json")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunDigestError(f"data_manifest.json is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunDigestError(f"data_manifest.json must contain a JSON object: {path}")
    raw_version = payload.get("schema_version", 0)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int) or raw_version < 0:
        raise RunDigestError(f"data_manifest.json has invalid schema_version: {raw_version!r}")
    return raw_version


def _validate_inventory_binding_names(artifact_hashes: Mapping[str, Any]) -> frozenset[str]:
    binding_names: set[str] = set()
    for raw_name in artifact_hashes:
        if raw_name == "schema_version":
            continue
        if not isinstance(raw_name, str):
            raise RunDigestError(f"artifact_hashes.json has an invalid artifact binding: {raw_name!r}")
        if _portable_posix_relative_parts(raw_name) is None:
            raise RunDigestError(f"artifact_hashes.json contains a non-canonical artifact path: {raw_name!r}")
        binding_names.add(raw_name)
    _require_no_portable_path_collisions(binding_names, error_type=RunDigestError)
    return frozenset(binding_names)


def _run_artifact_present(run_path: Path, name: str) -> bool:
    path = run_path / name
    return path.exists() or path.is_symlink()


def _uses_versioned_artifact_inventory(artifact_hashes: Mapping[str, Any]) -> bool:
    return "schema_version" in artifact_hashes or "data_manifest.json" in artifact_hashes


def _derive_inventory_registry_metadata(run_path: Path) -> dict[str, object] | None:
    manifest_path = run_path / "artifact_hashes.json"
    if not manifest_path.exists() or manifest_path.is_symlink():
        return None
    artifact_hashes = _read_artifact_hashes(manifest_path)
    if not _uses_versioned_artifact_inventory(artifact_hashes):
        return None
    profile = _resolve_inventory_profile(run_path, artifact_hashes)
    return {
        "schema_version": profile.contract_schema_version,
        "profile": profile.name,
    }


def _require_registry_inventory_profile(
    artifact_inventory: object,
    profile: RunArtifactInventoryProfile,
) -> None:
    if not isinstance(artifact_inventory, dict):
        raise RunDigestError("run digest artifact_inventory must be a JSON object")
    schema_version = artifact_inventory.get("schema_version")
    stored_profile = artifact_inventory.get("profile")
    if schema_version != RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION or not isinstance(stored_profile, str):
        raise RunDigestError(f"run digest artifact_inventory has an unsupported schema: {artifact_inventory!r}")
    if stored_profile != profile.name:
        raise RunDigestError(f"run digest inventory profile mismatch: stored={stored_profile}, actual={profile.name}")


def _read_entries(digest_path: Path) -> list[dict[str, Any]]:
    parent = digest_path.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise RunDigestError(f"run_digests.jsonl parent must be a canonical directory: {parent}")
    if digest_path.is_symlink():
        raise RunDigestError(f"run_digests.jsonl must be a regular, non-symlink file: {digest_path}")
    try:
        content = _read_regular_file_bytes(digest_path, artifact_name="run_digests.jsonl")
        if parent.is_symlink() or parent.resolve(strict=True) != parent:
            raise RunDigestError(f"run_digests.jsonl parent must be a canonical directory: {parent}")
    except (OSError, RunDigestError) as exc:
        raise RunDigestError(f"run_digests.jsonl is invalid: {digest_path}: {exc}") from exc
    return _parse_run_digest_content(content, digest_path)


def _parse_run_digest_content(content: bytes, digest_path: Path) -> list[dict[str, Any]]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RunDigestError(f"run_digests.jsonl is invalid: {digest_path}: {exc}") from exc

    entries: list[dict[str, Any]] = []
    run_ids_by_portable_key: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunDigestError(f"run_digests.jsonl entry {line_number} is invalid JSON: {digest_path}: {exc.msg}") from exc
        if not isinstance(entry, dict):
            raise RunDigestError(f"run_digests.jsonl entry {line_number} must be a JSON object: {digest_path}")
        run_id = entry.get("run_id")
        if not _is_normal_run_id(run_id):
            raise RunDigestError(f"run_digests.jsonl entry {line_number} has invalid run_id: {digest_path}")
        assert isinstance(run_id, str)
        portable_key = _portable_component_key(run_id)
        previous_run_id = run_ids_by_portable_key.get(portable_key)
        if previous_run_id is not None and previous_run_id != run_id:
            raise RunDigestError(
                f"run_digests.jsonl has a portable run_id collision: {previous_run_id!r} and {run_id!r}"
            )
        run_ids_by_portable_key[portable_key] = run_id
        artifact_hashes = entry.get("artifact_hashes")
        if not isinstance(artifact_hashes, str) or _DIGEST_RE.fullmatch(artifact_hashes) is None:
            raise RunDigestError(f"run_digests.jsonl entry {line_number} has invalid artifact_hashes: {digest_path}")
        entries.append(entry)
    return entries


def _hash_json_file(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunDigestError(f"artifact_hashes.json is invalid: {path}: {exc}") from exc
    return _hash_json_payload(payload)


def _hash_json_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _read_artifact_hashes(path: Path) -> dict[str, Any]:
    try:
        content = _read_regular_file_bytes(path, artifact_name="artifact_hashes.json")
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=_json_object_without_duplicates)
    except (RunDigestError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactHashesError(f"artifact_hashes.json is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactHashesError(f"artifact_hashes.json must contain a JSON object: {path}")
    return payload


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise RunDigestError(f"artifact_hashes.json contains duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _require_bound_artifacts_current(run_path: Path, artifact_hashes: Mapping[str, Any]) -> None:
    _require_no_portable_path_collisions(
        [name for name in artifact_hashes if isinstance(name, str) and name != "schema_version"],
        error_type=RunDigestError,
    )
    for name, expected in artifact_hashes.items():
        if name == "schema_version":
            continue
        if not isinstance(name, str) or not isinstance(expected, str) or _DIGEST_RE.fullmatch(expected) is None:
            raise RunDigestError(f"artifact_hashes.json has an invalid artifact binding: {name!r}")
        path = _safe_run_artifact_path(run_path, name)
        content = _read_regular_file_bytes(path, artifact_name=name)
        actual = _hash_bound_artifact(name, content)
        if actual != expected:
            raise RunDigestError(f"{name} hash mismatch: stored={expected}, actual={actual}")


def _hash_bound_artifact(name: str, content: bytes) -> str:
    if name == "metrics.json":
        return _hash_json_bytes(content, name=name, exclude_keys={"run_id"})
    if name == "environment.json":
        return _hash_json_bytes(content, name=name, exclude_keys={"run_timestamp"})
    if name in _CANONICAL_JSON_ARTIFACTS:
        return _hash_json_bytes(content, name=name)
    return _hash_bytes(content)


def _hash_json_bytes(content: bytes, *, name: str, exclude_keys: set[str] | None = None) -> str:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactHashesError(f"published JSON artifact is invalid: {name}: {exc}") from exc
    if isinstance(payload, dict) and exclude_keys:
        payload = {key: value for key, value in payload.items() if key not in exclude_keys}
    return _hash_json_payload(payload)


def _hash_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"


def _portable_component_key(value: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFKC", value).casefold())


def _portable_relative_key(name: object) -> tuple[str, ...] | None:
    parts = _portable_posix_relative_parts(name)
    if parts is None:
        return None
    return tuple(_portable_component_key(part) for part in parts)


def _portable_posix_relative_parts(name: object) -> tuple[str, ...] | None:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        return None
    raw_parts = tuple(name.split("/"))
    if any(part in {"", ".", ".."} for part in raw_parts):
        return None
    for part in raw_parts:
        if part.endswith((".", " ")):
            return None
        if any(ord(character) < 32 or character in _WINDOWS_ILLEGAL_CHARACTERS for character in part):
            return None
        device_name = _portable_component_key(part.split(".", 1)[0]).upper()
        if device_name in _WINDOWS_RESERVED_NAMES or _WINDOWS_RESERVED_NUMBERED_RE.fullmatch(device_name):
            return None
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or posix_path.parts != raw_parts
        or windows_path.parts != raw_parts
        or any(PureWindowsPath(part).drive for part in raw_parts)
    ):
        return None
    return raw_parts


def _require_no_portable_path_collisions(
    names: Collection[str],
    *,
    error_type: type[RunDigestError] = ArtifactHashesError,
) -> None:
    keyed: list[tuple[tuple[str, ...], str]] = []
    for name in names:
        key = _portable_relative_key(name)
        if key is None:
            continue
        for previous_key, previous_name in keyed:
            if name == previous_name:
                continue
            shared = min(len(key), len(previous_key))
            if key[:shared] == previous_key[:shared]:
                raise error_type(f"portable path collision: {previous_name!r} and {name!r}")
        keyed.append((key, name))


def _validate_replacement_sources(
    run_path: Path,
    replacement_sources: Mapping[str, str | Path | None],
    artifact_contents: Mapping[str, bytes],
    removed_artifacts: Collection[str],
) -> None:
    manifest_path = run_path / "artifact_hashes.json"
    digest_path = run_path.parent / "run_digests.jsonl"
    journal_path = _journal_path(run_path.parent)
    target_paths: list[Path] = [manifest_path, digest_path, digest_path.with_suffix(".jsonl.lock"), journal_path]
    path_targets: list[Path] = []
    atomic_targets: list[Path] = [manifest_path, digest_path, journal_path]
    sources: list[tuple[str, Path]] = []
    for name, source in replacement_sources.items():
        _validate_replacement_name(name)
        target = _safe_run_replacement_path(run_path, name)
        target_paths.append(target)
        path_targets.append(target)
        if source is not None:
            sources.append((name, Path(source)))
    for name in artifact_contents:
        target = _safe_run_artifact_path(run_path, name)
        target_paths.append(target)
        atomic_targets.append(target)
    for name in removed_artifacts:
        target = _safe_run_artifact_path(run_path, name)
        target_paths.append(target)
        atomic_targets.append(target)
    _require_managed_transaction_targets(target_paths, path_targets=path_targets, atomic_targets=atomic_targets)
    for name, source in sources:
        _snapshot_path(source, label=f"replacement source {name}")


def _validate_replacement_name(name: object) -> tuple[str, ...]:
    parts = _portable_posix_relative_parts(name)
    if parts is None:
        raise ArtifactHashesError(f"unsafe replacement path: {name!r}")
    return parts


def _safe_run_replacement_path(run_path: Path, name: str) -> Path:
    parts = _validate_replacement_name(name)
    if run_path.is_symlink() or not run_path.is_dir():
        raise ArtifactHashesError(f"run directory must be a real directory: {run_path}")
    run_root = run_path.resolve(strict=False)
    candidate = run_path.joinpath(*parts)
    current = run_path
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ArtifactHashesError(f"replacement target contains a symlink: {name}")
    try:
        candidate.resolve(strict=False).relative_to(run_root)
    except ValueError as exc:
        raise ArtifactHashesError(f"unsafe replacement path: {name!r}") from exc
    return candidate


def _require_non_overlapping_paths(paths: Collection[Path], message: str) -> None:
    resolved_paths = [path.resolve(strict=False) for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ArtifactHashesError(f"{message}: duplicate target")
    resolved = sorted(resolved_paths, key=lambda path: (len(path.parts), str(path)))
    for index, parent in enumerate(resolved):
        for child in resolved[index + 1 :]:
            if child.is_relative_to(parent):
                raise ArtifactHashesError(f"{message}: {parent} and {child}")


def _require_managed_transaction_targets(
    target_paths: Collection[Path],
    *,
    path_targets: Collection[Path],
    atomic_targets: Collection[Path],
) -> None:
    expanded = list(target_paths)
    for target in path_targets:
        expanded.extend(
            (
                target.parent / f".{target.name}.oxq-path-new",
                target.parent / f".{target.name}.oxq-path-old",
            )
        )
    _require_non_overlapping_paths(expanded, "managed transaction targets overlap")

    resolved_targets = [path.resolve(strict=False) for path in target_paths]
    resolved_atomic_targets = [path.resolve(strict=False) for path in atomic_targets]
    for target in resolved_targets:
        for managed in resolved_atomic_targets:
            if target == managed or target.parent != managed.parent:
                continue
            if target.name.startswith(f".{managed.name}.") and target.name.endswith(".tmp"):
                raise ArtifactHashesError(
                    f"managed atomic temp namespace collision: {target} aliases temporary files for {managed}"
                )


def _validate_artifact_name(name: object) -> tuple[str, ...]:
    parts = _portable_posix_relative_parts(name)
    if parts is None:
        raise ArtifactHashesError(f"unsafe artifact path in artifact_hashes.json: {name!r}")
    return parts


def _safe_run_artifact_path(run_path: Path, name: str) -> Path:
    try:
        parts = _validate_artifact_name(name)
    except ArtifactHashesError as exc:
        raise RunDigestError(str(exc)) from exc
    if run_path.is_symlink() or not run_path.is_dir():
        raise RunDigestError(f"run directory must be a real directory: {run_path}")
    run_root = run_path.resolve(strict=False)
    candidate = run_path.joinpath(*parts)
    current = run_path
    for part in parts:
        current /= part
        if current.is_symlink():
            raise RunDigestError(f"bound artifact must be a regular, non-symlink file in the run: {name}")
    try:
        candidate.resolve(strict=False).relative_to(run_root)
    except ValueError as exc:
        raise RunDigestError(f"unsafe artifact path in artifact_hashes.json: {name!r}") from exc
    return candidate


def _read_regular_file_bytes(path: Path, *, artifact_name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RunDigestError(f"bound artifact must be a regular, non-symlink file in the run: {artifact_name}: {exc}") from exc
    try:
        before_descriptor = os.fstat(descriptor)
        if not stat.S_ISREG(before_descriptor.st_mode):
            raise RunDigestError(f"bound artifact must be a regular, non-symlink file in the run: {artifact_name}")
        before_path = _require_open_file_identity(path, before_descriptor, artifact_name)
        chunks: list[bytes] = []
        bytes_read = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            bytes_read += len(chunk)
        after_descriptor = os.fstat(descriptor)
        after_path = _require_open_file_identity(path, after_descriptor, artifact_name)
        if (
            _stable_read_metadata(before_descriptor) != _stable_read_metadata(after_descriptor)
            or _file_identity(before_path) != _file_identity(after_path)
            or bytes_read != before_descriptor.st_size
            or bytes_read != after_descriptor.st_size
        ):
            raise RunDigestError(f"bound artifact changed while being read: {artifact_name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_open_file_identity(
    path: Path,
    metadata: os.stat_result,
    artifact_name: str,
) -> os.stat_result:
    try:
        path_metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RunDigestError(
            f"bound artifact must remain a regular, non-symlink file in the run: {artifact_name}: {exc}"
        ) from exc
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISREG(path_metadata.st_mode)
        or (path_metadata.st_dev, path_metadata.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        raise RunDigestError(
            f"bound artifact must remain a regular, non-symlink file in the run: {artifact_name}"
        )
    return path_metadata


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _stable_read_metadata(metadata: os.stat_result) -> tuple[tuple[int, int], int, int | float, int | float]:
    return (
        _file_identity(metadata),
        int(metadata.st_size),
        getattr(metadata, "st_mtime_ns", metadata.st_mtime),
        getattr(metadata, "st_ctime_ns", metadata.st_ctime),
    )


def _journal_path(parent: Path) -> Path:
    return parent / "run_digests.jsonl.journal"


def _encode_content(content: bytes | None) -> str | None:
    return None if content is None else base64.b64encode(content).decode("ascii")


def _decode_content(content: object) -> bytes | None:
    if content is None:
        return None
    if not isinstance(content, str):
        raise RunDigestError("run publication journal contains invalid file content")
    try:
        return base64.b64decode(content, validate=True)
    except ValueError as exc:
        raise RunDigestError("run publication journal contains invalid base64 content") from exc


def _snapshot_path(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        raise ArtifactHashesError(f"{label} must not contain a symlink: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if path.is_file():
        return {
            "type": "file",
            "mode": mode,
            "content": _encode_content(_read_regular_file_bytes(path, artifact_name=label)),
        }
    if not path.is_dir():
        raise ArtifactHashesError(f"{label} must be a regular file or directory: {path}")
    entries: list[dict[str, Any]] = []
    seen_portable_paths: dict[tuple[str, ...], str] = {}
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix()
        portable_key = _portable_relative_key(relative)
        if portable_key is None:
            raise ArtifactHashesError(f"{label} contains a nonportable path: {child}")
        previous = seen_portable_paths.get(portable_key)
        if previous is not None and previous != relative:
            raise ArtifactHashesError(f"{label} contains a portable path collision: {previous!r} and {relative!r}")
        seen_portable_paths[portable_key] = relative
        if child.is_symlink():
            raise ArtifactHashesError(f"{label} must not contain a symlink: {child}")
        child_mode = stat.S_IMODE(child.stat().st_mode)
        if child.is_dir():
            entries.append({"path": relative, "type": "directory", "mode": child_mode})
        elif child.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": child_mode,
                    "content": _encode_content(_read_regular_file_bytes(child, artifact_name=f"{label}/{relative}")),
                }
            )
        else:
            raise ArtifactHashesError(f"{label} contains a non-regular path: {child}")
    return {"type": "directory", "mode": mode, "entries": entries}


def _validated_path_snapshot(snapshot: object) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        raise RunDigestError("run publication journal contains an invalid path snapshot")
    path_type = snapshot.get("type")
    mode = snapshot.get("mode")
    if path_type not in {"file", "directory"} or isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777:
        raise RunDigestError("run publication journal contains an invalid path snapshot")
    if path_type == "file":
        content = _decode_content(snapshot.get("content"))
        if content is None:
            raise RunDigestError("run publication journal contains a file snapshot without content")
        return {"type": "file", "mode": mode, "content": _encode_content(content)}
    entries = snapshot.get("entries")
    if not isinstance(entries, list):
        raise RunDigestError("run publication journal contains a directory snapshot without entries")
    validated_entries: list[dict[str, Any]] = []
    seen: dict[tuple[str, ...], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RunDigestError("run publication journal contains an invalid directory entry")
        relative = entry.get("path")
        entry_type = entry.get("type")
        entry_mode = entry.get("mode")
        if not isinstance(relative, str):
            raise RunDigestError("run publication journal contains an invalid directory entry path")
        portable_key = _portable_relative_key(relative)
        if portable_key is None or portable_key in seen:
            raise RunDigestError("run publication journal contains an unsafe or duplicate directory entry")
        invalid_mode = isinstance(entry_mode, bool) or not isinstance(entry_mode, int) or not 0 <= entry_mode <= 0o7777
        if entry_type not in {"file", "directory"} or invalid_mode:
            raise RunDigestError("run publication journal contains an invalid directory entry")
        seen[portable_key] = relative
        validated = {"path": relative, "type": entry_type, "mode": entry_mode}
        if entry_type == "file":
            content = _decode_content(entry.get("content"))
            if content is None:
                raise RunDigestError("run publication journal contains a file entry without content")
            validated["content"] = _encode_content(content)
        validated_entries.append(validated)
    return {"type": "directory", "mode": mode, "entries": validated_entries}


def _journal_path_target(run_path: Path, name: str, source: str | Path | None) -> dict[str, Any]:
    target = _safe_run_replacement_path(run_path, name)
    return {
        "kind": "path",
        "name": name,
        "old_path": _snapshot_path(target, label=f"replacement target {name}"),
        "new_path": None if source is None else _snapshot_path(Path(source), label=f"replacement source {name}"),
    }


def _journal_target(kind: str, path: Path, new_content: bytes | None, *, name: str | None = None) -> dict[str, Any]:
    old_content = _read_optional_regular_file(path)
    return {
        "kind": kind,
        "name": name,
        "old": _encode_content(old_content),
        "new": _encode_content(new_content),
    }


def _begin_guarded_run_rollback_locked(
    run_path: Path,
    *,
    artifact_names: Collection[str],
    include_digest_state: bool,
    rollback_guard: Path,
) -> dict[str, Any]:
    """Start a rollback journal whose commit point is removal of a guard."""
    guard_path = Path(rollback_guard)
    if not guard_path.is_absolute():
        raise ArtifactHashesError("run rollback guard must be an absolute path")
    if guard_path.is_symlink() or not guard_path.is_file():
        raise ArtifactHashesError(f"run rollback guard must be a regular, non-symlink file: {guard_path}")
    names = list(artifact_names)
    for name in names:
        _validate_artifact_name(name)
        if name == "artifact_hashes.json":
            raise ArtifactHashesError("artifact_hashes.json must use the manifest transaction target")
    _require_no_portable_path_collisions(names)

    targets: list[dict[str, Any]] = []
    for name in names:
        target = _journal_target(
            "artifact",
            _safe_run_artifact_path(run_path, name),
            None,
            name=name,
        )
        target["new"] = target["old"]
        targets.append(target)
    if include_digest_state:
        for kind, path in (
            ("manifest", run_path / "artifact_hashes.json"),
            ("digest", run_path.parent / "run_digests.jsonl"),
        ):
            target = _journal_target(kind, path, None)
            target["new"] = target["old"]
            targets.append(target)
    if not targets:
        raise ArtifactHashesError("guarded run rollback requires at least one target")

    payload: dict[str, Any] = {
        "schema_version": _JOURNAL_SCHEMA_VERSION,
        "recovery": "rollback",
        "run_id": run_path.name,
        "run_root_identity": stable_filesystem_identity(run_path),
        "journal_root_identity": stable_filesystem_identity(run_path.parent),
        "rollback_guard": {
            "schema_version": _ROLLBACK_GUARD_SCHEMA_VERSION,
            "path": str(guard_path),
            "parent_identity": stable_filesystem_identity(guard_path.parent),
            "portable_name": _portable_component_key(guard_path.name),
        },
        "targets": targets,
    }
    _write_journal(_journal_path(run_path.parent), payload)
    return payload


def _seal_guarded_run_rollback_locked(
    run_path: Path,
    payload: dict[str, Any],
    *,
    expected_digest: str | None,
) -> None:
    """Record the committed target generation before the guard can be removed."""
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ArtifactHashesError("guarded run rollback journal has no targets")
    for target in targets:
        if not isinstance(target, dict):
            raise ArtifactHashesError("guarded run rollback journal has an invalid target")
        target_path = _target_path(run_path, target)
        target["new"] = _encode_content(_read_optional_regular_file(target_path))
    if expected_digest is None:
        payload.pop("postconditions", None)
    else:
        if _DIGEST_RE.fullmatch(expected_digest) is None:
            raise ArtifactHashesError(f"invalid artifact_hashes digest: {expected_digest}")
        payload["postconditions"] = {"artifact_hashes": expected_digest}
    _write_journal(_journal_path(run_path.parent), payload)


def _read_optional_regular_file(path: Path) -> bytes | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ArtifactHashesError(f"transaction target must be a regular, non-symlink file: {path}")
    return path.read_bytes()


def _target_path(run_path: Path, target: Mapping[str, Any]) -> Path:
    kind = target.get("kind")
    if kind == "digest":
        return run_path.parent / "run_digests.jsonl"
    if kind == "manifest":
        if run_path.is_symlink() or not run_path.is_dir():
            raise RunDigestError(f"run directory must be a real directory: {run_path}")
        return run_path / "artifact_hashes.json"
    if kind == "artifact" and isinstance(target.get("name"), str):
        return _safe_run_artifact_path(run_path, target["name"])
    if kind == "path" and isinstance(target.get("name"), str):
        return _safe_run_replacement_path(run_path, target["name"])
    raise RunDigestError("run publication journal contains an invalid target")


def _commit_journaled_targets(
    run_path: Path,
    targets: list[dict[str, Any]],
    commit: Callable[[], None],
    *,
    expected_digest: str | None = None,
    validate_completed: Callable[[], None] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": _JOURNAL_SCHEMA_VERSION,
        "recovery": "commit",
        "run_id": run_path.name,
        "run_root_identity": stable_filesystem_identity(run_path),
        "journal_root_identity": stable_filesystem_identity(run_path.parent),
        "targets": targets,
    }
    if expected_digest is not None:
        payload["postconditions"] = {"artifact_hashes": expected_digest}
    journal_path = _journal_path(run_path.parent)
    commit_started = False
    try:
        _write_journal(journal_path, payload, boundary="journal")
        commit_started = True
        commit()
        if validate_completed is not None:
            validate_completed()
        _clear_journal(journal_path, boundary="journal")
    except BaseException as cause:
        if commit_started or journal_path.exists():
            try:
                payload["recovery"] = "rollback"
                _write_journal(journal_path, payload)
                _recover_publication_locked(run_path.parent)
            except BaseException as recovery_error:
                raise ArtifactHashesError(f"run publication failed and could not be recovered: {recovery_error}") from cause
        raise


def _write_journal(path: Path, payload: Mapping[str, Any], *, boundary: str | None = None) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        boundary=boundary,
    )


def _clear_journal(path: Path, *, boundary: str | None = None) -> None:
    path.unlink(missing_ok=True)
    if boundary is not None:
        _publication_boundary(f"{boundary}.unlink")
    _fsync_directory(path.parent)
    if boundary is not None:
        _publication_boundary(f"{boundary}.unlink_dir_fsync")


def _guarded_recovery_mode(
    payload: Mapping[str, Any],
    recovery: str,
    journal_path: Path,
) -> str:
    guard = payload.get("rollback_guard")
    if guard is None:
        return recovery
    if recovery != "rollback" or not isinstance(guard, dict):
        raise RunDigestError(f"run publication journal has an invalid rollback guard: {journal_path}")
    if set(guard) != {"schema_version", "path", "parent_identity", "portable_name"}:
        raise RunDigestError(f"run publication journal has an invalid rollback guard: {journal_path}")
    guard_path_raw = guard.get("path")
    parent_identity = guard.get("parent_identity")
    portable_name = guard.get("portable_name")
    if (
        guard.get("schema_version") != _ROLLBACK_GUARD_SCHEMA_VERSION
        or not isinstance(guard_path_raw, str)
        or not isinstance(parent_identity, str)
        or not isinstance(portable_name, str)
    ):
        raise RunDigestError(f"run publication journal has an invalid rollback guard: {journal_path}")
    guard_path = Path(guard_path_raw)
    if (
        not guard_path.is_absolute()
        or _portable_component_key(guard_path.name) != portable_name
        or stable_filesystem_identity(guard_path.parent) != parent_identity
    ):
        raise RunDigestError(f"run publication journal rollback guard moved: {journal_path}")
    if guard_path.is_symlink() or (guard_path.exists() and not guard_path.is_file()):
        raise RunDigestError(f"run publication journal rollback guard is not a regular file: {journal_path}")
    return "rollback" if guard_path.exists() else "commit"


def _recover_publication_locked(parent: Path) -> None:
    parent = _canonical_existing_path(parent)
    journal_path = _journal_path(parent)
    if not journal_path.exists():
        _cleanup_atomic_temps(journal_path)
        return
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunDigestError(f"run publication journal is invalid: {journal_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _JOURNAL_SCHEMA_VERSION:
        raise RunDigestError(f"run publication journal has an unsupported schema: {journal_path}")
    run_id = payload.get("run_id")
    if not _is_normal_run_id(run_id):
        raise RunDigestError(f"run publication journal has an unsafe run_id: {journal_path}")
    assert isinstance(run_id, str)
    recorded_journal_root = payload.get("journal_root_identity")
    if recorded_journal_root is not None and recorded_journal_root != stable_filesystem_identity(parent):
        raise RunDigestError(f"run publication journal is bound to a different journal root: {journal_path}")
    recorded_recovery = payload.get("recovery")
    if recorded_recovery not in {"commit", "rollback"}:
        raise RunDigestError(f"run publication journal has an invalid recovery mode: {journal_path}")
    recovery = _guarded_recovery_mode(payload, recorded_recovery, journal_path)
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RunDigestError(f"run publication journal has no targets: {journal_path}")
    parent_root = parent
    try:
        run_path = _canonical_run_path(parent / run_id)
    except RunDigestError as exc:
        raise RunDigestError(f"run publication journal has an invalid run root: {journal_path}") from exc
    if run_path.parent != parent_root:
        raise RunDigestError(f"run publication journal has an invalid run root: {journal_path}")
    recorded_run_root = payload.get("run_root_identity")
    if recorded_run_root is not None and recorded_run_root != stable_filesystem_identity(run_path):
        raise RunDigestError(f"run publication journal is bound to a different run root: {journal_path}")
    run_root = run_path
    validated: list[tuple[Mapping[str, Any], Path, object, object]] = []
    seen_paths: set[Path] = set()
    path_targets: list[Path] = []
    atomic_targets: list[Path] = [journal_path]
    for target in targets:
        if not isinstance(target, dict):
            raise RunDigestError(f"run publication journal contains an invalid target: {journal_path}")
        target_path = _target_path(run_path, target)
        _require_recovery_target_contained(parent_root, run_root, target, target_path, journal_path)
        if any(_paths_share_location_or_file_identity(target_path, previous) for previous in seen_paths):
            raise RunDigestError(f"run publication journal contains duplicate targets: {journal_path}")
        seen_paths.add(target_path)
        if target.get("kind") == "path":
            old_content = _validated_path_snapshot(target.get("old_path"))
            new_content = _validated_path_snapshot(target.get("new_path"))
            path_targets.append(target_path)
        else:
            old_content = _decode_content(target.get("old"))
            new_content = _decode_content(target.get("new"))
            atomic_targets.append(target_path)
        if target.get("kind") == "artifact" and target.get("name") == "artifact_hashes.json":
            raise RunDigestError(f"run publication journal contains a reserved artifact target: {journal_path}")
        validated.append((target, target_path, old_content, new_content))
    try:
        _require_managed_transaction_targets(
            [*seen_paths, journal_path, parent / "run_digests.jsonl.lock"],
            path_targets=path_targets,
            atomic_targets=atomic_targets,
        )
    except ArtifactHashesError as exc:
        raise RunDigestError(str(exc)) from exc

    recovery_payload = payload
    if recovery != recorded_recovery:
        recovery_payload = {**payload, "recovery": recovery}
    expected_digest = _recovery_expected_digest(
        recovery_payload,
        validated,
        run_id,
        journal_path,
    )
    _cleanup_atomic_temps(journal_path)
    ordered = validated if recovery == "commit" else list(reversed(validated))
    for target, target_path, old_content, new_content in ordered:
        content = new_content if recovery == "commit" else old_content
        if target.get("kind") == "path":
            _atomic_replace_path_snapshot(target_path, content, boundary=None)
        else:
            _restore_file_bytes(target_path, content)
            _cleanup_atomic_temps(target_path)
    target_kinds = {target.get("kind") for target, *_rest in validated}
    if recovery == "commit" and expected_digest is not None:
        _validate_completed_run_state(run_path, expected_digest)
    elif {"manifest", "digest"} <= target_kinds:
        artifact_hashes = _read_artifact_hashes(run_path / "artifact_hashes.json")
        _validate_completed_run_state(run_path, _hash_json_payload(artifact_hashes))
    _clear_journal(journal_path)


def _recovery_expected_digest(
    payload: Mapping[str, Any],
    validated: Collection[tuple[Mapping[str, Any], Path, object, object]],
    run_id: str,
    journal_path: Path,
) -> str | None:
    if payload.get("recovery") != "commit":
        return None
    digest_targets = [item for item in validated if item[0].get("kind") == "digest"]
    postconditions = payload.get("postconditions")
    recorded_digest: str | None = None
    if postconditions is not None:
        if not isinstance(postconditions, dict) or set(postconditions) != {"artifact_hashes"}:
            raise RunDigestError(f"run publication journal has invalid postconditions: {journal_path}")
        candidate = postconditions.get("artifact_hashes")
        if not isinstance(candidate, str) or _DIGEST_RE.fullmatch(candidate) is None:
            raise RunDigestError(f"run publication journal has invalid postconditions: {journal_path}")
        recorded_digest = candidate
    if not digest_targets:
        if recorded_digest is not None:
            raise RunDigestError(f"run publication journal postcondition has no digest target: {journal_path}")
        return None

    _target, digest_path, _old_content, new_content = digest_targets[0]
    if not isinstance(new_content, bytes):
        raise RunDigestError(f"run publication journal digest target has no committed content: {journal_path}")
    entries = _parse_run_digest_content(new_content, digest_path)
    matches = [entry for entry in entries if entry["run_id"] == run_id]
    if len(matches) != 1:
        raise RunDigestError(
            f"run publication journal digest postcondition requires one row for {run_id}: {journal_path}"
        )
    target_digest = matches[0]["artifact_hashes"]
    if recorded_digest is not None and target_digest != recorded_digest:
        raise RunDigestError(
            "run publication journal digest target does not match its postcondition: "
            f"stored={target_digest}, expected={recorded_digest}"
        )
    return recorded_digest or target_digest


def _is_normal_run_id(value: object) -> bool:
    parts = _portable_posix_relative_parts(value)
    return parts is not None and len(parts) == 1


def _require_recovery_target_contained(
    parent_root: Path,
    run_root: Path,
    target: Mapping[str, Any],
    target_path: Path,
    journal_path: Path,
) -> None:
    kind = target.get("kind")
    resolved = target_path.resolve(strict=False)
    if kind == "digest":
        expected = parent_root / "run_digests.jsonl"
        contained = resolved == expected
    elif kind == "manifest":
        contained = resolved == run_root / "artifact_hashes.json"
    elif kind in {"artifact", "path"}:
        contained = resolved != run_root and resolved.is_relative_to(run_root)
    else:
        contained = False
    if not contained:
        raise RunDigestError(f"run publication journal target escapes the exact run root: {journal_path}")


def _atomic_replace_path_snapshot(
    path: Path,
    snapshot: object,
    *,
    boundary: str | None,
) -> None:
    desired = _validated_path_snapshot(snapshot)
    new_path = path.parent / f".{path.name}.oxq-path-new"
    old_path = path.parent / f".{path.name}.oxq-path-old"
    if _path_matches_snapshot(path, desired):
        removed = _remove_path(new_path) | _remove_path(old_path)
        if removed:
            _fsync_directory(path.parent)
        return

    _remove_path(new_path)
    if desired is not None:
        _materialize_path_snapshot(new_path, desired)
        if boundary is not None:
            _publication_boundary(f"{boundary}.temp_fsync")

    if path.exists() or path.is_symlink():
        _remove_path(old_path)
        os.replace(path, old_path)
        if boundary is not None:
            _publication_boundary(f"{boundary}.old_replace")
        _fsync_directory(path.parent)
        if boundary is not None:
            _publication_boundary(f"{boundary}.old_dir_fsync")

    if desired is not None:
        os.replace(new_path, path)
        if boundary is not None:
            _publication_boundary(f"{boundary}.replace")
        _fsync_directory(path.parent)
        if boundary is not None:
            _publication_boundary(f"{boundary}.dir_fsync")

    removed_old = _remove_path(old_path)
    if removed_old and boundary is not None:
        _publication_boundary(f"{boundary}.backup_unlink")
    if removed_old:
        _fsync_directory(path.parent)
        if boundary is not None:
            _publication_boundary(f"{boundary}.backup_unlink_dir_fsync")


def _path_matches_snapshot(path: Path, snapshot: dict[str, Any] | None) -> bool:
    try:
        current = _snapshot_path(path, label=f"transaction target {path.name}")
    except (OSError, RunDigestError):
        return False
    return current == snapshot


def _materialize_path_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot["type"] == "file":
        content = _decode_content(snapshot["content"])
        assert content is not None
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(snapshot["mode"])
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        return

    path.mkdir()
    entries = snapshot["entries"]
    directories = [entry for entry in entries if entry["type"] == "directory"]
    files = [entry for entry in entries if entry["type"] == "file"]
    for entry in sorted(directories, key=lambda item: (len(PurePosixPath(item["path"]).parts), item["path"])):
        _portable_child_path(path, entry["path"]).mkdir(parents=True, exist_ok=True)
    for entry in files:
        target = _portable_child_path(path, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _decode_content(entry["content"])
        assert content is not None
        with target.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(entry["mode"])
        with target.open("rb") as handle:
            os.fsync(handle.fileno())
    for entry in sorted(directories, key=lambda item: len(PurePosixPath(item["path"]).parts), reverse=True):
        directory = _portable_child_path(path, entry["path"])
        directory.chmod(entry["mode"])
        _fsync_directory(directory)
    path.chmod(snapshot["mode"])
    _fsync_directory(path)


def _portable_child_path(parent: Path, name: object) -> Path:
    parts = _portable_posix_relative_parts(name)
    if parts is None:
        raise RunDigestError(f"invalid portable relative path: {name!r}")
    return parent.joinpath(*parts)


def _remove_path(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise RunDigestError(f"transaction path is not a regular file or directory: {path}")
    return True


def _restore_file_bytes(path: Path, content: bytes | None) -> None:
    if _file_matches_bytes(path, content):
        return
    if content is None:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    else:
        _atomic_write_bytes(path, content)


def _file_matches_bytes(path: Path, content: bytes | None) -> bool:
    if content is None:
        return not path.exists() and not path.is_symlink()
    try:
        return not path.is_symlink() and path.is_file() and path.read_bytes() == content
    except OSError:
        return False


def _cleanup_atomic_temps(path: Path) -> None:
    removed = False
    for temp_path in path.parent.glob(f".{path.name}.*.tmp"):
        temp_path.unlink(missing_ok=True)
        removed = True
    if removed:
        _fsync_directory(path.parent)


def _publication_boundary(_label: str) -> None:
    """Fault-injection hook used by subprocess crash tests."""


def _atomic_write_text(path: Path, content: str, *, boundary: str | None = None) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"), boundary=boundary)


def _atomic_remove_file(path: Path, *, boundary: str | None = None) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise ArtifactHashesError(f"transaction target must be a regular, non-symlink file: {path}")
    path.unlink()
    if boundary is not None:
        _publication_boundary(f"{boundary}.unlink")
    _fsync_directory(path.parent)
    if boundary is not None:
        _publication_boundary(f"{boundary}.unlink_dir_fsync")


def _atomic_write_bytes(path: Path, content: bytes, *, boundary: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            if boundary is not None:
                _publication_boundary(f"{boundary}.temp_fsync")
        os.replace(temp_path, path)
        temp_path = None
        if boundary is not None:
            _publication_boundary(f"{boundary}.replace")
        if not _is_windows():
            _fsync_directory(path.parent)
            if boundary is not None:
                _publication_boundary(f"{boundary}.dir_fsync")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if _is_windows():
        # Windows does not expose a supported directory fsync through os.open.
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _is_windows() -> bool:
    return os.name == "nt"
