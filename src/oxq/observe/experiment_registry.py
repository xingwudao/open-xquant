"""Experiment Registry — JSONL-based experiment tracking.

Thin wrapper over a JSONL file that records every research run
to prevent selective memory. Complements the in-memory
:class:`oxq.observe.experiment.ExperimentLog`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
from datetime import datetime, timezone; UTC = timezone.utc  # py3.9 compat
from pathlib import Path
from typing import Any

from oxq.process_lock import (
    ProcessFileLock,
    stable_filesystem_identity,
    stable_path_location_identity,
)

DEFAULT_REGISTRY_PATH = "experiments.jsonl"
_VERSION_RE = re.compile(r"^v[0-9][A-Za-z0-9_-]*$")
_MONITOR_ARTIFACTS = (
    "reproducibility_audit.json",
    "research_bias_audit.json",
    "robustness.json",
)
_REGISTRY_TRANSACTION_SCHEMA_VERSION = 1
_REGISTRY_RUN_STATE_PROTOCOL = "shared_run_publication_v1"


class _MonitorRefreshError(RuntimeError):
    pass


class _RegistryAppendError(RuntimeError):
    pass


def add_experiment(
    run_dir: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    decision: str = "unknown",
    version_root: str | Path | None = None,
    backtest_phase_dir: str | Path | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    """Append a backtest run to the experiment registry.

    Reads metrics.json, strategy_spec.yaml, and spec_hash.txt from *run_dir*,
    constructs an experiment entry, and appends it as a JSON line to
    *registry_path*.

    Returns the entry dict with the generated experiment_id.
    """
    from oxq.run_digests import _canonical_run_path

    try:
        run_path = _canonical_run_path(run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"error": f"run directory could not be resolved: {run_dir}: {exc}"}
    explicit_parts = _explicit_backtest_parts(
        run_path,
        backtest_phase_dir=backtest_phase_dir,
        version_id=version_id,
    )
    if isinstance(explicit_parts, str):
        return {"error": explicit_parts}
    try:
        _validate_registry_location_before_parent_creation(run_path, registry_path)
        reg_path = _canonical_registry_path(registry_path)
        _validate_registry_managed_paths(run_path, reg_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"error": str(exc)}
    lock_path = reg_path.with_suffix(reg_path.suffix + ".lock")
    with ProcessFileLock(lock_path):
        try:
            _recover_pending_registry_transaction_locked(reg_path)
        except (OSError, ValueError) as exc:
            return {"error": f"experiment registry transaction recovery failed: {exc}"}

        from oxq.run_digests import run_digest_transaction

        with run_digest_transaction(run_path):
            metrics_path = run_path / "metrics.json"
            if not metrics_path.exists():
                return {"error": f"metrics.json not found in {run_dir}"}

            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                return {"error": f"metrics.json could not be parsed in {run_dir}: {exc}"}
            if not isinstance(metrics, dict):
                return {"error": f"metrics.json must contain an object in {run_dir}"}
            resolved_run_name = run_path.name
            if metrics.get("run_id") != resolved_run_name:
                return {
                    "error": (
                        "metrics.json run_id must match the resolved run directory name: "
                        f"{resolved_run_name}"
                    )
                }

            spec_path = run_path / "strategy_spec.yaml"
            if not spec_path.is_file():
                return {"error": f"strategy_spec.yaml not found in {run_dir}"}
            try:
                from oxq.spec.schema import StrategySpec

                spec = StrategySpec.from_yaml(spec_path)
            except Exception as exc:
                return {"error": f"strategy_spec.yaml could not be parsed in {run_dir}: {exc}"}
            if metrics.get("strategy_id") != spec.strategy_id:
                return {
                    "error": (
                        "metrics.json strategy_id must match strategy_spec.yaml strategy_id: "
                        f"{spec.strategy_id}"
                    )
                }

            spec_hash_path = run_path / "spec_hash.txt"
            if not spec_hash_path.is_file():
                return {"error": f"spec_hash.txt not found in {run_dir}"}
            try:
                spec_hash = spec_hash_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as exc:
                return {"error": f"spec_hash.txt could not be read in {run_dir}: {exc}"}
            if spec_hash != spec.compute_hash():
                return {"error": "spec_hash.txt must match the canonical strategy_spec.yaml hash"}

            bias_path = run_path / "research_bias_audit.json"
            from oxq.audit.research_bias import audit_research

            bias = audit_research(run_path)
            audit_status = bias.get("status", "unknown")
            entry = {
                "experiment_id": f"exp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}",
                "version_id": (
                    explicit_parts[0]
                    if explicit_parts is not None
                    else _infer_version_id(run_path, version_root=version_root)
                ),
                "run_path": _display_run_path(run_path),
                "run_role": _run_role_from_name(
                    explicit_parts[1]
                    if explicit_parts is not None
                    else _structural_run_name(run_path, version_root=version_root)
                ),
                "strategy_id": metrics.get("strategy_id", ""),
                "spec_hash": spec_hash,
                "run_id": metrics.get("run_id", ""),
                "metrics": metrics,
                "audit_status": audit_status,
                "decision": decision,
                "created_at": datetime.now(UTC).isoformat(),
            }
            try:
                _commit_registry_transaction_locked(
                    run_path,
                    reg_path,
                    bias_path,
                    (json.dumps(bias, indent=2) + "\n").encode("utf-8"),
                    entry,
                )
            except _MonitorRefreshError as exc:
                return {"error": f"monitor artifact integrity refresh failed: {exc}"}
            except _RegistryAppendError as exc:
                return {"error": f"experiment registry append failed: {exc}"}
            except (OSError, ValueError) as exc:
                return {"error": f"experiment registry transaction failed: {exc}"}

    return entry


def _commit_registry_transaction_locked(
    run_path: Path,
    reg_path: Path,
    bias_path: Path,
    bias_content: bytes,
    entry: dict[str, Any],
) -> None:
    _validate_registry_managed_paths(run_path, reg_path)
    journal_path = _registry_transaction_path(reg_path)
    snapshots = {
        "bias": _encode_optional(_read_optional_regular_file(bias_path)),
        "manifest": _encode_optional(_read_optional_regular_file(run_path / "artifact_hashes.json")),
        "digest": _encode_optional(_read_optional_regular_file(run_path.parent / "run_digests.jsonl")),
        "registry": _encode_optional(_read_optional_regular_file(reg_path)),
    }
    payload = {
        "schema_version": _REGISTRY_TRANSACTION_SCHEMA_VERSION,
        "recovery": "rollback",
        "run_path": str(run_path),
        "registry_path": str(reg_path),
        "run_root_identity": stable_filesystem_identity(run_path),
        "registry_root_identity": _registry_root_identity(reg_path),
        "run_state_protocol": _REGISTRY_RUN_STATE_PROTOCOL,
        "snapshots": snapshots,
    }
    mutation_started = False
    try:
        _atomic_write_text(journal_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        mutation_started = True
        from oxq.run_digests import (
            _begin_guarded_run_rollback_locked,
            _recover_publication_locked,
            _seal_guarded_run_rollback_locked,
        )

        run_rollback = _begin_guarded_run_rollback_locked(
            run_path,
            artifact_names=(bias_path.name,),
            include_digest_state=snapshots["manifest"] is not None,
            rollback_guard=journal_path,
        )
        _atomic_write_bytes(bias_path, bias_content)
        try:
            expected_digest = refresh_monitor_integrity(
                run_path,
                _run_transaction_locked=True,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise _MonitorRefreshError(str(exc)) from exc
        _seal_guarded_run_rollback_locked(
            run_path,
            run_rollback,
            expected_digest=expected_digest,
        )
        try:
            _append_registry_entry_locked(reg_path, entry)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise _RegistryAppendError(str(exc)) from exc
        _clear_transaction_journal(journal_path)
        _recover_publication_locked(run_path.parent)
    except BaseException as cause:
        if mutation_started or journal_path.exists() or journal_path.is_symlink():
            try:
                _recover_registry_transaction_payload_locked(reg_path, payload)
            except BaseException as recovery_error:
                raise ValueError(
                    f"experiment registry transaction failed and could not be recovered: {recovery_error}"
                ) from cause
        raise


def _append_registry_entry_locked(reg_path: Path, entry: dict[str, Any]) -> None:
    existing = _read_optional_regular_file(reg_path) or b""
    _atomic_write_bytes(reg_path, existing + (json.dumps(entry) + "\n").encode("utf-8"))


def refresh_monitor_integrity(
    run_dir: str | Path,
    *,
    _run_transaction_locked: bool = False,
) -> str | None:
    """Refresh hashes for present monitor outputs and the parent run digest."""
    run_path = Path(run_dir)
    artifact_hashes_path = run_path / "artifact_hashes.json"
    if not artifact_hashes_path.exists() and not artifact_hashes_path.is_symlink():
        return None

    from oxq.run_digests import (
        _update_artifact_hashes_and_run_digest_locked,
        update_artifact_hashes_and_run_digest,
    )

    def update(artifact_hashes: dict[str, Any]) -> None:
        for artifact_name in _MONITOR_ARTIFACTS:
            artifact_path = run_path / artifact_name
            if artifact_path.exists() or artifact_path.is_symlink():
                artifact_hashes[artifact_name] = _hash_file(artifact_path)

    if _run_transaction_locked:
        return _update_artifact_hashes_and_run_digest_locked(run_path, update)
    return update_artifact_hashes_and_run_digest(run_path, update)


def list_experiments(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Read all experiments from the registry file.

    Returns a list of experiment entry dicts. Returns empty list if the
    registry file does not exist.
    """
    supplied = _lexically_normalized_absolute_path(registry_path)
    if not supplied.parent.exists() and not supplied.parent.is_symlink():
        return []
    reg_path = _canonical_registry_path(registry_path, create_parent=False)
    lock_path = reg_path.with_suffix(reg_path.suffix + ".lock")
    with ProcessFileLock(lock_path):
        _recover_pending_registry_transaction_locked(reg_path)
        return _list_experiments_locked(reg_path)


def _list_experiments_locked(reg_path: Path) -> list[dict[str, Any]]:
    content = _read_optional_regular_file(reg_path)
    if content is None:
        return []

    entries: list[dict[str, Any]] = []
    for line in content.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _infer_version_id(run_path: Path, *, version_root: str | Path | None = None) -> str:
    structural = _structural_backtest_parts(run_path, version_root=version_root)
    return structural[0] if structural is not None else ""


def _infer_run_role(run_path: Path, *, version_root: str | Path | None = None) -> str:
    return _run_role_from_name(_structural_run_name(run_path, version_root=version_root))


def _structural_run_name(run_path: Path, *, version_root: str | Path | None = None) -> str | None:
    structural = _structural_backtest_parts(run_path, version_root=version_root)
    return structural[1] if structural is not None else None


def _run_role_from_name(run_name: str | None) -> str:
    if run_name is None:
        return "primary"
    if run_name.endswith("_cost_x2"):
        return "robustness_cost_x2"
    return "primary"


def _explicit_backtest_parts(
    run_path: Path,
    *,
    backtest_phase_dir: str | Path | None,
    version_id: str | None,
) -> tuple[str, str] | str | None:
    if backtest_phase_dir is None and version_id is None:
        return None
    if backtest_phase_dir is None or version_id is None or not _VERSION_RE.fullmatch(version_id):
        return "resolved backtest phase directory and safe version_id must be provided together"
    try:
        relative = run_path.resolve().relative_to(Path(backtest_phase_dir).resolve())
    except ValueError:
        return "run directory must stay within the resolved backtest phase directory"
    if len(relative.parts) != 1:
        return "run directory must stay within the resolved backtest phase directory"
    return version_id, relative.name


def _structural_backtest_parts(
    run_path: Path,
    *,
    version_root: str | Path | None = None,
) -> tuple[str, str] | None:
    if version_root is not None:
        try:
            relative_parts = run_path.resolve().relative_to(Path(version_root).resolve()).parts
        except ValueError:
            relative_parts = ()
        if (
            len(relative_parts) == 3
            and _VERSION_RE.fullmatch(relative_parts[0])
            and relative_parts[1] == "09_backtests"
        ):
            return relative_parts[0], relative_parts[2]
        return None
    parts = run_path.parts
    for backtest_index in range(len(parts) - 2, 0, -1):
        if parts[backtest_index] != "09_backtests":
            continue
        if backtest_index != len(parts) - 2:
            continue
        version_id = parts[backtest_index - 1]
        if _VERSION_RE.fullmatch(version_id):
            return version_id, parts[backtest_index + 1]
    return None


def _display_run_path(run_path: Path) -> str:
    try:
        return run_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return run_path.as_posix()


def _hash_file(path: Path) -> str:
    content = _read_optional_regular_file(path)
    if content is None:
        raise ValueError(f"monitor artifact must be a regular, non-symlink file: {path}")
    return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"transaction target must be a regular, non-symlink file: {path}")
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
        os.replace(temp_path, path)
        temp_path = None
        _fsync_directory(path.parent)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _canonical_registry_path(
    registry_path: str | Path,
    *,
    create_parent: bool = True,
) -> Path:
    supplied = _lexically_normalized_absolute_path(registry_path)
    if supplied.is_symlink():
        raise ValueError(f"experiment registry must be a regular, non-symlink file: {supplied}")
    if create_parent:
        supplied.parent.mkdir(parents=True, exist_ok=True)
    from oxq.run_digests import _canonical_existing_path

    parent = _canonical_existing_path(supplied.parent)
    path = parent / supplied.name
    if path.exists():
        path = _canonical_existing_path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"experiment registry must be a regular, non-symlink file: {path}")
    return path


def _lexically_normalized_absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _validate_registry_location_before_parent_creation(
    run_path: Path,
    registry_path: str | Path,
) -> None:
    candidate = _lexically_normalized_absolute_path(registry_path)
    missing_parts: list[str] = []
    existing = candidate
    while not existing.exists() and not existing.is_symlink():
        if existing.parent == existing:
            break
        missing_parts.append(existing.name)
        existing = existing.parent
    from oxq.run_digests import _canonical_existing_path

    canonical = _canonical_existing_path(existing).joinpath(*reversed(missing_parts))
    if canonical == run_path or canonical.is_relative_to(run_path):
        raise ValueError(f"experiment registry managed path overlap: {canonical} and {run_path}")


def _registry_root_identity(reg_path: Path) -> str:
    normalized_name = unicodedata.normalize(
        "NFC",
        unicodedata.normalize("NFKC", reg_path.name).casefold(),
    )
    return f"{stable_filesystem_identity(reg_path.parent)}:{normalized_name}"


def _validate_registry_managed_paths(run_path: Path, reg_path: Path) -> None:
    digest_path = run_path.parent / "run_digests.jsonl"
    registry_paths = [
        reg_path,
        reg_path.with_suffix(reg_path.suffix + ".lock"),
        _registry_transaction_path(reg_path),
    ]
    run_entries = list(run_path.rglob("*"))
    run_paths = [
        run_path,
        *run_entries,
        run_path / "artifact_hashes.json",
        digest_path,
        digest_path.with_suffix(digest_path.suffix + ".lock"),
        digest_path.with_suffix(digest_path.suffix + ".journal"),
    ]
    _require_distinct_managed_locations(registry_paths, "experiment registry internals overlap")
    _require_distinct_managed_locations(run_paths, "run transaction internals overlap")
    run_temp_index = _managed_temp_index(run_paths)
    for candidate in run_paths:
        managed = _managed_temp_alias(candidate, run_temp_index)
        if managed is not None and candidate != managed:
            raise ValueError(f"run transaction internals overlap: {candidate} and {managed}")

    from oxq.run_digests import _paths_share_location_or_file_identity

    for registry_path in registry_paths:
        overlapping = next(
            (
                path
                for path in run_paths
                if _paths_share_location_or_file_identity(registry_path, path)
            ),
            None,
        )
        if overlapping is not None:
            raise ValueError(f"experiment registry managed path overlap: {registry_path} and {overlapping}")
        if _paths_overlap_lexically(registry_path, run_path):
            raise ValueError(f"experiment registry managed path overlap: {registry_path} and {run_path}")
        managed = _managed_temp_alias(registry_path, run_temp_index)
        if managed is not None:
            raise ValueError(f"experiment registry transaction internals overlap: {registry_path} and {managed}")

    registry_temp_index = _managed_temp_index(registry_paths)
    for managed in run_paths:
        registry_path = _managed_temp_alias(managed, registry_temp_index)
        if registry_path is not None:
            raise ValueError(f"experiment registry transaction internals overlap: {managed} and {registry_path}")


def _require_distinct_managed_locations(paths: list[Path], message: str) -> None:
    seen_locations: dict[str, Path] = {}
    seen_files: dict[str, Path] = {}
    for path in paths:
        location_identity = stable_path_location_identity(path)
        previous = seen_locations.get(location_identity)
        if previous is not None and previous != path:
            raise ValueError(f"{message}: {previous} and {path}")
        seen_locations[location_identity] = path
        if path.exists():
            file_identity = stable_filesystem_identity(path)
            previous = seen_files.get(file_identity)
            if previous is not None and previous != path:
                raise ValueError(f"{message}: {previous} and {path}")
            seen_files[file_identity] = path


def _paths_overlap_lexically(left: Path, right: Path) -> bool:
    left_path = left.resolve(strict=False)
    right_path = right.resolve(strict=False)
    return left_path == right_path or left_path.is_relative_to(right_path) or right_path.is_relative_to(left_path)


def _managed_temp_index(paths: list[Path]) -> dict[str, dict[str, Path]]:
    from oxq.run_digests import _portable_component_key

    indexed: dict[str, dict[str, Path]] = {}
    for path in paths:
        parent_identity = stable_path_location_identity(path.parent)
        indexed.setdefault(parent_identity, {})[_portable_component_key(path.name)] = path
    return indexed


def _managed_temp_alias(
    candidate: Path,
    indexed: dict[str, dict[str, Path]],
) -> Path | None:
    from oxq.run_digests import _portable_component_key

    managed_names = indexed.get(stable_path_location_identity(candidate.parent))
    if not managed_names:
        return None
    candidate_name = _portable_component_key(candidate.name)
    if not candidate_name.startswith("."):
        return None
    for suffix in (".oxq-path-new", ".oxq-path-old"):
        if candidate_name.endswith(suffix):
            return managed_names.get(candidate_name[1 : -len(suffix)])
    if not candidate_name.endswith(".tmp"):
        return None
    body = candidate_name[1:-4]
    return next(
        (
            managed
            for index, character in enumerate(body)
            if character == "." and (managed := managed_names.get(body[:index])) is not None
        ),
        None,
    )


def _read_optional_regular_file(path: Path) -> bytes | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        raise ValueError(f"transaction target must be a regular, non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before_descriptor = os.fstat(descriptor)
        before_path = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before_descriptor.st_mode)
            or before_descriptor.st_nlink != 1
            or _registry_file_identity(before_descriptor) != _registry_file_identity(before_path)
        ):
            raise ValueError(f"transaction target must be a regular, non-symlink file: {path}")
        chunks: list[bytes] = []
        bytes_read = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            bytes_read += len(chunk)
        after_descriptor = os.fstat(descriptor)
        after_path = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(after_descriptor.st_mode)
            or after_descriptor.st_nlink != 1
            or _registry_file_identity(after_descriptor) != _registry_file_identity(after_path)
            or _registry_stable_read_metadata(before_descriptor)
            != _registry_stable_read_metadata(after_descriptor)
            or _registry_file_identity(before_path) != _registry_file_identity(after_path)
            or bytes_read != before_descriptor.st_size
            or bytes_read != after_descriptor.st_size
        ):
            raise ValueError(f"transaction target changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _registry_file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _registry_stable_read_metadata(
    metadata: os.stat_result,
) -> tuple[tuple[int, int], int, int | float, int | float]:
    return (
        _registry_file_identity(metadata),
        int(metadata.st_size),
        getattr(metadata, "st_mtime_ns", metadata.st_mtime),
        getattr(metadata, "st_ctime_ns", metadata.st_ctime),
    )


def _registry_transaction_path(reg_path: Path) -> Path:
    return reg_path.with_suffix(reg_path.suffix + ".transaction.json")


def _encode_optional(content: bytes | None) -> str | None:
    return None if content is None else base64.b64encode(content).decode("ascii")


def _decode_optional(content: object) -> bytes | None:
    if content is None:
        return None
    if not isinstance(content, str):
        raise ValueError("experiment registry transaction snapshot must be base64 text or null")
    try:
        return base64.b64decode(content, validate=True)
    except ValueError as exc:
        raise ValueError("experiment registry transaction snapshot is invalid base64") from exc


def _recover_pending_registry_transaction_locked(reg_path: Path) -> None:
    journal_path = _registry_transaction_path(reg_path)
    content = _read_optional_regular_file(journal_path)
    if content is None:
        return
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment registry transaction journal is invalid: {journal_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"experiment registry transaction journal must contain an object: {journal_path}")
    recorded_run = payload.get("run_path")
    if not isinstance(recorded_run, str):
        raise ValueError(f"experiment registry transaction journal has an invalid run path: {journal_path}")
    from oxq.run_digests import run_digest_transaction

    with run_digest_transaction(recorded_run):
        _recover_registry_transaction_payload_locked(reg_path, payload)


def _recover_registry_transaction_payload_locked(reg_path: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != _REGISTRY_TRANSACTION_SCHEMA_VERSION or payload.get("recovery") != "rollback":
        raise ValueError("experiment registry transaction journal has invalid metadata")
    run_state_protocol = payload.get("run_state_protocol")
    if run_state_protocol not in {None, _REGISTRY_RUN_STATE_PROTOCOL}:
        raise ValueError("experiment registry transaction journal has invalid run-state protocol")
    recorded_registry_root = payload.get("registry_root_identity")
    if recorded_registry_root is None:
        if payload.get("registry_path") != str(reg_path):
            raise ValueError("experiment registry transaction journal has invalid metadata")
    elif recorded_registry_root != _registry_root_identity(reg_path):
        raise ValueError("experiment registry transaction journal is bound to a different registry root")
    run_raw = payload.get("run_path")
    snapshots = payload.get("snapshots")
    if not isinstance(run_raw, str) or not isinstance(snapshots, dict):
        raise ValueError("experiment registry transaction journal has invalid snapshots")
    from oxq.run_digests import _canonical_run_path

    try:
        run_path = _canonical_run_path(run_raw)
    except ValueError as exc:
        raise ValueError("experiment registry transaction journal has a non-canonical run path") from exc
    recorded_run_root = payload.get("run_root_identity")
    if recorded_run_root is not None and recorded_run_root != stable_filesystem_identity(run_path):
        raise ValueError("experiment registry transaction journal is bound to a different run root")
    _validate_registry_managed_paths(run_path, reg_path)
    required = {"bias", "manifest", "digest", "registry"}
    if set(snapshots) != required:
        raise ValueError("experiment registry transaction journal has incomplete snapshots")
    if run_state_protocol == _REGISTRY_RUN_STATE_PROTOCOL:
        from oxq.run_digests import _recover_publication_locked

        _recover_publication_locked(run_path.parent)
        targets = ((reg_path, _decode_optional(snapshots["registry"])),)
    else:
        targets = (
            (reg_path, _decode_optional(snapshots["registry"])),
            (run_path.parent / "run_digests.jsonl", _decode_optional(snapshots["digest"])),
            (run_path / "artifact_hashes.json", _decode_optional(snapshots["manifest"])),
            (run_path / "research_bias_audit.json", _decode_optional(snapshots["bias"])),
        )
    for path, content in targets:
        _restore_optional_file(path, content)
    _clear_transaction_journal(_registry_transaction_path(reg_path))


def _restore_optional_file(path: Path, content: bytes | None) -> None:
    if content is None:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"transaction target must be a regular, non-symlink file: {path}")
        if path.exists():
            path.unlink()
            _fsync_directory(path.parent)
        return
    _atomic_write_bytes(path, content)


def _clear_transaction_journal(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"transaction journal must be a regular, non-symlink file: {path}")
    if path.exists():
        path.unlink()
        _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
