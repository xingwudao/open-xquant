"""Reproducibility Audit — verify same input produces same output."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from oxq.market_calendar import normalize_exchange_calendar
from oxq.run_digests import run_digest_transaction


def audit_reproducibility(run_dir: str | Path) -> dict:
    """Audit one coherent, transaction-bound generation of a backtest run."""
    run_path = Path(os.path.abspath(run_dir))
    with run_digest_transaction(run_path):
        return _audit_reproducibility_locked(run_path)


def _audit_reproducibility_locked(run_path: Path) -> dict:
    """Verify that a backtest run's core outputs are consistent.

    Checks spec hash, data manifest hash, trades hash, equity curve hash,
    and metrics hash. Returns a report dict with per-check status.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run directory (e.g. runs/20260616_153000_strategy_id/).

    Returns
    -------
    dict
        Audit result with 'status', 'checks', and summary fields.
    """
    checks: list[dict] = []

    # Check required files exist
    required_files = [
        "strategy_spec.yaml",
        "spec_hash.txt",
        "environment.json",
        "data_manifest.json",
        "metrics.json",
        "equity_curve.csv",
        "trades.csv",
        "artifact_hashes.json",
    ]
    required_paths: dict[str, Path] = {}
    invalid_required: list[str] = []
    for artifact_name in required_files:
        try:
            required_paths[artifact_name] = validate_run_artifact_path(
                run_path,
                artifact_name,
            )
        except OSError as exc:
            invalid_required.append(f"{artifact_name}: {exc}")
    if invalid_required:
        return {
            "status": "fail",
            "checks": [
                {
                    "id": "missing_files",
                    "status": "fail",
                    "severity": "fatal",
                    "message": f"Missing or unsafe files: {invalid_required}",
                }
            ],
            "fatal_count": 1,
            "warning_count": 0,
        }

    # Verify spec hash consistency — use the same canonical hash from StrategySpec
    parsed_spec = None
    try:
        from oxq.spec.schema import StrategySpec
        parsed_spec = StrategySpec.from_yaml(str(required_paths["strategy_spec.yaml"]))
        spec_hash_actual = parsed_spec.compute_hash()
    except Exception:
        spec_yaml = required_paths["strategy_spec.yaml"].read_text(encoding="utf-8")
        spec_hash_actual = f"sha256:{hashlib.sha256(spec_yaml.encode()).hexdigest()[:16]}"
    spec_hash_stored = required_paths["spec_hash.txt"].read_text(encoding="utf-8").strip()
    checks.append(
        _check(
            "spec_hash",
            spec_hash_actual == spec_hash_stored,
            "fatal",
            f"Spec hash mismatch: stored={spec_hash_stored}, actual={spec_hash_actual}",
        )
    )

    # Verify environment.json is valid
    env = {}
    try:
        parsed_env = json.loads(required_paths["environment.json"].read_text(encoding="utf-8"))
        if not isinstance(parsed_env, dict):
            checks.append(_check("environment", False, "fatal", "environment.json must be an object"))
        else:
            env = parsed_env
            has_spec_hash = "spec_hash" in env
            has_version = "open_xquant_version" in env
            checks.append(_check("environment", has_spec_hash and has_version, "warning", "environment.json missing spec_hash or version"))
    except Exception:
        checks.append(_check("environment", False, "warning", "environment.json is invalid JSON"))

    # Verify data_manifest.json is valid
    manifest_schema_version = 0
    manifest = {}
    try:
        parsed_manifest = json.loads(required_paths["data_manifest.json"].read_text(encoding="utf-8"))
        if not isinstance(parsed_manifest, dict):
            checks.append(_check("data_manifest", False, "fatal", "data_manifest.json must be an object"))
        else:
            manifest = parsed_manifest
    except (json.JSONDecodeError, OSError):
        checks.append(_check("data_manifest", False, "warning", "data_manifest.json is invalid JSON"))
    if manifest:
        try:
            manifest_schema_version = int(manifest.get("schema_version", 0) or 0)
        except (TypeError, ValueError):
            checks.append(_check("data_manifest", False, "fatal", "data_manifest.json has invalid schema_version"))
            manifest_schema_version = 1
        symbols = manifest.get("symbols")
        if not isinstance(symbols, list) or any(not isinstance(symbol, str) for symbol in symbols):
            checks.append(_check("data_manifest", False, "fatal", "data_manifest.json symbols must be a list of strings"))
        else:
            checks.append(_check("data_manifest", len(symbols) > 0, "warning", "data_manifest.json has no symbols"))

    required_artifact_hashes = {
        "data_manifest.json": "data_manifest_hash",
        "equity_curve.csv": "equity_hash",
        "trades.csv": "trades_hash",
        "metrics.json": "metrics_hash",
    }
    new_required_artifact_hashes = {
        **required_artifact_hashes,
        "strategy_spec.yaml": "strategy_spec_file_hash",
        "environment.json": "environment_hash",
        "execution_assumptions.json": "execution_assumptions_hash",
        "positions.csv": "positions_hash",
        "orders.csv": "orders_hash",
    }
    schema_3_required_artifact_hashes = {
        **new_required_artifact_hashes,
        "target_weights.csv": "target_weights_hash",
    }
    schema_4_required_artifact_hashes = {
        **schema_3_required_artifact_hashes,
        "compiled_plan.json": "compiled_plan_hash",
    }
    schema_5_required_artifact_hashes = {
        **schema_4_required_artifact_hashes,
        "strategy.py": "strategy_py_hash",
    }
    provenance_artifact_hashes = {
        "spec_audit.json": "spec_audit_hash",
        "conversation_hash.txt": "conversation_hash_file_hash",
        "component_catalog_hash.txt": "component_catalog_hash_file_hash",
        "recipe_catalog_hash.txt": "recipe_catalog_hash_file_hash",
    }
    optional_provenance_artifact_hashes = {
        "runtime_audit.json": "runtime_audit_hash",
        "component_manifest.json": "component_manifest_hash",
        "component_manifests.json": "component_manifests_hash",
        "component_bundle_hash.txt": "component_bundle_hash_file_hash",
    }
    artifact_schema_version = 0
    try:
        expected_hashes = json.loads(required_paths["artifact_hashes.json"].read_text(encoding="utf-8"))
        valid_hash_manifest = isinstance(expected_hashes, dict)
        unsafe_hash_keys: list[str] = []
        if not isinstance(expected_hashes, dict):
            checks.append(_check("artifact_hashes", False, "fatal", "artifact_hashes.json must be an object"))
            expected_hashes = {}
            missing_hash_keys = []
            required_hashes = required_artifact_hashes
        else:
            try:
                artifact_schema_version = int(expected_hashes.get("schema_version", 0) or 0)
                if manifest_schema_version >= 1 and artifact_schema_version < 1:
                    checks.append(_check(
                        "artifact_hashes",
                        False,
                        "fatal",
                        "artifact_hashes.json schema_version must be >= 1 for data_manifest schema_version >= 1",
                    ))
                if artifact_schema_version >= 5:
                    required_hashes = schema_5_required_artifact_hashes
                elif artifact_schema_version >= 4:
                    required_hashes = schema_4_required_artifact_hashes
                elif artifact_schema_version >= 3:
                    required_hashes = schema_3_required_artifact_hashes
                elif artifact_schema_version >= 2:
                    required_hashes = new_required_artifact_hashes
                elif manifest_schema_version >= 1 or artifact_schema_version >= 1:
                    required_hashes = {
                        **required_artifact_hashes,
                        "strategy_spec.yaml": "strategy_spec_file_hash",
                        "environment.json": "environment_hash",
                        "positions.csv": "positions_hash",
                        "orders.csv": "orders_hash",
                    }
                else:
                    required_hashes = required_artifact_hashes
                optional_artifact_hashes = {
                    "benchmark_curve.csv": "benchmark_hash",
                    "benchmark_equity_curve.csv": "benchmark_equity_hash",
                    "benchmark_prices.csv": "benchmark_prices_hash",
                    "robustness.json": "robustness_hash",
                }
                if artifact_schema_version >= 2:
                    optional_artifact_hashes["target_weights.csv"] = "target_weights_hash"
                for artifact_name, check_id in optional_artifact_hashes.items():
                    if artifact_name in expected_hashes or _artifact_path_present(run_path, artifact_name):
                        required_hashes = {
                            **required_hashes,
                            artifact_name: check_id,
                        }
                provenance_present = any(
                    artifact_name in expected_hashes or _artifact_path_present(run_path, artifact_name)
                    for artifact_name in provenance_artifact_hashes
                )
                if provenance_present:
                    required_hashes = {
                        **required_hashes,
                        **provenance_artifact_hashes,
                    }
                for artifact_name, check_id in optional_provenance_artifact_hashes.items():
                    if artifact_name in expected_hashes or _artifact_path_present(run_path, artifact_name):
                        required_hashes = {
                            **required_hashes,
                            artifact_name: check_id,
                        }
                if artifact_schema_version < 4 and "compiled_plan.json" in expected_hashes:
                    required_hashes = {
                        **required_hashes,
                        "compiled_plan.json": "compiled_plan_hash",
                    }
                if artifact_schema_version < 5 and "strategy.py" in expected_hashes:
                    required_hashes = {
                        **required_hashes,
                        "strategy.py": "strategy_py_hash",
                    }
                for artifact_name in expected_hashes:
                    if artifact_name == "schema_version":
                        continue
                    if not _is_safe_artifact_name(artifact_name):
                        unsafe_hash_keys.append(str(artifact_name))
                        continue
                    if artifact_name not in required_hashes:
                        required_hashes = {
                            **required_hashes,
                            artifact_name: _unknown_artifact_check_id(artifact_name),
                        }
                missing_hash_keys = sorted(set(required_hashes).difference(expected_hashes))
            except (TypeError, ValueError):
                checks.append(_check("artifact_hashes", False, "fatal", "artifact_hashes.json has invalid schema_version"))
                expected_hashes = {}
                missing_hash_keys = []
                required_hashes = required_artifact_hashes
        if valid_hash_manifest and not expected_hashes:
            checks.append(_check("artifact_hashes", False, "fatal", "artifact_hashes.json is empty"))
        elif valid_hash_manifest and missing_hash_keys:
            checks.append(_check(
                "artifact_hashes",
                False,
                "fatal",
                f"artifact_hashes.json missing required keys: {missing_hash_keys}",
            ))
        if valid_hash_manifest and unsafe_hash_keys:
            checks.append(_check(
                "artifact_hashes",
                False,
                "fatal",
                f"artifact_hashes.json contains unsafe artifact paths: {sorted(unsafe_hash_keys)}",
            ))
    except (json.JSONDecodeError, OSError):
        checks.append(_check("artifact_hashes", False, "fatal", "artifact_hashes.json is invalid JSON"))
        expected_hashes = {}
        missing_hash_keys = []
        required_hashes = required_artifact_hashes

    if expected_hashes and not missing_hash_keys:
        run_digest_check = _check_run_digest_locked(run_path)
        if run_digest_check is not None:
            checks.append(run_digest_check)
        for fname, check_id in required_hashes.items():
            try:
                actual = _hash_artifact(validate_run_artifact_path(run_path, fname))
                expected = expected_hashes.get(fname)
                checks.append(_check(check_id, actual == expected, "fatal", f"{fname} hash mismatch: stored={expected}, actual={actual}"))
            except (json.JSONDecodeError, OSError):
                checks.append(_check(check_id, False, "fatal", f"{fname} is corrupted or unreadable"))
        checks.extend(_check_component_bundle_hashes(run_path))
        if artifact_schema_version >= 4 or "compiled_plan.json" in expected_hashes:
            checks.append(_check_compiled_plan_consistency(run_path, spec_hash_actual, parsed_spec, env, manifest))

    hash_guard_failed = any(
        c["id"] in {"artifact_hashes", "environment_hash", "data_manifest_hash"}
        and c["severity"] == "fatal"
        and c["status"] == "fail"
        for c in checks
    )
    if manifest and not hash_guard_failed:
        checks.extend(_check_data_fingerprints(manifest, env.get("data_dir"), enforce=manifest_schema_version >= 1))

    fatal_count = sum(1 for c in checks if c["severity"] == "fatal" and c["status"] == "fail")
    warning_count = sum(1 for c in checks if c["severity"] == "warning" and c["status"] == "fail")
    has_fatal = any(c["severity"] == "fatal" and c["status"] == "fail" for c in checks)

    return {
        "status": "fail" if has_fatal else "pass",
        "checks": checks,
        "fatal_count": fatal_count,
        "warning_count": warning_count,
    }


def _check(check_id: str, passed: bool, severity: str, message: str) -> dict:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "severity": severity,
        "message": message if not passed else f"{check_id}: OK",
    }


def _hash_artifact(path: Path) -> str:
    if path.name == "metrics.json":
        return _hash_json_file(path, exclude_keys={"run_id"})
    if path.name == "environment.json":
        return _hash_json_file(path, exclude_keys={"run_timestamp"})
    if path.name in {
        "data_manifest.json",
        "execution_assumptions.json",
        "compiled_plan.json",
        "spec_audit.json",
        "runtime_audit.json",
        "component_manifest.json",
        "component_manifests.json",
    }:
        return _hash_json_file(path)
    content = path.read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"


def _check_component_bundle_hashes(run_path: Path) -> list[dict]:
    if not _artifact_path_present(run_path, "component_manifests.json"):
        return []
    try:
        summary_path = validate_run_artifact_path(run_path, "component_manifests.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [_check("component_bundle_hash", False, "fatal", f"component_manifests.json is invalid: {exc}")]
    if not isinstance(summary, list):
        return [_check("component_bundle_hash", False, "fatal", "component_manifests.json must be a list")]

    try:
        from oxq.core.component_manifest import compute_component_bundle_hash
    except Exception as exc:
        return [_check("component_bundle_hash", False, "fatal", f"component bundle verifier could not load: {exc}")]

    checks: list[dict] = []
    for index, item in enumerate(summary):
        if not isinstance(item, dict):
            checks.append(_check("component_bundle_hash", False, "fatal", f"component_manifests.json[{index}] must be an object"))
            continue
        recorded = item.get("bundle_hash")
        if not isinstance(recorded, str) or not recorded:
            checks.append(_check("component_bundle_hash", False, "fatal", f"component_manifests.json[{index}].bundle_hash is required"))
            continue
        try:
            manifest_path = _resolve_component_manifest_for_audit(run_path, item, recorded, len(summary))
            actual = compute_component_bundle_hash(manifest_path)
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_hash = manifest_payload.get("bundle_hash") if isinstance(manifest_payload, dict) else None
            if manifest_hash != recorded:
                checks.append(_check(
                    "component_bundle_hash",
                    False,
                    "fatal",
                    "component bundle "
                    f"{index} manifest hash mismatch: stored={recorded}, manifest={manifest_hash}",
                ))
                continue
        except Exception as exc:
            checks.append(_check("component_bundle_hash", False, "fatal", f"component bundle {index} could not be verified: {exc}"))
            continue
        checks.append(_check(
            "component_bundle_hash",
            actual == recorded,
            "fatal",
            f"component bundle {index} hash mismatch: stored={recorded}, actual={actual}",
        ))
    return checks


def _resolve_component_manifest_for_audit(run_path: Path, item: dict, recorded_hash: str, summary_count: int) -> Path:
    archived_path = item.get("archived_manifest_path")
    if isinstance(archived_path, str) and archived_path:
        if not _artifact_path_present(run_path, archived_path):
            raise OSError(f"archived component manifest not found: {archived_path}")
        try:
            archived = validate_run_artifact_path(run_path, archived_path)
        except OSError as exc:
            if "symlink" in str(exc):
                raise OSError(
                    f"archived component manifest must not be a symlink: {archived_path}"
                ) from exc
            raise
        _verify_run_local_component_manifest(run_path, archived, str(archived_path), recorded_hash)
        return archived

    if summary_count == 1 and _artifact_path_present(run_path, "component_manifest.json"):
        try:
            legacy = validate_run_artifact_path(run_path, "component_manifest.json")
        except OSError as exc:
            if "symlink" in str(exc):
                raise OSError(
                    "archived component manifest must not be a symlink: component_manifest.json"
                ) from exc
            raise
        _verify_run_local_component_manifest(run_path, legacy, "component_manifest.json", recorded_hash)
        return legacy

    manifest_path = item.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise OSError("component manifest path is required")
    candidate = Path(manifest_path)
    try:
        relative = candidate.relative_to(run_path) if candidate.is_absolute() else candidate
    except ValueError as exc:
        raise OSError(
            f"component manifest must stay within the run directory: {candidate}"
        ) from exc
    try:
        local_manifest = validate_run_artifact_path(run_path, relative.as_posix())
    except OSError as exc:
        raise OSError(f"component manifest must stay within the run directory: {candidate}") from exc
    _verify_run_local_component_manifest(
        run_path,
        local_manifest,
        str(manifest_path),
        recorded_hash,
    )
    return local_manifest


def _verify_run_local_component_manifest(run_path: Path, path: Path, label: str, recorded_hash: str) -> None:
    try:
        relative = path.relative_to(run_path).as_posix()
        path = validate_run_artifact_path(run_path, relative)
    except (OSError, ValueError) as exc:
        raise OSError(
            f"archived component manifest must be a run-local regular file without symlinks: {label}"
        ) from exc
    from oxq.core.component_manifest import compute_component_bundle_hash

    actual = compute_component_bundle_hash(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_hash = payload.get("bundle_hash") if isinstance(payload, dict) else None
    if manifest_hash == recorded_hash and actual == recorded_hash:
        return
    raise OSError(
        "archived component manifest hash mismatch: "
        f"stored={recorded_hash}, manifest={manifest_hash}, actual={actual}"
    )


def _unknown_artifact_check_id(artifact_name: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in artifact_name).strip("_")
    return f"{normalized}_hash" if normalized else "unknown_artifact_hash"


def validate_run_artifact_path(run_path: str | Path, artifact_name: str) -> Path:
    """Return one regular, non-symlinked artifact canonically contained by a run."""
    if not _is_safe_artifact_name(artifact_name):
        raise OSError(f"unsafe artifact path: {artifact_name}")
    lexical_run = Path(os.path.abspath(run_path))
    if lexical_run.is_symlink() or not lexical_run.is_dir():
        raise OSError(f"run directory must be a regular directory without symlinks: {lexical_run}")
    path = lexical_run / artifact_name
    current = lexical_run
    for part in Path(artifact_name).parts:
        current /= part
        if current.is_symlink():
            raise OSError(f"artifact path must not contain symlinks: {artifact_name}")
    if not path.is_file():
        raise OSError(f"artifact path must be a regular file: {artifact_name}")
    try:
        resolved_run = lexical_run.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_run)
    except (OSError, ValueError):
        raise OSError(f"unsafe artifact path escapes run directory: {artifact_name}")
    return path


def _artifact_path_present(run_path: Path, artifact_name: str) -> bool:
    if not _is_safe_artifact_name(artifact_name):
        return False
    path = run_path / artifact_name
    return path.exists() or path.is_symlink()


def _is_safe_artifact_name(artifact_name: str) -> bool:
    path = Path(artifact_name)
    return bool(artifact_name) and not path.is_absolute() and ".." not in path.parts


def _check_compiled_plan_consistency(
    run_path: Path,
    spec_hash_actual: str,
    spec: object | None = None,
    env: dict | None = None,
    manifest: dict | None = None,
) -> dict:
    if not _artifact_path_present(run_path, "compiled_plan.json"):
        return _check("compiled_plan_consistency", False, "fatal", "compiled_plan.json is missing")

    try:
        compiled_plan_path = validate_run_artifact_path(run_path, "compiled_plan.json")
        compiled_plan = json.loads(compiled_plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _check("compiled_plan_consistency", False, "fatal", f"compiled_plan.json consistency check failed: {exc}")
    if not isinstance(compiled_plan, dict):
        return _check("compiled_plan_consistency", False, "fatal", "compiled_plan.json must be an object")
    if compiled_plan.get("spec_hash") != spec_hash_actual:
        return _check(
            "compiled_plan_consistency",
            False,
            "fatal",
            "compiled_plan.json spec_hash mismatch: "
            f"stored={compiled_plan.get('spec_hash')}, actual={spec_hash_actual}",
        )
    if spec is None:
        return _check(
            "compiled_plan_consistency",
            False,
            "fatal",
            "strategy_spec.yaml could not be parsed; compiled_plan.json cannot be rebuilt",
        )
    try:
        from oxq.spec.compiler import (
            _build_compiled_plan_from_spec_metadata,
            _compiled_plan_material_differences,
        )

        expected_plan = _build_compiled_plan_from_spec_metadata(
            spec,
            effective_data_dir=_trusted_effective_data_dir(env or {}, manifest or {}),
            component_manifests=_read_component_manifests_for_plan(run_path),
        )
    except Exception as exc:
        return _check(
            "compiled_plan_consistency",
            False,
            "fatal",
            f"compiled_plan.json consistency check failed while rebuilding plan: {exc}",
        )
    required_oxq_version = getattr(spec, "required_oxq_version", None)
    if not isinstance(required_oxq_version, str) or not required_oxq_version:
        required_oxq_version = None
    differing = _compiled_plan_material_differences(
        compiled_plan,
        expected_plan,
        required_oxq_version=required_oxq_version,
    )
    if differing:
        return _check(
            "compiled_plan_consistency",
            False,
            "fatal",
            "compiled_plan.json material fields differ from strategy_spec.yaml: "
            + ", ".join(differing),
        )
    return _check(
        "compiled_plan_consistency",
        True,
        "fatal",
        "compiled_plan.json material fields match strategy_spec.yaml",
    )


def _read_component_manifests_for_plan(run_path: Path) -> list[dict]:
    if not _artifact_path_present(run_path, "component_manifests.json"):
        return []
    summary_path = validate_run_artifact_path(run_path, "component_manifests.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, list):
        raise ValueError("component_manifests.json must be a list")
    manifests: list[dict] = []
    for index, item in enumerate(summary):
        if not isinstance(item, dict):
            raise ValueError(f"component_manifests.json[{index}] must be an object")
        recorded = item.get("bundle_hash")
        if not isinstance(recorded, str) or not recorded:
            raise ValueError(f"component_manifests.json[{index}].bundle_hash is required")
        manifest_path = _resolve_component_manifest_for_audit(run_path, item, recorded, len(summary))
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"component manifest must be an object: {manifest_path}")
        manifests.append(payload)
    return manifests


def _trusted_effective_data_dir(env: dict, manifest: dict) -> str | None:
    data_dir = env.get("data_dir")
    if isinstance(data_dir, str) and data_dir:
        return data_dir
    for key in ("effective_data_dir", "data_dir"):
        data_dir = manifest.get(key)
        if isinstance(data_dir, str) and data_dir:
            return data_dir
    return None


def _hash_json_file(path: Path, exclude_keys: set[str] | None = None) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and exclude_keys:
        data = {key: value for key, value in data.items() if key not in exclude_keys}
    canonical = json.dumps(data, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _check_run_digest(run_path: Path) -> dict | None:
    with run_digest_transaction(run_path):
        return _check_run_digest_locked(run_path)


def _check_run_digest_locked(run_path: Path) -> dict | None:
    digest_path = run_path.parent / "run_digests.jsonl"
    if not digest_path.exists():
        return None

    from oxq.run_digests import RunDigestError, require_current_run_digest

    try:
        require_current_run_digest(run_path)
    except RunDigestError as exc:
        return _check("run_digest", False, "fatal", str(exc))
    return _check("run_digest", True, "fatal", "run digest matches artifact_hashes.json")


def _check_data_fingerprints(manifest: dict, data_dir: str | None, enforce: bool) -> list[dict]:
    fingerprints = manifest.get("data_fingerprints")
    if not fingerprints:
        severity = "fatal" if enforce else "warning"
        return [_check("data_fingerprint", False, severity, "data_manifest.json has no data_fingerprints")]
    if not isinstance(fingerprints, dict):
        severity = "fatal" if enforce else "warning"
        return [_check("data_fingerprint", False, severity, "data_manifest.json data_fingerprints must be an object")]
    if not data_dir:
        severity = "fatal" if enforce else "warning"
        return [_check("data_fingerprint", False, severity, "source data_dir unavailable; data fingerprints were not verified")]

    data_path = Path(data_dir).resolve()
    mismatches = []
    manifest_symbols = manifest.get("symbols")
    if isinstance(manifest_symbols, list):
        expected_symbols = set(manifest_symbols)
        fingerprint_symbols = set(fingerprints)
        if expected_symbols != fingerprint_symbols:
            missing = sorted(expected_symbols - fingerprint_symbols)
            extra = sorted(fingerprint_symbols - expected_symbols)
            details = []
            if missing:
                details.append(f"missing fingerprints for {missing}")
            if extra:
                details.append(f"unexpected fingerprints for {extra}")
            mismatches.append("; ".join(details))
    for symbol, expected in fingerprints.items():
        if not isinstance(expected, dict):
            mismatches.append(f"{symbol}: fingerprint must be an object")
            continue
        missing_fields = sorted({"start", "end", "columns", "content_hash"} - set(expected))
        if missing_fields:
            mismatches.append(f"{symbol}: fingerprint missing fields {missing_fields}")
            continue
        source_path = data_path / f"{symbol}.parquet"
        if _unsafe_data_symbol(symbol) or not source_path.resolve().is_relative_to(data_path):
            mismatches.append(f"{symbol}: unsafe source data path")
            continue
        if not source_path.exists():
            mismatches.append(f"{symbol}: source file missing")
            continue
        try:
            df = pd.read_parquet(source_path)
            df = _normalize_provider_index(df)
            df = _slice_to_manifest_range(df, manifest)
            df = _align_to_calendar_sessions(df, manifest.get("calendar"), manifest.get("start"), manifest.get("end"))
            actual = _fingerprint_dataframe(df, expected.get("columns") or None)
        except Exception as exc:
            mismatches.append(f"{symbol}: cannot fingerprint source data ({exc})")
            continue
        if actual != expected:
            mismatches.append(f"{symbol}: stored={expected}, actual={actual}")

    return [_check("data_fingerprint", not mismatches, "fatal", "Source data fingerprint mismatch: " + "; ".join(mismatches))]


def _unsafe_data_symbol(symbol: str) -> bool:
    if not symbol or "/" in symbol or "\\" in symbol:
        return True
    path = Path(symbol)
    return path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)


def _normalize_provider_index(df: pd.DataFrame) -> pd.DataFrame:
    """Match LocalMarketDataProvider's timezone handling for local parquet data."""
    if hasattr(df.index, "tz") and df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


def _slice_to_fingerprint_range(df: pd.DataFrame, expected: dict) -> pd.DataFrame:
    start = expected.get("start")
    end = expected.get("end")
    if df.empty or not start or not end:
        return df
    index = pd.DatetimeIndex(df.index)
    start_ts = _coerce_timestamp(start, index)
    end_ts = _coerce_timestamp(end, index)
    mask = (index >= start_ts) & (index <= end_ts)
    return df.loc[mask]


def _slice_to_manifest_range(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    start = manifest.get("start")
    end = manifest.get("end")
    if df.empty or not start or not end:
        return df
    index = pd.DatetimeIndex(df.index)
    start_date = pd.Timestamp(str(start)).date()
    end_date = pd.Timestamp(str(end)).date()
    session_dates = pd.Index([pd.Timestamp(idx).date() for idx in index])
    return df.loc[(session_dates >= start_date) & (session_dates <= end_date)]


def _align_to_calendar_sessions(df: pd.DataFrame, calendar: object, start: object, end: object) -> pd.DataFrame:
    if not isinstance(calendar, str) or not calendar:
        return df
    if not isinstance(start, str) or not isinstance(end, str) or not start or not end:
        return df
    import exchange_calendars as xcals

    cal = xcals.get_calendar(normalize_exchange_calendar(calendar))
    sessions = cal.sessions_in_range(pd.Timestamp(start).date(), pd.Timestamp(end).date())
    return _select_frame_for_session_fingerprint(df, pd.DatetimeIndex(sessions))


def _select_frame_for_session_fingerprint(df: pd.DataFrame, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty:
        return df.reindex(expected_index)
    source = df.copy()
    source_index = pd.DatetimeIndex(source.index)
    missing_index = _expected_index_with_source_tz(expected_index, source_index)
    session_dates = pd.Index([pd.Timestamp(idx).date() for idx in source.index])
    if session_dates.has_duplicates:
        raise ValueError("market data has multiple rows for the same market session")
    expected_dates = pd.Index([pd.Timestamp(idx).date() for idx in expected_index])
    source_by_date = dict(zip(session_dates, range(len(source)), strict=True))
    rows: list[pd.Series] = []
    index_values: list[object] = []
    for expected_date, missing_ts in zip(expected_dates, missing_index, strict=True):
        source_pos = source_by_date.get(expected_date)
        if source_pos is None:
            rows.append(pd.Series(index=source.columns, dtype="object"))
            index_values.append(missing_ts)
            continue
        rows.append(source.iloc[source_pos])
        index_values.append(source.index[source_pos])
    aligned = pd.DataFrame(rows)
    aligned.index = pd.Index(index_values)
    return aligned


def _expected_index_with_source_tz(
    expected_index: pd.DatetimeIndex,
    source_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    if source_index.tz is not None and expected_index.tz is None:
        return expected_index.tz_localize(source_index.tz)
    if source_index.tz is None and expected_index.tz is not None:
        return expected_index.tz_localize(None)
    if source_index.tz is not None and expected_index.tz is not None:
        return expected_index.tz_convert(source_index.tz)
    return expected_index


def _coerce_timestamp(value: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if index.tz is None:
        return ts.tz_localize(None) if ts.tz is not None else ts
    return ts.tz_localize(index.tz) if ts.tz is None else ts.tz_convert(index.tz)


def _fingerprint_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    if df.empty:
        return {
            "row_count": 0,
            "start": "",
            "end": "",
            "columns": columns or [],
            "content_hash": "sha256:e3b0c44298fc1c14",
        }
    frame = df.sort_index()
    check_columns = columns or list(frame.columns)
    frame = frame.reindex(columns=check_columns)
    records = []
    for idx, row in frame.iterrows():
        record = {"__index__": pd.Timestamp(idx).isoformat()}
        for col in check_columns:
            value = row[col]
            record[col] = None if pd.isna(value) else value
        records.append(record)
    payload = json.dumps({"columns": check_columns, "records": records}, sort_keys=True, default=str)
    index = pd.DatetimeIndex(frame.index)
    return {
        "row_count": int(len(frame)),
        "start": pd.Timestamp(index.min()).isoformat(),
        "end": pd.Timestamp(index.max()).isoformat(),
        "columns": check_columns,
        "content_hash": f"sha256:{hashlib.sha256(payload.encode()).hexdigest()[:16]}",
    }
