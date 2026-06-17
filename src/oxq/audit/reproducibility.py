"""Reproducibility Audit — verify same input produces same output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def audit_reproducibility(run_dir: str | Path) -> dict:
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
    run_path = Path(run_dir)
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
    missing = [f for f in required_files if not (run_path / f).exists()]
    if missing:
        return {
            "status": "fail",
            "checks": [{"id": "missing_files", "status": "fail", "severity": "fatal", "message": f"Missing files: {missing}"}],
            "fatal_count": 1,
            "warning_count": 0,
        }

    # Verify spec hash consistency — use the same canonical hash from StrategySpec
    try:
        from oxq.spec.schema import StrategySpec
        parsed = StrategySpec.from_yaml(str(run_path / "strategy_spec.yaml"))
        spec_hash_actual = parsed.compute_hash()
    except Exception:
        spec_yaml = (run_path / "strategy_spec.yaml").read_text(encoding="utf-8")
        spec_hash_actual = f"sha256:{hashlib.sha256(spec_yaml.encode()).hexdigest()[:16]}"
    spec_hash_stored = (run_path / "spec_hash.txt").read_text(encoding="utf-8").strip()
    checks.append(
        _check(
            "spec_hash",
            spec_hash_actual == spec_hash_stored,
            "fatal",
            f"Spec hash mismatch: stored={spec_hash_stored}, actual={spec_hash_actual}",
        )
    )

    # Verify environment.json is valid
    try:
        env = json.loads((run_path / "environment.json").read_text(encoding="utf-8"))
        has_spec_hash = "spec_hash" in env
        has_version = "open_xquant_version" in env
        checks.append(_check("environment", has_spec_hash and has_version, "warning", "environment.json missing spec_hash or version"))
    except Exception:
        checks.append(_check("environment", False, "warning", "environment.json is invalid JSON"))

    # Verify data_manifest.json is valid
    try:
        manifest = json.loads((run_path / "data_manifest.json").read_text(encoding="utf-8"))
        has_symbols = "symbols" in manifest and len(manifest["symbols"]) > 0
        checks.append(_check("data_manifest", has_symbols, "warning", "data_manifest.json has no symbols"))
    except Exception:
        checks.append(_check("data_manifest", False, "warning", "data_manifest.json is invalid JSON"))

    try:
        expected_hashes = json.loads((run_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        checks.append(_check("artifact_hashes", False, "fatal", "artifact_hashes.json is invalid JSON"))
        expected_hashes = {}

    if expected_hashes:
        for fname, check_id in [
            ("data_manifest.json", "data_manifest_hash"),
            ("equity_curve.csv", "equity_hash"),
            ("trades.csv", "trades_hash"),
            ("metrics.json", "metrics_hash"),
        ]:
            try:
                if fname == "metrics.json":
                    actual = _hash_json_file(run_path / fname, exclude_keys={"run_id"})
                elif fname == "data_manifest.json":
                    actual = _hash_json_file(run_path / fname)
                else:
                    content = (run_path / fname).read_bytes()
                    actual = f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"
                expected = expected_hashes.get(fname)
                checks.append(_check(check_id, actual == expected, "fatal", f"{fname} hash mismatch: stored={expected}, actual={actual}"))
            except (json.JSONDecodeError, OSError):
                checks.append(_check(check_id, False, "fatal", f"{fname} is corrupted or unreadable"))

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


def _hash_json_file(path: Path, exclude_keys: set[str] | None = None) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and exclude_keys:
        data = {key: value for key, value in data.items() if key not in exclude_keys}
    canonical = json.dumps(data, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
