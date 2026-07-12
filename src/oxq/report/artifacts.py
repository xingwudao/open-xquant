"""Stable readers for report-relevant run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

_GOVERNED_REPORT_REQUIRED_SOURCE_ARTIFACTS = frozenset(
    {
        "reproducibility_audit.json",
        "research_bias_audit.json",
    }
)


@dataclass(frozen=True)
class RunArtifacts:
    """Loaded report inputs for a single run directory."""

    run_dir: Path
    strategy_spec: dict[str, Any]
    metrics: dict[str, Any]
    execution_assumptions: dict[str, Any] | None
    reproducibility_audit: dict[str, Any] | None
    research_bias_audit: dict[str, Any] | None
    robustness: dict[str, Any] | None
    artifact_hashes: dict[str, Any] | None
    data_manifest: dict[str, Any] | None
    asset_manifest: dict[str, Any] | None
    equity_curve: pd.DataFrame
    benchmark_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    target_weights: pd.DataFrame

    @classmethod
    def load(cls, run_dir: str | Path) -> RunArtifacts:
        run_path = Path(run_dir)
        return cls(
            run_dir=run_path,
            strategy_spec=_read_yaml_object(run_path / "strategy_spec.yaml"),
            metrics=_read_json_object(run_path / "metrics.json") or {},
            execution_assumptions=_read_json_object(run_path / "execution_assumptions.json"),
            reproducibility_audit=_read_json_object(run_path / "reproducibility_audit.json"),
            research_bias_audit=_read_json_object(run_path / "research_bias_audit.json"),
            robustness=_read_json_object(run_path / "robustness.json"),
            artifact_hashes=_read_json_object(run_path / "artifact_hashes.json"),
            data_manifest=_read_json_object(run_path / "data_manifest.json"),
            asset_manifest=_read_json_object(run_path / "report_assets" / "manifest.json"),
            equity_curve=_read_csv(run_path / "equity_curve.csv"),
            benchmark_curve=_read_csv(run_path / "benchmark_curve.csv"),
            trades=_read_csv(run_path / "trades.csv"),
            positions=_read_csv(run_path / "positions.csv"),
            target_weights=_read_csv(run_path / "target_weights.csv"),
        )


def require_governed_report_source_hashes(run_dir: str | Path) -> None:
    """Require governed monitor audits and their artifact manifest entries."""
    run_path = Path(run_dir)
    missing_required_files = sorted(
        name
        for name in _GOVERNED_REPORT_REQUIRED_SOURCE_ARTIFACTS
        if not (run_path / name).is_file()
    )
    if missing_required_files:
        raise ValueError(
            "governed report source artifacts are required before report QA: "
            f"{missing_required_files}"
        )

    manifest_path = run_path / "artifact_hashes.json"
    try:
        artifact_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact_hashes.json must contain a valid JSON object") from exc
    if not isinstance(artifact_hashes, dict):
        raise ValueError("artifact_hashes.json must contain a valid JSON object")

    missing_hashes = sorted(
        name
        for name in _GOVERNED_REPORT_REQUIRED_SOURCE_ARTIFACTS
        if not isinstance(artifact_hashes.get(name), str) or not artifact_hashes[name]
    )
    if missing_hashes:
        raise ValueError(
            "artifact_hashes.json missing required hashes for governed report source artifacts: "
            f"{missing_hashes}"
        )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_yaml_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
