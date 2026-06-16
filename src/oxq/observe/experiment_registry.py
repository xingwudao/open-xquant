"""Experiment Registry — JSONL-based experiment tracking.

Thin wrapper over a JSONL file that records every research run
to prevent selective memory. Complements the in-memory
:class:`oxq.observe.experiment.ExperimentLog`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = "experiments.jsonl"


def add_experiment(
    run_dir: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    decision: str = "unknown",
) -> dict[str, Any]:
    """Append a backtest run to the experiment registry.

    Reads metrics.json and spec_hash.txt from *run_dir*, constructs
    an experiment entry, and appends it as a JSON line to *registry_path*.

    Returns the entry dict with the generated experiment_id.
    """
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    if not metrics_path.exists():
        return {"error": f"metrics.json not found in {run_dir}"}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    spec_hash = ""
    spec_hash_path = run_path / "spec_hash.txt"
    if spec_hash_path.exists():
        spec_hash = spec_hash_path.read_text(encoding="utf-8").strip()

    audit_status = "unknown"
    bias_path = run_path / "research_bias_audit.json"
    if bias_path.exists():
        bias = json.loads(bias_path.read_text(encoding="utf-8"))
        audit_status = bias.get("status", "unknown")

    entry = {
        "experiment_id": f"exp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "strategy_id": metrics.get("strategy_id", ""),
        "spec_hash": spec_hash,
        "run_id": metrics.get("run_id", ""),
        "metrics": metrics,
        "audit_status": audit_status,
        "decision": decision,
        "created_at": datetime.now(UTC).isoformat(),
    }

    reg_path = Path(registry_path)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def list_experiments(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Read all experiments from the registry file.

    Returns a list of experiment entry dicts. Returns empty list if the
    registry file does not exist.
    """
    reg_path = Path(registry_path)
    if not reg_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(reg_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
