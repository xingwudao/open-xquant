"""Report-adjacent tools for experiment tracking."""

from __future__ import annotations

from typing import Any

from oxq.tools.registry import registry


@registry.tool(
    name="experiment_add",
    description="Add a backtest run to the experiment registry (experiments.jsonl). "
    "Records experiment_id, strategy_id, spec_hash, run_id, metrics, audit_status, "
    "and created_at. Prevents selective memory in research.",
)
def experiment_add(run_dir: str, registry_path: str = "experiments.jsonl") -> dict[str, Any]:
    """Add a backtest run to the experiment registry."""
    from oxq.observe.experiment_registry import add_experiment

    entry = add_experiment(run_dir, registry_path=registry_path)
    if "error" in entry:
        return entry
    return {"status": "ok", "experiment_id": entry["experiment_id"], "strategy_id": entry["strategy_id"]}
