"""Report tools — generate research reports from backtest artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oxq.tools.registry import registry


@registry.tool(
    name="report_write",
    description="Generate a research_report.md from a backtest run directory. "
    "Reads strategy_spec.yaml, metrics.json, runs both audits, and produces a structured "
    "report with executive decision (REJECT/WATCHLIST/PAPER TRADING CANDIDATE). "
    "Returns the report path and executive decision.",
)
def report_write(run_dir: str, out: str | None = None) -> dict[str, Any]:
    """Generate a research report from a backtest run."""
    from oxq.report import generate_report

    report_md = generate_report(run_dir)
    output_path = Path(out) if out else Path(run_dir) / "research_report.md"
    output_path.write_text(report_md, encoding="utf-8")

    strategy_id = ""
    for line in report_md.split("\n"):
        if line.startswith("## 1. Executive Decision"):
            break
        if line.startswith("# Research Report: "):
            strategy_id = line.replace("# Research Report: ", "").strip()

    decision = ""
    for line in report_md.split("\n"):
        if line.startswith("**") and ("REJECT" in line or "WATCHLIST" in line or "CANDIDATE" in line):
            decision = line.strip("*").strip()
            break

    return {
        "status": "ok",
        "output": str(output_path),
        "strategy_id": strategy_id,
        "decision": decision,
    }


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
