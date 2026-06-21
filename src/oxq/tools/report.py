"""Report tools — generate research reports from backtest artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oxq.tools.registry import registry


@registry.tool(
    name="report_write",
    description="Generate research_report.md and research_report.html from a backtest run directory. "
    "Reads strategy_spec.yaml, metrics.json, runs both audits, and produces a structured "
    "report with executive decision (REJECT/WATCHLIST/PAPER TRADING CANDIDATE). "
    "Returns report paths and executive decision.",
)
def report_write(
    run_dir: str,
    out: str | None = None,
    lang: str = "zh",
    output_format: str = "all",
) -> dict[str, Any]:
    """Generate a research report from a backtest run."""
    from oxq.report import generate_report, write_report_files

    outputs = write_report_files(run_dir, lang=lang, output_format=output_format, out=out)
    report_md = outputs.markdown.read_text(encoding="utf-8") if outputs.markdown else generate_report(run_dir, lang=lang)

    strategy_id = ""
    for line in report_md.split("\n"):
        if line.startswith("## 1. Executive Decision"):
            break
        if line.startswith("# Research Report: "):
            strategy_id = line.replace("# Research Report: ", "").strip()
        if line.startswith("# 研究报告: "):
            strategy_id = line.replace("# 研究报告: ", "").strip()

    decision = ""
    for line in report_md.split("\n"):
        if line.startswith("**") and ("REJECT" in line or "WATCHLIST" in line or "CANDIDATE" in line):
            decision = line.strip("*").strip()
            break

    legacy_output = outputs.markdown or outputs.html or Path("")
    return {
        "status": "ok",
        "output": str(legacy_output),
        "markdown_output": str(outputs.markdown) if outputs.markdown else None,
        "html_output": str(outputs.html) if outputs.html else None,
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
