"""Audit tools — reproducibility and research bias audits."""

from __future__ import annotations

from typing import Any

from oxq.tools.registry import registry


@registry.tool(
    name="audit_reproducibility",
    description="Run reproducibility audit on a backtest run directory. "
    "Verifies spec hash, data manifest, artifact_hashes.json, and hashes for "
    "equity curve, trades, target_weights.csv, and metrics. "
    "Returns status and per-check results.",
)
def audit_reproducibility(run_dir: str) -> dict[str, Any]:
    """Verify reproducibility of a backtest run."""
    from oxq.audit.reproducibility import audit_reproducibility as _audit

    return _audit(run_dir)


@registry.tool(
    name="audit_research",
    description="Run research bias audit on a backtest run directory. "
    "Checks execution lag, cost model, OOS requirement, benchmark, survivorship bias, "
    "parameter count, trade count, drawdown severity, and data quality. "
    "Returns status and per-check results with fatal/warning counts.",
)
def audit_research(run_dir: str) -> dict[str, Any]:
    """Audit research bias of a backtest run."""
    from oxq.audit.research_bias import audit_research as _audit

    return _audit(run_dir)
