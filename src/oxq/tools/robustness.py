"""Robustness tools — stress-test backtest results."""

from __future__ import annotations

from typing import Any

from oxq.tools.registry import registry


@registry.tool(
    name="robustness_run",
    description="Run robustness tests on a backtest run directory. "
    "Checks cost stress, IS/OOS metric diff, parameter perturbation config, "
    "and regime analysis. Returns status (robust/warn/fragile/error), per-test "
    "results, writes robustness.json, and updates artifact_hashes.json.",
)
def robustness_run(run_dir: str) -> dict[str, Any]:
    """Run robustness tests on a backtest run."""
    from oxq.robustness import run_robustness

    return run_robustness(run_dir)
