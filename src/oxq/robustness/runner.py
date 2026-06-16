"""Robustness Runner — stress-test backtest results.

P0 tests: cost doubling, IS/OOS comparison, parameter perturbation,
and regime analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from oxq.spec.compiler import compile_run
from oxq.spec.schema import CostSection, StrategySpec


def run_robustness(run_dir: str | Path) -> dict:
    """Run P0 robustness tests on a backtest run.

    Reads the existing run's spec and metrics, re-runs with perturbed costs,
    and compares IS/OOS performance.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run directory.

    Returns
    -------
    dict
        Robustness result with 'status', 'tests', and summary.
    """
    run_path = Path(run_dir)
    tests: list[dict] = []

    # Load spec and baseline metrics
    spec_path = run_path / "strategy_spec.yaml"
    metrics_path = run_path / "metrics.json"
    if not spec_path.exists() or not metrics_path.exists():
        return {"status": "error", "tests": [], "message": "run directory missing spec or metrics"}

    spec = StrategySpec.from_yaml(str(spec_path))
    baseline_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0)

    # Preserve the effective data directory from the original run
    env_path = run_path / "environment.json"
    data_dir = None
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
        data_dir = env.get("data_dir")

    # --- Test 1: Cost x2 ---
    try:
        cost_x2_dir = run_path.parent / f"{run_path.name}_cost_x2"
        cost_spec = _clone_spec_with_cost_multiplier(spec, 2.0)
        cost_result, _ = compile_run(cost_spec, out_dir=str(cost_x2_dir), data_dir=data_dir)
        perturbed_sharpe = cost_result.sharpe_ratio()
        tests.append({
            "name": "cost_x2",
            "baseline_sharpe": round(baseline_sharpe, 4),
            "perturbed_sharpe": round(perturbed_sharpe, 4),
            "status": "fail" if perturbed_sharpe < 0 else ("warn" if perturbed_sharpe < baseline_sharpe * 0.5 else "pass"),
            "message": f"Sharpe drops from {baseline_sharpe:.2f} to {perturbed_sharpe:.2f} with 2x costs",
        })
    except Exception as e:
        tests.append({"name": "cost_x2", "status": "error", "message": str(e)})

    # --- Test 2: IS/OOS comparison ---
    train = spec.validation.train_period
    test = spec.validation.test_period
    if train and test and len(train) >= 2 and len(test) >= 2:
        tests.append({
            "name": "is_oos_comparison",
            "status": "warn",
            "message": f"IS: {train[0]} to {train[1]}, OOS: {test[0]} to {test[1]} — "
                       "IS/OOS metrics comparison not yet implemented",
        })
    else:
        tests.append({
            "name": "is_oos_comparison",
            "status": "warn",
            "message": "Train/test periods not fully specified — cannot compare IS/OOS",
        })

    # --- Test 3: Parameter perturbation — check sensitivity hints from spec ---
    perturbations = spec.robustness.parameter_perturbation
    if perturbations:
        tests.append({
            "name": "parameter_perturbation",
            "status": "warn",
            "message": f"Perturbation targets configured: {list(perturbations.keys())} — "
                       "re-running with perturbed parameters not yet implemented",
        })
    else:
        tests.append({
            "name": "parameter_perturbation",
            "status": "warn",
            "message": "No parameter perturbation targets configured in spec",
        })

    # --- Test 4: Regime analysis ---
    if spec.robustness.regime_analysis:
        tests.append({"name": "regime_analysis", "status": "pass", "message": "Regime analysis requested in spec"})
    else:
        tests.append({"name": "regime_analysis", "status": "warn", "message": "Regime analysis not configured"})

    # Summary
    failed = [t for t in tests if t["status"] == "fail"]
    warned = [t for t in tests if t["status"] == "warn"]
    errors = [t for t in tests if t["status"] == "error"]

    if errors:
        status = "error"
    elif failed:
        status = "fragile"
    elif warned:
        status = "warn"
    else:
        status = "robust"

    return {"status": status, "tests": tests, "baseline_sharpe": baseline_sharpe}


def _clone_spec_with_cost_multiplier(spec: StrategySpec, multiplier: float) -> StrategySpec:
    """Create a copy of spec with costs multiplied."""
    return StrategySpec(
        schema_version=spec.schema_version,
        strategy_id=f"{spec.strategy_id}_cost_x{int(multiplier)}",
        name=spec.name + f" (cost x{int(multiplier)})",
        required_oxq_version=spec.required_oxq_version,
        research=spec.research,
        market=spec.market,
        universe=spec.universe,
        data=spec.data,
        signal=spec.signal,
        portfolio=spec.portfolio,
        execution=spec.execution,
        cost=CostSection(
            fee_rate=spec.cost.fee_rate * multiplier,
            fee_min=spec.cost.fee_min,
            slippage_rate=spec.cost.slippage_rate * multiplier,
        ),
        benchmark=spec.benchmark,
        validation=spec.validation,
        robustness=spec.robustness,
        decision_policy=spec.decision_policy,
    )
