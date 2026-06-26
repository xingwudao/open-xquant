"""Robustness Runner — stress-test backtest results.

P0 tests: cost doubling, IS/OOS comparison, parameter perturbation,
and regime analysis.
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from oxq.portfolio.metrics_profile import compute_equity_curve_metrics
from oxq.spec.compiler import _append_run_digest, _hash_file, _hash_json_file, compile_run
from oxq.spec.schema import CostSection, StrategySpec

_BENCHMARK_CURVE_FILES = ("benchmark_curve.csv", "benchmark_equity_curve.csv", "benchmark_prices.csv")
_COMPONENT_PROVENANCE_FILES = ("component_manifest.json", "component_manifests.json", "component_bundle_hash.txt")
_COMPONENT_EXTENSION_ARCHIVE_DIR = "component_extensions"


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
    from oxq.core.component_manifest import scoped_component_registries

    with scoped_component_registries():
        return _run_robustness_scoped(run_dir)


def _run_robustness_scoped(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    tests: list[dict] = []
    try:
        from oxq.core.component_manifest import load_component_manifests_from_run

        load_component_manifests_from_run(run_path)
    except Exception as exc:
        return {"status": "error", "tests": [], "message": f"run component manifests could not be loaded: {exc}"}

    # Load spec and baseline metrics
    spec_path = run_path / "strategy_spec.yaml"
    metrics_path = run_path / "metrics.json"
    if not spec_path.exists() or not metrics_path.exists():
        return {"status": "error", "tests": [], "message": "run directory missing spec or metrics"}

    spec = StrategySpec.from_yaml(str(spec_path))
    baseline_metrics, error = _read_json_object(metrics_path, "metrics.json")
    if error is not None:
        return {"status": "error", "tests": [], "message": error}
    baseline_sharpe = _finite_float(baseline_metrics.get("sharpe_ratio"))

    # Preserve the effective data directory from the original run
    env_path = run_path / "environment.json"
    data_dir = None
    if env_path.exists():
        env, error = _read_json_object(env_path, "environment.json")
        if error is not None:
            return {"status": "error", "tests": [], "message": error}
        data_dir = env.get("data_dir")

    # --- Test 1: Cost x2 ---
    try:
        cost_x2_dir = run_path.parent / f"{run_path.name}_cost_x2"
        cost_spec = _clone_spec_with_cost_multiplier(spec, 2.0)
        _, cost_run_dir = compile_run(cost_spec, out_dir=str(cost_x2_dir), data_dir=data_dir)
        _copy_component_provenance(run_path, Path(cost_run_dir))
        perturbed_sharpe = _read_metric_sharpe(Path(cost_run_dir) / "metrics.json")
        if baseline_sharpe is None or perturbed_sharpe is None:
            tests.append({
                "name": "cost_x2",
                "baseline_sharpe": baseline_sharpe,
                "perturbed_sharpe": perturbed_sharpe,
                "status": "warn",
                "message": "Sharpe unavailable for cost x2 robustness comparison",
            })
        else:
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
    tests.append(_compare_is_oos(spec, baseline_metrics))

    # --- Test 3: Parameter perturbation — check sensitivity hints from spec ---
    tests.append(_run_parameter_perturbations(spec, run_path, data_dir, baseline_sharpe))

    # --- Test 4: Regime analysis ---
    tests.append(_analyze_regimes(spec, run_path))

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

    result = {"status": status, "tests": tests, "baseline_sharpe": baseline_sharpe}
    _write_robustness_artifact(run_path, result)
    return result


def _read_json_object(path: Path, name: str) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {}, f"{name} is invalid JSON: {exc}"
    if not isinstance(value, dict):
        return {}, f"{name} must be a JSON object"
    return value, None


def _read_metric_sharpe(path: Path) -> float | None:
    metrics, error = _read_json_object(path, path.name)
    if error is not None:
        return None
    return _finite_float(metrics.get("sharpe_ratio"))


def _compare_is_oos(spec: StrategySpec, metrics: dict[str, Any]) -> dict[str, Any]:
    train = spec.validation.train_period
    test = spec.validation.test_period
    if not (train and test and len(train) >= 2 and len(test) >= 2):
        return {
            "name": "is_oos_comparison",
            "status": "warn",
            "message": "Train/test periods not fully specified — cannot compare IS/OOS",
        }

    is_metrics = _split_metrics(metrics, "is")
    oos_metrics = _split_metrics(metrics, "oos")
    missing = [
        f"{prefix}_{name}"
        for prefix, values in (("is", is_metrics), ("oos", oos_metrics))
        for name, value in values.items()
        if value is None
    ]
    policy_breaches = _reject_policy_breaches(spec, oos_metrics)
    if missing:
        oos_sharpe = oos_metrics["sharpe_ratio"]
        negative_oos_sharpe = oos_sharpe is not None and oos_sharpe < 0
        status = "fail" if policy_breaches or negative_oos_sharpe else "warn"
        message = f"IS/OOS metrics unavailable or non-finite: {missing}"
        if policy_breaches:
            message += f"; breached reject policy: {policy_breaches}"
        if negative_oos_sharpe:
            message += "; negative OOS Sharpe"
        return {
            "name": "is_oos_comparison",
            "status": status,
            "is_period": train,
            "oos_period": test,
            "is": is_metrics,
            "oos": oos_metrics,
            "message": message,
        }

    degradation = {
        "total_return": _higher_is_better_degradation(is_metrics["total_return"], oos_metrics["total_return"]),
        "sharpe_ratio": _higher_is_better_degradation(is_metrics["sharpe_ratio"], oos_metrics["sharpe_ratio"]),
        "calmar_ratio": _higher_is_better_degradation(is_metrics["calmar_ratio"], oos_metrics["calmar_ratio"]),
        "max_drawdown": _drawdown_degradation(is_metrics["max_drawdown"], oos_metrics["max_drawdown"]),
    }
    oos_sharpe = oos_metrics["sharpe_ratio"]
    material_degradations = {
        name: value
        for name, value in degradation.items()
        if value is not None and value > 0.5
    }
    if policy_breaches:
        status = "fail"
    elif oos_sharpe is not None and oos_sharpe < 0:
        status = "fail"
    elif material_degradations:
        status = "warn"
    else:
        status = "pass"
    message = f"Compared IS {train[0]} to {train[1]} with OOS {test[0]} to {test[1]}"
    if policy_breaches:
        message += f"; breached reject policy: {policy_breaches}"
    return {
        "name": "is_oos_comparison",
        "status": status,
        "is_period": train,
        "oos_period": test,
        "is": is_metrics,
        "oos": oos_metrics,
        "degradation": degradation,
        "message": message,
    }


def _reject_policy_breaches(spec: StrategySpec, oos_metrics: dict[str, float | None]) -> list[str]:
    reject_if = spec.decision_policy.reject_if
    breaches: list[str] = []
    if "oos_sharpe_lt" in reject_if:
        threshold = _finite_float(reject_if["oos_sharpe_lt"])
        oos_sharpe = oos_metrics["sharpe_ratio"]
        if threshold is not None and oos_sharpe is not None and threshold > oos_sharpe:
            breaches.append("oos_sharpe_lt")
    if "max_drawdown_lt" in reject_if:
        threshold = _finite_float(reject_if["max_drawdown_lt"])
        max_drawdown = oos_metrics["max_drawdown"]
        if threshold is not None and max_drawdown is not None and threshold > max_drawdown:
            breaches.append("max_drawdown_lt")
    return breaches


def _split_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float | None]:
    return {
        "total_return": _finite_float(metrics.get(f"{prefix}_total_return")),
        "sharpe_ratio": _finite_float(metrics.get(f"{prefix}_sharpe_ratio")),
        "max_drawdown": _finite_float(metrics.get(f"{prefix}_max_drawdown")),
        "calmar_ratio": _finite_float(metrics.get(f"{prefix}_calmar_ratio")),
    }


def _higher_is_better_degradation(in_sample: float | None, out_of_sample: float | None) -> float | None:
    if in_sample is None or out_of_sample is None or in_sample == 0:
        return None
    return round((in_sample - out_of_sample) / abs(in_sample), 6)


def _drawdown_degradation(in_sample: float | None, out_of_sample: float | None) -> float | None:
    if in_sample is None or out_of_sample is None:
        return None
    if in_sample == 0:
        return 1.0 if out_of_sample != 0 else 0.0
    return round((abs(out_of_sample) - abs(in_sample)) / abs(in_sample), 6)


def _run_parameter_perturbations(
    spec: StrategySpec,
    run_path: Path,
    data_dir: str | None,
    baseline_sharpe: float | None,
) -> dict[str, Any]:
    perturbations = spec.robustness.parameter_perturbation
    if not perturbations:
        return {
            "name": "parameter_perturbation",
            "status": "warn",
            "message": "No parameter perturbation targets configured in spec",
        }

    results: list[dict[str, Any]] = []
    for target, values in perturbations.items():
        if not isinstance(values, list):
            results.append({
                "target": target,
                "status": "error",
                "message": f"robustness.parameter_perturbation.{target} must be a list of values",
            })
            continue
        if not values:
            results.append({
                "target": target,
                "status": "warn",
                "message": f"robustness.parameter_perturbation.{target} must include at least one value",
            })
            continue
        for value in values:
            if _is_non_finite_number(value):
                results.append({
                    "target": target,
                    "value": str(value),
                    "status": "error",
                    "message": f"robustness.parameter_perturbation.{target} values must be finite",
                })
                continue
            perturbed_spec = copy.deepcopy(spec)
            try:
                _apply_perturbation(perturbed_spec, target, value)
                value_slug = _slugify(str(value))
                target_slug = _slugify(target)
                perturbed_spec.strategy_id = f"{spec.strategy_id}_perturb_{target_slug}_{value_slug}"
                out_dir = run_path.parent / f"{run_path.name}_perturb_{target_slug}_{value_slug}"
                _, perturbed_run_dir = compile_run(perturbed_spec, out_dir=str(out_dir), data_dir=data_dir)
                _copy_component_provenance(run_path, Path(perturbed_run_dir))
                perturbed_sharpe = _read_metric_sharpe(Path(perturbed_run_dir) / "metrics.json")
                results.append(_perturbation_result(target, value, baseline_sharpe, perturbed_sharpe, perturbed_run_dir))
            except Exception as exc:
                results.append({
                    "target": target,
                    "value": value,
                    "status": "error",
                    "message": str(exc),
                })

    statuses = {item["status"] for item in results}
    if statuses == {"error"}:
        status = "error"
    elif "fail" in statuses:
        status = "fail"
    elif "warn" in statuses or "error" in statuses:
        status = "warn"
    elif not results:
        status = "warn"
    else:
        status = "pass"
    message = (
        "No parameter perturbation reruns were executed"
        if not results
        else f"Ran {len(results)} one-at-a-time parameter perturbations"
    )
    return {
        "name": "parameter_perturbation",
        "status": status,
        "results": results,
        "message": message,
    }


def _copy_component_provenance(source_run: Path, child_run: Path) -> None:
    copied: list[str] = []
    for filename in _COMPONENT_PROVENANCE_FILES:
        source = source_run / filename
        if not source.exists():
            continue
        target = child_run / filename
        shutil.copy2(source, target)
        copied.append(filename)
    source_extensions = source_run / _COMPONENT_EXTENSION_ARCHIVE_DIR
    if source_extensions.exists():
        target_extensions = child_run / _COMPONENT_EXTENSION_ARCHIVE_DIR
        shutil.copytree(
            source_extensions,
            target_extensions,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.pyc", "*.pyo"),
        )
    _copy_legacy_component_roots(source_run, child_run)
    if not copied:
        return

    hashes_path = child_run / "artifact_hashes.json"
    hashes: dict[str, Any] = {}
    if hashes_path.exists():
        hashes, error = _read_json_object(hashes_path, "artifact_hashes.json")
        if error is not None:
            hashes = {}
    for filename in copied:
        path = child_run / filename
        hashes[filename] = _hash_json_file(path) if path.suffix == ".json" else _hash_file(path)
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_run_digest(child_run, _hash_json_file(hashes_path))


def _copy_legacy_component_roots(source_run: Path, child_run: Path) -> None:
    """Copy run-local legacy component roots referenced by component_manifest.json."""

    manifest_path = source_run / "component_manifest.json"
    summary_path = source_run / "component_manifests.json"
    if not manifest_path.exists() or not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(summary, list) or len(summary) != 1 or not isinstance(summary[0], dict):
        return
    if summary[0].get("archived_manifest_path"):
        return
    raw_root = manifest.get("extension_root") or manifest.get("extension_id")
    if not isinstance(raw_root, str) or not raw_root:
        return
    root_path = Path(raw_root)
    if root_path.is_absolute() or ".." in root_path.parts:
        return
    source_root = (source_run / root_path).resolve()
    source_run_resolved = source_run.resolve()
    if not source_root.is_dir() or not source_root.is_relative_to(source_run_resolved):
        return
    target_root = child_run / root_path
    shutil.copytree(
        source_root,
        target_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.pyc", "*.pyo"),
    )


def _perturbation_result(
    target: str,
    value: float | int,
    baseline_sharpe: float | None,
    perturbed_sharpe: float | None,
    run_dir: str | Path,
) -> dict[str, Any]:
    if baseline_sharpe is None or perturbed_sharpe is None:
        status = "warn"
        message = "Sharpe unavailable for parameter perturbation comparison"
    elif perturbed_sharpe < 0:
        status = "fail"
        message = f"Sharpe is negative after perturbing {target} to {value}"
    elif perturbed_sharpe < baseline_sharpe * 0.5:
        status = "warn"
        message = f"Sharpe drops from {baseline_sharpe:.2f} to {perturbed_sharpe:.2f}"
    else:
        status = "pass"
        message = f"Sharpe remains {perturbed_sharpe:.2f} after perturbing {target}"
    return {
        "target": target,
        "value": value,
        "status": status,
        "baseline_sharpe": _round_metric(baseline_sharpe),
        "perturbed_sharpe": _round_metric(perturbed_sharpe),
        "run_dir": str(run_dir),
        "message": message,
    }


def _apply_perturbation(spec: StrategySpec, target: str, value: float | int) -> None:
    parts = target.split(".")
    if len(parts) == 2 and parts[0] in spec.signal.indicators:
        indicator = spec.signal.indicators[parts[0]]
        if parts[1] not in indicator.params:
            raise ValueError(f"Perturbation target '{target}' not found")
        indicator.params[parts[1]] = value
        return

    if parts[:2] == ["signal", "params"] and len(parts) == 3:
        if len(spec.signal.rules) != 1:
            raise ValueError("signal.params shorthand requires exactly one signal rule")
        rule = next(iter(spec.signal.rules.values()))
        if parts[2] not in rule.params:
            raise ValueError(f"Perturbation target '{target}' not found")
        rule.params[parts[2]] = value
        return

    if len(parts) == 5 and parts[:2] == ["signal", "indicators"] and parts[3] == "params":
        indicator = spec.signal.indicators.get(parts[2])
        if indicator is None or parts[4] not in indicator.params:
            raise ValueError(f"Perturbation target '{target}' not found")
        indicator.params[parts[4]] = value
        return

    if len(parts) == 5 and parts[:2] == ["signal", "rules"] and parts[3] == "params":
        rule = spec.signal.rules.get(parts[2])
        if rule is None or parts[4] not in rule.params:
            raise ValueError(f"Perturbation target '{target}' not found")
        rule.params[parts[4]] = value
        return

    if len(parts) == 3 and parts[:2] == ["portfolio", "params"]:
        if parts[2] not in spec.portfolio.params:
            raise ValueError(f"Perturbation target '{target}' not found")
        spec.portfolio.params[parts[2]] = value
        return

    raise ValueError(f"Unsupported perturbation target '{target}'")


def _analyze_regimes(spec: StrategySpec, run_path: Path) -> dict[str, Any]:
    if not spec.robustness.regime_analysis:
        return {"name": "regime_analysis", "status": "warn", "message": "Regime analysis not configured"}

    equity_path = run_path / "equity_curve.csv"
    if not equity_path.exists():
        return {
            "name": "regime_analysis",
            "status": "warn",
            "message": "equity_curve.csv not found — cannot compute regime metrics",
        }

    equity, error = _read_curve_csv(equity_path, "equity_curve.csv")
    if error is not None:
        return {
            "name": "regime_analysis",
            "status": "warn",
            "message": error,
        }

    if len(equity) < 2:
        return {
            "name": "regime_analysis",
            "status": "warn",
            "message": "equity_curve.csv needs at least two valid rows for regime metrics",
        }

    strategy_returns = equity["value"].pct_change().fillna(0.0)
    reference, regime_source = _regime_reference_curve(spec, run_path, equity)
    regime_returns = reference["value"].pct_change().fillna(0.0)
    abs_returns = regime_returns.abs()
    median_abs_return = float(abs_returns.median())
    masks = {
        "uptrend": regime_returns > 0,
        "downtrend": regime_returns <= 0,
        "high_vol": abs_returns >= median_abs_return,
        "low_vol": abs_returns < median_abs_return,
    }
    trade_counts = _read_trade_date_counts(run_path / "trades.csv")
    regimes = {
        name: _regime_bucket(equity, strategy_returns, mask, spec, trade_counts)
        for name, mask in masks.items()
    }
    return {
        "name": "regime_analysis",
        "status": "pass",
        "regime_source": regime_source,
        "regimes": regimes,
        "message": f"Computed realized regime metrics using {regime_source} segmentation",
    }


def _read_curve_csv(path: Path, name: str) -> tuple[pd.DataFrame, str | None]:
    try:
        curve = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), f"{name} could not be read: {exc}"
    required_columns = {"date", "value"}
    if not required_columns.issubset(curve.columns):
        return pd.DataFrame(), f"{name} must contain date and value columns"

    curve = curve.loc[:, ["date", "value"]].copy()
    curve["date"] = _parse_local_dates(curve["date"])
    curve["value"] = pd.to_numeric(curve["value"], errors="coerce")
    curve = curve.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return curve, None


def _regime_reference_curve(spec: StrategySpec, run_path: Path, equity: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if spec.benchmark.symbols:
        for filename in _BENCHMARK_CURVE_FILES:
            benchmark_path = run_path / filename
            if not benchmark_path.exists():
                continue
            if not _artifact_hash_matches(run_path, filename):
                continue
            benchmark, error = _read_curve_csv(benchmark_path, filename)
            if error is not None or len(benchmark) < 2:
                continue
            aligned = _align_curve_to_equity_dates(equity, benchmark)
            if aligned is not None:
                return aligned, "benchmark"
    return equity, "strategy_equity"


def _align_curve_to_equity_dates(equity: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame | None:
    aligned = equity.loc[:, ["date"]].merge(reference, on="date", how="left")
    aligned["value"] = aligned["value"].ffill()
    aligned = aligned.dropna(subset=["value"]).reset_index(drop=True)
    if len(aligned) != len(equity) or aligned["value"].nunique(dropna=True) < 2:
        return None
    return aligned


def _regime_bucket(
    equity: pd.DataFrame,
    returns: pd.Series,
    mask: pd.Series,
    spec: StrategySpec,
    trade_counts: dict[Any, int],
) -> dict[str, Any]:
    segment = equity.loc[mask, ["date", "value"]]
    dates = set(segment["date"].tolist())
    selected_returns = returns.loc[mask]
    curve = _equity_curve_from_returns(selected_returns)
    metrics = compute_equity_curve_metrics(curve, spec.metrics)
    return {
        "date_count": int(len(segment)),
        "trade_count": int(sum(trade_counts.get(date, 0) for date in dates)),
        **{key: _round_metric(_finite_float(value)) for key, value in metrics.items()},
    }


def _equity_curve_from_returns(returns: pd.Series) -> list[tuple[int, float]]:
    values = [1.0]
    for value in returns.tolist():
        parsed = _finite_float(value)
        if parsed is None:
            continue
        values.append(values[-1] * (1.0 + parsed))
    return list(enumerate(values))


def _read_trade_date_counts(path: Path) -> dict[Any, int]:
    if not path.exists():
        return {}
    try:
        trades = pd.read_csv(path)
    except Exception:
        return {}
    if "filled_at" not in trades.columns:
        return {}
    dates = _parse_local_dates(trades["filled_at"]).dropna()
    counts = dates.value_counts()
    return {date: int(count) for date, count in counts.items()}


def _parse_local_dates(values: pd.Series) -> pd.Series:
    parsed = values.map(lambda value: pd.to_datetime(value, errors="coerce"))
    return parsed.map(lambda value: value.date() if not pd.isna(value) else pd.NaT)


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_non_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return not math.isfinite(float(value))


def _artifact_hash_matches(run_path: Path, filename: str) -> bool:
    artifact_hashes_path = run_path / "artifact_hashes.json"
    if not artifact_hashes_path.exists():
        return False
    try:
        artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(artifact_hashes, dict):
        return False
    expected = artifact_hashes.get(filename)
    if not isinstance(expected, str):
        return False
    try:
        actual = _hash_file(run_path / filename)
    except OSError:
        return False
    return actual == expected


def _round_metric(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-")
    slug = slug.replace(".", "_")
    return slug or "value"


def _write_robustness_artifact(run_path: Path, result: dict[str, Any]) -> None:
    artifact_path = run_path / "robustness.json"
    artifact_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_hashes_path = run_path / "artifact_hashes.json"
    if not artifact_hashes_path.exists():
        return
    try:
        artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(artifact_hashes, dict):
        return
    artifact_hashes["robustness.json"] = _hash_file(artifact_path)
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8")
    _append_run_digest(run_path, _hash_json_file(artifact_hashes_path))


def _clone_spec_with_cost_multiplier(spec: StrategySpec, multiplier: float) -> StrategySpec:
    """Create a copy of spec with costs multiplied."""
    cloned = copy.deepcopy(spec)
    cloned.strategy_id = f"{spec.strategy_id}_cost_x{int(multiplier)}"
    cloned.name = spec.name + f" (cost x{int(multiplier)})"
    cloned.cost = CostSection(
        fee_rate=spec.cost.fee_rate * multiplier,
        fee_min=spec.cost.fee_min * multiplier,
        slippage_rate=spec.cost.slippage_rate * multiplier,
    )
    return cloned
