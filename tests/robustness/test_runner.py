from __future__ import annotations

import json
from pathlib import Path

import yaml

from oxq.portfolio.analytics import RunResult
from oxq.robustness.runner import _clone_spec_with_cost_multiplier, run_robustness
from oxq.spec.schema import IndicatorDef, StrategySpec


def test_regime_analysis_request_is_not_reported_as_pass_when_unimplemented(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="regime_requested", hypothesis="regime analysis needs real metrics")
    spec.robustness.regime_analysis = True
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 1.0}), encoding="utf-8")

    result = run_robustness(tmp_path)

    regime = next(test for test in result["tests"] if test["name"] == "regime_analysis")
    assert regime["status"] == "warn"
    assert "equity_curve.csv" in regime["message"]


def test_cost_multiplier_multiplies_minimum_fee() -> None:
    spec = StrategySpec.template(strategy_id="cost_multiplier", hypothesis="minimum fees are part of costs")
    spec.cost.fee_rate = 0.001
    spec.cost.fee_min = 2.5
    spec.cost.slippage_rate = 0.001
    spec.metrics.profile = "xquant_production"
    spec.metrics.return_type = "log"
    spec.metrics.risk_free_rate = 0.02

    cloned = _clone_spec_with_cost_multiplier(spec, 2.0)

    assert cloned.cost.fee_rate == 0.002
    assert cloned.cost.fee_min == 5.0
    assert cloned.cost.slippage_rate == 0.002
    assert cloned.metrics.profile == "xquant_production"
    assert cloned.metrics.return_type == "log"
    assert cloned.metrics.risk_free_rate == 0.02


def test_run_robustness_handles_unavailable_baseline_sharpe(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="none_sharpe", hypothesis="missing sharpe should not crash robustness")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": None}), encoding="utf-8")

    class FakeResult:
        def sharpe_ratio(self) -> float:
            return 0.0

    def fake_compile_run(*args, **kwargs) -> tuple[RunResult, object]:
        return FakeResult(), tmp_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    cost_test = next(test for test in result["tests"] if test["name"] == "cost_x2")
    assert cost_test["status"] == "warn"
    assert cost_test["baseline_sharpe"] is None


def test_run_robustness_compares_perturbed_metrics_artifact(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="profile_sharpe", hypothesis="robustness should compare artifact sharpe")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 1.2}), encoding="utf-8")

    class FakeResult:
        def sharpe_ratio(self) -> float:
            return 99.0

    def fake_compile_run(_spec, *, out_dir: str, data_dir=None):
        del data_dir
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 0.7}), encoding="utf-8")
        return FakeResult(), out_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    cost_test = next(test for test in result["tests"] if test["name"] == "cost_x2")
    assert cost_test["baseline_sharpe"] == 1.2
    assert cost_test["perturbed_sharpe"] == 0.7


def test_run_robustness_does_not_fallback_to_legacy_sharpe(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_perturbed_sharpe", hypothesis="robustness uses artifact semantics")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 1.2}), encoding="utf-8")

    class FakeResult:
        def sharpe_ratio(self) -> float:
            return 99.0

    def fake_compile_run(_spec, *, out_dir: str, data_dir=None):
        del data_dir
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": None}), encoding="utf-8")
        return FakeResult(), out_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    cost_test = next(test for test in result["tests"] if test["name"] == "cost_x2")
    assert cost_test["status"] == "warn"
    assert cost_test["perturbed_sharpe"] is None


def test_run_robustness_returns_error_for_corrupt_metrics(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="corrupt_metrics", hypothesis="corrupt metrics should not crash")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text("{not-json", encoding="utf-8")

    result = run_robustness(tmp_path)

    assert result["status"] == "error"
    assert "metrics.json is invalid JSON" in result["message"]


def test_run_robustness_returns_error_for_non_object_metrics(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="list_metrics", hypothesis="metrics schema should be an object")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text("[]", encoding="utf-8")

    result = run_robustness(tmp_path)

    assert result["status"] == "error"
    assert result["message"] == "metrics.json must be a JSON object"


def test_run_robustness_returns_error_for_non_object_environment(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="list_env", hypothesis="environment schema should be an object")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 1.0}), encoding="utf-8")
    (tmp_path / "environment.json").write_text("[]", encoding="utf-8")

    result = run_robustness(tmp_path)

    assert result["status"] == "error"
    assert result["message"] == "environment.json must be a JSON object"


def test_run_robustness_writes_robustness_json(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="write_robustness", hypothesis="robustness should leave an artifact")
    _write_run_inputs(tmp_path, spec, {"sharpe_ratio": 1.0})

    def fake_compile_run(_spec, *, out_dir: str, data_dir=None):
        del data_dir
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 0.9}), encoding="utf-8")
        return object(), out_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    artifact = tmp_path / "robustness.json"
    assert artifact.exists()
    assert json.loads(artifact.read_text(encoding="utf-8")) == result


def test_is_oos_comparison_uses_metrics_json_numeric_values(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="is_oos_values", hypothesis="robustness should compare split metrics")
    _write_run_inputs(
        tmp_path,
        spec,
        {
            "sharpe_ratio": 1.0,
            "is_total_return": 0.5,
            "is_sharpe_ratio": 2.0,
            "is_max_drawdown": -0.1,
            "is_calmar_ratio": 5.0,
            "oos_total_return": 0.25,
            "oos_sharpe_ratio": 1.0,
            "oos_max_drawdown": -0.2,
            "oos_calmar_ratio": 1.25,
        },
    )

    def fake_compile_run(_spec, *, out_dir: str, data_dir=None):
        del data_dir
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 0.9}), encoding="utf-8")
        return object(), out_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    comparison = next(test for test in result["tests"] if test["name"] == "is_oos_comparison")
    assert comparison["is"]["sharpe_ratio"] == 2.0
    assert comparison["oos"]["sharpe_ratio"] == 1.0
    assert comparison["degradation"]["sharpe_ratio"] == 0.5
    assert "not yet implemented" not in comparison["message"]


def test_parameter_perturbation_reruns_one_at_a_time(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="perturb_once", hypothesis="robustness should perturb independently")
    spec.signal.indicators["mom"] = IndicatorDef(type="Momentum", params={"period": 10})
    spec.signal.indicators["vol"] = IndicatorDef(type="Volatility", params={"period": 20})
    spec.robustness.parameter_perturbation = {
        "mom.period": [9, 11],
        "vol.period": [18, 22],
    }
    _write_run_inputs(tmp_path, spec, {"sharpe_ratio": 1.0})
    seen: list[tuple[int, int]] = []

    def fake_compile_run(_spec, *, out_dir: str, data_dir=None):
        del data_dir
        seen.append((_spec.signal.indicators["mom"].params["period"], _spec.signal.indicators["vol"].params["period"]))
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 0.9}), encoding="utf-8")
        return object(), out_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    perturbation = next(test for test in result["tests"] if test["name"] == "parameter_perturbation")
    assert len(perturbation["results"]) == 4
    assert seen[1:] == [(9, 20), (11, 20), (10, 18), (10, 22)]


def test_invalid_perturbation_path_reports_target_error_only(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="perturb_errors", hypothesis="invalid perturbation paths should not stop valid ones")
    spec.signal.indicators["mom"] = IndicatorDef(type="Momentum", params={"period": 10})
    spec.robustness.parameter_perturbation = {
        "mom.period": [20],
        "missing.period": [30],
    }
    _write_run_inputs(tmp_path, spec, {"sharpe_ratio": 1.0})
    compile_calls = 0

    def fake_compile_run(_spec, *, out_dir: str, data_dir=None):
        nonlocal compile_calls
        del data_dir
        compile_calls += 1
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 0.8}), encoding="utf-8")
        return object(), out_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    perturbation = next(test for test in result["tests"] if test["name"] == "parameter_perturbation")
    assert compile_calls == 2
    assert any(item["target"] == "mom.period" and item["status"] in {"pass", "warn", "fail"} for item in perturbation["results"])
    assert any(item["target"] == "missing.period" and item["status"] == "error" for item in perturbation["results"])


def test_regime_analysis_returns_all_buckets(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="regimes", hypothesis="regime analysis should segment realized behavior")
    spec.robustness.regime_analysis = True
    _write_run_inputs(tmp_path, spec, {"sharpe_ratio": 1.0})
    _write_equity_curve(
        tmp_path,
        [
            ("2022-01-01", 100),
            ("2022-01-02", 104),
            ("2022-01-03", 101),
            ("2022-01-04", 108),
            ("2022-01-05", 102),
            ("2022-01-06", 109),
            ("2022-01-07", 99),
            ("2022-01-08", 100),
        ],
    )
    _write_trades(tmp_path, [("2022-01-02", "SPY"), ("2022-01-07", "SPY")])

    def fake_compile_run(_spec, *, out_dir: str, data_dir=None):
        del data_dir
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "metrics.json").write_text(json.dumps({"sharpe_ratio": 0.9}), encoding="utf-8")
        return object(), out_path

    monkeypatch.setattr("oxq.robustness.runner.compile_run", fake_compile_run)

    result = run_robustness(tmp_path)

    regimes = next(test for test in result["tests"] if test["name"] == "regime_analysis")["regimes"]
    assert set(regimes) == {"uptrend", "downtrend", "high_vol", "low_vol"}
    assert all("date_count" in bucket for bucket in regimes.values())
    assert sum(bucket["trade_count"] for bucket in regimes.values()) >= 2


def _write_run_inputs(tmp_path: Path, spec: StrategySpec, metrics: dict) -> None:
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    _write_equity_curve(tmp_path, [("2022-01-01", 100), ("2022-01-02", 101), ("2022-01-03", 102)])
    _write_trades(tmp_path, [])


def _write_equity_curve(tmp_path: Path, rows: list[tuple[str, float]]) -> None:
    lines = ["date,value"]
    lines.extend(f"{date},{value}" for date, value in rows)
    (tmp_path / "equity_curve.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_trades(tmp_path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["symbol,side,shares,filled_price,filled_at,fee"]
    lines.extend(f"{symbol},buy,1,100,{filled_at},1" for filled_at, symbol in rows)
    (tmp_path / "trades.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
