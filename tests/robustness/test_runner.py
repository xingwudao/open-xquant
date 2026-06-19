from __future__ import annotations

import json
from pathlib import Path

import yaml

from oxq.portfolio.analytics import RunResult
from oxq.robustness.runner import _clone_spec_with_cost_multiplier, run_robustness
from oxq.spec.schema import StrategySpec


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
    assert "not yet implemented" in regime["message"]


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
