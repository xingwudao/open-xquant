from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd
import pytest

from oxq.audit.reproducibility import audit_reproducibility
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _build_optimizer, _write_artifacts, compile_run, compile_strategy
from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec


def test_artifact_spec_hash_matches_serialized_spec(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="hash_test", hypothesis="hash artifacts are reproducible")
    spec.execution.initial_cash = 100_000

    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"


def test_missing_ratio_ignores_derived_indicator_nans(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_ratio", hypothesis="derived warmup nans are not raw data missing")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                    "sma_50": [None, None, 1.0],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_counts_absent_required_columns(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_required", hypothesis="absent required columns are missing data")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0


def test_compile_run_rejects_strategy_id_path_traversal(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="../outside", hypothesis="strategy ids cannot be paths")

    with pytest.raises(ValueError, match="strategy_id"):
        compile_run(spec, out_dir=tmp_path)


def test_compile_run_rejects_multiple_crossover_rules(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="multi_cross", hypothesis="each crossover has an exit")
    spec.signal.indicators = {
        "fast_a": IndicatorDef(type="SMA", params={"period": 2}),
        "slow_a": IndicatorDef(type="SMA", params={"period": 3}),
        "fast_b": IndicatorDef(type="SMA", params={"period": 4}),
        "slow_b": IndicatorDef(type="SMA", params={"period": 5}),
    }
    spec.signal.rules = {
        "cross_a": SignalRuleDef(type="Crossover", params={"fast": "fast_a", "slow": "slow_a"}),
        "cross_b": SignalRuleDef(type="Crossover", params={"fast": "fast_b", "slow": "slow_b"}),
    }

    with pytest.raises(ValueError, match="Multiple Crossover"):
        compile_run(spec, out_dir=tmp_path)


def test_metrics_artifact_hash_excludes_run_id(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="metrics_hash", hypothesis="run metadata is not metrics fingerprint")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    first = tmp_path / "run_1"
    second = tmp_path / "run_2"
    first.mkdir()
    second.mkdir()
    _write_artifacts(spec, result, first, Engine())
    _write_artifacts(spec, result, second, Engine())

    first_hashes = json.loads((first / "artifact_hashes.json").read_text(encoding="utf-8"))
    second_hashes = json.loads((second / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert first_hashes["metrics.json"] == second_hashes["metrics.json"]


def test_reproducibility_audit_fails_when_artifact_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="tamper_test", hypothesis="artifact tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "metrics.json").write_text('{"strategy_id": "tampered"}\n', encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "metrics_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_data_manifest_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="manifest_tamper", hypothesis="manifest tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["price_adjustment"] = "raw"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_handles_corrupt_metrics_json(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="corrupt_metrics", hypothesis="corrupt metrics fail audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "metrics.json").write_text("{not-json", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "metrics_hash" and check["status"] == "fail" for check in audit["checks"])


def test_data_manifest_uses_test_period_start_when_train_period_absent(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="test_only", hypothesis="test-only specs record actual range")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["start"] == "2024-01-01"
    assert manifest["end"] == "2024-01-03"


def test_data_manifest_start_includes_min_start_date(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="warmup_start", hypothesis="manifest includes warmup data range")
    spec.data.min_start_date = "2023-12-01"
    dates = pd.bdate_range("2023-12-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["start"] == "2023-12-01"


def test_reproducibility_audit_handles_corrupt_artifact_hashes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="corrupt_hashes", hypothesis="corrupt hash manifest fails cleanly")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "artifact_hashes.json").write_text("{not-json", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_crossover_latch_can_be_reset_after_exit() -> None:
    spec = StrategySpec.template(strategy_id="cross_reset", hypothesis="crossover exits clear active entry state")
    spec.signal.indicators = {
        "fast": IndicatorDef(type="SMA", params={"period": 2}),
        "slow": IndicatorDef(type="SMA", params={"period": 3}),
    }
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
    }

    optimizer = _build_optimizer(spec)
    entry_bar = pd.DataFrame({"cross": [True]})
    inactive_bar = pd.DataFrame({"cross": [False]})

    assert optimizer.optimize({"SPY": entry_bar}, {"SPY": entry_bar}) == {"SPY": 1.0}

    optimizer.reset_symbols(["SPY"])

    assert optimizer.optimize({"SPY": inactive_bar}, {"SPY": inactive_bar}) == {"CASH": 1.0}


def test_compile_strategy_rejects_unsupported_universe_type() -> None:
    spec = StrategySpec.template(strategy_id="unsupported_universe", hypothesis="unsupported universes fail clearly")
    spec.universe.type = "filter"

    with pytest.raises(ValueError, match="Unsupported universe.type 'filter'"):
        compile_strategy(spec)
