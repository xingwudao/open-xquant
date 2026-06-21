from __future__ import annotations

import json

import pytest
import yaml

from oxq.report.artifacts import RunArtifacts
from oxq.report.facts import build_report_facts
from oxq.spec.schema import StrategySpec


def test_run_artifacts_load_reads_named_files_and_data_frames(tmp_path) -> None:
    run_dir = _write_run_with_curves(tmp_path)
    (run_dir / "research_bias_audit.json").write_text(
        json.dumps({"status": "pass", "fatal_count": 0, "warning_count": 0}),
        encoding="utf-8",
    )

    artifacts = RunArtifacts.load(run_dir)

    assert artifacts.run_dir == run_dir
    assert artifacts.strategy_spec["strategy_id"] == "facts_case"
    assert artifacts.metrics["trade_count"] == 3
    assert artifacts.research_bias_audit == {"status": "pass", "fatal_count": 0, "warning_count": 0}
    assert list(artifacts.equity_curve["value"]) == [100.0, 110.0, 99.0, 120.0]
    assert list(artifacts.trades["side"]) == ["BUY", "SELL", "BUY"]


def test_report_facts_compute_dates_months_oos_trades_and_known_numbers(tmp_path) -> None:
    run_dir = _write_run_with_curves(tmp_path)

    facts = build_report_facts(RunArtifacts.load(run_dir))

    assert facts.configured_end_date == "2024-03-31"
    assert facts.effective_last_trading_day == "2024-03-29"
    assert facts.positive_month_count == 2
    assert facts.negative_month_count == 1
    assert facts.oos_trade_count == 2
    assert facts.monthly_returns == [
        {"month": "2024-01", "return": pytest.approx(0.10)},
        {"month": "2024-02", "return": pytest.approx(-0.10)},
        {"month": "2024-03", "return": pytest.approx(120.0 / 99.0 - 1.0)},
    ]
    assert any(item["name"] == "metric.sharpe_ratio" and item["value"] == 1.23 for item in facts.known_numbers)
    assert any(item["name"] == "spec.cost.fee_rate" and item["value"] == 0.001 for item in facts.known_numbers)
    assert any(item["name"] == "spec.cost.slippage_rate" and item["value"] == 0.0005 for item in facts.known_numbers)
    assert any(item["name"] == "spec.execution.initial_cash" and item["value"] == 100000.0 for item in facts.known_numbers)
    assert any(item["name"] == "fact.positive_month_count" and item["value"] == 2.0 for item in facts.known_numbers)


def _write_run_with_curves(tmp_path):
    spec = StrategySpec.template(strategy_id="facts_case", hypothesis="facts should be artifact-derived")
    spec.validation.train_period = ["2024-01-02", "2024-01-31"]
    spec.validation.test_period = ["2024-02-01", "2024-03-31"]
    spec.validation.required_oos = True
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.0005
    spec.execution.initial_cash = 100000
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": "facts-run", "trade_count": 3, "oos_trade_count": 2, "sharpe_ratio": 1.23}),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n"
        "2024-01-02,100\n"
        "2024-01-31,110\n"
        "2024-02-29,99\n"
        "2024-03-29,120\n",
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "symbol,side,shares,filled_price,filled_at,fee\n"
        "AAA,BUY,1,10,2024-01-15,0\n"
        "AAA,SELL,1,11,2024-02-15,0\n"
        "BBB,BUY,1,20,2024-03-01,0\n",
        encoding="utf-8",
    )
    return run_dir
