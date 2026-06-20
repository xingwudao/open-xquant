from __future__ import annotations

import json

import pandas as pd
import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.spec.schema import StrategySpec


def test_robustness_run_exits_nonzero_for_error(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run_robustness(_run_dir: str) -> dict:
        return {"status": "error", "tests": [], "message": "missing data"}

    monkeypatch.setattr("oxq.robustness.run_robustness", fake_run_robustness)

    result = CliRunner().invoke(main, ["robustness", "run", str(run_dir), "--json"])

    assert result.exit_code == 1
    assert "missing data" in result.output


def test_robustness_run_exits_nonzero_for_fragile(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run_robustness(_run_dir: str) -> dict:
        return {
            "status": "fragile",
            "tests": [{"name": "cost_x2", "status": "fail", "message": "sharpe collapsed"}],
            "baseline_sharpe": 1.0,
        }

    monkeypatch.setattr("oxq.robustness.run_robustness", fake_run_robustness)

    result = CliRunner().invoke(main, ["robustness", "run", str(run_dir)])

    assert result.exit_code == 1
    assert "Status: FRAGILE" in result.output


def test_spec_init_generates_path_safe_strategy_id(tmp_path) -> None:
    out = tmp_path / "strategy_spec.yaml"

    result = CliRunner().invoke(main, ["spec", "init", "SMA/RSI crossover!!!", "--out", str(out)])

    assert result.exit_code == 0, result.output
    spec = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert spec["strategy_id"] == "sma_rsi_crossover"


def test_backtest_run_json_outputs_machine_readable_artifact_manifest(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="json_backtest", hypothesis="json output is agent friendly")
    spec.market.calendar = "XNYS"
    spec.universe.symbols = ["SPY"]
    spec.benchmark.symbols = []
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-04"]
    spec.validation.required_oos = False

    spec_file = tmp_path / "strategy_spec.yaml"
    spec_file.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1000, 1000],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_file),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["run_id"]
    assert payload["run_dir"]
    assert set(payload["summary_metrics"]) >= {"total_return", "sharpe_ratio", "max_drawdown", "trade_count"}
    assert isinstance(payload["warnings"], list)
    assert payload["errors"] == []
    assert "target_weights.csv" in payload["artifacts"]
    assert payload["artifacts"]["target_weights.csv"].endswith("target_weights.csv")
