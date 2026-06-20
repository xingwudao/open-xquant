import json
from pathlib import Path

import pandas as pd
import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec


def _write_spec_and_data(tmp_path, *, evaluation_window: str = "full"):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 102, 104, 103, 106, 108],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC"),
    )
    frame.to_parquet(data_dir / "SPY.parquet")

    spec = StrategySpec.template(
        strategy_id="json_backtest",
        hypothesis="json backtest output supports agents",
    )
    spec.universe.symbols = ["SPY"]
    spec.universe.point_in_time = True
    spec.signal.indicators = {
        "roc_1": IndicatorDef(type="ROC", params={"column": "close", "period": 1})
    }
    spec.signal.rules = {
        "positive": SignalRuleDef(
            type="Threshold",
            params={"column": "roc_1", "threshold": 0, "relationship": "gt"},
        )
    }
    spec.validation.train_period = ["2024-01-02", "2024-01-04"]
    spec.validation.test_period = ["2024-01-05", "2024-01-09"]
    spec.benchmark.symbols = ["SPY"]
    spec.metrics.evaluation_window = evaluation_window
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    return spec_path, data_dir


def test_backtest_run_json_outputs_artifact_paths(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
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
    assert payload["metrics"]["trade_count"] >= 0
    assert payload["metrics"] == json.loads(Path(payload["artifacts"]["metrics_json"]).read_text(encoding="utf-8"))
    assert payload["warnings"] == []
    assert payload["errors"] == []
    assert set(payload["artifacts"]) >= {
        "strategy_spec_yaml",
        "environment_json",
        "data_manifest_json",
        "execution_assumptions_json",
        "equity_curve_csv",
        "trades_csv",
        "positions_csv",
        "orders_csv",
        "target_weights_csv",
        "benchmark_curve_csv",
        "metrics_json",
        "artifact_hashes_json",
        "run_log_jsonl",
    }
    assert payload["artifacts"]["target_weights_csv"].endswith("target_weights.csv")
    assert payload["artifacts"]["benchmark_curve_csv"].endswith("benchmark_curve.csv")
    assert payload["artifacts"]["artifact_hashes_json"].endswith("artifact_hashes.json")


def test_backtest_run_json_reports_validation_failure(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path)
    raw = spec_path.read_text(encoding="utf-8")
    spec_path.write_text(raw.replace("fee_rate: 0.001", "fee_rate: -0.001"), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["run_id"] == ""
    assert payload["run_dir"] == ""
    assert payload["errors"]
    assert payload["artifacts"] == {}


def test_backtest_run_json_reports_runtime_failure(tmp_path) -> None:
    spec_path, _data_dir = _write_spec_and_data(tmp_path)
    missing_data_dir = tmp_path / "missing_data"
    missing_data_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(missing_data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert payload["run_id"] == ""
    assert payload["run_dir"] == ""
    assert payload["artifacts"] == {}
    assert payload["metrics"] == {}
    assert payload["errors"][0]["check"] == "runtime_error"


def test_backtest_run_json_uses_artifact_metrics_for_oos_window(tmp_path) -> None:
    spec_path, data_dir = _write_spec_and_data(tmp_path, evaluation_window="oos")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    artifact_metrics = json.loads(Path(payload["artifacts"]["metrics_json"]).read_text(encoding="utf-8"))
    assert payload["metrics"] == artifact_metrics
    assert payload["metrics"]["metric_assumptions"]["evaluation_window"] == "oos"
