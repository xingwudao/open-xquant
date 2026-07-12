from __future__ import annotations

import json

import pytest
import yaml

from oxq.spec.schema import StrategySpec
from oxq.tools import registry


def test_report_write_tool_is_not_registered() -> None:
    names = {tool.name for tool in registry.all_tools()}

    assert "report_write" not in names
    assert "experiment_add" in names
    with pytest.raises(KeyError):
        registry.get("report_write")


def test_experiment_add_tool_forwards_configured_version_root(tmp_path) -> None:
    from oxq.tools.report import experiment_add

    version_root = tmp_path / "research_versions"
    run_dir = version_root / "v003/09_backtests/run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_root", "run_id": "run_1"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(run_dir, "custom_root")
    registry_path = tmp_path / "experiments.jsonl"

    result = experiment_add(
        str(run_dir),
        registry_path=str(registry_path),
        version_root=str(version_root),
    )

    assert result["status"] == "ok"
    entry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert entry["version_id"] == "v003"


def test_experiment_add_tool_forwards_manifest_backtest_phase(tmp_path) -> None:
    from oxq.tools.report import experiment_add

    backtest_phase_dir = tmp_path / "research_versions/v003/artifacts/backtests"
    run_dir = backtest_phase_dir / "run_1_cost_x2"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_phase", "run_id": "run_1_cost_x2"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(run_dir, "custom_phase")
    registry_path = tmp_path / "experiments.jsonl"

    result = experiment_add(
        str(run_dir),
        registry_path=str(registry_path),
        backtest_phase_dir=str(backtest_phase_dir),
        version_id="v003",
    )

    assert result["status"] == "ok"
    entry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert entry["version_id"] == "v003"
    assert entry["run_role"] == "robustness_cost_x2"


def _write_identity_artifacts(run_dir, strategy_id: str) -> None:
    spec = StrategySpec.template(strategy_id=strategy_id, hypothesis="experiment identity fixture")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")
