from __future__ import annotations

import json

import yaml

from oxq.observe.experiment_registry import add_experiment
from oxq.spec.schema import StrategySpec


def test_add_experiment_runs_and_persists_research_audit(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"

    spec = StrategySpec.template(strategy_id="audit_registry", hypothesis="audit status is preserved")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "audit_registry", "run_id": "run_1", "trade_count": 12, "max_drawdown": -0.10}),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash(), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    audit_path = run_dir / "research_bias_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert entry["audit_status"] == audit["status"]
    assert entry["audit_status"] != "unknown"
