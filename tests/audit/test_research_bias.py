from __future__ import annotations

import json

import yaml

from oxq.audit.research_bias import audit_research
from oxq.spec.schema import StrategySpec


def test_research_audit_fails_when_metrics_json_is_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_metrics", hypothesis="metrics are required for audit")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "fail"
    assert any(check["id"] == "metrics_json" and check["severity"] == "fatal" for check in result["checks"])


def test_research_audit_fails_when_required_metrics_are_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="partial_metrics", hypothesis="key metrics are required for audit")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"trade_count": 12}), encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "fail"
    assert any(check["id"] == "metrics_required" and check["severity"] == "fatal" for check in result["checks"])
