from __future__ import annotations

import json

import yaml

from oxq.audit.research_bias import audit_research
from oxq.spec.schema import SignalRuleDef, StrategySpec


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


def test_research_audit_returns_fatal_when_spec_cannot_parse(tmp_path) -> None:
    (tmp_path / "strategy_spec.yaml").write_text(
        """
strategy_id: bad_spec
validation:
  test_period: 2024-01-01
""",
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "fail"
    assert any(check["id"] == "spec_parse_error" and check["severity"] == "fatal" for check in result["checks"])


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


def test_research_audit_fails_when_required_metrics_are_invalid(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="invalid_metrics", hypothesis="metric schema is required")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps({"trade_count": None, "max_drawdown": "bad"}), encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "fail"
    assert any(check["id"] == "metrics_schema" and check["severity"] == "fatal" for check in result["checks"])


def test_research_audit_fails_when_metrics_json_is_not_object(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="metrics_array", hypothesis="metric artifact schema is required")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(json.dumps([]), encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "fail"
    assert any(check["id"] == "metrics_schema" and check["severity"] == "fatal" for check in result["checks"])


def test_research_audit_fails_on_negative_costs(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="negative_costs", hypothesis="negative costs are invalid")
    spec.cost.fee_rate = -0.001
    spec.cost.slippage_rate = -0.001
    spec.cost.fee_min = -1.0
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"trade_count": 12, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "fail"
    assert any(check["id"] == "cost_model" and check["severity"] == "fatal" for check in result["checks"])


def test_research_audit_warns_on_zero_cost_with_explicit_replay_execution(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="zero_cost_explicit_replay", hypothesis="zero costs are replay-style")
    spec.cost.fee_rate = 0.0
    spec.cost.slippage_rate = 0.0
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"trade_count": 12, "oos_trade_count": 12, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "pass"
    cost_model = next(check for check in result["checks"] if check["id"] == "cost_model")
    assert cost_model["status"] == "fail"
    assert cost_model["severity"] == "warning"


def test_research_audit_surfaces_fatal_spec_validation_findings(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="peak_bias", hypothesis="future data signals fail audits")
    spec.signal.rules = {"peak": SignalRuleDef(type="Peak", params={"column": "close"})}
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"trade_count": 12, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    assert result["status"] == "fail"
    assert any(check["id"] == "spec_validation" and check["severity"] == "fatal" for check in result["checks"])


def test_research_audit_flags_quoted_false_point_in_time(tmp_path) -> None:
    (tmp_path / "strategy_spec.yaml").write_text(
        """
strategy_id: quoted_point_in_time
research:
  hypothesis: quoted booleans should not bypass audits
universe:
  type: static
  symbols: ["SPY"]
  point_in_time: "false"
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2024-01-01", "2024-12-31"]
""",
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"trade_count": 12, "oos_trade_count": 12, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    check = next(item for item in result["checks"] if item["id"] == "static_universe_survivorship")
    assert check["status"] == "fail"
    assert check["severity"] == "warning"


def test_research_audit_uses_oos_trade_count_when_available(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="oos_trade_count", hypothesis="OOS trades are statistically relevant")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"trade_count": 25, "oos_trade_count": 0, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    trade_count = next(check for check in result["checks"] if check["id"] == "trade_count")
    assert trade_count["status"] == "fail"
    assert "OOS" in trade_count["message"]


def test_research_audit_handles_invalid_missing_ratio(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_missing_ratio", hypothesis="missing ratio must be numeric")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"trade_count": 25, "oos_trade_count": 25, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": "bad"}),
        encoding="utf-8",
    )

    result = audit_research(tmp_path)

    missing_data = next(check for check in result["checks"] if check["id"] == "missing_data")
    assert missing_data["status"] == "fail"


def test_research_audit_handles_non_object_data_manifest(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="manifest_array", hypothesis="manifest artifact schema is required")
    (tmp_path / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"trade_count": 25, "oos_trade_count": 25, "max_drawdown": -0.1}),
        encoding="utf-8",
    )
    (tmp_path / "data_manifest.json").write_text(json.dumps([]), encoding="utf-8")

    result = audit_research(tmp_path)

    missing_data = next(check for check in result["checks"] if check["id"] == "missing_data")
    assert missing_data["status"] == "fail"
