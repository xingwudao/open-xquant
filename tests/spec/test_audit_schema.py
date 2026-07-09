from __future__ import annotations

import json
from typing import Any

from oxq.core.component_catalog import _catalog_hash
from oxq.spec.audit_schema import validate_spec_audit
from oxq.spec.schema import StrategySpec


def _payload(field_audits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        "strategy_idea_brief": "versions/v001/01_brainstorm/strategy_idea_brief.json",
        "strategy_idea_audit": "versions/v001/02_idea_audit/strategy_idea_audit.json",
        "strategy_idea_brief_hash": "sha256:" + "5" * 16,
        "strategy_idea_audit_hash": "sha256:" + "6" * 16,
        "recipe_matches": [],
        "field_audits": field_audits,
        "component_audits": [],
        "unsupported_mappings": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }


def _confirmed(path: str, value: Any, evidence: str | None = None) -> dict[str, Any]:
    return {
        "field_path": path,
        "spec_value": value,
        "status": "confirmed",
        "material_category": "execution_assumption",
        "evidence": [evidence or f"User: {path} = {json.dumps(value, sort_keys=True, default=str)}"],
        "blocking": False,
    }


def test_strict_confirmed_coverage_accepts_direct_user_evidence() -> None:
    spec = {"execution": {"initial_cash": 100000}}
    payload = _payload(
        [
            _confirmed(
                "execution.initial_cash",
                100000,
                "User: run the strategy with initial cash 100000.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_pass_audit_rejects_non_empty_blocking_findings() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["blocking_findings"] = [
        {
            "message": "Previously blocked question was resolved by user confirmation.",
            "resolution": "confirmed",
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "blocking_findings" for error in result["errors"])


def test_pass_audit_rejects_unresolved_exception_lists() -> None:
    payload = _payload([_confirmed("execution.price_type", "open")])
    payload["agent_added_fields"] = [
        {
            "field_path": "execution.price_type",
            "message": "Agent expanded next_session_open.",
            "resolution": "confirmed",
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "agent_added_fields" for error in result["errors"])


def test_strict_confirmed_coverage_rejects_stale_default_duplicate_row() -> None:
    spec = {"execution": {"initial_cash": 100000}}
    payload = _payload(
        [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000,
                "status": "default",
                "material_category": "execution_assumption",
                "evidence": ["Default checklist row before user confirmation."],
                "blocking": False,
            },
            _confirmed("execution.initial_cash", 100000),
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("conflicting non-confirmed audit rows" in error["message"] for error in result["errors"])


def test_strict_confirmed_coverage_rejects_framework_default_evidence() -> None:
    spec = {"market": {"region": "us"}}
    payload = _payload(
        [
            _confirmed(
                "market.region",
                "us",
                "Framework default; calendar XSHG is configured explicitly.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("evidence denies user confirmation" in error["message"] for error in result["errors"])


def test_strict_confirmed_coverage_rejects_effective_default_coverage_evidence() -> None:
    spec = {"cost": {"buy_fee_rate": None}}
    payload = _payload(
        [
            _confirmed(
                "cost.buy_fee_rate",
                None,
                "Effective StrategySpec default value. Inherits from fee_rate when buy_fee_rate is absent from YAML. "
                "Documented for full SPEC coverage.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("evidence denies user confirmation" in error["message"] for error in result["errors"])


def test_strict_confirmed_coverage_accepts_user_confirmed_default_checklist() -> None:
    spec = {"metrics": {"risk_free_rate": 0.0}}
    payload = _payload(
        [
            _confirmed(
                "metrics.risk_free_rate",
                0.0,
                "User confirmed the Default Confirmation Checklist metrics group.",
            )
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_spec_hash_must_match_strategy_spec_compute_hash_when_spec_is_supplied() -> None:
    spec = StrategySpec.from_dict(
        {
            "schema_version": "0.1",
            "strategy_id": "hash_gate_test",
            "name": "Hash Gate Test",
            "universe": {"symbols": ["SPY"]},
            "benchmark": {"symbols": ["SPY"]},
            "execution": {"lot_size_config": {"default": 1}},
            "cost": {"fee_rate": 0.001, "slippage_rate": 0.001},
            "validation": {"test_period": ["2022-01-01", "2023-01-01"], "required_oos": False},
        }
    )
    payload = _payload([])
    payload["spec_hash"] = "sha256:" + "f" * 16

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_hash" and "strategy spec hash" in error["message"] for error in result["errors"])

    payload["spec_hash"] = spec.compute_hash()

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_catalog_hash_must_match_component_catalog_payload_when_catalog_is_supplied() -> None:
    catalog = {"components": [], "recipes": []}
    catalog["catalog_hash"] = _catalog_hash(catalog)
    payload = _payload([])
    payload["catalog_hash"] = "sha256:" + "9" * 16

    result = validate_spec_audit(payload, component_catalog=catalog)

    assert result["status"] == "fail"
    assert any(error["path"] == "catalog_hash" and "component catalog hash" in error["message"] for error in result["errors"])

    payload["catalog_hash"] = catalog["catalog_hash"]

    result = validate_spec_audit(payload, component_catalog=catalog)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_pass_audit_rejects_blocking_unsupported_mappings() -> None:
    payload = _payload([])
    payload["unsupported_mappings"] = [
        {
            "source_field": "portfolio.constraints.min_position_value",
            "requested_semantic": "minimum notional position size",
            "reason": "current strategy_spec.yaml parses the field but cannot execute it",
            "disposition": "blocked",
            "blocking": True,
        }
    ]

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "unsupported_mappings[0].blocking" for error in result["errors"])


def test_pass_audit_accepts_explicit_empty_unsupported_mappings() -> None:
    payload = _payload([])
    payload["unsupported_mappings"] = []

    result = validate_spec_audit(payload)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_blocked_audit_may_omit_spec_confirmation_table() -> None:
    payload = _payload([])
    payload["status"] = "block"
    payload["audit_conclusion"] = "blocked"
    payload["user_confirmation_status"] = "pending"
    payload["spec_provenance_pass"] = False
    payload["blocking_findings"] = [{"message": "execution.initial_cash mistranslated"}]
    payload.pop("spec_confirmation_table")

    result = validate_spec_audit(payload)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_all_pass_pending_audit_requires_spec_confirmation_table() -> None:
    payload = _payload([])
    payload["status"] = "block"
    payload["audit_conclusion"] = "all_pass"
    payload["user_confirmation_status"] = "pending"
    payload.pop("spec_confirmation_table")

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "spec_confirmation_table" for error in result["errors"])


def test_spec_audit_requires_idea_and_mapping_contract_fields() -> None:
    payload = _payload([])
    for field in (
        "strategy_idea_brief",
        "strategy_idea_audit",
        "strategy_idea_brief_hash",
        "strategy_idea_audit_hash",
        "unsupported_mappings",
    ):
        payload.pop(field)

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    paths = {error["path"] for error in result["errors"]}
    assert "strategy_idea_brief" in paths
    assert "strategy_idea_audit" in paths
    assert "strategy_idea_brief_hash" in paths
    assert "strategy_idea_audit_hash" in paths
    assert "unsupported_mappings" in paths


def test_field_audits_require_material_category() -> None:
    payload = _payload([_confirmed("execution.initial_cash", 100000)])
    payload["field_audits"][0].pop("material_category")

    result = validate_spec_audit(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_audits[0].material_category" for error in result["errors"])


def test_spec_supplied_rejects_yaml_only_field_audit_path() -> None:
    spec = {"execution": {"initial_cash": 100000.0}}
    payload = _payload(
        [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000.0,
                "status": "contradiction",
                "material_category": "execution_assumption",
                "evidence": ["Effective value differs from user-confirmed source value."],
                "blocking": True,
            },
            {
                "field_path": "portfolio.initial_cash",
                "spec_value": 1000000,
                "status": "contradiction",
                "material_category": "execution_assumption",
                "evidence": ["YAML-only misplaced source path; not effective StrategySpec semantics."],
                "blocking": True,
            },
        ]
    )
    payload["status"] = "block"
    payload["audit_conclusion"] = "blocked"
    payload["user_confirmation_status"] = "rejected"
    payload["spec_provenance_pass"] = False
    payload["spec_confirmation_table"] = None
    payload["blocking_findings"] = [{"message": "initial_cash is mapped to a non-operative YAML path"}]

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_audits[portfolio.initial_cash]" for error in result["errors"])
    assert any("effective StrategySpec field" in error["message"] for error in result["errors"])


def test_yaml_only_source_path_belongs_in_contradiction_not_field_audits() -> None:
    spec = {"execution": {"initial_cash": 100000.0}}
    payload = _payload(
        [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000.0,
                "status": "contradiction",
                "material_category": "execution_assumption",
                "evidence": ["User confirmed 1000000, but effective value remains default 100000.0."],
                "blocking": True,
            }
        ]
    )
    payload["status"] = "block"
    payload["audit_conclusion"] = "blocked"
    payload["user_confirmation_status"] = "rejected"
    payload["spec_provenance_pass"] = False
    payload["spec_confirmation_table"] = None
    payload["contradictions"] = [
        {
            "message": "initial_cash was placed under a non-operative YAML path",
            "effective_field_path": "execution.initial_cash",
            "source_yaml_path": "portfolio.initial_cash",
            "expected_value": 1000000,
            "effective_value": 100000.0,
            "builder_required_fix": "move the value to execution.initial_cash and remove portfolio.initial_cash",
        }
    ]
    payload["blocking_findings"] = [{"message": "return to build for initial_cash mapping fix"}]

    result = validate_spec_audit(payload, spec=spec)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_strict_confirmed_coverage_flattens_structured_lists_like_spec_fields() -> None:
    spec = {
        "signal": {
            "rules": {
                "foo": {
                    "params": {
                        "conditions": [
                            {"column": "ret_20", "threshold": 0},
                            {"column": "vol_20", "threshold": 0.2},
                        ]
                    }
                }
            }
        }
    }
    payload = _payload(
        [
            _confirmed("signal.rules.foo.params.conditions[0].column", "ret_20"),
            _confirmed("signal.rules.foo.params.conditions[0].threshold", 0),
            _confirmed("signal.rules.foo.params.conditions[1].column", "vol_20"),
            _confirmed("signal.rules.foo.params.conditions[1].threshold", 0.2),
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "pass"
    assert result["errors"] == []
