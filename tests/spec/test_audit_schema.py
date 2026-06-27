from __future__ import annotations

import json
from typing import Any

from oxq.spec.audit_schema import validate_spec_audit


def _payload(field_audits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        "recipe_matches": [],
        "field_audits": field_audits,
        "component_audits": [],
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


def test_strict_confirmed_coverage_rejects_stale_default_duplicate_row() -> None:
    spec = {"execution": {"initial_cash": 100000}}
    payload = _payload(
        [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000,
                "status": "default",
                "evidence": ["Default checklist row before user confirmation."],
                "blocking": False,
            },
            _confirmed("execution.initial_cash", 100000),
        ]
    )

    result = validate_spec_audit(payload, spec=spec, require_confirmed_coverage=True)

    assert result["status"] == "fail"
    assert any("conflicting non-confirmed audit rows" in error["message"] for error in result["errors"])


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
