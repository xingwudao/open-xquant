from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
GOVERNANCE = ROOT / "docs/strategy-workflow-artifact-governance.md"
GUIDE = ROOT / "docs/agent-guide.md"
COMPARISON_SKILL = ROOT / "agent/skills/compare-strategy-versions/SKILL.md"
COMPARISON_ROLE = ROOT / "agent/roles/oxq-experiment-comparator-worker.md"
COORDINATOR = ROOT / "agent/roles/oxq-coordinator.md"
ROUTER = ROOT / "agent/skills/open-xquant/SKILL.md"
SELECTOR = ROOT / "agent/skills/select-final-version/SKILL.md"
CHARTS = ROOT / "agent/skills/build-report-charts/SKILL.md"
WRITER = ROOT / "agent/skills/write-research-report/SKILL.md"
REVIEWER = ROOT / "agent/skills/review-research-report/SKILL.md"
LINEAGE = ROOT / "agent/skills/audit-artifact-lineage/SKILL.md"

ALL = (GOVERNANCE, GUIDE, COMPARISON_SKILL, COMPARISON_ROLE, COORDINATOR, ROUTER, SELECTOR, CHARTS, WRITER, REVIEWER, LINEAGE)


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalized(path: Path) -> str:
    return " ".join(text(path).lower().split())


def json_examples(path: Path) -> list[object]:
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", text(path), re.DOTALL)]


def object_example(path: Path, **fields: object) -> dict[str, object]:
    for item in json_examples(path):
        if isinstance(item, dict) and all(item.get(key) == value for key, value in fields.items()):
            return item
    raise AssertionError((path, fields))


def test_current_schema_registry_accepts_only_round_26_versions() -> None:
    for path in (GOVERNANCE, GUIDE, COORDINATOR, ROUTER, SELECTOR):
        value = normalized(path)
        assert "schema `5`" in value or "schema 5" in value, path
        assert "schema `3`" in value, path
        for name in ("candidate", "policy", "comparison", "lineage"):
            assert name in value, (path, name)

    decision = object_example(SELECTOR, schema_version=5, status="selected")
    assert "report_revision" in decision and "selection_request_id" in decision
    pointers = [item for item in json_examples(SELECTOR) if isinstance(item, dict) and item.get("final_decision")]
    assert pointers


def test_historical_refresh_restarts_selection_only_after_new_comparison_inputs() -> None:
    for path in (GOVERNANCE, COORDINATOR, ROUTER, WRITER, REVIEWER, LINEAGE, COMPARISON_SKILL, COMPARISON_ROLE):
        value = normalized(path)
        assert "write -> review -> lineage -> prepare new selection -> comparison -> resume" in value, path
        assert "fresh selection" in value, path
        assert "candidate" in value and "hash" in value, path
        assert "fresh" in value and "comparison" in value, path
        assert ("old selection" in value or "prior selection" in value) and "must not" in value, path


def test_confirmation_event_is_closed_and_binds_user_and_coordinator_provenance() -> None:
    required = ("schema_version", "phase", "timestamp", "confirmed_by", "producer", "coordinator", "raw_line_hash")
    for path in (GOVERNANCE, COORDINATOR):
        value = normalized(path)
        assert "closed" in value and "confirmation" in value, path
        for field in required:
            assert f"`{field}`" in value, (path, field)
        assert "missing" in value and "mismatched" in value, path
        assert "producer" in value and "coordinator" in value, path


def test_schema_three_comparison_request_is_exact_and_shared() -> None:
    request = object_example(GOVERNANCE, schema_version=3, mode="build_selection_comparison")
    assert set(request) == {
        "schema_version", "mode", "selection_id", "selection_request_id",
        "selection_policy", "candidate_set", "comparison_population",
    }
    assert request["selection_id"] == "selection_20260712_190000"
    assert request["selection_request_id"] == "selection-request-20260712-1"
    for path in (COMPARISON_SKILL, COMPARISON_ROLE, COORDINATOR, ROUTER):
        value = normalized(path)
        assert "schema `3`" in value and "build_selection_comparison" in value, path
        for field in ("selection_id", "selection_request_id", "selection_policy", "candidate_set", "comparison_population"):
            assert field in value, (path, field)


def test_chart_retry_cannot_seal_blocked_revision_before_fresh_chart_attempt() -> None:
    for path in (CHARTS, WRITER, REVIEWER, GOVERNANCE):
        value = normalized(path)
        assert "unsealed" in value and "revision" in value, path
        assert "retry" in value and "chart" in value, path
        assert "seal" in value and "completion" in value, path
    for path in (CHARTS, WRITER, GOVERNANCE):
        assert "fresh `report_revision_id`" in normalized(path), path


def test_default_chart_pack_uses_canonical_requested_set_and_closed_skips() -> None:
    canonical = object_example(CHARTS, schema_version=1, chart_decision="default_professional_chart_pack")["requested"]
    assert isinstance(canonical, list) and canonical
    for path in (CHARTS, WRITER, GOVERNANCE):
        value = normalized(path)
        assert "canonical requested set" in value, path
        assert "requested equality" in value or "cannot be narrowed" in value, path
        assert "omitted charts" in value or "omitted chart" in value, path
        assert "closed skip reason" in value or "closed skip reasons" in value, path
    result = object_example(CHARTS, schema_version=1, status="complete")
    assert result["requested"] == canonical
    generated = {item["id"] for item in result["generated"]}
    skipped = {item["id"] for item in result["skipped"]}
    assert generated | skipped == set(canonical)


def test_round_26_contracts_fail_closed_when_schema_four_pointer_is_supplied() -> None:
    pointer = {"schema_version": 4, "selection_id": "S1", "final_decision": {}}
    assert pointer["schema_version"] != 5
    with pytest.raises(AssertionError):
        assert pointer["schema_version"] == 5
