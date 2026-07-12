from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

CHART_SKILL = Path("agent/skills/build-report-charts/SKILL.md")
WRITER_SKILL = Path("agent/skills/write-research-report/SKILL.md")
WRITER_ROLE = Path("agent/roles/oxq-report-writer-worker.md")
REVIEWER_SKILL = Path("agent/skills/review-research-report/SKILL.md")
REVIEWER_ROLE = Path("agent/roles/oxq-report-reviewer-worker.md")
LINEAGE_SKILL = Path("agent/skills/audit-artifact-lineage/SKILL.md")
LINEAGE_ROLE = Path("agent/roles/oxq-lineage-auditor-worker.md")
COMPARATOR_SKILL = Path("agent/skills/compare-strategy-versions/SKILL.md")
COMPARATOR_ROLE = Path("agent/roles/oxq-experiment-comparator-worker.md")
SELECTOR_SKILL = Path("agent/skills/select-final-version/SKILL.md")
SELECTOR_ROLE = Path("agent/roles/oxq-final-selector-worker.md")
VERSION_SKILL = Path("agent/skills/manage-strategy-version/SKILL.md")
VERSION_ROLE = Path("agent/roles/oxq-version-manager-worker.md")
GOVERNOR_SKILL = Path("agent/skills/govern-research-workspace/SKILL.md")
GOVERNOR_ROLE = Path("agent/roles/oxq-artifact-governor-worker.md")
COORDINATOR_ROLE = Path("agent/roles/oxq-coordinator.md")
ROUTER_SKILL = Path("agent/skills/open-xquant/SKILL.md")
GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
AGENT_GUIDE = Path("docs/agent-guide.md")

REVISION_SCHEMA_CONTRACTS = (REVIEWER_SKILL, REVIEWER_ROLE, GOVERNANCE_DOC)
REVISION_SELECTION_CONTRACTS = (SELECTOR_SKILL, SELECTOR_ROLE, GOVERNANCE_DOC)
REVISION_CONSUMERS = (
    LINEAGE_SKILL,
    LINEAGE_ROLE,
    COMPARATOR_SKILL,
    COMPARATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COORDINATOR_ROLE,
    GOVERNANCE_DOC,
)
GOVERNANCE_PUBLISHERS = (
    VERSION_SKILL,
    VERSION_ROLE,
    GOVERNOR_SKILL,
    GOVERNOR_ROLE,
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    GOVERNANCE_DOC,
    AGENT_GUIDE,
)
POINTER_CONTRACTS = (
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    GOVERNANCE_DOC,
    AGENT_GUIDE,
)
POLICY_CONTRACTS = (
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    GOVERNANCE_DOC,
)
HISTORICAL_ROUTING_CONTRACTS = (
    CHART_SKILL,
    WRITER_SKILL,
    WRITER_ROLE,
    REVIEWER_SKILL,
    REVIEWER_ROLE,
    LINEAGE_SKILL,
    LINEAGE_ROLE,
    COMPARATOR_SKILL,
    COMPARATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    VERSION_SKILL,
    VERSION_ROLE,
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    GOVERNANCE_DOC,
)
CHART_HANDOFF_CONSUMERS = (
    WRITER_SKILL,
    WRITER_ROLE,
    REVIEWER_SKILL,
    REVIEWER_ROLE,
    LINEAGE_SKILL,
    LINEAGE_ROLE,
    COMPARATOR_SKILL,
    COMPARATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    GOVERNANCE_DOC,
)
COMPARATOR_RESULT_CONTRACTS = (COMPARATOR_SKILL, COMPARATOR_ROLE, GOVERNANCE_DOC)
COMPARATOR_ROUTERS = (COORDINATOR_ROLE, ROUTER_SKILL, SELECTOR_SKILL, GOVERNANCE_DOC)
FINAL_OUTPUT_CONTRACTS = (
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    LINEAGE_SKILL,
    LINEAGE_ROLE,
    GOVERNANCE_DOC,
    AGENT_GUIDE,
)

FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
RETRY_BLOCKER_CODES = {
    "comparison_id_collision",
    "comparison_build_failed",
    "comparison_publication_failed",
}
RESTART_BLOCKER_CODES = {
    "stale_confirmation_event",
    "stale_selection_policy",
    "stale_candidate_set",
    "stale_candidate_evidence",
    "stale_report_revision",
    "stale_review_revision",
    "stale_lineage_audit",
    "selection_binding_mismatch",
}
SKIP_REASON_CODES = {
    "missing_optional_input",
    "empty_optional_input",
    "structurally_insufficient_input",
    "not_applicable_to_strategy",
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).lower().split())


def _json_examples(path: Path) -> list[Any]:
    return [
        json.loads(block)
        for block in re.findall(r"```json\n(.*?)\n```", _text(path), flags=re.DOTALL)
    ]


def _json_object(path: Path, **fields: object) -> dict[str, Any]:
    payload = next(
        (
            item
            for item in _json_examples(path)
            if isinstance(item, dict)
            and all(item.get(key) == value for key, value in fields.items())
        ),
        None,
    )
    assert payload is not None, (path, fields)
    return payload


def _assert_ref(reference: object, suffix: str) -> None:
    assert isinstance(reference, dict)
    assert set(reference) == {"path", "sha256"}
    assert str(reference["path"]).endswith(suffix)
    assert FULL_SHA256.fullmatch(str(reference["sha256"]))


def test_report_and_review_revisions_are_immutable_and_selection_bound() -> None:
    for path in REVISION_SCHEMA_CONTRACTS:
        review = _json_object(path, schema_version=2, review_revision_id="review_20260712_181500")
        assert set(review) >= {
            "version_id",
            "run_id",
            "review_revision_id",
            "report_revision",
            "reviewed_artifacts",
            "decision_inputs",
        }
        _assert_ref(review["report_revision"], "/candidate_manifest.json")

    for path in REVISION_SELECTION_CONTRACTS:
        candidate_set = next(
            item
            for item in _json_examples(path)
            if isinstance(item, dict)
            and item.get("schema_version") == 3
            and item.get("selection_id") == "selection_20260712_190000"
            and "candidates" in item
        )
        candidates = candidate_set["candidates"]
        assert candidates
        for candidate in candidates:
            assert set(candidate) == {
                "ordinal",
                "identity",
                "primary_run",
                "report_revision",
                "report_review",
                "lineage_audit",
            }
            _assert_ref(candidate["report_revision"], "/candidate_manifest.json")
            _assert_ref(candidate["report_review"], "/report_review.json")

    for path in REVISION_CONSUMERS:
        normalized = _normalized(path)
        for phrase in (
            "immutable report revision",
            "immutable review revision",
            "exact `{path, sha256}` report-revision reference",
            "exact `{path, sha256}` review-revision reference",
            "evidence reachable from any prior selection",
            "never overwrite",
        ):
            assert phrase in normalized, (path, phrase)


def test_governance_publication_is_one_recoverable_locked_transaction() -> None:
    for path in GOVERNANCE_PUBLISHERS:
        normalized = _normalized(path)
        for phrase in (
            "`<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json`",
            "`workspace-governance.lock`",
            "`final-selection.lock` last",
            "prepared -> committing -> committed",
            "durable backup",
            "unchanged-byte",
            "before the first replacement",
            "after a non-pointer replacement",
            "after `current.json` replacement",
            "roll forward",
            "roll back",
        ):
            assert phrase in normalized, (path, phrase)


def test_final_pointer_fsyncs_final_dir_and_recovers_post_rename_sync_failure() -> None:
    prohibited = (
        "atomic `current_final.json` replacement the final fallible publication operation",
        "no validation, hashing, cleanup, directory sync",
        "successful replacement 后不得再执行可能把成功转换成 failure",
    )
    for path in POINTER_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "`fsync(<final_dir>)` after atomic replacement",
            "post-rename directory-sync failure",
            "publication outcome is indeterminate",
            "must not claim that the prior pointer is unchanged",
            "recover under `final-selection.lock`",
            "exact new pointer bytes",
            "exact prior pointer bytes",
            "any other bytes",
        ):
            assert phrase in normalized, (path, phrase)
        for phrase in prohibited:
            assert phrase not in normalized, (path, phrase)


def _event_hash(event: dict[str, Any]) -> str:
    raw = json.dumps(event, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _validate_confirmation(
    lines: list[dict[str, Any]],
    reference: dict[str, Any],
    *,
    selection_request_id: str,
    policy_hash: str,
) -> None:
    assert reference["path"].endswith("/confirmations.jsonl")
    line_number = reference["line_number"]
    assert isinstance(line_number, int) and 1 <= line_number <= len(lines)
    event = lines[line_number - 1]
    assert _event_hash(event) == reference["event_hash"]
    assert event["event_id"] == reference["event_id"]
    assert event["decision"] == reference["decision"] == "confirmed"
    assert event["selection_request_id"] == reference["selection_request_id"]
    assert event["selection_request_id"] == selection_request_id
    assert event["policy_hash"] == reference["policy_hash"] == policy_hash


def test_selection_policy_uses_coordinator_confirmation_journal_not_self_attestation() -> None:
    for path in POLICY_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "coordinator is the sole producer",
            "append-only `confirmations.jsonl`",
            "`confirmations.jsonl.lock`",
            "exact raw jsonl line bytes",
            "fabricated",
            "stale",
            "mismatched",
            "caller self-attestation is invalid",
        ):
            assert phrase in normalized, (path, phrase)
        assert "confirmed_selection_policy.json" not in _text(path), path

        request = _json_object(path, schema_version=3, mode="prepare_selection")
        policy = request["selection_policy"]
        assert set(policy) == {"payload", "policy_hash", "confirmation_event"}
        assert "confirmed_by_user" not in policy["payload"]
        event_ref = policy["confirmation_event"]
        assert set(event_ref) == {
            "path",
            "event_id",
            "line_number",
            "event_hash",
            "decision",
            "selection_request_id",
            "policy_hash",
        }
        assert FULL_SHA256.fullmatch(event_ref["event_hash"])
        assert FULL_SHA256.fullmatch(event_ref["policy_hash"])

    policy_hash = f"sha256:{'a' * 64}"
    event = {
        "schema_version": 1,
        "event_id": "selection-policy-confirmation-1",
        "timestamp": "2026-07-12T18:00:00Z",
        "phase": "final_selection_policy",
        "selection_request_id": "selection-request-1",
        "decision": "confirmed",
        "policy_hash": policy_hash,
        "confirmed_by": "user",
    }
    reference = {
        "path": "conversations/conv_1/confirmations.jsonl",
        "event_id": event["event_id"],
        "line_number": 1,
        "event_hash": _event_hash(event),
        "decision": "confirmed",
        "selection_request_id": event["selection_request_id"],
        "policy_hash": policy_hash,
    }
    _validate_confirmation(
        [event],
        reference,
        selection_request_id="selection-request-1",
        policy_hash=policy_hash,
    )

    fabricated = {**reference, "event_hash": f"sha256:{'f' * 64}"}
    with pytest.raises(AssertionError):
        _validate_confirmation(
            [event],
            fabricated,
            selection_request_id="selection-request-1",
            policy_hash=policy_hash,
        )
    with pytest.raises(AssertionError):
        _validate_confirmation(
            [event],
            reference,
            selection_request_id="selection-request-2",
            policy_hash=policy_hash,
        )
    with pytest.raises(AssertionError):
        _validate_confirmation(
            [event],
            reference,
            selection_request_id="selection-request-1",
            policy_hash=f"sha256:{'b' * 64}",
        )


def _validate_script_bindings(manifest: dict[str, Any], files: dict[str, bytes]) -> None:
    for asset in manifest["assets"]:
        source = asset["source"]
        script_path = source["script"]
        assert script_path.startswith("scripts/") and ".." not in Path(script_path).parts
        digest = f"sha256:{hashlib.sha256(files[script_path]).hexdigest()}"
        assert source["script_sha256"] == digest


def test_report_manifest_binds_source_script_path_and_hash_and_rejects_mutation() -> None:
    manifest = _json_object(CHART_SKILL, schema_version=2)
    scripts: dict[str, bytes] = {}
    for asset in manifest["assets"]:
        source = asset["source"]
        assert set(source) == {"script", "script_sha256", "input_artifacts"}
        assert FULL_SHA256.fullmatch(source["script_sha256"])
        scripts.setdefault(source["script"], b"plot-v1")
        source["script_sha256"] = f"sha256:{hashlib.sha256(b'plot-v1').hexdigest()}"
    _validate_script_bindings(manifest, scripts)

    scripts[next(iter(scripts))] = b"plot-v2"
    with pytest.raises(AssertionError):
        _validate_script_bindings(manifest, scripts)

    for path in (CHART_SKILL, *CHART_HANDOFF_CONSUMERS):
        normalized = _normalized(path)
        for phrase in (
            "safe package-relative `source.script`",
            "full lowercase `source.script_sha256`",
            "recompute the script sha-256",
            "script mutation",
        ):
            assert phrase in normalized, (path, phrase)


def test_inactive_candidate_revision_workflow_never_reactivates_or_overwrites() -> None:
    for path in HISTORICAL_ROUTING_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "`candidate_scoped_historical_report_revision`",
            "inactive version",
            "current-state guard",
            "fresh `report_revision_id`",
            "fresh `review_revision_id`",
            "must not reactivate",
            "must not overwrite",
        ):
            assert phrase in normalized, (path, phrase)

    for path in (COORDINATOR_ROLE, ROUTER_SKILL, GOVERNANCE_DOC):
        normalized = _normalized(path)
        for phrase in (
            "write -> review -> lineage -> comparison -> reselection",
            "fresh lineage audit",
            "fresh `comparison_id`",
            "`restart_selection`",
            "prior revision bytes remain reachable",
        ):
            assert phrase in normalized, (path, phrase)


def _route_comparator_result(result: dict[str, Any]) -> str:
    status = result["status"]
    action = result["next_action"]
    codes = set(result["blocker_codes"])
    if status == "comparison_ready":
        assert action == "resume_selection" and not codes
        return action
    assert status in {"blocked", "fail"} and codes
    if action == "retry_with_fresh_comparison_id":
        assert codes <= RETRY_BLOCKER_CODES
        return action
    assert action == "restart_selection" and codes <= RESTART_BLOCKER_CODES
    return action


def test_comparator_results_have_closed_next_action_and_blocker_mapping() -> None:
    for path in COMPARATOR_RESULT_CONTRACTS:
        ready = _json_object(path, schema_version=3, status="comparison_ready")
        retry = _json_object(path, schema_version=3, status="blocked")
        restart = _json_object(path, schema_version=3, status="fail")
        for result in (ready, retry, restart):
            assert set(result) == {
                "schema_version",
                "status",
                "selection_id",
                "selection_policy",
                "candidate_set",
                "comparison_ref",
                "next_action",
                "blocker_codes",
                "blocking_findings",
            }
            _route_comparator_result(result)

        normalized = _normalized(path)
        assert "closed blocker-code mapping" in normalized, path
        assert "retry_with_fresh_comparison_id" in normalized, path

    for path in COMPARATOR_ROUTERS:
        normalized = _normalized(path)
        for phrase in (
            "`retry_with_fresh_comparison_id`",
            "`restart_selection`",
            "unknown or mixed blocker codes",
            "deterministic protocol violation",
        ):
            assert phrase in normalized, (path, phrase)


def test_chart_build_result_is_hash_bound_complete_inventory() -> None:
    result = _json_object(CHART_SKILL, schema_version=1, status="complete")
    assert set(result) == {
        "schema_version",
        "status",
        "version_id",
        "run_id",
        "report_revision_id",
        "chart_decision",
        "hash_algorithm",
        "requested",
        "applicable",
        "generated",
        "skipped",
        "manifest",
        "blocking_findings",
    }
    requested = result["requested"]
    applicable = result["applicable"]
    generated = result["generated"]
    skipped = result["skipped"]
    generated_ids = [item["id"] for item in generated]
    skipped_ids = [item["id"] for item in skipped]
    assert generated_ids == applicable
    assert set(generated_ids).isdisjoint(skipped_ids)
    assert set(generated_ids) | set(skipped_ids) == set(requested)
    for item in generated:
        assert set(item) == {"id", "asset"}
        _assert_ref(item["asset"], ".png")
    for item in skipped:
        assert set(item) == {"id", "reason_code", "input_artifacts"}
        assert item["reason_code"] in SKIP_REASON_CODES
    _assert_ref(result["manifest"], "/report_assets/manifest.json")

    for path in (CHART_SKILL, *CHART_HANDOFF_CONSUMERS):
        normalized = _normalized(path)
        for phrase in (
            "`chart_build_result.json`",
            "requested/applicable/generated/skipped",
            "closed skip reason codes",
            "exact `{path, sha256}` manifest reference",
            "set invariants",
        ):
            assert phrase in normalized, (path, phrase)


def test_final_decision_json_is_the_only_required_decision_output() -> None:
    for path in FINAL_OUTPUT_CONTRACTS:
        normalized = _normalized(path)
        assert "`final_decision.json` is the sole canonical decision artifact" in normalized, path
        assert "final_decision.md" not in _text(path), path

    for path in REVISION_SELECTION_CONTRACTS:
        decision = _json_object(path, schema_version=5, status="selected")
        pointer = next(
            item
            for item in _json_examples(path)
            if isinstance(item, dict)
            and item.get("schema_version") == 4
            and item.get("selection_request_id") == decision["selection_request_id"]
            and "final_decision" in item
        )
        assert set(decision) == {
            "schema_version",
            "status",
            "selection_id",
            "selection_request_id",
            "selected_version_id",
            "selected_run_id",
            "selected_as",
            "selected_run",
            "report_revision",
            "report_review",
            "lineage_audit",
            "selection_policy",
            "candidate_set",
            "comparison_refs",
            "blocking_findings",
        }
        assert set(pointer) == {
            "schema_version",
            "selection_id",
            "selection_request_id",
            "selected_version_id",
            "selected_run_id",
            "candidate_set",
            "final_decision",
        }
        for field in (
            "selection_id",
            "selection_request_id",
            "selected_version_id",
            "selected_run_id",
            "candidate_set",
        ):
            assert pointer[field] == decision[field]

        mismatched = json.loads(json.dumps(pointer))
        mismatched["selected_run_id"] = "run_other"
        with pytest.raises(AssertionError):
            for field in (
                "selection_id",
                "selection_request_id",
                "selected_version_id",
                "selected_run_id",
                "candidate_set",
            ):
                assert mismatched[field] == decision[field]
