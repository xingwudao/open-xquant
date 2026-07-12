from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from oxq.selection_lock import hold_final_selection_lock

GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
SELECTOR_SKILL = Path("agent/skills/select-final-version/SKILL.md")
SELECTOR_ROLE = Path("agent/roles/oxq-final-selector-worker.md")
COMPARATOR_SKILL = Path("agent/skills/compare-strategy-versions/SKILL.md")
COMPARATOR_ROLE = Path("agent/roles/oxq-experiment-comparator-worker.md")
LINEAGE_SKILL = Path("agent/skills/audit-artifact-lineage/SKILL.md")
LINEAGE_ROLE = Path("agent/roles/oxq-lineage-auditor-worker.md")
CHART_SKILL = Path("agent/skills/build-report-charts/SKILL.md")
WRITER_SKILL = Path("agent/skills/write-research-report/SKILL.md")
WRITER_ROLE = Path("agent/roles/oxq-report-writer-worker.md")
REVIEWER_SKILL = Path("agent/skills/review-research-report/SKILL.md")
REVIEWER_ROLE = Path("agent/roles/oxq-report-reviewer-worker.md")
MONITOR_SKILL = Path("agent/skills/monitor-strategy-run/SKILL.md")
MONITOR_ROLE = Path("agent/roles/oxq-monitor-worker.md")
RUNNER_SKILL = Path("agent/skills/run-authorized-backtest/SKILL.md")
RUNNER_ROLE = Path("agent/roles/oxq-runner-worker.md")
COORDINATOR_ROLE = Path("agent/roles/oxq-coordinator.md")
ROUTER_SKILL = Path("agent/skills/open-xquant/SKILL.md")

PREPARE_CONTRACTS = (
    ROUTER_SKILL,
    COORDINATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    GOVERNANCE_DOC,
)
DIRECT_PUBLISHERS = (
    COMPARATOR_SKILL,
    COMPARATOR_ROLE,
    LINEAGE_SKILL,
    LINEAGE_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
)
RUNTIME_PUBLISHER_GUIDANCE = (
    RUNNER_SKILL,
    RUNNER_ROLE,
    MONITOR_SKILL,
    MONITOR_ROLE,
    COORDINATOR_ROLE,
)
FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).lower().split())


def _json_examples(path: Path) -> list[dict[str, object]]:
    return [
        payload
        for block in re.findall(r"```json\n(.*?)\n```", _text(path), flags=re.DOTALL)
        if isinstance(payload := json.loads(block), dict)
    ]


def _example(path: Path, **fields: object) -> dict[str, object]:
    payload = next(
        (
            item
            for item in _json_examples(path)
            if all(item.get(key) == value for key, value in fields.items())
        ),
        None,
    )
    assert payload is not None, (path, fields)
    return payload


def _validate_confirmed_payload(payload: object) -> None:
    assert isinstance(payload, dict)
    assert set(payload) == {
        "schema_version",
        "confirmed_by_user",
        "confirmation",
        "eligible_if",
        "rank_by",
        "tie_breakers",
    }
    assert payload["schema_version"] == 1
    assert payload["confirmed_by_user"] is True
    confirmation = payload["confirmation"]
    assert isinstance(confirmation, dict)
    assert set(confirmation) == {"source_conversation", "confirmed_at"}
    assert isinstance(confirmation["source_conversation"], str)
    assert confirmation["source_conversation"]
    assert isinstance(confirmation["confirmed_at"], str)
    assert confirmation["confirmed_at"].endswith("Z")
    assert isinstance(payload["eligible_if"], dict) and payload["eligible_if"]
    assert isinstance(payload["rank_by"], list) and payload["rank_by"]
    assert isinstance(payload["tie_breakers"], list)


def _validate_policy_input(policy_input: object) -> None:
    assert isinstance(policy_input, dict)
    assert set(policy_input) == {"source", "payload", "reference"}
    if policy_input["source"] == "confirmed_payload":
        _validate_confirmed_payload(policy_input["payload"])
        assert policy_input["reference"] is None
    else:
        assert policy_input["source"] == "hash_bound_reference"
        assert policy_input["payload"] is None
        reference = policy_input["reference"]
        assert isinstance(reference, dict)
        assert set(reference) == {"path", "sha256"}
        assert FULL_SHA256.fullmatch(reference["sha256"])


def test_prepare_selection_v2_carries_exact_confirmed_policy_input() -> None:
    expected_policy: object = None
    for path in PREPARE_CONTRACTS:
        request = _example(path, mode="prepare_selection")
        assert set(request) == {
            "schema_version",
            "mode",
            "selection_id_policy",
            "selection_policy",
            "candidate_population",
        }
        assert request["schema_version"] == 2
        _validate_policy_input(request["selection_policy"])
        if expected_policy is None:
            expected_policy = request["selection_policy"]
        assert request["selection_policy"] == expected_policy

        normalized = _normalized(path)
        for phrase in (
            "must not infer policy fields",
            "schema-version-2 `selection_policy.json`",
            "exact user-confirmed payload",
            "hash-bound source reference",
            "atomically publishes it inside the generated selection directory",
        ):
            assert phrase in normalized, (path, phrase)


def test_policy_binding_is_shared_by_candidate_set_and_all_staged_handoffs() -> None:
    for path in (SELECTOR_SKILL, SELECTOR_ROLE, GOVERNANCE_DOC):
        candidate_set = next(
            item
            for item in _json_examples(path)
            if item.get("schema_version") == 2 and "candidates" in item
        )
        assert set(candidate_set) == {
            "schema_version",
            "selection_id",
            "hash_algorithm",
            "selection_policy",
            "candidates",
        }
        policy_ref = candidate_set["selection_policy"]
        assert isinstance(policy_ref, dict)
        assert set(policy_ref) == {"path", "sha256"}
        assert FULL_SHA256.fullmatch(policy_ref["sha256"])

        prepared = _example(path, status="candidate_set_ready")
        resume = _example(path, mode="resume_selection")
        assert prepared["schema_version"] == resume["schema_version"] == 2
        assert prepared["selection_policy"] == resume["selection_policy"] == policy_ref

        normalized = _normalized(path)
        for phrase in (
            "reject stale or cross-selection policy",
            "exact selection id",
            "policy reference must equal",
            "schema-version-1 prepare requests and candidate sets are historical only",
            "restart_selection",
        ):
            assert phrase in normalized, (path, phrase)


def test_policy_binding_closes_comparison_final_and_pointer_schema_chain() -> None:
    for path in (SELECTOR_SKILL, SELECTOR_ROLE, GOVERNANCE_DOC):
        examples = _json_examples(path)
        policy = next(
            item
            for item in examples
            if item.get("schema_version") == 2 and "policy_payload" in item
        )
        candidate_set = next(
            item
            for item in examples
            if item.get("schema_version") == 2 and "candidates" in item
        )
        decision = next(
            item
            for item in examples
            if item.get("schema_version") == 4 and "candidate_set" in item
        )
        pointer = next(
            item
            for item in examples
            if item.get("schema_version") == 3 and "final_decision" in item
        )
        assert policy["selection_id"] == candidate_set["selection_id"] == decision["selection_id"]
        assert decision["selection_policy"] == candidate_set["selection_policy"]
        assert pointer["candidate_set"] == decision["candidate_set"]

    expected_policy: object = None
    expected_candidate_set: object = None
    for path in (COMPARATOR_SKILL, COMPARATOR_ROLE, GOVERNANCE_DOC):
        manifest = next(
            item
            for item in _json_examples(path)
            if item.get("schema_version") == 2
            and "comparison_id" in item
            and "selection_id" in item
        )
        assert set(manifest["selection_policy"]) == {"path", "sha256"}
        assert set(manifest["candidate_set"]) == {"path", "sha256"}
        if expected_policy is None:
            expected_policy = manifest["selection_policy"]
            expected_candidate_set = manifest["candidate_set"]
        assert manifest["selection_policy"] == expected_policy
        assert manifest["candidate_set"] == expected_candidate_set


def test_exactly_one_candidate_resumes_directly_without_comparator() -> None:
    for path in PREPARE_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "exactly one candidate",
            "`next_action: resume_selection`",
            "`comparison_refs: []`",
            "must not invoke the comparator",
            "two or more candidates",
            "`next_action: compare_then_resume`",
        ):
            assert phrase in normalized, (path, phrase)

    population = [{"ordinal": 0, "identity": {"version_id": "v001", "run_id": "runA"}}]
    prepared = {
        "schema_version": 2,
        "status": "candidate_set_ready",
        "selection_id": "selection_single",
        "selection_policy": {
            "path": "<final_dir>/selection_single/selection_policy.json",
            "sha256": f"sha256:{'1' * 64}",
        },
        "candidate_set": {
            "path": "<final_dir>/selection_single/candidate_set.json",
            "sha256": f"sha256:{'2' * 64}",
        },
        "comparison_refs": [],
        "next_action": "resume_selection",
        "blocking_findings": [],
    }
    resume = {
        "schema_version": 2,
        "mode": "resume_selection",
        "selection_id": prepared["selection_id"],
        "selection_policy": prepared["selection_policy"],
        "candidate_set": prepared["candidate_set"],
        "comparison_refs": [],
    }
    decision = {
        "schema_version": 4,
        "selection_id": resume["selection_id"],
        "selection_policy": resume["selection_policy"],
        "candidate_set": resume["candidate_set"],
        "comparison_refs": resume["comparison_refs"],
        "selected": population[0]["identity"],
    }
    pointer = {
        "schema_version": 3,
        "selection_id": decision["selection_id"],
        "candidate_set": decision["candidate_set"],
        "final_decision": {
            "path": "<final_dir>/selection_single/final_decision.json",
            "sha256": f"sha256:{'3' * 64}",
        },
    }
    assert resume["comparison_refs"] == decision["comparison_refs"] == []
    assert prepared["next_action"] == "resume_selection"
    assert pointer["selection_id"] == decision["selection_id"]


def test_agent_publishers_use_runtime_selection_lock_protocol() -> None:
    for path in DIRECT_PUBLISHERS:
        normalized = _normalized(path)
        for phrase in (
            "`governing_workspace_root(subject)`",
            "`final_selection_lock_path(subject)`",
            "`hold_final_selection_lock(precomputed_path)`",
            "nearest ancestor `.open-xquant/workspace.yaml`",
            "valid non-governed",
            "malformed or unsafe governed configuration",
            "fail closed",
            "last lock acquired",
        ):
            assert phrase in normalized, (path, phrase)

    for path in RUNTIME_PUBLISHER_GUIDANCE:
        normalized = _normalized(path)
        for phrase in (
            "runtime publisher acquires the final-selection lock centrally",
            "run_digests.jsonl.lock",
            "final-selection.lock",
            "innermost",
            "must not pre-acquire",
        ):
            assert phrase in normalized, (path, phrase)


def test_selector_uses_only_direct_byte_checks_inside_final_lock() -> None:
    for path in (SELECTOR_SKILL, SELECTOR_ROLE, GOVERNANCE_DOC):
        normalized = _normalized(path)
        for phrase in (
            "release every run and registry lock before",
            "inside the final-selection lock",
            "only direct byte snapshots",
            "must not invoke `validate_run_artifact_inventory`",
            "must not invoke `require_current_run_digest`",
            "must not invoke run-locking apis",
        ):
            assert phrase in normalized, (path, phrase)


def test_evidence_publisher_waits_for_pointer_byte_sweep(tmp_path: Path) -> None:
    lock_path = tmp_path / ".open-xquant/locks/final-selection.lock"
    lock_path.parent.parent.mkdir()
    evidence = tmp_path / "report_review.json"
    attempted = tmp_path / "publisher-attempted"
    evidence.write_bytes(b'{"status":"old"}\n')
    script = """
import os
import sys
from pathlib import Path
from oxq.selection_lock import hold_final_selection_lock

lock_path = Path(sys.argv[1])
evidence = Path(sys.argv[2])
attempted = Path(sys.argv[3])
attempted.write_text("ready", encoding="utf-8")
with hold_final_selection_lock(lock_path):
    temporary = evidence.with_suffix(".tmp")
    temporary.write_bytes(b'{"status":"new"}\\n')
    os.replace(temporary, evidence)
"""

    with hold_final_selection_lock(lock_path):
        before_sweep = evidence.read_bytes()
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(lock_path), str(evidence), str(attempted)],
            env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
        )
        deadline = time.monotonic() + 5
        while not attempted.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert attempted.exists()
        time.sleep(0.1)
        assert process.poll() is None
        assert evidence.read_bytes() == before_sweep

    assert process.wait(timeout=5) == 0
    assert evidence.read_bytes() == b'{"status":"new"}\n'
