from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
SELECTOR_SKILL = Path("agent/skills/select-final-version/SKILL.md")
SELECTOR_ROLE = Path("agent/roles/oxq-final-selector-worker.md")
COMPARATOR_SKILL = Path("agent/skills/compare-strategy-versions/SKILL.md")
COMPARATOR_ROLE = Path("agent/roles/oxq-experiment-comparator-worker.md")
COORDINATOR_ROLE = Path("agent/roles/oxq-coordinator.md")
ROUTER_SKILL = Path("agent/skills/open-xquant/SKILL.md")

SELECTOR_CONTRACTS = (SELECTOR_SKILL, SELECTOR_ROLE, GOVERNANCE_DOC)
COMPARATOR_CONTRACTS = (COMPARATOR_SKILL, COMPARATOR_ROLE, GOVERNANCE_DOC)
ORCHESTRATION_CONTRACTS = (
    ROUTER_SKILL,
    COORDINATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COMPARATOR_SKILL,
    COMPARATOR_ROLE,
    GOVERNANCE_DOC,
)
POINTER_PUBLICATION_CONTRACTS = SELECTOR_CONTRACTS
FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).lower().split())


def _json_examples(path: Path) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for block in re.findall(r"```json\n(.*?)\n```", _text(path), flags=re.DOTALL):
        payload = json.loads(block)
        if isinstance(payload, dict):
            examples.append(payload)
    return examples


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


def _validate_ref(reference: object, suffix: str) -> None:
    assert isinstance(reference, dict)
    assert set(reference) == {"path", "sha256"}
    assert isinstance(reference["path"], str)
    assert reference["path"].endswith(suffix)
    assert FULL_SHA256.fullmatch(reference["sha256"])


def _pointer_publication_attempt(
    *,
    decision_bytes_at_validation: bytes,
    decision_bytes_after_validation: bytes,
    evidence_at_decision: dict[str, bytes],
    evidence_before_pointer: dict[str, bytes],
    prior_pointer: dict[str, object],
) -> dict[str, object]:
    # This scenario model mirrors the normative all-evidence pointer gate.
    assert evidence_before_pointer == evidence_at_decision
    assert decision_bytes_after_validation == decision_bytes_at_validation
    return {
        "selection_id": "selection_20260712_180000",
        "final_decision_sha256": hashlib.sha256(
            decision_bytes_after_validation
        ).hexdigest(),
        "replaced": prior_pointer,
    }


@pytest.mark.parametrize(
    "mutated_path",
    ("selection_policy.json", "metrics_comparison.json"),
)
def test_pointer_gate_rejects_post_decision_transitive_mutation_and_preserves_pointer(
    mutated_path: str,
) -> None:
    prior_pointer = {"selection_id": "selection_previous"}
    evidence_at_decision = {
        "selection_policy.json": b'{"confirmed_by_user":true}\n',
        "comparison_manifest.json": b'{"schema_version":1}\n',
        "metrics_comparison.json": b'{"winner":"v002"}\n',
        "candidate_set.json": b'{"schema_version":1}\n',
        "lineage_audit.json": b'{"schema_version":2}\n',
        "artifact_hashes.json": b'{"schema_version":5}\n',
    }
    current_evidence = copy.deepcopy(evidence_at_decision)
    current_evidence[mutated_path] += b"mutated"

    with pytest.raises(AssertionError):
        _pointer_publication_attempt(
            decision_bytes_at_validation=b'{"schema_version":4}\n',
            decision_bytes_after_validation=b'{"schema_version":4}\n',
            evidence_at_decision=evidence_at_decision,
            evidence_before_pointer=current_evidence,
            prior_pointer=prior_pointer,
        )
    assert prior_pointer == {"selection_id": "selection_previous"}

    for path in POINTER_PUBLICATION_CONTRACTS:
        normalized = _normalized(path)
        phrases = (
                "immediately before `current_final.json` publication",
                "selection policy",
            "comparison manifests and every required comparison output",
            "candidate set",
            "complete lineage-v2 validator",
            "`validate_run_artifact_inventory(run_dir)`",
            "re-read the decision bytes after that validation",
            "byte-for-byte unchanged",
                "leave the prior `current_final.json` unchanged",
        )
        if path in (SELECTOR_SKILL, SELECTOR_ROLE):
            phrases += ("schema-version-5 decision validation again",)
        else:
            phrases += ("full schema-version-4 decision validation",)
        for phrase in phrases:
            assert phrase in normalized, (path, phrase)


def test_staged_selection_examples_complete_one_two_candidate_request() -> None:
    expected_candidate_ref: dict[str, object] | None = None
    expected_selection_id: object = None

    for path in SELECTOR_CONTRACTS:
        prepared = _example(path, status="candidate_set_ready")
        assert set(prepared) == {
            "schema_version",
            "status",
            "selection_id",
            "selection_policy",
            "candidate_set",
            "comparison_refs",
            "next_action",
            "blocking_findings",
        }
        assert prepared["schema_version"] == 2
        assert prepared["comparison_refs"] == []
        assert prepared["next_action"] == "compare_then_resume"
        assert prepared["blocking_findings"] == []
        _validate_ref(prepared["candidate_set"], "/candidate_set.json")
        _validate_ref(prepared["selection_policy"], "/selection_policy.json")

        resume = _example(path, mode="resume_selection")
        assert set(resume) == {
            "schema_version",
            "mode",
            "selection_id",
            "selection_policy",
            "candidate_set",
            "comparison_refs",
        }
        assert resume["selection_id"] == prepared["selection_id"]
        assert resume["selection_policy"] == prepared["selection_policy"]
        assert resume["candidate_set"] == prepared["candidate_set"]
        assert isinstance(resume["comparison_refs"], list)
        assert len(resume["comparison_refs"]) == 1
        _validate_ref(
            resume["comparison_refs"][0],
            "/comparison_manifest.json",
        )

        if expected_candidate_ref is None:
            expected_candidate_ref = prepared["candidate_set"]
            expected_selection_id = prepared["selection_id"]
        assert prepared["candidate_set"] == expected_candidate_ref
        assert prepared["selection_id"] == expected_selection_id

    for path in COMPARATOR_CONTRACTS:
        request = _example(
            path,
            schema_version=3,
            mode="build_selection_comparison",
        )
        assert set(request) == {
            "schema_version",
            "mode",
            "selection_id",
            "selection_request_id",
            "selection_policy",
            "candidate_set",
            "comparison_population",
        }
        assert request["selection_request_id"] == "selection-request-20260712-1"
        assert request["selection_id"] == "selection_20260712_190000"
        _validate_ref(request["selection_policy"], "/selection_policy.json")
        _validate_ref(request["candidate_set"], "/candidate_set.json")
        assert "selection_20260712_190000" in request["selection_policy"]["path"]
        assert "selection_20260712_190000" in request["candidate_set"]["path"]
        assert len(request["comparison_population"]) == 2

        ready = _example(path, status="comparison_ready")
        assert set(ready) == {
            "schema_version",
            "status",
            "selection_id",
            "selection_policy",
            "candidate_set",
            "comparison_ref",
            "blocking_findings",
        }
        assert ready["selection_id"] == expected_selection_id
        assert ready["selection_policy"] == prepared["selection_policy"]
        assert ready["candidate_set"] == expected_candidate_ref
        _validate_ref(ready["comparison_ref"], "/comparison_manifest.json")
        assert ready["blocking_findings"] == []


def test_router_and_coordinator_define_normal_prepare_compare_resume_state_machine() -> None:
    for path in ORCHESTRATION_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "prepare_selection",
            "candidate_set_ready",
            "build_selection_comparison",
            "comparison_ready",
            "resume_selection",
            "same `selection_id`",
            "same exact candidate-set reference",
        ):
            assert phrase in normalized, (path, phrase)


def test_current_selection_comparison_and_selector_result_schemas_are_v3() -> None:
    coordinator = _normalized(COORDINATOR_ROLE)
    router = _normalized(ROUTER_SKILL)

    assert "hash exact final bytes into the schema-version-3 comparison manifest" in coordinator
    assert "hash exact final bytes into the schema-version-2 comparison manifest" not in coordinator
    assert "every current selector result uses schema version 3" in router
    assert "every current selector result uses schema version 2" not in router

    for path in (ROUTER_SKILL, COORDINATOR_ROLE, SELECTOR_SKILL, SELECTOR_ROLE):
        normalized = _normalized(path)
        for phrase in (
            "normal nonterminal handoff",
            "without a new user request",
            "must not write `final_decision.json`",
            "must not update `current_final.json`",
        ):
            assert phrase in normalized, (path, phrase)


def test_selector_result_contract_defines_resume_and_restart_block_states() -> None:
    for path in SELECTOR_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "`status: blocked` with `next_action: resume_selection`",
            "missing or incomplete comparison coverage",
            "same immutable candidate set",
            "`status: blocked` with `next_action: restart_selection`",
            "candidate-set or transitive candidate evidence",
            "prior `current_final.json` remains unchanged",
        ):
            assert phrase in normalized, (path, phrase)


def _validate_connected_coverage(
    candidates: list[str],
    comparisons: list[list[str]],
    selected: str,
) -> None:
    candidate_set = set(candidates)
    assert selected in candidate_set
    covered: set[str] = set()
    graph = {candidate: set() for candidate in candidates}
    for population in comparisons:
        assert len(population) == len(set(population)) >= 2
        assert set(population) <= candidate_set
        assert population == [item for item in candidates if item in population]
        covered.update(population)
        for left in population:
            graph[left].update(right for right in population if right != left)

    if len(candidates) == 2:
        assert all(population == candidates for population in comparisons)
        return

    assert covered == candidate_set
    assert selected in covered
    visited = {candidates[0]}
    frontier = [candidates[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in graph[current] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    assert visited == candidate_set


def test_three_candidate_chain_allows_selecting_each_node_but_rejects_bad_coverage() -> None:
    candidates = ["v001/runA", "v002/runB", "v003/runC"]
    chain = [candidates[:2], candidates[1:]]
    for selected in candidates:
        _validate_connected_coverage(candidates, chain, selected)

    with pytest.raises(AssertionError):
        _validate_connected_coverage(candidates, [candidates[:2]], candidates[0])
    with pytest.raises(AssertionError):
        _validate_connected_coverage(
            candidates,
            [[candidates[0], "v999/runZ"], candidates[1:]],
            candidates[0],
        )
    with pytest.raises(AssertionError):
        _validate_connected_coverage(
            candidates,
            [[candidates[0], candidates[1]], [candidates[2], candidates[2]]],
            candidates[1],
        )

    for path in SELECTOR_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "selected identity must appear exactly once in the union of validated comparison populations",
            "must not require the selected identity in every referenced comparison",
            "exactly equal the two-candidate ordered population",
            "comparison coverage graph must be connected",
            "reject an omitted candidate",
            "unrelated replacement",
        ):
            assert phrase in normalized, (path, phrase)
        assert (
            "require that the selected version_id/run_id identity appears exactly once"
            not in normalized
        ), path
