from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from oxq.run_digests import (
    RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION,
    RunArtifactInventoryProfile,
    RunDigestError,
    require_current_run_digest,
    validate_run_artifact_inventory,
)

GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
REPORT_REVIEW_CONTRACTS = (
    Path("agent/skills/review-research-report/SKILL.md"),
    Path("agent/roles/oxq-report-reviewer-worker.md"),
)
LINEAGE_CONTRACTS = (
    Path("agent/skills/audit-artifact-lineage/SKILL.md"),
    Path("agent/roles/oxq-lineage-auditor-worker.md"),
)
COMPARISON_CONTRACTS = (
    Path("agent/skills/compare-strategy-versions/SKILL.md"),
    Path("agent/roles/oxq-experiment-comparator-worker.md"),
)
FINAL_SELECTOR_CONTRACTS = (
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
)
HISTORICAL_HANDOFF_CONTRACTS = (*REPORT_REVIEW_CONTRACTS, GOVERNANCE_DOC)
HISTORICAL_STATE_CONTRACTS = (
    Path("agent/skills/manage-strategy-version/SKILL.md"),
    Path("agent/roles/oxq-version-manager-worker.md"),
    Path("agent/roles/oxq-coordinator.md"),
)
INVENTORY_GATE_CONTRACTS = (
    *REPORT_REVIEW_CONTRACTS,
    *LINEAGE_CONTRACTS,
    *COMPARISON_CONTRACTS,
    *FINAL_SELECTOR_CONTRACTS,
    GOVERNANCE_DOC,
)
FINAL_SCHEMA_CONTRACTS = (*FINAL_SELECTOR_CONTRACTS, GOVERNANCE_DOC)

FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
RUN_DIGEST = re.compile(r"sha256:[0-9a-f]{16}")
CANDIDATE_SET_KEYS = {
    "schema_version",
    "selection_id",
    "hash_algorithm",
    "selection_policy",
    "candidates",
}
CANDIDATE_KEYS = {"ordinal", "identity", "primary_run", "lineage_audit"}
FINAL_DECISION_KEYS = {
    "schema_version",
    "selection_id",
    "status",
    "selected_version_id",
    "selected_run_id",
    "selected_as",
    "hash_algorithm",
    "candidate_set",
    "selected_run",
    "report_artifacts",
    "report_review",
    "lineage_audit",
    "selection_policy",
    "comparison_refs",
    "blocked_candidates",
    "blocking_findings",
    "created_by_role",
}
CURRENT_FINAL_KEYS = {
    "schema_version",
    "selection_id",
    "selected_version_id",
    "selected_run_id",
    "candidate_set",
    "final_decision",
}


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


def _example_matching(path: Path, predicate: object) -> dict[str, object]:
    assert callable(predicate)
    example = next((item for item in _json_examples(path) if predicate(item)), None)
    assert example is not None, path
    return example


def _canonical_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _file_digest(payload: dict[str, object]) -> str:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _artifact_digest(name: str, content: bytes) -> str:
    if name in {"data_manifest.json", "metrics.json", "environment.json"}:
        payload = json.loads(content)
        if name == "metrics.json":
            payload.pop("run_id", None)
        else:
            payload.pop("run_timestamp", None)
        return _canonical_digest(payload)
    return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"


def _write_v1_run(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / "09_backtests" / "runA"
    run_dir.mkdir(parents=True)
    artifacts = {
        "data_manifest.json": b'{"schema_version": 1}\n',
        "equity_curve.csv": b"date,equity\n2026-01-01,1.0\n",
        "trades.csv": b"date,symbol\n",
        "metrics.json": b'{"run_id": "runA", "sharpe": 1.0}\n',
        "strategy_spec.yaml": b"schema_version: 1\nstrategy_id: demo\n",
        "environment.json": b'{"python": "3.12", "run_timestamp": "ignored"}\n',
        "positions.csv": b"date,symbol,quantity\n",
        "orders.csv": b"date,symbol,quantity\n",
    }
    for name, content in artifacts.items():
        (run_dir / name).write_bytes(content)
    manifest: dict[str, object] = {
        "schema_version": 1,
        **{name: _artifact_digest(name, content) for name, content in artifacts.items()},
    }
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, manifest


def _validate_candidate_set(
    payload: dict[str, object],
    expected_identities: list[tuple[str, str]],
) -> None:
    assert set(payload) == CANDIDATE_SET_KEYS
    assert payload["schema_version"] == 2
    assert payload["hash_algorithm"] == "sha256-file-bytes-v1"
    assert isinstance(payload["selection_id"], str) and payload["selection_id"]
    assert set(payload["selection_policy"]) == {"path", "sha256"}
    assert payload["selection_policy"]["path"].endswith("/selection_policy.json")
    assert FULL_SHA256.fullmatch(payload["selection_policy"]["sha256"])
    candidates = payload["candidates"]
    assert isinstance(candidates, list) and candidates
    assert [item["ordinal"] for item in candidates] == list(range(len(candidates)))
    assert all(set(item) == CANDIDATE_KEYS for item in candidates)

    identities: list[tuple[str, str]] = []
    for candidate in candidates:
        identity = candidate["identity"]
        assert set(identity) == {"version_id", "run_id"}
        pair = (identity["version_id"], identity["run_id"])
        assert all(isinstance(value, str) and value for value in pair)
        identities.append(pair)

        primary_run = candidate["primary_run"]
        assert set(primary_run) == {"path", "digest"}
        assert primary_run["path"].endswith(f"/{pair[1]}")
        assert RUN_DIGEST.fullmatch(primary_run["digest"])

        lineage_audit = candidate["lineage_audit"]
        assert set(lineage_audit) == {"path", "sha256", "scope"}
        assert FULL_SHA256.fullmatch(lineage_audit["sha256"])
        assert lineage_audit["scope"] == {"version_id": pair[0], "run_id": pair[1]}

    assert len(identities) == len(set(identities))
    assert identities == expected_identities


def _candidate_identities(payload: dict[str, object]) -> list[tuple[str, str]]:
    return [
        (item["identity"]["version_id"], item["identity"]["run_id"])
        for item in payload["candidates"]
    ]


def _validate_comparison_coverage(
    candidate_set: dict[str, object],
    comparisons: list[list[tuple[str, str]]],
) -> None:
    candidates = _candidate_identities(candidate_set)
    candidate_members = set(candidates)
    if len(candidates) > 1:
        assert comparisons, "multiple candidates require comparison evidence"

    covered: set[tuple[str, str]] = set()
    graph: dict[tuple[str, str], set[tuple[str, str]]] = {
        candidate: set() for candidate in candidates
    }
    for population in comparisons:
        assert len(population) >= 2
        assert len(population) == len(set(population))
        assert set(population) <= candidate_members, "unrelated comparison candidate"
        assert population == [candidate for candidate in candidates if candidate in population]
        covered.update(population)
        for left in population:
            graph[left].update(right for right in population if right != left)

    if len(candidates) == 2:
        assert all(population == candidates for population in comparisons)
    elif len(candidates) > 2:
        assert covered == candidate_members, "incomplete candidate population coverage"
        visited = {candidates[0]}
        frontier = [candidates[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in graph[current] - visited:
                visited.add(neighbor)
                frontier.append(neighbor)
        assert visited == candidate_members, "comparison coverage graph is disconnected"


def test_public_inventory_v1_rejects_omission_alias_and_digest_profile_mismatch(
    tmp_path: Path,
) -> None:
    run_dir, manifest = _write_v1_run(tmp_path)
    profile = validate_run_artifact_inventory(run_dir)
    assert isinstance(profile, RunArtifactInventoryProfile)
    assert profile.contract_schema_version == RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION == 1
    assert profile.name == "artifact_hashes_v1"

    omitted = copy.deepcopy(manifest)
    omitted.pop("orders.csv")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(omitted), encoding="utf-8")
    with pytest.raises(RunDigestError, match="missing required bindings"):
        validate_run_artifact_inventory(run_dir)

    aliased = copy.deepcopy(manifest)
    aliased["./orders.csv"] = aliased.pop("orders.csv")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(aliased), encoding="utf-8")
    with pytest.raises(RunDigestError, match="non-canonical artifact path"):
        validate_run_artifact_inventory(run_dir)

    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps(
            {
                "run_id": "runA",
                "artifact_hashes": _canonical_digest(manifest),
                "artifact_inventory": {
                    "schema_version": 1,
                    "profile": "artifact_hashes_v2",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RunDigestError, match="inventory profile mismatch"):
        require_current_run_digest(run_dir)


def test_all_consumers_independently_invoke_and_pin_inventory_v1_pre_and_post() -> None:
    for path in INVENTORY_GATE_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "independently invoke `validate_run_artifact_inventory(run_dir)`",
            "before evidence consumption and again immediately before publication",
            "`profile.contract_schema_version == run_artifact_inventory_schema_version == 1`",
            "`digest_row.artifact_inventory == {\"schema_version\": 1, \"profile\": profile.name}`",
            "independent of the digest-row check",
        ):
            assert phrase in normalized, (path, phrase)


def test_candidate_set_v2_and_hash_bindings_are_exact_and_shared() -> None:
    candidate_sets = []
    final_decisions = []
    pointers = []
    for path in FINAL_SCHEMA_CONTRACTS:
        candidate_set = _example_matching(path, lambda item: "candidates" in item)
        final_decision = _example_matching(path, lambda item: item.get("schema_version") == 4)
        pointer = _example_matching(
            path,
            lambda item: item.get("schema_version") == 3 and "final_decision" in item,
        )
        expected = [("v001", "runA"), ("v002", "run_20260712_173012")]
        _validate_candidate_set(candidate_set, expected)
        assert set(final_decision) == FINAL_DECISION_KEYS
        assert set(pointer) == CURRENT_FINAL_KEYS
        assert final_decision["candidate_set"] == pointer["candidate_set"]
        assert final_decision["candidate_set"]["path"].endswith("/candidate_set.json")
        assert FULL_SHA256.fullmatch(final_decision["candidate_set"]["sha256"])

        normalized = _normalized(path)
        for phrase in (
            "`candidate_set.json` uses schema version 2",
            "exact ordered equality",
            "revalidate `candidate_set.json` immediately before final-decision publication",
            "revalidate it again immediately before current-pointer publication",
            "do not mutate or backfill an existing candidate set",
        ):
            assert phrase in normalized, (path, phrase)

        candidate_sets.append(candidate_set)
        final_decisions.append(final_decision)
        pointers.append(pointer)

    assert candidate_sets[0] == candidate_sets[1] == candidate_sets[2]
    assert final_decisions[0] == final_decisions[1] == final_decisions[2]
    assert pointers[0] == pointers[1] == pointers[2]


@pytest.mark.parametrize(
    "mutation",
    ("omission", "reorder", "duplicate", "wrong_identity", "wrong_run", "wrong_lineage"),
)
def test_candidate_set_mutations_are_rejected(mutation: str) -> None:
    payload = copy.deepcopy(_example_matching(FINAL_SELECTOR_CONTRACTS[0], lambda item: "candidates" in item))
    expected = [("v001", "runA"), ("v002", "run_20260712_173012")]
    candidates = payload["candidates"]
    if mutation == "omission":
        candidates.pop()
    elif mutation == "reorder":
        candidates.reverse()
    elif mutation == "duplicate":
        candidates[1]["identity"] = copy.deepcopy(candidates[0]["identity"])
    elif mutation == "wrong_identity":
        candidates[1]["identity"]["version_id"] = "v999"
    elif mutation == "wrong_run":
        candidates[1]["primary_run"]["path"] = "versions/v002/09_backtests/run_other"
    else:
        candidates[1]["lineage_audit"]["scope"]["run_id"] = "run_other"

    with pytest.raises(AssertionError):
        _validate_candidate_set(payload, expected)


def test_candidate_set_hash_mutation_is_rejected_by_decision_and_pointer() -> None:
    payload = copy.deepcopy(_example_matching(FINAL_SELECTOR_CONTRACTS[0], lambda item: "candidates" in item))
    bound_hash = _file_digest(payload)
    decision_ref = {"path": "final/selection_1/candidate_set.json", "sha256": bound_hash}
    pointer_ref = copy.deepcopy(decision_ref)
    payload["candidates"].reverse()
    current_hash = _file_digest(payload)
    assert current_hash != decision_ref["sha256"]
    assert current_hash != pointer_ref["sha256"]


def test_comparison_coverage_is_exact_for_two_and_complete_for_larger_sets() -> None:
    candidate_set = copy.deepcopy(
        _example_matching(FINAL_SELECTOR_CONTRACTS[0], lambda item: "candidates" in item)
    )
    two = _candidate_identities(candidate_set)
    _validate_comparison_coverage(candidate_set, [two])

    with pytest.raises(AssertionError, match="unrelated comparison candidate"):
        _validate_comparison_coverage(candidate_set, [[two[1], ("v999", "runZ")]])
    with pytest.raises(AssertionError, match="multiple candidates require comparison evidence"):
        _validate_comparison_coverage(candidate_set, [])

    third = copy.deepcopy(candidate_set["candidates"][1])
    third["ordinal"] = 2
    third["identity"] = {"version_id": "v003", "run_id": "runC"}
    third["primary_run"] = {
        "path": "versions/v003/09_backtests/runC",
        "digest": "sha256:3333333333333333",
    }
    third["lineage_audit"] = {
        "path": "governance/lineage_audit_v003_runC.json",
        "sha256": "sha256:" + "3" * 64,
        "scope": {"version_id": "v003", "run_id": "runC"},
    }
    candidate_set["candidates"].append(third)
    three = _candidate_identities(candidate_set)
    _validate_comparison_coverage(candidate_set, [three[:2], three[1:]])
    with pytest.raises(AssertionError, match="incomplete candidate population coverage"):
        _validate_comparison_coverage(candidate_set, [three[:2]])

    for path in FINAL_SCHEMA_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "comparison population must be a subset of the hash-bound candidate set",
            "exactly equal the two-candidate ordered population",
            "union must exactly equal the complete candidate-set population",
            "comparison coverage graph must be connected",
            "comparison_refs must be non-empty whenever candidate_set has multiple candidates",
        ):
            assert phrase in normalized, (path, phrase)


def test_current_schema_inactive_candidate_can_receive_guarded_stale_rereview() -> None:
    handoffs = []
    for path in HISTORICAL_HANDOFF_CONTRACTS:
        handoff = _example_matching(path, lambda item: item.get("mode") == "candidate_scoped_historical_rereview")
        assert set(handoff) == {
            "mode",
            "version_id",
            "run_id",
            "current_state_guard",
            "reason",
            "requested_by_role",
        }
        assert handoff["version_id"] != handoff["current_state_guard"]["active_version"]
        assert handoff["reason"] in {"missing_report_review", "stale_report_review"}

        normalized = _normalized(path)
        for phrase in (
            "any explicit inactive candidate",
            "missing or stale",
            "current-schema `report_review.json`",
            "must not change `current.json`",
            "rerun deterministic report qa",
            "rerun artifact lineage audit",
            "regenerate every comparison",
            "rerun final selection",
        ):
            assert phrase in normalized, (path, phrase)
        assert "pre-round-17" not in normalized, path
        handoffs.append(handoff)

    assert handoffs[0] == handoffs[1] == handoffs[2]

    for path in HISTORICAL_STATE_CONTRACTS:
        normalized = _normalized(path)
        assert "any explicit inactive candidate" in normalized, path
        assert "review is missing or stale" in normalized, path
        assert "do not update `current.json`" in normalized, path
        assert "pre-round-17" not in normalized, path
