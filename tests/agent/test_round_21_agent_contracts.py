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
LINEAGE_SKILL = Path("agent/skills/audit-artifact-lineage/SKILL.md")
LINEAGE_ROLE = Path("agent/roles/oxq-lineage-auditor-worker.md")
COORDINATOR_ROLE = Path("agent/roles/oxq-coordinator.md")
ROUTER_SKILL = Path("agent/skills/open-xquant/SKILL.md")

SELECTOR_CONTRACTS = (SELECTOR_SKILL, SELECTOR_ROLE, GOVERNANCE_DOC)
COMPARATOR_CONTRACTS = (COMPARATOR_SKILL, COMPARATOR_ROLE, GOVERNANCE_DOC)
PREPARE_CONTRACTS = (
    ROUTER_SKILL,
    COORDINATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    GOVERNANCE_DOC,
)
BINDING_CONTRACTS = (
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COMPARATOR_SKILL,
    COMPARATOR_ROLE,
    LINEAGE_SKILL,
    LINEAGE_ROLE,
    GOVERNANCE_DOC,
)
LOCK_CONTRACTS = (
    ROUTER_SKILL,
    COORDINATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    GOVERNANCE_DOC,
)
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


def _candidate(index: int) -> dict[str, object]:
    letter = chr(ord("A") + index)
    digit = str(index + 1)
    return {
        "ordinal": index,
        "identity": {"version_id": f"v00{index + 1}", "run_id": f"run{letter}"},
        "primary_run": {
            "path": f"<phase_paths.09_backtests>/run{letter}",
            "digest": f"sha256:{digit * 16}",
        },
        "lineage_audit": {
            "path": f"<governance_dir>/lineage_audit_v00{index + 1}_run{letter}.json",
            "sha256": f"sha256:{digit * 64}",
            "scope": {"version_id": f"v00{index + 1}", "run_id": f"run{letter}"},
        },
    }


def _validate_candidate(candidate: object, ordinal: int) -> None:
    assert isinstance(candidate, dict)
    assert set(candidate) == {"ordinal", "identity", "primary_run", "lineage_audit"}
    assert candidate["ordinal"] == ordinal
    identity = candidate["identity"]
    primary_run = candidate["primary_run"]
    lineage_audit = candidate["lineage_audit"]
    assert isinstance(identity, dict) and set(identity) == {"version_id", "run_id"}
    assert isinstance(primary_run, dict) and set(primary_run) == {"path", "digest"}
    assert isinstance(lineage_audit, dict)
    assert set(lineage_audit) == {"path", "sha256", "scope"}
    assert lineage_audit["scope"] == identity
    assert FULL_SHA256.fullmatch(lineage_audit["sha256"])


def _locked_pointer_attempt(
    *,
    dependency_order: tuple[str, ...],
    mutation_after_read: str | None,
    prior_pointer: bytes,
) -> bytes:
    current = {name: f"{name}:v1".encode() for name in dependency_order}
    snapshots: dict[str, bytes] = {}
    lock_held = True
    for name in dependency_order:
        assert lock_held
        snapshots[name] = current[name]
        if mutation_after_read == name:
            current[name] += b":mutated"

    assert lock_held
    for name in dependency_order:
        assert current[name] == snapshots[name]

    pointer = json.dumps(
        {
            "schema_version": 3,
            "final_decision": hashlib.sha256(current["final_decision.json"]).hexdigest(),
        },
        sort_keys=True,
    ).encode()
    return pointer if lock_held else prior_pointer


@pytest.mark.parametrize(
    "mutated_dependency",
    (
        "selection_policy.json",
        "candidate_set.json",
        "lineage_audit.json",
        "artifact_hashes.json",
        "comparison_manifest.json",
        "metrics_comparison.json",
        "report_review.json",
        "final_decision.json",
    ),
)
def test_selection_lock_rejects_mutation_after_each_dependency_read(
    mutated_dependency: str,
) -> None:
    dependency_order = (
        "selection_policy.json",
        "candidate_set.json",
        "lineage_audit.json",
        "artifact_hashes.json",
        "comparison_manifest.json",
        "metrics_comparison.json",
        "report_review.json",
        "final_decision.json",
    )
    prior_pointer = b'{"selection_id":"selection_previous"}\n'
    with pytest.raises(AssertionError):
        _locked_pointer_attempt(
            dependency_order=dependency_order,
            mutation_after_read=mutated_dependency,
            prior_pointer=prior_pointer,
        )
    assert prior_pointer == b'{"selection_id":"selection_previous"}\n'

    for path in SELECTOR_CONTRACTS:
        normalized = _normalized(path)
        phrases = (
            "canonical `<workspace_root>/.open-xquant/locks/final-selection.lock`",
            "exclusive advisory file lock",
            "hold it continuously",
            "snapshot the exact bytes at each dependency read",
            "unchanged-byte sweep",
            "atomic `current_final.json` replacement",
            "previous pointer byte-for-byte",
            "final fallible publication operation",
            "never unlink the lock file",
        )
        if path in (SELECTOR_SKILL, SELECTOR_ROLE):
            phrases += ("before direct schema-version-5 transitive revalidation",)
        else:
            phrases += ("before direct schema-version-4 transitive revalidation",)
        for phrase in phrases:
            assert phrase in normalized, (path, phrase)


def test_selection_lock_order_and_ownership_are_workspace_wide() -> None:
    for path in LOCK_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "final selector owns the lock lifecycle",
            "last lock acquired",
            "must not acquire another lock while holding it",
            "all governed writers of selection-transitive evidence",
            "leave the prior `current_final.json` byte-for-byte unchanged",
        ):
            assert phrase in normalized, (path, phrase)


def test_prepare_selection_request_envelope_is_exact_schema_v2() -> None:
    expected_identities: object = None
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
        assert request["selection_id_policy"] == {
            "source": "generated",
            "selection_id": None,
        }
        assert set(request["selection_policy"]) == {"source", "payload", "reference"}
        assert request["selection_policy"]["source"] == "confirmed_payload"
        assert request["selection_policy"]["reference"] is None
        assert request["selection_policy"]["payload"]["confirmed_by_user"] is True
        population = request["candidate_population"]
        assert isinstance(population, list) and len(population) == 2
        for ordinal, candidate in enumerate(population):
            _validate_candidate(candidate, ordinal)
        identities = [candidate["identity"] for candidate in population]
        if expected_identities is None:
            expected_identities = identities
        assert identities == expected_identities

        normalized = _normalized(path)
        for phrase in (
            "`source: generated` requires `selection_id: null`",
            "`source: provided` requires",
            "no implicit default",
            "exact ordered equality",
            "must not rediscover",
        ):
            assert phrase in normalized, (path, phrase)


def _comparison_manifest(
    *,
    selection_id: str,
    selection_policy: dict[str, str] | None = None,
    candidate_set: dict[str, str],
    population: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "selection_id": selection_id,
        "selection_policy": copy.deepcopy(
            selection_policy
            or {
                "path": f"<final_dir>/{selection_id}/selection_policy.json",
                "sha256": f"sha256:{'0' * 64}",
            }
        ),
        "candidate_set": copy.deepcopy(candidate_set),
        "candidate_identities": [copy.deepcopy(item["identity"]) for item in population],
        "candidate_evidence": copy.deepcopy(population),
    }


def _validate_manifest_binding(
    manifest: dict[str, object],
    *,
    selection_id: str,
    selection_policy: dict[str, str] | None = None,
    candidate_set: dict[str, str],
    population: list[dict[str, object]],
) -> None:
    assert manifest["schema_version"] == 2
    assert manifest["selection_id"] == selection_id
    if selection_policy is not None:
        assert manifest["selection_policy"] == selection_policy
    assert manifest["candidate_set"] == candidate_set
    assert manifest["candidate_identities"] == [item["identity"] for item in population]
    assert manifest["candidate_evidence"] == population


@pytest.mark.parametrize("candidate_count", (2, 3))
def test_comparison_manifest_cannot_cross_selection_substitute(
    candidate_count: int,
) -> None:
    population = [_candidate(index) for index in range(candidate_count)]
    s1_ref = {
        "path": "<final_dir>/selection_S1/candidate_set.json",
        "sha256": f"sha256:{'a' * 64}",
    }
    s2_ref = {
        "path": "<final_dir>/selection_S2/candidate_set.json",
        "sha256": f"sha256:{'b' * 64}",
    }
    manifest = _comparison_manifest(
        selection_id="selection_S1",
        candidate_set=s1_ref,
        population=population,
    )
    _validate_manifest_binding(
        manifest,
        selection_id="selection_S1",
        candidate_set=s1_ref,
        population=population,
    )
    with pytest.raises(AssertionError):
        _validate_manifest_binding(
            manifest,
            selection_id="selection_S2",
            candidate_set=s2_ref,
            population=population,
        )

    for path in BINDING_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "schema-version-2 comparison manifest",
            "exact `selection_id`",
            "exact `{path, sha256}` candidate-set reference",
            "candidate evidence must equal the exact ordered projection",
            "cross-selection substitution",
            "schema-version-1 comparison manifests are historical only",
            "regenerate",
        ):
            assert phrase in normalized, (path, phrase)


@pytest.mark.parametrize("candidate_count", (2, 3))
def test_complete_prepare_compare_resume_pointer_handoff(candidate_count: int) -> None:
    population = [_candidate(index) for index in range(candidate_count)]
    request = {
        "schema_version": 2,
        "mode": "prepare_selection",
        "selection_id_policy": {"source": "generated", "selection_id": None},
        "selection_policy": {
            "source": "confirmed_payload",
            "payload": {"confirmed_by_user": True},
            "reference": None,
        },
        "candidate_population": copy.deepcopy(population),
    }
    selection_id = "selection_generated"
    policy_ref = {
        "path": f"<final_dir>/{selection_id}/selection_policy.json",
        "sha256": f"sha256:{'0' * 64}",
    }
    candidate_set = {
        "schema_version": 2,
        "selection_id": selection_id,
        "selection_policy": policy_ref,
        "candidates": copy.deepcopy(request["candidate_population"]),
    }
    candidate_ref = {
        "path": f"<final_dir>/{selection_id}/candidate_set.json",
        "sha256": f"sha256:{'a' * 64}",
    }
    prepared = {
        "schema_version": 2,
        "status": "candidate_set_ready",
        "selection_id": selection_id,
        "selection_policy": policy_ref,
        "candidate_set": candidate_ref,
        "comparison_refs": [],
        "next_action": "compare_then_resume",
        "blocking_findings": [],
    }
    comparison_populations = (
        [population] if candidate_count == 2 else [population[:2], population[1:]]
    )
    comparison_requests = [
        {
            "schema_version": 2,
            "mode": "build_selection_comparison",
            "selection_id": prepared["selection_id"],
            "selection_policy": prepared["selection_policy"],
            "candidate_set": prepared["candidate_set"],
            "comparison_population": [item["identity"] for item in items],
        }
        for items in comparison_populations
    ]
    manifests = [
        _comparison_manifest(
            selection_id=request_item["selection_id"],
            selection_policy=request_item["selection_policy"],
            candidate_set=request_item["candidate_set"],
            population=items,
        )
        for request_item, items in zip(
            comparison_requests,
            comparison_populations,
            strict=True,
        )
    ]
    for manifest, items in zip(manifests, comparison_populations, strict=True):
        _validate_manifest_binding(
            manifest,
            selection_id=selection_id,
            selection_policy=policy_ref,
            candidate_set=candidate_ref,
            population=items,
        )
    comparison_results = [
        {
            "schema_version": 2,
            "status": "comparison_ready",
            "selection_id": manifest["selection_id"],
            "selection_policy": manifest["selection_policy"],
            "candidate_set": manifest["candidate_set"],
            "comparison_ref": {
                "path": f"<comparisons_dir>/cmp_{index}/comparison_manifest.json",
                "sha256": f"sha256:{str(index + 1) * 64}",
            },
            "blocking_findings": [],
        }
        for index, manifest in enumerate(manifests)
    ]
    resume = {
        "schema_version": 2,
        "mode": "resume_selection",
        "selection_id": prepared["selection_id"],
        "selection_policy": prepared["selection_policy"],
        "candidate_set": prepared["candidate_set"],
        "comparison_refs": [item["comparison_ref"] for item in comparison_results],
    }
    decision = {
        "schema_version": 4,
        "selection_id": resume["selection_id"],
        "selection_policy": resume["selection_policy"],
        "candidate_set": resume["candidate_set"],
        "comparison_refs": resume["comparison_refs"],
    }
    pointer = {
        "schema_version": 3,
        "selection_id": decision["selection_id"],
        "candidate_set": decision["candidate_set"],
        "final_decision": {
            "path": f"<final_dir>/{selection_id}/final_decision.json",
            "sha256": f"sha256:{'f' * 64}",
        },
    }
    assert candidate_set["candidates"] == request["candidate_population"]
    assert prepared["selection_id"] == candidate_set["selection_id"] == selection_id
    assert all(item["selection_id"] == selection_id for item in comparison_requests)
    assert all(item["selection_id"] == selection_id for item in comparison_results)
    assert pointer["selection_id"] == decision["selection_id"] == selection_id
    assert pointer["candidate_set"] == candidate_ref

    for path in PREPARE_CONTRACTS:
        normalized = _normalized(path)
        assert "complete two-candidate and three-candidate handoffs" in normalized, path
        assert "request through pointer publication" in normalized, path
