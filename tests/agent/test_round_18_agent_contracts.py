from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import pytest

from oxq.run_digests import RunDigestError, require_current_run_digest

GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
LINEAGE_CONTRACTS = (
    Path("agent/skills/audit-artifact-lineage/SKILL.md"),
    Path("agent/roles/oxq-lineage-auditor-worker.md"),
)
REPORT_REVIEW_CONTRACTS = (
    Path("agent/skills/review-research-report/SKILL.md"),
    Path("agent/roles/oxq-report-reviewer-worker.md"),
)
COMPARISON_CONTRACTS = (
    Path("agent/skills/compare-strategy-versions/SKILL.md"),
    Path("agent/roles/oxq-experiment-comparator-worker.md"),
)
FINAL_SELECTOR_CONTRACTS = (
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
)
HISTORICAL_HANDOFF_CONTRACTS = (
    *REPORT_REVIEW_CONTRACTS,
    GOVERNANCE_DOC,
)
FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
RUN_DIGEST = re.compile(r"sha256:[0-9a-f]{16}")

MANDATORY_LINEAGE_PATHS = (
    "<version_root>/v002/version_manifest.json",
    "<phase_paths.04_spec_build>/strategy_spec.yaml",
    "<phase_paths.06_spec_audit>/spec_audit.json",
    "<phase_paths.07_compile_preview>/compiled_plan.json",
    "<phase_paths.08_runtime_audit>/runtime_audit.json",
    "<phase_paths.09_backtests>/run_20260712_173012/artifact_hashes.json",
    "<phase_paths.09_backtests>/run_20260712_173012/reproducibility_audit.json",
    "<phase_paths.09_backtests>/run_20260712_173012/research_bias_audit.json",
    "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
)
COMPARISON_OUTPUTS = {
    "comparability_audit.json",
    "metrics_comparison.json",
    "spec_diff.yaml",
    "comparison_report.md",
    "figures",
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


def _example_with(path: Path, field: str) -> dict[str, object]:
    example = next((item for item in _json_examples(path) if field in item), None)
    assert example is not None, (path, field)
    return example


def _canonical_symbolic_path(path: str) -> str:
    candidate = PurePosixPath(path)
    assert not candidate.is_absolute(), path
    assert ".." not in candidate.parts, path
    return str(candidate)


def _validate_lineage_inventory(payload: dict[str, object]) -> None:
    references = payload["input_hashes"]
    assert isinstance(references, list)
    paths = [reference["path"] for reference in references]
    assert all(isinstance(path, str) for path in paths)
    assert len(paths) == len(set(paths)), "duplicate recorded path"
    canonical = [_canonical_symbolic_path(path) for path in paths]
    assert len(canonical) == len(set(canonical)), "duplicate canonical target"
    assert set(paths) == set(MANDATORY_LINEAGE_PATHS), "exact set equality"


def _validate_comparison_manifest(
    identity_fragment: dict[str, object],
    evidence_fragment: dict[str, object],
) -> None:
    identities = identity_fragment["candidate_identities"]
    evidence = evidence_fragment["candidate_evidence"]
    assert isinstance(identities, list)
    assert isinstance(evidence, list)
    identity_pairs = [(item["version_id"], item["run_id"]) for item in identities]
    evidence_pairs = [(item["version_id"], item["run_id"]) for item in evidence]
    assert evidence_pairs == identity_pairs, "candidate evidence identity equality"
    assert set(evidence_fragment["evidence_hashes"]) == COMPARISON_OUTPUTS


def _producer_artifact_hash(name: str, data: bytes) -> str:
    if name in {"data_manifest.json", "metrics.json"}:
        payload = json.loads(data)
        if name == "metrics.json" and isinstance(payload, dict):
            payload.pop("run_id", None)
        data = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"sha256:{hashlib.sha256(data).hexdigest()[:16]}"


def _canonical_manifest_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _validate_all_manifest_entries(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    for name, recorded in manifest.items():
        if name == "schema_version":
            continue
        path = run_dir / name
        assert path.is_file(), f"missing manifest entry target: {name}"
        assert recorded == _producer_artifact_hash(name, path.read_bytes()), f"stale manifest entry: {name}"


def _historical_target(current: dict[str, object], handoff: dict[str, object]) -> tuple[str, str]:
    assert handoff["mode"] == "candidate_scoped_historical_rereview"
    guard = handoff["current_state_guard"]
    assert isinstance(guard, dict)
    assert guard["path"] == "current.json"
    assert guard["active_version"] == current["active_version"]
    assert FULL_SHA256.fullmatch(guard["sha256"])
    assert handoff["version_id"] != current["active_version"]
    return str(handoff["version_id"]), str(handoff["run_id"])


def test_unrefreshed_artifact_mutation_requires_full_contract_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "09_backtests" / "runA"
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_bytes(b'{"sharpe": 1.0}\n')
    legacy_artifacts = {
        "data_manifest.json": b'{"schema_version": 0}\n',
        "equity_curve.csv": b"date,equity\n2026-01-01,1.0\n",
        "trades.csv": b"date,symbol\n",
    }
    for name, content in legacy_artifacts.items():
        (run_dir / name).write_bytes(content)
    manifest = {
        "schema_version": 0,
        **{name: _producer_artifact_hash(name, content) for name, content in legacy_artifacts.items()},
        "metrics.json": _producer_artifact_hash("metrics.json", metrics_path.read_bytes()),
    }
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps(
                {
                    "run_id": "runA",
                    "artifact_hashes": _canonical_manifest_digest(manifest),
                    "artifact_inventory": {
                        "schema_version": 1,
                        "profile": "artifact_hashes_v0_legacy",
                    },
                }
        )
        + "\n",
        encoding="utf-8",
    )

    require_current_run_digest(run_dir)
    _validate_all_manifest_entries(run_dir)

    metrics_path.write_bytes(b'{"sharpe": 9.0}\n')

    assert _canonical_manifest_digest(manifest) == json.loads(
        (run_dir.parent / "run_digests.jsonl").read_text(encoding="utf-8")
    )["artifact_hashes"]
    with pytest.raises(RunDigestError, match="metrics.json hash mismatch"):
        require_current_run_digest(run_dir)
    with pytest.raises(AssertionError, match="stale manifest entry: metrics.json"):
        _validate_all_manifest_entries(run_dir)

    for path in (*REPORT_REVIEW_CONTRACTS, *LINEAGE_CONTRACTS, *FINAL_SELECTOR_CONTRACTS):
        normalized = _normalized(path)
        assert "full manifest-entry integrity validation" in normalized, path
        assert "is not the complete current-evidence gate" in normalized, path
        assert "callers must independently validate" in normalized, path
        assert "mutation without a manifest refresh" in normalized, path


def test_lineage_v2_inventory_is_exact_shared_and_not_producer_defined() -> None:
    schemas = []
    for path in (*LINEAGE_CONTRACTS, GOVERNANCE_DOC):
        schema = _example_with(path, "input_hashes")
        schemas.append(schema)
        _validate_lineage_inventory(schema)
        normalized = _normalized(path)
        assert "these nine paths are the complete mandatory inventory" in normalized, path
        assert "exact set equality" in normalized, path
        assert "duplicate canonical target" in normalized, path
        assert "regenerate" in normalized, path

    assert schemas[0] == schemas[1] == schemas[2]


@pytest.mark.parametrize(
    "mutation",
    ("omission", "addition", "duplicate_path", "duplicate_canonical", "wrong_run"),
)
def test_lineage_v2_inventory_negative_mutations_are_rejected(mutation: str) -> None:
    payload = copy.deepcopy(_example_with(LINEAGE_CONTRACTS[0], "input_hashes"))
    references = payload["input_hashes"]
    assert isinstance(references, list)

    if mutation == "omission":
        references.pop()
    elif mutation == "addition":
        references.append(
            {
                "path": "<phase_paths.09_backtests>/run_20260712_173012/metrics.json",
                "sha256": "sha256:" + "a" * 64,
            }
        )
    elif mutation == "duplicate_path":
        references[-1]["path"] = references[0]["path"]
    elif mutation == "duplicate_canonical":
        references[-1]["path"] = "<phase_paths.04_spec_build>/./strategy_spec.yaml"
        references.append(copy.deepcopy(references[1]))
        references.pop(1)
    else:
        references[-1]["path"] = "<phase_paths.10_reports>/run_other/report_review.json"

    with pytest.raises(AssertionError):
        _validate_lineage_inventory(payload)


def test_comparison_normative_example_is_complete_and_shared() -> None:
    identity_fragments = []
    evidence_fragments = []
    for path in (*COMPARISON_CONTRACTS, GOVERNANCE_DOC):
        identity = _example_with(path, "candidate_identities")
        evidence = _example_with(path, "evidence_hashes")
        identity_fragments.append(identity)
        evidence_fragments.append(evidence)
        _validate_comparison_manifest(identity, evidence)

        for candidate in evidence["candidate_evidence"]:
            assert set(candidate) == {"version_id", "run_id", "selected_run", "lineage_audit"}
            assert RUN_DIGEST.fullmatch(candidate["selected_run"]["digest"])
            assert FULL_SHA256.fullmatch(candidate["lineage_audit"]["sha256"])
        for name, reference in evidence["evidence_hashes"].items():
            references = reference if name == "figures" else [reference]
            assert references or name == "figures"
            assert all(FULL_SHA256.fullmatch(item["sha256"]) for item in references)

        normalized = _normalized(path)
        assert "identity-only manifest" in normalized, path
        assert "independently recompute" in normalized, path
        assert "regenerate the comparison" in normalized, path

    assert identity_fragments[0] == identity_fragments[1] == identity_fragments[2]
    assert evidence_fragments[0] == evidence_fragments[1] == evidence_fragments[2]


def test_comparison_negative_mutations_are_rejected() -> None:
    identity = copy.deepcopy(_example_with(COMPARISON_CONTRACTS[0], "candidate_identities"))
    evidence = copy.deepcopy(_example_with(COMPARISON_CONTRACTS[0], "evidence_hashes"))

    missing_non_selected = copy.deepcopy(evidence)
    missing_non_selected["candidate_evidence"].pop(0)
    with pytest.raises(AssertionError, match="candidate evidence identity equality"):
        _validate_comparison_manifest(identity, missing_non_selected)

    missing_output = copy.deepcopy(evidence)
    missing_output["evidence_hashes"].pop("spec_diff.yaml")
    with pytest.raises(AssertionError):
        _validate_comparison_manifest(identity, missing_output)


def test_final_selector_recursively_revalidates_every_comparison_candidate() -> None:
    for path in FINAL_SELECTOR_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "every `candidate_evidence` entry, including non-selected candidates",
            "complete lineage-v2 validator",
            "recursively revalidate the bound `report_review.json`",
            "full manifest-entry integrity validation",
            "do not trust `status: pass` or the lineage-audit file hash",
        ):
            assert phrase in normalized, (path, phrase)


def test_normative_run_digest_selection_requires_one_row_without_order_fallback() -> None:
    normalized = _normalized(GOVERNANCE_DOC)
    assert "exactly one valid matching `run_id` row" in normalized
    assert "zero 或 multiple matching rows" in normalized
    assert "do not use file order or choose a last matching row" in normalized
    assert "`require_current_run_digest()`" in normalized
    assert "按 file order 最后一个 matching `run_id` entry" not in normalized


def test_active_v002_can_rereview_historical_v001_without_reactivation() -> None:
    handoffs = []
    for path in HISTORICAL_HANDOFF_CONTRACTS:
        handoff = _example_with(path, "current_state_guard")
        handoffs.append(handoff)
        assert _historical_target({"active_version": "v002"}, handoff) == ("v001", "runA")

        normalized = _normalized(path)
        for phrase in (
            "candidate-scoped historical re-review",
            "must not change `current.json`",
            "does not reactivate `v001`",
            "rerun deterministic report qa",
            "rerun artifact lineage audit",
            "regenerate every comparison",
            "rerun final selection",
        ):
            assert phrase in normalized, (path, phrase)

    assert handoffs[0] == handoffs[1] == handoffs[2]

    wrong_guard = copy.deepcopy(handoffs[0])
    wrong_guard["current_state_guard"]["active_version"] = "v001"
    with pytest.raises(AssertionError):
        _historical_target({"active_version": "v002"}, wrong_guard)

    for path in (
        Path("agent/skills/manage-strategy-version/SKILL.md"),
        Path("agent/roles/oxq-version-manager-worker.md"),
        Path("agent/roles/oxq-coordinator.md"),
    ):
        normalized = _normalized(path)
        assert "historical re-review" in normalized, path
        assert "do not update `current.json`" in normalized, path
