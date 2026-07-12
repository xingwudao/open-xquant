from __future__ import annotations

import json
import re
from pathlib import Path

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
FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
RUN_DIGEST = re.compile(r"sha256:[0-9a-f]{16}")


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


def test_lineage_v2_has_one_closed_candidate_derived_input_set() -> None:
    selector = _normalized(FINAL_SELECTOR_CONTRACTS[0])

    for path in LINEAGE_CONTRACTS:
        lineage = _normalized(path)
        for phrase in (
            "mandatory logical inputs",
            "exact set equality",
            "reject omissions",
            "wrong phase",
            "wrong run",
            "duplicate recorded path or duplicate canonical target",
        ):
            assert phrase in lineage, (path, phrase)

    for phrase in (
        "derive the mandatory lineage input paths independently",
        "before trusting `input_hashes`",
        "exact set equality",
        "reject omissions",
        "wrong phase",
        "wrong run",
        "duplicate recorded path or duplicate canonical target",
        "recompute sha-256 over every mandatory current input",
    ):
        assert phrase in selector, phrase


def test_incomplete_lineage_v2_is_regenerated_not_backfilled() -> None:
    for path in LINEAGE_CONTRACTS:
        normalized = _normalized(path)
        assert "incomplete schema-version-2 lineage audit" in normalized, path
        assert "do not append missing hashes" in normalized, path
        assert "rerun artifact lineage audit" in normalized, path


def test_report_review_binds_current_run_and_external_decision_inputs() -> None:
    expected_inputs = {
        "strategy_spec.yaml",
        "spec_audit.json",
        "compiled_plan.json",
        "runtime_audit.json",
        "report_assets/manifest.json",
    }

    schemas = []
    for path in REPORT_REVIEW_CONTRACTS:
        schema = _example_with(path, "reviewed_artifacts")
        schemas.append(schema)
        assert set(schema["source_run"]) == {"path", "digest"}, path
        assert schema["source_run"]["path"].endswith("/<run_id>"), path
        assert RUN_DIGEST.fullmatch(schema["source_run"]["digest"]), path
        decision_inputs = schema["decision_inputs"]
        assert set(decision_inputs) == expected_inputs, path
        for name, reference in decision_inputs.items():
            assert set(reference) == {"path", "sha256"}, (path, name)
            assert reference["path"].endswith(f"/{name}"), (path, name)
            assert FULL_SHA256.fullmatch(reference["sha256"]), (path, name)

        normalized = _normalized(path)
        assert "require exactly one valid matching `run_id` row" in normalized, path
        assert "canonical current run digest" in normalized, path
        assert "registered figure and source-script hashes" in normalized, path
        assert "robustness mutation followed by an integrity refresh" in normalized, path

    assert schemas[0] == schemas[1]


def test_final_selector_revalidates_report_review_transitive_evidence() -> None:
    normalized = _normalized(FINAL_SELECTOR_CONTRACTS[0])

    for phrase in (
        "recompute the report review's canonical current run digest",
        "recompute every `decision_inputs` file hash",
        "revalidate every registered figure and source-script hash",
        "robustness mutation followed by an integrity refresh invalidates the old review",
    ):
        assert phrase in normalized, phrase


def test_comparison_manifest_binds_all_inputs_outputs_and_figures() -> None:
    required_outputs = {
        "comparability_audit.json",
        "metrics_comparison.json",
        "spec_diff.yaml",
        "comparison_report.md",
        "figures",
    }

    fragments = []
    for path in COMPARISON_CONTRACTS:
        fragment = _example_with(path, "evidence_hashes")
        fragments.append(fragment)
        assert fragment["hash_algorithm"] == "sha256-file-bytes-v1", path
        candidate_evidence = fragment["candidate_evidence"]
        assert isinstance(candidate_evidence, list) and len(candidate_evidence) >= 2, path
        for candidate in candidate_evidence:
            assert set(candidate) == {"version_id", "run_id", "selected_run", "lineage_audit"}, path
            assert set(candidate["selected_run"]) == {"path", "digest"}, path
            assert RUN_DIGEST.fullmatch(candidate["selected_run"]["digest"]), path
            assert set(candidate["lineage_audit"]) == {"path", "sha256"}, path
            assert FULL_SHA256.fullmatch(candidate["lineage_audit"]["sha256"]), path

        evidence_hashes = fragment["evidence_hashes"]
        assert set(evidence_hashes) == required_outputs, path
        for name in required_outputs - {"figures"}:
            reference = evidence_hashes[name]
            assert set(reference) == {"path", "sha256"}, (path, name)
            assert reference["path"].endswith(f"/{name}"), (path, name)
            assert FULL_SHA256.fullmatch(reference["sha256"]), (path, name)
        assert isinstance(evidence_hashes["figures"], list) and evidence_hashes["figures"], path
        assert all(set(reference) == {"path", "sha256"} for reference in evidence_hashes["figures"]), path

    assert fragments[0] == fragments[1]


def test_legacy_comparison_v1_is_regenerated_not_backfilled() -> None:
    for path in COMPARISON_CONTRACTS:
        normalized = _normalized(path)
        assert "schema-version-1 comparison manifest" in normalized, path
        assert "do not backfill" in normalized, path
        assert "regenerate the comparison" in normalized, path

    selector = _normalized(FINAL_SELECTOR_CONTRACTS[0])
    assert "reject a legacy comparison manifest" in selector
    assert "regenerate the comparison" in selector


def test_final_selector_recomputes_comparison_manifest_transitive_evidence() -> None:
    normalized = _normalized(FINAL_SELECTOR_CONTRACTS[0])

    for phrase in (
        "recompute every candidate's canonical current run digest",
        "recompute every candidate lineage-audit file hash",
        "recompute all four required comparison output hashes",
        "exact current regular-file set under `figures/`",
        "reject omitted, extra, duplicate, stale, or non-candidate evidence",
    ):
        assert phrase in normalized, phrase


def test_final_selector_uses_require_current_run_digest_row_cardinality() -> None:
    for path in FINAL_SELECTOR_CONTRACTS:
        normalized = _normalized(path)
        assert "require exactly one valid matching `run_id` row" in normalized, path
        assert "do not use file order or choose a last matching row" in normalized, path
        assert "`require_current_run_digest()`" in normalized, path
