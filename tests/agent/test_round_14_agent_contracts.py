from __future__ import annotations

import json
import re
from pathlib import Path

FINAL_SELECTOR_CONTRACTS = (
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
)
REPORT_REVIEW_CONTRACTS = (
    Path("agent/skills/review-research-report/SKILL.md"),
    Path("agent/roles/oxq-report-reviewer-worker.md"),
)
WORKSPACE_GOVERNOR_CONTRACTS = (
    Path("agent/skills/govern-research-workspace/SKILL.md"),
    Path("agent/roles/oxq-artifact-governor-worker.md"),
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


def test_final_decision_schema_is_exact_and_hash_bound() -> None:
    expected_fields = {
        "schema_version",
        "selection_id",
        "status",
        "selected_version_id",
        "selected_run_id",
        "selected_as",
        "hash_algorithm",
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
    expected_report_artifacts = {
        "research_report.md",
        "research_report.html",
        "writer_result.json",
    }
    schemas = []

    for path in FINAL_SELECTOR_CONTRACTS:
        schema = _example_with(path, "selected_run")
        schemas.append(schema)

        assert set(schema) == expected_fields, path
        assert schema["schema_version"] == 3, path
        assert schema["status"] == "selected", path
        assert schema["hash_algorithm"] == "sha256-file-bytes-v1", path

        selected_run = schema["selected_run"]
        assert isinstance(selected_run, dict), path
        assert set(selected_run) == {"path", "digest"}, path
        assert RUN_DIGEST.fullmatch(selected_run["digest"]), path

        report_artifacts = schema["report_artifacts"]
        assert isinstance(report_artifacts, dict), path
        assert set(report_artifacts) == expected_report_artifacts, path
        for artifact_name, reference in report_artifacts.items():
            assert set(reference) == {"path", "sha256"}, (path, artifact_name)
            assert reference["path"].endswith(f"/{artifact_name}"), (path, artifact_name)
            assert FULL_SHA256.fullmatch(reference["sha256"]), (path, artifact_name)

        for field in ("report_review", "selection_policy"):
            reference = schema[field]
            assert isinstance(reference, dict), (path, field)
            assert set(reference) == {"path", "sha256"}, (path, field)
            assert FULL_SHA256.fullmatch(reference["sha256"]), (path, field)

    assert schemas[0] == schemas[1]


def test_current_final_schema_binds_final_decision_bytes() -> None:
    expected_fields = {
        "schema_version",
        "selection_id",
        "selected_version_id",
        "selected_run_id",
        "final_decision",
    }
    schemas = []

    for path in FINAL_SELECTOR_CONTRACTS:
        schema = _example_with(path, "final_decision")
        schemas.append(schema)

        assert set(schema) == expected_fields, path
        assert schema["schema_version"] == 2, path
        final_decision = schema["final_decision"]
        assert isinstance(final_decision, dict), path
        assert set(final_decision) == {"path", "sha256"}, path
        assert final_decision["path"].endswith("/final_decision.json"), path
        assert FULL_SHA256.fullmatch(final_decision["sha256"]), path

    assert schemas[0] == schemas[1]


def test_final_selector_validates_hash_bindings_before_publication() -> None:
    for path in FINAL_SELECTOR_CONTRACTS:
        normalized = _normalized(path)

        assert "last matching `run_id` entry" in normalized, path
        assert "recompute the selected run digest" in normalized, path
        assert "recompute the full sha-256 over the exact current file bytes" in normalized, path
        assert "validate the complete in-memory `final_decision.json` payload before any write" in normalized, path
        assert "re-read the published `final_decision.json` bytes" in normalized, path
        assert "write `current_final.json` last" in normalized, path
        assert "do not update `current_final.json`" in normalized, path


def test_final_selection_migration_regenerates_instead_of_backfilling_hashes() -> None:
    for path in FINAL_SELECTOR_CONTRACTS:
        normalized = _normalized(path)

        assert "schema-version-1" in normalized, path
        assert "schema-version-2" in normalized, path
        assert "historical" in normalized, path
        assert "do not infer" in normalized or "never infer" in normalized, path
        assert "in place" in normalized, path
        assert "rerun final selection" in normalized, path
        assert "existing current pointer unchanged" in normalized or "existing pointer unchanged" in normalized, path


def test_report_review_result_invariants_are_exact_and_consistent() -> None:
    for path in REPORT_REVIEW_CONTRACTS:
        normalized = _normalized(path)

        assert "`status: pass` if and only if" in normalized, path
        assert "`verdict: consistent`" in normalized, path
        for field in ("blocking_findings", "required_report_edits", "errors"):
            assert f"`{field}` is empty" in normalized, (path, field)
        assert "`status: blocked` requires `verdict: needs_revision`" in normalized, path
        assert "`status: fail` requires `verdict: inconsistent`" in normalized, path
        assert "validate these field and cross-field invariants before writing" in normalized, path


def test_report_review_missing_artifact_maps_to_blocked_without_proven_inconsistency() -> None:
    for path in REPORT_REVIEW_CONTRACTS:
        normalized = _normalized(path)

        assert "an unavailable artifact by itself requires `status: blocked`" in normalized, path
        assert "independently proves a semantic inconsistency" in normalized, path


def test_report_review_migration_revalidates_schema_v1_cross_field_invariants() -> None:
    for path in REPORT_REVIEW_CONTRACTS:
        normalized = _normalized(path)

        assert "existing schema-version-1 report reviews are not grandfathered" in normalized, path
        assert "rerun semantic review" in normalized, path
        assert "contradictory" in normalized, path


def test_final_selector_rejects_contradictory_review_results_even_with_valid_hashes() -> None:
    contradictions = (
        "`status: pass` with `verdict: needs_revision` or `verdict: inconsistent`",
        "`status: pass` with non-empty `blocking_findings`",
        "`status: pass` with non-empty `required_report_edits`",
        "`status: pass` with non-empty `errors`",
    )

    for path in FINAL_SELECTOR_CONTRACTS:
        normalized = _normalized(path)

        assert "status exactly `pass`" in normalized, path
        assert "verdict exactly `consistent`" in normalized, path
        for field in ("blocking_findings", "required_report_edits", "errors"):
            assert f"`{field}` exactly empty" in normalized, (path, field)
        for contradiction in contradictions:
            assert contradiction in normalized, (path, contradiction)
        assert "even when every recorded hash matches" in normalized, path


def test_governor_inventory_schema_includes_direct_children_and_reference_sources() -> None:
    expected_fields = {
        "schema_version",
        "status",
        "blocking_findings",
        "warnings",
        "layout_version",
        "next_required_phase",
        "inventory_sources",
        "version_inventory",
    }
    expected_sources = {
        "lineage_version_ids",
        "selection_version_ids",
        "version_root_direct_children",
    }
    expected_inventory_fields = {
        "version_id",
        "path",
        "sources",
        "naming_status",
        "manifest_status",
        "reference_status",
        "findings",
    }
    schemas = []

    for path in WORKSPACE_GOVERNOR_CONTRACTS:
        schema = _example_with(path, "version_inventory")
        schemas.append(schema)

        assert set(schema) == expected_fields, path
        assert schema["schema_version"] == 2, path
        assert set(schema["inventory_sources"]) == expected_sources, path
        inventory = schema["version_inventory"]
        assert isinstance(inventory, list) and inventory, path
        assert all(set(item) == expected_inventory_fields for item in inventory), path
        assert any(item["sources"] == ["directory"] for item in inventory), path

    assert schemas[0] == schemas[1]


def test_governor_reports_orphan_and_unreferenced_direct_child_cases() -> None:
    expected_cases = (
        ("v003", "unreferenced_version", "valid matching manifest"),
        ("v004", "orphaned_version", "missing or invalid manifest"),
        ("draft", "invalid_version_directory_name", "does not match `^v[0-9]{3,}$`"),
    )

    for path in WORKSPACE_GOVERNOR_CONTRACTS:
        normalized = _normalized(path)

        assert "enumerate every direct child directory" in normalized, path
        assert "union" in normalized and "target_version_ids" in normalized, path
        assert "sort `target_version_ids` lexicographically" in normalized, path
        for version_id, finding, condition in expected_cases:
            assert version_id in normalized, (path, version_id)
            assert finding in normalized, (path, finding)
            assert condition in normalized, (path, condition)
        assert "never omit a direct child because lineage or selection artifacts do not reference it" in normalized, path


def test_governor_migration_regenerates_v2_inventory_from_live_sources() -> None:
    for path in WORKSPACE_GOVERNOR_CONTRACTS:
        normalized = _normalized(path)

        assert "schema-version-1 `workspace_audit.json` is historical" in normalized, path
        assert "rerun the governor" in normalized, path
        assert "live lineage, selection, and direct-child sources" in normalized, path
        assert "never synthesize `version_inventory` from the old audit" in normalized, path
