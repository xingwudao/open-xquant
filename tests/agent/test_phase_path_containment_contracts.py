from __future__ import annotations

import json
import re
from pathlib import Path

PHASE_CONSUMING_SKILLS = {
    Path("agent/skills/audit-artifact-lineage/SKILL.md"),
    Path("agent/skills/audit-runtime-semantics/SKILL.md"),
    Path("agent/skills/audit-strategy-idea/SKILL.md"),
    Path("agent/skills/audit-strategy-spec/SKILL.md"),
    Path("agent/skills/author-component/SKILL.md"),
    Path("agent/skills/brainstorm-strategy-idea/SKILL.md"),
    Path("agent/skills/build-report-charts/SKILL.md"),
    Path("agent/skills/build-strategy-spec/SKILL.md"),
    Path("agent/skills/compare-strategy-versions/SKILL.md"),
    Path("agent/skills/explore-data/SKILL.md"),
    Path("agent/skills/govern-research-workspace/SKILL.md"),
    Path("agent/skills/manage-strategy-version/SKILL.md"),
    Path("agent/skills/monitor-strategy-run/SKILL.md"),
    Path("agent/skills/review-performance/SKILL.md"),
    Path("agent/skills/review-research-report/SKILL.md"),
    Path("agent/skills/run-authorized-backtest/SKILL.md"),
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/skills/write-research-report/SKILL.md"),
}

PHASE_CONSUMING_ROLES = {
    Path("agent/roles/oxq-artifact-governor-worker.md"),
    Path("agent/roles/oxq-component-author-worker.md"),
    Path("agent/roles/oxq-coordinator.md"),
    Path("agent/roles/oxq-data-inspection-worker.md"),
    Path("agent/roles/oxq-experiment-comparator-worker.md"),
    Path("agent/roles/oxq-lineage-auditor-worker.md"),
    Path("agent/roles/oxq-monitor-worker.md"),
    Path("agent/roles/oxq-report-reviewer-worker.md"),
    Path("agent/roles/oxq-report-writer-worker.md"),
    Path("agent/roles/oxq-runner-worker.md"),
    Path("agent/roles/oxq-runtime-auditor-worker.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
    Path("agent/roles/oxq-spec-auditor-worker.md"),
    Path("agent/roles/oxq-strategy-brainstorm-worker.md"),
    Path("agent/roles/oxq-strategy-builder-worker.md"),
    Path("agent/roles/oxq-strategy-idea-auditor-worker.md"),
    Path("agent/roles/oxq-version-manager-worker.md"),
}

PHASE_CONSUMER_CONTRACTS = PHASE_CONSUMING_SKILLS | PHASE_CONSUMING_ROLES

CROSS_VERSION_INSPECTION_CONTRACTS = {
    Path("agent/skills/audit-artifact-lineage/SKILL.md"),
    Path("agent/skills/compare-strategy-versions/SKILL.md"),
    Path("agent/skills/govern-research-workspace/SKILL.md"),
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/roles/oxq-artifact-governor-worker.md"),
    Path("agent/roles/oxq-experiment-comparator-worker.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
    Path("agent/roles/oxq-lineage-auditor-worker.md"),
}

WORKSPACE_GOVERNOR_CONTRACTS = {
    Path("agent/skills/govern-research-workspace/SKILL.md"),
    Path("agent/roles/oxq-artifact-governor-worker.md"),
}

REPORT_REVIEW_CONTRACTS = {
    Path("agent/skills/review-research-report/SKILL.md"),
    Path("agent/roles/oxq-report-reviewer-worker.md"),
}

FINAL_SELECTOR_CONTRACTS = {
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).lower().split())


def test_phase_consumer_inventory_is_explicit_and_complete() -> None:
    discovered_skills = {path for path in Path("agent/skills").glob("*/SKILL.md") if "phase_paths" in _text(path)}
    discovered_roles = {path for path in Path("agent/roles").glob("*.md") if "phase_paths" in _text(path)}

    assert discovered_skills == PHASE_CONSUMING_SKILLS
    assert discovered_roles == PHASE_CONSUMING_ROLES

    assert Path("agent/skills/compare-strategy-versions/SKILL.md") in PHASE_CONSUMING_SKILLS
    assert Path("agent/roles/oxq-experiment-comparator-worker.md") in PHASE_CONSUMING_ROLES


def test_every_phase_consumer_requires_identity_and_containment_preflight() -> None:
    for path in sorted(PHASE_CONSUMER_CONTRACTS):
        text = _text(path)
        normalized = _normalized(path)

        assert "Phase Path Containment Preflight" in text, path
        assert "current.json" in text and "active_version" in text, path
        assert "expected_version_id" in text, path
        assert re.search(
            r"manifest(?:'s|\.)? `?version_id`?.{0,100}must equal.{0,100}"
            r"`?expected_version_id`?",
            normalized,
        ), path
        assert "workspace-relative" in normalized, path
        assert "absolute" in normalized, path
        assert "`..`" in normalized and "path segment" in normalized, path
        assert "symlink" in normalized and "escape" in normalized, path
        assert "canonical" in normalized and "intended version directory" in normalized, path
        assert re.search(
            r"intended version directory.{0,160}"
            r"(?:inside|under|descendant of).{0,80}canonical version root",
            normalized,
        ), path
        assert "before" in normalized, path
        for operation in ("read", "write", "directory creation", "command", "handoff"):
            assert operation in normalized, (path, operation)
        assert "block" in normalized, path


def test_phase_preflight_contracts_cover_unsafe_and_safe_examples() -> None:
    rejected_examples = (
        "strategy_store/v001/../v002/04_spec_build",
        "strategy_store/v002/04_spec_build",
        "/tmp/04_spec_build",
        "strategy_store/v001/escape/04_spec_build",
    )
    accepted_example = "strategy_store/v001/custom/phases/04_spec_build"

    for path in sorted(PHASE_CONSUMER_CONTRACTS):
        text = _text(path)
        normalized = _normalized(path)

        for example in rejected_examples:
            assert example in text, (path, example)
        assert "v001" in text and "v002" in text, path
        assert "symlink" in normalized and "outside" in normalized, path
        assert accepted_example in text, path
        assert re.search(r"allow(?:ed)? .{0,80}custom nested", normalized), path


def test_cross_version_and_bootstrap_exceptions_remain_identity_bound() -> None:
    for path in sorted(PHASE_CONSUMER_CONTRACTS):
        normalized = _normalized(path)

        assert "explicitly owns cross-version inspection" in normalized, path
        assert "referenced version id" in normalized, path
        assert "new-version bootstrap" in normalized, path
        assert "before publishing" in normalized, path


def test_cross_version_owner_inventory_explicitly_includes_workspace_governor() -> None:
    assert WORKSPACE_GOVERNOR_CONTRACTS <= CROSS_VERSION_INSPECTION_CONTRACTS
    assert CROSS_VERSION_INSPECTION_CONTRACTS <= PHASE_CONSUMER_CONTRACTS


def test_workspace_governor_resolves_each_historical_version_from_its_own_manifest() -> None:
    for path in sorted(WORKSPACE_GOVERNOR_CONTRACTS):
        text = _text(path)
        normalized = _normalized(path)

        assert "target_version_ids" in text, path
        assert "governed lineage and selection scope" in normalized, path
        assert "before comparison or migration" in normalized, path
        assert "<version_root>/<target_version_id>/version_manifest.json" in text, path
        assert "manifest `version_id` must equal `target_version_id`" in normalized, path
        assert "for every target version" in normalized, path
        assert "exact `phase_paths`" in normalized, path
        assert "workspace-relative" in normalized and "absolute" in normalized, path
        assert "`..`" in normalized and "path segment" in normalized, path
        assert "canonical" in normalized and "target version directory" in normalized, path
        assert "cross-version" in normalized, path
        assert "symlink" in normalized and "escape" in normalized, path
        assert "per-version resolved phase paths" in normalized, path
        assert "read-only" in normalized, path
        assert "<governance_dir>/workspace_audit" in text, path


def test_comparator_candidates_are_exact_direct_runs_from_their_claimed_version_manifest() -> None:
    comparator_contracts = (
        Path("agent/skills/compare-strategy-versions/SKILL.md"),
        Path("agent/roles/oxq-experiment-comparator-worker.md"),
    )
    rejected_examples = (
        "legacy_root/v001/09_backtests/run_001",
        "research_store/v002/09_backtests/run_001",
        "research_store/v001/09_backtests/nested/run_001",
        "research_store/v001/09_backtests/escape/run_001",
    )

    for path in comparator_contracts:
        text = _text(path)
        normalized = _normalized(path)

        assert "<version_root>/<candidate.version_id>/version_manifest.json" in text, path
        assert "manifest `version_id` must equal `candidate.version_id`" in normalized, path
        assert "phase_paths.09_backtests" in text, path
        assert "do not fall back" in normalized, path
        assert "exactly one direct run directory" in normalized, path
        assert "resolved_run.parent == resolved_backtest_phase" in text, path
        assert "resolved_run.name == candidate.run_id" in text, path
        for example in rejected_examples:
            assert example in text, (path, example)


def test_report_review_contract_has_deterministic_candidate_identity_schema() -> None:
    required_top_level = {
        "schema_version",
        "version_id",
        "run_id",
        "hash_algorithm",
        "status",
        "verdict",
        "findings",
        "blocking_findings",
        "required_report_edits",
        "reviewed_artifacts",
        "errors",
    }
    required_artifacts = {
        "research_report.md",
        "research_report.html",
        "writer_result.json",
        "metrics.json",
    }

    for path in sorted(REPORT_REVIEW_CONTRACTS):
        examples = [
            json.loads(block)
            for block in re.findall(r"```json\n(.*?)\n```", _text(path), flags=re.DOTALL)
        ]
        schema = next(
            (item for item in examples if isinstance(item, dict) and "reviewed_artifacts" in item),
            None,
        )

        assert schema is not None, path
        assert required_top_level <= set(schema), path
        assert schema["schema_version"] == 1, path
        assert schema["hash_algorithm"] == "sha256-file-bytes-v1", path
        assert isinstance(schema["version_id"], str) and schema["version_id"], path
        assert isinstance(schema["run_id"], str) and schema["run_id"], path
        assert isinstance(schema["reviewed_artifacts"], dict), path
        assert set(schema["reviewed_artifacts"]) == required_artifacts, path
        for artifact_name, identity in schema["reviewed_artifacts"].items():
            assert set(identity) == {"path", "sha256"}, (path, artifact_name)
            assert identity["path"].endswith(f"/{artifact_name}"), (path, artifact_name)
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", identity["sha256"]), (path, artifact_name)


def test_final_selector_inventory_and_candidate_resolution_are_cross_version_bound() -> None:
    assert FINAL_SELECTOR_CONTRACTS <= PHASE_CONSUMER_CONTRACTS
    assert FINAL_SELECTOR_CONTRACTS <= CROSS_VERSION_INSPECTION_CONTRACTS

    rejected_cases = (
        "legacy_root/v001/09_backtests/run_001",
        "research_store/v002/custom/backtests/run_001",
        "research_store/v001/custom/backtests/nested/run_001",
        "research_store/v001/custom/reports/run_001/copied_report_review.json",
    )
    accepted_case = "research_store/v001/custom/reports/run_001/report_review.json"

    for path in sorted(FINAL_SELECTOR_CONTRACTS):
        text = _text(path)
        normalized = _normalized(path)

        assert "<version_root>/<candidate.version_id>/version_manifest.json" in text, path
        assert "manifest `version_id` must equal `candidate.version_id`" in normalized, path
        for phase in ("phase_paths.09_backtests", "phase_paths.10_reports"):
            assert phase in text, (path, phase)
        assert "resolved_run.parent == resolved_backtest_phase" in text, path
        assert "resolved_run.name == candidate.run_id" in text, path
        assert "resolved_report.parent == resolved_report_phase" in text, path
        assert "resolved_report.name == candidate.run_id" in text, path
        assert "metrics.json" in text and "run_id" in text and "strategy_id" in text, path
        assert "report_review.json" in text and "reviewed_artifacts" in text, path
        assert "sha256-file-bytes-v1" in text, path
        assert "copied" in normalized and "stale" in normalized and "cross-version" in normalized, path
        for rejected_case in rejected_cases:
            assert rejected_case in text, (path, rejected_case)
        assert accepted_case in text, path
