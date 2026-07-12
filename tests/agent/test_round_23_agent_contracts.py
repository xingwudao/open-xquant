from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from oxq.selection_lock import final_selection_lock_path, hold_final_selection_lock

CHART_SKILL = Path("agent/skills/build-report-charts/SKILL.md")
WRITER_SKILL = Path("agent/skills/write-research-report/SKILL.md")
REVIEWER_SKILL = Path("agent/skills/review-research-report/SKILL.md")
WRITER_ROLE = Path("agent/roles/oxq-report-writer-worker.md")
REVIEWER_ROLE = Path("agent/roles/oxq-report-reviewer-worker.md")
COMPARATOR_SKILL = Path("agent/skills/compare-strategy-versions/SKILL.md")
COMPARATOR_ROLE = Path("agent/roles/oxq-experiment-comparator-worker.md")
SELECTOR_SKILL = Path("agent/skills/select-final-version/SKILL.md")
SELECTOR_ROLE = Path("agent/roles/oxq-final-selector-worker.md")
COORDINATOR_ROLE = Path("agent/roles/oxq-coordinator.md")
ROUTER_SKILL = Path("agent/skills/open-xquant/SKILL.md")
AGENT_GUIDE = Path("docs/agent-guide.md")
ARCHITECTURE_DOC = Path("docs/architecture.md")
GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")

REPORT_PUBLISHER_CONTRACTS = (
    CHART_SKILL,
    WRITER_SKILL,
    REVIEWER_SKILL,
    WRITER_ROLE,
    REVIEWER_ROLE,
    AGENT_GUIDE,
    ARCHITECTURE_DOC,
    GOVERNANCE_DOC,
)
SELECTION_ID_CONTRACTS = (
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    GOVERNANCE_DOC,
)
SELECTION_COMPARISON_CONTRACTS = (
    COMPARATOR_SKILL,
    COMPARATOR_ROLE,
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COORDINATOR_ROLE,
    GOVERNANCE_DOC,
)
RESULT_PRODUCERS = (
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    GOVERNANCE_DOC,
)
SELECTION_ID_RE = re.compile(r"selection_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).lower().split())


def _allocate_selection(final_dir: Path, selection_id: str) -> Path:
    if SELECTION_ID_RE.fullmatch(selection_id) is None:
        raise ValueError("unsafe selection id")
    canonical_final = final_dir.resolve(strict=True)
    if final_dir.is_symlink() or canonical_final != final_dir:
        raise ValueError("unsafe final directory")
    candidate = final_dir / selection_id
    if candidate.parent.resolve(strict=True) != canonical_final:
        raise ValueError("selection is not a direct child")
    os.mkdir(candidate, mode=0o700)
    return candidate


def test_report_contracts_use_one_runtime_batch_publisher() -> None:
    for path in REPORT_PUBLISHER_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "`publish_report_artifacts(report_dir, artifacts, *, lock_subject=none)`",
            "safe relative keys",
            "complete `bytes`",
            "`none` deletes",
            "callable builder executes under the final-selection lock",
            "baseline check",
            "atomic all-or-rollback batch",
            "`lock_subject=source_run_dir`",
            "`run_digest_transaction(source_run_dir)`",
            "run lock first and the final-selection lock second",
        ):
            assert phrase in normalized, (path, phrase)

    assert "oxq report asset add" not in _text(CHART_SKILL)
    assert ".write_text(" not in _text(WRITER_SKILL)


def test_report_publisher_lock_subject_blocks_external_export(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "versions/v001/09_backtests/run_001"
    report_dir = tmp_path / "external-report"
    config_dir = workspace / ".open-xquant"
    config_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
        encoding="utf-8",
    )
    report_dir.mkdir()
    report = report_dir / "research_report.md"
    report.write_bytes(b"old\n")
    attempted = tmp_path / "publisher-attempted"
    lock_path = final_selection_lock_path(run_dir)
    assert lock_path is not None

    script = """
import sys
from pathlib import Path
from oxq.report.assets import publish_report_artifacts

report_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
attempted = Path(sys.argv[3])
attempted.write_bytes(b"ready")
publish_report_artifacts(
    report_dir,
    {"research_report.md": b"new\\n"},
    lock_subject=run_dir,
)
"""
    with hold_final_selection_lock(lock_path):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(report_dir), str(run_dir), str(attempted)],
            env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
        )
        deadline = time.monotonic() + 5
        while not attempted.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert attempted.exists()
        time.sleep(0.1)
        assert process.poll() is None
        assert report.read_bytes() == b"old\n"

    assert process.wait(timeout=5) == 0
    assert report.read_bytes() == b"new\n"


@pytest.mark.parametrize(
    "selection_id",
    (
        "",
        ".",
        "..",
        "selection_",
        "selection_../escape",
        "selection_a/b",
        r"selection_a\b",
        "/selection_abs",
        r"C:\selection_abs",
        "selection_a.b",
        "selection_alias/../selection_ok",
        "selection_\u4e2d\u6587",
        "selection_" + "a" * 65,
    ),
)
def test_selection_id_rejects_traversal_alias_and_out_of_grammar_forms(
    tmp_path: Path,
    selection_id: str,
) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    with pytest.raises((ValueError, OSError)):
        _allocate_selection(final_dir, selection_id)


def test_selection_id_allocation_is_exclusive_and_rejects_symlink_parent(
    tmp_path: Path,
) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    allocated = _allocate_selection(final_dir, "selection_20260712_180000")
    assert allocated.parent == final_dir
    with pytest.raises(FileExistsError):
        _allocate_selection(final_dir, allocated.name)

    real_final = tmp_path / "real-final"
    alias_final = tmp_path / "alias-final"
    real_final.mkdir()
    alias_final.symlink_to(real_final, target_is_directory=True)
    with pytest.raises(ValueError):
        _allocate_selection(alias_final, "selection_alias")


def test_selection_id_contract_is_one_bounded_direct_child_grammar() -> None:
    for path in SELECTION_ID_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            r"`\aselection_[a-za-z0-9][a-za-z0-9_-]{0,63}\z`",
            "generated and provided ids use the same grammar",
            "one normal direct-child component",
            "resolved parent must equal the canonical `<final_dir>` exactly",
            "no symlink parent",
            "exclusive atomic `mkdir`",
            "provided collision",
            "generated collision",
        ):
            assert phrase in normalized, (path, phrase)


def test_same_population_selections_and_failed_retry_are_immutable(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    comparisons_dir = tmp_path / "comparisons"
    final_dir.mkdir()
    comparisons_dir.mkdir()
    prior_pointer = b'{"selection_id":"selection_first"}\n'
    pointer = final_dir / "current_final.json"
    pointer.write_bytes(prior_pointer)

    first = _allocate_selection(final_dir, "selection_first")
    first_comparison = comparisons_dir / first.name / "cmp_population"
    first_comparison.mkdir(parents=True)
    first_manifest = first_comparison / "comparison_manifest.json"
    first_manifest.write_bytes(b'{"schema_version":2,"selection_id":"selection_first"}\n')
    first_bytes = first_manifest.read_bytes()

    second = _allocate_selection(final_dir, "selection_second")
    second_comparison = comparisons_dir / second.name / "cmp_population"
    second_comparison.mkdir(parents=True)
    (second_comparison / "comparison_manifest.json").write_bytes(b"partial\n")
    # The second selection fails before pointer publication.

    assert first != second
    assert first_manifest.read_bytes() == first_bytes
    assert pointer.read_bytes() == prior_pointer
    with pytest.raises(FileExistsError):
        second_comparison.mkdir(parents=True)


def test_final_selection_comparison_v2_paths_are_selection_scoped() -> None:
    scoped_root = "<comparisons_dir>/<selection_id>/<comparison_id>/"
    scoped_example = (
        "<comparisons_dir>/selection_20260712_180000/"
        "cmp_v001_runA_vs_v002_runB/comparison_manifest.json"
    )
    for path in SELECTION_COMPARISON_CONTRACTS:
        text = _text(path)
        normalized = _normalized(path)
        assert scoped_root in text, path
        assert scoped_example in text, path
        comparison_schema_phrase = (
            "schema-version-3 comparison manifest"
            if path == COORDINATOR_ROLE
            else "schema-version-2 comparison manifest"
        )
        for phrase in (
            "reject an existing output directory",
            "never overwrite",
            "immutable selection-scoped directory",
            "prior `current_final.json`",
            "retry uses a fresh `comparison_id`",
            "same `selection_id`",
            comparison_schema_phrase,
            "hash exact final bytes",
        ):
            assert phrase in normalized, (path, phrase)


def test_current_selector_result_producers_reject_obsolete_schema_two() -> None:
    stale = re.compile(r"schema-version-1 selector result", flags=re.IGNORECASE)
    for path in RESULT_PRODUCERS:
        text = _text(path)
        normalized = _normalized(path)
        assert stale.search(text) is None, path
        phrases = (
            "`restart_selection` allocates a new selection id",
            "must not reuse or overwrite the failed selection directory",
        )
        if path in (SELECTOR_SKILL, SELECTOR_ROLE):
            phrases += (
                "the schema-version-2 selector result envelope above is a historical",
                "current selector result handoffs use schema version 3",
                "selected result publishes the schema-version-5",
            )
        elif path in (ROUTER_SKILL, COORDINATOR_ROLE):
            phrases += ("every current selector result uses schema version 3",)
        else:
            phrases += ("every current selector result uses schema version 2",)
        for phrase in phrases:
            assert phrase in normalized, (path, phrase)
