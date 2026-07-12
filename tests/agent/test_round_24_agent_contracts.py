from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SELECTOR_SKILL = Path("agent/skills/select-final-version/SKILL.md")
SELECTOR_ROLE = Path("agent/roles/oxq-final-selector-worker.md")
COORDINATOR_ROLE = Path("agent/roles/oxq-coordinator.md")
ROUTER_SKILL = Path("agent/skills/open-xquant/SKILL.md")
GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
CHART_SKILL = Path("agent/skills/build-report-charts/SKILL.md")
WRITER_SKILL = Path("agent/skills/write-research-report/SKILL.md")
WRITER_ROLE = Path("agent/roles/oxq-report-writer-worker.md")

SELECTOR_PRODUCER_CONTRACTS = (
    SELECTOR_SKILL,
    SELECTOR_ROLE,
    GOVERNANCE_DOC,
)
SELECTION_ROUTING_CONTRACTS = (
    COORDINATOR_ROLE,
    ROUTER_SKILL,
    GOVERNANCE_DOC,
)


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
    result = next(
        (
            item
            for item in _json_examples(path)
            if isinstance(item, dict)
            and all(item.get(key) == value for key, value in fields.items())
        ),
        None,
    )
    assert result is not None, (path, fields)
    return result


def _comparison_ref_arrays(path: Path) -> list[list[dict[str, str]]]:
    return [
        item
        for item in _json_examples(path)
        if isinstance(item, list)
        and all(
            isinstance(reference, dict)
            and set(reference) == {"path", "sha256"}
            for reference in item
        )
    ]


def test_final_selector_is_the_locked_comparison_refs_producer() -> None:
    for path in SELECTOR_PRODUCER_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "the final selector is the sole producer of `comparison_refs.json`",
            "existing selection directory",
            "under `final-selection.lock`",
            "safe workspace-relative direct regular file",
            "same-directory temporary regular file",
            "flush and `fsync` the temporary file",
            "atomic `os.replace`",
            "`fsync` the selection directory",
            "the coordinator and router never write `comparison_refs.json`",
        ):
            assert phrase in normalized, (path, phrase)

    for path in SELECTION_ROUTING_CONTRACTS:
        normalized = _normalized(path)
        assert "the final selector is the sole producer of `comparison_refs.json`" in normalized, path
        assert "must not write `comparison_refs.json`" in normalized, path


def test_comparison_refs_ledger_examples_cover_singleton_and_multi_candidate_flows() -> None:
    expected_multi: list[dict[str, str]] | None = None

    for path in SELECTOR_PRODUCER_CONTRACTS:
        arrays = _comparison_ref_arrays(path)
        assert arrays[0] == [], path
        assert len(arrays) >= 2, path

        persisted_multi = arrays[1]
        resume = _json_object(path, mode="resume_selection")
        decision = _json_object(path, schema_version=4, status="selected")
        assert persisted_multi == resume["comparison_refs"], path
        assert persisted_multi == decision["comparison_refs"], path

        if expected_multi is None:
            expected_multi = persisted_multi
        assert persisted_multi == expected_multi, path

        normalized = _normalized(path)
        for phrase in (
            "persist the literal utf-8 bytes `[]`",
            "before returning `candidate_set_ready`",
            "validated `comparison_ready` results",
            "exact ordered array",
            "same `selection_id` and exact candidate-set reference",
            "before ranking or writing `final_decision.json`",
        ):
            assert phrase in normalized, (path, phrase)


def test_comparison_refs_resume_decision_and_pointer_bind_one_persisted_array() -> None:
    for path in SELECTOR_PRODUCER_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "`resume_selection` must re-read",
            "request `comparison_refs` to equal the persisted array exactly",
            "`final_decision.comparison_refs` must equal that same persisted array exactly",
            "pointer-time validation re-reads `comparison_refs.json`",
            "transitively binds the same array",
            "hold the same lock continuously through decision and pointer publication",
            "idempotent no-op",
            "stale or different array",
            "must not overwrite",
            "must not allocate a new selection id",
        ):
            assert phrase in normalized, (path, phrase)

    for path in SELECTION_ROUTING_CONTRACTS:
        normalized = _normalized(path)
        for phrase in (
            "comparator dispatch order, not completion order",
            "without alteration",
            "same `selection_id` and exact candidate-set reference",
            "must not write `comparison_refs.json`",
        ):
            assert phrase in normalized, (path, phrase)


def test_report_asset_manifest_paths_are_package_relative() -> None:
    examples = [
        item
        for item in _json_examples(CHART_SKILL)
        if isinstance(item, dict) and isinstance(item.get("assets"), list)
    ]
    assert len(examples) == 1

    assets = examples[0]["assets"]
    assert [asset["path"] for asset in assets] == [
        "figures/equity_curve.png",
        "figures/drawdown.png",
        "figures/trade_curve.png",
    ]
    assert {asset["source"]["script"] for asset in assets} == {
        "scripts/plot_report_charts.py"
    }

    text = _text(CHART_SKILL)
    normalized = _normalized(CHART_SKILL)
    assert '"path": "report_assets/figures/' not in text
    for phrase in (
        "manifest asset `path` values are relative to the `report_assets/manifest.json` package",
        "`figures/<name>.png`",
        "`source.script` values use `scripts/<name>.py`",
        "never `report_assets/figures/<name>.png`",
    ):
        assert phrase in normalized


def test_report_publisher_keys_and_report_image_urls_remain_report_relative() -> None:
    chart = _normalized(CHART_SKILL)
    for phrase in (
        "publisher mapping keys remain relative to the report directory",
        "`report_assets/manifest.json`",
        "`report_assets/figures/<name>.png`",
        "`report_assets/scripts/<name>.py`",
        "report image url remains `report_assets/figures/<name>.png`",
    ):
        assert phrase in chart, phrase

    for path in (WRITER_SKILL, WRITER_ROLE):
        normalized = _normalized(path)
        for phrase in (
            "manifest path `figures/<name>.png`",
            "report image url `report_assets/figures/<name>.png`",
            "do not embed the package-relative manifest path directly",
        ):
            assert phrase in normalized, (path, phrase)
