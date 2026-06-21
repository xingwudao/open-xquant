from __future__ import annotations

from pathlib import Path


def test_report_chart_builder_skill_documents_chart_asset_workflow() -> None:
    skill = Path("agent/skills/report-chart-builder.md")

    text = skill.read_text(encoding="utf-8")

    assert "report-chart-builder" in text
    assert "discuss chart requirements" in text
    assert "plotting Python" in text
    assert "report_assets/figures" in text
    assert "report_assets/scripts" in text
    assert "oxq report asset add" in text
    assert "oxq report write" in text
    assert "research_report.md" in text
    assert "research_report.html" in text
    assert "Do not modify metrics" in text
    assert "Do not modify audit" in text
