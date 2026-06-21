from __future__ import annotations

import json
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


def test_opencode_quant_reporter_can_write_html_and_report_assets() -> None:
    config = json.loads(Path("agent/opencode/opencode.json").read_text(encoding="utf-8"))

    write_permissions = config["agents"]["quant-reporter"]["permissions"]["write"]

    assert "runs/*/research_report.md" in write_permissions
    assert "runs/*/research_report.html" in write_permissions
    assert "runs/*/report_assets/**" in write_permissions
