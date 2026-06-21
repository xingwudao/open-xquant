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
    assert "research_report.md" in text
    assert "research_report.html" in text
    assert "research-report-writer" in text
    assert "oxq report write" not in text
    assert "report_evidence.md" not in text
    assert "Do not modify metrics" in text
    assert "Do not modify audit" in text


def test_opencode_quant_reporter_can_write_html_and_report_assets() -> None:
    config = json.loads(Path("agent/opencode/opencode.json").read_text(encoding="utf-8"))

    write_permissions = config["agents"]["quant-reporter"]["permissions"]["write"]

    assert "runs/*/research_report.md" in write_permissions
    assert "runs/*/research_report.html" in write_permissions
    assert "runs/*/report_assets/**" in write_permissions
    assert "runs/*/report_evidence.md" not in write_permissions
    assert "report_write" not in config["agents"]["quant-reporter"]["tools"]


def test_research_report_writer_skill_requires_agent_authored_final_report() -> None:
    skill = Path("agent/skills/research-report-writer.md")

    text = skill.read_text(encoding="utf-8")

    assert "research-report-writer" in text
    assert "research_report.md" in text
    assert "research_report.html" in text
    assert "render_markdown_html_report" in text
    assert "human researcher" in text
    assert "potential investor" in text
    assert "report_evidence.md" not in text
    assert "oxq report write" not in text
    assert "Do not invent evidence" in text


def test_quant_reporter_routes_final_report_through_writer_skill() -> None:
    reporter = Path("agent/opencode/agents/quant-reporter.md").read_text(encoding="utf-8")
    command = Path("agent/opencode/commands/quant-report.md").read_text(encoding="utf-8")

    combined = reporter + "\n" + command

    assert "research-report-writer" in combined
    assert "render_markdown_html_report" in combined
    assert "report_evidence.md" not in combined
    assert "oxq report write" not in combined
