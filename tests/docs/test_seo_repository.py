from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
POSITIONING = (
    "open-xquant 是中文友好的 AI 量化研究框架，面向 AI Coding Agent "
    "和人类量化研究者，提供策略回测、因子研究、稳健性检验、"
    "审计报告与实盘交易工作流。"
)
PROJECT_DESCRIPTION = (
    "AI 量化研究框架 | Agentic Quant Research Kernel for reproducible "
    "and auditable research"
)
PROJECT_KEYWORDS = {
    "ai-quant",
    "quantitative-finance",
    "quant-research",
    "ai-agents",
    "backtesting",
    "factor-research",
    "algorithmic-trading",
}


def test_readme_leads_with_ai_quant_positioning() -> None:
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# open-xquant"
    assert lines[2] == POSITIONING
    assert "https://xingwudao.github.io/open-xquant/" in "\n".join(lines[:30])


def test_python_project_metadata_matches_positioning() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert project["description"] == PROJECT_DESCRIPTION
    assert set(project["keywords"]) == PROJECT_KEYWORDS


def test_baseline_records_reproducible_observations() -> None:
    text = (ROOT / "docs/seo/baseline-2026-09-02.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "## Repository Metadata",
        "## Search Samples",
        "## Inventory",
        "## Measurement Rules",
    ):
        assert heading in text
