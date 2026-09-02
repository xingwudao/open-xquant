from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
PAGES = {
    "website/index.md": ("AI 量化框架 | open-xquant", "AI 量化研究框架"),
    "website/guide/ai-quant-framework.md": ("什么是 AI 量化框架 | open-xquant", "AI 量化框架"),
    "website/guide/agentic-quant-research.md": ("AI Agent 量化研究 | open-xquant", "AI Agent 量化研究"),
    "website/guide/reproducible-quant-research.md": ("可复现量化研究 | open-xquant", "可复现量化研究"),
    "website/workflows/strategy-backtest.md": ("AI 量化回测 | open-xquant", "AI 量化回测"),
    "website/workflows/factor-research.md": ("AI 因子研究 | open-xquant", "AI 因子研究"),
    "website/workflows/research-audit.md": ("量化回测审计 | open-xquant", "量化回测审计"),
    "website/workflows/robustness-testing.md": ("量化策略稳健性检验 | open-xquant", "量化策略稳健性检验"),
    "website/workflows/live-trading.md": ("AI 量化实盘交易 | open-xquant", "AI 量化实盘交易"),
    "website/examples/index.md": ("量化研究示例 | open-xquant", "量化研究示例"),
    "website/faq/index.md": ("AI 量化常见问题 | open-xquant", "AI 量化常见问题"),
}


def _front_matter(text: str) -> dict[str, object]:
    assert text.startswith("---\n")
    _empty, raw, _body = text.split("---", 2)
    payload = yaml.safe_load(raw)
    assert isinstance(payload, dict)
    return payload


def test_search_intent_pages_have_unique_metadata_and_h1() -> None:
    titles: set[str] = set()
    descriptions: set[str] = set()
    for relative, (title, h1) in PAGES.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        metadata = _front_matter(text)
        assert metadata["title"] == title
        assert isinstance(metadata["description"], str)
        assert len(metadata["description"]) >= 45
        assert f"# {h1}" in text
        titles.add(str(metadata["title"]))
        descriptions.add(str(metadata["description"]))
    assert len(titles) == len(PAGES)
    assert len(descriptions) == len(PAGES)


def test_homepage_uses_real_architecture_asset_and_source_links() -> None:
    text = (ROOT / "website/index.md").read_text(encoding="utf-8")
    assert "/images/open-xquant-subagent-collaboration.png" in text
    assert "https://github.com/xingwudao/open-xquant" in text
    assert "/workflows/strategy-backtest" in text
    assert "/skills/" in text
