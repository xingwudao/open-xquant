from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_node_and_vitepress_are_pinned() -> None:
    package = json.loads((ROOT / "website/package.json").read_text(encoding="utf-8"))
    assert package["private"] is True
    assert package["type"] == "module"
    assert package["engines"]["node"] == ">=22 <23"
    assert package["devDependencies"]["vitepress"] == "1.6.4"
    assert (ROOT / "website/.node-version").read_text(encoding="utf-8") == "22\n"


def test_site_config_locks_production_urls() -> None:
    config = (ROOT / "website/.vitepress/config.mts").read_text(encoding="utf-8")
    seo = (ROOT / "website/.vitepress/seo.mts").read_text(encoding="utf-8")
    assert "base: '/open-xquant/'" in config
    assert "titleTemplate: false" in config
    assert "provider: 'local'" in config
    assert "https://xingwudao.github.io/open-xquant/" in config
    assert "https://xingwudao.github.io" in seo
    assert "rel: 'canonical'" in seo
    assert "application/ld+json" in seo
    assert "og:site_name" in seo
    assert "og:image:alt" in seo
    assert "twitter:card" in seo
    assert "SoftwareApplication" in seo
    assert "TechArticle" in seo
    assert "FAQPage" in seo
    assert "ItemList" in seo


def test_robots_declares_production_sitemap() -> None:
    robots = (ROOT / "website/public/robots.txt").read_text(encoding="utf-8")
    assert robots == (
        "User-agent: *\n"
        "Allow: /\n"
        "Sitemap: https://xingwudao.github.io/open-xquant/sitemap.xml\n"
    )


def test_llms_txt_declares_ai_quant_entry_points() -> None:
    text = (ROOT / "website/public/llms.txt").read_text(encoding="utf-8")
    assert text.startswith("# open-xquant\n")
    for url in (
        "https://xingwudao.github.io/open-xquant/",
        "https://xingwudao.github.io/open-xquant/guide/ai-quant-framework",
        "https://xingwudao.github.io/open-xquant/workflows/strategy-backtest",
        "https://xingwudao.github.io/open-xquant/skills/",
        "https://xingwudao.github.io/open-xquant/tools/",
    ):
        assert url in text
    assert "AI 量化研究框架" in text


def test_pages_workflow_manual_dispatch_can_deploy() -> None:
    workflow = (ROOT / ".github/workflows/docs-pages.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "actions/upload-pages-artifact" in workflow
    assert "actions/deploy-pages" in workflow
