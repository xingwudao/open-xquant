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


def test_robots_declares_production_sitemap() -> None:
    robots = (ROOT / "website/public/robots.txt").read_text(encoding="utf-8")
    assert robots == (
        "User-agent: *\n"
        "Allow: /\n"
        "Sitemap: https://xingwudao.github.io/open-xquant/sitemap.xml\n"
    )
