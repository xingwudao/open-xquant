from __future__ import annotations

from pathlib import Path

import pytest
from website.scripts.validate_site import SiteValidationError, validate_site

VALID_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<title>AI 量化框架 | open-xquant</title>
<meta name="description" content="中文友好的 AI 量化研究框架，覆盖回测、因子、审计和稳健性工作流。">
<link rel="canonical" href="https://xingwudao.github.io/open-xquant/">
<meta property="og:title" content="AI 量化框架 | open-xquant">
<meta property="og:description" content="中文友好的 AI 量化研究框架，覆盖回测、因子、审计和稳健性工作流。">
<meta property="og:url" content="https://xingwudao.github.io/open-xquant/">
<meta property="og:image" content="https://xingwudao.github.io/open-xquant/images/open-xquant-subagent-collaboration.png">
<script type="application/ld+json">{"@type":"WebSite"}</script>
</head><body><h1>AI 量化研究框架</h1></body></html>"""


def _write_site(tmp_path: Path, html: str) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(html, encoding="utf-8")
    (dist / "sitemap.xml").write_text(
        "<urlset><url><loc>https://xingwudao.github.io/open-xquant/</loc></url></urlset>",
        encoding="utf-8",
    )
    (dist / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nSitemap: https://xingwudao.github.io/open-xquant/sitemap.xml\n",
        encoding="utf-8",
    )
    return dist


def test_validate_site_accepts_complete_metadata(tmp_path: Path) -> None:
    validate_site(_write_site(tmp_path, VALID_HTML))


def test_validate_site_rejects_missing_canonical(tmp_path: Path) -> None:
    html = VALID_HTML.replace(
        '<link rel="canonical" href="https://xingwudao.github.io/open-xquant/">',
        "",
    )
    with pytest.raises(SiteValidationError, match="canonical"):
        validate_site(_write_site(tmp_path, html))


def test_validate_site_rejects_duplicate_titles(tmp_path: Path) -> None:
    dist = _write_site(tmp_path, VALID_HTML)
    nested = dist / "guide"
    nested.mkdir()
    (nested / "duplicate.html").write_text(VALID_HTML, encoding="utf-8")
    with pytest.raises(SiteValidationError, match="duplicate title"):
        validate_site(dist)
