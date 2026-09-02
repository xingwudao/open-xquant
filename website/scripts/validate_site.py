from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

PRODUCTION_BASE_URL = "https://xingwudao.github.io/open-xquant/"
PRODUCTION_SITEMAP_URL = f"{PRODUCTION_BASE_URL}sitemap.xml"


class SiteValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PageSignals:
    path: Path
    title: str
    description: str
    canonical: str
    h1_count: int
    og_title: str
    og_description: str
    og_url: str
    og_image: str
    json_ld_count: int


class _PageParser(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.title_parts: list[str] = []
        self.description = ""
        self.canonical = ""
        self.h1_count = 0
        self.og_title = ""
        self.og_description = ""
        self.og_url = ""
        self.og_image = ""
        self.json_ld_count = 0
        self.body_has_content = False
        self._in_title = False
        self._in_body = False
        self._content_stack: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        attr_map = {name.lower(): value or "" for name, value in attrs}

        if tag == "title":
            self._in_title = True
        elif tag == "body":
            self._in_body = True
        elif tag == "h1":
            self.h1_count += 1
        elif tag == "meta":
            name = attr_map.get("name", "").lower()
            prop = attr_map.get("property", "").lower()
            content = attr_map.get("content", "").strip()
            if name == "description":
                self.description = content
            elif prop == "og:title":
                self.og_title = content
            elif prop == "og:description":
                self.og_description = content
            elif prop == "og:url":
                self.og_url = content
            elif prop == "og:image":
                self.og_image = content
        elif tag == "link":
            rel = {value.lower() for value in attr_map.get("rel", "").split()}
            if "canonical" in rel:
                self.canonical = attr_map.get("href", "").strip()
        elif tag == "script":
            script_type = attr_map.get("type", "").lower()
            if script_type == "application/ld+json":
                self.json_ld_count += 1

        if self._in_body:
            self._content_stack.append(tag)
            if tag not in {"body", "script", "style", "template", "noscript"}:
                self.body_has_content = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag == "body":
            self._in_body = False

        if self._in_body:
            for index in range(len(self._content_stack) - 1, -1, -1):
                if self._content_stack[index] == tag:
                    del self._content_stack[index:]
                    break

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._in_body and data.strip() and not self._is_in_ignored_body_tag():
            self.body_has_content = True

    def _is_in_ignored_body_tag(self) -> bool:
        return any(tag in {"script", "style", "template"} for tag in self._content_stack)

    def signals(self) -> PageSignals:
        return PageSignals(
            path=self.path,
            title=" ".join("".join(self.title_parts).split()),
            description=self.description,
            canonical=self.canonical,
            h1_count=self.h1_count,
            og_title=self.og_title,
            og_description=self.og_description,
            og_url=self.og_url,
            og_image=self.og_image,
            json_ld_count=self.json_ld_count,
        )


def inspect_page(path: Path) -> PageSignals:
    parser = _PageParser(path)
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    signals = parser.signals()

    missing = [
        name
        for name, value in (
            ("title", signals.title),
            ("description", signals.description),
            ("canonical", signals.canonical),
            ("og:title", signals.og_title),
            ("og:description", signals.og_description),
            ("og:url", signals.og_url),
            ("og:image", signals.og_image),
        )
        if not value
    ]
    if missing:
        raise SiteValidationError(f"{path}: missing {', '.join(missing)}")
    if signals.json_ld_count == 0:
        raise SiteValidationError(f"{path}: missing JSON-LD")
    is_not_found_page = path.name == "404.html"
    if not is_not_found_page and signals.h1_count != 1:
        raise SiteValidationError(
            f"{path}: expected exactly one H1, found {signals.h1_count}"
        )
    if not signals.canonical.startswith(PRODUCTION_BASE_URL):
        raise SiteValidationError(
            f"{path}: canonical leaves production prefix: {signals.canonical}"
        )
    if not is_not_found_page and not parser.body_has_content:
        raise SiteValidationError(f"{path}: empty body")

    return signals


def validate_site(dist: Path) -> tuple[PageSignals, ...]:
    html_paths = tuple(sorted(dist.rglob("*.html")))
    if not html_paths:
        raise SiteValidationError(f"{dist}: no HTML files found")

    pages = tuple(inspect_page(path) for path in html_paths)
    indexable_pages = tuple(page for page in pages if page.path.name != "404.html")
    _validate_unique_metadata(indexable_pages, "title")
    _validate_unique_metadata(indexable_pages, "description")
    _validate_unique_metadata(indexable_pages, "canonical")
    _validate_robots(dist)
    not_found_pages = tuple(page for page in pages if page.path.name == "404.html")
    _validate_sitemap(dist, indexable_pages, not_found_pages)
    return pages


def _validate_unique_metadata(pages: tuple[PageSignals, ...], field_name: str) -> None:
    seen: dict[str, Path] = {}
    for page in pages:
        value = str(getattr(page, field_name))
        if value in seen:
            raise SiteValidationError(
                f"{page.path}: duplicate {field_name} also used by {seen[value]}"
            )
        seen[value] = page.path


def _validate_robots(dist: Path) -> None:
    robots_path = dist / "robots.txt"
    if not robots_path.exists():
        raise SiteValidationError(f"{robots_path}: missing robots.txt")
    robots = robots_path.read_text(encoding="utf-8")
    if f"Sitemap: {PRODUCTION_SITEMAP_URL}" not in robots.splitlines():
        raise SiteValidationError(
            f"{robots_path}: missing production sitemap {PRODUCTION_SITEMAP_URL}"
        )


def _validate_sitemap(
    dist: Path, pages: tuple[PageSignals, ...], not_found_pages: tuple[PageSignals, ...]
) -> None:
    sitemap_path = dist / "sitemap.xml"
    if not sitemap_path.exists():
        raise SiteValidationError(f"{sitemap_path}: missing sitemap.xml")
    sitemap_urls = _read_sitemap_urls(sitemap_path)
    canonical_urls = {page.canonical for page in pages}
    ignored_urls = {page.canonical for page in not_found_pages}
    if sitemap_urls - ignored_urls != canonical_urls:
        raise SiteValidationError(
            f"{sitemap_path}: sitemap entries do not equal HTML canonical URLs"
        )


def _read_sitemap_urls(path: Path) -> set[str]:
    try:
        root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    except ElementTree.ParseError as exc:
        raise SiteValidationError(f"{path}: invalid sitemap XML: {exc}") from exc

    urls: set[str] = set()
    for element in root.iter():
        if _local_name(element.tag) != "loc":
            continue
        url = (element.text or "").strip()
        if not url:
            continue
        if not url.startswith(PRODUCTION_BASE_URL):
            raise SiteValidationError(f"{path}: sitemap URL leaves production prefix: {url}")
        urls.add(url)
    return urls


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated docs HTML.")
    parser.add_argument("dist", type=Path)
    args = parser.parse_args(argv)

    try:
        validate_site(args.dist)
    except SiteValidationError as exc:
        print(f"site validation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
