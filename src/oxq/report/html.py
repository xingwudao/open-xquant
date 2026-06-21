"""Static HTML report renderer."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from oxq.report.generator import generate_report

_IMAGE_RE = re.compile(r"^!\[(?P<alt>.*)]\((?P<src>.*)\)$")


def render_html_report(run_dir: str | Path, lang: str = "zh") -> str:
    """Render a static, offline HTML report from run artifacts."""
    markdown = generate_report(run_dir, lang=lang)
    body = _markdown_to_html(markdown)
    return "\n".join(
        [
            "<!doctype html>",
            f'<html lang="{escape(lang, quote=True)}">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>open-xquant report</title>",
            "<style>",
            _stylesheet(),
            "</style>",
            "</head>",
            "<body>",
            '<main class="report">',
            body,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _markdown_to_html(markdown: str) -> str:
    html_lines: list[str] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        image = _IMAGE_RE.match(line)
        if image:
            caption = ""
            skip_until = i + 1
            if i + 2 < len(lines) and not lines[i + 1].strip() and _looks_like_figure_caption(lines[i + 2]):
                caption = lines[i + 2]
                skip_until = i + 3
            html_lines.append("<figure>")
            html_lines.append(
                f'<img src="{escape(image.group("src"), quote=True)}" '
                f'alt="{escape(image.group("alt"), quote=True)}">'
            )
            if caption:
                html_lines.append(f"<figcaption>{escape(caption)}</figcaption>")
            html_lines.append("</figure>")
            i = skip_until
            continue

        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{_inline_markdown(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{_inline_markdown(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{_inline_markdown(line[4:])}</h3>")
        elif line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            html_lines.append(f'<pre class="markdown-table">{escape(chr(10).join(table_lines))}</pre>')
            continue
        elif line.startswith("- "):
            html_lines.append(f"<p class=\"bullet\">{_inline_markdown(line)}</p>")
        else:
            html_lines.append(f"<p>{_inline_markdown(line)}</p>")
        i += 1
    return "\n".join(html_lines)


def _looks_like_figure_caption(line: str) -> bool:
    return line.startswith("图 ") or line.startswith("Figure ")


def _inline_markdown(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[(?P<label>[^]]+)]\((?P<href>[^)]+)\)", _link_repl, escaped)
    return escaped


def _link_repl(match: re.Match[str]) -> str:
    label = match.group("label")
    href = match.group("href")
    return f'<a href="{escape(href, quote=True)}">{label}</a>'


def _stylesheet() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f8fafc;
  --text: #1f2937;
  --muted: #667085;
  --border: #d9e1ec;
  --panel: #ffffff;
  --accent: #0f766e;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.65;
}
.report {
  max-width: 980px;
  margin: 0 auto;
  padding: 32px 20px 56px;
}
h1, h2, h3 {
  line-height: 1.25;
  margin: 1.4em 0 0.55em;
}
h1 {
  margin-top: 0;
  font-size: 32px;
}
h2 {
  border-top: 1px solid var(--border);
  padding-top: 18px;
  font-size: 24px;
}
h3 {
  font-size: 18px;
}
p {
  margin: 0 0 10px;
}
.bullet {
  padding-left: 16px;
}
figure {
  margin: 22px 0;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
}
img {
  display: block;
  max-width: 100%;
  height: auto;
}
figcaption {
  margin-top: 10px;
  color: var(--muted);
  font-size: 14px;
}
.markdown-table {
  overflow-x: auto;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
}
a {
  color: var(--accent);
}
""".strip()
