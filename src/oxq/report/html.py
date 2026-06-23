"""Static HTML report renderer."""

from __future__ import annotations

import re
from html import escape, unescape
from pathlib import Path
from urllib.parse import urlsplit

from oxq.report.generator import generate_report

_IMAGE_RE = re.compile(r"^!\[(?P<alt>.*)]\((?P<src>.*)\)$")
_DECISIONS = {"REJECT", "NO EVIDENCE", "WATCHLIST", "PAPER TRADING CANDIDATE"}
_SAFE_HREF_SCHEMES = {"", "http", "https", "mailto"}


def render_html_report(run_dir: str | Path, lang: str = "zh") -> str:
    """Render a static, offline HTML report from run artifacts."""
    markdown = generate_report(run_dir, lang=lang)
    return render_markdown_html_report(markdown, lang=lang)


def render_markdown_html_report(markdown: str, lang: str = "zh") -> str:
    """Render a static, offline HTML report from an already generated Markdown report."""
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
            '<main class="report-shell">',
            '<article class="report">',
            '<div class="report-content">',
            body,
            "</div>",
            "</article>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _markdown_to_html(markdown: str) -> str:
    html_lines: list[str] = []
    lines = markdown.splitlines()
    i = 0
    seen_h1 = False
    while i < len(lines):
        line = lines[i]
        image = _IMAGE_RE.match(line)
        if image:
            src = image.group("src")
            alt = image.group("alt")
            caption = ""
            skip_until = i + 1
            if i + 2 < len(lines) and not lines[i + 1].strip() and _looks_like_figure_caption(lines[i + 2]):
                caption = lines[i + 2]
                skip_until = i + 3
            if not _is_safe_image_src(src):
                if alt:
                    html_lines.append(f"<p>{escape(alt)}</p>")
                i = skip_until
                continue
            html_lines.append('<figure class="figure-card">')
            html_lines.append(f'<img src="{escape(src, quote=True)}" alt="{escape(alt, quote=True)}">')
            if caption:
                html_lines.append(f"<figcaption>{escape(caption)}</figcaption>")
            html_lines.append("</figure>")
            i = skip_until
            continue

        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            heading = _inline_markdown(line[2:])
            if not seen_h1:
                html_lines.append('<header class="report-hero">')
                html_lines.append('<p class="report-kicker">open-xquant research report</p>')
                html_lines.append(f"<h1>{heading}</h1>")
                html_lines.append("</header>")
                seen_h1 = True
            else:
                html_lines.append(f"<h1>{heading}</h1>")
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
            html_lines.append(_markdown_table_to_html(table_lines))
            continue
        elif line.startswith("- "):
            html_lines.append(f"<p class=\"bullet\">{_inline_markdown(line)}</p>")
        else:
            decision = _decision_text(line)
            if decision is not None:
                html_lines.append(
                    f'<p class="decision-badge decision-{_decision_class(decision)}"><strong>{escape(decision)}</strong></p>'
                )
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


def _decision_text(line: str) -> str | None:
    stripped = line.strip()
    if not (stripped.startswith("**") and stripped.endswith("**")):
        return None
    candidate = stripped.strip("*")
    return candidate if candidate in _DECISIONS else None


def _decision_class(decision: str) -> str:
    return decision.lower().replace(" ", "-")


def _markdown_table_to_html(lines: list[str]) -> str:
    rows = [_split_table_row(line) for line in lines]
    rows = [row for row in rows if row and not _is_separator_row(row)]
    if not rows:
        return ""
    header = rows[0]
    body_rows = rows[1:]
    wrap_class = _table_wrap_class(header)
    html = [f'<div class="{wrap_class}">', "<table>", "<thead>", "<tr>"]
    html.extend(f"<th>{_inline_markdown(cell)}</th>" for cell in header)
    html.extend(["</tr>", "</thead>", "<tbody>"])
    for row in body_rows:
        html.append("<tr>")
        html.extend(f"<td>{_inline_markdown(cell)}</td>" for cell in row)
        html.append("</tr>")
    html.extend(["</tbody>", "</table>", "</div>"])
    return "\n".join(html)


def _table_wrap_class(header: list[str]) -> str:
    normalized = " ".join(cell.strip().lower() for cell in header)
    metric_terms = ("metric", "指标", "measure")
    value_terms = ("value", "数值", "结果", "result")
    status_terms = ("status", "状态", "audit", "审计", "robustness", "稳健")
    if any(term in normalized for term in metric_terms) and any(term in normalized for term in value_terms):
        return "table-wrap metric-table-wrap"
    if any(term in normalized for term in status_terms):
        return "table-wrap status-table-wrap"
    return "table-wrap"


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(row: list[str]) -> bool:
    return all(cell and set(cell) <= {"-", ":"} for cell in row)


def _link_repl(match: re.Match[str]) -> str:
    label = match.group("label")
    href = unescape(match.group("href"))
    if not _is_safe_href(href):
        return label
    return f'<a href="{escape(href, quote=True)}">{label}</a>'


def _is_safe_href(href: str) -> bool:
    stripped = href.strip()
    if not stripped:
        return False
    if any(ord(char) < 32 for char in stripped):
        return False
    return urlsplit(stripped).scheme.lower() in _SAFE_HREF_SCHEMES


def _is_safe_image_src(src: str) -> bool:
    stripped = src.strip()
    if not stripped or "\\" in stripped or "%" in stripped:
        return False
    if any(ord(char) < 32 for char in stripped):
        return False
    parsed = urlsplit(stripped)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return False
    path = parsed.path
    if not path.startswith("report_assets/"):
        return False
    return all(part not in {"", ".", ".."} for part in path.split("/"))


def _stylesheet() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f3f6f8;
  --text: #172033;
  --muted: #627084;
  --border: #d5dee8;
  --panel: #ffffff;
  --panel-soft: #f8fbfb;
  --accent: #0f766e;
  --accent-strong: #115e59;
  --warning: #b45309;
  --danger: #b42318;
  --shadow: 0 18px 45px rgba(23, 32, 51, 0.08);
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
    "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  line-height: 1.65;
}
.report-shell {
  min-height: 100vh;
  padding: 28px 16px 64px;
}
.report {
  max-width: 980px;
  margin: 0 auto;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.report-content {
  padding: 0 34px 52px;
}
.report-hero {
  margin: 0 -34px 26px;
  padding: 34px 34px 30px;
  background:
    linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(180, 83, 9, 0.08)),
    var(--panel-soft);
  border-bottom: 1px solid var(--border);
}
.report-kicker {
  margin: 0 0 8px;
  color: var(--accent-strong);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}
h1, h2, h3 {
  line-height: 1.25;
  margin: 1.4em 0 0.55em;
}
h1 {
  margin-top: 0;
  margin-bottom: 0;
  max-width: 760px;
  font-size: 34px;
  letter-spacing: 0;
}
h2 {
  border-top: 1px solid var(--border);
  padding-top: 18px;
  font-size: 24px;
  letter-spacing: 0;
}
h3 {
  font-size: 18px;
  letter-spacing: 0;
}
p {
  margin: 0 0 10px;
}
.bullet {
  position: relative;
  padding-left: 18px;
}
.bullet::before {
  content: "";
  position: absolute;
  left: 4px;
  top: 0.82em;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent);
}
.figure-card {
  margin: 24px 0 28px;
  padding: 16px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: 0 10px 28px rgba(23, 32, 51, 0.06);
}
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
  border-radius: 6px;
}
figcaption {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 14px;
}
.table-wrap {
  margin: 16px 0 24px;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0;
  background: var(--panel);
}
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
tbody tr:last-child td {
  border-bottom: 0;
}
th {
  background: #edf7f5;
  color: #143c37;
  font-weight: 650;
}
.metric-table-wrap table {
  min-width: 520px;
}
.metric-table-wrap tbody td:first-child {
  color: var(--muted);
  font-weight: 600;
}
.metric-table-wrap tbody td:last-child {
  color: var(--text);
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}
.status-table-wrap th {
  background: #fff7ed;
  color: #7c2d12;
}
.decision-badge {
  display: inline-block;
  margin: 2px 0 16px;
  padding: 8px 13px;
  border-radius: 8px;
  font-size: 18px;
  letter-spacing: 0;
}
.decision-reject {
  background: #fee2e2;
  color: #991b1b;
  border: 1px solid #fecaca;
}
.decision-no-evidence {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fde68a;
}
.decision-watchlist {
  background: #e0f2fe;
  color: #075985;
  border: 1px solid #bae6fd;
}
.decision-paper-trading-candidate {
  background: #dcfce7;
  color: #166534;
  border: 1px solid #bbf7d0;
}
pre {
  overflow-x: auto;
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
}
a {
  color: var(--accent);
}
@media (max-width: 720px) {
  .report-shell {
    padding: 0;
  }
  .report {
    border: 0;
    border-radius: 0;
    box-shadow: none;
  }
  .report-content {
    padding: 0 18px 40px;
  }
  .report-hero {
    margin: 0 -18px 22px;
    padding: 28px 18px 24px;
  }
  h1 {
    font-size: 28px;
  }
  h2 {
    font-size: 21px;
  }
}
@media print {
  body {
    background: #ffffff;
  }
  .report-shell {
    padding: 0;
  }
  .report {
    max-width: none;
    border: 0;
    box-shadow: none;
  }
  .report-content {
    padding: 0;
  }
  .report-hero {
    margin: 0 0 18px;
    padding: 0 0 16px;
    background: #ffffff;
  }
  h2, .figure-card, .table-wrap {
    break-inside: avoid;
  }
  a {
    color: var(--text);
  }
}
""".strip()
