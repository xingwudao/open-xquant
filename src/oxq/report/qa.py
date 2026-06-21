"""Quality checks for final open-xquant research reports."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from oxq.report.artifacts import RunArtifacts
from oxq.report.facts import ReportFacts, build_report_facts

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\((?P<src>[^)]+)\)")
_PERCENT_RE = re.compile(r"(?<![\w.])(?P<value>-?\d+(?:\.\d+)?)%")
_PLAIN_NUMBER_RE = re.compile(r"(?<![\w.%/-])(?P<value>-?\d+(?:\.\d+)?)(?!(?:\.\d)|[\w%/-])")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_CJK_FONT_NAMES = (
    "Noto Sans CJK",
    "Noto Serif CJK",
    "Source Han",
    "SourceHan",
    "SimHei",
    "Microsoft YaHei",
    "PingFang",
    "Heiti",
    "Hiragino Sans GB",
    "Arial Unicode",
    "WenQuanYi",
    "Songti",
    "STSong",
    "KaiTi",
    "Kaiti",
    "FangSong",
    "Yu Gothic",
    "Meiryo",
    "Malgun Gothic",
)
_ASSET_KIND_REQUIRED_PREFIX = {"figure": "figures", "attachment": "attachments"}


@dataclass(frozen=True)
class ReportQAFinding:
    id: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "severity": self.severity, "message": self.message}


@dataclass(frozen=True)
class ReportQAResult:
    status: str
    findings: list[ReportQAFinding]
    facts: ReportFacts

    @property
    def fatal_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "fatal")

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "fatal_count": self.fatal_count,
            "warning_count": self.warning_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "facts": self.facts.to_dict(),
        }


def run_report_qa(run_dir: str | Path) -> ReportQAResult:
    """Run final report QA checks for a backtest run directory."""
    run_path = Path(run_dir)
    artifacts = RunArtifacts.load(run_path)
    facts = build_report_facts(artifacts)
    findings: list[ReportQAFinding] = []

    markdown_path = run_path / "research_report.md"
    html_path = run_path / "research_report.html"
    markdown = _read_text(markdown_path, findings, "research_report.md")
    html = _read_text(html_path, findings, "research_report.html")

    manifest_assets = _manifest_assets(run_path, findings)
    registered_paths = {
        f"report_assets/{asset.get('path')}"
        for asset in manifest_assets
        if isinstance(asset.get("path"), str)
    }
    registered_figure_paths = _registered_figure_paths(manifest_assets)

    markdown_images = _markdown_image_sources(markdown)
    html_images = _html_image_sources(html)

    _check_image_counts(markdown_images, html_images, findings)
    _check_image_sources(markdown_images, html_images, findings)
    _check_markdown_images(markdown_images, registered_paths, registered_figure_paths, findings)
    _check_html_images(html_images, registered_paths, registered_figure_paths, findings)
    _check_manifest_assets(run_path, manifest_assets, findings)
    _check_required_date_disclosure(markdown, html, facts, findings)
    _check_cjk_font_risk(run_path, manifest_assets, findings)
    _check_numeric_claims(markdown, facts, findings, source_label="Markdown")
    _check_numeric_claims(_html_text(html), facts, findings, source_label="HTML")

    status = "fail" if any(f.severity == "fatal" for f in findings) else ("warn" if findings else "pass")
    return ReportQAResult(status=status, findings=findings, facts=facts)


def _read_text(path: Path, findings: list[ReportQAFinding], label: str) -> str:
    if not path.exists():
        findings.append(ReportQAFinding("report_file_missing", "fatal", f"{label} is missing"))
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        findings.append(ReportQAFinding("report_file_unreadable", "fatal", f"{label} could not be read: {exc}"))
        return ""


def _manifest_assets(run_path: Path, findings: list[ReportQAFinding]) -> list[dict[str, Any]]:
    path = run_path / "report_assets" / "manifest.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        findings.append(ReportQAFinding("manifest_unreadable", "fatal", f"report asset manifest could not be read: {exc}"))
        return []
    assets = raw.get("assets") if isinstance(raw, dict) else None
    if not isinstance(assets, list):
        findings.append(ReportQAFinding("manifest_invalid", "fatal", "report asset manifest must contain an assets array"))
        return []
    typed_assets = [asset for asset in assets if isinstance(asset, dict)]
    expected = sorted(typed_assets, key=_asset_sort_key)
    if typed_assets != expected:
        findings.append(
            ReportQAFinding(
                "manifest_order",
                "warning",
                "report asset manifest is not sorted by section, order, and id",
            )
        )
    return typed_assets


def _asset_sort_key(asset: dict[str, Any]) -> tuple[str, int, str]:
    return (str(asset.get("section", "results")), _int_value(asset.get("order"), 100), str(asset.get("id", "")))


def _registered_figure_paths(assets: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for asset in assets:
        relative_path = asset.get("path")
        if (
            asset.get("kind") == "figure"
            and isinstance(relative_path, str)
            and _manifest_path_matches_kind(asset.get("kind"), relative_path)
        ):
            paths.add(f"report_assets/{relative_path}")
    return paths


def _markdown_image_sources(markdown: str) -> list[str]:
    return [match.group("src").strip() for match in _MD_IMAGE_RE.finditer(markdown)]


def _html_image_sources(html: str) -> list[str]:
    parser = _ImageParser()
    parser.feed(html)
    return parser.sources


def _html_text(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return parser.text()


def _check_image_counts(markdown_images: list[str], html_images: list[str], findings: list[ReportQAFinding]) -> None:
    if len(markdown_images) != len(html_images):
        findings.append(
            ReportQAFinding(
                "image_count_mismatch",
                "fatal",
                f"Markdown image count ({len(markdown_images)}) does not match HTML image count ({len(html_images)})",
            )
        )


def _check_image_sources(markdown_images: list[str], html_images: list[str], findings: list[ReportQAFinding]) -> None:
    if len(markdown_images) == len(html_images) and markdown_images != html_images:
        findings.append(
            ReportQAFinding(
                "image_source_mismatch",
                "fatal",
                "Markdown and HTML image sources differ even though image counts match",
            )
        )


def _check_markdown_images(
    markdown_images: list[str],
    registered_paths: set[str],
    registered_figure_paths: set[str],
    findings: list[ReportQAFinding],
) -> None:
    for src in markdown_images:
        if not _safe_report_asset_src(src):
            findings.append(ReportQAFinding("markdown_image_path", "fatal", f"Markdown image path is not safe: {src}"))
            continue
        if src not in registered_paths:
            findings.append(ReportQAFinding("markdown_image_unregistered", "fatal", f"Markdown image is not registered: {src}"))
            continue
        if src not in registered_figure_paths:
            findings.append(ReportQAFinding("embedded_image_not_figure", "fatal", f"Markdown image is not a figure asset: {src}"))


def _check_html_images(
    html_images: list[str],
    registered_paths: set[str],
    registered_figure_paths: set[str],
    findings: list[ReportQAFinding],
) -> None:
    for src in html_images:
        if not _safe_report_asset_src(src):
            findings.append(ReportQAFinding("html_image_path", "fatal", f"HTML image must use report_assets/...: {src}"))
            continue
        if src not in registered_paths:
            findings.append(ReportQAFinding("html_image_unregistered", "fatal", f"HTML image is not registered: {src}"))
            continue
        if src not in registered_figure_paths:
            findings.append(ReportQAFinding("embedded_image_not_figure", "fatal", f"HTML image is not a figure asset: {src}"))


def _check_manifest_assets(run_path: Path, assets: list[dict[str, Any]], findings: list[ReportQAFinding]) -> None:
    for asset in assets:
        asset_id = str(asset.get("id", "unknown"))
        relative_path = asset.get("path")
        if not isinstance(relative_path, str) or not _safe_manifest_asset_path(relative_path):
            findings.append(ReportQAFinding("asset_path_invalid", "fatal", f"asset {asset_id} has unsafe path: {relative_path}"))
            continue
        if not _manifest_path_matches_kind(asset.get("kind"), relative_path):
            expected_prefix = _expected_asset_kind_prefix(asset.get("kind"))
            findings.append(
                ReportQAFinding(
                    "asset_kind_path_mismatch",
                    "fatal",
                    f"asset {asset_id} kind {asset.get('kind')} must use {expected_prefix}/ path: {relative_path}",
                )
            )
            continue
        asset_path = run_path / "report_assets" / relative_path
        if not asset_path.exists():
            findings.append(ReportQAFinding("asset_file_missing", "fatal", f"asset {asset_id} file is missing: {relative_path}"))
            continue
        if not asset_path.is_file():
            findings.append(
                ReportQAFinding("asset_file_not_regular", "fatal", f"asset {asset_id} path is not a regular file: {relative_path}")
            )
            continue
        size = asset_path.stat().st_size
        if size <= 0:
            findings.append(ReportQAFinding("asset_file_empty", "fatal", f"asset {asset_id} file is empty: {relative_path}"))
            continue
        expected_hash = asset.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            findings.append(ReportQAFinding("asset_hash_missing", "fatal", f"asset {asset_id} is missing sha256"))
        else:
            actual_hash = _sha256(asset_path)
            if actual_hash != expected_hash:
                findings.append(
                    ReportQAFinding(
                        "asset_hash_mismatch",
                        "fatal",
                        f"asset {asset_id} hash mismatch: expected {expected_hash}, got {actual_hash}",
                    )
                )
        if asset.get("kind") == "figure":
            dimensions = _image_dimensions(asset_path)
            if dimensions is None:
                findings.append(
                    ReportQAFinding("image_dimensions_unreadable", "warning", f"figure {asset_id} dimensions could not be read")
                )
            elif dimensions[0] <= 0 or dimensions[1] <= 0:
                findings.append(ReportQAFinding("image_dimensions_invalid", "fatal", f"figure {asset_id} has invalid dimensions"))


def _check_required_date_disclosure(markdown: str, html: str, facts: ReportFacts, findings: list[ReportQAFinding]) -> None:
    if facts.effective_last_trading_day is None:
        findings.append(
            ReportQAFinding(
                "effective_last_trading_day_unavailable",
                "fatal",
                "effective last trading day could not be computed from equity_curve.csv",
            )
        )
    else:
        if facts.effective_last_trading_day not in markdown:
            findings.append(
                ReportQAFinding(
                    "markdown_effective_last_trading_day_missing",
                    "fatal",
                    f"Markdown report must disclose effective last trading day {facts.effective_last_trading_day}",
                )
            )
        if facts.effective_last_trading_day not in html:
            findings.append(
                ReportQAFinding(
                    "html_effective_last_trading_day_missing",
                    "fatal",
                    f"HTML report must disclose effective last trading day {facts.effective_last_trading_day}",
                )
            )

    if facts.configured_end_date is None:
        findings.append(
            ReportQAFinding(
                "configured_end_date_unavailable",
                "fatal",
                "configured end date could not be computed from strategy_spec.yaml or data_manifest.json",
            )
        )
    else:
        if facts.configured_end_date not in markdown:
            findings.append(
                ReportQAFinding(
                    "markdown_configured_end_date_missing",
                    "fatal",
                    f"Markdown report must disclose configured end date {facts.configured_end_date}",
                )
            )
        if facts.configured_end_date not in html:
            findings.append(
                ReportQAFinding(
                    "html_configured_end_date_missing",
                    "fatal",
                    f"HTML report must disclose configured end date {facts.configured_end_date}",
                )
            )


def _check_cjk_font_risk(run_path: Path, assets: list[dict[str, Any]], findings: list[ReportQAFinding]) -> None:
    for asset in assets:
        if asset.get("kind") != "figure":
            continue
        title_caption = f"{asset.get('title', '')}\n{asset.get('caption', '')}"
        source = asset.get("source")
        script_text = ""
        script_path = None
        script_reference = source.get("script") if isinstance(source, dict) else None
        if isinstance(script_reference, str):
            if not _safe_source_script_path(script_reference):
                findings.append(
                    ReportQAFinding(
                        "source_script_path_invalid",
                        "fatal",
                        f"figure {asset.get('id', 'unknown')} has unsafe source script path: {script_reference}",
                    )
                )
                continue
            script_path = run_path / "report_assets" / script_reference
            script_text = _read_optional_text(script_path)
        has_cjk = _contains_cjk(title_caption) or _contains_cjk(script_text)
        if not has_cjk:
            continue
        if not script_text or not _has_verified_cjk_font(script_text):
            script_label = str(script_path.relative_to(run_path)) if script_path and script_path.exists() else "missing source script"
            findings.append(
                ReportQAFinding(
                    "cjk_font_unverified",
                    "warning",
                    f"figure {asset.get('id', 'unknown')} contains CJK labels but no CJK font configuration was verified in {script_label}",
                )
            )


def _check_numeric_claims(text: str, facts: ReportFacts, findings: list[ReportQAFinding], *, source_label: str) -> None:
    percent_values = _known_values(facts, _is_percent_known_number)
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        context_percent_values = _percent_context_known_values(line, facts)
        known_percent_values = context_percent_values if context_percent_values is not None else percent_values
        for match in _PERCENT_RE.finditer(line):
            parsed = _finite_float(match.group("value"))
            if parsed is None:
                continue
            value = parsed / 100.0
            if not any(_numbers_close(value, known) for known in known_percent_values):
                findings.append(
                    ReportQAFinding(
                        "numeric_claim_unverified",
                        "warning",
                        f"{source_label} percentage claim {match.group(0)} on line {line_number} was not found in metrics or report facts",
                    )
                )
    all_known_values = _known_values(facts, lambda _name: True)
    for line_number, line in enumerate(lines, start=1):
        if _skip_plain_number_line(line):
            continue
        context_values = _context_known_values(line, facts)
        known_values = context_values if context_values else all_known_values
        for match in _PLAIN_NUMBER_RE.finditer(line):
            parsed = _finite_float(match.group("value"))
            if parsed is None:
                continue
            if not any(_numbers_close(parsed, known) for known in known_values):
                findings.append(
                    ReportQAFinding(
                        "numeric_claim_unverified",
                        "warning",
                        f"{source_label} numeric claim {match.group(0)} on line {line_number} was not found in metrics or report facts",
                    )
                )


def _has_verified_cjk_font(script_text: str) -> bool:
    return any(name in script_text for name in _CJK_FONT_NAMES)


def _known_values(facts: ReportFacts, include_name: Callable[[str], bool]) -> list[float]:
    values: list[float] = []
    for item in facts.known_numbers:
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not isinstance(value, int | float):
            continue
        if include_name(name):
            values.append(float(value))
    return values


def _is_percent_known_number(name: str) -> bool:
    lowered = name.lower()
    if any(marker in lowered for marker in ("count", "trade", "cost", "fee", "commission", "sharpe", "calmar", "sortino")):
        return False
    return any(
        marker in lowered
        for marker in (
            "return",
            "drawdown",
            "volatility",
            "cagr",
            "rate",
            "pct",
            "percentage",
        )
    )


def _context_known_values(line: str, facts: ReportFacts) -> list[float]:
    lowered = line.lower()
    names: set[str] = set()
    if ("oos" in lowered or "样本外" in line) and ("trade" in lowered or "交易" in line):
        names.update({"metric.oos_trade_count", "fact.oos_trade_count"})
        if _mentions_total_trades(line, lowered):
            names.add("metric.trade_count")
    elif "trade" in lowered or "交易" in line:
        names.update({"metric.trade_count", "metric.oos_trade_count", "fact.oos_trade_count"})
    if "positive month" in lowered or "positive months" in lowered or "正收益月份" in line or "正收益月" in line:
        names.add("fact.positive_month_count")
    if "negative month" in lowered or "negative months" in lowered or "负收益月份" in line or "负收益月" in line:
        names.add("fact.negative_month_count")
    if "sharpe" in lowered or "夏普" in line:
        names.update(_known_number_names(facts, "sharpe"))
    if "calmar" in lowered or "卡玛" in line:
        names.update(_known_number_names(facts, "calmar"))
    if "sortino" in lowered:
        names.update(_known_number_names(facts, "sortino"))
    if any(marker in lowered for marker in ("cost", "fee", "commission")) or any(marker in line for marker in ("费用", "成本", "佣金")):
        for marker in ("cost", "fee", "commission"):
            names.update(_known_number_names(facts, marker))
    if not names:
        return []
    return _known_values(facts, lambda name: name in names)


def _percent_context_known_values(line: str, facts: ReportFacts) -> list[float] | None:
    lowered = line.lower()
    names: set[str] = set()
    drawdown_context = "drawdown" in lowered or "回撤" in line
    if drawdown_context:
        names.update(_known_number_names(facts, "drawdown"))
    if any(marker in lowered for marker in ("annualized return", "annual return", "cagr")) or "年化收益" in line:
        names.update(_known_number_names(facts, "annualized_return"))
        names.update(_known_number_names(facts, "cagr"))
    if any(marker in lowered for marker in ("total return", "cumulative return", "excess return")) or any(
        marker in line for marker in ("总收益", "累计收益", "超额收益")
    ):
        names.update(_known_number_names(facts, "total_return"))
        names.update(_known_number_names(facts, "excess_total_return"))
    elif "return" in lowered or "收益" in line:
        names.update(_known_number_names(facts, "return"))
    if "volatility" in lowered or "波动" in line:
        names.update(_known_number_names(facts, "volatility"))
    if "win rate" in lowered or "胜率" in line:
        names.update(_known_number_names(facts, "rate"))
    if not names:
        return None
    values = _known_values(facts, lambda name: name in names)
    if drawdown_context:
        values.extend(abs(value) for value in list(values))
    return values


def _known_number_names(facts: ReportFacts, marker: str) -> set[str]:
    return {
        str(item["name"])
        for item in facts.known_numbers
        if isinstance(item.get("name"), str) and marker in str(item["name"]).lower()
    }


def _skip_plain_number_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if re.match(r"^(?:\d+[.)]|\d+(?:\.\d+)+)\s+\D", stripped):
        return True
    if stripped.startswith("!["):
        return True
    if re.match(r"^(图|figure)\s+\d+[\s.:：]", stripped, flags=re.IGNORECASE):
        return True
    return False


def _mentions_total_trades(line: str, lowered: str) -> bool:
    if any(marker in lowered for marker in ("total trade", "total trades", "trade count", "all trades")):
        return True
    return any(marker in line for marker in ("总交易", "交易总数", "全部交易"))


def _safe_report_asset_src(src: str) -> bool:
    stripped = src.strip()
    if not stripped or "\\" in stripped or "%" in stripped:
        return False
    if any(ord(char) < 32 for char in stripped):
        return False
    parsed = urlsplit(stripped)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return False
    if not parsed.path.startswith("report_assets/"):
        return False
    return all(part not in {"", ".", ".."} for part in parsed.path.split("/"))


def _safe_manifest_asset_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path or "%" in path:
        return False
    parts = path.split("/")
    return parts[0] in {"figures", "attachments"} and all(part not in {"", ".", ".."} for part in parts)


def _manifest_path_matches_kind(kind: Any, path: str) -> bool:
    return isinstance(kind, str) and path.split("/", 1)[0] == _ASSET_KIND_REQUIRED_PREFIX.get(kind)


def _expected_asset_kind_prefix(kind: Any) -> str:
    return _ASSET_KIND_REQUIRED_PREFIX.get(kind, "figure|attachment")


def _safe_source_script_path(path: str) -> bool:
    stripped = path.strip()
    if not stripped or stripped.startswith("/") or "\\" in stripped or "%" in stripped:
        return False
    if any(ord(char) < 32 for char in stripped):
        return False
    parsed = urlsplit(stripped)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return False
    parts = parsed.path.split("/")
    return parts[0] == "scripts" and all(part not in {"", ".", ".."} for part in parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(data)
    if suffix == ".svg":
        text = data[:4096].decode("utf-8", errors="ignore")
        width = _svg_dimension(text, "width")
        height = _svg_dimension(text, "height")
        return (width, height) if width is not None and height is not None else None
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            return None
        length = int.from_bytes(data[index:index + 2], "big")
        if length < 2 or index + length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if index + 7 > len(data):
                return None
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height
        index += length
    return None


def _svg_dimension(text: str, attr: str) -> int | None:
    match = re.search(rf'{attr}=["\'](?P<value>\d+)(?:\.\d+)?(?:px)?["\']', text)
    return int(match.group("value")) if match else None


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except (OSError, UnicodeDecodeError):
        return ""


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _numbers_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-4, abs_tol=5e-4)


class _ImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        for name, value in attrs:
            if name.lower() == "src" and value is not None:
                self.sources.append(value.strip())
                return


class _HTMLTextParser(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "div",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    _IGNORED_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._lines: list[str] = []
        self._current: list[str] = []
        self._ignored_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in self._IGNORED_TAGS:
            self._ignored_stack.append(lowered)
            return
        if lowered in self._BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._ignored_stack and lowered == self._ignored_stack[-1]:
            self._ignored_stack.pop()
            return
        if lowered in self._BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._ignored_stack:
            return
        text = " ".join(data.split())
        if text:
            self._current.append(text)

    def text(self) -> str:
        self._flush()
        return "\n".join(self._lines)

    def _flush(self) -> None:
        if not self._current:
            return
        self._lines.append(" ".join(self._current))
        self._current = []
