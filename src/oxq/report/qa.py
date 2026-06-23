"""Quality checks for final open-xquant research reports."""

from __future__ import annotations

import ast
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
_NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_PERCENT_RE = re.compile(rf"(?<![\w.,])(?P<value>{_NUMBER_PATTERN})%")
_PLAIN_NUMBER_RE = re.compile(rf"(?<![\w.%/-])(?P<value>{_NUMBER_PATTERN})(?!(?:\.\d)|[\w%/-])")
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
_CJK_FONT_PATH_MARKERS = (
    "STHeiti",
    "PingFang",
    "NotoSansCJK",
    "Noto Sans CJK",
    "NotoSerifCJK",
    "Noto Serif CJK",
    "SourceHan",
    "Source Han",
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi",
)
_ASSET_KIND_REQUIRED_PREFIX = {"figure": "figures", "attachment": "attachments"}
_EFFECTIVE_LAST_TRADING_DAY_LABELS = ("effective last trading day", "有效数据最后交易日")
_CONFIGURED_END_DATE_LABELS = ("configured end date", "配置结束日")
_MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


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


def run_report_qa(run_dir: str | Path, *, include_advisory_checks: bool = True) -> ReportQAResult:
    """Run report QA checks for a backtest run directory.

    Deterministic artifact checks always run. Advisory semantic checks are kept
    available for targeted tests and tooling, but the CLI leaves them to the
    research-report-reviewer skill by default.
    """
    run_path = Path(run_dir)
    artifacts = RunArtifacts.load(run_path)
    facts = build_report_facts(artifacts)
    findings: list[ReportQAFinding] = []
    _check_metrics_artifact(run_path, findings)

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
    _check_source_script_paths(manifest_assets, findings)
    _check_required_date_disclosure(markdown, html, facts, findings)
    if include_advisory_checks:
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


def _check_metrics_artifact(run_path: Path, findings: list[ReportQAFinding]) -> None:
    path = run_path / "metrics.json"
    if not path.exists():
        findings.append(ReportQAFinding("metrics_unreadable", "fatal", "metrics.json is missing"))
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        findings.append(ReportQAFinding("metrics_unreadable", "fatal", f"metrics.json could not be read: {exc}"))
        return
    if not isinstance(value, dict):
        findings.append(ReportQAFinding("metrics_unreadable", "fatal", "metrics.json must contain a JSON object"))


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
    typed_assets = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            findings.append(
                ReportQAFinding(
                    "manifest_asset_invalid",
                    "fatal",
                    f"report asset manifest entry {index} must be an object",
                )
            )
            continue
        typed_assets.append(asset)
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


def _check_source_script_paths(assets: list[dict[str, Any]], findings: list[ReportQAFinding]) -> None:
    for asset in assets:
        source = asset.get("source")
        script_reference = source.get("script") if isinstance(source, dict) else None
        if isinstance(script_reference, str) and not _safe_source_script_path(script_reference):
            asset_label = "figure" if asset.get("kind") == "figure" else "asset"
            findings.append(
                ReportQAFinding(
                    "source_script_path_invalid",
                    "fatal",
                    f"{asset_label} {asset.get('id', 'unknown')} has unsafe source script path: {script_reference}",
                )
            )


def _check_required_date_disclosure(markdown: str, html: str, facts: ReportFacts, findings: list[ReportQAFinding]) -> None:
    html_text = _html_text(html)
    if facts.effective_last_trading_day is None:
        findings.append(
            ReportQAFinding(
                "effective_last_trading_day_unavailable",
                "fatal",
                "effective last trading day could not be computed from equity_curve.csv",
            )
        )
    else:
        if not _has_labeled_date(markdown, facts.effective_last_trading_day, _EFFECTIVE_LAST_TRADING_DAY_LABELS):
            findings.append(
                ReportQAFinding(
                    "markdown_effective_last_trading_day_missing",
                    "fatal",
                    f"Markdown report must disclose effective last trading day {facts.effective_last_trading_day}",
                )
            )
        if not _has_labeled_date(html_text, facts.effective_last_trading_day, _EFFECTIVE_LAST_TRADING_DAY_LABELS):
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
        if not _has_labeled_date(markdown, facts.configured_end_date, _CONFIGURED_END_DATE_LABELS):
            findings.append(
                ReportQAFinding(
                    "markdown_configured_end_date_missing",
                    "fatal",
                    f"Markdown report must disclose configured end date {facts.configured_end_date}",
                )
            )
        if not _has_labeled_date(html_text, facts.configured_end_date, _CONFIGURED_END_DATE_LABELS):
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
        has_scope_marker = bool(_scope_markers(line))
        for match in _PERCENT_RE.finditer(line):
            parsed = _finite_float(match.group("value"))
            if parsed is None:
                continue
            value = parsed / 100.0
            claim_scope = _claim_metric_scope(line, match.start())
            scope_override = claim_scope if claim_scope is not None else ("overall" if has_scope_marker else None)
            context_percent_values = _percent_context_known_values(
                line,
                facts,
                claim_start=match.start(),
                claim_end=match.end(),
                scope_override=scope_override,
                use_line_scope=not has_scope_marker,
            )
            known_percent_values = context_percent_values if context_percent_values is not None else percent_values
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
        scan_line = _plain_number_scan_line(line)
        if scan_line is None:
            continue
        has_scope_marker = bool(_scope_markers(scan_line))
        for match in _PLAIN_NUMBER_RE.finditer(scan_line):
            parsed = _finite_float(match.group("value"))
            if parsed is None:
                continue
            claim_scope = _claim_metric_scope(scan_line, match.start())
            scope_override = claim_scope if claim_scope is not None else ("overall" if has_scope_marker else None)
            context_values = _context_known_values(
                scan_line,
                facts,
                claim_start=match.start(),
                claim_end=match.end(),
                scope_override=scope_override,
                use_line_scope=not has_scope_marker,
            )
            known_values = context_values if context_values is not None else all_known_values
            if not any(_plain_numbers_close(parsed, known) for known in known_values):
                findings.append(
                    ReportQAFinding(
                        "numeric_claim_unverified",
                        "warning",
                        f"{source_label} numeric claim {match.group(0)} on line {line_number} was not found in metrics or report facts",
                    )
                )


def _has_labeled_date(text: str, expected_date: str, labels: tuple[str, ...]) -> bool:
    normalized = _normalize_label_text(text)
    for label in labels:
        label_pattern = re.escape(_normalize_label_text(label))
        date_pattern = re.escape(expected_date)
        if re.search(rf"{label_pattern}[^\d]{{0,80}}{date_pattern}", normalized):
            return True
    return False


def _normalize_label_text(text: str) -> str:
    normalized = text.replace("`", "").replace("*", "").replace("_", "").lower()
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _has_verified_cjk_font(script_text: str) -> bool:
    try:
        tree = ast.parse(script_text)
    except SyntaxError:
        return False

    parent_map = _ast_parent_map(tree)
    function_call_lines = _top_level_function_call_lines(tree, parent_map)
    cjk_text_calls: list[ast.Call] = []
    global_font_config_lines: list[int] = []
    cjk_legend_labels: list[tuple[str | None, int]] = []
    fonted_legends: list[tuple[str | None, int, bool]] = []
    for node in ast.walk(tree):
        before_lineno = _execution_before_lineno(node, parent_map, function_call_lines)
        cjk_font_paths = _assigned_cjk_font_path_names(tree, before_lineno)
        if isinstance(node, ast.Assign):
            context = _registered_cjk_font_context(tree, cjk_font_paths, before_lineno)
            if any(_is_rcparams_font_target(target) for target in node.targets) and (
                _ast_contains_cjk_font_name(node.value)
                or _ast_uses_registered_cjk_font_property(
                    node.value,
                    context[1],
                    context[0],
                    context[2],
                    context[3],
                )
            ):
                global_font_config_lines.append(before_lineno or _node_lineno(node) or 0)
        elif isinstance(node, ast.AnnAssign):
            context = _registered_cjk_font_context(tree, cjk_font_paths, before_lineno)
            if _is_rcparams_font_target(node.target) and (
                _ast_contains_cjk_font_name(node.value)
                or _ast_uses_registered_cjk_font_property(
                    node.value,
                    context[1],
                    context[0],
                    context[2],
                    context[3],
                )
            ):
                global_font_config_lines.append(before_lineno or _node_lineno(node) or 0)
        elif isinstance(node, ast.Call):
            context = _registered_cjk_font_context(tree, cjk_font_paths, before_lineno)
            if (
                _is_rcparams_update_font_call(
                    node,
                    context[1],
                    context[0],
                    context[2],
                    context[3],
                )
                or _is_matplotlib_rc_font_call(
                    node,
                    context[1],
                    context[0],
                    context[2],
                    context[3],
                )
            ):
                global_font_config_lines.append(before_lineno or _node_lineno(node) or 0)
            if _call_applies_to_cjk_text(node):
                cjk_text_calls.append(node)
                execution_lineno = before_lineno or _node_lineno(node) or 0
                if not (
                    _has_prior_global_font_config(global_font_config_lines, execution_lineno)
                    or _call_applies_cjk_font_property(
                        node,
                        _assigned_cjk_font_property_names(tree, cjk_font_paths, before_lineno),
                        cjk_font_paths,
                    )
                ):
                    return False
            elif _call_has_cjk_legend_label(node):
                cjk_legend_labels.append((_call_receiver_key(node), before_lineno or _node_lineno(node) or 0))
            if _is_legend_call(node) and _call_applies_cjk_font_property(
                node,
                _assigned_cjk_font_property_names(tree, cjk_font_paths, before_lineno),
                cjk_font_paths,
                require_cjk_text=False,
            ):
                fonted_legends.append(
                    (
                        _call_receiver_key(node),
                        before_lineno or _node_lineno(node) or 0,
                        _call_has_cjk_legend_label(node),
                    )
                )
    if cjk_text_calls:
        return True
    if _has_verified_cjk_legend_font(cjk_legend_labels, fonted_legends):
        return True
    if global_font_config_lines:
        return True
    return False


def _assigned_cjk_font_path_names(tree: ast.AST, before_lineno: int | None = None) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not _node_is_before(node, before_lineno):
            continue
        value, targets = _assignment_value_and_targets(node)
        if value is None:
            continue
        if not _ast_contains_cjk_font_path(value):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _assigned_cjk_font_property_names(
    tree: ast.AST,
    cjk_font_paths: set[str],
    before_lineno: int | None = None,
) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not _node_is_before(node, before_lineno):
            continue
        value, targets = _assignment_value_and_targets(node)
        if value is None:
            continue
        node_cjk_font_paths = _assigned_cjk_font_path_names(tree, _node_lineno(node))
        if not _is_cjk_font_property_call(value, node_cjk_font_paths):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _registered_cjk_font_context(
    tree: ast.AST,
    cjk_font_paths: set[str],
    before_lineno: int | None,
) -> tuple[set[str], set[str], bool, set[str]]:
    registered_cjk_font_paths, registered_cjk_font_path_literals = _registered_cjk_font_path_refs(
        tree,
        cjk_font_paths,
        before_lineno,
    )
    has_registered_cjk_font = bool(registered_cjk_font_paths or registered_cjk_font_path_literals)
    registered_cjk_font_properties = _assigned_registered_cjk_font_property_names(
        tree,
        registered_cjk_font_paths,
        has_registered_cjk_font,
        registered_cjk_font_path_literals,
        before_lineno,
    )
    return registered_cjk_font_paths, registered_cjk_font_properties, has_registered_cjk_font, registered_cjk_font_path_literals


def _registered_cjk_font_path_refs(
    tree: ast.AST,
    cjk_font_paths: set[str],
    before_lineno: int | None,
) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    literals: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "addfont":
            continue
        if not _node_is_before(node, before_lineno):
            continue
        payloads = [*node.args, *(keyword.value for keyword in node.keywords)]
        for payload in payloads:
            literals.update(_cjk_font_path_literals(payload))
            names.update(_cjk_font_path_ref_names(payload, cjk_font_paths))
    return names, literals


def _assigned_registered_cjk_font_property_names(
    tree: ast.AST,
    registered_cjk_font_paths: set[str],
    has_registered_cjk_font: bool,
    registered_cjk_font_path_literals: set[str],
    before_lineno: int | None,
) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not _node_is_before(node, before_lineno):
            continue
        value, targets = _assignment_value_and_targets(node)
        if value is None:
            continue
        node_cjk_font_paths = _assigned_cjk_font_path_names(tree, _node_lineno(node))
        node_registered_paths, node_registered_literals = _registered_cjk_font_path_refs(
            tree,
            node_cjk_font_paths,
            _node_lineno(node),
        )
        node_has_registered_font = bool(node_registered_paths or node_registered_literals)
        if _is_registered_cjk_font_property_call(
            value,
            node_registered_paths,
            node_has_registered_font,
            node_registered_literals,
        ) or _ast_get_name_uses_registered_cjk_font_property(
            value,
            names,
            node_registered_paths,
            node_has_registered_font,
            node_registered_literals,
        ):
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _assignment_value_and_targets(node: ast.AST) -> tuple[ast.AST | None, list[ast.expr]]:
    if isinstance(node, ast.Assign):
        return node.value, list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return node.value, [node.target]
    return None, []


def _ast_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _node_is_inside_function(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> bool:
    parent = parent_map.get(node)
    while parent is not None:
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            return True
        parent = parent_map.get(parent)
    return False


def _execution_before_lineno(
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    function_call_lines: dict[str, int],
) -> int | None:
    function_name = _enclosing_function_name(node, parent_map)
    if function_name is not None:
        return function_call_lines.get(function_name)
    return _node_lineno(node)


def _enclosing_function_name(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> str | None:
    parent = parent_map.get(node)
    while parent is not None:
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
            return parent.name
        if isinstance(parent, ast.Lambda):
            return None
        parent = parent_map.get(parent)
    return None


def _top_level_function_call_lines(tree: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> dict[str, int]:
    lines: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _node_is_inside_function(node, parent_map):
            continue
        name = _call_name(node.func)
        lineno = _node_lineno(node)
        if name is None or lineno is None:
            continue
        lines[name] = min(lines.get(name, lineno), lineno)
    return lines


def _node_lineno(node: ast.AST) -> int | None:
    return getattr(node, "lineno", None)


def _node_is_before(node: ast.AST, before_lineno: int | None) -> bool:
    lineno = _node_lineno(node)
    return before_lineno is None or lineno is None or lineno < before_lineno


def _is_cjk_font_property_call(node: ast.AST | None, cjk_font_paths: set[str]) -> bool:
    if node is None:
        return False
    if not isinstance(node, ast.Call) or _call_name(node.func) != "FontProperties":
        return False
    payloads = [*node.args, *(keyword.value for keyword in node.keywords if keyword.arg == "fname")]
    return any(
        _ast_contains_cjk_font_path(payload) or bool(_cjk_font_path_ref_names(payload, cjk_font_paths))
        for payload in payloads
    )


def _is_registered_cjk_font_property_call(
    node: ast.AST | None,
    registered_cjk_font_paths: set[str],
    has_registered_cjk_font: bool,
    registered_cjk_font_path_literals: set[str],
) -> bool:
    if node is None:
        return False
    if not isinstance(node, ast.Call) or _call_name(node.func) != "FontProperties":
        return False
    payloads = [*node.args, *(keyword.value for keyword in node.keywords if keyword.arg == "fname")]
    return any(
        (has_registered_cjk_font and bool(_cjk_font_path_literals(payload) & registered_cjk_font_path_literals))
        or bool(_cjk_font_path_ref_names(payload, registered_cjk_font_paths))
        for payload in payloads
    )


def _ast_uses_cjk_font_property(node: ast.AST | None, cjk_font_properties: set[str], cjk_font_paths: set[str]) -> bool:
    if node is None:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in cjk_font_properties:
            return True
        if _is_cjk_font_property_call(child, cjk_font_paths):
            return True
    return False


def _ast_uses_registered_cjk_font_property(
    node: ast.AST | None,
    registered_cjk_font_properties: set[str],
    registered_cjk_font_paths: set[str],
    has_registered_cjk_font: bool,
    registered_cjk_font_path_literals: set[str],
) -> bool:
    if node is None:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in registered_cjk_font_properties:
            return True
        if _is_registered_cjk_font_property_call(
            child,
            registered_cjk_font_paths,
            has_registered_cjk_font,
            registered_cjk_font_path_literals,
        ):
            return True
    return False


def _ast_get_name_uses_registered_cjk_font_property(
    node: ast.AST | None,
    registered_cjk_font_properties: set[str],
    registered_cjk_font_paths: set[str],
    has_registered_cjk_font: bool,
    registered_cjk_font_path_literals: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "get_name":
        return False
    receiver = node.func.value
    if isinstance(receiver, ast.Name) and receiver.id in registered_cjk_font_properties:
        return True
    return _is_registered_cjk_font_property_call(
        receiver,
        registered_cjk_font_paths,
        has_registered_cjk_font,
        registered_cjk_font_path_literals,
    )


def _is_rcparams_font_target(target: ast.expr) -> bool:
    if not isinstance(target, ast.Subscript):
        return False
    if not _is_rcparams_object(target.value):
        return False
    key = _literal_string(target.slice)
    return key is not None and key.startswith("font.")


def _is_rcparams_object(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "rcParams"
    return isinstance(node, ast.Attribute) and node.attr == "rcParams"


def _is_rcparams_update_font_call(
    node: ast.Call,
    registered_cjk_font_properties: set[str],
    registered_cjk_font_paths: set[str],
    has_registered_cjk_font: bool,
    registered_cjk_font_path_literals: set[str],
) -> bool:
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and _is_rcparams_object(node.func.value)
    ):
        return False
    payloads = [*node.args, *(keyword.value for keyword in node.keywords)]
    return any(
        _dict_sets_font_key(payload)
        and (
            _ast_contains_cjk_font_name(payload)
            or _ast_uses_registered_cjk_font_property(
                payload,
                registered_cjk_font_properties,
                registered_cjk_font_paths,
                has_registered_cjk_font,
                registered_cjk_font_path_literals,
            )
        )
        for payload in payloads
    )


def _is_matplotlib_rc_font_call(
    node: ast.Call,
    registered_cjk_font_properties: set[str],
    registered_cjk_font_paths: set[str],
    has_registered_cjk_font: bool,
    registered_cjk_font_path_literals: set[str],
) -> bool:
    name = _call_name(node.func)
    if name != "rc":
        return False
    if not node.args or _literal_string(node.args[0]) != "font":
        return False
    payloads = [*node.args[1:], *(keyword.value for keyword in node.keywords)]
    return any(
        _ast_contains_cjk_font_name(payload)
        or _ast_uses_registered_cjk_font_property(
            payload,
            registered_cjk_font_properties,
            registered_cjk_font_paths,
            has_registered_cjk_font,
            registered_cjk_font_path_literals,
        )
        for payload in payloads
    )


def _call_applies_cjk_font_property(
    node: ast.Call,
    cjk_font_properties: set[str],
    cjk_font_paths: set[str],
    *,
    require_cjk_text: bool = True,
) -> bool:
    if require_cjk_text and not _call_applies_to_cjk_text(node):
        return False
    for keyword in node.keywords:
        if keyword.arg in {"fontproperties", "font_properties", "prop"} and _ast_uses_cjk_font_property(
            keyword.value,
            cjk_font_properties,
            cjk_font_paths,
        ):
            return True
        if keyword.arg == "fontdict" and _fontdict_uses_cjk_font_property(
            keyword.value,
            cjk_font_properties,
            cjk_font_paths,
        ):
            return True
    return False


def _call_applies_to_cjk_text(node: ast.Call) -> bool:
    name = _call_name(node.func)
    if name not in {
        "annotate",
        "figtext",
        "set_title",
        "set_xlabel",
        "set_ylabel",
        "set_zlabel",
        "suptitle",
        "text",
        "title",
        "xlabel",
        "ylabel",
        "zlabel",
    }:
        return False
    return any(_ast_contains_cjk_text(payload) for payload in [*node.args, *(keyword.value for keyword in node.keywords)])


def _call_has_cjk_legend_label(node: ast.Call) -> bool:
    if _is_legend_call(node):
        return any(_ast_contains_cjk_text(arg) for arg in node.args) or any(
            keyword.arg == "labels" and _ast_contains_cjk_text(keyword.value) for keyword in node.keywords
        )
    return any(keyword.arg == "label" and _ast_contains_cjk_text(keyword.value) for keyword in node.keywords)


def _is_legend_call(node: ast.Call) -> bool:
    return _call_name(node.func) == "legend"


def _call_receiver_key(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return ast.dump(node.func.value, include_attributes=False)
    return None


def _has_verified_cjk_legend_font(
    cjk_legend_labels: list[tuple[str | None, int]],
    fonted_legends: list[tuple[str | None, int, bool]],
) -> bool:
    for receiver, legend_lineno, legend_has_cjk_label in fonted_legends:
        if legend_has_cjk_label:
            return True
        if any(label_receiver == receiver and label_lineno < legend_lineno for label_receiver, label_lineno in cjk_legend_labels):
            return True
    return False


def _has_prior_global_font_config(global_font_config_lines: list[int], lineno: int) -> bool:
    return any(config_lineno < lineno for config_lineno in global_font_config_lines)


def _fontdict_uses_cjk_font_property(
    node: ast.AST,
    cjk_font_properties: set[str],
    cjk_font_paths: set[str],
) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    for raw_key, value in zip(node.keys, node.values, strict=True):
        key = _literal_string(raw_key) if raw_key is not None else None
        if key in {"fontproperties", "font_properties", "prop"} and _ast_uses_cjk_font_property(
            value,
            cjk_font_properties,
            cjk_font_paths,
        ):
            return True
    return False


def _ast_contains_cjk_text(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and _CJK_RE.search(child.value):
            return True
    return False


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dict_sets_font_key(node: ast.expr) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    return any(
        raw_key is not None and (key := _literal_string(raw_key)) is not None and key.startswith("font.")
        for raw_key in node.keys
    )


def _ast_contains_cjk_font_name(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if any(name in child.value for name in _CJK_FONT_NAMES):
                return True
    return False


def _ast_contains_cjk_font_path(node: ast.AST | None) -> bool:
    return bool(_cjk_font_path_literals(node))


def _cjk_font_path_literals(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    paths: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if any(marker in child.value for marker in _CJK_FONT_PATH_MARKERS):
                paths.add(child.value)
    return paths


def _cjk_font_path_ref_names(node: ast.AST | None, cjk_font_paths: set[str]) -> set[str]:
    if isinstance(node, ast.Name) and node.id in cjk_font_paths:
        return {node.id}
    if isinstance(node, ast.Call) and _call_name(node.func) in {"Path", "str"} and len(node.args) == 1:
        return _cjk_font_path_ref_names(node.args[0], cjk_font_paths)
    return set()


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


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
    if "fee_rate" in lowered or "slippage_rate" in lowered:
        return True
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


def _context_known_values(
    line: str,
    facts: ReportFacts,
    *,
    claim_start: int | None = None,
    claim_end: int | None = None,
    scope_override: str | None = None,
    use_line_scope: bool = True,
) -> list[float] | None:
    lowered = line.lower()
    scope = scope_override if scope_override is not None else (_line_metric_scope(line, lowered) if use_line_scope else None)
    ratio_scope = scope if scope is not None else "overall"
    claim_month_names = _claim_month_count_names(line, claim_start, claim_end)
    if claim_month_names is not None:
        return _known_values(facts, lambda name: name in claim_month_names)
    claim_metric_names = _claim_trade_or_ratio_names(line, facts, claim_start, claim_end, ratio_scope)
    if claim_metric_names is not None:
        return _known_values(facts, lambda name: name in claim_metric_names)
    names: set[str] = set()
    matched_context = False
    claim_trade_names = _claim_trade_names(line, facts, claim_start, claim_end)
    if claim_trade_names is not None:
        matched_context = True
        names.update(claim_trade_names)
    elif ("oos" in lowered or "样本外" in line) and ("trade" in lowered or "交易" in line):
        matched_context = True
        names.update({"metric.oos_trade_count", "fact.oos_trade_count"})
        if _mentions_total_trades(line, lowered):
            names.add("metric.trade_count")
    elif "trade" in lowered or "交易" in line:
        matched_context = True
        names.add("metric.trade_count")
    if "positive month" in lowered or "positive months" in lowered or "正收益月份" in line or "正收益月" in line:
        matched_context = True
        names.add("fact.positive_month_count")
    if "negative month" in lowered or "negative months" in lowered or "负收益月份" in line or "负收益月" in line:
        matched_context = True
        names.add("fact.negative_month_count")
    claim_ratio_names = _claim_ratio_names(line, facts, claim_start, claim_end, ratio_scope)
    if claim_ratio_names is not None:
        matched_context = True
        names.update(claim_ratio_names)
    else:
        if "sharpe" in lowered or "夏普" in line:
            matched_context = True
            names.update(_known_number_names(facts, "sharpe", scope=ratio_scope))
        if "calmar" in lowered or "卡玛" in line:
            matched_context = True
            names.update(_known_number_names(facts, "calmar", scope=ratio_scope))
        if "sortino" in lowered:
            matched_context = True
            names.update(_known_number_names(facts, "sortino", scope=ratio_scope))
    claim_cost_names = _claim_cost_names(line, facts, claim_start, claim_end)
    if claim_cost_names is not None:
        matched_context = True
        names.update(claim_cost_names)
    elif any(marker in lowered for marker in ("cost", "fee", "commission", "slippage")) or any(
        marker in line for marker in ("费用", "成本", "佣金", "滑点")
    ):
        matched_context = True
        for marker in ("cost", "fee", "commission", "slippage"):
            names.update(_known_number_names(facts, marker))
    if "initial cash" in lowered or "初始资金" in line or "初始现金" in line:
        matched_context = True
        names.update(_known_number_names(facts, "initial_cash"))
    if not names:
        return [] if matched_context else None
    return _known_values(facts, lambda name: name in names)


def _percent_context_known_values(
    line: str,
    facts: ReportFacts,
    *,
    claim_start: int | None = None,
    claim_end: int | None = None,
    scope_override: str | None = None,
    use_line_scope: bool = True,
) -> list[float] | None:
    lowered = line.lower()
    scope = scope_override if scope_override is not None else (_line_metric_scope(line, lowered) if use_line_scope else None)
    context_line = _claim_context_text(line, claim_start)
    context_lowered = context_line.lower()
    return_scope = scope if scope is not None else "overall"
    names: set[str] = set()
    matched_context = False
    drawdown_context = "drawdown" in context_lowered or "回撤" in context_line
    if drawdown_context:
        matched_context = True
        names.update(_known_number_names(facts, "drawdown", scope=return_scope))
    if any(marker in context_lowered for marker in ("annualized return", "annual return", "cagr")) or "年化收益" in context_line:
        matched_context = True
        names.update(_known_number_names(facts, "annualized_return", scope=return_scope))
        names.update(_known_number_names(facts, "cagr", scope=return_scope))
    else:
        monthly_names = _monthly_return_names_for_line(context_line, facts)
        if monthly_names:
            matched_context = True
            names.update(monthly_names)
        elif "excess return" in context_lowered or "超额收益" in context_line:
            matched_context = True
            names.update(_known_number_names(facts, "excess_total_return", scope=return_scope))
        elif ("benchmark" in context_lowered or "基准" in context_line) and (
            any(marker in context_lowered for marker in ("total return", "cumulative return")) or any(
                marker in context_line for marker in ("总收益", "累计收益")
            )
        ):
            matched_context = True
            names.update(_known_benchmark_total_return_names(facts, scope=return_scope))
        elif any(marker in context_lowered for marker in ("total return", "cumulative return")) or any(
            marker in context_line for marker in ("总收益", "累计收益")
        ):
            matched_context = True
            names.update(_known_strategy_total_return_names(facts, scope=return_scope))
        elif ("benchmark" in context_lowered or "基准" in context_line) and (
            "return" in context_lowered or "收益" in context_line
        ):
            matched_context = True
            names.update(_known_benchmark_total_return_names(facts, scope=return_scope))
        elif "excess" in context_lowered or "超额" in context_line:
            matched_context = True
            names.update(_known_number_names(facts, "excess_total_return", scope=return_scope))
        elif "return" in context_lowered or "收益" in context_line:
            matched_context = True
            names.update(_known_strategy_total_return_names(facts, scope=return_scope))
    if "volatility" in context_lowered or "波动" in context_line:
        matched_context = True
        names.update(_known_number_names(facts, "volatility", scope=return_scope))
    if "win rate" in context_lowered or "胜率" in context_line:
        matched_context = True
        names.update(_known_number_names(facts, "win_rate", scope=return_scope))
        names.update(_known_number_names(facts, "winrate", scope=return_scope))
    claim_cost_names = _claim_cost_names(line, facts, claim_start, claim_end, rate_only=True)
    if claim_cost_names is not None:
        matched_context = True
        names.update(claim_cost_names)
    elif any(marker in context_lowered for marker in ("cost", "fee", "commission", "slippage")) or any(
        marker in context_line for marker in ("费用", "成本", "佣金", "滑点")
    ):
        matched_context = True
        for marker in ("cost", "fee", "commission", "slippage"):
            names.update(_known_number_names(facts, marker))
    if not names:
        return [] if matched_context else None
    values = _known_values(facts, lambda name: name in names)
    if drawdown_context:
        values.extend(abs(value) for value in list(values))
    return values


def _monthly_return_names_for_line(line: str, facts: ReportFacts) -> set[str]:
    lowered = line.lower()
    if "return" not in lowered and "收益" not in line:
        return set()
    known_months = _known_monthly_return_months(facts)
    mentioned_months = _mentioned_months(line, known_months)
    return {f"fact.monthly_return.{month}" for month in mentioned_months}


def _known_monthly_return_months(facts: ReportFacts) -> set[str]:
    prefix = "fact.monthly_return."
    return {
        str(item["name"])[len(prefix) :]
        for item in facts.known_numbers
        if isinstance(item.get("name"), str) and str(item["name"]).startswith(prefix)
    }


def _mentioned_months(line: str, known_months: set[str]) -> set[str]:
    if not known_months:
        return set()
    mentioned: set[str] = set()
    for match in re.finditer(r"\b(?P<year>[12]\d{3})[-/](?P<month>0?[1-9]|1[0-2])\b", line):
        month = f"{match.group('year')}-{int(match.group('month')):02d}"
        if month in known_months:
            mentioned.add(month)
    for match in re.finditer(r"(?:(?P<year>[12]\d{3})年)?(?P<month>0?[1-9]|1[0-2])月", line):
        mentioned.update(_known_months_for_label(known_months, int(match.group("month")), match.group("year")))
    lowered = line.lower()
    years = set(re.findall(r"\b([12]\d{3})\b", line))
    for name, month_number in _MONTH_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            mentioned.update(_known_months_for_label(known_months, month_number, next(iter(years)) if len(years) == 1 else None))
    return mentioned


def _known_months_for_label(known_months: set[str], month_number: int, year: str | None) -> set[str]:
    if year is not None:
        month = f"{year}-{month_number:02d}"
        return {month} if month in known_months else set()
    suffix = f"-{month_number:02d}"
    return {month for month in known_months if month.endswith(suffix)}


def _claim_cost_names(
    line: str,
    facts: ReportFacts,
    claim_start: int | None,
    claim_end: int | None,
    *,
    rate_only: bool = False,
) -> set[str] | None:
    if claim_start is None or claim_end is None:
        return None
    label = _nearest_claim_label(
        line,
        _claim_label_markers(
            line,
            (
                ("slippage", r"\bslippage\b|滑点"),
                ("fee", r"\bfees?\b|\bcommissions?\b|佣金|手续费"),
                ("cost", r"\bcosts?\b|费用|成本"),
            ),
        ),
        claim_start,
        claim_end,
    )
    if label is None:
        return None
    if label == "slippage":
        return _filter_rate_names(_known_number_names(facts, "slippage"), rate_only=rate_only)
    if label == "fee":
        fee_names = _known_number_names(facts, "fee") | _known_number_names(facts, "commission")
        return _filter_rate_names(fee_names, rate_only=rate_only)
    names: set[str] = set()
    for marker in ("cost", "fee", "commission", "slippage"):
        names.update(_known_number_names(facts, marker))
    return _filter_rate_names(names, rate_only=rate_only)


def _filter_rate_names(names: set[str], *, rate_only: bool) -> set[str]:
    if not rate_only:
        return names
    return {name for name in names if _name_looks_like_rate(name)}


def _name_looks_like_rate(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in ("rate", "pct", "percentage"))


def _claim_trade_names(line: str, facts: ReportFacts, claim_start: int | None, claim_end: int | None) -> set[str] | None:
    if claim_start is None or claim_end is None:
        return None
    label = _nearest_claim_label(
        line,
        _claim_label_markers(
            line,
            (
                ("oos", r"\boos\s+trades?\b|\bout-of-sample\s+trades?\b|\bout of sample\s+trades?\b|样本外交易"),
                ("total", r"\btotal\s+trades?\b|\btrade\s+count\b|\ball\s+trades?\b|总交易|交易总数|全部交易"),
            ),
        ),
        claim_start,
        claim_end,
    )
    if label is None:
        return None
    if label == "oos":
        return {"metric.oos_trade_count", "fact.oos_trade_count"}
    return {"metric.trade_count"}


def _claim_month_count_names(line: str, claim_start: int | None, claim_end: int | None) -> set[str] | None:
    if claim_start is None or claim_end is None:
        return None
    label = _nearest_claim_label(
        line,
        _claim_label_markers(
            line,
            (
                ("positive", r"\bpositive\s+months?\b|\bpositive\s+return\s+months?\b|正收益月份|正收益月"),
                ("negative", r"\bnegative\s+months?\b|\bnegative\s+return\s+months?\b|负收益月份|负收益月"),
            ),
        ),
        claim_start,
        claim_end,
    )
    if label is None:
        return None
    if label == "positive":
        return {"fact.positive_month_count"}
    return {"fact.negative_month_count"}


def _claim_trade_or_ratio_names(
    line: str,
    facts: ReportFacts,
    claim_start: int | None,
    claim_end: int | None,
    scope: str | None,
) -> set[str] | None:
    if claim_start is None or claim_end is None:
        return None
    label = _nearest_claim_label(
        line,
        _claim_label_markers(
            line,
            (
                ("trade:oos", r"\boos\s+trades?\b|\bout-of-sample\s+trades?\b|\bout of sample\s+trades?\b|样本外交易"),
                ("trade:total", r"\btotal\s+trades?\b|\btrade\s+count\b|\ball\s+trades?\b|总交易|交易总数|全部交易"),
                ("ratio:sharpe", r"\bsharpe(?:\s+ratio)?\b|夏普"),
                ("ratio:calmar", r"\bcalmar(?:\s+ratio)?\b|卡玛"),
                ("ratio:sortino", r"\bsortino(?:\s+ratio)?\b"),
            ),
        ),
        claim_start,
        claim_end,
    )
    if label is None:
        return None
    if label == "trade:oos":
        return {"metric.oos_trade_count", "fact.oos_trade_count"}
    if label == "trade:total":
        return {"metric.trade_count"}
    return _known_number_names(facts, label.removeprefix("ratio:"), scope=scope)


def _claim_ratio_names(
    line: str,
    facts: ReportFacts,
    claim_start: int | None,
    claim_end: int | None,
    scope: str | None,
) -> set[str] | None:
    if claim_start is None or claim_end is None:
        return None
    label = _nearest_claim_label(
        line,
        _claim_label_markers(
            line,
            (
                ("sharpe", r"\bsharpe(?:\s+ratio)?\b|夏普"),
                ("calmar", r"\bcalmar(?:\s+ratio)?\b|卡玛"),
                ("sortino", r"\bsortino(?:\s+ratio)?\b"),
            ),
        ),
        claim_start,
        claim_end,
    )
    if label is None:
        return None
    return _known_number_names(facts, label, scope=scope)


def _claim_label_markers(line: str, patterns: tuple[tuple[str, str], ...]) -> list[tuple[int, int, str]]:
    markers: list[tuple[int, int, str]] = []
    for label, pattern in patterns:
        markers.extend((match.start(), match.end(), label) for match in re.finditer(pattern, line, flags=re.IGNORECASE))
    return sorted(markers)


def _nearest_claim_label(
    line: str, markers: list[tuple[int, int, str]], claim_start: int, claim_end: int
) -> str | None:
    if not markers:
        return None
    segment_start, segment_end = _claim_segment_bounds(line, claim_start)
    markers = [marker for marker in markers if segment_start <= marker[0] < segment_end]
    if not markers:
        return None

    def distance(marker: tuple[int, int, str]) -> int:
        start, end, _label = marker
        if end < claim_start:
            return claim_start - end
        if start > claim_end:
            return start - claim_end
        return 0

    nearest = min(markers, key=distance)
    return nearest[2] if distance(nearest) <= 40 else None


def _known_number_names(facts: ReportFacts, marker: str, *, scope: str | None = None) -> set[str]:
    names = {
        str(item["name"])
        for item in facts.known_numbers
        if isinstance(item.get("name"), str) and marker in str(item["name"]).lower()
    }
    return _scope_known_number_names(names, scope)


def _known_strategy_total_return_names(facts: ReportFacts, *, scope: str | None = None) -> set[str]:
    names = {
        str(item["name"])
        for item in facts.known_numbers
        if isinstance(item.get("name"), str) and _is_strategy_total_return_name(str(item["name"]))
    }
    return _scope_known_number_names(names, scope)


def _known_benchmark_total_return_names(facts: ReportFacts, *, scope: str | None = None) -> set[str]:
    names = {
        str(item["name"])
        for item in facts.known_numbers
        if isinstance(item.get("name"), str) and _is_benchmark_total_return_name(str(item["name"]))
    }
    return _scope_known_number_names(names, scope)


def _is_strategy_total_return_name(name: str) -> bool:
    leaf = name.lower().rsplit(".", maxsplit=1)[-1]
    return leaf in {"total_return", "is_total_return", "oos_total_return"}


def _is_benchmark_total_return_name(name: str) -> bool:
    return name.lower().rsplit(".", maxsplit=1)[-1] == "benchmark_total_return"


def _scope_known_number_names(names: set[str], scope: str | None) -> set[str]:
    if scope == "oos":
        return {name for name in names if _metric_name_scope(name) == "oos"}
    if scope == "is":
        return {name for name in names if _metric_name_scope(name) == "is"}
    if scope == "overall":
        return {name for name in names if _metric_name_scope(name) is None}
    return names


def _metric_name_scope(name: str) -> str | None:
    lowered = name.lower()
    if ".oos_" in lowered:
        return "oos"
    if ".is_" in lowered:
        return "is"
    return None


def _line_metric_scope(line: str, lowered: str) -> str | None:
    scopes = {scope for _, scope in _scope_markers(line, lowered)}
    if len(scopes) != 1:
        return None
    return next(iter(scopes))


def _claim_metric_scope(line: str, index: int) -> str | None:
    lowered = line.lower()
    markers = _scope_markers(line, lowered)
    if not markers:
        return None
    segment_start, segment_end = _claim_segment_bounds(line, index)
    segment_markers = [marker for marker in markers if segment_start <= marker[0] < segment_end]
    before = [marker for marker in segment_markers if marker[0] <= index]
    if before:
        return max(before, key=lambda marker: marker[0])[1]
    after = [marker for marker in segment_markers if marker[0] > index]
    if after:
        return min(after, key=lambda marker: marker[0])[1]
    return None


def _claim_segment_bounds(line: str, index: int) -> tuple[int, int]:
    boundaries = [match.start() for match in re.finditer(r"[,，;；。!?！？|]", line)]
    boundaries.extend(
        match.start()
        for match in re.finditer(r"\.", line)
        if not (
            match.start() > 0
            and match.start() + 1 < len(line)
            and line[match.start() - 1].isdigit()
            and line[match.start() + 1].isdigit()
        )
    )
    start = max((position + 1 for position in boundaries if position < index), default=0)
    end = min((position for position in boundaries if position > index), default=len(line))
    return start, end


def _claim_context_text(line: str, index: int | None) -> str:
    if index is None:
        return line
    start, end = _claim_segment_bounds(line, index)
    return line[start:end]


def _scope_markers(line: str, lowered: str | None = None) -> list[tuple[int, str]]:
    lowered = line.lower() if lowered is None else lowered
    markers: list[tuple[int, str]] = []
    markers.extend((match.start(), "oos") for match in re.finditer(r"\boos\b", lowered))
    markers.extend((match.start(), "oos") for match in re.finditer(r"out-of-sample|out of sample", lowered))
    markers.extend((match.start(), "oos") for match in re.finditer("样本外", line))
    markers.extend((match.start(), "is") for match in re.finditer(r"\bIS\b", line))
    markers.extend((match.start(), "is") for match in re.finditer(r"in-sample|in sample", lowered))
    markers.extend((match.start(), "is") for match in re.finditer("样本内", line))
    return sorted(markers)


def _plain_number_scan_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return None
    if re.match(r"^\d+(?:\.\d+)+\s+\D", stripped) and not _plain_number_line_has_metric_context(stripped):
        return None
    stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
    stripped = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:UTC|Z))?\b", "", stripped)
    if stripped.startswith("!["):
        return None
    stripped = re.sub(r"^(图|figure)\s+\d+[\s.:：]\s*", "", stripped, count=1, flags=re.IGNORECASE)
    if not stripped:
        return None
    return stripped


def _plain_number_line_has_metric_context(line: str) -> bool:
    lowered = line.lower()
    return any(
        marker in lowered
        for marker in (
            "return",
            "drawdown",
            "volatility",
            "sharpe",
            "calmar",
            "sortino",
            "win rate",
            "fee",
            "commission",
            "slippage",
            "trade",
        )
    ) or any(marker in line for marker in ("收益", "回撤", "波动", "夏普", "卡玛", "胜率", "费用", "佣金", "滑点", "交易"))


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
    if suffix == ".webp":
        return _webp_dimensions(data)
    if suffix == ".svg":
        text = data[:4096].decode("utf-8", errors="ignore")
        width = _svg_dimension(text, "width")
        height = _svg_dimension(text, "height")
        return (width, height) if width is not None and height is not None else None
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    index = 12
    while index + 8 <= len(data):
        chunk_type = data[index:index + 4]
        chunk_size = int.from_bytes(data[index + 4:index + 8], "little")
        chunk_start = index + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            return None
        payload = data[chunk_start:chunk_end]
        if chunk_type == b"VP8X" and len(payload) >= 10:
            width = int.from_bytes(payload[4:7] + b"\x00", "little") + 1
            height = int.from_bytes(payload[7:10] + b"\x00", "little") + 1
            return width, height
        if chunk_type == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
            bits = int.from_bytes(payload[1:5], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height
        if chunk_type == b"VP8 " and len(payload) >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            width = int.from_bytes(payload[6:8], "little") & 0x3FFF
            height = int.from_bytes(payload[8:10], "little") & 0x3FFF
            return width, height
        index = chunk_end + (chunk_size % 2)
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
    if isinstance(value, str):
        value = value.replace(",", "")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _numbers_close(left: float, right: float, *, abs_tol: float = 5e-5) -> bool:
    return math.isclose(left, right, rel_tol=1e-4, abs_tol=abs_tol)


def _plain_numbers_close(left: float, right: float) -> bool:
    abs_tol = 5e-3 if max(abs(left), abs(right)) >= 1 else 5e-5
    return _numbers_close(left, right, abs_tol=abs_tol)


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
        "tfoot",
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
