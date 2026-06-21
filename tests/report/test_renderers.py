from __future__ import annotations

import json

import yaml

from oxq.report import generate_report
from oxq.report.assets import add_report_asset
from oxq.report.generator import write_report_files
from oxq.report.html import render_html_report, render_markdown_html_report
from oxq.spec.schema import StrategySpec


def test_generate_report_defaults_to_chinese_and_embeds_registered_figures(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    figure = tmp_path / "equity.png"
    figure.write_bytes(b"png")
    add_report_asset(
        run_dir,
        figure,
        asset_id="equity",
        title="策略净值与基准对比",
        caption="由 equity_curve.csv 生成。",
        section="results",
        order=10,
    )

    report = generate_report(run_dir)

    assert report.startswith("# 研究报告: renderer_case")
    assert "## 7. 图表资产" in report
    assert "![策略净值与基准对比](report_assets/figures/equity.png)" in report
    assert "图 1. 由 equity_curve.csv 生成。" in report
    assert "- **sha256**: sha256:" in report


def test_generate_report_supports_explicit_english(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    report = generate_report(run_dir, lang="en")

    assert report.startswith("# Research Report: renderer_case")
    assert "## 1. Executive Decision" in report
    assert "No chart assets registered." in report


def test_render_html_report_is_static_and_embeds_registered_figures(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    figure = tmp_path / "drawdown.png"
    figure.write_bytes(b"png")
    add_report_asset(
        run_dir,
        figure,
        asset_id="drawdown",
        title="最大回撤曲线",
        caption="由 equity_curve.csv 生成。",
    )

    html = render_html_report(run_dir)

    assert html.startswith("<!doctype html>")
    assert '<html lang="zh">' in html
    assert "<figure>" in html
    assert 'src="report_assets/figures/drawdown.png"' in html
    assert 'alt="最大回撤曲线"' in html
    assert "<figcaption>图 1. 由 equity_curve.csv 生成。</figcaption>" in html
    assert "<script" not in html.lower()


def test_render_html_report_adds_professional_decision_and_table_markup(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    html = render_html_report(run_dir, lang="en")

    assert 'class="decision-badge decision-' in html
    assert "<table>" in html
    assert "<th>Metric</th>" in html
    assert "<td>Total Return</td>" in html
    assert "markdown-table" not in html
    assert "<script" not in html.lower()


def test_render_markdown_html_report_rejects_active_url_schemes() -> None:
    html = render_markdown_html_report(
        "# Report\n\n[unsafe](javascript:alert(1))\n\n[ok](https://example.com)\n",
        lang="en",
    )

    assert "javascript:" not in html
    assert '<a href="https://example.com">ok</a>' in html
    assert "unsafe" in html


def test_write_report_files_outputs_markdown_and_html_by_default(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    outputs = write_report_files(run_dir)

    assert outputs.markdown == run_dir / "research_report.md"
    assert outputs.html == run_dir / "research_report.html"
    assert outputs.markdown.read_text(encoding="utf-8").startswith("# 研究报告: renderer_case")
    assert outputs.html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_write_report_files_format_all_uses_distinct_paths_for_html_out(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    out = tmp_path / "exports" / "report.html"

    outputs = write_report_files(run_dir, output_format="all", out=out)

    assert outputs.markdown == out.with_suffix(".md")
    assert outputs.html == out
    assert outputs.markdown.exists()
    assert outputs.html.exists()
    assert outputs.markdown.read_text(encoding="utf-8").startswith("# 研究报告: renderer_case")
    assert outputs.html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_write_report_files_copies_asset_bundle_next_to_custom_out(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    figure = tmp_path / "equity.png"
    figure.write_bytes(b"png")
    add_report_asset(run_dir, figure, asset_id="equity", title="策略净值")
    out = tmp_path / "exports" / "report.md"

    outputs = write_report_files(run_dir, output_format="all", out=out)

    assert outputs.markdown == out
    assert outputs.html == out.with_suffix(".html")
    assert (out.parent / "report_assets/figures/equity.png").read_bytes() == b"png"
    assert "![策略净值](report_assets/figures/equity.png)" in out.read_text(encoding="utf-8")
    assert 'src="report_assets/figures/equity.png"' in outputs.html.read_text(encoding="utf-8")


def test_write_report_files_renders_html_from_same_markdown(monkeypatch, tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    monkeypatch.setattr("oxq.report.generator.generate_report", lambda run_dir, lang="zh": "# One\n\nsame markdown")
    monkeypatch.setattr("oxq.report.html.generate_report", lambda run_dir, lang="zh": "# Two\n\nstale markdown")

    outputs = write_report_files(run_dir, output_format="all")

    assert outputs.markdown.read_text(encoding="utf-8") == "# One\n\nsame markdown"
    html = outputs.html.read_text(encoding="utf-8")
    assert "same markdown" in html
    assert "stale markdown" not in html


def _write_report_run(tmp_path):
    spec = StrategySpec.template(
        strategy_id="renderer_case",
        hypothesis="renderer should include assets",
    )
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "renderer-run",
                "trade_count": 12,
                "max_drawdown": -0.05,
                "total_return": 0.1,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 1.1,
            }
        ),
        encoding="utf-8",
    )
    return run_dir
