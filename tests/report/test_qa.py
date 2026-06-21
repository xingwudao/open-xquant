from __future__ import annotations

import base64
import json

import yaml

from oxq.report.assets import add_report_asset
from oxq.report.html import render_markdown_html_report
from oxq.report.qa import run_report_qa
from oxq.spec.schema import StrategySpec


def test_report_qa_passes_complete_registered_report(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    script = run_dir / "report_assets/scripts/plot.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        'import matplotlib.pyplot as plt\nplt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]\n',
        encoding="utf-8",
    )
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(
        run_dir,
        figure,
        asset_id="equity",
        title="策略净值",
        caption="由 equity_curve.csv 生成。",
        section="results",
        order=10,
        source_script=script,
        source_artifacts=["equity_curve.csv"],
    )
    markdown = (
        "# 研究报告\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "总收益为 20.00%，正收益月份 2 个，负收益月份 1 个。\n\n"
        "![策略净值](report_assets/figures/equity.png)\n\n"
        "图 1. 由 equity_curve.csv 生成。\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"
    assert result.fatal_count == 0
    assert result.warning_count == 0
    assert result.facts.configured_end_date == "2024-03-31"
    assert result.facts.effective_last_trading_day == "2024-03-29"


def test_report_qa_flags_report_image_manifest_hash_and_number_problems(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(run_dir, figure, asset_id="equity", title="Equity", section="results", order=10)
    (run_dir / "report_assets/figures/equity.png").write_bytes(b"changed")
    markdown = (
        "# Report\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "总收益为 99.00%。\n\n"
        "![Unregistered](report_assets/figures/unregistered.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        '<!doctype html><html><body><img src="../outside.png"></body></html>',
        encoding="utf-8",
    )

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    finding_ids = {finding.id for finding in result.findings}
    assert "asset_hash_mismatch" in finding_ids
    assert "markdown_image_unregistered" in finding_ids
    assert "html_image_path" in finding_ids
    assert "numeric_claim_unverified" in finding_ids


def test_report_qa_warns_when_cjk_chart_lacks_font_check(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    script = run_dir / "report_assets/scripts/plot.py"
    script.parent.mkdir(parents=True)
    script.write_text('import matplotlib.pyplot as plt\nplt.title("策略净值")\n', encoding="utf-8")
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(
        run_dir,
        figure,
        asset_id="equity",
        title="策略净值",
        caption="由 equity_curve.csv 生成。",
        source_script=script,
    )
    markdown = (
        "# 研究报告\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "![策略净值](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "cjk_font_unverified" for finding in result.findings)


def test_report_qa_flags_missing_html_date_disclosures(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = "# 研究报告\n\n有效数据最后交易日：2024-03-29\n\n配置结束日：2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text("<!doctype html><html><body><h1>研究报告</h1></body></html>", encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    finding_ids = {finding.id for finding in result.findings}
    assert "html_effective_last_trading_day_missing" in finding_ids
    assert "html_configured_end_date_missing" in finding_ids


def test_report_qa_flags_same_count_different_html_image_sources(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    equity = tmp_path / "equity.png"
    drawdown = tmp_path / "drawdown.png"
    _write_png(equity)
    _write_png(drawdown)
    add_report_asset(run_dir, equity, asset_id="equity", title="Equity", section="results", order=10)
    add_report_asset(run_dir, drawdown, asset_id="drawdown", title="Drawdown", section="results", order=20)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/figures/equity.png)\n"
    )
    html = '<!doctype html><html><body><p>2024-03-29 2024-03-31</p><img src="report_assets/figures/drawdown.png"></body></html>'
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(html, encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "image_source_mismatch" for finding in result.findings)


def test_report_qa_rejects_embedded_attachment_images(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    attachment = tmp_path / "notes.pdf"
    attachment.write_bytes(b"%PDF-1.4")
    add_report_asset(run_dir, attachment, asset_id="notes", title="Notes", section="appendix", order=10)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Notes](report_assets/attachments/notes.pdf)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "embedded_image_not_figure" for finding in result.findings)


def test_report_qa_requires_available_date_facts(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    (run_dir / "equity_curve.csv").unlink()
    (run_dir / "research_report.md").write_text("# Report\n\nConfigured end date: 2024-03-31\n", encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        "<!doctype html><html><body>Configured end date: 2024-03-31</body></html>",
        encoding="utf-8",
    )

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "effective_last_trading_day_unavailable" for finding in result.findings)


def test_report_qa_flags_non_percent_numeric_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The report claims 99 OOS trades, 10 positive months, and Sharpe 9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "99" in messages
    assert "10" in messages
    assert "9.99" in messages


def test_report_qa_does_not_match_percent_claims_against_counts(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "trade_count": 2, "oos_trade_count": 1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The invented total return was 200.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "200.00%" in finding.message for finding in result.findings)


def test_report_qa_does_not_treat_generic_font_sans_serif_as_cjk_font(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    script = run_dir / "report_assets/scripts/plot.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        'import matplotlib.pyplot as plt\nplt.rcParams["font.sans-serif"] = ["DejaVu Sans"]\nplt.title("策略净值")\n',
        encoding="utf-8",
    )
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(run_dir, figure, asset_id="equity", title="策略净值", source_script=script)
    markdown = "# 研究报告\n\n有效数据最后交易日：2024-03-29\n\n配置结束日：2024-03-31\n\n![策略净值](report_assets/figures/equity.png)\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "cjk_font_unverified" for finding in result.findings)


def test_report_qa_checks_numeric_claims_in_html_text(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The total return was 20.00%.\n"
    )
    html = (
        "<!doctype html><html><body>"
        "<p>Effective last trading day: 2024-03-29</p>"
        "<p>Configured end date: 2024-03-31</p>"
        "<p>The total return was 99.00%.</p>"
        "</body></html>"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(html, encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(
        finding.id == "numeric_claim_unverified" and "HTML" in finding.message and "99.00%" in finding.message
        for finding in result.findings
    )


def test_report_qa_rejects_figure_kind_outside_figures_dir(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    asset_path = run_dir / "report_assets/attachments/equity.png"
    asset_path.parent.mkdir(parents=True)
    _write_png(asset_path)
    _write_manifest(
        run_dir,
        [
            {
                "id": "equity",
                "kind": "figure",
                "path": "attachments/equity.png",
                "title": "Equity",
                "caption": "",
                "section": "results",
                "order": 10,
                "mime_type": "image/png",
                "sha256": "",
            }
        ],
    )
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/attachments/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "asset_kind_path_mismatch" for finding in result.findings)


def test_report_qa_allows_total_and_oos_trade_counts_on_same_line(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 2 total trades and 1 OOS trade.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_skips_numbered_markdown_headings(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "## 7. Executive Decision\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_reports_unsafe_source_script_path(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(run_dir, figure, asset_id="equity", title="策略净值")
    manifest = json.loads((run_dir / "report_assets/manifest.json").read_text(encoding="utf-8"))
    manifest["assets"][0]["source"] = {"script": "../plot.py"}
    (run_dir / "report_assets/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    markdown = (
        "# 研究报告\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "![策略净值](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "source_script_path_invalid" for finding in result.findings)


def _write_qa_run(tmp_path):
    spec = StrategySpec.template(strategy_id="qa_case", hypothesis="qa should validate final reports")
    spec.validation.train_period = ["2024-01-02", "2024-01-31"]
    spec.validation.test_period = ["2024-02-01", "2024-03-31"]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": "qa-run", "trade_count": 2, "oos_trade_count": 1, "total_return": 0.2}),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n"
        "2024-01-02,100\n"
        "2024-01-31,110\n"
        "2024-02-29,99\n"
        "2024-03-29,120\n",
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "symbol,side,shares,filled_price,filled_at,fee\n"
        "AAA,BUY,1,10,2024-01-15,0\n"
        "AAA,SELL,1,11,2024-02-15,0\n",
        encoding="utf-8",
    )
    return run_dir


def _write_png(path) -> None:
    path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR4nGNgYGAAAAAEAAHIiY1AAAAAAElFTkSuQmCC"))


def _write_manifest(run_dir, assets: list[dict]) -> None:
    path = run_dir / "report_assets/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "assets": assets}), encoding="utf-8")
