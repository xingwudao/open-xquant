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
    script.write_text('import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\n', encoding="utf-8")
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


def test_report_qa_does_not_validate_chart_text_rendering(tmp_path) -> None:
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

    assert result.status == "pass"
    assert result.warning_count == 0


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


def test_report_qa_allows_table_date_disclosures(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        "| Effective last trading day | 2024-03-29 |\n"
        "| Configured end date | 2024-03-31 |\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


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


def test_report_qa_matches_percent_claims_to_metric_context(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "max_drawdown": -0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The max drawdown was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_treats_unscoped_drawdown_as_overall(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"max_drawdown": -0.05, "oos_max_drawdown": -0.1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Max drawdown was 10.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "10.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_annualized_return_context_exclusive(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"annualized_return": 0.1, "total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Annualized return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_excess_return_context_exclusive(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "excess_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Excess return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_total_return_context_exclusive(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "benchmark_total_return": 0.07, "excess_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 5.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "5.00%" in finding.message for finding in result.findings)


def test_report_qa_treats_unscoped_total_return_as_overall(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_binds_strategy_and_benchmark_returns_per_claim(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "benchmark_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Strategy total return 5.00%; benchmark total return 5.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "5.00%" in finding.message for finding in result.findings)


def test_report_qa_binds_generic_strategy_return_to_strategy_total_return(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "benchmark_total_return": 0.05, "excess_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The strategy returned 5.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "5.00%" in finding.message for finding in result.findings)


def test_report_qa_parses_signed_positive_percentage_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was +99.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "+99.00%" in finding.message for finding in result.findings)


def test_report_qa_excludes_benchmark_window_strategy_return_from_generic_total(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n"
        "2024-01-31,100\n"
        "2024-03-29,100\n",
        encoding="utf-8",
    )
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 9.09%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.09%" in finding.message for finding in result.findings)


def test_report_qa_matches_monthly_return_claims_to_mentioned_month(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "February return was 10.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "10.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_cost_rate_claims_field_specific(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["cost"] = {"fee_rate": 0.001, "slippage_rate": 0.0005}
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Fee: 0.050%, Slippage: 0.100%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "0.050%" in messages
    assert "0.100%" in messages


def test_report_qa_excludes_fee_min_from_fee_rate_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["cost"] = {"fee_rate": 0.001, "fee_min": 0.0}
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Fee: 0.000%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "0.000%" in finding.message for finding in result.findings)


def test_report_qa_respects_oos_scope_for_percent_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "oos_total_return": -0.1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The OOS total return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_is_oos_percent_claims_scoped_on_mixed_lines(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"is_total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "IS total return 20.00%, OOS total return 10.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "20.00%" in messages
    assert "10.00%" in messages


def test_report_qa_does_not_scope_prior_total_return_to_later_oos_label(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return 10.00%, OOS total return 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_binds_scope_markers_after_percent_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "oos_total_return": -0.1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "20.00% OOS total return.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_restricts_prior_unscoped_return_on_mixed_scope_line(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return 20.00%, OOS total return 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_checks_numbers_inside_ordered_list_items(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "1. Sharpe ratio was 9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.99" in finding.message for finding in result.findings)


def test_report_qa_skips_generated_timestamp_clock_components(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "**Generated**: 2024-03-31 12:34:56 UTC\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_accepts_registered_webp_dimensions(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    figure = tmp_path / "equity.webp"
    _write_webp_vp8x(figure, width=2, height=3)
    add_report_asset(run_dir, figure, asset_id="equity", title="Equity", section="results", order=10)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/figures/equity.webp)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"
    assert not any(finding.id == "image_dimensions_unreadable" for finding in result.findings)


def test_report_qa_requires_date_labels_when_required_dates_are_equal(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["validation"]["test_period"][1] = "2024-03-29"
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    finding_ids = {finding.id for finding in result.findings}
    assert result.status == "fail"
    assert "markdown_configured_end_date_missing" in finding_ids
    assert "html_configured_end_date_missing" in finding_ids


def test_report_qa_allows_strategy_spec_cost_and_cash_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = StrategySpec.from_yaml(str(run_dir / "strategy_spec.yaml"))
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.0005
    spec.execution.initial_cash = 100000
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Fee: 0.100%.\n\n"
        "Slippage: 0.050%.\n\n"
        "Initial Cash: $100,000.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


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


def test_report_qa_preserves_html_table_row_context_for_numbers(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "max_drawdown": -0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
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
        "<table><tr><td>Max drawdown</td><td>20.00%</td></tr></table>"
        "</body></html>"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(html, encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(
        finding.id == "numeric_claim_unverified" and "HTML" in finding.message and "20.00%" in finding.message
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


def test_report_qa_requires_manifest_asset_hash(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    asset_path = run_dir / "report_assets/attachments/notes.txt"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text("notes", encoding="utf-8")
    _write_manifest(
        run_dir,
        [
            {
                "id": "notes",
                "kind": "attachment",
                "path": "attachments/notes.txt",
                "title": "Notes",
                "caption": "",
                "section": "appendix",
                "order": 10,
                "mime_type": "text/plain",
                "sha256": "",
            }
        ],
    )
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "asset_hash_missing" for finding in result.findings)


def test_report_qa_rejects_directory_manifest_asset_paths(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    (run_dir / "report_assets/attachments").mkdir(parents=True)
    _write_manifest(
        run_dir,
        [
            {
                "id": "notes",
                "kind": "attachment",
                "path": "attachments",
                "title": "Notes",
                "caption": "",
                "section": "appendix",
                "order": 10,
                "mime_type": "text/plain",
                "sha256": "sha256:unused",
            }
        ],
    )
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "asset_file_not_regular" for finding in result.findings)


def test_report_qa_rejects_non_object_manifest_entries(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    _write_manifest(run_dir, ["bad"])
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "manifest_asset_invalid" for finding in result.findings)


def test_report_qa_fails_when_metrics_are_missing(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    (run_dir / "metrics.json").unlink()
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "metrics_unreadable" for finding in result.findings)


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


def test_report_qa_keeps_total_and_oos_trade_counts_claim_specific(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 1 total trades and 2 OOS trades.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "1" in messages
    assert "2" in messages


def test_report_qa_keeps_positive_and_negative_month_counts_claim_specific(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Positive months 1, negative months 2.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "1" in messages
    assert "2" in messages


def test_report_qa_does_not_validate_generic_trade_claims_with_oos_count(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 1 trade overall.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "1" in finding.message for finding in result.findings)


def test_report_qa_allows_rounded_plain_number_ratio_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.234})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe Ratio | 1.23\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_parses_signed_positive_plain_number_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe +9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "+9.99" in finding.message for finding in result.findings)


def test_report_qa_checks_leading_decimal_metric_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "9.99 Sharpe ratio.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.99" in finding.message for finding in result.findings)


def test_report_qa_treats_unscoped_ratio_claims_as_overall(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 0.5, "oos_sharpe_ratio": 2.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe 2.00.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "2.00" in finding.message for finding in result.findings)


def test_report_qa_matches_win_rate_claims_only_to_win_rate_metrics(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["cost"] = {"fee_rate": 0.001, "slippage_rate": 0.0005}
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Win rate was 0.10%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "0.10%" in finding.message for finding in result.findings)


def test_report_qa_binds_same_line_ratio_claims_to_labels(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.0, "calmar_ratio": 2.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe 2.00, Calmar 1.00.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "2.00" in messages
    assert "1.00" in messages


def test_report_qa_keeps_trade_counts_out_of_ratio_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"trade_count": 2, "sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 2 total trades and Sharpe 2.00.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "2.00" in finding.message for finding in result.findings)


def test_report_qa_checks_plain_numbers_inside_figure_captions(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"oos_sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Figure 1. OOS Sharpe 9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.99" in finding.message for finding in result.findings)


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


def test_report_qa_parses_comma_formatted_percentage_claims_as_single_value(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] = 10.0
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 1,000.00%.\n"
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


def _write_webp_vp8x(path, *, width: int, height: int) -> None:
    payload = b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    chunk = b"VP8X" + len(payload).to_bytes(4, "little") + payload
    data = b"WEBP" + chunk
    path.write_bytes(b"RIFF" + len(data).to_bytes(4, "little") + data)


def _write_manifest(run_dir, assets: list[dict]) -> None:
    path = run_dir / "report_assets/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "assets": assets}), encoding="utf-8")
