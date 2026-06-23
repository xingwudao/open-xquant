from __future__ import annotations

import base64
import json

from click.testing import CliRunner

from oxq.cli.main import main
from oxq.report.assets import add_report_asset
from oxq.report.html import render_markdown_html_report


def test_report_qa_command_prints_facts_and_pass_status(tmp_path) -> None:
    run_dir = _write_cli_qa_run(tmp_path)
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(run_dir, figure, asset_id="equity", title="Equity", section="results", order=10)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 20.00%.\n\n"
        "![Equity](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = CliRunner().invoke(main, ["report", "qa", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "Status: PASS" in result.output
    assert "Configured end date: 2024-03-31" in result.output
    assert "Effective last trading day: 2024-03-29" in result.output


def test_report_qa_command_exits_nonzero_on_fatal_findings(tmp_path) -> None:
    run_dir = _write_cli_qa_run(tmp_path)
    (run_dir / "research_report.md").write_text("# Report\n\n![Missing](report_assets/figures/missing.png)\n", encoding="utf-8")
    (run_dir / "research_report.html").write_text("<!doctype html><html><body></body></html>", encoding="utf-8")

    result = CliRunner().invoke(main, ["report", "qa", str(run_dir)])

    assert result.exit_code == 1
    assert "Status: FAIL" in result.output
    assert "markdown_image_unregistered" in result.output


def test_report_qa_command_can_emit_json(tmp_path) -> None:
    run_dir = _write_cli_qa_run(tmp_path)
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = CliRunner().invoke(main, ["report", "qa", str(run_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["facts"]["configured_end_date"] == "2024-03-31"
    assert payload["facts"]["effective_last_trading_day"] == "2024-03-29"


def test_report_qa_command_leaves_semantic_advisory_checks_to_skill(tmp_path) -> None:
    run_dir = _write_cli_qa_run(tmp_path)
    figure = tmp_path / "equity.png"
    _write_png(figure)
    script = run_dir / "report_assets/scripts/plot.py"
    script.parent.mkdir(parents=True)
    script.write_text('import matplotlib.pyplot as plt\nplt.title("策略净值")\n', encoding="utf-8")
    add_report_asset(run_dir, figure, asset_id="equity", title="策略净值", source_script=script)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 999.00% and Sharpe was 99.00.\n\n"
        "![策略净值](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = CliRunner().invoke(main, ["report", "qa", str(run_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    finding_ids = {finding["id"] for finding in payload["findings"]}
    assert "numeric_claim_unverified" not in finding_ids
    assert payload["warning_count"] == 0


def test_report_qa_command_keeps_source_script_path_validation_deterministic(tmp_path) -> None:
    run_dir = _write_cli_qa_run(tmp_path)
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(run_dir, figure, asset_id="equity", title="Equity")
    manifest_path = run_dir / "report_assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][0]["source"] = {"script": "../plot.py"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = CliRunner().invoke(main, ["report", "qa", str(run_dir), "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    finding_ids = {finding["id"] for finding in payload["findings"]}
    assert "source_script_path_invalid" in finding_ids


def _write_cli_qa_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        "strategy_id: cli_qa\nvalidation:\n  train_period: [2024-01-02, 2024-01-31]\n  test_period: [2024-02-01, 2024-03-31]\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"total_return": 0.2}), encoding="utf-8")
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02,100\n2024-01-31,110\n2024-02-29,99\n2024-03-29,120\n",
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text("symbol,side,shares,filled_price,filled_at,fee\n", encoding="utf-8")
    return run_dir


def _write_png(path) -> None:
    path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR4nGNgYGAAAAAEAAHIiY1AAAAAAElFTkSuQmCC"))
