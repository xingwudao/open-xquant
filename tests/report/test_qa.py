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
