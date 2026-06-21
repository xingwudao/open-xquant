from __future__ import annotations

import json

import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.spec.schema import StrategySpec


def test_report_asset_add_registers_figure_with_source_metadata(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    figure = tmp_path / "equity.png"
    figure.write_bytes(b"png")
    script = tmp_path / "plot_equity.py"
    script.write_text("print('plot')\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "report",
            "asset",
            "add",
            str(run_dir),
            str(figure),
            "--id",
            "equity_vs_benchmark",
            "--title",
            "策略净值与基准对比",
            "--caption",
            "由 equity_curve.csv 生成。",
            "--section",
            "results",
            "--order",
            "10",
            "--source-script",
            str(script),
            "--source-artifact",
            "equity_curve.csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Added report asset equity_vs_benchmark" in result.output
    manifest = json.loads((run_dir / "report_assets/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["assets"][0]
    assert entry["id"] == "equity_vs_benchmark"
    assert entry["path"] == "figures/equity_vs_benchmark.png"
    assert entry["source"]["script"] == "scripts/plot_equity.py"
    assert entry["source"]["input_artifacts"] == ["equity_curve.csv"]


def test_report_asset_list_prints_registered_assets(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    figure = tmp_path / "drawdown.png"
    figure.write_bytes(b"png")
    add_result = CliRunner().invoke(
        main,
        [
            "report",
            "asset",
            "add",
            str(run_dir),
            str(figure),
            "--id",
            "drawdown",
            "--title",
            "最大回撤曲线",
        ],
    )
    assert add_result.exit_code == 0, add_result.output

    result = CliRunner().invoke(main, ["report", "asset", "list", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "drawdown" in result.output
    assert "figure" in result.output
    assert "最大回撤曲线" in result.output
    assert "figures/drawdown.png" in result.output
    assert "sha256:" in result.output


def test_report_asset_list_prints_empty_state(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = CliRunner().invoke(main, ["report", "asset", "list", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "No report assets registered." in result.output


def test_report_write_defaults_to_markdown_and_html_in_chinese(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    result = CliRunner().invoke(main, ["report", "write", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "research_report.md" in result.output
    assert "research_report.html" in result.output
    assert (run_dir / "research_report.md").read_text(encoding="utf-8").startswith("# 研究报告: cli_report_case")
    assert (run_dir / "research_report.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_write_can_generate_html_only_in_english(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    result = CliRunner().invoke(main, ["report", "write", str(run_dir), "--lang", "en", "--format", "html"])

    assert result.exit_code == 0, result.output
    assert not (run_dir / "research_report.md").exists()
    html = (run_dir / "research_report.html").read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert "Research Report: cli_report_case" in html


def _write_report_run(tmp_path):
    spec = StrategySpec.template(
        strategy_id="cli_report_case",
        hypothesis="cli should write report bundle",
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
                "run_id": "cli-report-run",
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
