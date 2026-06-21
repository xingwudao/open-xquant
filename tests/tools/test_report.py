from __future__ import annotations

import json

import yaml

from oxq.spec.schema import StrategySpec
from oxq.tools.report import report_write


def test_report_write_tool_returns_markdown_and_html_outputs(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    result = report_write(str(run_dir), lang="en")

    assert result["status"] == "ok"
    assert result["output"] == result["markdown_output"]
    assert result["markdown_output"].endswith("research_report.md")
    assert result["html_output"].endswith("research_report.html")
    assert (run_dir / "research_report.md").exists()
    assert (run_dir / "research_report.html").exists()
    assert result["strategy_id"] == "tool_report_case"
    assert result["decision"] in {"REJECT", "WATCHLIST", "PAPER TRADING CANDIDATE"}


def _write_report_run(tmp_path):
    spec = StrategySpec.template(
        strategy_id="tool_report_case",
        hypothesis="tool should write report outputs",
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
                "run_id": "tool-report-run",
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
