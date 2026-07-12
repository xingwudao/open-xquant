from __future__ import annotations

import json

import pandas as pd
import yaml

from oxq.audit.reproducibility import audit_reproducibility
from oxq.audit.research_bias import audit_research
from oxq.observe.experiment_registry import add_experiment
from oxq.report.html import render_markdown_html_report
from oxq.report.qa import run_report_qa
from oxq.robustness.runner import run_robustness
from oxq.run_digests import publish_run_artifacts, require_current_run_digest
from oxq.spec.compiler import compile_run
from oxq.spec.schema import StrategySpec


def test_governed_monitor_workflow_refreshes_integrity_before_report_qa(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    backtest_dir = workspace / "versions/v001/09_backtests"
    report_phase_dir = workspace / "versions/v001/10_reports"
    data_dir = workspace / "data"
    config_dir = workspace / ".open-xquant"
    for path in (backtest_dir, report_phase_dir, data_dir, config_dir):
        path.mkdir(parents=True)
    monkeypatch.chdir(workspace)

    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"versions_dir": "versions"},
                "workflow": {"layout": "version_governed"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(
        json.dumps({"active_version": "v001"}),
        encoding="utf-8",
    )
    (workspace / "versions/v001/version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "phase_paths": {
                    "09_backtests": "versions/v001/09_backtests",
                    "10_reports": "versions/v001/10_reports",
                },
            }
        ),
        encoding="utf-8",
    )

    dates = pd.bdate_range("2024-01-02", "2024-01-12", tz="UTC")
    pd.DataFrame(
        {
            "open": range(100, 100 + len(dates)),
            "high": range(101, 101 + len(dates)),
            "low": range(99, 99 + len(dates)),
            "close": range(100, 100 + len(dates)),
            "volume": [1000] * len(dates),
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")
    spec = StrategySpec.template(
        strategy_id="governed_monitor",
        hypothesis="monitor artifacts remain reportable",
    )
    spec.validation.train_period = ["2024-01-02", "2024-01-05"]
    spec.validation.test_period = ["2024-01-08", "2024-01-12"]

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=backtest_dir)

    reproducibility = audit_reproducibility(run_dir)
    assert reproducibility["status"] == "pass"
    research_bias = audit_research(run_dir)
    publish_run_artifacts(
        run_dir,
        {
            "reproducibility_audit.json": (json.dumps(reproducibility, indent=2) + "\n").encode(),
            "research_bias_audit.json": (json.dumps(research_bias, indent=2) + "\n").encode(),
        },
    )
    robustness = run_robustness(run_dir)
    assert robustness["status"] in {"robust", "warn", "fragile"}

    entry = add_experiment(
        run_dir,
        registry_path=workspace / "experiments.jsonl",
        backtest_phase_dir=backtest_dir,
        version_id="v001",
    )
    assert "error" not in entry

    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert {
        "reproducibility_audit.json",
        "research_bias_audit.json",
        "robustness.json",
    }.issubset(hashes)
    require_current_run_digest(run_dir)

    report_dir = report_phase_dir / run_dir.name
    report_dir.mkdir()
    (report_dir / "writer_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "version_id": "v001",
                "run_id": run_dir.name,
                "strategy_id": spec.strategy_id,
                "source_run_dir": f"versions/v001/09_backtests/{run_dir.name}",
            }
        ),
        encoding="utf-8",
    )
    markdown = "# Report\n\nEffective last trading day: 2024-01-12\n\nConfigured end date: 2024-01-12\n"
    (report_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown, lang="en"),
        encoding="utf-8",
    )

    qa = run_report_qa(report_dir, include_advisory_checks=False)

    assert qa.status == "pass"
    assert qa.fatal_count == 0
