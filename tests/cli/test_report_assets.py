from __future__ import annotations

import json

from click.testing import CliRunner

from oxq.cli.main import main


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


def test_report_asset_add_batch_registers_multiple_assets(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    equity = tmp_path / "equity.png"
    drawdown = tmp_path / "drawdown.png"
    script = tmp_path / "plot.py"
    equity.write_bytes(b"equity")
    drawdown.write_bytes(b"drawdown")
    script.write_text("print('plot')\n", encoding="utf-8")
    batch = tmp_path / "assets.json"
    batch.write_text(
        json.dumps(
            [
                {
                    "id": "equity",
                    "file_path": str(equity),
                    "title": "净值曲线",
                    "caption": "由 equity_curve.csv 生成。",
                    "section": "results",
                    "order": 10,
                    "source_script": str(script),
                    "source_artifacts": ["equity_curve.csv"],
                },
                {
                    "id": "drawdown",
                    "file_path": str(drawdown),
                    "title": "回撤曲线",
                    "section": "risk",
                    "order": 20,
                    "source_script": str(script),
                    "source_artifacts": ["equity_curve.csv"],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["report", "asset", "add-batch", str(run_dir), str(batch)])

    assert result.exit_code == 0, result.output
    assert "Added 2 report assets" in result.output
    assert "equity" in result.output
    assert "drawdown" in result.output
    manifest = json.loads((run_dir / "report_assets/manifest.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in manifest["assets"]] == ["equity", "drawdown"]
    assert manifest["assets"][0]["source"]["script"] == "scripts/plot.py"


def test_report_write_command_is_not_registered(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = CliRunner().invoke(main, ["report", "write", str(run_dir)])

    assert result.exit_code != 0
    assert "No such command 'write'" in result.output


def test_report_help_does_not_advertise_write_command() -> None:
    result = CliRunner().invoke(main, ["report", "--help"])

    assert result.exit_code == 0, result.output
    assert "write" not in result.output
    assert "asset" in result.output
