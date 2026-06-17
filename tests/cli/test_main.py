from __future__ import annotations

from click.testing import CliRunner

from oxq.cli.main import main


def test_robustness_run_exits_nonzero_for_error(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run_robustness(_run_dir: str) -> dict:
        return {"status": "error", "tests": [], "message": "missing data"}

    monkeypatch.setattr("oxq.robustness.run_robustness", fake_run_robustness)

    result = CliRunner().invoke(main, ["robustness", "run", str(run_dir), "--json"])

    assert result.exit_code == 1
    assert "missing data" in result.output


def test_robustness_run_exits_nonzero_for_fragile(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run_robustness(_run_dir: str) -> dict:
        return {
            "status": "fragile",
            "tests": [{"name": "cost_x2", "status": "fail", "message": "sharpe collapsed"}],
            "baseline_sharpe": 1.0,
        }

    monkeypatch.setattr("oxq.robustness.run_robustness", fake_run_robustness)

    result = CliRunner().invoke(main, ["robustness", "run", str(run_dir)])

    assert result.exit_code == 1
    assert "Status: FRAGILE" in result.output
