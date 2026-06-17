from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from oxq.cli.doctor import _check_data
from oxq.cli.main import main


def _write_source(root: Path) -> None:
    skills = root / "agent" / "skills"
    skills.mkdir(parents=True)
    (skills / "strategy-builder.md").write_text(
        "---\nname: strategy-builder\ndescription: Build quant strategies\n---\n\n# Strategy Builder\n",
        encoding="utf-8",
    )


def test_doctor_json_reports_missing_workspace_fix(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(main, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checks"]["agent"]["status"] == "ok"
    assert payload["checks"]["workspace"]["status"] == "missing"
    assert "oxq research init" in payload["fixes"]


def test_doctor_json_fix_outputs_only_json(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["doctor", "--json", "--fix"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checks"]["workspace"]["status"] == "ok"
    assert (work / ".open-xquant" / "workspace.yaml").exists()


def test_doctor_data_check_uses_market_data_directory(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    (home / ".oxq/data").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    result = _check_data()

    assert result["status"] == "warn"
    assert result["path"].endswith(".oxq/data/market")
