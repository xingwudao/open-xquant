from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from oxq.cli.main import main


def _write_source(root: Path, description: str = "Build quant strategies") -> None:
    skills = root / "agent" / "skills"
    skills.mkdir(parents=True)
    (skills / "strategy-builder.md").write_text(
        f"---\nname: strategy-builder\ndescription: >-\n  {description}\n---\n\n# Strategy Builder\n",
        encoding="utf-8",
    )
    (skills / "backtest-runner.md").write_text(
        "---\nname: backtest-runner\ndescription: Run backtests\n---\n\n# Backtest Runner\n",
        encoding="utf-8",
    )


def test_agent_install_all_targets_writes_managed_skills(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    codex_home = home / ".codex-profile"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--all-targets", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert (home / ".agents/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".config/opencode/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".claude/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".cursor/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".openclaw/skills/strategy-builder/SKILL.md").exists()
    assert (codex_home / "AGENTS.md").read_text(encoding="utf-8").count("open-xquant:begin") == 1
    assert "open-xquant:begin" in (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    assert "open-xquant:begin" in (home / ".claude/CLAUDE.md").read_text(encoding="utf-8")

    marker = home / ".agents/skills/strategy-builder/.open-xquant-managed.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["target"] == "codex"

    manifest = home / ".config/open-xquant/agent-install.json"
    targets = json.loads(manifest.read_text(encoding="utf-8"))["targets"]
    assert set(targets) == {"codex", "opencode", "claude-code", "cursor", "openclaw"}


def test_agent_status_json_reports_installed_targets(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(main, ["agent", "status", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["targets"]["opencode"]["installed"] is True
    assert payload["targets"]["opencode"]["skills"]["installed"] == 2
