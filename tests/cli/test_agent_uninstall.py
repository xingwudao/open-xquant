from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from oxq.cli.main import main


def _write_source(root: Path) -> None:
    skills = root / "agent" / "skills"
    skills.mkdir(parents=True)
    (skills / "strategy-builder.md").write_text(
        "---\nname: strategy-builder\ndescription: Build quant strategies\n---\n\n# Strategy Builder\n",
        encoding="utf-8",
    )


def test_agent_uninstall_removes_only_managed_files(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    agents = home / ".config/opencode/AGENTS.md"
    agents.write_text(
        "user content\n" + agents.read_text(encoding="utf-8") + "\nmore user content\n",
        encoding="utf-8",
    )
    data_dir = home / ".oxq" / "data"
    data_dir.mkdir(parents=True)

    result = CliRunner().invoke(main, ["agent", "uninstall", "--target", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert not (home / ".config/opencode/skills/strategy-builder").exists()
    assert "open-xquant:begin" not in agents.read_text(encoding="utf-8")
    assert "user content" in agents.read_text(encoding="utf-8")
    assert data_dir.exists()
