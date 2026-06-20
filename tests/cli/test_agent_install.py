from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from oxq.cli.agent_manifest import read_yaml_file
from oxq.cli.main import main


def _write_source(root: Path, skills: dict[str, str] | None = None) -> None:
    skill_descriptions = skills or {
        "strategy-builder": "Build quant strategies",
        "backtest-runner": "Run backtests",
    }
    skills = root / "agent" / "skills"
    skills.mkdir(parents=True)
    for name, description in skill_descriptions.items():
        title = name.replace("-", " ").title()
        (skills / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: >-\n  {description}\n---\n\n# {title}\n",
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
    assert (codex_home / "skills/strategy-builder/SKILL.md").exists()
    assert (home / ".config/opencode/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".claude/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".cursor/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".openclaw/skills/strategy-builder/SKILL.md").exists()
    assert (home / ".trae/skills/strategy-builder/SKILL.md").exists()
    assert (codex_home / "AGENTS.md").read_text(encoding="utf-8").count("open-xquant:begin") == 1
    assert "open-xquant:begin" in (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    assert "open-xquant:begin" in (home / ".claude/CLAUDE.md").read_text(encoding="utf-8")

    marker = codex_home / "skills/strategy-builder/.open-xquant-managed.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["target"] == "codex"

    manifest = home / ".config/open-xquant/agent-install.json"
    targets = json.loads(manifest.read_text(encoding="utf-8"))["targets"]
    assert set(targets) == {"codex", "opencode", "claude-code", "cursor", "openclaw", "trae"}


def test_agent_install_trae_writes_global_skills(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "trae", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    installed = home / ".trae/skills/strategy-builder/SKILL.md"
    assert installed.exists()
    assert "Build quant strategies" in installed.read_text(encoding="utf-8")

    marker = home / ".trae/skills/strategy-builder/.open-xquant-managed.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["target"] == "trae"

    config = read_yaml_file(home / ".config/open-xquant/agent.yaml")
    assert config["preferred_runner"] == f"uv run --project {source.resolve()} oxq"


def test_agent_install_writes_cross_directory_runner(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    raw_config = (home / ".config/open-xquant/agent.yaml").read_text(encoding="utf-8")
    config = read_yaml_file(home / ".config/open-xquant/agent.yaml")
    assert config["preferred_runner"] == f"uv run --project {source.resolve()} oxq"
    assert f"preferred_runner: uv run --project {source.resolve()} oxq" in raw_config

    instructions = (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    assert "agent.yaml" in instructions
    assert "agent-install.json" in instructions
    assert "uv run --project <source.path> oxq" in instructions


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


def test_agent_upgrade_reconciles_added_and_removed_managed_skills(monkeypatch, tmp_path) -> None:
    old_source = tmp_path / "old-source"
    new_source = tmp_path / "new-source"
    home = tmp_path / "home"
    _write_source(old_source, {"old-skill": "Old skill", "kept-skill": "Kept skill"})
    _write_source(new_source, {"kept-skill": "Kept skill updated", "new-skill": "New skill"})
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(old_source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    upgrade = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(new_source), "--yes"],
    )

    assert upgrade.exit_code == 0, upgrade.output
    skills_dir = home / ".config/opencode/skills"
    assert not (skills_dir / "old-skill").exists()
    assert (skills_dir / "kept-skill/SKILL.md").exists()
    assert (skills_dir / "new-skill/SKILL.md").exists()
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert names == {"kept-skill", "new-skill"}


def test_agent_upgrade_preserves_modified_removed_managed_skill(monkeypatch, tmp_path) -> None:
    old_source = tmp_path / "old-source"
    new_source = tmp_path / "new-source"
    home = tmp_path / "home"
    _write_source(old_source, {"old-skill": "Old skill", "kept-skill": "Kept skill"})
    _write_source(new_source, {"kept-skill": "Kept skill updated"})
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(old_source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    old_skill_file = home / ".config/opencode/skills/old-skill/SKILL.md"
    old_skill_file.write_text(old_skill_file.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")

    upgrade = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(new_source), "--yes"],
    )

    assert upgrade.exit_code == 0, upgrade.output
    assert old_skill_file.exists()
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert names == {"old-skill", "kept-skill"}


def test_agent_install_repair_restores_missing_managed_skill(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    skill_file = home / ".config/opencode/skills/strategy-builder/SKILL.md"
    skill_file.unlink()

    repair = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--repair", "--yes"],
    )

    assert repair.exit_code == 0, repair.output
    assert skill_file.exists()


def test_agent_install_repair_preserves_modified_skill_manifest_record(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    skill_file = home / ".config/opencode/skills/strategy-builder/SKILL.md"
    skill_file.write_text(skill_file.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")

    repair = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--repair", "--yes"],
    )

    assert repair.exit_code == 0, repair.output
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert names == {"strategy-builder", "backtest-runner"}


def test_agent_uninstall_requires_explicit_target_or_all_targets(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(main, ["agent", "uninstall", "--yes"])

    assert result.exit_code != 0
    assert "Use --target or --all-targets" in result.output
    assert (home / ".config/opencode/skills/strategy-builder/SKILL.md").exists()


def test_agent_install_rejects_unsafe_skill_names(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    skills = source / "agent" / "skills"
    skills.mkdir(parents=True)
    (skills / "escape.md").write_text(
        "---\nname: ../AGENTS\ndescription: Escape target\n---\n\n# Escape\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code != 0
    assert "invalid skill name" in result.output
    assert not (home / ".config/opencode/AGENTS/SKILL.md").exists()
