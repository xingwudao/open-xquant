from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from oxq.cli.agent import _upgrade_source
from oxq.cli.main import main


def _write_source(root: Path, body: str) -> None:
    skills = root / "agent" / "skills"
    skills.mkdir(parents=True)
    (skills / "strategy-builder.md").write_text(
        "---\nname: strategy-builder\ndescription: Build quant strategies\n---\n\n"
        f"# Strategy Builder\n\n{body}\n",
        encoding="utf-8",
    )


def test_agent_upgrade_replaces_unmodified_managed_skill(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    installed = home / ".cursor/skills/strategy-builder/SKILL.md"
    assert "new workflow" in installed.read_text(encoding="utf-8")


def test_agent_upgrade_skips_locally_modified_skill(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output

    installed = home / ".cursor/skills/strategy-builder/SKILL.md"
    installed.write_text(installed.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "local edit" in installed.read_text(encoding="utf-8")
    assert "new workflow" not in installed.read_text(encoding="utf-8")
    assert "modified" in result.output


def test_upgrade_source_uses_safe_cache_path_for_path_like_ref(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    source = tmp_path / "cloned"
    _write_source(source, "from git")
    clone_destinations: list[Path] = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("oxq.cli.agent.resolve_source_root", lambda _path: source)

    def fake_run(cmd, check):
        assert check is True
        clone_destinations.append(Path(cmd[-1]).resolve())

    monkeypatch.setattr("oxq.cli.agent.subprocess.run", fake_run)

    result = _upgrade_source(None, "https://example.invalid/repo.git", "..")

    cache_root = (home / ".config/open-xquant/cache/open-xquant").resolve()
    assert result == source
    assert clone_destinations
    assert clone_destinations[0].is_relative_to(cache_root)
    assert clone_destinations[0] != cache_root.parent
