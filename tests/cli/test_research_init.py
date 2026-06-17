from __future__ import annotations

from click.testing import CliRunner

from oxq.cli.main import main


def test_research_init_creates_workspace_and_preserves_agents_md(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / "AGENTS.md").write_text("user note\n", encoding="utf-8")
        created = runner.invoke(
            main,
            ["research", "init", "--name", "demo", "--data-dir", "~/.oxq/data/market"],
        )

        assert created.exit_code == 0, created.output
        assert (cwd_path / ".open-xquant/workspace.yaml").exists()
        assert (cwd_path / "strategy_specs").is_dir()
        assert (cwd_path / "runs").is_dir()
        assert (cwd_path / "reports").is_dir()
        assert (cwd_path / "experiments.jsonl").exists()
        agents_text = (cwd_path / "AGENTS.md").read_text(encoding="utf-8")
        assert "user note" in agents_text
        assert "open-xquant-workspace:begin" in agents_text

        again = runner.invoke(main, ["research", "init"])
        assert again.exit_code == 0, again.output
        assert (cwd_path / "AGENTS.md").read_text(encoding="utf-8").count("open-xquant-workspace:begin") == 1
