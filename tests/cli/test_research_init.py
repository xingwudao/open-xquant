from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.cli.sdk_bundle import install_workspace_sdk


def test_research_init_creates_workspace_and_preserves_agents_md(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
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
        assert (cwd_path / "runs").is_dir()
        assert (cwd_path / "runs/final").is_dir()
        assert (cwd_path / "comparisons").is_dir()
        assert (cwd_path / "experiments.jsonl").exists()
        assert not (cwd_path / "strategy_specs").exists()
        assert not (cwd_path / "reports").exists()
        agents_text = (cwd_path / "AGENTS.md").read_text(encoding="utf-8")
        assert "user note" in agents_text
        assert "open-xquant-workspace:begin" in agents_text
        assert "open-xquant-subagents:begin" in agents_text
        assert "use the installed `open-xquant` skill first" in agents_text
        assert "Do not run `oxq`" in agents_text
        assert "For open-xquant workflows, prefer SubAgents by default" in agents_text
        assert "If SubAgent tools are unavailable" in agents_text

        again = runner.invoke(main, ["research", "init"])
        assert again.exit_code == 0, again.output
        assert (cwd_path / "AGENTS.md").read_text(encoding="utf-8").count("open-xquant-workspace:begin") == 1
        assert (cwd_path / "AGENTS.md").read_text(encoding="utf-8").count("open-xquant-subagents:begin") == 1


def test_research_init_skips_subagent_policy_for_standalone_profile(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    config_dir = home / ".config" / "open-xquant"
    config_dir.mkdir(parents=True)
    (config_dir / "agent.yaml").write_text("agent_profile: standalone-agent\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / "AGENTS.md").write_text(
            "<!-- open-xquant-subagents:begin -->\nold\n<!-- open-xquant-subagents:end -->\n",
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        agents_text = (cwd_path / "AGENTS.md").read_text(encoding="utf-8")
        assert "open-xquant-workspace:begin" in agents_text
        assert "open-xquant-subagents:begin" not in agents_text
        assert "For open-xquant workflows, prefer SubAgents by default" not in agents_text


def test_research_init_workspace_paths_match_run_centric_layout(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        workspace = yaml.safe_load((cwd_path / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["paths"] == {
            "current_spec": "strategy_spec.yaml",
            "runs_dir": "runs",
            "final_dir": "runs/final",
            "comparisons_dir": "comparisons",
            "experiment_registry": "experiments.jsonl",
            "comparison_registry": "comparisons/comparisons.jsonl",
        }


def test_research_init_repairs_paths_from_existing_workspace_config(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    "paths:",
                    "  specs_dir: strategy_specs",
                    "  runs_dir: runs",
                    "  reports_dir: reports",
                    "  experiment_registry: experiments.jsonl",
                ]
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / "strategy_specs").is_dir()
        assert (cwd_path / "runs").is_dir()
        assert (cwd_path / "reports").is_dir()
        assert (cwd_path / "experiments.jsonl").exists()
        assert not (cwd_path / "runs/final").exists()
        assert not (cwd_path / "comparisons" / "comparisons.jsonl").exists()
        workspace = yaml.safe_load((cwd_path / ".open-xquant" / "workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["paths"]["specs_dir"] == "strategy_specs"


def test_research_init_defaults_to_market_data_directory(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        result = runner.invoke(main, ["research", "init", "--minimal"])

        assert result.exit_code == 0, result.output
        workspace = yaml.safe_load((cwd_path / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["data"]["market_data_dir"] == "~/.oxq/data/market"


def test_research_init_sdk_installs_from_agent_bundle(monkeypatch, tmp_path) -> None:
    installed: list[tuple[Path, Path, bool]] = []

    def install(cwd: Path, venv: Path, *, force: bool = False) -> dict:
        installed.append((cwd, venv, force))
        return {
            "enabled": True,
            "bundle_id": "bundle-test",
            "profile": "full-research",
            "venv": ".venv",
            "runner": ".venv/bin/oxq",
            "python": ".venv/bin/python",
            "wheel_sha256": "wheel-sha",
            "lock_sha256": "lock-sha",
        }

    monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install, raising=False)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        result = runner.invoke(main, ["research", "init", "--sdk"])

        assert result.exit_code == 0, result.output
        assert installed == [(cwd_path.resolve(), cwd_path.resolve() / ".venv", False)]
        workspace = yaml.safe_load((cwd_path / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["sdk"] == {
            "enabled": True,
            "bundle_id": "bundle-test",
            "profile": "full-research",
            "venv": ".venv",
            "runner": ".venv/bin/oxq",
            "python": ".venv/bin/python",
            "wheel_sha256": "wheel-sha",
            "lock_sha256": "lock-sha",
        }


def test_research_init_force_does_not_force_sdk_venv_replacement(monkeypatch, tmp_path) -> None:
    forced: list[bool] = []

    def install(cwd: Path, venv: Path, *, force: bool = False) -> dict:
        del cwd, venv
        forced.append(force)
        return {
            "enabled": True,
            "bundle_id": "bundle-test",
            "profile": "full-research",
            "venv": ".venv",
            "runner": ".venv/bin/oxq",
            "python": ".venv/bin/python",
            "wheel_sha256": "wheel-sha",
            "lock_sha256": "lock-sha",
        }

    monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install, raising=False)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["research", "init", "--sdk", "--force"])

    assert result.exit_code == 0, result.output
    assert forced == [False]


def test_research_init_sdk_allows_custom_venv(monkeypatch, tmp_path) -> None:
    installed: list[Path] = []

    def install(cwd: Path, venv: Path, *, force: bool = False) -> dict:
        del cwd, force
        installed.append(venv)
        return {
            "enabled": True,
            "bundle_id": "bundle-test",
            "profile": "full-research",
            "venv": "envs/oxq",
            "runner": "envs/oxq/bin/oxq",
            "python": "envs/oxq/bin/python",
            "wheel_sha256": "wheel-sha",
            "lock_sha256": "lock-sha",
        }

    monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install, raising=False)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        result = runner.invoke(main, ["research", "init", "--sdk", "--sdk-venv", "envs/oxq"])

        assert result.exit_code == 0, result.output
        assert installed == [cwd_path.resolve() / "envs/oxq"]


def test_research_init_sdk_expands_env_absolute_venv(monkeypatch, tmp_path) -> None:
    installed: list[Path] = []
    target_venv = tmp_path / "external venv"

    def install(cwd: Path, venv: Path, *, force: bool = False) -> dict:
        del cwd, force
        installed.append(venv)
        return {
            "enabled": True,
            "bundle_id": "bundle-test",
            "profile": "full-research",
            "venv": str(target_venv),
            "runner": str(target_venv / "bin/oxq"),
            "python": str(target_venv / "bin/python"),
            "wheel_sha256": "wheel-sha",
            "lock_sha256": "lock-sha",
        }

    monkeypatch.setenv("OXQ_TEST_VENV", str(target_venv))
    monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install, raising=False)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["research", "init", "--sdk", "--sdk-venv", "$OXQ_TEST_VENV"])

    assert result.exit_code == 0, result.output
    assert installed == [target_venv.resolve()]


def test_install_workspace_sdk_rejects_research_directory_as_venv(tmp_path) -> None:
    with pytest.raises(Exception, match="research directory"):
        install_workspace_sdk(tmp_path, tmp_path, force=True)


def test_install_workspace_sdk_rejects_existing_non_venv_path(tmp_path) -> None:
    venv = tmp_path / "not-a-venv"
    venv.mkdir()
    (venv / "README.txt").write_text("project files\n", encoding="utf-8")

    with pytest.raises(Exception, match="non-virtualenv"):
        install_workspace_sdk(tmp_path, venv, force=True)
