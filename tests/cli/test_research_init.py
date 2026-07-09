from __future__ import annotations

import json
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
        assert (cwd_path / "versions").is_dir()
        assert (cwd_path / "versions" / "v001").is_dir()
        assert (cwd_path / "versions" / "v001" / "version_manifest.json").exists()
        assert (cwd_path / "versions" / "v001" / "phase_state.json").exists()
        for phase_dir in (
            "01_brainstorm",
            "02_idea_audit",
            "03_component_authoring",
            "04_spec_build",
            "05_data_inspection",
            "06_spec_audit",
            "07_compile_preview",
            "08_runtime_audit",
            "09_backtests",
            "10_reports",
        ):
            assert (cwd_path / "versions" / "v001" / phase_dir).is_dir()
        assert (cwd_path / "conversations").is_dir()
        assert (cwd_path / "components").is_dir()
        assert (cwd_path / "final").is_dir()
        assert (cwd_path / "governance").is_dir()
        assert not (cwd_path / "runs").exists()
        assert not (cwd_path / "runs/final").exists()
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
        assert "Version-Governed Artifact Contract" in agents_text
        assert "versions/v001/01_brainstorm/strategy_idea_brief.json" in agents_text
        assert "Do not write root-level `strategy_spec.yaml`" in agents_text
        assert "versions/v001/10_reports/<run_id>/research_report.md" in agents_text
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


def test_research_init_workspace_paths_match_version_governed_layout(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        workspace = yaml.safe_load((cwd_path / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["paths"] == {
            "versions_dir": "versions",
            "conversations_dir": "conversations",
            "components_dir": "components",
            "governance_dir": "governance",
            "runs_dir": "runs",
            "final_dir": "final",
            "comparisons_dir": "comparisons",
            "current_manifest": "current.json",
            "lineage_manifest": "lineage.json",
            "workflow_manifest": "workflow_manifest.json",
            "experiment_registry": "experiments.jsonl",
            "comparison_registry": "comparisons/comparisons.jsonl",
        }
        assert workspace["workflow"]["layout"] == "version_governed"
        assert workspace["workflow"]["default_output_dir"] == "versions/{active_version}/09_backtests"


def test_research_init_version_governed_workspace_does_not_require_legacy_runs_dir(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        init_result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert init_result.exit_code == 0, init_result.output
        assert not (cwd_path / "runs").exists()

        doctor_result = runner.invoke(main, ["doctor", "--json"])

        assert doctor_result.exit_code == 0, doctor_result.output
        payload = json.loads(doctor_result.output)
        assert payload["checks"]["workspace"]["status"] == "ok"
        assert str(cwd_path / "runs") not in payload["checks"]["workspace"]["missing"]


def test_research_init_creates_active_v001_governance_manifests(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert result.exit_code == 0, result.output
        current = json.loads((cwd_path / "current.json").read_text(encoding="utf-8"))
        lineage = json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))
        version_manifest = json.loads(
            (cwd_path / "versions" / "v001" / "version_manifest.json").read_text(encoding="utf-8")
        )
        phase_state = json.loads(
            (cwd_path / "versions" / "v001" / "phase_state.json").read_text(encoding="utf-8")
        )

        assert current["active_version"] == "v001"
        assert current["active_phase"] == "01_brainstorm"
        assert current["active_run"] == ""
        assert lineage["versions"] == [
            {
                "version_id": "v001",
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "active",
            }
        ]
        assert version_manifest["version_id"] == "v001"
        assert version_manifest["strategy_family_id"] == "demo"
        assert version_manifest["active_phase"] == "01_brainstorm"
        assert version_manifest["phase_paths"]["04_spec_build"] == "versions/v001/04_spec_build"
        assert phase_state["version_id"] == "v001"
        assert phase_state["current_phase"] == "01_brainstorm"
        assert phase_state["status"] == "active"


def test_research_init_minimal_creates_governance_manifests(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd

        result = runner.invoke(main, ["research", "init", "--minimal"])

        assert result.exit_code == 0, result.output
        assert json.loads((cwd_path / "current.json").read_text(encoding="utf-8"))["active_version"] == "v001"
        assert (cwd_path / "lineage.json").exists()
        assert (cwd_path / "workflow_manifest.json").exists()
        assert (cwd_path / "versions/v001/04_spec_build").is_dir()


def test_research_init_rejects_unsafe_existing_active_version(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "unsafe-active-version",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": "current.json",
                        "lineage_manifest": "lineage.json",
                        "workflow_manifest": "workflow_manifest.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "../escape"}),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "active_version is unsafe" in result.output
        assert not (cwd_path.parent / "escape").exists()


def test_research_init_preserves_active_phase_in_repaired_version_manifests(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "phase-preserve",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": "current.json",
                        "lineage_manifest": "lineage.json",
                        "workflow_manifest": "workflow_manifest.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v003", "active_phase": "06_spec_audit", "active_run": "run_001"}),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        version_manifest = json.loads(
            (cwd_path / "versions/v003/version_manifest.json").read_text(encoding="utf-8")
        )
        phase_state = json.loads((cwd_path / "versions/v003/phase_state.json").read_text(encoding="utf-8"))
        assert version_manifest["active_phase"] == "06_spec_audit"
        assert phase_state["current_phase"] == "06_spec_audit"


def test_research_init_appends_active_version_to_existing_lineage(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "lineage-preserve",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": "current.json",
                        "lineage_manifest": "lineage.json",
                        "workflow_manifest": "workflow_manifest.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v003"}),
            encoding="utf-8",
        )
        (cwd_path / "lineage.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_family_id": "lineage-preserve",
                    "versions": [
                        {
                            "version_id": "v002",
                            "parent_version_id": "v001",
                            "created_reason": "parameter_change",
                            "status": "superseded",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        lineage = json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))
        assert [item["version_id"] for item in lineage["versions"]] == ["v002", "v003"]
        assert lineage["versions"][0]["created_reason"] == "parameter_change"


def test_research_init_repairs_version_governed_workspace_without_manifest_path_keys(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        (config_dir / "workspace.yaml").write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    "name: legacy-versioned",
                    "paths:",
                    "  versions_dir: versions",
                    "workflow:",
                    "  layout: version_governed",
                    "  default_output_dir: versions/{active_version}/09_backtests",
                ]
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / "current.json").exists()
        assert (cwd_path / "lineage.json").exists()
        assert (cwd_path / "workflow_manifest.json").exists()
        assert (cwd_path / "versions" / "v001" / "phase_state.json").exists()


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


def test_research_init_repairs_hidden_manifest_paths_to_root_manifests(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "hidden-manifest-repair",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": ".open-xquant/current.json",
                        "lineage_manifest": ".open-xquant/lineage.json",
                        "workflow_manifest": ".open-xquant/workflow_manifest.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant" / "current.json").write_text(
            json.dumps({"active_version": "v003"}),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant" / "lineage.json").write_text(
            json.dumps({"versions": [{"version_id": "v003"}]}),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant" / "workflow_manifest.json").write_text(
            json.dumps({"workflow": "legacy-hidden"}),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / "current.json").exists()
        assert (cwd_path / "lineage.json").exists()
        assert (cwd_path / "workflow_manifest.json").exists()
        assert json.loads((cwd_path / "current.json").read_text(encoding="utf-8"))["active_version"] == "v003"
        assert json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))["versions"][0]["version_id"] == "v003"
        assert json.loads((cwd_path / "workflow_manifest.json").read_text(encoding="utf-8"))["workflow"] == "legacy-hidden"
        workspace = yaml.safe_load((cwd_path / ".open-xquant" / "workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["paths"]["current_manifest"] == "current.json"
        assert workspace["paths"]["lineage_manifest"] == "lineage.json"
        assert workspace["paths"]["workflow_manifest"] == "workflow_manifest.json"


def test_research_init_hidden_manifest_source_wins_over_stale_root_manifest(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "hidden-source-wins",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": ".open-xquant/current.json",
                        "lineage_manifest": ".open-xquant/lineage.json",
                        "workflow_manifest": ".open-xquant/workflow_manifest.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "v001"}), encoding="utf-8")
        (cwd_path / ".open-xquant" / "current.json").write_text(
            json.dumps({"active_version": "v004", "active_phase": "05_data_inspection"}),
            encoding="utf-8",
        )
        (cwd_path / "lineage.json").write_text(json.dumps({"versions": [{"version_id": "v001"}]}), encoding="utf-8")
        (cwd_path / ".open-xquant" / "lineage.json").write_text(
            json.dumps({"versions": [{"version_id": "v004"}]}),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant" / "workflow_manifest.json").write_text(
            json.dumps({"layout": "version_governed", "strategy_family_id": "hidden-source-wins"}),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert json.loads((cwd_path / "current.json").read_text(encoding="utf-8"))["active_version"] == "v004"
        lineage = json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))
        assert [item["version_id"] for item in lineage["versions"]] == ["v004"]
        assert (cwd_path / "versions/v004/05_data_inspection").is_dir()


def test_install_workspace_sdk_rejects_existing_non_venv_path(tmp_path) -> None:
    venv = tmp_path / "not-a-venv"
    venv.mkdir()
    (venv / "README.txt").write_text("project files\n", encoding="utf-8")

    with pytest.raises(Exception, match="non-virtualenv"):
        install_workspace_sdk(tmp_path, venv, force=True)
