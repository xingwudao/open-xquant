from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.cli.research import VERSION_PHASE_DIRS
from oxq.cli.sdk_bundle import install_workspace_sdk


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX directory descriptors")
def test_governance_commit_parent_swap_before_first_rename_preserves_journal(
    monkeypatch,
    tmp_path,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="parent-swap")
    destination_parent = workspace / "versions/v001"
    destination = destination_parent / "phase_state.json"
    original_content = destination.read_text(encoding="utf-8")
    replacement_content = original_content.replace('"status": "active"', '"status": "paused"')
    displaced_parent = workspace / "versions/v001-displaced"
    external_parent = workspace / "versions/v001-external"
    original_replace = os.replace
    swapped = False

    def swap_before_replace(source, target, *args, **kwargs) -> None:
        nonlocal swapped
        if not swapped and Path(source).name == destination.name and Path(target).name.startswith(f".{destination.name}.backup-"):
            swapped = True
            original_replace(destination_parent, displaced_parent)
            destination_parent.mkdir()
            destination.write_text("unrelated\n", encoding="utf-8")
        original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(research_module.os, "replace", swap_before_replace)

    with pytest.raises(Exception, match="parent changed|journal preserved"):
        research_module._write_governance_files_atomically(
            workspace,
            {destination: replacement_content},
        )

    assert swapped
    assert destination.read_text(encoding="utf-8") == "unrelated\n"
    journal = workspace / ".open-xquant/governance-transaction.json"
    assert journal.is_file()

    original_replace(destination_parent, external_parent)
    original_replace(displaced_parent, destination_parent)
    monkeypatch.setattr(research_module.os, "replace", original_replace)
    research_module._recover_governance_transaction(workspace)

    assert destination.read_text(encoding="utf-8") == original_content
    assert (external_parent / destination.name).read_text(encoding="utf-8") == "unrelated\n"
    assert not journal.exists()
    assert not list(destination_parent.glob(f".{destination.name}.stage-*"))
    assert not list(destination_parent.glob(f".{destination.name}.backup-*"))


def _workspace_snapshot(root: Path) -> dict[str, bytes | None]:
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None for path in sorted(root.rglob("*"))}


@pytest.mark.parametrize("instruction_file", ["AGENTS.md", "CLAUDE.md"])
@pytest.mark.parametrize("marker", ["open-xquant-workspace", "open-xquant-subagents"])
def test_research_init_rejects_malformed_managed_markers_without_mutation(
    monkeypatch,
    tmp_path,
    instruction_file: str,
    marker: str,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / instruction_file).write_text(
            f"user content\n<!-- {marker}:begin -->\nmanaged content\n",
            encoding="utf-8",
        )
        sdk_calls: list[Path] = []

        def install(cwd: Path, _venv: Path) -> dict[str, object]:
            sdk_calls.append(cwd)
            return {"venv": ".venv"}

        monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init", "--sdk"])

        assert result.exit_code == 1
        assert f"Partial marker block for {marker}" in result.output
        assert sdk_calls == []
        assert _workspace_snapshot(cwd_path) == before


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
        assert "`paths.versions_dir`" in agents_text
        assert "`current.json.active_version`" in agents_text
        assert "`phase_paths`" in agents_text
        assert "Do not write root-level `strategy_spec.yaml`" in agents_text
        assert "<phase_paths.10_reports>/<run_id>/research_report.md" in agents_text
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
        version_manifest = json.loads((cwd_path / "versions" / "v001" / "version_manifest.json").read_text(encoding="utf-8"))
        phase_state = json.loads((cwd_path / "versions" / "v001" / "phase_state.json").read_text(encoding="utf-8"))

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


def test_research_init_version_manifest_honors_custom_versions_dir(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "custom-version-root",
                    "paths": {
                        "versions_dir": "research_versions",
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

        result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert result.exit_code == 0, result.output
        version_manifest = json.loads((cwd_path / "research_versions" / "v001" / "version_manifest.json").read_text(encoding="utf-8"))
        assert (cwd_path / "research_versions/v001/04_spec_build").is_dir()
        assert version_manifest["phase_paths"]["04_spec_build"] == "research_versions/v001/04_spec_build"
        assert not (cwd_path / "versions/v001/04_spec_build").exists()
        workspace = yaml.safe_load((cwd_path / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["workflow"]["default_output_dir"] == "research_versions/{active_version}/09_backtests"
        agents_text = (cwd_path / "AGENTS.md").read_text(encoding="utf-8")
        assert "`research_versions`" in agents_text
        assert "<phase_paths.04_spec_build>/strategy_spec.yaml" in agents_text
        assert "`versions/v001/04_spec_build/strategy_spec.yaml`" not in agents_text


def test_research_init_workspace_instructions_resolve_current_active_version_phase_paths(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "research_versions/v002"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "active-version-instructions",
                    "paths": {"versions_dir": "research_versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "04_spec_build"}),
            encoding="utf-8",
        )
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version_id": "v002",
                    "parent_version_id": "v001",
                    "created_reason": "existing_strategy_version",
                    "status": "active",
                    "active_phase": "04_spec_build",
                    "phase_paths": {
                        phase: f"research_versions/v002/{phase}"
                        for phase in (
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
                        )
                    },
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        agents_text = (cwd_path / "AGENTS.md").read_text(encoding="utf-8")
        assert "`current.json.active_version`" in agents_text
        assert "`research_versions`" in agents_text
        assert "version_manifest.json" in agents_text
        assert "`phase_paths`" in agents_text
        assert "research_versions/v001/" not in agents_text
        assert "research_versions/v002/04_spec_build/strategy_spec.yaml" not in agents_text
        assert "<phase_paths.04_spec_build>/strategy_spec.yaml" in agents_text


def test_research_init_rewrites_root_runs_auto_for_custom_versions_dir(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "custom-version-root",
                    "paths": {
                        "versions_dir": "research_versions",
                        "current_manifest": "current.json",
                        "lineage_manifest": "lineage.json",
                        "workflow_manifest": "workflow_manifest.json",
                    },
                    "workflow": {
                        "layout": "version_governed",
                        "default_output_dir": "runs/auto",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert result.exit_code == 0, result.output
        workspace = yaml.safe_load((cwd_path / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["workflow"]["default_output_dir"] == "research_versions/{active_version}/09_backtests"


def test_research_init_rewrites_nested_root_runs_auto_for_custom_versions_dir(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "custom-version-root",
                    "paths": {
                        "versions_dir": "research_versions",
                        "current_manifest": "current.json",
                        "lineage_manifest": "lineage.json",
                        "workflow_manifest": "workflow_manifest.json",
                    },
                    "workflow": {
                        "layout": "version_governed",
                        "default_output_dir": "runs/auto/runs/runs/{active_version}",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert result.exit_code == 0, result.output
        workspace = yaml.safe_load((cwd_path / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
        assert workspace["workflow"]["default_output_dir"] == "research_versions/{active_version}/09_backtests"


def test_research_init_rejects_unsafe_versions_dir(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "unsafe-version-root",
                    "paths": {
                        "versions_dir": "../escape",
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

        result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert result.exit_code == 1
        assert "workspace paths.versions_dir must be a safe relative path" in result.output
        assert not (cwd_path.parent / "escape").exists()


def test_research_init_rejects_symlinked_versions_dir_escape(tmp_path) -> None:
    runner = CliRunner()
    outside = tmp_path / "outside"
    outside.mkdir()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / "versions_link").symlink_to(outside, target_is_directory=True)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "symlink-version-root",
                    "paths": {
                        "versions_dir": "versions_link",
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

        result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert result.exit_code == 1
        assert "workspace paths.versions_dir must not contain symlink components" in result.output
        assert not (outside / "v001").exists()


def test_research_init_rejects_symlinked_active_version_dir_escape_without_mutation(tmp_path) -> None:
    runner = CliRunner()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("unchanged\n", encoding="utf-8")
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        (cwd_path / "versions").mkdir()
        (cwd_path / "versions/v002").symlink_to(outside, target_is_directory=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "active-version-symlink",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "04_spec_build"}),
            encoding="utf-8",
        )
        (cwd_path / "lineage.json").write_text(
            json.dumps(
                {
                    "versions": [
                        {
                            "version_id": "v002",
                            "parent_version_id": "v001",
                            "created_reason": "signal_semantics_change",
                            "status": "active",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)
        outside_before = _workspace_snapshot(outside)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "active version directory must stay within the workspace" in result.output
        assert _workspace_snapshot(cwd_path) == before
        assert _workspace_snapshot(outside) == outside_before


def test_research_init_preserves_safe_custom_phase_paths_and_creates_their_dirs(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "research_versions/v001"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "custom-phase-path",
                    "paths": {"versions_dir": "research_versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v001", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        identity = {
            "version_id": "v001",
            "parent_version_id": "",
            "created_reason": "initial_strategy_version",
            "status": "active",
        }
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": [identity]}),
            encoding="utf-8",
        )
        custom_spec_path = "research_versions/v001/artifacts/specs/generated"
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    **identity,
                    "active_phase": "04_spec_build",
                    "phase_paths": {"04_spec_build": custom_spec_path},
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        manifest = json.loads((version_dir / "version_manifest.json").read_text(encoding="utf-8"))
        assert manifest["phase_paths"]["04_spec_build"] == custom_spec_path
        assert manifest["phase_paths"]["09_backtests"] == "research_versions/v001/09_backtests"
        assert (cwd_path / custom_spec_path).is_dir()
        assert not (version_dir / "04_spec_build").exists()


def test_research_init_rejects_cross_version_phase_path_without_mutation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v002"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "cross-version-phase",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "04_spec_build"}),
            encoding="utf-8",
        )
        identity = {
            "version_id": "v002",
            "parent_version_id": "v001",
            "created_reason": "signal_semantics_change",
            "status": "active",
        }
        (cwd_path / "lineage.json").write_text(json.dumps({"versions": [identity]}), encoding="utf-8")
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    **identity,
                    "active_phase": "04_spec_build",
                    "phase_paths": {"04_spec_build": "versions/v001/04_spec_build"},
                }
            ),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "phase_paths.04_spec_build must stay within the active version" in result.output
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_rejects_stale_version_manifest_phase_paths_for_custom_versions_dir(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / "research_versions/v001").mkdir(parents=True)
        (cwd_path / "current.json").write_text(
            json.dumps({"schema_version": 1, "active_version": "v001", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        (cwd_path / "research_versions/v001/version_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version_id": "v001",
                    "strategy_family_id": "stale",
                    "parent_version_id": "",
                    "created_reason": "initial_strategy_version",
                    "status": "active",
                    "active_phase": "04_spec_build",
                    "phase_paths": {"04_spec_build": "versions/v001/04_spec_build"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "custom-version-root",
                    "paths": {
                        "versions_dir": "research_versions",
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

        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init", "--name", "demo"])

        assert result.exit_code == 1
        assert "phase_paths.04_spec_build must stay within the active version" in result.output
        assert _workspace_snapshot(cwd_path) == before


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
        (cwd_path / "versions/v003").mkdir(parents=True)
        (cwd_path / "versions/v003/version_manifest.json").write_text(
            json.dumps(
                {
                    "version_id": "v003",
                    "parent_version_id": "v002",
                    "created_reason": "existing_strategy_version",
                    "status": "active",
                    "active_phase": "04_spec_build",
                    "phase_paths": {"04_spec_build": "versions/v003/04_spec_build"},
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        version_manifest = json.loads((cwd_path / "versions/v003/version_manifest.json").read_text(encoding="utf-8"))
        phase_state = json.loads((cwd_path / "versions/v003/phase_state.json").read_text(encoding="utf-8"))
        assert version_manifest["active_phase"] == "06_spec_audit"
        assert phase_state["current_phase"] == "06_spec_audit"


def test_research_init_reconciles_stale_phase_state_to_current_and_version_manifest(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "phase-reconcile"])
        assert created.exit_code == 0, created.output
        current_path = cwd_path / "current.json"
        manifest_path = cwd_path / "versions/v001/version_manifest.json"
        phase_state_path = cwd_path / "versions/v001/phase_state.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        current["active_phase"] = "06_spec_audit"
        current_path.write_text(json.dumps(current), encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["active_phase"] = "04_spec_build"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
        phase_state["current_phase"] = "05_data_inspection"
        phase_state["completed_phases"] = ["01_brainstorm", "02_idea_audit"]
        phase_state_path.write_text(json.dumps(phase_state), encoding="utf-8")

        repaired = runner.invoke(main, ["research", "init"])

        assert repaired.exit_code == 0, repaired.output
        repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        repaired_phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
        assert repaired_manifest["active_phase"] == "06_spec_audit"
        assert repaired_phase_state["current_phase"] == "06_spec_audit"
        assert repaired_phase_state["completed_phases"] == ["01_brainstorm", "02_idea_audit"]


@pytest.mark.parametrize("missing_active_phase", [True, False])
def test_research_init_atomically_defaults_incomplete_current_phase_across_active_state(
    tmp_path,
    missing_active_phase: bool,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "phase-repair"])
        assert created.exit_code == 0, created.output
        current_path = cwd_path / "current.json"
        manifest_path = cwd_path / "versions/v001/version_manifest.json"
        phase_state_path = cwd_path / "versions/v001/phase_state.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        if missing_active_phase:
            current.pop("active_phase")
        else:
            current["active_phase"] = ""
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["active_phase"] = "06_spec_audit"
        phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
        phase_state["current_phase"] = "06_spec_audit"
        current_path.write_text(json.dumps(current), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        phase_state_path.write_text(json.dumps(phase_state), encoding="utf-8")

        repaired = runner.invoke(main, ["research", "init"])

        assert repaired.exit_code == 0, repaired.output
        assert json.loads(current_path.read_text(encoding="utf-8"))["active_phase"] == "01_brainstorm"
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["active_phase"] == "01_brainstorm"
        assert json.loads(phase_state_path.read_text(encoding="utf-8"))["current_phase"] == "01_brainstorm"


def test_research_init_rolls_back_incomplete_phase_reconciliation_on_replace_failure(
    monkeypatch,
    tmp_path,
) -> None:
    import oxq.cli.research as research_module

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "phase-rollback"])
        assert created.exit_code == 0, created.output
        current_path = cwd_path / "current.json"
        manifest_path = cwd_path / "versions/v001/version_manifest.json"
        phase_state_path = cwd_path / "versions/v001/phase_state.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        current.pop("active_phase")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["active_phase"] = "06_spec_audit"
        phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
        phase_state["current_phase"] = "06_spec_audit"
        current_path.write_text(json.dumps(current), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        phase_state_path.write_text(json.dumps(phase_state), encoding="utf-8")
        before = _workspace_snapshot(cwd_path)
        original_replace = research_module._GovernanceMutationParent.replace_file
        failed_once = False

        def fail_phase_state_replace(self, source: str, target: str) -> None:
            nonlocal failed_once
            if self.path / target == phase_state_path and not failed_once:
                failed_once = True
                raise OSError("injected phase-state replace failure")
            original_replace(self, source, target)

        monkeypatch.setattr(
            research_module._GovernanceMutationParent,
            "replace_file",
            fail_phase_state_replace,
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert failed_once
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_repairs_family_identity_from_authoritative_current(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "workspace-name"])
        assert created.exit_code == 0, created.output
        current_path = cwd_path / "current.json"
        lineage_path = cwd_path / "lineage.json"
        workflow_path = cwd_path / "workflow_manifest.json"
        manifest_path = cwd_path / "versions/v001/version_manifest.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        current["strategy_family_id"] = "authoritative-family"
        current_path.write_text(json.dumps(current), encoding="utf-8")
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        lineage["strategy_family_id"] = "stale-family"
        lineage["versions"][0]["strategy_family_id"] = "stale-family"
        lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        workflow["strategy_family_id"] = "stale-family"
        workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["strategy_family_id"] = "stale-family"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        repaired = runner.invoke(main, ["research", "init"])

        assert repaired.exit_code == 0, repaired.output
        assert json.loads(current_path.read_text(encoding="utf-8"))["strategy_family_id"] == "authoritative-family"
        repaired_lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        assert repaired_lineage["strategy_family_id"] == "authoritative-family"
        assert repaired_lineage["versions"][0]["strategy_family_id"] == "authoritative-family"
        assert json.loads(workflow_path.read_text(encoding="utf-8"))["strategy_family_id"] == "authoritative-family"
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["strategy_family_id"] == "authoritative-family"


@pytest.mark.parametrize("missing_family", [True, False])
def test_research_init_repairs_missing_current_family_from_workspace_identity(
    tmp_path,
    missing_family: bool,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "family-reject"])
        assert created.exit_code == 0, created.output
        current_path = cwd_path / "current.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        if missing_family:
            current.pop("strategy_family_id")
        else:
            current["strategy_family_id"] = ""
        current_path.write_text(json.dumps(current), encoding="utf-8")
        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert json.loads(current_path.read_text(encoding="utf-8"))["strategy_family_id"] == "family-reject"


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("schema_version", None, True),
        ("schema_version", "1", False),
        ("layout", "legacy", False),
        ("strategy_family_id", "", False),
        ("paths", [], False),
    ],
)
def test_research_init_fully_validates_existing_workflow_manifest_before_mutation(
    monkeypatch,
    tmp_path,
    field: str,
    value: object,
    remove: bool,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "workflow-schema"])
        assert created.exit_code == 0, created.output
        workflow_path = cwd_path / "workflow_manifest.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        if remove:
            workflow.pop(field)
        else:
            workflow[field] = value
        workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
        sdk_calls: list[Path] = []

        def install(cwd: Path, _venv: Path) -> dict[str, object]:
            sdk_calls.append(cwd)
            return {"venv": ".venv"}

        monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init", "--sdk"])

        assert result.exit_code == 1
        assert "workflow_manifest.json schema is invalid" in result.output
        assert sdk_calls == []
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("schema_version", None, True),
        ("version_id", "v002", False),
        ("current_phase", "11_unknown", False),
        ("status", "superseded", False),
        ("completed_phases", "01_brainstorm", False),
        ("blocked_phase", [], False),
    ],
)
def test_research_init_fully_validates_existing_phase_state_before_mutation(
    monkeypatch,
    tmp_path,
    field: str,
    value: object,
    remove: bool,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "phase-schema"])
        assert created.exit_code == 0, created.output
        phase_state_path = cwd_path / "versions/v001/phase_state.json"
        phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
        if remove:
            phase_state.pop(field)
        else:
            phase_state[field] = value
        phase_state_path.write_text(json.dumps(phase_state), encoding="utf-8")
        sdk_calls: list[Path] = []

        def install(cwd: Path, _venv: Path) -> dict[str, object]:
            sdk_calls.append(cwd)
            return {"venv": ".venv"}

        monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init", "--sdk"])

        assert result.exit_code == 1
        assert "phase_state.json schema is invalid" in result.output
        assert sdk_calls == []
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_validates_default_phase_state_when_current_manifest_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v001"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "default-phase-preflight",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (version_dir / "phase_state.json").write_text(
            json.dumps({"schema_version": 1, "version_id": "v001"}),
            encoding="utf-8",
        )
        sdk_calls: list[Path] = []

        def install(cwd: Path, _venv: Path) -> dict[str, object]:
            sdk_calls.append(cwd)
            return {"venv": ".venv"}

        monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init", "--sdk"])

        assert result.exit_code == 1
        assert "phase_state.json schema is invalid" in result.output
        assert sdk_calls == []
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_rejects_invalid_current_active_phase_without_mutation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "invalid-active-phase"])
        assert created.exit_code == 0, created.output
        current_path = cwd_path / "current.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        current["active_phase"] = "11_unknown"
        current_path.write_text(json.dumps(current), encoding="utf-8")
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "current.json active_phase is invalid" in result.output
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize(
    ("created_reason", "parent_version_id"),
    [
        ("initial_strategy_version", "v000"),
        ("signal_semantics_change", None),
        ("signal_semantics_change", ""),
    ],
)
def test_research_init_enforces_created_reason_parent_contract_without_mutation(
    tmp_path,
    created_reason: str,
    parent_version_id: str | None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        created = runner.invoke(main, ["research", "init", "--name", "parent-contract"])
        assert created.exit_code == 0, created.output
        lineage_path = cwd_path / "lineage.json"
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        lineage["versions"][0]["created_reason"] = created_reason
        lineage["versions"][0]["parent_version_id"] = parent_version_id
        lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
        manifest_path = cwd_path / "versions/v001/version_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["created_reason"] = created_reason
        manifest["parent_version_id"] = parent_version_id
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "lineage identity" in result.output
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_rejects_active_version_missing_lineage_metadata_without_mutation(tmp_path) -> None:
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

        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "v003" in result.output
        assert "version_manifest.json" in result.output
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize("lineage_status", ["draft", None])
def test_research_init_rejects_unknown_or_missing_lineage_status_without_mutation(
    tmp_path,
    lineage_status: str | None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v002"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "lineage-status-validation",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": "current.json",
                        "lineage_manifest": "lineage.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        lineage_identity = {
            "version_id": "v002",
            "parent_version_id": "v001",
            "created_reason": "signal_semantics_change",
        }
        if lineage_status is not None:
            lineage_identity["status"] = lineage_status
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": [lineage_identity]}),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "status must be active or superseded" in result.output
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_appends_active_version_from_matching_manifest(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v003"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "lineage-migrate",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v003", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        (cwd_path / "lineage.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_family_id": "lineage-migrate",
                    "versions": [
                        {
                            "version_id": "v002",
                            "parent_version_id": "v001",
                            "created_reason": "parameter_change",
                            "status": "active",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        identity = {
            "version_id": "v003",
            "parent_version_id": "v002",
            "created_reason": "signal_semantics_change",
            "status": "active",
        }
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_family_id": "lineage-migrate",
                    **identity,
                    "active_phase": "06_spec_audit",
                    "phase_paths": {"09_backtests": "versions/v003/09_backtests"},
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        lineage = json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))
        assert lineage["versions"] == [
            {
                "version_id": "v002",
                "parent_version_id": "v001",
                "created_reason": "parameter_change",
                "status": "superseded",
            },
            identity,
        ]


@pytest.mark.parametrize(
    "lineage_versions",
    [
        [
            {
                "version_id": "v002",
                "parent_version_id": "v001",
                "created_reason": "signal_semantics_change",
                "status": "superseded",
            }
        ],
        [
            {
                "version_id": "v001",
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "superseded",
            },
            {
                "version_id": "v002",
                "parent_version_id": "v001",
                "created_reason": "signal_semantics_change",
                "status": "superseded",
            },
        ],
    ],
)
def test_research_init_repairs_current_version_to_only_active_lineage_entry(
    tmp_path,
    lineage_versions: list[dict[str, str]],
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v002"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "lineage-repair",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": lineage_versions}),
            encoding="utf-8",
        )
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    "version_id": "v002",
                    "parent_version_id": "v001",
                    "created_reason": "signal_semantics_change",
                    "status": "active",
                    "active_phase": "06_spec_audit",
                    "phase_paths": {"09_backtests": "versions/v002/09_backtests"},
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        lineage = json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))
        active = [entry for entry in lineage["versions"] if entry["status"] == "active"]
        assert active == [
            {
                "version_id": "v002",
                "parent_version_id": "v001",
                "created_reason": "signal_semantics_change",
                "status": "active",
            }
        ]


def test_research_init_preserves_active_lineage_identity_when_manifest_is_missing(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "lineage-bootstrap",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        lineage_identity = {
            "version_id": "v002",
            "parent_version_id": "v001",
            "created_reason": "signal_semantics_change",
            "status": "active",
        }
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": [lineage_identity]}),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        version_manifest = json.loads((cwd_path / "versions/v002/version_manifest.json").read_text(encoding="utf-8"))
        assert {key: version_manifest[key] for key in ("version_id", "parent_version_id", "created_reason", "status")} == lineage_identity


def test_research_init_accepts_matching_nullable_initial_parent_idempotently(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v001"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "nullable-initial-parent",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_family_id": "nullable-initial-parent",
                    "active_version": "v001",
                    "active_phase": "01_brainstorm",
                    "active_run": "",
                }
            ),
            encoding="utf-8",
        )
        identity = {
            "version_id": "v001",
            "parent_version_id": None,
            "created_reason": "initial_strategy_version",
            "status": "active",
        }
        (cwd_path / "lineage.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_family_id": "nullable-initial-parent",
                    "versions": [identity],
                }
            ),
            encoding="utf-8",
        )
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_family_id": "nullable-initial-parent",
                    **identity,
                    "active_phase": "01_brainstorm",
                    "source_conversation": "",
                    "phase_paths": {phase: f"versions/v001/{phase}" for phase in VERSION_PHASE_DIRS},
                }
            ),
            encoding="utf-8",
        )

        first = runner.invoke(main, ["research", "init"])

        assert first.exit_code == 0, first.output
        after_first = _workspace_snapshot(cwd_path)
        second = runner.invoke(main, ["research", "init"])
        assert second.exit_code == 0, second.output
        assert _workspace_snapshot(cwd_path) == after_first
        lineage = json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))
        manifest = json.loads((version_dir / "version_manifest.json").read_text(encoding="utf-8"))
        assert lineage["versions"][0]["parent_version_id"] is None
        assert manifest["parent_version_id"] is None


@pytest.mark.parametrize("missing_from", ["lineage", "version_manifest"])
def test_research_init_rejects_missing_initial_parent_identity_key(
    tmp_path,
    missing_from: str,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v001"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "missing-initial-parent",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v001", "active_phase": "01_brainstorm"}),
            encoding="utf-8",
        )
        lineage_identity = {
            "version_id": "v001",
            "parent_version_id": None,
            "created_reason": "initial_strategy_version",
            "status": "active",
        }
        manifest_identity = dict(lineage_identity)
        if missing_from == "lineage":
            lineage_identity.pop("parent_version_id")
        else:
            manifest_identity.pop("parent_version_id")
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": [lineage_identity]}),
            encoding="utf-8",
        )
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    **manifest_identity,
                    "active_phase": "01_brainstorm",
                    "phase_paths": {phase: f"versions/v001/{phase}" for phase in VERSION_PHASE_DIRS},
                }
            ),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "lineage identity" in result.output
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize(
    "lineage_identity",
    [
        {"version_id": "v002", "created_reason": "signal_semantics_change", "status": "active"},
        {"version_id": "v002", "parent_version_id": "v001", "status": "active"},
        {
            "version_id": "v002",
            "parent_version_id": None,
            "created_reason": "signal_semantics_change",
            "status": "active",
        },
        {
            "version_id": "v002",
            "parent_version_id": 1,
            "created_reason": "signal_semantics_change",
            "status": "active",
        },
        {
            "version_id": "v002",
            "parent_version_id": "v001",
            "created_reason": "",
            "status": "active",
        },
    ],
)
def test_research_init_rejects_incomplete_active_lineage_identity_before_bootstrap(
    tmp_path,
    lineage_identity: dict[str, object],
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "incomplete-lineage-bootstrap",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": [lineage_identity]}),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "active version v002 lineage identity is incomplete" in result.output
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize(
    ("manifest_status", "lineage_entry"),
    [
        (
            "superseded",
            {
                "version_id": "v002",
                "parent_version_id": "v001",
                "created_reason": "signal_semantics_change",
                "status": "active",
            },
        ),
        (
            "active",
            {
                "version_id": "v002",
                "parent_version_id": "v001",
                "created_reason": "different_identity",
                "status": "superseded",
            },
        ),
    ],
)
def test_research_init_rejects_untrustworthy_lineage_status_repair_without_mutation(
    tmp_path,
    manifest_status: str,
    lineage_entry: dict[str, str],
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v002"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "lineage-reject",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": [lineage_entry]}),
            encoding="utf-8",
        )
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    "version_id": "v002",
                    "parent_version_id": "v001",
                    "created_reason": "signal_semantics_change",
                    "status": manifest_status,
                    "active_phase": "06_spec_audit",
                    "phase_paths": {"09_backtests": "versions/v002/09_backtests"},
                }
            ),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "v002" in result.output
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize(
    ("identity_field", "lineage_value"),
    [
        ("parent_version_id", "v000"),
        ("created_reason", "different_identity"),
    ],
)
def test_research_init_rejects_active_lineage_identity_mismatch_without_mutation(
    tmp_path,
    identity_field: str,
    lineage_value: str,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions/v002"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "lineage-identity-reject",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v002", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        lineage_identity = {
            "version_id": "v002",
            "parent_version_id": "v001",
            "created_reason": "signal_semantics_change",
            "status": "active",
        }
        lineage_identity[identity_field] = lineage_value
        (cwd_path / "lineage.json").write_text(
            json.dumps({"versions": [lineage_identity]}),
            encoding="utf-8",
        )
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    "version_id": "v002",
                    "parent_version_id": "v001",
                    "created_reason": "signal_semantics_change",
                    "status": "active",
                    "active_phase": "06_spec_audit",
                    "phase_paths": {"09_backtests": "versions/v002/09_backtests"},
                }
            ),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "lineage identity does not match version_manifest.json" in result.output
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_rejects_multiple_active_lineage_entries_without_mutation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant/workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "multiple-active",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "v003"}), encoding="utf-8")
        (cwd_path / "lineage.json").write_text(
            json.dumps(
                {
                    "versions": [
                        {"version_id": "v002", "status": "active"},
                        {"version_id": "v003", "status": "active"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "multiple active" in result.output.lower()
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize("manifest_name", ["current.json", "lineage.json", "workflow_manifest.json"])
def test_research_init_rejects_corrupt_governance_manifest_without_mutation(
    tmp_path,
    manifest_name: str,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "corrupt-manifest",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": "current.json",
                        "lineage_manifest": "lineage.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        corrupt_bytes = b'{"schema_version": 1, invalid\n'
        (cwd_path / manifest_name).write_bytes(corrupt_bytes)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert f"{manifest_name} must contain a valid JSON object" in result.output
        assert _workspace_snapshot(cwd_path) == before
        assert (cwd_path / manifest_name).read_bytes() == corrupt_bytes


@pytest.mark.parametrize("manifest_name", ["version_manifest.json", "phase_state.json"])
def test_research_init_rejects_corrupt_active_version_manifest_without_mutation(
    tmp_path,
    manifest_name: str,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "versions" / "v003"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "corrupt-version-manifest",
                    "paths": {"versions_dir": "versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v003", "active_phase": "04_spec_build"}),
            encoding="utf-8",
        )
        if manifest_name == "phase_state.json":
            (version_dir / "version_manifest.json").write_text(
                json.dumps(
                    {
                        "version_id": "v003",
                        "parent_version_id": "v002",
                        "created_reason": "existing_strategy_version",
                        "status": "active",
                        "phase_paths": {"04_spec_build": "versions/v003/04_spec_build"},
                    }
                ),
                encoding="utf-8",
            )
        corrupt_bytes = b'{"schema_version": 1, invalid\n'
        (version_dir / manifest_name).write_bytes(corrupt_bytes)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert f"{manifest_name} must contain a valid JSON object" in result.output
        assert _workspace_snapshot(cwd_path) == before
        assert (version_dir / manifest_name).read_bytes() == corrupt_bytes


@pytest.mark.parametrize(
    "invalid_case",
    ["versions_dir", "governance_dir", "experiment_registry", "active_version"],
)
def test_research_init_preflight_validation_is_mutation_free(
    monkeypatch,
    tmp_path,
    invalid_case: str,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        paths = {"versions_dir": "versions"}
        if invalid_case != "active_version":
            paths[invalid_case] = "../escape"
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "preflight-mutation-free",
                    "paths": paths,
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        if invalid_case == "active_version":
            (cwd_path / "current.json").write_text(
                json.dumps({"active_version": "../escape"}),
                encoding="utf-8",
            )

        def install(cwd: Path, _venv: Path) -> dict[str, object]:
            (cwd / "sdk-mutated").write_text("called\n", encoding="utf-8")
            return {"venv": ".venv"}

        monkeypatch.setattr("oxq.cli.research.install_workspace_sdk", install)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init", "--sdk"])

        assert result.exit_code == 1
        assert _workspace_snapshot(cwd_path) == before


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


def _write_hidden_manifest_migration_workspace(cwd: Path) -> None:
    config_dir = cwd / ".open-xquant"
    config_dir.mkdir()
    (config_dir / "workspace.yaml").write_text(
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
    (config_dir / "current.json").write_text(
        json.dumps({"active_version": "v003"}),
        encoding="utf-8",
    )
    (config_dir / "lineage.json").write_text(
        json.dumps(
            {
                "versions": [
                    {
                        "version_id": "v003",
                        "parent_version_id": "v002",
                        "created_reason": "existing_strategy_version",
                        "status": "active",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout": "version_governed",
                "strategy_family_id": "hidden-manifest-repair",
                "paths": {
                    "versions_dir": "versions",
                    "current_manifest": ".open-xquant/current.json",
                    "lineage_manifest": ".open-xquant/lineage.json",
                    "workflow_manifest": ".open-xquant/workflow_manifest.json",
                },
                "workflow": "legacy-hidden",
            }
        ),
        encoding="utf-8",
    )
    (cwd / "current.json").write_text(
        json.dumps({"active_version": ""}),
        encoding="utf-8",
    )
    (cwd / "lineage.json").write_text(
        json.dumps({"versions": []}),
        encoding="utf-8",
    )
    (cwd / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout": "version_governed",
                "strategy_family_id": "stale-root",
                "paths": {
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
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "interrupted_destination",
    ["current.json", "lineage.json", "workflow_manifest.json", "workspace.yaml"],
)
@pytest.mark.parametrize("boundary", ["backup", "install"])
def test_hidden_manifest_migration_recovers_each_publication_on_second_init(
    monkeypatch,
    tmp_path,
    interrupted_destination: str,
    boundary: str,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_hidden_manifest_migration_workspace(workspace)
    original_replace = research_module._GovernanceMutationParent.replace_file
    original_recover = research_module._recover_governance_transaction
    interrupted = False

    def interrupt_after_install(self, source: str, target: str) -> None:
        nonlocal interrupted
        original_replace(self, source, target)
        target_name = target
        at_boundary = (boundary == "backup" and target_name.startswith(f".{interrupted_destination}.backup-")) or (
            boundary == "install" and ".stage-" in source and target_name == interrupted_destination
        )
        if not interrupted and at_boundary:
            interrupted = True
            raise KeyboardInterrupt("injected hidden migration interruption")

    def terminate_rollback(root: Path) -> None:
        if (root / ".open-xquant/governance-transaction.json").exists():
            raise SystemExit("injected rollback interruption")
        original_recover(root)

    monkeypatch.setattr(
        research_module._GovernanceMutationParent,
        "replace_file",
        interrupt_after_install,
    )
    monkeypatch.setattr(
        research_module,
        "_recover_governance_transaction",
        terminate_rollback,
    )

    with pytest.raises(SystemExit, match="rollback interruption"):
        research_module.initialize_workspace(workspace)

    assert interrupted
    assert (workspace / ".open-xquant/governance-transaction.json").is_file()
    monkeypatch.setattr(
        research_module._GovernanceMutationParent,
        "replace_file",
        original_replace,
    )
    monkeypatch.setattr(
        research_module,
        "_recover_governance_transaction",
        original_recover,
    )

    research_module.initialize_workspace(workspace)

    workspace_config = yaml.safe_load((workspace / ".open-xquant/workspace.yaml").read_text(encoding="utf-8"))
    assert workspace_config["paths"]["current_manifest"] == "current.json"
    assert json.loads((workspace / "current.json").read_text(encoding="utf-8"))["active_version"] == "v003"
    assert not (workspace / ".open-xquant/governance-transaction.json").exists()


def test_hidden_manifest_migration_publishes_workspace_config_last(
    monkeypatch,
    tmp_path,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_hidden_manifest_migration_workspace(workspace)
    original_replace = research_module._GovernanceMutationParent.replace_file
    installs: list[tuple[str, str]] = []

    def record_installs(self, source: str, target: str) -> None:
        if ".stage-" in source:
            transaction_id = source.rsplit(".stage-", 1)[1]
            installs.append((transaction_id, target))
        original_replace(self, source, target)

    monkeypatch.setattr(
        research_module._GovernanceMutationParent,
        "replace_file",
        record_installs,
    )

    research_module.initialize_workspace(workspace)

    migration_ids = [transaction_id for transaction_id, name in installs if name == "workspace.yaml"]
    assert len(migration_ids) == 1
    migration_installs = [name for transaction_id, name in installs if transaction_id == migration_ids[0]]
    assert migration_installs == [
        "current.json",
        "lineage.json",
        "workflow_manifest.json",
        "workspace.yaml",
    ]


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
            json.dumps(
                {
                    "versions": [
                        {
                            "version_id": "v003",
                            "parent_version_id": "v002",
                            "created_reason": "existing_strategy_version",
                            "status": "active",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant" / "workflow_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "layout": "version_governed",
                    "strategy_family_id": "hidden-manifest-repair",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": ".open-xquant/current.json",
                        "lineage_manifest": ".open-xquant/lineage.json",
                        "workflow_manifest": ".open-xquant/workflow_manifest.json",
                    },
                    "workflow": "legacy-hidden",
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / "current.json").exists()
        assert (cwd_path / "lineage.json").exists()
        assert (cwd_path / "workflow_manifest.json").exists()
        assert json.loads((cwd_path / "current.json").read_text(encoding="utf-8"))["active_version"] == "v003"
        assert json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))["versions"][0]["version_id"] == "v003"
        repaired_workflow = json.loads((cwd_path / "workflow_manifest.json").read_text(encoding="utf-8"))
        assert repaired_workflow["workflow"] == "legacy-hidden"
        assert repaired_workflow["paths"]["current_manifest"] == "current.json"
        assert repaired_workflow["paths"]["lineage_manifest"] == "lineage.json"
        assert repaired_workflow["paths"]["workflow_manifest"] == "workflow_manifest.json"
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
            json.dumps(
                {
                    "versions": [
                        {
                            "version_id": "v004",
                            "parent_version_id": "v003",
                            "created_reason": "existing_strategy_version",
                            "status": "active",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant" / "workflow_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "layout": "version_governed",
                    "strategy_family_id": "hidden-source-wins",
                    "paths": {"versions_dir": "versions"},
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert json.loads((cwd_path / "current.json").read_text(encoding="utf-8"))["active_version"] == "v004"
        lineage = json.loads((cwd_path / "lineage.json").read_text(encoding="utf-8"))
        assert [item["version_id"] for item in lineage["versions"]] == ["v004"]
        assert (cwd_path / "versions/v004/05_data_inspection").is_dir()


def test_research_init_hidden_current_ignores_stale_root_version_artifacts(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        stale_version = cwd_path / "versions/v001"
        config_dir.mkdir()
        stale_version.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "hidden-precedence",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": ".open-xquant/current.json",
                        "lineage_manifest": ".open-xquant/lineage.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "v001"}), encoding="utf-8")
        (config_dir / "current.json").write_text(
            json.dumps({"active_version": "v004", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        (config_dir / "lineage.json").write_text(
            json.dumps(
                {
                    "versions": [
                        {
                            "version_id": "v004",
                            "parent_version_id": "v003",
                            "created_reason": "existing_strategy_version",
                            "status": "active",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        corrupt = b'{"schema_version": 1, invalid\n'
        (stale_version / "version_manifest.json").write_bytes(corrupt)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        assert json.loads((cwd_path / "current.json").read_text(encoding="utf-8"))["active_version"] == "v004"
        assert (stale_version / "version_manifest.json").read_bytes() == corrupt


def test_research_init_preflights_hidden_active_version_before_manifest_migration(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        hidden_version_dir = cwd_path / "versions/v004"
        config_dir.mkdir()
        hidden_version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "hidden-preflight",
                    "paths": {
                        "versions_dir": "versions",
                        "current_manifest": ".open-xquant/current.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v001", "active_phase": "01_brainstorm"}),
            encoding="utf-8",
        )
        (config_dir / "current.json").write_text(
            json.dumps({"active_version": "v004", "active_phase": "06_spec_audit"}),
            encoding="utf-8",
        )
        corrupt_bytes = b'{"schema_version": 1, invalid\n'
        (hidden_version_dir / "version_manifest.json").write_bytes(corrupt_bytes)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "version_manifest.json must contain a valid JSON object" in result.output
        assert _workspace_snapshot(cwd_path) == before
        assert (hidden_version_dir / "version_manifest.json").read_bytes() == corrupt_bytes


def test_install_workspace_sdk_rejects_existing_non_venv_path(tmp_path) -> None:
    venv = tmp_path / "not-a-venv"
    venv.mkdir()
    (venv / "README.txt").write_text("project files\n", encoding="utf-8")

    with pytest.raises(Exception, match="non-virtualenv"):
        install_workspace_sdk(tmp_path, venv, force=True)


@pytest.mark.parametrize("versions_dir", [None, "", 7])
@pytest.mark.parametrize("explicit_layout", [False, True])
def test_research_init_rejects_malformed_configured_versions_dir_without_mutation(
    tmp_path,
    versions_dir: object,
    explicit_layout: bool,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        workspace: dict[str, object] = {
            "schema_version": 1,
            "name": "invalid-version-root",
            "paths": {"versions_dir": versions_dir},
        }
        if explicit_layout:
            workspace["workflow"] = {"layout": "version_governed"}
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(workspace, sort_keys=False),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "workspace paths.versions_dir must be a non-empty string" in result.output
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_rejects_workspace_root_drift_without_mutation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        initialized = runner.invoke(main, ["research", "init", "--name", "root-drift"])
        assert initialized.exit_code == 0, initialized.output
        workspace_path = cwd_path / ".open-xquant/workspace.yaml"
        workspace = yaml.safe_load(workspace_path.read_text(encoding="utf-8"))
        workspace["paths"]["versions_dir"] = "research_versions"
        workspace_path.write_text(
            yaml.safe_dump(workspace, sort_keys=False),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "workflow_manifest.json paths.versions_dir does not match workspace config" in result.output
        assert "migration" in result.output
        assert _workspace_snapshot(cwd_path) == before
        assert not (cwd_path / "research_versions").exists()


def test_research_init_rejects_internal_versions_dir_symlink_without_mutation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        internal_target = cwd_path / "internal_versions"
        config_dir.mkdir()
        internal_target.mkdir()
        (cwd_path / "versions_link").symlink_to(internal_target, target_is_directory=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "internal-symlink",
                    "paths": {"versions_dir": "versions_link"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "workspace paths.versions_dir must not contain symlink components" in result.output
        assert _workspace_snapshot(cwd_path) == before
        assert not (internal_target / "v001").exists()


def test_research_init_rejects_incomplete_path_snapshot_before_relocation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        initialized = runner.invoke(main, ["research", "init", "--name", "snapshot-drift"])
        assert initialized.exit_code == 0, initialized.output
        workspace_path = cwd_path / ".open-xquant/workspace.yaml"
        workspace = yaml.safe_load(workspace_path.read_text(encoding="utf-8"))
        workspace["paths"]["versions_dir"] = "relocated_versions"
        workspace_path.write_text(
            yaml.safe_dump(workspace, sort_keys=False),
            encoding="utf-8",
        )
        workflow_path = cwd_path / "workflow_manifest.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        workflow["paths"].pop("versions_dir")
        workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "workflow_manifest.json schema is invalid" in result.output
        assert _workspace_snapshot(cwd_path) == before
        assert not (cwd_path / "relocated_versions").exists()


def test_research_init_persists_complete_effective_path_snapshot(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "effective-paths",
                    "paths": {"versions_dir": "research_versions"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 0, result.output
        workflow = json.loads((cwd_path / "workflow_manifest.json").read_text(encoding="utf-8"))
        assert workflow["paths"] == {
            "versions_dir": "research_versions",
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


def test_research_init_rejects_internal_phase_path_symlink_without_mutation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        initialized = runner.invoke(main, ["research", "init", "--name", "phase-link"])
        assert initialized.exit_code == 0, initialized.output
        phase_path = cwd_path / "versions/v001/04_spec_build"
        phase_path.rmdir()
        internal_target = cwd_path / "versions/v001/internal-spec"
        internal_target.mkdir()
        phase_path.symlink_to(internal_target, target_is_directory=True)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert "phase_paths.04_spec_build must not contain symlink components" in result.output
        assert _workspace_snapshot(cwd_path) == before


def test_spec_init_rejects_internal_phase_path_symlink_before_mutation(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        initialized = runner.invoke(main, ["research", "init", "--name", "phase-link"])
        assert initialized.exit_code == 0, initialized.output
        phase_path = cwd_path / "versions/v001/04_spec_build"
        phase_path.rmdir()
        internal_target = cwd_path / "versions/v001/internal-spec"
        internal_target.mkdir()
        phase_path.symlink_to(internal_target, target_is_directory=True)
        before = _workspace_snapshot(cwd_path)

        result = runner.invoke(main, ["spec", "init", "internal phase link"])

        assert result.exit_code == 1
        assert "phase_paths.04_spec_build must not contain symlink components" in result.output
        assert _workspace_snapshot(cwd_path) == before


@pytest.mark.parametrize("boundary", ["backup", "install"])
def test_research_init_rolls_back_keyboard_interrupt_at_replace_boundaries(
    monkeypatch,
    tmp_path,
    boundary: str,
) -> None:
    import oxq.cli.research as research_module

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        initialized = runner.invoke(main, ["research", "init", "--name", "interrupt"])
        assert initialized.exit_code == 0, initialized.output
        current_path = cwd_path / "current.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        current.pop("active_phase")
        current_path.write_text(json.dumps(current), encoding="utf-8")
        manifest_path = cwd_path / "versions/v001/version_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["active_phase"] = "06_spec_audit"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        phase_state_path = cwd_path / "versions/v001/phase_state.json"
        phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
        phase_state["current_phase"] = "06_spec_audit"
        phase_state_path.write_text(json.dumps(phase_state), encoding="utf-8")
        before = _workspace_snapshot(cwd_path)
        original_replace = research_module._GovernanceMutationParent.replace_file
        interrupted = False

        def interrupt_after_replace(self, source: str, target: str) -> None:
            nonlocal interrupted
            original_replace(self, source, target)
            is_boundary = (boundary == "backup" and ".backup-" in target) or (
                boundary == "install" and ".stage-" in source and target == "current.json"
            )
            if is_boundary and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("injected replace interruption")

        monkeypatch.setattr(
            research_module._GovernanceMutationParent,
            "replace_file",
            interrupt_after_replace,
        )

        result = runner.invoke(main, ["research", "init"])

        assert result.exit_code == 1
        assert interrupted
        assert _workspace_snapshot(cwd_path) == before


def test_research_init_recovers_interrupted_transaction_on_next_invocation(
    monkeypatch,
    tmp_path,
) -> None:
    import oxq.cli.research as research_module

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = tmp_path / cwd
        initialized = runner.invoke(main, ["research", "init", "--name", "recovery"])
        assert initialized.exit_code == 0, initialized.output
        current_path = cwd_path / "current.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        current.pop("active_phase")
        current_path.write_text(json.dumps(current), encoding="utf-8")
        manifest_path = cwd_path / "versions/v001/version_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["active_phase"] = "06_spec_audit"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        phase_state_path = cwd_path / "versions/v001/phase_state.json"
        phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
        phase_state["current_phase"] = "06_spec_audit"
        phase_state_path.write_text(json.dumps(phase_state), encoding="utf-8")
        governance_paths = (
            current_path,
            cwd_path / "workflow_manifest.json",
            cwd_path / "lineage.json",
            manifest_path,
            phase_state_path,
        )
        before = {path: path.read_bytes() for path in governance_paths}
        original_replace = research_module._GovernanceMutationParent.replace_file
        original_recover = research_module._recover_governance_transaction
        interrupted = False

        def interrupt_after_install(self, source: str, target: str) -> None:
            nonlocal interrupted
            original_replace(self, source, target)
            if ".stage-" in source and target == "current.json" and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("injected process interruption")

        def terminate_recovery(root: Path) -> None:
            journal = root / ".open-xquant/governance-transaction.json"
            if journal.exists():
                raise SystemExit("injected termination during rollback")
            original_recover(root)

        monkeypatch.setattr(
            research_module._GovernanceMutationParent,
            "replace_file",
            interrupt_after_install,
        )
        monkeypatch.setattr(
            research_module,
            "_recover_governance_transaction",
            terminate_recovery,
        )

        interrupted_result = runner.invoke(main, ["research", "init"])

        assert interrupted_result.exit_code == 1
        assert interrupted
        assert (cwd_path / ".open-xquant/governance-transaction.json").exists()

        monkeypatch.setattr(
            research_module._GovernanceMutationParent,
            "replace_file",
            original_replace,
        )
        monkeypatch.setattr(
            research_module,
            "_recover_governance_transaction",
            original_recover,
        )
        (cwd_path / "AGENTS.md").write_text(
            "<!-- open-xquant-workspace:begin -->\nunterminated\n",
            encoding="utf-8",
        )

        recovered_result = runner.invoke(main, ["research", "init"])

        assert recovered_result.exit_code == 1
        assert "Partial marker block" in recovered_result.output
        assert {path: path.read_bytes() for path in governance_paths} == before
        assert not (cwd_path / ".open-xquant/governance-transaction.json").exists()
        assert not list(cwd_path.rglob("*.backup-*"))
        assert not list(cwd_path.rglob("*.stage-*"))


@pytest.mark.parametrize("state", ["prepared", "committed"])
@pytest.mark.parametrize(
    "duplicate_destination",
    [
        "versions/v001/version_manifest.json",
        "versions/v001/./version_manifest.json",
    ],
    ids=["exact-duplicate", "lexical-alias"],
)
def test_research_init_recovery_rejects_duplicate_normalized_destinations_without_mutation(
    tmp_path,
    state: str,
    duplicate_destination: str,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="journal-duplicate")
    transaction_id = "a" * 32
    destination = workspace / "versions/v001/version_manifest.json"
    stage, backup = research_module._governance_transaction_artifacts(
        destination,
        transaction_id,
    )
    destination.write_bytes(b"installed data\n")
    stage.write_bytes(b"staged data\n")
    backup.write_bytes(b"original data\n")
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": research_module._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "state": state,
                "journal_parent_identity": research_module._governance_identity_payload(journal_path.parent.stat()),
                "entries": [
                    {
                        "destination": "versions/v001/version_manifest.json",
                        "had_original": True,
                        "parent_identity": research_module._governance_identity_payload(destination.parent.stat()),
                    },
                    {
                        "destination": duplicate_destination,
                        "had_original": False,
                        "parent_identity": research_module._governance_identity_payload(destination.parent.stat()),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = (destination, stage, backup, journal_path)
    before = {path: path.read_bytes() for path in artifacts}

    with pytest.raises(Exception, match="duplicate.*destination"):
        research_module._recover_governance_transaction(workspace)

    assert {path: path.read_bytes() for path in artifacts} == before


@pytest.mark.parametrize("symlink_timing", ["before", "after"])
def test_research_init_recovery_rejects_symlinked_destination_parent_without_external_mutation(
    tmp_path,
    symlink_timing: str,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="journal-symlink")
    version_dir = workspace / "versions/v001"
    external_dir = tmp_path / "external-version"
    external_dir.mkdir()
    external_manifest = external_dir / "version_manifest.json"
    external_manifest.write_text("external data\n", encoding="utf-8")
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal = {
        "schema_version": research_module._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": "a" * 32,
        "state": "prepared",
        "journal_parent_identity": research_module._governance_identity_payload(journal_path.parent.stat()),
        "entries": [
            {
                "destination": "versions/v001/version_manifest.json",
                "had_original": False,
                "parent_identity": research_module._governance_identity_payload(version_dir.stat()),
            }
        ],
    }

    if symlink_timing == "before":
        original_version_dir = workspace / "versions/v001-original"
        version_dir.rename(original_version_dir)
        version_dir.symlink_to(external_dir, target_is_directory=True)
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    if symlink_timing == "after":
        original_version_dir = workspace / "versions/v001-original"
        version_dir.rename(original_version_dir)
        version_dir.symlink_to(external_dir, target_is_directory=True)

    with pytest.raises(Exception, match="symlink"):
        research_module.initialize_workspace(workspace)

    assert external_manifest.read_text(encoding="utf-8") == "external data\n"
    assert journal_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlink semantics")
@pytest.mark.parametrize("state", ["prepared", "committed"])
def test_research_init_recovery_rejects_parent_swap_after_final_validation(
    monkeypatch,
    tmp_path,
    state: str,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="journal-swap")
    transaction_id = "d" * 32
    version_dir = workspace / "versions/v001"
    destination = version_dir / "version_manifest.json"
    stage, backup = research_module._governance_transaction_artifacts(
        destination,
        transaction_id,
    )
    if state == "prepared":
        destination.replace(backup)
        destination.write_text("installed data\n", encoding="utf-8")
    else:
        backup.write_text("backup data\n", encoding="utf-8")
    stage.write_text("staged data\n", encoding="utf-8")

    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": research_module._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "state": state,
                "journal_parent_identity": research_module._governance_identity_payload(journal_path.parent.stat()),
                "entries": [
                    {
                        "destination": "versions/v001/version_manifest.json",
                        "had_original": True,
                        "parent_identity": research_module._governance_identity_payload(version_dir.stat()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    external_dir = tmp_path / "external-version"
    external_dir.mkdir()
    external_paths = {
        "version_manifest.json": b"external destination\n",
        stage.name: b"external stage\n",
        backup.name: b"external backup\n",
    }
    for name, content in external_paths.items():
        (external_dir / name).write_bytes(content)

    original_validate = research_module._validate_workspace_transaction_path
    parked_version_dir = workspace / "versions/v001-validated"
    swapped = False

    def swap_after_final_validation(root: Path, path: Path, label: str) -> Path:
        nonlocal swapped
        relative = original_validate(root, path, label)
        if path == backup and label == "backup" and not swapped:
            version_dir.rename(parked_version_dir)
            version_dir.symlink_to(external_dir, target_is_directory=True)
            swapped = True
        return relative

    monkeypatch.setattr(
        research_module,
        "_validate_workspace_transaction_path",
        swap_after_final_validation,
    )

    with pytest.raises(Exception, match="symlink|changed|ambiguous"):
        research_module._recover_governance_transaction(workspace)

    assert swapped
    assert {name: (external_dir / name).read_bytes() for name in external_paths} == external_paths
    assert journal_path.exists()


def _record_governance_durability_events(monkeypatch, research_module):
    events: list[tuple[object, ...]] = []
    original_fsync = os.fsync
    original_replace = os.replace
    original_unlink = os.unlink
    original_write_journal = research_module._write_governance_transaction_journal

    def directory_identity(path: object, descriptor: int | None) -> tuple[int, int]:
        status = os.fstat(descriptor) if descriptor is not None else Path(path).parent.stat()
        return status.st_dev, status.st_ino

    def record_fsync(descriptor: int) -> None:
        status = os.fstat(descriptor)
        events.append(("fsync", (status.st_dev, status.st_ino)))
        original_fsync(descriptor)

    def record_replace(
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        identity = directory_identity(destination, dst_dir_fd)
        events.append(
            (
                "replace",
                identity,
                Path(source).name,
                Path(destination).name,
            )
        )
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def record_unlink(path: object, *, dir_fd: int | None = None) -> None:
        identity = directory_identity(path, dir_fd)
        events.append(("unlink", identity, Path(path).name))
        original_unlink(path, dir_fd=dir_fd)

    def record_write_journal(cwd: Path, journal: dict[str, object]) -> None:
        events.append(("journal", journal["state"]))
        original_write_journal(cwd, journal)

    monkeypatch.setattr(research_module.os, "fsync", record_fsync)
    monkeypatch.setattr(research_module.os, "replace", record_replace)
    monkeypatch.setattr(research_module.os, "unlink", record_unlink)
    monkeypatch.setattr(
        research_module,
        "_write_governance_transaction_journal",
        record_write_journal,
    )
    return events


def _assert_immediate_parent_fsyncs(
    events: list[tuple[object, ...]],
    mutation_names: set[str],
) -> None:
    mutations = [
        index
        for index, event in enumerate(events)
        if event[0] in {"replace", "unlink"} and any(name in mutation_names for name in event[2:])
    ]
    assert mutations
    for index in mutations:
        assert events[index + 1] == ("fsync", events[index][1]), events


def test_governance_commit_syncs_each_destination_rename_before_committed_journal(
    monkeypatch,
    tmp_path,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="durable-commit")
    current_path = workspace / "current.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current["active_phase"] = "06_spec_audit"
    events = _record_governance_durability_events(monkeypatch, research_module)

    research_module._write_governance_payloads_atomically(
        workspace,
        {current_path: current},
    )

    transaction_id = next(event[3].rsplit("-", 1)[-1] for event in events if event[0] == "replace" and ".backup-" in str(event[3]))
    stage_name = f".{current_path.name}.stage-{transaction_id}"
    backup_name = f".{current_path.name}.backup-{transaction_id}"
    commit_mutations = [
        index
        for index, event in enumerate(events)
        if event[0] == "replace" and (event[3] == backup_name or (event[2] == stage_name and event[3] == current_path.name))
    ]
    assert len(commit_mutations) == 2
    for index in commit_mutations:
        assert events[index + 1] == ("fsync", events[index][1]), events
    committed_index = events.index(("journal", "committed"))
    install_index = next(
        index for index, event in enumerate(events) if event[0] == "replace" and event[2] == stage_name and event[3] == current_path.name
    )
    assert install_index + 1 < committed_index


@pytest.mark.parametrize("state", ["prepared", "committed"])
def test_governance_recovery_syncs_each_rollback_and_cleanup_mutation(
    monkeypatch,
    tmp_path,
    state: str,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="durable-recovery")
    transaction_id = "e" * 32
    destination = workspace / "versions/v001/version_manifest.json"
    stage, backup = research_module._governance_transaction_artifacts(
        destination,
        transaction_id,
    )
    entry = {
        "destination": "versions/v001/version_manifest.json",
        "had_original": True,
        "parent_identity": research_module._governance_identity_payload(
            destination.parent.stat()
        ),
    }
    if state == "prepared":
        destination.replace(backup)
        destination.write_text("installed data\n", encoding="utf-8")
        stage.write_text("staged data\n", encoding="utf-8")
    else:
        backup.write_text("backup data\n", encoding="utf-8")
        replacement = destination.read_bytes()
        original = backup.read_bytes()
        entry.update(
            {
                "progress": "installed",
                "replacement_identity": research_module._governance_identity_payload(
                    destination.stat()
                ),
                "replacement_sha256": research_module.hashlib.sha256(
                    replacement
                ).hexdigest(),
                "original_identity": research_module._governance_identity_payload(
                    backup.stat()
                ),
                "original_sha256": research_module.hashlib.sha256(
                    original
                ).hexdigest(),
            }
        )
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": research_module._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "state": state,
                "journal_parent_identity": research_module._governance_identity_payload(journal_path.parent.stat()),
                "entries": [entry],
            }
        ),
        encoding="utf-8",
    )
    events = _record_governance_durability_events(monkeypatch, research_module)

    research_module._recover_governance_transaction(workspace)

    _assert_immediate_parent_fsyncs(
        events,
        {destination.name, stage.name, backup.name, journal_path.name},
    )


def test_research_init_recovery_rejects_non_governance_destination_without_mutation(
    tmp_path,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="journal-allowlist")
    notes_path = workspace / "notes.json"
    notes_path.write_text("user data\n", encoding="utf-8")
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": research_module._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
                "transaction_id": "b" * 32,
                "state": "prepared",
                "journal_parent_identity": research_module._governance_identity_payload(journal_path.parent.stat()),
                "entries": [
                    {
                        "destination": "notes.json",
                        "had_original": False,
                        "parent_identity": research_module._governance_identity_payload(workspace.stat()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="destination"):
        research_module.initialize_workspace(workspace)

    assert notes_path.read_text(encoding="utf-8") == "user data\n"
    assert journal_path.exists()


def test_research_init_recovery_rejects_inactive_version_destination_without_mutation(
    tmp_path,
) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="journal-active-version")
    inactive_dir = workspace / "versions/v999"
    inactive_dir.mkdir()
    inactive_manifest = inactive_dir / "version_manifest.json"
    inactive_manifest.write_text("user data\n", encoding="utf-8")
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": research_module._GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
                "transaction_id": "c" * 32,
                "state": "prepared",
                "journal_parent_identity": research_module._governance_identity_payload(journal_path.parent.stat()),
                "entries": [
                    {
                        "destination": "versions/v999/version_manifest.json",
                        "had_original": False,
                        "parent_identity": research_module._governance_identity_payload(inactive_dir.stat()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="destination"):
        research_module.initialize_workspace(workspace)

    assert inactive_manifest.read_text(encoding="utf-8") == "user data\n"
    assert journal_path.exists()


def test_governance_directory_fsync_is_explicitly_skipped_on_windows(
    monkeypatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    import oxq.cli.research as research_module

    calls: list[object] = []
    monkeypatch.setattr(
        research_module,
        "os",
        SimpleNamespace(
            name="nt",
            open=lambda *args, **kwargs: calls.append((args, kwargs)),
        ),
    )

    research_module._fsync_directory(tmp_path)

    assert calls == []


def test_research_init_serializes_recovery_and_mutation_across_processes(tmp_path) -> None:
    import oxq.cli.research as research_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research_module.initialize_workspace(workspace, name="process-lock")
    current_path = workspace / "current.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current.pop("active_phase")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    manifest_path = workspace / "versions/v001/version_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_phase"] = "06_spec_audit"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    phase_state_path = workspace / "versions/v001/phase_state.json"
    phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
    phase_state["current_phase"] = "06_spec_audit"
    phase_state_path.write_text(json.dumps(phase_state), encoding="utf-8")

    paused = tmp_path / "paused"
    release = tmp_path / "release"
    first_code = """
import pathlib
import sys
import time
import oxq.cli.research as research

workspace, paused, release = map(pathlib.Path, sys.argv[1:])
original = research._write_governance_transaction_journal

def pause_after_prepared(cwd, journal):
    original(cwd, journal)
    if journal.get('state') == 'prepared':
        paused.write_text('ready', encoding='utf-8')
        deadline = time.monotonic() + 15
        while not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('release was not signaled')
            time.sleep(0.02)

research._write_governance_transaction_journal = pause_after_prepared
research.initialize_workspace(workspace)
"""
    second_code = """
import pathlib
import sys
import oxq.cli.research as research
research.initialize_workspace(pathlib.Path(sys.argv[1]))
"""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    first = subprocess.Popen(
        [sys.executable, "-c", first_code, str(workspace), str(paused), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    second: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + 10
        while not paused.exists() and first.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        assert paused.exists(), first.communicate(timeout=1)

        second = subprocess.Popen(
            [sys.executable, "-c", second_code, str(workspace)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        time.sleep(0.4)
        assert second.poll() is None
        release.write_text("continue", encoding="utf-8")
        first_stdout, first_stderr = first.communicate(timeout=15)
        second_stdout, second_stderr = second.communicate(timeout=15)
    finally:
        release.touch()
        if first.poll() is None:
            first.kill()
            first.communicate()
        if second is not None and second.poll() is None:
            second.kill()
            second.communicate()

    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    assert not (workspace / ".open-xquant/governance-transaction.json").exists()
    assert not list(workspace.rglob("*.backup-*"))
    assert not list(workspace.rglob("*.stage-*"))
    repaired_current = json.loads(current_path.read_text(encoding="utf-8"))
    repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repaired_phase_state = json.loads(phase_state_path.read_text(encoding="utf-8"))
    assert repaired_current["active_phase"] == repaired_manifest["active_phase"]
    assert repaired_current["active_phase"] == repaired_phase_state["current_phase"]
