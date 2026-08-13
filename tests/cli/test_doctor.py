from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from oxq.cli.doctor import _check_data, _check_deps, _check_workspace
from oxq.cli.main import main
from oxq.cli.research import VERSION_PHASE_DIRS


def _workspace_snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def _write_source(root: Path) -> None:
    skills = root / "agent" / "skills"
    skill_dir = skills / "build-strategy-spec"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: build-strategy-spec\ndescription: Build quant strategies\n---\n\n# Strategy Builder\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlinks")
def test_doctor_fails_closed_for_broken_workspace_marker(monkeypatch, tmp_path) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    (config_dir / "workspace.yaml").symlink_to(tmp_path / "missing.yaml")
    monkeypatch.chdir(tmp_path)

    result = _check_workspace()

    assert result["status"] == "fail"
    assert "symlink" in result["error"]


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symlinks")
def test_doctor_does_not_suggest_force_init_for_symlinked_config_directory(monkeypatch, tmp_path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "workspace.yaml").write_text("paths: [broken\n", encoding="utf-8")
    (tmp_path / ".open-xquant").symlink_to(external, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    result = _check_workspace()

    assert result["fixes"] == ["Replace the .open-xquant symlink with a real directory, then run oxq research init"]
    assert "oxq research init --force" not in result["fixes"]


def _write_governed_workspace(work: Path, *, active_phase: str = "01_brainstorm") -> Path:
    config_dir = work / ".open-xquant"
    version_dir = work / "versions/v001"
    config_dir.mkdir(parents=True)
    version_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        "schema_version: 1\n"
        "paths:\n"
        "  versions_dir: versions\n"
        "  current_manifest: current.json\n"
        "  lineage_manifest: lineage.json\n"
        "workflow:\n"
        "  layout: version_governed\n",
        encoding="utf-8",
    )
    (work / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "active_version": "v001",
                "active_phase": active_phase,
                "active_run": "",
            }
        ),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "versions": [
                    {
                        "version_id": "v001",
                        "parent_version_id": "",
                        "created_reason": "initial_strategy_version",
                        "status": "active",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return version_dir


def _write_valid_active_version_state(version_dir: Path, *, active_phase: str = "01_brainstorm") -> None:
    (version_dir / "phase_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version_id": "v001",
                "current_phase": active_phase,
                "status": "active",
                "completed_phases": [],
                "blocked_phase": "",
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "version_id": "v001",
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "active",
                "active_phase": active_phase,
                "source_conversation": "",
                "phase_paths": {
                    phase: f"versions/v001/{phase}"
                    for phase in VERSION_PHASE_DIRS
                },
            }
        ),
        encoding="utf-8",
    )


def _write_required_governance_paths(work: Path) -> None:
    for dirname in ("conversations", "components", "governance", "final", "comparisons"):
        (work / dirname).mkdir()
    (work / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout": "version_governed",
                "strategy_family_id": "work",
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
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    (work / "comparisons/comparisons.jsonl").write_text("", encoding="utf-8")


@pytest.fixture(autouse=True)
def fake_sdk_bundle(monkeypatch):
    def build(source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict:
        del source_root
        root = config_root / "sdk-bundles" / "bundle-test"
        wheel = root / "dist" / "open_xquant-0.1.0-py3-none-any.whl"
        lock = root / "requirements.lock.txt"
        packages = root / "packages.json"
        python = root / "runner" / ".venv" / "bin" / "python"
        runner = root / "runner" / ".venv" / "bin" / "oxq"
        if not dry_run:
            wheel.parent.mkdir(parents=True, exist_ok=True)
            wheel.write_text("wheel", encoding="utf-8")
            lock.write_text("open-xquant @ file://wheel\n", encoding="utf-8")
            packages.write_text("[]\n", encoding="utf-8")
            runner.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            runner.chmod(0o755)
        return {
            "id": "bundle-test",
            "root": str(root),
            "profile": "full-research",
            "extras": ["chart", "scipy", "yfinance", "akshare", "live", "mcp", "agent"],
            "excluded_extras": ["dev", "docs", "talib"],
            "wheel": {"path": str(wheel), "sha256": "wheel-sha", "version": "0.1.0", "source_commit": "commit-sha"},
            "dependencies": {
                    "lock_file": str(lock),
                    "lock_sha256": "lock-sha",
                    "packages_file": str(packages),
                    "packages_count": 1,
                },
            "runner": {
                "venv": str(root / "runner" / ".venv"),
                "python": str(python),
                "oxq": str(runner),
                "argv": [str(runner)],
            },
            "uv_cache_dir": str(root / "uv-cache"),
        }

    monkeypatch.setattr("oxq.cli.agent.build_sdk_bundle", build)


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


def test_doctor_uses_persisted_codex_root_after_env_change(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    work = tmp_path / "work"
    original_codex_home = tmp_path / "codex-original"
    replacement_codex_home = tmp_path / "codex-replacement"
    work.mkdir()
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(original_codex_home))
    monkeypatch.chdir(work)
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    replacement_codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(replacement_codex_home))

    result = CliRunner().invoke(main, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checks"]["agent"]["status"] == "ok"
    assert not any(replacement_codex_home.iterdir())


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


@pytest.mark.parametrize("instruction_file", ["AGENTS.md", "CLAUDE.md"])
@pytest.mark.parametrize("marker", ["open-xquant-workspace", "open-xquant-subagents"])
def test_doctor_fix_rejects_malformed_managed_markers_without_mutation(
    monkeypatch,
    tmp_path,
    instruction_file: str,
    marker: str,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / instruction_file).write_text(
        f"user content\n<!-- {marker}:begin -->\nmanaged content\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(work)
    before = _workspace_snapshot(work)

    result = CliRunner().invoke(main, ["doctor", "--fix"])

    assert result.exit_code == 1
    assert f"Partial marker block for {marker}" in result.output
    assert _workspace_snapshot(work) == before


def test_doctor_data_check_uses_market_data_directory(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    (home / ".oxq/data").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    result = _check_data()

    assert result["status"] == "warn"
    assert result["path"].endswith(".oxq/data/market")


def test_doctor_accepts_legacy_workspace_layout(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
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
    (work / "strategy_specs").mkdir()
    (work / "runs").mkdir()
    (work / "reports").mkdir()
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "ok"
    assert result["missing"] == []


def test_doctor_warns_when_configured_comparison_registry_is_missing(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  runs_dir: runs",
                "  final_dir: runs/final",
                "  comparisons_dir: comparisons",
                "  experiment_registry: experiments.jsonl",
                "  comparison_registry: comparisons/comparisons.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    (work / "runs/final").mkdir(parents=True)
    (work / "comparisons").mkdir()
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert str(work / "comparisons" / "comparisons.jsonl") in result["missing"]


def test_doctor_warns_when_version_governed_workspace_has_no_active_version(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions",
                "  conversations_dir: conversations",
                "  components_dir: components",
                "  governance_dir: governance",
                "  runs_dir: runs",
                "  final_dir: final",
                "  comparisons_dir: comparisons",
                "  current_manifest: current.json",
                "  lineage_manifest: lineage.json",
                "  workflow_manifest: workflow_manifest.json",
                "  experiment_registry: experiments.jsonl",
                "  comparison_registry: comparisons/comparisons.jsonl",
                "workflow:",
                "  layout: version_governed",
            ]
        ),
        encoding="utf-8",
    )
    for dirname in ("versions", "conversations", "components", "governance", "runs", "final", "comparisons"):
        (work / dirname).mkdir()
    (work / "current.json").write_text(
        json.dumps({"schema_version": 1, "strategy_family_id": "work", "active_version": "", "active_phase": "", "active_run": ""}),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps({"schema_version": 1, "strategy_family_id": "work", "versions": []}),
        encoding="utf-8",
    )
    (work / "workflow_manifest.json").write_text(
        json.dumps({"schema_version": 1, "layout": "version_governed", "strategy_family_id": "work"}),
        encoding="utf-8",
    )
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    (work / "comparisons" / "comparisons.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "active_version_missing" in result["governance_warnings"]
    assert "lineage_versions_empty" in result["governance_warnings"]


def test_doctor_treats_versions_dir_workspace_as_version_governed(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions",
                "  current_manifest: current.json",
                "  lineage_manifest: lineage.json",
            ]
        ),
        encoding="utf-8",
    )
    (work / "versions").mkdir()
    (work / "current.json").write_text(json.dumps({"active_version": "", "active_phase": ""}), encoding="utf-8")
    (work / "lineage.json").write_text(json.dumps({"versions": []}), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "active_version_missing" in result["governance_warnings"]
    assert "lineage_versions_empty" in result["governance_warnings"]


def test_doctor_uses_defaults_for_omitted_required_version_governance_paths(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "schema_version: 1\npaths:\n  versions_dir: versions\n",
        encoding="utf-8",
    )
    (work / "versions").mkdir()
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert str(work / "workflow_manifest.json") in result["missing"]
    assert str(work / "experiments.jsonl") in result["missing"]


def test_doctor_warns_when_versions_dir_symlink_escapes_workspace(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    (work / ".open-xquant").mkdir(parents=True)
    (outside / "v001").mkdir(parents=True)
    (work / "versions_link").symlink_to(outside, target_is_directory=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions_link",
                "  current_manifest: current.json",
                "  lineage_manifest: lineage.json",
                "workflow:",
                "  layout: version_governed",
            ]
        ),
        encoding="utf-8",
    )
    (work / "current.json").write_text(
        json.dumps({"active_version": "v001", "active_phase": "04_spec_build"}),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps({"versions": [{"version_id": "v001", "status": "active"}]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "versions_dir_unsafe" in result["governance_warnings"]


@pytest.mark.parametrize(
    "path_key,is_registry",
    [
        ("components_dir", False),
        ("conversations_dir", False),
        ("governance_dir", False),
        ("experiment_registry", True),
        ("comparison_registry", True),
    ],
)
@pytest.mark.parametrize("escape_kind", ["traversal", "symlink"])
def test_doctor_warns_for_every_unsafe_configured_path(
    monkeypatch,
    tmp_path,
    path_key: str,
    is_registry: bool,
    escape_kind: str,
) -> None:
    work = tmp_path / "work"
    outside = tmp_path / ("outside.jsonl" if is_registry else "outside")
    (work / ".open-xquant").mkdir(parents=True)
    if is_registry:
        outside.write_text("", encoding="utf-8")
    else:
        outside.mkdir()
    if escape_kind == "traversal":
        configured = f"../{outside.name}"
    else:
        link = work / ("registry_link" if is_registry else "components_link")
        link.symlink_to(outside, target_is_directory=not is_registry)
        configured = link.name
    (work / ".open-xquant" / "workspace.yaml").write_text(
        f"schema_version: 1\npaths:\n  {path_key}: {configured}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert f"{path_key}_unsafe" in result["governance_warnings"]


def test_doctor_warns_when_version_governed_workspace_uses_hidden_root_manifests(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions",
                "  current_manifest: .open-xquant/current.json",
                "  lineage_manifest: lineage.json",
                "  workflow_manifest: workflow_manifest.json",
                "workflow:",
                "  layout: version_governed",
            ]
        ),
        encoding="utf-8",
    )
    (work / "versions" / "v001").mkdir(parents=True)
    (work / ".open-xquant" / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v001", "active_phase": "04_spec_build"}),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps({"schema_version": 1, "versions": [{"version_id": "v001", "status": "active"}]}),
        encoding="utf-8",
    )
    (work / "workflow_manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "root_manifest_path_invalid:current_manifest" in result["governance_warnings"]


def test_doctor_warns_when_active_version_is_not_path_safe(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions",
                "  current_manifest: current.json",
                "  lineage_manifest: lineage.json",
                "  workflow_manifest: workflow_manifest.json",
                "workflow:",
                "  layout: version_governed",
            ]
        ),
        encoding="utf-8",
    )
    (work / "versions").mkdir()
    (work / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "../escape", "active_phase": "04_spec_build"}),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps({"schema_version": 1, "versions": [{"version_id": "../escape", "status": "active"}]}),
        encoding="utf-8",
    )
    (work / "workflow_manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "active_version_invalid" in result["governance_warnings"]


def test_doctor_warns_when_lineage_has_multiple_active_versions_with_custom_root(
    monkeypatch,
    tmp_path,
) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    versions_dir = work / "research_versions"
    config_dir.mkdir(parents=True)
    (versions_dir / "v002").mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        "schema_version: 1\n"
        "paths:\n"
        "  versions_dir: research_versions\n"
        "  current_manifest: current.json\n"
        "  lineage_manifest: lineage.json\n"
        "workflow:\n"
        "  layout: version_governed\n",
        encoding="utf-8",
    )
    (work / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "active_version": "v002",
                "active_phase": "06_spec_audit",
                "active_run": "",
            }
        ),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps(
            {
                "versions": [
                    {"version_id": "v001", "status": "active"},
                    {"version_id": "v002", "status": "active"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "lineage_multiple_active_versions" in result["governance_warnings"]


def test_doctor_warns_when_only_active_lineage_entry_differs_from_current_with_custom_root(
    monkeypatch,
    tmp_path,
) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    version_dir = work / "research_versions/v002"
    config_dir.mkdir(parents=True)
    version_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        "schema_version: 1\n"
        "paths:\n"
        "  versions_dir: research_versions\n"
        "  current_manifest: current.json\n"
        "  lineage_manifest: lineage.json\n"
        "workflow:\n"
        "  layout: version_governed\n",
        encoding="utf-8",
    )
    (work / "current.json").write_text(
        json.dumps({"active_version": "v002", "active_phase": "06_spec_audit"}),
        encoding="utf-8",
    )
    current_identity = {
        "version_id": "v002",
        "parent_version_id": "v001",
        "created_reason": "signal_semantics_change",
        "status": "superseded",
    }
    (work / "lineage.json").write_text(
        json.dumps(
            {
                "versions": [
                    {
                        "version_id": "v001",
                        "parent_version_id": "",
                        "created_reason": "initial_strategy_version",
                        "status": "active",
                    },
                    current_identity,
                ]
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps({**current_identity, "active_phase": "06_spec_audit"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "lineage_active_version_mismatch:current" in result["governance_warnings"]


@pytest.mark.parametrize("identity_field", ["parent_version_id", "created_reason"])
def test_doctor_warns_when_active_lineage_identity_disagrees_with_manifest_in_custom_root(
    monkeypatch,
    tmp_path,
    identity_field: str,
) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    version_dir = work / "research_versions/v002"
    config_dir.mkdir(parents=True)
    version_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        "schema_version: 1\n"
        "paths:\n"
        "  versions_dir: research_versions\n"
        "  current_manifest: current.json\n"
        "  lineage_manifest: lineage.json\n"
        "workflow:\n"
        "  layout: version_governed\n",
        encoding="utf-8",
    )
    (work / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "active_version": "v002",
                "active_phase": "06_spec_audit",
                "active_run": "",
            }
        ),
        encoding="utf-8",
    )
    lineage_identity = {
        "version_id": "v002",
        "parent_version_id": "v001",
        "created_reason": "signal_semantics_change",
        "status": "active",
    }
    lineage_identity[identity_field] = (
        "v000" if identity_field == "parent_version_id" else "different"
    )
    (work / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "versions": [lineage_identity],
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "version_id": "v002",
                "parent_version_id": "v001",
                "created_reason": "signal_semantics_change",
                "status": "active",
                "active_phase": "06_spec_audit",
                "source_conversation": "",
                "phase_paths": {
                    phase: f"research_versions/v002/{phase}"
                    for phase in VERSION_PHASE_DIRS
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "active_lineage_identity_mismatch:version_manifest" in result["governance_warnings"]


@pytest.mark.parametrize(
    ("manifest_name", "contents", "expected_warning"),
    [
        ("version_manifest.json", None, "active_version_manifest_missing"),
        ("version_manifest.json", "{broken\n", "active_version_manifest_invalid"),
        ("phase_state.json", None, "active_phase_state_missing"),
        ("phase_state.json", "[]\n", "active_phase_state_invalid"),
    ],
)
def test_doctor_warns_for_missing_or_malformed_active_version_state_files(
    monkeypatch,
    tmp_path,
    manifest_name: str,
    contents: str | None,
    expected_warning: str,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    valid_files = {
        "version_manifest.json": {
            "version_id": "v001",
            "parent_version_id": "",
            "created_reason": "initial_strategy_version",
            "status": "active",
            "active_phase": "01_brainstorm",
            "phase_paths": {"10_reports": "versions/v001/10_reports"},
        },
        "phase_state.json": {
            "version_id": "v001",
            "current_phase": "01_brainstorm",
            "status": "active",
        },
    }
    for filename, payload in valid_files.items():
        if filename != manifest_name:
            (version_dir / filename).write_text(json.dumps(payload), encoding="utf-8")
    if contents is not None:
        (version_dir / manifest_name).write_text(contents, encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert expected_warning in result["governance_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("schema_version", None, True),
        ("schema_version", 999, False),
        ("schema_version", "1", False),
        ("version_id", "v002", False),
        ("version_id", None, True),
        ("version_id", 1, False),
        ("status", "superseded", False),
        ("status", None, True),
        ("status", 1, False),
        ("current_phase", None, True),
        ("current_phase", 1, False),
        ("current_phase", "11_unknown", False),
        ("completed_phases", None, True),
        ("completed_phases", "01_brainstorm", False),
        ("completed_phases", [1], False),
        ("completed_phases", ["11_unknown"], False),
        ("completed_phases", ["01_brainstorm", "01_brainstorm"], False),
        ("blocked_phase", None, True),
        ("blocked_phase", [], False),
        ("blocked_phase", "11_unknown", False),
    ],
)
def test_doctor_warns_for_invalid_active_phase_state_schema(
    monkeypatch,
    tmp_path,
    field: str,
    value: object,
    remove: bool,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    phase_state = json.loads((version_dir / "phase_state.json").read_text(encoding="utf-8"))
    if remove:
        phase_state.pop(field)
    else:
        phase_state[field] = value
    (version_dir / "phase_state.json").write_text(json.dumps(phase_state), encoding="utf-8")
    monkeypatch.chdir(work)
    before = _workspace_snapshot(work)

    result = _check_workspace()

    assert result["missing"] == []
    assert result["status"] == "warn"
    assert "active_phase_state_invalid" in result["governance_warnings"]
    assert _workspace_snapshot(work) == before


@pytest.mark.parametrize(
    "entry",
    [
        [],
        {"status": "superseded"},
        {
            "version_id": 1,
            "parent_version_id": "",
            "created_reason": "initial_strategy_version",
            "status": "superseded",
        },
        {
            "version_id": "../v000",
            "parent_version_id": "",
            "created_reason": "initial_strategy_version",
            "status": "superseded",
        },
        {
            "version_id": "v000",
            "created_reason": "initial_strategy_version",
            "status": "superseded",
        },
        {
            "version_id": "v000",
            "parent_version_id": 1,
            "created_reason": "initial_strategy_version",
            "status": "superseded",
        },
        {
            "version_id": "v000",
            "parent_version_id": "",
            "status": "superseded",
        },
        {
            "version_id": "v000",
            "parent_version_id": "",
            "created_reason": 1,
            "status": "superseded",
        },
        {
            "version_id": "v000",
            "parent_version_id": "",
            "created_reason": "",
            "status": "superseded",
        },
        {
            "version_id": "v000",
            "parent_version_id": "",
            "created_reason": "initial_strategy_version",
        },
    ],
)
def test_doctor_warns_for_malformed_non_active_lineage_entry(
    monkeypatch,
    tmp_path,
    entry: object,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    lineage = json.loads((work / "lineage.json").read_text(encoding="utf-8"))
    lineage["versions"].insert(0, entry)
    (work / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    monkeypatch.chdir(work)
    before = _workspace_snapshot(work)

    result = _check_workspace()

    assert result["missing"] == []
    assert result["status"] == "warn"
    assert any(
        warning.startswith("lineage_entry_invalid:")
        for warning in result["governance_warnings"]
    )
    assert _workspace_snapshot(work) == before


def test_doctor_accepts_nullable_governance_fields_and_unique_completed_phases(
    monkeypatch,
    tmp_path,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work, active_phase="02_idea_audit")
    _write_valid_active_version_state(version_dir, active_phase="02_idea_audit")
    _write_required_governance_paths(work)
    lineage = json.loads((work / "lineage.json").read_text(encoding="utf-8"))
    lineage["versions"][0]["parent_version_id"] = None
    (work / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    version_manifest = json.loads(
        (version_dir / "version_manifest.json").read_text(encoding="utf-8")
    )
    version_manifest["parent_version_id"] = None
    (version_dir / "version_manifest.json").write_text(
        json.dumps(version_manifest),
        encoding="utf-8",
    )
    phase_state = json.loads((version_dir / "phase_state.json").read_text(encoding="utf-8"))
    phase_state["completed_phases"] = ["01_brainstorm"]
    phase_state["blocked_phase"] = None
    (version_dir / "phase_state.json").write_text(json.dumps(phase_state), encoding="utf-8")
    monkeypatch.chdir(work)
    before = _workspace_snapshot(work)

    result = _check_workspace()

    assert result == {"status": "ok", "missing": [], "governance_warnings": []}
    assert _workspace_snapshot(work) == before


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("schema_version", None, True),
        ("schema_version", 2, False),
        ("strategy_family_id", None, True),
        ("strategy_family_id", "", False),
        ("active_version", None, True),
        ("active_version", 1, False),
        ("active_phase", None, True),
        ("active_phase", "11_unknown", False),
        ("active_run", None, True),
        ("active_run", 1, False),
    ],
)
def test_doctor_warns_for_invalid_current_manifest_schema(
    monkeypatch,
    tmp_path,
    field: str,
    value: object,
    remove: bool,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    current_path = work / "current.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if remove:
        current.pop(field)
    else:
        current[field] = value
    current_path.write_text(json.dumps(current), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert "current_manifest_invalid" in result["governance_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("schema_version", None, True),
        ("schema_version", "1", False),
        ("strategy_family_id", None, True),
        ("strategy_family_id", "", False),
        ("versions", None, True),
        ("versions", {}, False),
        ("versions", [], False),
    ],
)
def test_doctor_warns_for_invalid_lineage_manifest_schema(
    monkeypatch,
    tmp_path,
    field: str,
    value: object,
    remove: bool,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    lineage_path = work / "lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if remove:
        lineage.pop(field)
    else:
        lineage[field] = value
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert "lineage_manifest_invalid" in result["governance_warnings"]


def test_doctor_warns_when_workflow_manifest_is_missing(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    (work / "workflow_manifest.json").unlink()
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert "workflow_manifest_missing" in result["governance_warnings"]


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("schema_version", None, True),
        ("schema_version", "1", False),
        ("layout", None, True),
        ("layout", "legacy", False),
        ("strategy_family_id", "", False),
        ("paths", [], False),
    ],
)
def test_doctor_warns_for_incomplete_workflow_manifest_schema(
    monkeypatch,
    tmp_path,
    field: str,
    value: object,
    remove: bool,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    workflow_path = work / "workflow_manifest.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    if remove:
        workflow.pop(field)
    else:
        workflow[field] = value
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert "workflow_manifest_invalid" in result["governance_warnings"]


@pytest.mark.parametrize(
    ("created_reason", "parent_version_id"),
    [
        ("initial_strategy_version", "v000"),
        ("signal_semantics_change", None),
        ("signal_semantics_change", ""),
    ],
)
def test_doctor_enforces_created_reason_parent_contract(
    monkeypatch,
    tmp_path,
    created_reason: str,
    parent_version_id: str | None,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    lineage_path = work / "lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["versions"][0]["created_reason"] = created_reason
    lineage["versions"][0]["parent_version_id"] = parent_version_id
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    manifest_path = version_dir / "version_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_reason"] = created_reason
    manifest["parent_version_id"] = parent_version_id
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    warnings = result["governance_warnings"]
    assert "lineage_entry_invalid:v001" in warnings
    assert "active_version_manifest_invalid" in warnings


@pytest.mark.parametrize(
    ("field", "value", "remove"),
    [
        ("schema_version", None, True),
        ("schema_version", 2, False),
        ("strategy_family_id", None, True),
        ("strategy_family_id", "", False),
        ("version_id", None, True),
        ("version_id", "v002", False),
        ("parent_version_id", None, True),
        ("parent_version_id", 1, False),
        ("created_reason", None, True),
        ("created_reason", "", False),
        ("status", None, True),
        ("status", "superseded", False),
        ("active_phase", None, True),
        ("active_phase", "11_unknown", False),
        ("source_conversation", None, True),
        ("source_conversation", 1, False),
        ("phase_paths", None, True),
        ("phase_paths", [], False),
    ],
)
def test_doctor_warns_for_invalid_active_version_manifest_schema(
    monkeypatch,
    tmp_path,
    field: str,
    value: object,
    remove: bool,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    manifest_path = version_dir / "version_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if remove:
        manifest.pop(field)
    else:
        manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert "active_version_manifest_invalid" in result["governance_warnings"]


@pytest.mark.parametrize("phase", VERSION_PHASE_DIRS)
@pytest.mark.parametrize("invalid_kind", ["missing", "traversal", "symlink"])
def test_doctor_validates_every_required_phase_path_with_canonical_containment(
    monkeypatch,
    tmp_path,
    phase: str,
    invalid_kind: str,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    manifest_path = version_dir / "version_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if invalid_kind == "missing":
        manifest["phase_paths"].pop(phase)
    elif invalid_kind == "traversal":
        manifest["phase_paths"][phase] = f"../outside/{phase}"
    else:
        outside = tmp_path / f"outside-{phase}"
        outside.mkdir()
        link = version_dir / f"{phase}-link"
        link.symlink_to(outside, target_is_directory=True)
        manifest["phase_paths"][phase] = link.relative_to(work).as_posix()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert f"phase_path_unsafe:{phase}" in result["governance_warnings"]


def test_doctor_warns_for_duplicate_lineage_version_ids(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    lineage = json.loads((work / "lineage.json").read_text(encoding="utf-8"))
    lineage["versions"].insert(
        0,
        {
            "version_id": "v001",
            "parent_version_id": None,
            "created_reason": "earlier_version_record",
            "status": "superseded",
        },
    )
    (work / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    monkeypatch.chdir(work)
    before = _workspace_snapshot(work)

    result = _check_workspace()

    assert result["missing"] == []
    assert result["status"] == "warn"
    assert "lineage_duplicate_version_id:v001" in result["governance_warnings"]
    assert _workspace_snapshot(work) == before


def test_doctor_warns_for_unknown_status_on_non_active_lineage_entry(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    (work / "lineage.json").write_text(
        json.dumps(
            {
                "versions": [
                    {
                        "version_id": "v000",
                        "parent_version_id": "",
                        "created_reason": "initial_strategy_version",
                        "status": "retired",
                    },
                    {
                        "version_id": "v001",
                        "parent_version_id": "",
                        "created_reason": "initial_strategy_version",
                        "status": "active",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["missing"] == []
    assert result["status"] == "warn"
    assert "lineage_status_invalid:v000" in result["governance_warnings"]


def test_doctor_scans_custom_safe_report_phase_path_for_passing_review(
    monkeypatch,
    tmp_path,
) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    custom_reports = version_dir / "artifacts/reports"
    (custom_reports / "run-1").mkdir(parents=True)
    (version_dir / "phase_state.json").write_text(
        json.dumps({"version_id": "v001", "current_phase": "01_brainstorm", "status": "active"}),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "active",
                "active_phase": "01_brainstorm",
                "phase_paths": {"10_reports": "versions/v001/artifacts/reports"},
            }
        ),
        encoding="utf-8",
    )
    (custom_reports / "run-1/report_review.json").write_text(
        json.dumps({"status": "pass", "verdict": "consistent"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert "active_phase_stale:report_review_passed" in result["governance_warnings"]
    assert "phase_state_stale:report_review_passed" in result["governance_warnings"]
    assert "version_manifest_phase_stale:report_review_passed" in result["governance_warnings"]


def test_doctor_warns_for_unsafe_custom_report_phase_path(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    version_dir = _write_governed_workspace(work)
    (version_dir / "phase_state.json").write_text(
        json.dumps({"version_id": "v001", "current_phase": "01_brainstorm", "status": "active"}),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "active",
                "active_phase": "01_brainstorm",
                "phase_paths": {"10_reports": "../outside/reports"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert "phase_path_unsafe:10_reports" in result["governance_warnings"]


@pytest.mark.parametrize("escape_kind", ["run", "review"])
def test_doctor_rejects_symlink_escapes_while_scanning_report_reviews(
    monkeypatch,
    tmp_path,
    escape_kind: str,
) -> None:
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    version_dir = _write_governed_workspace(work)
    reports_dir = version_dir / "10_reports"
    external_run = outside / "run-1"
    reports_dir.mkdir()
    external_run.mkdir(parents=True)
    _write_valid_active_version_state(version_dir)
    external_review = external_run / "report_review.json"
    external_review.write_text(
        json.dumps({"status": "pass", "verdict": "consistent"}),
        encoding="utf-8",
    )
    if escape_kind == "run":
        (reports_dir / "run-1").symlink_to(external_run, target_is_directory=True)
    else:
        local_run = reports_dir / "run-1"
        local_run.mkdir()
        (local_run / "report_review.json").symlink_to(external_review)
    monkeypatch.chdir(work)

    result = _check_workspace()

    warnings = result["governance_warnings"]
    assert "phase_path_unsafe:10_reports" in warnings
    assert "active_phase_stale:report_review_passed" not in warnings
    assert "phase_state_stale:report_review_passed" not in warnings
    assert "version_manifest_phase_stale:report_review_passed" not in warnings


def test_doctor_warns_when_version_governed_workspace_has_root_phase_artifacts(
    monkeypatch, tmp_path
) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions",
                "  conversations_dir: conversations",
                "  components_dir: components",
                "  governance_dir: governance",
                "  runs_dir: runs",
                "  final_dir: final",
                "  comparisons_dir: comparisons",
                "  current_manifest: current.json",
                "  lineage_manifest: lineage.json",
                "  workflow_manifest: workflow_manifest.json",
                "  experiment_registry: experiments.jsonl",
                "  comparison_registry: comparisons/comparisons.jsonl",
                "workflow:",
                "  layout: version_governed",
            ]
        ),
        encoding="utf-8",
    )
    for dirname in ("versions", "conversations", "components", "governance", "runs", "final", "comparisons"):
        (work / dirname).mkdir()
    (work / "versions" / "v001").mkdir()
    (work / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "active_version": "v001",
                "active_phase": "04_spec_build",
                "active_run": "",
            }
        ),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "versions": [{"version_id": "v001", "status": "active"}],
            }
        ),
        encoding="utf-8",
    )
    (work / "workflow_manifest.json").write_text(
        json.dumps({"schema_version": 1, "layout": "version_governed", "strategy_family_id": "work"}),
        encoding="utf-8",
    )
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    (work / "comparisons" / "comparisons.jsonl").write_text("", encoding="utf-8")
    (work / "strategy_spec.yaml").write_text("strategy_id: polluted\n", encoding="utf-8")
    (work / "spec_mapping_contract.json").write_text("{}\n", encoding="utf-8")
    (work / "component_manifest.json").write_text("{}\n", encoding="utf-8")
    (work / "compile_preview").mkdir()
    (work / "result.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "root_phase_artifact:strategy_spec.yaml" in result["governance_warnings"]
    assert "root_phase_artifact:spec_mapping_contract.json" in result["governance_warnings"]
    assert "root_phase_artifact:component_manifest.json" in result["governance_warnings"]
    assert "root_phase_artifact:compile_preview" in result["governance_warnings"]
    assert "root_phase_artifact:result.json" in result["governance_warnings"]


def test_doctor_warns_when_report_review_passes_but_active_phase_is_stale(
    monkeypatch, tmp_path
) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions",
                "  conversations_dir: conversations",
                "  components_dir: components",
                "  governance_dir: governance",
                "  runs_dir: runs",
                "  final_dir: final",
                "  comparisons_dir: comparisons",
                "  current_manifest: current.json",
                "  lineage_manifest: lineage.json",
                "  workflow_manifest: workflow_manifest.json",
                "  experiment_registry: experiments.jsonl",
                "  comparison_registry: comparisons/comparisons.jsonl",
                "workflow:",
                "  layout: version_governed",
            ]
        ),
        encoding="utf-8",
    )
    for dirname in ("versions", "conversations", "components", "governance", "runs", "final", "comparisons"):
        (work / dirname).mkdir()
    version = work / "versions" / "v001"
    (version / "10_reports" / "run-1").mkdir(parents=True)
    (version / "phase_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version_id": "v001",
                "current_phase": "01_brainstorm",
                "status": "active",
                "completed_phases": [],
                "blocked_phase": "",
            }
        ),
        encoding="utf-8",
    )
    (version / "version_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "version_id": "v001",
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "active",
                "active_phase": "01_brainstorm",
                "source_conversation": "",
                "phase_paths": {
                    phase: f"versions/v001/{phase}"
                    for phase in VERSION_PHASE_DIRS
                },
            }
        ),
        encoding="utf-8",
    )
    (version / "10_reports" / "run-1" / "report_review.json").write_text(
        json.dumps({"status": "pass", "verdict": "consistent"}),
        encoding="utf-8",
    )
    (work / "current.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "active_version": "v001",
                "active_phase": "01_brainstorm",
                "active_run": "run-1",
            }
        ),
        encoding="utf-8",
    )
    (work / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy_family_id": "work",
                "versions": [
                    {
                        "version_id": "v001",
                        "parent_version_id": "",
                        "created_reason": "initial_strategy_version",
                        "status": "active",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (work / "workflow_manifest.json").write_text(
        json.dumps({"schema_version": 1, "layout": "version_governed", "strategy_family_id": "work"}),
        encoding="utf-8",
    )
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    (work / "comparisons" / "comparisons.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "active_phase_stale:report_review_passed" in result["governance_warnings"]


@pytest.mark.parametrize(
    ("artifact", "expected_warning"),
    [
        ("lineage_root", "strategy_family_id_mismatch:lineage"),
        ("lineage_entry", "strategy_family_id_mismatch:lineage:v001"),
        ("workflow", "strategy_family_id_mismatch:workflow_manifest"),
        ("version_manifest", "strategy_family_id_mismatch:version_manifest"),
    ],
)
def test_doctor_warns_for_every_strategy_family_identity_mismatch(
    monkeypatch,
    tmp_path,
    artifact: str,
    expected_warning: str,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    version_dir = _write_governed_workspace(work)
    _write_valid_active_version_state(version_dir)
    _write_required_governance_paths(work)
    paths = {
        "lineage_root": work / "lineage.json",
        "lineage_entry": work / "lineage.json",
        "workflow": work / "workflow_manifest.json",
        "version_manifest": version_dir / "version_manifest.json",
    }
    path = paths[artifact]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if artifact == "lineage_entry":
        payload["versions"][0]["strategy_family_id"] = "other-family"
    else:
        payload["strategy_family_id"] = "other-family"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert expected_warning in result["governance_warnings"]


def test_doctor_json_reports_malformed_workspace_config(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text("paths: [broken", encoding="utf-8")
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checks"]["workspace"]["status"] == "fail"
    assert "workspace.yaml" in payload["checks"]["workspace"]["path"]
    assert "oxq research init --force" in payload["fixes"]


def test_doctor_deps_separates_core_and_optional_missing(monkeypatch) -> None:
    missing = {"pyarrow", "pandas", "numpy", "yaml", "scipy", "matplotlib", "yfinance"}

    def fake_find_spec(module: str):
        return None if module in missing else object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    result = _check_deps()

    assert result["status"] == "fail"
    assert "pyarrow" in result["missing_core"]
    assert "pandas" in result["missing_core"]
    assert "numpy" in result["missing_core"]
    assert "yaml" in result["missing_core"]
    assert "scipy" in result["missing_optional"]
    assert "matplotlib" in result["missing_optional"]
    assert "yfinance" in result["missing_optional"]
    assert "uv sync --all-extras" in result["fixes"]


def test_doctor_deps_warns_when_only_optional_missing(monkeypatch) -> None:
    missing = {"scipy", "mplfinance", "seaborn"}

    def fake_find_spec(module: str):
        return None if module in missing else object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    result = _check_deps()

    assert result["status"] == "warn"
    assert result["missing_core"] == []
    assert result["missing_optional"] == ["mplfinance", "scipy", "seaborn"]
    assert "uv sync --extra scipy" in result["fixes"]
    assert "uv sync --extra chart" in result["fixes"]


@pytest.mark.parametrize("versions_dir", [None, "", 7])
def test_doctor_classifies_malformed_configured_versions_dir_as_invalid_governed(
    monkeypatch,
    tmp_path,
    versions_dir: object,
) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    config_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": 1, "paths": {"versions_dir": versions_dir}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "versions_dir_invalid" in result["governance_warnings"]


def test_doctor_warns_when_workflow_manifest_paths_drift_from_workspace_config(
    monkeypatch,
    tmp_path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    initialized = CliRunner().invoke(main, ["research", "init", "--name", "root-drift"])
    assert initialized.exit_code == 0, initialized.output
    workspace_path = work / ".open-xquant/workspace.yaml"
    config = yaml.safe_load(workspace_path.read_text(encoding="utf-8"))
    config["paths"]["versions_dir"] = "research_versions"
    workspace_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "workflow_manifest_path_mismatch:versions_dir" in result["governance_warnings"]


def test_doctor_warns_for_incomplete_workflow_path_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    initialized = CliRunner().invoke(main, ["research", "init", "--name", "snapshot"])
    assert initialized.exit_code == 0, initialized.output
    workflow_path = work / "workflow_manifest.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["paths"].pop("versions_dir")
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "workflow_manifest_invalid" in result["governance_warnings"]


def test_doctor_warns_for_internal_phase_path_symlink(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    initialized = CliRunner().invoke(main, ["research", "init", "--name", "phase-link"])
    assert initialized.exit_code == 0, initialized.output
    phase_path = work / "versions/v001/04_spec_build"
    phase_path.rmdir()
    internal_target = work / "versions/v001/internal-spec"
    internal_target.mkdir()
    phase_path.symlink_to(internal_target, target_is_directory=True)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert "phase_path_unsafe:04_spec_build" in result["governance_warnings"]
