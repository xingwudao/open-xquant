"""Research workspace initialization commands."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import click

from oxq.cli.agent_manifest import read_yaml_file, remove_marker_block, upsert_marker_block, write_text_file, write_yaml_file
from oxq.cli.sdk_bundle import install_workspace_sdk

AGENT_PROFILE_MULTI = "multi-agent"
AGENT_PROFILE_STANDALONE = "standalone-agent"
_WORKSPACE_VERSION_RE = re.compile(r"^v[0-9][A-Za-z0-9_-]*$")

WORKSPACE_BLOCK = """This is an open-xquant research workspace.

For quant strategy, factor, backtest, audit, robustness, report, chart asset,
SDK, and live trading tasks, use the installed `open-xquant` skill first.
Do not run `oxq`, SDK code, scripts, or write report files until that router
skill has selected the specific open-xquant skill for the task.

Use `.open-xquant/workspace.yaml` for local paths.

## Version-Governed Artifact Contract

This workspace uses version-governed research artifacts. Read `current.json`
before writing any research artifact. The active version starts as `v001`.

Write phase artifacts only under the active version:

- `versions/v001/01_brainstorm/strategy_idea_brief.json`
- `versions/v001/02_idea_audit/strategy_idea_audit.json`
- `versions/v001/04_spec_build/strategy_spec.yaml`
- `versions/v001/06_spec_audit/spec_confirmation_table.md`
- `versions/v001/07_compile_preview/compiled_plan.json`
- `versions/v001/08_runtime_audit/runtime_audit.json`
- `versions/v001/09_backtests/<run_id>/strategy_spec.yaml`
- `versions/v001/10_reports/<run_id>/research_report.md`

Do not write root-level `strategy_idea_brief.json`,
`strategy_idea_audit.json`, `strategy_spec.yaml`, `spec_audit.json`,
`runtime_audit.json`, `research_report.md`, or `research_report.html`.
Do not write root-level `strategy_spec.yaml`.
Root-level phase artifacts are layout pollution, even if `oxq doctor` says the
workspace skeleton is OK."""


SUBAGENT_POLICY_BLOCK = """## SubAgent policy

- For open-xquant workflows, prefer SubAgents by default whenever SubAgent or
  multi-agent tools are available.
- The main agent acts as coordinator, reviewer, and final verifier.
- Before running `oxq`, SDK code, or report scripts, first check whether
  SubAgent tools are available.
- If SubAgent tools are unavailable, explicitly say so before continuing in
  the main thread.
- Delegate independent phases to workers:
  - version manager worker
  - artifact governor worker
  - strategy brainstorm worker
  - strategy idea audit worker
  - strategy builder worker
  - data inspection worker
  - spec audit worker
  - runtime audit worker
  - backtest runner worker
  - monitor worker
  - report writer/reviewer worker
  - comparison/final selection worker
- Do not force parallel execution when phases are strictly dependent. Use
  sequential SubAgents with artifact handoff instead."""


@click.group()
def research() -> None:
    """Manage open-xquant research workspaces."""


@research.command(name="init")
@click.option("--name", default=None, help="Workspace name.")
@click.option("--data-dir", default="~/.oxq/data/market", help="Market data directory.")
@click.option("--minimal", is_flag=True, help="Create only required workspace files.")
@click.option("--force", is_flag=True, help="Replace managed workspace config.")
@click.option("--sdk", is_flag=True, help="Install the cached open-xquant SDK bundle into the research workspace.")
@click.option("--sdk-venv", default=".venv", help="Virtual environment path for --sdk.")
def init_workspace(name: str | None, data_dir: str, minimal: bool, force: bool, sdk: bool, sdk_venv: str) -> None:
    """Initialize the current directory as an open-xquant research workspace."""

    initialize_workspace(Path.cwd(), name=name, data_dir=data_dir, minimal=minimal, force=force, sdk=sdk, sdk_venv=sdk_venv)


def initialize_workspace(
    cwd: Path,
    *,
    name: str | None = None,
    data_dir: str = "~/.oxq/data/market",
    minimal: bool = False,
    force: bool = False,
    sdk: bool = False,
    sdk_venv: str = ".venv",
) -> None:
    """Create open-xquant workspace files under cwd."""

    cwd = cwd.resolve()
    config_dir = cwd / ".open-xquant"
    workspace_file = config_dir / "workspace.yaml"
    sdk_state = None
    workspace_config: dict[str, object] | None = None
    created_workspace_config = False
    if sdk:
        sdk_state = install_workspace_sdk(cwd, _resolve_sdk_venv(cwd, sdk_venv))
    if workspace_file.exists() and not force:
        click.echo("open-xquant workspace already initialized")
        workspace_config = read_yaml_file(workspace_file)
        if sdk_state is not None:
            workspace_config["sdk"] = sdk_state
            write_yaml_file(workspace_file, workspace_config)
            click.echo(f"SDK config written to {workspace_file}")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        workspace_config = _workspace_payload(cwd, name, data_dir, sdk_state=sdk_state)
        write_yaml_file(workspace_file, workspace_config)
        created_workspace_config = True
        click.echo(f"Workspace config written to {workspace_file}")

    workspace_config = workspace_config or {}
    migrated_manifests = _migrate_hidden_root_manifest_files(cwd, workspace_config)
    normalized_config = _normalize_root_manifest_paths(workspace_config)
    normalized_config = _normalize_default_output_dir(cwd, workspace_config) or normalized_config
    if normalized_config:
        write_yaml_file(workspace_file, workspace_config)
        click.echo(f"Workspace config normalized in {workspace_file}")
    elif migrated_manifests:
        click.echo("Workspace root manifests migrated from .open-xquant/")
    if not minimal:
        _create_configured_workspace_dirs(cwd, workspace_config)
    experiments = _configured_path(cwd, workspace_config, "experiment_registry") or (cwd / "experiments.jsonl")
    if not experiments.exists():
        write_text_file(experiments, "")
    comparison_registry = _configured_path(cwd, workspace_config, "comparison_registry")
    if comparison_registry is None and created_workspace_config:
        comparison_registry = cwd / "comparisons" / "comparisons.jsonl"
    if not minimal and comparison_registry is not None and not comparison_registry.exists():
        write_text_file(comparison_registry, "")
    _create_default_governance_manifests(cwd, workspace_config)
    upsert_marker_block(cwd / "AGENTS.md", "open-xquant-workspace", _workspace_block(cwd, workspace_config))
    if _installed_agent_profile() == AGENT_PROFILE_STANDALONE:
        remove_marker_block(cwd / "AGENTS.md", "open-xquant-subagents")
    else:
        upsert_marker_block(cwd / "AGENTS.md", "open-xquant-subagents", SUBAGENT_POLICY_BLOCK)


def _workspace_payload(cwd: Path, name: str | None, data_dir: str, *, sdk_state: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": name or cwd.name,
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
        "data": {
            "market_data_dir": data_dir,
            "provider": "local",
        },
        "workflow": {
            "layout": "version_governed",
            "require_validate_before_backtest": True,
            "require_audit_before_report": True,
            "default_output_dir": "versions/{active_version}/09_backtests",
        },
    }
    if sdk_state is not None:
        payload["sdk"] = sdk_state
    return payload


def _create_configured_workspace_dirs(cwd: Path, workspace: dict[str, object]) -> None:
    created_configured_dir = False
    for key in (
        "specs_dir",
        "versions_dir",
        "conversations_dir",
        "components_dir",
        "governance_dir",
        "runs_dir",
        "reports_dir",
        "final_dir",
        "comparisons_dir",
    ):
        if key == "runs_dir" and _uses_version_local_backtest_output(cwd, workspace):
            continue
        path = _configured_path(cwd, workspace, key)
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            created_configured_dir = True
    if not created_configured_dir:
        (cwd / "versions").mkdir(exist_ok=True)
        (cwd / "conversations").mkdir(exist_ok=True)
        (cwd / "components").mkdir(exist_ok=True)
        (cwd / "governance").mkdir(exist_ok=True)
        if not _uses_version_local_backtest_output(cwd, workspace):
            (cwd / "runs").mkdir(exist_ok=True)
        (cwd / "final").mkdir(exist_ok=True)
        (cwd / "comparisons").mkdir(exist_ok=True)


def _create_default_governance_manifests(cwd: Path, workspace: dict[str, object]) -> None:
    if not _is_version_governed_workspace(workspace):
        return
    name = str(workspace.get("name") or cwd.name)
    paths = workspace.get("paths") if isinstance(workspace.get("paths"), dict) else {}
    current_path = _configured_root_manifest_path(cwd, workspace, "current_manifest", "current.json")
    current_payload = _read_json_object(current_path)
    active_version = current_payload.get("active_version")
    active_phase = current_payload.get("active_phase")
    if not isinstance(active_phase, str) or not active_phase:
        active_phase = "01_brainstorm"
    if active_version:
        if not isinstance(active_version, str) or not _WORKSPACE_VERSION_RE.fullmatch(active_version):
            raise click.ClickException(
                f"workspace current.json active_version is unsafe: {active_version}; "
                "repair current.json before running research init"
            )
        version_id = active_version
    else:
        version_id = "v001"
    version_dir = (_configured_path(cwd, workspace, "versions_dir") or (cwd / "versions")) / version_id
    version_dir_display = _display_workspace_path(cwd, version_dir)
    _create_version_phase_dirs(version_dir)

    workflow_path = _configured_root_manifest_path(cwd, workspace, "workflow_manifest", "workflow_manifest.json")
    if workflow_path is not None and not workflow_path.exists():
        write_text_file(
            workflow_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "layout": "version_governed",
                    "strategy_family_id": name,
                    "paths": paths,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )

    if not current_payload.get("active_version"):
        write_text_file(
            current_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_family_id": name,
                    "active_version": version_id,
                    "active_phase": active_phase,
                    "active_run": "",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )

    lineage_path = _configured_root_manifest_path(cwd, workspace, "lineage_manifest", "lineage.json")
    lineage_payload = _read_json_object(lineage_path)
    versions = lineage_payload.get("versions")
    if isinstance(versions, list):
        lineage_versions = list(versions)
    else:
        lineage_versions = []
    if not any(isinstance(item, dict) and item.get("version_id") == version_id for item in lineage_versions):
        lineage_versions.append(
            {
                "version_id": version_id,
                "parent_version_id": "",
                "created_reason": "initial_strategy_version",
                "status": "active",
            }
        )
    if lineage_payload.get("versions") != lineage_versions:
        lineage_payload = {
            **lineage_payload,
            "schema_version": lineage_payload.get("schema_version", 1),
            "strategy_family_id": lineage_payload.get("strategy_family_id", name),
            "versions": lineage_versions,
        }
        write_text_file(
            lineage_path,
            json.dumps(lineage_payload, indent=2, ensure_ascii=False) + "\n",
        )

    expected_phase_paths = {
        phase: f"{version_dir_display}/{phase}" for phase in VERSION_PHASE_DIRS
    }
    version_manifest = version_dir / "version_manifest.json"
    if version_manifest.exists():
        version_manifest_payload = _read_json_object(version_manifest)
        repaired_manifest = {
            **version_manifest_payload,
            "schema_version": version_manifest_payload.get("schema_version", 1),
            "version_id": version_id,
            "strategy_family_id": version_manifest_payload.get("strategy_family_id", name),
            "active_phase": active_phase,
            "phase_paths": expected_phase_paths,
        }
        if version_manifest_payload != repaired_manifest:
            write_text_file(
                version_manifest,
                json.dumps(repaired_manifest, indent=2, ensure_ascii=False) + "\n",
            )
    else:
        write_text_file(
            version_manifest,
            json.dumps(
                {
                    "schema_version": 1,
                    "version_id": version_id,
                    "strategy_family_id": name,
                    "parent_version_id": "",
                    "created_reason": "initial_strategy_version",
                    "status": "active",
                    "active_phase": active_phase,
                    "source_conversation": "",
                    "phase_paths": expected_phase_paths,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )

    phase_state = version_dir / "phase_state.json"
    if not phase_state.exists():
        write_text_file(
            phase_state,
            json.dumps(
                {
                    "schema_version": 1,
                    "version_id": version_id,
                    "current_phase": active_phase,
                    "status": "active",
                    "completed_phases": [],
                    "blocked_phase": "",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )


def _configured_path(cwd: Path, workspace: dict[str, object], key: str) -> Path | None:
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        return None
    value = paths.get(key)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise click.ClickException(f"workspace paths.{key} must be a safe relative path")
    resolved_cwd = cwd.resolve()
    candidate = cwd / path
    try:
        candidate.resolve(strict=False).relative_to(resolved_cwd)
    except ValueError as exc:
        raise click.ClickException(f"workspace paths.{key} must stay within the workspace") from exc
    return candidate


def _display_workspace_path(cwd: Path, path: Path) -> str:
    try:
        return path.relative_to(cwd).as_posix()
    except ValueError:
        return path.as_posix()


def _workspace_versions_dir_display(cwd: Path, workspace: dict[str, object]) -> str:
    return _display_workspace_path(cwd, _configured_path(cwd, workspace, "versions_dir") or (cwd / "versions"))


def _workspace_backtest_output_template(cwd: Path, workspace: dict[str, object]) -> str:
    return f"{_workspace_versions_dir_display(cwd, workspace)}/{{active_version}}/09_backtests"


def _workspace_block(cwd: Path, workspace: dict[str, object]) -> str:
    version_root = _workspace_versions_dir_display(cwd, workspace)
    return WORKSPACE_BLOCK.replace("versions/v001", f"{version_root}/v001")


def _configured_root_manifest_path(cwd: Path, workspace: dict[str, object], key: str, filename: str) -> Path:
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        return cwd / filename
    value = paths.get(key)
    if not isinstance(value, str) or not value:
        return cwd / filename
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != filename:
        return cwd / filename
    return cwd / value


def _normalize_root_manifest_paths(workspace: dict[str, object]) -> bool:
    if not _is_version_governed_workspace(workspace):
        return False
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        paths = {}
        workspace["paths"] = paths
    changed = False
    for key, filename in (
        ("current_manifest", "current.json"),
        ("lineage_manifest", "lineage.json"),
        ("workflow_manifest", "workflow_manifest.json"),
    ):
        if paths.get(key) != filename:
            paths[key] = filename
            changed = True
    return changed


def _normalize_default_output_dir(cwd: Path, workspace: dict[str, object]) -> bool:
    if not _is_version_governed_workspace(workspace):
        return False
    workflow = workspace.get("workflow")
    if not isinstance(workflow, dict):
        workflow = {}
        workspace["workflow"] = workflow
    expected = _workspace_backtest_output_template(cwd, workspace)
    configured = workflow.get("default_output_dir")
    if configured == "versions/{active_version}/09_backtests" and configured != expected:
        workflow["default_output_dir"] = expected
        return True
    if configured in {"runs/auto", "runs/auto/runs/runs/{active_version}", "runs", "runs/{active_version}"}:
        workflow["default_output_dir"] = expected
        return True
    if not isinstance(configured, str) or not configured:
        workflow["default_output_dir"] = expected
        return True
    return False


def _migrate_hidden_root_manifest_files(cwd: Path, workspace: dict[str, object]) -> bool:
    if not _is_version_governed_workspace(workspace):
        return False
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        return False
    migrated = False
    for key, filename in (
        ("current_manifest", "current.json"),
        ("lineage_manifest", "lineage.json"),
        ("workflow_manifest", "workflow_manifest.json"),
    ):
        configured = paths.get(key)
        if not isinstance(configured, str) or not configured:
            continue
        configured_path = Path(configured)
        if configured_path.is_absolute() or ".." in configured_path.parts:
            continue
        if len(configured_path.parts) != 2 or configured_path.parts[0] != ".open-xquant":
            continue
        source = cwd / configured_path
        target = cwd / filename
        if source.is_file():
            source_content = source.read_text(encoding="utf-8")
            target_content = target.read_text(encoding="utf-8") if target.exists() else None
            if target_content != source_content:
                write_text_file(target, source_content)
                migrated = True
    return migrated


VERSION_PHASE_DIRS = (
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


def _is_version_governed_workspace(workspace: dict[str, object]) -> bool:
    workflow = workspace.get("workflow")
    if isinstance(workflow, dict) and workflow.get("layout") == "version_governed":
        return True
    paths = workspace.get("paths")
    return isinstance(paths, dict) and "versions_dir" in paths


def _uses_version_local_backtest_output(cwd: Path, workspace: dict[str, object]) -> bool:
    workflow = workspace.get("workflow")
    if not isinstance(workflow, dict):
        return False
    return workflow.get("layout") == "version_governed" and workflow.get(
        "default_output_dir"
    ) == _workspace_backtest_output_template(cwd, workspace)


def _create_version_phase_dirs(version_dir: Path) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    for phase in VERSION_PHASE_DIRS:
        (version_dir / phase).mkdir(parents=True, exist_ok=True)


def _read_json_object(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_sdk_venv(cwd: Path, raw_path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if expanded.is_absolute():
        return expanded.resolve()
    return (cwd / expanded).resolve()


def _installed_agent_profile() -> str:
    config_path = Path.home() / ".config" / "open-xquant" / "agent.yaml"
    if not config_path.exists():
        return AGENT_PROFILE_MULTI
    value = read_yaml_file(config_path).get("agent_profile")
    if value == AGENT_PROFILE_STANDALONE:
        return AGENT_PROFILE_STANDALONE
    return AGENT_PROFILE_MULTI
