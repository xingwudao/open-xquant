"""Research workspace initialization commands."""

from __future__ import annotations

import os
from pathlib import Path

import click

from oxq.cli.agent_manifest import read_yaml_file, remove_marker_block, upsert_marker_block, write_text_file, write_yaml_file
from oxq.cli.sdk_bundle import install_workspace_sdk

AGENT_PROFILE_MULTI = "multi-agent"
AGENT_PROFILE_STANDALONE = "standalone-agent"

WORKSPACE_BLOCK = """This is an open-xquant research workspace.

For quant strategy, factor, backtest, audit, robustness, report, chart asset,
SDK, and live trading tasks, use the installed `open-xquant` skill first.
Do not run `oxq`, SDK code, scripts, or write report files until that router
skill has selected the specific open-xquant skill for the task.

Use `.open-xquant/workspace.yaml` for local paths."""


SUBAGENT_POLICY_BLOCK = """## SubAgent policy

- For open-xquant workflows, prefer SubAgents by default whenever SubAgent or
  multi-agent tools are available.
- The main agent acts as coordinator, reviewer, and final verifier.
- Before running `oxq`, SDK code, or report scripts, first check whether
  SubAgent tools are available.
- If SubAgent tools are unavailable, explicitly say so before continuing in
  the main thread.
- Delegate independent phases to workers:
  - strategy builder worker
  - data inspection worker
  - spec audit worker
  - runtime audit worker
  - backtest runner worker
  - monitor/report worker
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
    upsert_marker_block(cwd / "AGENTS.md", "open-xquant-workspace", WORKSPACE_BLOCK)
    if _installed_agent_profile() == AGENT_PROFILE_STANDALONE:
        remove_marker_block(cwd / "AGENTS.md", "open-xquant-subagents")
    else:
        upsert_marker_block(cwd / "AGENTS.md", "open-xquant-subagents", SUBAGENT_POLICY_BLOCK)


def _workspace_payload(cwd: Path, name: str | None, data_dir: str, *, sdk_state: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": name or cwd.name,
        "paths": {
            "current_spec": "strategy_spec.yaml",
            "runs_dir": "runs",
            "final_dir": "runs/final",
            "comparisons_dir": "comparisons",
            "experiment_registry": "experiments.jsonl",
            "comparison_registry": "comparisons/comparisons.jsonl",
        },
        "data": {
            "market_data_dir": data_dir,
            "provider": "local",
        },
        "workflow": {
            "require_validate_before_backtest": True,
            "require_audit_before_report": True,
            "default_output_dir": "runs/auto",
        },
    }
    if sdk_state is not None:
        payload["sdk"] = sdk_state
    return payload


def _create_configured_workspace_dirs(cwd: Path, workspace: dict[str, object]) -> None:
    created_configured_dir = False
    for key in ("specs_dir", "runs_dir", "reports_dir", "final_dir", "comparisons_dir"):
        path = _configured_path(cwd, workspace, key)
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            created_configured_dir = True
    if not created_configured_dir:
        (cwd / "runs" / "final").mkdir(parents=True, exist_ok=True)
        (cwd / "comparisons").mkdir(exist_ok=True)


def _configured_path(cwd: Path, workspace: dict[str, object], key: str) -> Path | None:
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        return None
    value = paths.get(key)
    if not isinstance(value, str) or not value:
        return None
    return cwd / value


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
