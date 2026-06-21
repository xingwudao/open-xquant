"""Research workspace initialization commands."""

from __future__ import annotations

from pathlib import Path

import click

from oxq.cli.agent_manifest import expand_path, read_yaml_file, upsert_marker_block, write_text_file, write_yaml_file
from oxq.cli.sdk_bundle import install_workspace_sdk

WORKSPACE_BLOCK = """This is an open-xquant research workspace.

For quant strategy, factor, backtest, audit, robustness, report,
and live trading tasks, use the installed open-xquant skills.

Use `.open-xquant/workspace.yaml` for local paths."""


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
    if sdk:
        sdk_state = install_workspace_sdk(cwd, _resolve_sdk_venv(cwd, sdk_venv), force=force)
    if workspace_file.exists() and not force:
        click.echo("open-xquant workspace already initialized")
        if sdk_state is not None:
            workspace = read_yaml_file(workspace_file)
            workspace["sdk"] = sdk_state
            write_yaml_file(workspace_file, workspace)
            click.echo(f"SDK config written to {workspace_file}")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        write_yaml_file(workspace_file, _workspace_payload(cwd, name, data_dir, sdk_state=sdk_state))
        click.echo(f"Workspace config written to {workspace_file}")

    if not minimal:
        for dirname in ("strategy_specs", "runs", "reports"):
            (cwd / dirname).mkdir(exist_ok=True)
    experiments = cwd / "experiments.jsonl"
    if not experiments.exists():
        write_text_file(experiments, "")
    upsert_marker_block(cwd / "AGENTS.md", "open-xquant-workspace", WORKSPACE_BLOCK)


def _workspace_payload(cwd: Path, name: str | None, data_dir: str, *, sdk_state: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": name or cwd.name,
        "paths": {
            "specs_dir": "strategy_specs",
            "runs_dir": "runs",
            "reports_dir": "reports",
            "experiment_registry": "experiments.jsonl",
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


def _resolve_sdk_venv(cwd: Path, raw_path: str) -> Path:
    path = expand_path(raw_path)
    if Path(raw_path).expanduser().is_absolute():
        return path
    return (cwd / raw_path).resolve()
