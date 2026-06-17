"""Research workspace initialization commands."""

from __future__ import annotations

from pathlib import Path

import click

from oxq.cli.agent_manifest import upsert_marker_block, write_text_file, write_yaml_file

WORKSPACE_BLOCK = """This is an open-xquant research workspace.

For quant strategy, factor, backtest, audit, robustness, report,
and live trading tasks, use the installed open-xquant skills.

Use `.open-xquant/workspace.yaml` for local paths."""


@click.group()
def research() -> None:
    """Manage open-xquant research workspaces."""


@research.command(name="init")
@click.option("--name", default=None, help="Workspace name.")
@click.option("--data-dir", default="~/.oxq/data", help="Market data directory.")
@click.option("--minimal", is_flag=True, help="Create only required workspace files.")
@click.option("--force", is_flag=True, help="Replace managed workspace config.")
def init_workspace(name: str | None, data_dir: str, minimal: bool, force: bool) -> None:
    """Initialize the current directory as an open-xquant research workspace."""

    initialize_workspace(Path.cwd(), name=name, data_dir=data_dir, minimal=minimal, force=force)


def initialize_workspace(
    cwd: Path,
    *,
    name: str | None = None,
    data_dir: str = "~/.oxq/data",
    minimal: bool = False,
    force: bool = False,
) -> None:
    """Create open-xquant workspace files under cwd."""

    cwd = cwd.resolve()
    config_dir = cwd / ".open-xquant"
    workspace_file = config_dir / "workspace.yaml"
    if workspace_file.exists() and not force:
        click.echo("open-xquant workspace already initialized")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)
        write_yaml_file(workspace_file, _workspace_payload(cwd, name, data_dir))
        click.echo(f"Workspace config written to {workspace_file}")

    if not minimal:
        for dirname in ("strategy_specs", "runs", "reports"):
            (cwd / dirname).mkdir(exist_ok=True)
    experiments = cwd / "experiments.jsonl"
    if not experiments.exists():
        write_text_file(experiments, "")
    upsert_marker_block(cwd / "AGENTS.md", "open-xquant-workspace", WORKSPACE_BLOCK)


def _workspace_payload(cwd: Path, name: str | None, data_dir: str) -> dict[str, object]:
    return {
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
