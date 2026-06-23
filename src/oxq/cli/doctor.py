"""Environment checks for open-xquant Agent workflows."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import click

from oxq.cli.agent import manifest_path
from oxq.cli.agent_manifest import read_json_file, read_yaml_file
from oxq.cli.research import initialize_workspace


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
@click.option("--fix", is_flag=True, help="Apply safe fixes.")
def doctor(as_json: bool, fix: bool) -> None:
    """Check CLI, Agent, workspace, data, and optional dependency readiness."""

    if fix and not (Path.cwd() / ".open-xquant" / "workspace.yaml").exists():
        if as_json:
            with redirect_stdout(StringIO()):
                initialize_workspace(Path.cwd())
        else:
            initialize_workspace(Path.cwd())
    payload = _doctor_payload()
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo("open-xquant doctor")
    click.echo("")
    for name in ("cli", "agent", "workspace", "data", "deps"):
        click.echo(f"{name.upper()}: {payload['checks'][name]['status'].upper()}")
    if payload["fixes"]:
        click.echo("")
        click.echo("Suggested fixes:")
        for fix_cmd in payload["fixes"]:
            click.echo(f"- {fix_cmd}")


def _doctor_payload() -> dict[str, Any]:
    checks = {
        "cli": _check_cli(),
        "agent": _check_agent(),
        "workspace": _check_workspace(),
        "data": _check_data(),
        "deps": _check_deps(),
    }
    fixes: list[str] = []
    for check in checks.values():
        fixes.extend(check.get("fixes", []))
    statuses = {check["status"] for check in checks.values()}
    status = "fail" if "fail" in statuses else ("warn" if "warn" in statuses or "missing" in statuses else "ok")
    return {"status": status, "checks": checks, "fixes": fixes}


def _check_cli() -> dict[str, Any]:
    return {"status": "ok" if sys.version_info >= (3, 12) else "fail", "python": sys.version.split()[0]}


def _check_agent() -> dict[str, Any]:
    if not manifest_path().exists():
        return {"status": "missing", "fixes": ["oxq agent install"]}
    manifest = read_json_file(manifest_path())
    targets = manifest.get("targets", {}) if isinstance(manifest.get("targets"), dict) else {}
    installed_targets = {
        target_id: state
        for target_id, state in targets.items()
        if isinstance(state, dict) and state.get("installed")
    }
    missing_paths: list[str] = []
    installed_count = 0
    expected_count = 0
    for state in installed_targets.values():
        skills = state.get("skills", []) if isinstance(state.get("skills"), list) else []
        expected_count += len(skills)
        for record in skills:
            if not isinstance(record, dict):
                continue
            if Path(record["dest"]).exists():
                installed_count += 1
            else:
                missing_paths.append(record["dest"])
    return {
        "status": "ok" if installed_targets and not missing_paths else "warn",
        "targets": sorted(installed_targets),
        "skills": {"installed": installed_count, "expected": expected_count},
        "missing_paths": missing_paths,
    }


def _check_workspace() -> dict[str, Any]:
    workspace = Path.cwd() / ".open-xquant" / "workspace.yaml"
    if not workspace.exists():
        return {"status": "missing", "fixes": ["oxq research init"]}
    try:
        config = read_yaml_file(workspace)
    except Exception as exc:
        return {
            "status": "fail",
            "path": str(workspace),
            "error": str(exc),
            "fixes": ["oxq research init --force"],
        }
    configured_paths = _workspace_required_paths(config)
    missing = [
        str(path)
        for path in configured_paths
        if not path.exists()
    ]
    return {"status": "ok" if not missing else "warn", "missing": missing}


def _workspace_required_paths(config: dict[str, Any]) -> list[Path]:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        paths = {}
    required_keys = (
        "specs_dir",
        "runs_dir",
        "reports_dir",
        "final_dir",
        "comparisons_dir",
        "experiment_registry",
        "comparison_registry",
    )
    configured = [
        Path.cwd() / value
        for key in required_keys
        if isinstance((value := paths.get(key)), str) and value
    ]
    if configured:
        return configured
    return [
        Path.cwd() / "runs",
        Path.cwd() / "runs" / "final",
        Path.cwd() / "comparisons",
        Path.cwd() / "experiments.jsonl",
    ]


def _check_data() -> dict[str, Any]:
    data_dir = Path.home() / ".oxq" / "data" / "market"
    return {"status": "ok" if data_dir.exists() else "warn", "path": str(data_dir)}


def _check_deps() -> dict[str, Any]:
    core_modules = ("pandas", "numpy", "pyarrow", "yaml", "click", "exchange_calendars")
    optional_modules = {
        "yfinance": "uv sync --extra yfinance",
        "akshare": "uv sync --extra akshare",
        "scipy": "uv sync --extra scipy",
        "matplotlib": "uv sync --extra chart",
        "mplfinance": "uv sync --extra chart",
        "seaborn": "uv sync --extra chart",
        "httpx": "uv sync --extra live",
        "socksio": "uv sync --extra live",
        "websockets": "uv sync --extra live",
        "tabulate": "uv sync --extra dev",
    }
    missing_core = [
        module
        for module in core_modules
        if importlib.util.find_spec(module) is None
    ]
    missing_optional = sorted(
        module
        for module in optional_modules
        if importlib.util.find_spec(module) is None
    )
    fixes = sorted({optional_modules[module] for module in missing_optional})
    if missing_core:
        fixes.insert(0, "uv sync --all-extras")
    status = "fail" if missing_core else ("warn" if missing_optional else "ok")
    return {
        "status": status,
        "missing": missing_core + missing_optional,
        "missing_core": missing_core,
        "missing_optional": missing_optional,
        "fixes": fixes,
    }
