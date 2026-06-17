"""Environment checks for open-xquant Agent workflows."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import click

from oxq.cli.agent import manifest_path
from oxq.cli.agent_manifest import read_json_file
from oxq.cli.research import initialize_workspace


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
@click.option("--fix", is_flag=True, help="Apply safe fixes.")
def doctor(as_json: bool, fix: bool) -> None:
    """Check CLI, Agent, workspace, data, and optional dependency readiness."""

    if fix and not (Path.cwd() / ".open-xquant" / "workspace.yaml").exists():
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
    missing = [
        str(path)
        for path in (
            Path.cwd() / "strategy_specs",
            Path.cwd() / "runs",
            Path.cwd() / "reports",
            Path.cwd() / "experiments.jsonl",
        )
        if not path.exists()
    ]
    return {"status": "ok" if not missing else "warn", "missing": missing}


def _check_data() -> dict[str, Any]:
    data_dir = Path.home() / ".oxq" / "data" / "market"
    return {"status": "ok" if data_dir.exists() else "warn", "path": str(data_dir)}


def _check_deps() -> dict[str, Any]:
    missing = [
        module
        for module in ("yfinance", "mplfinance", "httpx", "websockets")
        if importlib.util.find_spec(module) is None
    ]
    fixes = []
    if "mplfinance" in missing:
        fixes.append("uv sync --extra chart")
    if "httpx" in missing or "websockets" in missing:
        fixes.append("uv sync --extra live")
    if "yfinance" in missing:
        fixes.append("uv sync --extra yfinance")
    return {"status": "ok" if not missing else "warn", "missing": missing, "fixes": fixes}
