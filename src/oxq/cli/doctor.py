"""Environment checks for open-xquant Agent workflows."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import click

from oxq.cli.agent import manifest_path
from oxq.cli.agent_manifest import read_json_file, read_yaml_file
from oxq.cli.research import initialize_workspace

_WORKSPACE_VERSION_RE = re.compile(r"^v[0-9][A-Za-z0-9_-]*$")


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
    governance_warnings = _workspace_governance_warnings(config)
    return {
        "status": "ok" if not missing and not governance_warnings else "warn",
        "missing": missing,
        "governance_warnings": governance_warnings,
    }


def _workspace_required_paths(config: dict[str, Any]) -> list[Path]:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        paths = {}
    optional_keys: set[str] = set()
    if _uses_version_local_backtest_output(config):
        optional_keys.add("runs_dir")
    required_keys = (
        "specs_dir",
        "versions_dir",
        "conversations_dir",
        "components_dir",
        "governance_dir",
        "runs_dir",
        "reports_dir",
        "final_dir",
        "comparisons_dir",
        "current_manifest",
        "lineage_manifest",
        "workflow_manifest",
        "experiment_registry",
        "comparison_registry",
    )
    configured = [
        Path.cwd() / value
        for key in required_keys
        if key not in optional_keys
        if isinstance((value := paths.get(key)), str) and value
    ]
    if configured:
        return configured
    return [
        Path.cwd() / "versions",
        Path.cwd() / "conversations",
        Path.cwd() / "components",
        Path.cwd() / "governance",
        Path.cwd() / "final",
        Path.cwd() / "comparisons",
        Path.cwd() / "current.json",
        Path.cwd() / "lineage.json",
        Path.cwd() / "workflow_manifest.json",
        Path.cwd() / "experiments.jsonl",
    ]


def _uses_version_local_backtest_output(config: dict[str, Any]) -> bool:
    workflow = config.get("workflow")
    if not isinstance(workflow, dict):
        return False
    return (
        workflow.get("layout") == "version_governed"
        and workflow.get("default_output_dir") == "versions/{active_version}/09_backtests"
    )


def _workspace_governance_warnings(config: dict[str, Any]) -> list[str]:
    workflow = config.get("workflow")
    paths = config.get("paths")
    if not isinstance(workflow, dict) or workflow.get("layout") != "version_governed":
        return []
    if not isinstance(paths, dict):
        paths = {}

    warnings: list[str] = []
    warnings.extend(_root_manifest_path_warnings(paths))
    current_path = Path.cwd() / str(paths.get("current_manifest", "current.json"))
    lineage_path = Path.cwd() / str(paths.get("lineage_manifest", "lineage.json"))

    current = _read_json_object(current_path)
    lineage = _read_json_object(lineage_path)

    active_version = current.get("active_version")
    active_phase = current.get("active_phase")
    if not isinstance(active_version, str) or not active_version:
        warnings.append("active_version_missing")
    elif not _WORKSPACE_VERSION_RE.fullmatch(active_version):
        warnings.append("active_version_invalid")
    if not isinstance(active_phase, str) or not active_phase:
        warnings.append("active_phase_missing")

    versions = lineage.get("versions")
    if not isinstance(versions, list) or not versions:
        warnings.append("lineage_versions_empty")

    if isinstance(active_version, str) and active_version and _WORKSPACE_VERSION_RE.fullmatch(active_version):
        version_dir = Path.cwd() / str(paths.get("versions_dir", "versions")) / active_version
        if not version_dir.exists():
            warnings.append("active_version_dir_missing")
        if versions and isinstance(versions, list) and not any(
            isinstance(item, dict) and item.get("version_id") == active_version for item in versions
        ):
            warnings.append("active_version_not_in_lineage")
        if version_dir.exists():
            phase_state = _read_json_object(version_dir / "phase_state.json")
            version_manifest = _read_json_object(version_dir / "version_manifest.json")
            phase_state_phase = phase_state.get("current_phase")
            version_manifest_phase = version_manifest.get("active_phase")
            if isinstance(phase_state_phase, str) and phase_state_phase and phase_state_phase != active_phase:
                warnings.append("active_phase_mismatch:phase_state")
            if (
                isinstance(version_manifest_phase, str)
                and version_manifest_phase
                and version_manifest_phase != active_phase
            ):
                warnings.append("active_phase_mismatch:version_manifest")
            if _has_passing_report_review(version_dir):
                if active_phase != "10_reports":
                    warnings.append("active_phase_stale:report_review_passed")
                if phase_state_phase != "10_reports":
                    warnings.append("phase_state_stale:report_review_passed")
                if version_manifest_phase != "10_reports":
                    warnings.append("version_manifest_phase_stale:report_review_passed")

    for artifact in ROOT_PHASE_ARTIFACTS:
        if (Path.cwd() / artifact).exists():
            warnings.append(f"root_phase_artifact:{artifact}")

    return warnings


def _root_manifest_path_warnings(paths: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key, filename in (
        ("current_manifest", "current.json"),
        ("lineage_manifest", "lineage.json"),
        ("workflow_manifest", "workflow_manifest.json"),
    ):
        raw_value = paths.get(key, filename)
        if not isinstance(raw_value, str) or not raw_value:
            raw_value = filename
        path = Path(raw_value)
        if path.is_absolute() or len(path.parts) != 1 or path.name != filename:
            warnings.append(f"root_manifest_path_invalid:{key}")
    return warnings


def _has_passing_report_review(version_dir: Path) -> bool:
    reports_dir = version_dir / "10_reports"
    if not reports_dir.exists():
        return False
    for review_path in reports_dir.glob("*/report_review.json"):
        review = _read_json_object(review_path)
        if review.get("status") == "pass":
            return True
    return False


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


ROOT_PHASE_ARTIFACTS = (
    "strategy_idea_brief.json",
    "strategy_idea_audit.json",
    "data_inspection_result.json",
    "data_availability_report.md",
    "strategy_spec.yaml",
    "component_request.json",
    "component_manifest.json",
    "component_catalog.json",
    "spec_build_notes.md",
    "spec_mapping_notes.md",
    "spec_mapping_contract.json",
    "builder_phase_result.json",
    "spec_audit.json",
    "audit_notes.md",
    "spec_confirmation_table.md",
    "compile_preview",
    "runtime_audit.json",
    "compiled_plan.json",
    "backtest_authorization.json",
    "runner_result.json",
    "result.json",
    "research_report.md",
    "research_report.html",
    "writer_result.json",
    "report_review.json",
    "report_assets",
)


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
