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

from oxq.cli.agent import agent_lifecycle_lock, manifest_path
from oxq.cli.agent_manifest import read_json_file
from oxq.cli.research import (
    _LINEAGE_STATUSES,
    VERSION_PHASE_DIRS,
    _classify_workspace_governance,
    _configured_path,
    _lineage_identity_matches,
    _lineage_parent_version_id_is_valid,
    _resolve_active_version_dir,
    _resolve_version_phase_path,
    _workflow_manifest_is_valid,
    _workflow_manifest_path_mismatches,
    _workspace_config_directory_is_link,
    initialize_workspace,
)
from oxq.cli.research import (
    _phase_state_is_valid as _active_phase_state_is_valid,
)
from oxq.core.workspace_config import load_workspace_config

_WORKSPACE_VERSION_RE = re.compile(r"^v[0-9][A-Za-z0-9_-]*$")


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
@click.option("--fix", is_flag=True, help="Apply safe fixes.")
def doctor(as_json: bool, fix: bool) -> None:
    """Check CLI, Agent, workspace, data, and optional dependency readiness."""

    if (
        fix
        and not (Path.cwd() / ".open-xquant" / "workspace.yaml").exists()
        and not (Path.cwd() / ".open-xquant" / "workspace.yaml").is_symlink()
        and not _workspace_config_directory_is_link(Path.cwd())
    ):
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
    with agent_lifecycle_lock():
        return _check_agent_locked()


def _check_agent_locked() -> dict[str, Any]:
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
    if _workspace_config_directory_is_link(Path.cwd()):
        return {
            "status": "fail",
            "path": str(workspace),
            "error": "workspace configuration directory must not be a symlink or reparse point",
            "fixes": ["Replace the .open-xquant symlink with a real directory, then run oxq research init"],
        }
    if not workspace.exists() and not workspace.is_symlink():
        return {"status": "missing", "fixes": ["oxq research init"]}
    try:
        config = load_workspace_config(workspace, allow_empty=True)
    except Exception as exc:
        fixes = (
            ["Replace the .open-xquant symlink with a real directory, then run oxq research init"]
            if _workspace_config_directory_is_link(Path.cwd())
            else ["oxq research init --force"]
        )
        return {
            "status": "fail",
            "path": str(workspace),
            "error": str(exc),
            "fixes": fixes,
        }
    configured_paths, path_warnings = _workspace_required_paths_and_warnings(config)
    missing = [
        str(path)
        for path in configured_paths
        if not path.exists()
    ]
    governance_warnings = list(dict.fromkeys([*path_warnings, *_workspace_governance_warnings(config)]))
    return {
        "status": "ok" if not missing and not governance_warnings else "warn",
        "missing": missing,
        "governance_warnings": governance_warnings,
    }


def _workspace_required_paths(config: dict[str, Any]) -> list[Path]:
    configured, _warnings = _workspace_required_paths_and_warnings(config)
    return configured


def _workspace_required_paths_and_warnings(config: dict[str, Any]) -> tuple[list[Path], list[str]]:
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
    defaults = (
        {
            "versions_dir": "versions",
            "conversations_dir": "conversations",
            "components_dir": "components",
            "governance_dir": "governance",
            "final_dir": "final",
            "comparisons_dir": "comparisons",
            "current_manifest": "current.json",
            "lineage_manifest": "lineage.json",
            "workflow_manifest": "workflow_manifest.json",
            "experiment_registry": "experiments.jsonl",
            "comparison_registry": "comparisons/comparisons.jsonl",
        }
        if _is_version_governed_workspace(config)
        else {}
    )
    configured: list[Path] = []
    warnings: list[str] = []
    for key in required_keys:
        if key in optional_keys:
            continue
        value = paths.get(key, defaults.get(key))
        if not isinstance(value, str) or not value:
            continue
        try:
            configured_path = (
                _configured_path(Path.cwd(), config, key)
                if key in paths
                else Path.cwd() / value
            )
        except click.ClickException:
            warnings.append(f"{key}_unsafe")
            continue
        if configured_path is not None:
            configured.append(configured_path)
    if configured:
        return configured, warnings
    if warnings:
        return [], warnings
    return (
        [
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
        ],
        warnings,
    )


def _uses_version_local_backtest_output(config: dict[str, Any]) -> bool:
    workflow = config.get("workflow")
    if not isinstance(workflow, dict):
        return _is_version_governed_workspace(config)
    return (
        workflow.get("layout") == "version_governed"
        and workflow.get("default_output_dir") == "versions/{active_version}/09_backtests"
    ) or _is_version_governed_workspace(config)


def _workspace_governance_warnings(config: dict[str, Any]) -> list[str]:
    paths = config.get("paths")
    governed, classification_error = _classify_workspace_governance(config)
    if not governed:
        return []
    if not isinstance(paths, dict):
        paths = {}

    warnings: list[str] = []
    if classification_error is not None:
        warnings.append("versions_dir_invalid")
    warnings.extend(_root_manifest_path_warnings(paths))
    try:
        versions_dir = _configured_path(Path.cwd(), config, "versions_dir") or (Path.cwd() / "versions")
    except click.ClickException:
        versions_dir = None
        warnings.append("versions_dir_unsafe")
    try:
        current_path = _configured_path(Path.cwd(), config, "current_manifest") or (Path.cwd() / "current.json")
    except click.ClickException:
        current_path = None
        warnings.append("current_manifest_unsafe")
    try:
        lineage_path = _configured_path(Path.cwd(), config, "lineage_manifest") or (Path.cwd() / "lineage.json")
    except click.ClickException:
        lineage_path = None
        warnings.append("lineage_manifest_unsafe")
    try:
        workflow_path = _configured_path(Path.cwd(), config, "workflow_manifest") or (
            Path.cwd() / "workflow_manifest.json"
        )
    except click.ClickException:
        workflow_path = None
        warnings.append("workflow_manifest_unsafe")

    current, current_status = (
        _read_json_object_state(current_path)
        if current_path is not None
        else ({}, "missing")
    )
    lineage, lineage_status = (
        _read_json_object_state(lineage_path)
        if lineage_path is not None
        else ({}, "missing")
    )
    workflow_manifest, workflow_manifest_status = (
        _read_json_object_state(workflow_path)
        if workflow_path is not None
        else ({}, "missing")
    )
    if current_status != "ok" or not _current_manifest_is_valid(current):
        warnings.append(
            f"current_manifest_{current_status if current_status != 'ok' else 'invalid'}"
        )
    if lineage_status != "ok" or not _lineage_manifest_root_is_valid(lineage):
        warnings.append(
            f"lineage_manifest_{lineage_status if lineage_status != 'ok' else 'invalid'}"
        )
    if workflow_manifest_status != "ok" or not _workflow_manifest_is_valid(
        workflow_manifest
    ):
        warnings.append(
            "workflow_manifest_"
            + (
                workflow_manifest_status
                if workflow_manifest_status != "ok"
                else "invalid"
            )
        )
    else:
        warnings.extend(
            f"workflow_manifest_path_mismatch:{key}"
            for key in _workflow_manifest_path_mismatches(config, workflow_manifest)
        )

    strategy_family_id = current.get("strategy_family_id")
    if isinstance(strategy_family_id, str) and strategy_family_id:
        if lineage.get("strategy_family_id") != strategy_family_id:
            warnings.append("strategy_family_id_mismatch:lineage")
        if workflow_manifest.get("strategy_family_id") != strategy_family_id:
            warnings.append("strategy_family_id_mismatch:workflow_manifest")

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
    if isinstance(versions, list):
        version_ids: list[str] = []
        for item in versions:
            if not _lineage_entry_is_valid(item):
                warnings.append(f"lineage_entry_invalid:{_lineage_entry_label(item)}")
            if isinstance(item, dict) and item.get("status") not in _LINEAGE_STATUSES:
                warnings.append(f"lineage_status_invalid:{_lineage_entry_label(item)}")
            if isinstance(item, dict) and isinstance(item.get("version_id"), str):
                version_ids.append(item["version_id"])
            if (
                isinstance(item, dict)
                and "strategy_family_id" in item
                and isinstance(strategy_family_id, str)
                and strategy_family_id
                and item.get("strategy_family_id") != strategy_family_id
            ):
                warnings.append(
                    f"strategy_family_id_mismatch:lineage:{_lineage_entry_label(item)}"
                )
        warnings.extend(
            f"lineage_duplicate_version_id:{version_id}"
            for version_id in dict.fromkeys(version_ids)
            if version_ids.count(version_id) > 1
        )
    active_lineage = (
        [item for item in versions if isinstance(item, dict) and item.get("status") == "active"]
        if isinstance(versions, list)
        else []
    )
    if not active_lineage:
        warnings.append("lineage_active_version_missing")
    elif len(active_lineage) > 1:
        warnings.append("lineage_multiple_active_versions")

    if isinstance(active_version, str) and active_version and _WORKSPACE_VERSION_RE.fullmatch(active_version):
        if len(active_lineage) == 1 and active_lineage[0].get("version_id") != active_version:
            warnings.append("lineage_active_version_mismatch:current")
        try:
            version_dir = (
                _resolve_active_version_dir(Path.cwd(), versions_dir, active_version)
                if versions_dir is not None
                else None
            )
        except click.ClickException:
            version_dir = None
            warnings.append("active_version_dir_unsafe")
        if version_dir is None:
            return warnings
        if not version_dir.exists():
            warnings.append("active_version_dir_missing")
        if versions and isinstance(versions, list) and not any(
            isinstance(item, dict) and item.get("version_id") == active_version for item in versions
        ):
            warnings.append("active_version_not_in_lineage")
        if version_dir.exists():
            phase_state, phase_state_status = _read_json_object_state(version_dir / "phase_state.json")
            version_manifest, version_manifest_status = _read_json_object_state(
                version_dir / "version_manifest.json"
            )
            if phase_state_status == "ok" and not _active_phase_state_is_valid(phase_state, active_version):
                phase_state = {}
                phase_state_status = "invalid"
            if phase_state_status != "ok":
                warnings.append(f"active_phase_state_{phase_state_status}")
            version_manifest_schema_valid = (
                version_manifest_status == "ok"
                and _active_version_manifest_is_valid(version_manifest, active_version)
            )
            if not version_manifest_schema_valid:
                warnings.append(
                    "active_version_manifest_"
                    + (
                        version_manifest_status
                        if version_manifest_status != "ok"
                        else "invalid"
                    )
                )
            if (
                isinstance(strategy_family_id, str)
                and strategy_family_id
                and version_manifest.get("strategy_family_id") != strategy_family_id
            ):
                warnings.append("strategy_family_id_mismatch:version_manifest")
            matching_active_lineage = [
                item
                for item in active_lineage
                if item.get("version_id") == active_version
            ]
            if (
                len(matching_active_lineage) == 1
                and version_manifest_schema_valid
                and not _lineage_identity_matches(matching_active_lineage[0], version_manifest)
            ):
                warnings.append("active_lineage_identity_mismatch:version_manifest")
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
            resolved_phase_paths: dict[str, Path] = {}
            if version_manifest_status == "ok":
                phase_paths = version_manifest.get("phase_paths")
                for phase in VERSION_PHASE_DIRS:
                    raw_phase_path = (
                        phase_paths.get(phase)
                        if isinstance(phase_paths, dict) and phase in phase_paths
                        else None
                    )
                    try:
                        resolved_phase_paths[phase] = _resolve_version_phase_path(
                            Path.cwd(),
                            version_dir,
                            phase,
                            raw_phase_path,
                        )
                    except click.ClickException:
                        warnings.append(f"phase_path_unsafe:{phase}")
            report_review_passed = False
            reports_dir = resolved_phase_paths.get("10_reports")
            if reports_dir is not None:
                report_review_passed, report_path_unsafe = _scan_passing_report_reviews(reports_dir)
                if report_path_unsafe:
                    warnings.append("phase_path_unsafe:10_reports")
            if report_review_passed:
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


def _is_version_governed_workspace(config: dict[str, Any]) -> bool:
    governed, _error = _classify_workspace_governance(config)
    return governed


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


def _scan_passing_report_reviews(reports_dir: Path) -> tuple[bool, bool]:
    if not reports_dir.exists():
        return False, False
    resolved_reports_dir = reports_dir.resolve(strict=False)
    passing_review = False
    unsafe_path = False
    for review_path in reports_dir.glob("*/report_review.json"):
        run_path = review_path.parent
        if (
            run_path.is_symlink()
            or review_path.is_symlink()
            or not _is_canonically_within(run_path, resolved_reports_dir)
            or not _is_canonically_within(review_path, resolved_reports_dir)
        ):
            unsafe_path = True
            continue
        review = _read_json_object(review_path)
        if review.get("status") == "pass":
            passing_review = True
    return passing_review, unsafe_path


def _is_canonically_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_object_state(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "invalid"
    if not isinstance(payload, dict) or not payload:
        return {}, "invalid"
    return payload, "ok"


def _schema_version_is_current(payload: dict[str, Any]) -> bool:
    return type(payload.get("schema_version")) is int and payload["schema_version"] == 1


def _strategy_family_id_is_valid(payload: dict[str, Any]) -> bool:
    strategy_family_id = payload.get("strategy_family_id")
    return isinstance(strategy_family_id, str) and bool(strategy_family_id)


def _current_manifest_is_valid(payload: dict[str, Any]) -> bool:
    active_version = payload.get("active_version")
    return (
        _schema_version_is_current(payload)
        and _strategy_family_id_is_valid(payload)
        and isinstance(active_version, str)
        and bool(_WORKSPACE_VERSION_RE.fullmatch(active_version))
        and payload.get("active_phase") in VERSION_PHASE_DIRS
        and isinstance(payload.get("active_run"), str)
    )


def _lineage_manifest_root_is_valid(payload: dict[str, Any]) -> bool:
    versions = payload.get("versions")
    return (
        _schema_version_is_current(payload)
        and _strategy_family_id_is_valid(payload)
        and isinstance(versions, list)
        and bool(versions)
    )


def _active_version_manifest_is_valid(
    payload: dict[str, Any],
    active_version: str,
) -> bool:
    phase_paths = payload.get("phase_paths")
    return (
        _schema_version_is_current(payload)
        and _strategy_family_id_is_valid(payload)
        and payload.get("version_id") == active_version
        and _lineage_parent_version_id_is_valid(payload)
        and isinstance(payload.get("created_reason"), str)
        and bool(payload["created_reason"])
        and payload.get("status") == "active"
        and payload.get("active_phase") in VERSION_PHASE_DIRS
        and isinstance(payload.get("source_conversation"), str)
        and isinstance(phase_paths, dict)
        and all(phase in phase_paths for phase in VERSION_PHASE_DIRS)
    )


def _lineage_entry_is_valid(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    version_id = item.get("version_id")
    created_reason = item.get("created_reason")
    return (
        isinstance(version_id, str)
        and bool(_WORKSPACE_VERSION_RE.fullmatch(version_id))
        and _lineage_parent_version_id_is_valid(item)
        and isinstance(created_reason, str)
        and bool(created_reason)
        and item.get("status") in _LINEAGE_STATUSES
    )


def _lineage_entry_label(item: object) -> str:
    if not isinstance(item, dict):
        return "unknown"
    version_id = item.get("version_id")
    return version_id if isinstance(version_id, str) and version_id else "unknown"


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
