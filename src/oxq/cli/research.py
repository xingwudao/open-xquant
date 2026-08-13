"""Research workspace initialization commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import click
import yaml

from oxq.cli.agent import read_recovered_agent_profile
from oxq.cli.agent_manifest import (
    MarkerBlockError,
    read_yaml_file,
    remove_marker_block,
    upsert_marker_block,
    validate_marker_block,
    write_text_file,
)
from oxq.cli.sdk_bundle import install_workspace_sdk
from oxq.core.workspace_config import load_workspace_config
from oxq.process_lock import ProcessFileLock, stable_path_location_identity, verified_user_runtime_root

AGENT_PROFILE_MULTI = "multi-agent"
AGENT_PROFILE_STANDALONE = "standalone-agent"
_WORKSPACE_VERSION_RE = re.compile(r"^v[0-9][A-Za-z0-9_-]*$")
_LINEAGE_STATUSES = frozenset({"active", "superseded"})
_GOVERNANCE_TRANSACTION_FILENAME = "governance-transaction.json"
_GOVERNANCE_TRANSACTION_SCHEMA_VERSION = 2
_GOVERNANCE_ENTRY_PROGRESS = frozenset({"pending", "staged", "backed_up", "installed"})
_GOVERNANCE_ENTRY_EVIDENCE_FIELDS = frozenset(
    {
        "progress",
        "replacement_identity",
        "replacement_sha256",
        "original_identity",
        "original_sha256",
    }
)
_WORKSPACE_PATH_DEFAULTS = {
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
_WORKSPACE_PATH_KEYS = frozenset(_WORKSPACE_PATH_DEFAULTS)

WORKSPACE_BLOCK = """This is an open-xquant research workspace.

For quant strategy, factor, backtest, audit, robustness, report, chart asset,
SDK, and live trading tasks, use the installed `open-xquant` skill first.
Do not run `oxq`, SDK code, scripts, or write report files until that router
skill has selected the specific open-xquant skill for the task.

Use `.open-xquant/workspace.yaml` for local paths.

## Version-Governed Artifact Contract

This workspace uses version-governed research artifacts. Read `current.json`
before writing any research artifact. Resolve `version_root` from
`.open-xquant/workspace.yaml` `paths.versions_dir` (`__VERSION_ROOT__` for this
workspace), then resolve the active version from `current.json.active_version`.
Read `<version_root>/<active_version>/version_manifest.json` and use its exact
`phase_paths` entries. Do not assume that the active version or phase paths are
the initialization defaults.

Write phase artifacts only under the active version:

- `<phase_paths.01_brainstorm>/strategy_idea_brief.json`
- `<phase_paths.02_idea_audit>/strategy_idea_audit.json`
- `<phase_paths.04_spec_build>/strategy_spec.yaml`
- `<phase_paths.06_spec_audit>/spec_confirmation_table.md`
- `<phase_paths.07_compile_preview>/compiled_plan.json`
- `<phase_paths.08_runtime_audit>/runtime_audit.json`
- `<phase_paths.09_backtests>/<run_id>/strategy_spec.yaml`
- `<phase_paths.10_reports>/<run_id>/research_report.md`

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
    with ProcessFileLock(_workspace_init_transition_lock_path(cwd)):
        cwd.mkdir(parents=True, exist_ok=True)
        cwd = cwd.resolve()
        with ProcessFileLock(_workspace_init_lock_path(cwd)):
            _initialize_workspace_locked(
                cwd,
                name=name,
                data_dir=data_dir,
                minimal=minimal,
                force=force,
                sdk=sdk,
                sdk_venv=sdk_venv,
            )


def _initialize_workspace_locked(
    cwd: Path,
    *,
    name: str | None,
    data_dir: str,
    minimal: bool,
    force: bool,
    sdk: bool,
    sdk_venv: str,
) -> None:
    agent_profile = _installed_agent_profile()
    _require_safe_workspace_config_directory(cwd)
    _recover_governance_transaction(cwd)
    _preflight_workspace_instruction_markers(cwd)
    config_dir = cwd / ".open-xquant"
    workspace_file = config_dir / "workspace.yaml"
    sdk_state = None
    workspace_config: dict[str, object] | None = None
    created_workspace_config = False
    if (workspace_file.exists() or workspace_file.is_symlink()) and not force:
        click.echo("open-xquant workspace already initialized")
        workspace_config = load_workspace_config(workspace_file, allow_empty=True)
    else:
        workspace_config = _workspace_payload(cwd, name, data_dir)
        created_workspace_config = True

    _validate_existing_governance_manifests(cwd, workspace_config)

    if sdk:
        sdk_state = install_workspace_sdk(cwd, _resolve_sdk_venv(cwd, sdk_venv))
        workspace_config["sdk"] = sdk_state

    if created_workspace_config:
        config_dir.mkdir(parents=True, exist_ok=True)
        _write_workspace_config(cwd, workspace_file, workspace_config)
        click.echo(f"Workspace config written to {workspace_file}")
    elif sdk_state is not None:
        _write_workspace_config(cwd, workspace_file, workspace_config)
        click.echo(f"SDK config written to {workspace_file}")

    workspace_config = workspace_config or {}
    migration_files = _hidden_root_manifest_migration_files(cwd, workspace_config)
    normalized_config = _normalize_root_manifest_paths(workspace_config)
    normalized_config = _normalize_default_output_dir(cwd, workspace_config) or normalized_config
    if migration_files:
        workflow_path = cwd / "workflow_manifest.json"
        if workflow_path in migration_files:
            workflow_payload = json.loads(migration_files[workflow_path])
            assert isinstance(workflow_payload, dict)
            workflow_payload["paths"] = _effective_workspace_path_map(workspace_config)
            migration_files[workflow_path] = json.dumps(workflow_payload, indent=2, ensure_ascii=False) + "\n"
        migration_files[workspace_file] = yaml.safe_dump(
            workspace_config,
            sort_keys=False,
            width=1000,
        )
        _write_governance_files_atomically(cwd, migration_files)
        click.echo("Workspace root manifests migrated from .open-xquant/")
    elif normalized_config:
        _write_workspace_config(cwd, workspace_file, workspace_config)
        click.echo(f"Workspace config normalized in {workspace_file}")
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
    if agent_profile == AGENT_PROFILE_STANDALONE:
        remove_marker_block(cwd / "AGENTS.md", "open-xquant-subagents")
    else:
        upsert_marker_block(cwd / "AGENTS.md", "open-xquant-subagents", SUBAGENT_POLICY_BLOCK)


def _workspace_init_lock_path(cwd: Path) -> Path:
    identity = hashlib.sha256(stable_path_location_identity(cwd).encode("utf-8")).hexdigest()[:24]
    return verified_user_runtime_root() / "research" / f"{identity}.lock"


def _workspace_init_transition_lock_path(cwd: Path) -> Path:
    """Return a location lock that is unchanged when the workspace appears."""

    candidate = Path(os.path.abspath(cwd.expanduser()))
    parent_identity = stable_path_location_identity(candidate.parent)
    normalized_name = unicodedata.normalize(
        "NFC",
        unicodedata.normalize("NFKC", candidate.name).casefold(),
    )
    digest = hashlib.sha256()
    for component in (parent_identity, normalized_name):
        encoded = component.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    identity = digest.hexdigest()[:24]
    return verified_user_runtime_root() / "research" / f"transition-{identity}.lock"


def _workspace_config_directory_is_link(cwd: Path) -> bool:
    try:
        status = (cwd / ".open-xquant").lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(status.st_mode) or _is_windows_reparse_point(status)


def _require_safe_workspace_config_directory(cwd: Path) -> None:
    if _workspace_config_directory_is_link(cwd):
        raise click.ClickException("workspace configuration directory must not be a symlink or reparse point")


def _write_workspace_config(cwd: Path, path: Path, payload: dict[str, object]) -> None:
    temporary_name = f".{path.name}.tmp-{uuid4().hex}"
    try:
        parent = _GovernanceMutationParent(cwd, path.parent)
    except OSError as exc:
        raise click.ClickException(
            "workspace configuration directory contains a symlink or reparse point, or changed after validation"
        ) from exc
    try:
        parent.write_text_file(temporary_name, yaml.safe_dump(payload, sort_keys=False, width=1000))
        parent.replace_file(temporary_name, path.name)
    finally:
        try:
            parent.unlink_file(temporary_name)
        finally:
            parent.close()


def _preflight_workspace_instruction_markers(cwd: Path) -> None:
    try:
        for filename in ("AGENTS.md", "CLAUDE.md"):
            for marker in ("open-xquant-workspace", "open-xquant-subagents"):
                validate_marker_block(cwd / filename, marker)
    except MarkerBlockError as exc:
        raise click.ClickException(str(exc)) from exc


def _workspace_payload(cwd: Path, name: str | None, data_dir: str, *, sdk_state: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": name or cwd.name,
        "paths": dict(_WORKSPACE_PATH_DEFAULTS),
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
    paths = _effective_workspace_path_map(workspace)
    current_path = _configured_root_manifest_path(cwd, workspace, "current_manifest", "current.json")
    current_payload = _read_json_object(current_path)
    active_version = current_payload.get("active_version")
    active_phase = current_payload.get("active_phase")
    if not isinstance(active_phase, str) or not active_phase:
        active_phase = "01_brainstorm"
    if active_version:
        if not isinstance(active_version, str) or not _WORKSPACE_VERSION_RE.fullmatch(active_version):
            raise click.ClickException(
                f"workspace current.json active_version is unsafe: {active_version}; repair current.json before running research init"
            )
        version_id = active_version
        current_family_id = current_payload.get("strategy_family_id")
        strategy_family_id = current_family_id if isinstance(current_family_id, str) and current_family_id else name
    else:
        version_id = "v001"
        strategy_family_id = name
    versions_dir = _configured_path(cwd, workspace, "versions_dir") or (cwd / "versions")
    version_dir = _resolve_active_version_dir(cwd, versions_dir, version_id)
    version_dir_display = _display_workspace_path(cwd, version_dir)
    updates: dict[Path, dict[str, object]] = {}

    workflow_path = _configured_root_manifest_path(cwd, workspace, "workflow_manifest", "workflow_manifest.json")
    workflow_payload = _read_json_object(workflow_path)
    expected_workflow = (
        {
            **workflow_payload,
            "strategy_family_id": strategy_family_id,
            "paths": paths,
        }
        if workflow_payload
        else {
            "schema_version": 1,
            "layout": "version_governed",
            "strategy_family_id": strategy_family_id,
            "paths": paths,
        }
    )
    if workflow_payload != expected_workflow:
        updates[workflow_path] = expected_workflow

    expected_current = (
        {
            **current_payload,
            "strategy_family_id": strategy_family_id,
            "active_phase": active_phase,
        }
        if active_version
        else {
            "schema_version": 1,
            "strategy_family_id": strategy_family_id,
            "active_version": version_id,
            "active_phase": active_phase,
            "active_run": "",
        }
    )

    lineage_path = _configured_root_manifest_path(cwd, workspace, "lineage_manifest", "lineage.json")
    lineage_payload = _read_json_object(lineage_path)
    versions = lineage_payload.get("versions")
    if isinstance(versions, list):
        lineage_versions = list(versions)
    else:
        lineage_versions = []
    matching_lineage = [item for item in lineage_versions if isinstance(item, dict) and item.get("version_id") == version_id]
    if matching_lineage:
        lineage_versions = [
            {
                **item,
                "status": "active" if item.get("version_id") == version_id else "superseded",
            }
            if isinstance(item, dict) and (item.get("version_id") == version_id or item.get("status") == "active")
            else item
            for item in lineage_versions
        ]
    else:
        if active_version:
            version_manifest_payload = _read_json_object(version_dir / "version_manifest.json")
            identity = _lineage_identity_from_version_manifest(version_manifest_payload, version_id)
            lineage_versions = [
                {**item, "status": "superseded"} if isinstance(item, dict) and item.get("status") == "active" else item
                for item in lineage_versions
            ]
            lineage_versions.append(identity)
        else:
            lineage_versions.append(
                {
                    "version_id": version_id,
                    "parent_version_id": "",
                    "created_reason": "initial_strategy_version",
                    "status": "active",
                }
            )
    lineage_versions = [
        {
            **item,
            "strategy_family_id": strategy_family_id,
        }
        if isinstance(item, dict) and "strategy_family_id" in item
        else item
        for item in lineage_versions
    ]
    expected_lineage = {
        **lineage_payload,
        "schema_version": lineage_payload.get("schema_version", 1),
        "strategy_family_id": strategy_family_id,
        "versions": lineage_versions,
    }
    if lineage_payload != expected_lineage:
        updates[lineage_path] = expected_lineage

    version_manifest = version_dir / "version_manifest.json"
    if version_manifest.exists():
        version_manifest_payload = _read_json_object(version_manifest)
        expected_phase_paths = _effective_version_phase_paths(
            cwd,
            version_dir,
            version_dir_display,
            version_manifest_payload,
        )
        _create_version_phase_dirs(cwd, version_dir, expected_phase_paths)
        repaired_manifest = {
            **version_manifest_payload,
            "schema_version": version_manifest_payload.get("schema_version", 1),
            "version_id": version_id,
            "strategy_family_id": strategy_family_id,
            "active_phase": active_phase,
            "phase_paths": expected_phase_paths,
        }
        if version_manifest_payload != repaired_manifest:
            updates[version_manifest] = repaired_manifest
    else:
        expected_phase_paths = _effective_version_phase_paths(
            cwd,
            version_dir,
            version_dir_display,
            {},
        )
        _create_version_phase_dirs(cwd, version_dir, expected_phase_paths)
        manifest_identity = {
            "version_id": version_id,
            "parent_version_id": "",
            "created_reason": "initial_strategy_version",
            "status": "active",
        }
        if matching_lineage:
            lineage_identity = matching_lineage[0]
            if (
                lineage_identity.get("status") == "active"
                and _lineage_parent_version_id_is_valid(lineage_identity)
                and isinstance(lineage_identity.get("created_reason"), str)
                and lineage_identity.get("created_reason")
            ):
                manifest_identity = {key: lineage_identity[key] for key in ("version_id", "parent_version_id", "created_reason", "status")}
        updates[version_manifest] = {
            "schema_version": 1,
            "strategy_family_id": strategy_family_id,
            **manifest_identity,
            "active_phase": active_phase,
            "source_conversation": "",
            "phase_paths": expected_phase_paths,
        }

    phase_state = version_dir / "phase_state.json"
    if phase_state.exists():
        phase_state_payload = _read_json_object(phase_state)
        repaired_phase_state = {
            **phase_state_payload,
            "current_phase": active_phase,
        }
        if phase_state_payload != repaired_phase_state:
            updates[phase_state] = repaired_phase_state
    else:
        updates[phase_state] = {
            "schema_version": 1,
            "version_id": version_id,
            "current_phase": active_phase,
            "status": "active",
            "completed_phases": [],
            "blocked_phase": "",
        }
    if current_payload != expected_current:
        updates[current_path] = expected_current
    _write_governance_payloads_atomically(cwd, updates)


def _write_governance_payloads_atomically(
    cwd: Path,
    payloads: dict[Path, dict[str, object]],
) -> None:
    _write_governance_files_atomically(
        cwd,
        {destination: json.dumps(payload, indent=2, ensure_ascii=False) + "\n" for destination, payload in payloads.items()},
    )


def _write_governance_files_atomically(
    cwd: Path,
    files: dict[Path, str],
) -> None:
    if not files:
        return
    transaction_id = uuid4().hex
    destinations: list[tuple[Path, Path]] = []
    for destination in files:
        destination_relative = _validate_governance_destination(cwd, destination)
        stage, backup = _governance_transaction_artifacts(
            destination,
            transaction_id,
        )
        _validate_workspace_transaction_path(cwd, stage, "stage")
        _validate_workspace_transaction_path(cwd, backup, "backup")
        destinations.append((destination, destination_relative))
    recovery_safe = True
    try:
        with ExitStack() as stack:
            parent_handles: dict[Path, _GovernanceMutationParent] = {}
            for parent in {
                _governance_transaction_path(cwd).parent,
                *(destination.parent for destination, _relative in destinations),
            }:
                try:
                    handle = _GovernanceMutationParent(cwd, parent)
                except OSError as exc:
                    recovery_safe = False
                    raise click.ClickException(
                        "governance transaction parent contains a symlink or reparse point, or changed after validation; journal preserved"
                    ) from exc
                stack.callback(handle.close)
                parent_handles[parent] = handle
            token = _ACTIVE_GOVERNANCE_PARENTS.set(parent_handles)
            stack.callback(_ACTIVE_GOVERNANCE_PARENTS.reset, token)
            entries: list[dict[str, object]] = []
            for destination, destination_relative in destinations:
                parent = parent_handles[destination.parent]
                original_evidence = parent.file_evidence(destination.name)
                replacement_sha256 = hashlib.sha256(files[destination].encode("utf-8")).hexdigest()
                entries.append(
                    {
                        "destination": destination_relative.as_posix(),
                        "had_original": original_evidence is not None,
                        "parent_identity": parent.identity_payload(),
                        "progress": "pending",
                        "replacement_identity": None,
                        "replacement_sha256": replacement_sha256,
                        "original_identity": original_evidence[0] if original_evidence is not None else None,
                        "original_sha256": original_evidence[1] if original_evidence is not None else None,
                    }
                )
            journal = {
                "schema_version": _GOVERNANCE_TRANSACTION_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "state": "prepared",
                "journal_parent_identity": parent_handles[_governance_transaction_path(cwd).parent].identity_payload(),
                "entries": entries,
            }
            _write_governance_transaction_journal(cwd, journal)
            try:
                for entry, (destination, _destination_relative) in zip(entries, destinations, strict=True):
                    stage, _backup = _governance_transaction_artifacts(
                        destination,
                        transaction_id,
                    )
                    parent = parent_handles[destination.parent]
                    parent.write_text_file(
                        stage.name,
                        files[destination],
                    )
                    replacement_evidence = parent.file_evidence(stage.name)
                    if replacement_evidence is None or replacement_evidence[1] != entry["replacement_sha256"]:
                        raise click.ClickException(
                            f"governance transaction stage content changed while being prepared; journal preserved: {stage}"
                        )
                    entry["replacement_identity"] = replacement_evidence[0]
                    entry["progress"] = "staged"
                    _write_governance_transaction_journal(cwd, journal)
                for entry in entries:
                    destination = cwd / str(entry["destination"])
                    stage, backup = _governance_transaction_artifacts(
                        destination,
                        transaction_id,
                    )
                    parent = parent_handles[destination.parent]
                    if entry["had_original"]:
                        if parent.exists(backup.name):
                            raise click.ClickException(
                                f"governance transaction backup has unrecognized content; journal preserved: {backup}"
                            )
                        parent.assert_file_evidence(
                            destination.name,
                            entry["original_identity"],
                            entry["original_sha256"],
                        )
                        parent.replace_file(destination.name, backup.name)
                        parent.assert_file_evidence(
                            backup.name,
                            entry["original_identity"],
                            entry["original_sha256"],
                        )
                        entry["progress"] = "backed_up"
                        _write_governance_transaction_journal(cwd, journal)
                    elif parent.exists(destination.name):
                        raise click.ClickException(
                            f"governance transaction destination has unrecognized content; journal preserved: {destination}"
                        )
                    parent.assert_file_evidence(
                        stage.name,
                        entry["replacement_identity"],
                        entry["replacement_sha256"],
                    )
                    parent.replace_file(stage.name, destination.name)
                    parent.assert_file_evidence(
                        destination.name,
                        entry["replacement_identity"],
                        entry["replacement_sha256"],
                    )
                    entry["progress"] = "installed"
                    _write_governance_transaction_journal(cwd, journal)
                journal["state"] = "committed"
                _write_governance_transaction_journal(cwd, journal)
                for parent in parent_handles.values():
                    parent.assert_unchanged()
            except BaseException:
                try:
                    for parent in parent_handles.values():
                        parent.assert_unchanged()
                except BaseException:
                    recovery_safe = False
                    raise
                raise
    except BaseException:
        if recovery_safe:
            _recover_governance_transaction(cwd)
        raise
    _recover_governance_transaction(cwd)


def _governance_transaction_path(cwd: Path) -> Path:
    return cwd / ".open-xquant" / _GOVERNANCE_TRANSACTION_FILENAME


def _governance_transaction_artifacts(
    destination: Path,
    transaction_id: str,
) -> tuple[Path, Path]:
    return (
        destination.parent / f".{destination.name}.stage-{transaction_id}",
        destination.parent / f".{destination.name}.backup-{transaction_id}",
    )


def _write_governance_transaction_journal(
    cwd: Path,
    journal: dict[str, object],
) -> None:
    journal_path = _governance_transaction_path(cwd)
    _validate_workspace_transaction_path(cwd, journal_path, "journal")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    transaction_id = str(journal["transaction_id"])
    temporary = journal_path.parent / f".{journal_path.name}.tmp-{transaction_id}"
    _validate_workspace_transaction_path(cwd, temporary, "journal temporary")
    active_parents = _ACTIVE_GOVERNANCE_PARENTS.get()
    active_parent = active_parents.get(journal_path.parent) if active_parents is not None else None
    try:
        content = json.dumps(journal, indent=2, ensure_ascii=False) + "\n"
        if active_parent is not None:
            active_parent.write_text_file(temporary.name, content)
            active_parent.replace_file(temporary.name, journal_path.name)
        else:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, journal_path)
            _fsync_directory(journal_path.parent)
    finally:
        if active_parent is not None:
            active_parent.unlink_file(temporary.name)
        elif temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _recover_governance_transaction(cwd: Path) -> None:
    journal_path = _governance_transaction_path(cwd)
    _validate_workspace_transaction_path(cwd, journal_path, "journal")
    if not (journal_path.exists() or journal_path.is_symlink()):
        return
    if journal_path.is_symlink():
        raise click.ClickException("governance transaction journal must not be a symlink")
    if not journal_path.is_file():
        raise click.ClickException("governance transaction journal must be a regular file")
    journal_identity = _transaction_file_identity(journal_path.lstat())
    journal = _read_governance_transaction_journal(journal_path)
    if _transaction_file_identity(journal_path.lstat()) != journal_identity:
        raise click.ClickException("governance transaction journal changed while being read; manual recovery is required")
    transaction_id = str(journal["transaction_id"])
    state = str(journal["state"])
    entries = journal["entries"]
    assert isinstance(entries, list)
    allowed_destinations = _governance_recovery_destination_allowlist(
        cwd,
        transaction_id,
    )
    resolved_entries: list[_GovernanceRecoveryEntry] = []
    canonical_destinations: set[Path] = set()
    for entry in entries:
        assert isinstance(entry, dict)
        destination = cwd / str(entry["destination"])
        destination_relative = _validate_governance_destination(
            cwd,
            destination,
            allowed_destinations=allowed_destinations,
        )
        destination = cwd / destination_relative
        canonical_destination = destination.resolve(strict=False)
        if canonical_destination in canonical_destinations:
            raise click.ClickException(
                "governance transaction journal contains duplicate normalized destinations; manual recovery is required"
            )
        canonical_destinations.add(canonical_destination)
        stage, backup = _governance_transaction_artifacts(
            destination,
            transaction_id,
        )
        _validate_workspace_transaction_path(cwd, stage, "stage")
        _validate_workspace_transaction_path(cwd, backup, "backup")
        has_evidence = _GOVERNANCE_ENTRY_EVIDENCE_FIELDS.issubset(entry)
        resolved_entries.append(
            _GovernanceRecoveryEntry(
                destination=destination,
                stage=stage,
                backup=backup,
                had_original=bool(entry["had_original"]),
                parent_identity=entry["parent_identity"],
                progress=str(entry["progress"]) if has_evidence else None,
                replacement_identity=entry["replacement_identity"] if has_evidence else None,
                replacement_sha256=str(entry["replacement_sha256"]) if has_evidence else None,
                original_identity=entry["original_identity"] if has_evidence else None,
                original_sha256=str(entry["original_sha256"]) if has_evidence and entry["original_sha256"] is not None else None,
            )
        )

    with ExitStack() as stack:
        parent_handles: dict[Path, _GovernanceMutationParent] = {}
        for parent in {
            journal_path.parent,
            *(entry.destination.parent for entry in resolved_entries),
        }:
            try:
                handle = _GovernanceMutationParent(cwd, parent)
            except OSError as exc:
                raise click.ClickException(
                    "governance transaction parent contains a symlink or reparse point, or changed after validation; journal preserved"
                ) from exc
            stack.callback(handle.close)
            parent_handles[parent] = handle

        journal_parent = parent_handles[journal_path.parent]
        journal_parent.assert_identity_payload(journal["journal_parent_identity"])
        journal_evidence = journal_parent.file_evidence(journal_path.name)
        if (
            journal_evidence is None
            or _governance_identity_tuple(journal_evidence[0]) != journal_identity
        ):
            raise click.ClickException(
                "governance transaction journal changed during recovery; "
                "journal preserved"
            )
        for entry in resolved_entries:
            parent_handles[entry.destination.parent].assert_identity_payload(entry.parent_identity)
            _preflight_governance_recovery_entry(
                parent_handles[entry.destination.parent],
                entry,
                state,
            )

        if state == "prepared":
            for entry in reversed(resolved_entries):
                parent = parent_handles[entry.destination.parent]
                if entry.has_evidence:
                    _rollback_evidenced_governance_entry(parent, entry)
                elif entry.had_original:
                    if parent.exists(entry.backup.name):
                        parent.unlink_file(entry.destination.name)
                        parent.replace_file(entry.backup.name, entry.destination.name)

        for entry in resolved_entries:
            parent = parent_handles[entry.destination.parent]
            if entry.has_evidence:
                _cleanup_evidenced_governance_artifacts(parent, entry)
            else:
                parent.unlink_file(entry.stage.name)
                parent.unlink_file(entry.backup.name)
        journal_parent.unlink_file(
            journal_path.name,
            expected_identity=journal_identity,
            expected_sha256=journal_evidence[1],
        )


def _read_governance_transaction_journal(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise click.ClickException("governance transaction journal is invalid; manual recovery is required") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("governance transaction journal is invalid; manual recovery is required")
    transaction_id = payload.get("transaction_id")
    state = payload.get("state")
    entries = payload.get("entries")
    valid_entries = isinstance(entries, list) and bool(entries)
    normalized_destinations: set[Path] = set()
    if valid_entries:
        for entry in entries:
            if not isinstance(entry, dict):
                valid_entries = False
                break
            destination = entry.get("destination")
            if (
                not isinstance(destination, str)
                or not destination
                or Path(destination).is_absolute()
                or ".." in Path(destination).parts
                or type(entry.get("had_original")) is not bool
            ):
                valid_entries = False
                break
            try:
                entry["parent_identity"] = _validated_governance_identity(entry.get("parent_identity"))
            except ValueError:
                valid_entries = False
                break
            evidence_fields = _GOVERNANCE_ENTRY_EVIDENCE_FIELDS.intersection(entry)
            if evidence_fields and evidence_fields != _GOVERNANCE_ENTRY_EVIDENCE_FIELDS:
                valid_entries = False
                break
            if evidence_fields:
                try:
                    _validate_governance_entry_evidence(entry)
                except ValueError:
                    valid_entries = False
                    break
            normalized_destination = Path(destination)
            if normalized_destination in normalized_destinations:
                raise click.ClickException(
                    "governance transaction journal contains duplicate normalized destinations; manual recovery is required"
                )
            normalized_destinations.add(normalized_destination)
    if (
        payload.get("schema_version") != _GOVERNANCE_TRANSACTION_SCHEMA_VERSION
        or not isinstance(transaction_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", transaction_id)
        or state not in {"prepared", "committed"}
        or not valid_entries
    ):
        raise click.ClickException("governance transaction journal is invalid; manual recovery is required")
    try:
        payload["journal_parent_identity"] = _validated_governance_identity(payload.get("journal_parent_identity"))
    except ValueError as exc:
        raise click.ClickException("governance transaction journal is invalid; manual recovery is required") from exc
    return payload


@dataclass(frozen=True)
class _GovernanceRecoveryEntry:
    destination: Path
    stage: Path
    backup: Path
    had_original: bool
    parent_identity: dict[str, int]
    progress: str | None
    replacement_identity: dict[str, int] | None
    replacement_sha256: str | None
    original_identity: dict[str, int] | None
    original_sha256: str | None

    @property
    def has_evidence(self) -> bool:
        return self.progress is not None


def _validate_governance_entry_evidence(entry: dict[str, object]) -> None:
    progress = entry.get("progress")
    replacement_identity = entry.get("replacement_identity")
    replacement_sha256 = entry.get("replacement_sha256")
    original_identity = entry.get("original_identity")
    original_sha256 = entry.get("original_sha256")
    had_original = bool(entry["had_original"])
    if progress not in _GOVERNANCE_ENTRY_PROGRESS:
        raise ValueError("invalid governance entry progress")
    if not isinstance(replacement_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", replacement_sha256) is None:
        raise ValueError("invalid governance replacement digest")
    if replacement_identity is not None:
        entry["replacement_identity"] = _validated_governance_identity(replacement_identity)
    if progress == "pending" and replacement_identity is not None:
        raise ValueError("pending governance entry must not have replacement identity")
    if progress != "pending" and replacement_identity is None:
        raise ValueError("advanced governance entry requires replacement identity")
    if progress == "backed_up" and not had_original:
        raise ValueError("new governance destination cannot be backed up")
    if had_original:
        entry["original_identity"] = _validated_governance_identity(original_identity)
        if not isinstance(original_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", original_sha256) is None:
            raise ValueError("invalid governance original digest")
    elif original_identity is not None or original_sha256 is not None:
        raise ValueError("new governance destination cannot have original evidence")


def _preflight_governance_recovery_entry(
    parent: _GovernanceMutationParent,
    entry: _GovernanceRecoveryEntry,
    state: str,
) -> None:
    if state == "committed" and (
        not entry.has_evidence or entry.progress != "installed"
    ):
        raise click.ClickException(
            "governance transaction committed entry progress must be installed; "
            "journal preserved for manual recovery"
        )
    if not entry.has_evidence:
        if state == "prepared" and not entry.had_original and parent.exists(entry.destination.name):
            _raise_unrecognized_governance_artifact(entry.destination, "destination")
        return

    stage_evidence = parent.file_evidence(entry.stage.name)
    backup_evidence = parent.file_evidence(entry.backup.name)
    destination_evidence = parent.file_evidence(entry.destination.name)
    if stage_evidence is not None and not _governance_evidence_matches(
        stage_evidence,
        entry.replacement_identity,
        entry.replacement_sha256,
        allow_missing_identity=True,
    ):
        _raise_unrecognized_governance_artifact(entry.stage, "stage")
    if backup_evidence is not None and not _governance_evidence_matches(
        backup_evidence,
        entry.original_identity,
        entry.original_sha256,
    ):
        _raise_unrecognized_governance_artifact(entry.backup, "backup")
    if state == "committed":
        if destination_evidence is None:
            _raise_unrecognized_governance_artifact(
                entry.destination,
                "missing destination",
            )
        if not _governance_evidence_matches(
            destination_evidence,
            entry.replacement_identity,
            entry.replacement_sha256,
        ):
            _raise_unrecognized_governance_artifact(entry.destination, "destination")
        return

    if entry.had_original:
        destination_is_original = _governance_evidence_matches(
            destination_evidence,
            entry.original_identity,
            entry.original_sha256,
        )
        destination_is_replacement = _governance_evidence_matches(
            destination_evidence,
            entry.replacement_identity,
            entry.replacement_sha256,
        )
        if destination_evidence is None:
            if backup_evidence is None:
                _raise_unrecognized_governance_artifact(entry.destination, "missing destination")
        elif destination_is_original:
            if backup_evidence is not None:
                _raise_unrecognized_governance_artifact(entry.destination, "ambiguous destination")
        elif destination_is_replacement:
            if backup_evidence is None or stage_evidence is not None:
                _raise_unrecognized_governance_artifact(entry.destination, "ambiguous destination")
        else:
            _raise_unrecognized_governance_artifact(entry.destination, "destination")
        return

    if backup_evidence is not None:
        _raise_unrecognized_governance_artifact(entry.backup, "backup")
    if destination_evidence is None:
        return
    if entry.replacement_identity is None or not _governance_evidence_matches(
        destination_evidence,
        entry.replacement_identity,
        entry.replacement_sha256,
    ):
        _raise_unrecognized_governance_artifact(entry.destination, "destination")
    if stage_evidence is not None:
        _raise_unrecognized_governance_artifact(entry.destination, "ambiguous destination")


def _rollback_evidenced_governance_entry(
    parent: _GovernanceMutationParent,
    entry: _GovernanceRecoveryEntry,
) -> None:
    if entry.had_original:
        if not parent.exists(entry.backup.name):
            return
        if parent.exists(entry.destination.name):
            parent.assert_file_evidence(
                entry.destination.name,
                entry.replacement_identity,
                entry.replacement_sha256,
            )
            parent.unlink_file(
                entry.destination.name,
                expected_identity=_governance_identity_tuple(entry.replacement_identity),
                expected_sha256=entry.replacement_sha256,
            )
        parent.assert_file_evidence(
            entry.backup.name,
            entry.original_identity,
            entry.original_sha256,
        )
        parent.replace_file(entry.backup.name, entry.destination.name)
        return
    if parent.exists(entry.destination.name):
        parent.assert_file_evidence(
            entry.destination.name,
            entry.replacement_identity,
            entry.replacement_sha256,
        )
        parent.unlink_file(
            entry.destination.name,
            expected_identity=_governance_identity_tuple(entry.replacement_identity),
            expected_sha256=entry.replacement_sha256,
        )


def _cleanup_evidenced_governance_artifacts(
    parent: _GovernanceMutationParent,
    entry: _GovernanceRecoveryEntry,
) -> None:
    if parent.exists(entry.stage.name):
        parent.assert_file_evidence(
            entry.stage.name,
            entry.replacement_identity,
            entry.replacement_sha256,
            allow_missing_identity=True,
        )
        parent.unlink_file(
            entry.stage.name,
            expected_identity=_governance_identity_tuple(entry.replacement_identity),
            expected_sha256=entry.replacement_sha256,
        )
    if parent.exists(entry.backup.name):
        parent.assert_file_evidence(
            entry.backup.name,
            entry.original_identity,
            entry.original_sha256,
        )
        parent.unlink_file(
            entry.backup.name,
            expected_identity=_governance_identity_tuple(entry.original_identity),
            expected_sha256=entry.original_sha256,
        )


def _governance_evidence_matches(
    actual: tuple[dict[str, int], str] | None,
    expected_identity: dict[str, int] | None,
    expected_sha256: str | None,
    *,
    allow_missing_identity: bool = False,
) -> bool:
    if actual is None or expected_sha256 is None or actual[1] != expected_sha256:
        return False
    if expected_identity is None:
        return allow_missing_identity
    return actual[0] == expected_identity


def _governance_identity_tuple(identity: dict[str, int] | None) -> tuple[int, int] | None:
    if identity is None:
        return None
    return identity["device"], identity["inode"]


def _raise_unrecognized_governance_artifact(path: Path, label: str) -> None:
    raise click.ClickException(f"governance transaction {label} has unrecognized content; journal preserved for manual recovery: {path}")


def _transaction_file_identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _governance_identity_payload(status: os.stat_result) -> dict[str, int]:
    return {"device": int(status.st_dev), "inode": int(status.st_ino)}


def _governance_status_snapshot(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _validated_governance_identity(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"device", "inode"}:
        raise ValueError("directory identity must contain device and inode")
    device = value.get("device")
    inode = value.get("inode")
    if (
        not isinstance(device, int)
        or isinstance(device, bool)
        or device < 0
        or not isinstance(inode, int)
        or isinstance(inode, bool)
        or inode < 0
    ):
        raise ValueError("directory identity values must be non-negative integers")
    return {"device": device, "inode": inode}


class _GovernanceMutationParent:
    def __init__(self, cwd: Path, path: Path) -> None:
        self.cwd = cwd
        self.path = path
        self.relative = path.relative_to(cwd)
        self.root_descriptor: int | None = None
        self.descriptor: int | None = None
        self.windows_handles: list[int] = []
        self._identity: tuple[int, int]
        if os.name == "nt":
            self.windows_handles = _pin_windows_directory_chain(cwd, self.relative)
            self._identity = _transaction_file_identity(os.stat(self.path, follow_symlinks=False))
        else:
            self.root_descriptor = _open_posix_directory(cwd)
            try:
                self.descriptor = _open_posix_relative_directory(
                    self.root_descriptor,
                    self.relative,
                )
            except BaseException:
                os.close(self.root_descriptor)
                self.root_descriptor = None
                raise
            self._identity = _transaction_file_identity(os.fstat(self.descriptor))

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        if self.root_descriptor is not None:
            os.close(self.root_descriptor)
            self.root_descriptor = None
        while self.windows_handles:
            _close_windows_handle(self.windows_handles.pop())

    def exists(self, name: str) -> bool:
        return self._status(name) is not None

    def file_evidence(self, name: str) -> tuple[dict[str, int], str] | None:
        status = self._status(name)
        if status is None:
            return None
        self._require_file_artifact(name, status)
        digest = hashlib.sha256()
        if self.descriptor is not None:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self.descriptor,
            )
            try:
                opened_status = os.fstat(descriptor)
                self._require_file_artifact(name, opened_status)
                if _governance_status_snapshot(opened_status) != _governance_status_snapshot(status):
                    raise click.ClickException(
                        f"governance transaction artifact changed while being inspected; journal preserved: {self.path / name}"
                    )
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                final_status = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        else:
            with (self.path / name).open("rb") as handle:
                opened_status = os.fstat(handle.fileno())
                self._require_file_artifact(name, opened_status)
                if _governance_status_snapshot(opened_status) != _governance_status_snapshot(status):
                    raise click.ClickException(
                        f"governance transaction artifact changed while being inspected; journal preserved: {self.path / name}"
                    )
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                final_status = os.fstat(handle.fileno())
        current_status = self._status(name)
        if (
            _governance_status_snapshot(final_status) != _governance_status_snapshot(opened_status)
            or current_status is None
            or _governance_status_snapshot(current_status) != _governance_status_snapshot(final_status)
        ):
            raise click.ClickException(
                f"governance transaction artifact changed while being inspected; journal preserved: {self.path / name}"
            )
        return _governance_identity_payload(final_status), digest.hexdigest()

    def assert_file_evidence(
        self,
        name: str,
        expected_identity: dict[str, int] | None,
        expected_sha256: str | None,
        *,
        allow_missing_identity: bool = False,
    ) -> None:
        actual = self.file_evidence(name)
        if not _governance_evidence_matches(
            actual,
            expected_identity,
            expected_sha256,
            allow_missing_identity=allow_missing_identity,
        ):
            _raise_unrecognized_governance_artifact(self.path / name, "artifact")

    def identity_payload(self) -> dict[str, int]:
        self._assert_parent_unchanged()
        if self.descriptor is not None:
            return _governance_identity_payload(os.fstat(self.descriptor))
        return {"device": self._identity[0], "inode": self._identity[1]}

    def assert_identity_payload(self, expected: dict[str, int]) -> None:
        if self.identity_payload() != expected:
            raise click.ClickException("governance transaction parent identity does not match journal; journal preserved")

    def write_text_file(self, name: str, content: str) -> None:
        self._assert_parent_unchanged()
        if self.descriptor is not None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(name, flags, 0o600, dir_fd=self.descriptor)
            try:
                data = memoryview(content.encode("utf-8"))
                while data:
                    written = os.write(descriptor, data)
                    data = data[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            with (self.path / name).open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        self.sync()
        self._assert_parent_unchanged()

    def unlink_file(
        self,
        name: str,
        *,
        expected_identity: tuple[int, int] | None = None,
        expected_sha256: str | None = None,
    ) -> bool:
        status = self._status(name)
        if status is None:
            return False
        self._require_file_artifact(name, status)
        if expected_identity is not None or expected_sha256 is not None:
            quarantine = f".{name}.quarantine-{uuid4().hex}"
            self.replace_file(name, quarantine)
            evidence = self.file_evidence(quarantine)
            identity_matches = (
                evidence is not None
                and (
                    expected_identity is None
                    or _governance_identity_tuple(evidence[0]) == expected_identity
                )
            )
            content_matches = (
                evidence is not None
                and (
                    expected_sha256 is None
                    or evidence[1] == expected_sha256
                )
            )
            if not identity_matches or not content_matches:
                raise click.ClickException(
                    "governance transaction artifact changed during recovery; "
                    f"journal preserved and mismatched artifact quarantined: {self.path / quarantine}"
                )
            return self.unlink_file(quarantine)
        self._assert_parent_unchanged()
        if self.descriptor is not None:
            os.unlink(name, dir_fd=self.descriptor)
        else:
            os.unlink(self.path / name)
        self.sync()
        self._assert_parent_unchanged()
        return True

    def replace_file(self, source: str, destination: str) -> None:
        source_status = self._status(source)
        if source_status is None:
            raise click.ClickException(f"governance transaction artifact disappeared during recovery: {self.path / source}")
        self._require_file_artifact(source, source_status)
        destination_status = self._status(destination)
        if destination_status is not None:
            self._require_file_artifact(destination, destination_status)
        self._assert_parent_unchanged()
        if self.descriptor is not None:
            os.replace(
                source,
                destination,
                src_dir_fd=self.descriptor,
                dst_dir_fd=self.descriptor,
            )
        else:
            os.replace(self.path / source, self.path / destination)
        self.sync()
        self._assert_parent_unchanged()

    def sync(self) -> None:
        if self.descriptor is not None:
            os.fsync(self.descriptor)

    def assert_unchanged(self) -> None:
        self._assert_parent_unchanged()

    def _status(self, name: str) -> os.stat_result | None:
        self._assert_parent_unchanged()
        try:
            if self.descriptor is not None:
                return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            status = os.lstat(self.path / name)
        except FileNotFoundError:
            return None
        if _is_windows_reparse_point(status):
            raise click.ClickException("governance transaction artifact is a Windows reparse point; journal preserved")
        return status

    def _assert_parent_unchanged(self) -> None:
        if self.descriptor is not None:
            assert self.root_descriptor is not None
            try:
                probe = _open_posix_relative_directory(
                    self.root_descriptor,
                    self.relative,
                )
            except OSError as exc:
                raise click.ClickException("governance transaction parent changed after validation; journal preserved") from exc
            try:
                expected = os.fstat(self.descriptor)
                actual = os.fstat(probe)
            finally:
                os.close(probe)
            if _transaction_file_identity(expected) != _transaction_file_identity(actual):
                raise click.ClickException("governance transaction parent changed after validation; journal preserved")
            return
        for handle in self.windows_handles:
            if _windows_handle_is_reparse_point(handle):
                raise click.ClickException("governance transaction parent became a Windows reparse point; journal preserved")
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise click.ClickException("governance transaction parent changed after validation; journal preserved") from exc
        if _transaction_file_identity(current) != self._identity:
            raise click.ClickException("governance transaction parent changed after validation; journal preserved")

    def _require_file_artifact(self, name: str, status: os.stat_result) -> None:
        if not stat.S_ISREG(status.st_mode):
            raise click.ClickException(f"governance transaction artifact is not a regular file: {self.path / name}")


_ACTIVE_GOVERNANCE_PARENTS: ContextVar[dict[Path, _GovernanceMutationParent] | None] = ContextVar("active_governance_parents", default=None)


def _open_posix_directory(path: Path) -> int:
    return os.open(path, _posix_directory_open_flags())


def _posix_directory_open_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if directory is None or no_follow is None:
        raise OSError("secure directory handles are unavailable on this POSIX platform")
    return os.O_RDONLY | directory | no_follow | getattr(os, "O_CLOEXEC", 0)


def _open_posix_relative_directory(root_descriptor: int, relative: Path) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for part in relative.parts:
            next_descriptor = os.open(
                part,
                _posix_directory_open_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _pin_windows_directory_chain(cwd: Path, relative: Path) -> list[int]:
    handles: list[int] = []
    current = cwd
    try:
        handles.append(_open_windows_directory_handle(current))
        for part in relative.parts:
            current /= part
            handles.append(_open_windows_directory_handle(current))
    except BaseException:
        while handles:
            _close_windows_handle(handles.pop())
        raise
    return handles


def _open_windows_directory_handle(path: Path) -> int:
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    file_share_read = 0x00000001
    file_share_write = 0x00000002
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    handle = create_file(
        str(path),
        0,
        file_share_read | file_share_write,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error), str(path))
    integer_handle = int(handle)
    if _windows_handle_is_reparse_point(integer_handle):
        _close_windows_handle(integer_handle)
        raise OSError(f"directory is a Windows reparse point: {path}")
    return integer_handle


def _windows_handle_is_reparse_point(handle: int) -> bool:
    import ctypes
    from ctypes import wintypes

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    get_info = ctypes.WinDLL(
        "kernel32",
        use_last_error=True,
    ).GetFileInformationByHandleEx
    get_info.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_info.restype = wintypes.BOOL
    info = FileAttributeTagInfo()
    if not get_info(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error))
    return bool(info.file_attributes & 0x00000400)


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    close_handle(handle)


def _is_windows_reparse_point(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = _open_posix_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_governance_destination(
    cwd: Path,
    destination: Path,
    *,
    allowed_destinations: set[Path] | None = None,
) -> Path:
    relative = _validate_workspace_transaction_path(
        cwd,
        destination,
        "destination",
    )
    if allowed_destinations is not None:
        if relative in allowed_destinations:
            return relative
        raise click.ClickException(f"governance transaction destination is not allowlisted: {relative.as_posix()}")
    root_destinations = {
        Path("current.json"),
        Path("lineage.json"),
        Path("workflow_manifest.json"),
        Path(".open-xquant/workspace.yaml"),
    }
    if relative in root_destinations:
        return relative

    versions_dir = _governance_versions_dir(cwd)
    try:
        versions_relative = versions_dir.relative_to(cwd)
    except ValueError as exc:
        raise click.ClickException("governance transaction destinations must stay within the workspace") from exc
    version_prefix = relative.parts[: len(versions_relative.parts)]
    suffix = relative.parts[len(versions_relative.parts) :]
    if (
        version_prefix == versions_relative.parts
        and len(suffix) == 2
        and _WORKSPACE_VERSION_RE.fullmatch(suffix[0]) is not None
        and suffix[1] in {"version_manifest.json", "phase_state.json"}
    ):
        return relative
    raise click.ClickException(f"governance transaction destination is not allowlisted: {relative.as_posix()}")


def _governance_recovery_destination_allowlist(
    cwd: Path,
    transaction_id: str,
) -> set[Path]:
    allowed = {
        Path("current.json"),
        Path("lineage.json"),
        Path("workflow_manifest.json"),
        Path(".open-xquant/workspace.yaml"),
    }
    current_path = cwd / "current.json"
    _current_stage, current_backup = _governance_transaction_artifacts(
        current_path,
        transaction_id,
    )
    active_versions: set[str] = set()
    for candidate in (current_path, current_backup):
        _validate_workspace_transaction_path(cwd, candidate, "current manifest")
        if not candidate.is_file():
            continue
        active_version = _read_json_object(candidate).get("active_version")
        if isinstance(active_version, str) and _WORKSPACE_VERSION_RE.fullmatch(active_version) is not None:
            active_versions.add(active_version)
    if not active_versions:
        active_versions.add("v001")

    versions_dir = _governance_versions_dir(cwd, transaction_id=transaction_id)
    versions_relative = versions_dir.relative_to(cwd)
    for active_version in active_versions:
        version_relative = versions_relative / active_version
        allowed.add(version_relative / "version_manifest.json")
        allowed.add(version_relative / "phase_state.json")
    return allowed


def _governance_versions_dir(
    cwd: Path,
    *,
    transaction_id: str | None = None,
) -> Path:
    workspace_path = cwd / ".open-xquant" / "workspace.yaml"
    candidates = [workspace_path]
    if transaction_id is not None:
        stage, backup = _governance_transaction_artifacts(
            workspace_path,
            transaction_id,
        )
        candidates.extend((backup, stage))
    for candidate in candidates:
        _validate_workspace_transaction_path(cwd, candidate, "workspace config")
        if candidate.is_symlink():
            raise click.ClickException("governance transaction workspace config must not be a symlink")
        if candidate.is_file():
            workspace = read_yaml_file(candidate)
            return _configured_path(cwd, workspace, "versions_dir") or cwd / "versions"
    raise click.ClickException("governance transaction destination allowlist requires workspace.yaml")


def _validate_workspace_transaction_path(
    cwd: Path,
    path: Path,
    label: str,
) -> Path:
    try:
        relative = path.relative_to(cwd)
    except ValueError as exc:
        raise click.ClickException(f"governance transaction {label} must stay within the workspace") from exc
    current = cwd
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise click.ClickException(f"governance transaction {label} must not contain symlink components")
    try:
        path.resolve(strict=False).relative_to(cwd)
    except ValueError as exc:
        raise click.ClickException(f"governance transaction {label} must stay within the workspace") from exc
    return relative


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
    if key == "versions_dir" and _path_has_symlink_component(candidate, resolved_cwd):
        raise click.ClickException("workspace paths.versions_dir must not contain symlink components")
    try:
        candidate.resolve(strict=False).relative_to(resolved_cwd)
    except ValueError as exc:
        raise click.ClickException(f"workspace paths.{key} must stay within the workspace") from exc
    return candidate


def _path_has_symlink_component(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _resolve_active_version_dir(cwd: Path, versions_dir: Path, version_id: str) -> Path:
    workspace_root = cwd.resolve()
    versions_root = versions_dir if versions_dir.is_absolute() else cwd / versions_dir
    candidate = versions_root / version_id
    if candidate.is_symlink():
        raise click.ClickException("active version directory must stay within the workspace and must not be a symlink")
    try:
        candidate.resolve(strict=False).relative_to(workspace_root)
    except ValueError as exc:
        raise click.ClickException("active version directory must stay within the workspace") from exc
    return candidate


def _resolve_version_phase_path(
    cwd: Path,
    version_dir: Path,
    phase: str,
    raw_path: object,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise click.ClickException(f"active version_manifest.json requires phase_paths.{phase}")
    phase_path = Path(raw_path)
    if phase_path.is_absolute() or ".." in phase_path.parts:
        raise click.ClickException(f"active version_manifest.json phase_paths.{phase} must be a safe relative path")
    workspace_root = cwd.resolve()
    resolved_version_dir = version_dir.resolve(strict=False)
    resolved_phase_path = (cwd / phase_path).resolve(strict=False)
    if _path_has_symlink_component(cwd / phase_path, workspace_root):
        raise click.ClickException(f"active version_manifest.json phase_paths.{phase} must not contain symlink components")
    try:
        resolved_version_dir.relative_to(workspace_root)
        resolved_phase_path.relative_to(resolved_version_dir)
    except ValueError as exc:
        raise click.ClickException(f"active version_manifest.json phase_paths.{phase} must stay within the active version") from exc
    return cwd / phase_path


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
    return WORKSPACE_BLOCK.replace("__VERSION_ROOT__", version_root)


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


def _hidden_root_manifest_migration_files(
    cwd: Path,
    workspace: dict[str, object],
) -> dict[Path, str]:
    if not _is_version_governed_workspace(workspace):
        return {}
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        return {}
    files: dict[Path, str] = {}
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
            files[target] = source.read_text(encoding="utf-8")
    return files


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


def _classify_workspace_governance(workspace: dict[str, object]) -> tuple[bool, str | None]:
    workflow = workspace.get("workflow")
    paths = workspace.get("paths")
    versions_dir_configured = isinstance(paths, dict) and "versions_dir" in paths
    governed = (isinstance(workflow, dict) and workflow.get("layout") == "version_governed") or versions_dir_configured
    if versions_dir_configured and (not isinstance(paths["versions_dir"], str) or not paths["versions_dir"]):
        return True, "workspace paths.versions_dir must be a non-empty string"
    return governed, None


def _is_version_governed_workspace(workspace: dict[str, object]) -> bool:
    governed, error = _classify_workspace_governance(workspace)
    if error is not None:
        raise click.ClickException(error)
    return governed


def _workflow_manifest_path_mismatches(
    workspace: dict[str, object],
    workflow_manifest: dict[str, object],
) -> list[str]:
    persisted_paths = workflow_manifest.get("paths")
    if not isinstance(persisted_paths, dict):
        return []
    effective_paths = _effective_workspace_path_map(workspace)
    return sorted(key for key in effective_paths.keys() | persisted_paths.keys() if effective_paths.get(key) != persisted_paths.get(key))


def _effective_workspace_path_map(
    workspace: dict[str, object],
) -> dict[str, str]:
    configured_paths = workspace.get("paths")
    effective = dict(_WORKSPACE_PATH_DEFAULTS)
    if isinstance(configured_paths, dict):
        effective.update(
            {key: value for key, value in configured_paths.items() if isinstance(key, str) and key and isinstance(value, str) and value}
        )
    return effective


def _uses_version_local_backtest_output(cwd: Path, workspace: dict[str, object]) -> bool:
    workflow = workspace.get("workflow")
    if not isinstance(workflow, dict):
        return False
    return workflow.get("layout") == "version_governed" and workflow.get("default_output_dir") == _workspace_backtest_output_template(
        cwd, workspace
    )


def _effective_version_phase_paths(
    cwd: Path,
    version_dir: Path,
    version_dir_display: str,
    version_manifest: dict[str, object],
) -> dict[str, str]:
    raw_phase_paths = version_manifest.get("phase_paths")
    if raw_phase_paths is not None and not isinstance(raw_phase_paths, dict):
        raise click.ClickException("active version_manifest.json phase_paths must be an object")
    configured = raw_phase_paths if isinstance(raw_phase_paths, dict) else {}
    effective: dict[str, str] = {}
    for phase in VERSION_PHASE_DIRS:
        raw_path = configured.get(phase, f"{version_dir_display}/{phase}")
        _resolve_version_phase_path(cwd, version_dir, phase, raw_path)
        effective[phase] = raw_path
    return effective


def _create_version_phase_dirs(
    cwd: Path,
    version_dir: Path,
    phase_paths: dict[str, str],
) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    for phase, raw_path in phase_paths.items():
        _resolve_version_phase_path(cwd, version_dir, phase, raw_path).mkdir(
            parents=True,
            exist_ok=True,
        )


def _read_json_object(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _lineage_identity_from_version_manifest(
    payload: dict[str, object],
    version_id: str,
) -> dict[str, object]:
    if payload.get("version_id") != version_id:
        raise click.ClickException(f"active version {version_id} requires a matching version_manifest.json")
    parent_version_id = payload.get("parent_version_id")
    created_reason = payload.get("created_reason")
    status = payload.get("status")
    if not _lineage_parent_version_id_is_valid(payload) or not isinstance(created_reason, str) or not created_reason:
        raise click.ClickException(f"active version {version_id} version_manifest.json lacks lineage identity")
    if status != "active":
        raise click.ClickException(f"active version {version_id} version_manifest.json must have status active")
    return {
        "version_id": version_id,
        "parent_version_id": parent_version_id,
        "created_reason": created_reason,
        "status": status,
    }


def _lineage_parent_version_id_is_valid(payload: dict[str, object]) -> bool:
    if "parent_version_id" not in payload:
        return False
    parent_version_id = payload["parent_version_id"]
    created_reason = payload.get("created_reason")
    if parent_version_id is None or parent_version_id == "":
        return created_reason == "initial_strategy_version"
    if created_reason == "initial_strategy_version":
        return False
    return isinstance(parent_version_id, str) and bool(_WORKSPACE_VERSION_RE.fullmatch(parent_version_id))


def _workflow_manifest_is_valid(payload: dict[str, object]) -> bool:
    paths = payload.get("paths")
    return (
        type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and payload.get("layout") == "version_governed"
        and isinstance(payload.get("strategy_family_id"), str)
        and bool(payload["strategy_family_id"])
        and isinstance(paths, dict)
        and _WORKSPACE_PATH_KEYS.issubset(paths)
        and all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in paths.items())
    )


def _legacy_hidden_workflow_manifest_is_valid(payload: dict[str, object]) -> bool:
    paths = payload.get("paths")
    return (
        type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and payload.get("layout") == "version_governed"
        and isinstance(payload.get("strategy_family_id"), str)
        and bool(payload["strategy_family_id"])
        and isinstance(paths, dict)
        and bool(paths)
        and all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in paths.items())
    )


def _phase_state_is_valid(payload: dict[str, object], active_version: str) -> bool:
    current_phase = payload.get("current_phase")
    completed_phases = payload.get("completed_phases")
    blocked_phase = payload.get("blocked_phase")
    return (
        type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and payload.get("version_id") == active_version
        and isinstance(current_phase, str)
        and current_phase in VERSION_PHASE_DIRS
        and payload.get("status") == "active"
        and isinstance(completed_phases, list)
        and all(isinstance(phase, str) and phase in VERSION_PHASE_DIRS for phase in completed_phases)
        and len(completed_phases) == len(set(completed_phases))
        and "blocked_phase" in payload
        and (blocked_phase is None or blocked_phase == "" or (isinstance(blocked_phase, str) and blocked_phase in VERSION_PHASE_DIRS))
    )


def _lineage_identity_matches(
    left: dict[str, object],
    right: dict[str, object],
    *,
    include_status: bool = True,
) -> bool:
    keys = ["version_id", "parent_version_id", "created_reason"]
    if include_status:
        keys.append("status")
    return all(key in left and key in right and left[key] == right[key] for key in keys)


def _validate_existing_governance_manifests(cwd: Path, workspace: dict[str, object]) -> None:
    if not _is_version_governed_workspace(workspace):
        return
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
        "current_manifest",
        "lineage_manifest",
        "workflow_manifest",
        "experiment_registry",
        "comparison_registry",
    ):
        _configured_path(cwd, workspace, key)

    checked: set[Path] = set()
    payloads: dict[Path, dict[str, object]] = {}
    current_payloads: dict[Path, dict[str, object]] = {}
    legacy_hidden_workflow_paths: set[Path] = set()
    for key, filename in (
        ("current_manifest", "current.json"),
        ("lineage_manifest", "lineage.json"),
        ("workflow_manifest", "workflow_manifest.json"),
    ):
        candidates = [cwd / filename]
        configured = _configured_path(cwd, workspace, key)
        if configured is not None:
            candidates.append(configured)
        for path in candidates:
            if path in checked or not path.exists():
                continue
            checked.add(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise click.ClickException(f"{filename} must contain a valid JSON object") from exc
            if not isinstance(payload, dict):
                raise click.ClickException(f"{filename} must contain a valid JSON object")
            if filename == "workflow_manifest.json" and not _workflow_manifest_is_valid(payload):
                is_legacy_hidden_source = (
                    configured is not None
                    and path == configured
                    and path.parent == cwd / ".open-xquant"
                    and _legacy_hidden_workflow_manifest_is_valid(payload)
                )
                if not is_legacy_hidden_source:
                    raise click.ClickException("workflow_manifest.json schema is invalid")
                legacy_hidden_workflow_paths.add(path)
            payloads[path] = payload
            if filename == "current.json":
                current_payloads[path] = payload

    configured_workflow = _configured_path(cwd, workspace, "workflow_manifest")
    authoritative_workflow = (
        configured_workflow if configured_workflow is not None and configured_workflow.exists() else cwd / "workflow_manifest.json"
    )
    workflow_payload = payloads.get(authoritative_workflow)
    if workflow_payload is not None:
        if authoritative_workflow in legacy_hidden_workflow_paths:
            effective_paths = _effective_workspace_path_map(workspace)
            persisted_paths = workflow_payload["paths"]
            assert isinstance(persisted_paths, dict)
            mismatches = sorted(key for key, persisted_value in persisted_paths.items() if effective_paths.get(key) != persisted_value)
        else:
            mismatches = _workflow_manifest_path_mismatches(
                workspace,
                workflow_payload,
            )
        if mismatches:
            key = mismatches[0]
            raise click.ClickException(
                f"workflow_manifest.json paths.{key} does not match workspace config; "
                "workspace root relocation requires an explicit migration"
            )

    configured_current = _configured_path(cwd, workspace, "current_manifest")
    authoritative_current = configured_current if configured_current is not None and configured_current.exists() else cwd / "current.json"
    active_versions: set[str] = set()
    for current_path, current_payload in current_payloads.items():
        if current_path == authoritative_current:
            active_phase = current_payload.get("active_phase")
            if active_phase not in (None, "") and active_phase not in VERSION_PHASE_DIRS:
                raise click.ClickException(
                    f"workspace current.json active_phase is invalid: {active_phase}; repair current.json before running research init"
                )
        active_version = current_payload.get("active_version")
        if not active_version:
            continue
        if not isinstance(active_version, str) or not _WORKSPACE_VERSION_RE.fullmatch(active_version):
            raise click.ClickException(
                f"workspace current.json active_version is unsafe: {active_version}; repair current.json before running research init"
            )
        if current_path == authoritative_current:
            active_versions.add(active_version)

    versions_dir = _configured_path(cwd, workspace, "versions_dir") or (cwd / "versions")
    configured_lineage = _configured_path(cwd, workspace, "lineage_manifest")
    authoritative_lineage = configured_lineage if configured_lineage is not None and configured_lineage.exists() else cwd / "lineage.json"
    lineage_payload = payloads.get(authoritative_lineage, {})
    lineage_raw = lineage_payload.get("versions")
    if lineage_raw is not None and not isinstance(lineage_raw, list):
        raise click.ClickException("lineage.json versions must be a list")
    lineage_versions = lineage_raw if isinstance(lineage_raw, list) else []
    for item in lineage_versions:
        if isinstance(item, dict) and item.get("status") not in _LINEAGE_STATUSES:
            raise click.ClickException(f"lineage.json version {item.get('version_id')} status must be active or superseded")
    active_lineage = [item for item in lineage_versions if isinstance(item, dict) and item.get("status") == "active"]
    if len(active_lineage) > 1:
        raise click.ClickException("lineage.json contains multiple active version entries")

    for active_version in sorted(active_versions):
        version_dir = _resolve_active_version_dir(cwd, versions_dir, active_version)
        matching_lineage = [item for item in lineage_versions if isinstance(item, dict) and item.get("version_id") == active_version]
        if len(matching_lineage) > 1:
            raise click.ClickException(f"lineage.json contains duplicate entries for active version {active_version}")
        version_manifest = version_dir / "version_manifest.json"
        version_manifest_payload: dict[str, object] | None = None
        if version_manifest.exists():
            try:
                raw_version_manifest = json.loads(version_manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise click.ClickException("version_manifest.json must contain a valid JSON object") from exc
            if not isinstance(raw_version_manifest, dict):
                raise click.ClickException("version_manifest.json must contain a valid JSON object")
            version_manifest_payload = raw_version_manifest
            manifest_identity = _lineage_identity_from_version_manifest(
                version_manifest_payload,
                active_version,
            )
            _effective_version_phase_paths(
                cwd,
                version_dir,
                _display_workspace_path(cwd, version_dir),
                version_manifest_payload,
            )
            for item in matching_lineage:
                if not _lineage_identity_matches(
                    item,
                    manifest_identity,
                    include_status=False,
                ):
                    raise click.ClickException(f"active version {active_version} lineage identity does not match version_manifest.json")
        if not matching_lineage:
            if not version_manifest.exists():
                raise click.ClickException(
                    f"active version {active_version} is absent from lineage.json and requires a matching version_manifest.json"
                )
        elif matching_lineage[0].get("status") == "superseded" and version_manifest_payload is None:
            raise click.ClickException(
                f"active version {active_version} is superseded in lineage.json and requires an active version_manifest.json for repair"
            )
        elif version_manifest_payload is None:
            lineage_identity = matching_lineage[0]
            if (
                lineage_identity.get("version_id") != active_version
                or not _lineage_parent_version_id_is_valid(lineage_identity)
                or not isinstance(lineage_identity.get("created_reason"), str)
                or not lineage_identity.get("created_reason")
                or lineage_identity.get("status") != "active"
            ):
                raise click.ClickException(f"active version {active_version} lineage identity is incomplete")
        for filename in ("version_manifest.json", "phase_state.json"):
            path = version_dir / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise click.ClickException(f"{filename} must contain a valid JSON object") from exc
            if not isinstance(payload, dict):
                raise click.ClickException(f"{filename} must contain a valid JSON object")
            if filename == "phase_state.json" and not _phase_state_is_valid(
                payload,
                active_version,
            ):
                raise click.ClickException("phase_state.json schema is invalid")

    if not active_versions:
        default_phase_state = (
            _resolve_active_version_dir(
                cwd,
                versions_dir,
                "v001",
            )
            / "phase_state.json"
        )
        if default_phase_state.exists():
            try:
                default_phase_payload = json.loads(default_phase_state.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise click.ClickException("phase_state.json must contain a valid JSON object") from exc
            if not isinstance(default_phase_payload, dict):
                raise click.ClickException("phase_state.json must contain a valid JSON object")
            if not _phase_state_is_valid(default_phase_payload, "v001"):
                raise click.ClickException("phase_state.json schema is invalid")

    if not active_versions and lineage_versions:
        raise click.ClickException("lineage.json is non-empty but current.json has no active_version")


def _resolve_sdk_venv(cwd: Path, raw_path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if expanded.is_absolute():
        return expanded.resolve()
    return (cwd / expanded).resolve()


def _installed_agent_profile() -> str:
    return read_recovered_agent_profile()
