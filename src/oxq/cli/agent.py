"""Agent lifecycle commands for installing open-xquant skills."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

import click

from oxq.cli.agent_manifest import (
    MarkerBlockError,
    expand_path,
    read_json_file,
    read_yaml_file,
    remove_marker_block,
    sha256_file,
    upsert_marker_block,
    write_json_file,
    write_text_file,
    write_yaml_file,
)
from oxq.cli.agent_targets import (
    CONCRETE_TARGETS,
    ROLE_TARGETS,
    SUPPORTED_TARGETS,
    AgentTarget,
    SkillValidationError,
    detect_targets,
    discover_agent_roles,
    discover_skills,
    render_agent_role_for_target,
    render_skill_for_target,
    resolve_source_root,
    resolve_target,
)
from oxq.cli.sdk_bundle import (
    build_sdk_bundle,
    remove_sdk_bundle,
    sdk_bundle_can_be_removed,
    sdk_bundle_contains_active_runner,
)
from oxq.process_lock import (
    ProcessFileLock,
    stable_filesystem_identity,
    stable_path_location_identity,
    verified_user_runtime_root,
)

MANAGED_MARKER = ".open-xquant-managed.json"
CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
LEGACY_PURGE_TRANSACTION_SCHEMA_VERSION = 1
RECORDED_ROOT_PURGE_TRANSACTION_SCHEMA_VERSION = 2
PURGE_TRANSACTION_SCHEMA_VERSION = 3
LIFECYCLE_TRANSACTION_SCHEMA_VERSION = 3
LIFECYCLE_MANIFEST_WITNESS_SCHEMA_VERSION = 1
AGENT_PROFILE_MULTI = "multi-agent"
AGENT_PROFILE_STANDALONE = "standalone-agent"
AGENT_PROFILES = (AGENT_PROFILE_MULTI, AGENT_PROFILE_STANDALONE)
MULTI_AGENT_RECOMMENDED_TARGETS = {"codex", "opencode", "claude-code", "cursor"}
DEPRECATED_SKILLS = {
    "authorized-backtest-runner",
    "backtest-runner",
    "chart-indicator",
    "component-author",
    "component-creator",
    "data-explorer",
    "experiment-comparator",
    "factor-evaluator",
    "factor-screening",
    "live-trader",
    "parameter-tuner",
    "performance-reviewer",
    "quant-research",
    "report-chart-builder",
    "research-report-reviewer",
    "research-report-writer",
    "rule-builder",
    "runtime-auditor",
    "spec-auditor",
    "strategy-builder",
    "strategy-builder-standalone",
    "strategy-monitor",
    "trade-executor",
    "universe-builder",
}


def config_dir() -> Path:
    return Path.home().joinpath(".config", "open-xquant").resolve()


def manifest_path() -> Path:
    return config_dir() / "agent-install.json"


def agent_config_path() -> Path:
    return config_dir() / "agent.yaml"


def purge_transaction_path() -> Path:
    return config_dir() / "agent-uninstall-transaction.json"


def lifecycle_transaction_path() -> Path:
    return config_dir() / "agent-lifecycle-transaction.json"


def lifecycle_manifest_witness_path() -> Path:
    return config_dir() / "agent-lifecycle-manifest-witness.json"


def lifecycle_manifest_witness_digest_path() -> Path:
    return config_dir() / "agent-lifecycle-manifest-witness.sha256"


def lifecycle_lock_path() -> Path:
    root = config_dir()
    identity_source = (
        f"filesystem:{stable_filesystem_identity(root)}"
        if root.exists()
        else _lifecycle_config_location_identity(root)
    )
    identity = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()[:24]
    return verified_user_runtime_root() / "agent" / f"{identity}.lock"


def _lifecycle_config_location_identity(root: Path) -> str:
    return stable_path_location_identity(root)


def _lifecycle_bootstrap_lock_path() -> Path:
    identity = hashlib.sha256(_lifecycle_config_location_identity(config_dir()).encode("utf-8")).hexdigest()[:24]
    return verified_user_runtime_root() / "agent" / f"{identity}.lock"


def _lifecycle_transition_lock_path() -> Path:
    return verified_user_runtime_root() / "agent" / "lifecycle-transition.lock"


@contextmanager
def _agent_lifecycle_lock() -> Iterator[None]:
    with agent_lifecycle_lock():
        yield


@contextmanager
def agent_lifecycle_lock() -> Iterator[None]:
    """Return the reentrant lock shared by lifecycle writers and readers."""

    with ExitStack() as locks:
        acquired: set[Path] = set()
        lock_paths = [_lifecycle_transition_lock_path(), _lifecycle_bootstrap_lock_path()]
        if config_dir().exists():
            lock_paths.append(lifecycle_lock_path())
        for lock_path in lock_paths:
            if lock_path in acquired:
                continue
            locks.enter_context(ProcessFileLock(lock_path))
            acquired.add(lock_path)
        yield


def _serialized_agent_lifecycle[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def serialized(*args: P.args, **kwargs: P.kwargs) -> R:
        with _agent_lifecycle_lock():
            return function(*args, **kwargs)

    return serialized


def default_agent_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "default_target": "auto",
        "installed_targets": [],
        "default_data_dir": "~/.oxq/data/market",
        "auto_init_workspace": True,
        "allow_auto_download": "ask",
        "preferred_runner": "uv run oxq",
    }


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


GLOBAL_AGENT_BLOCK = f"""## open-xquant

When the user asks about quant strategy, backtest, factor evaluation,
parameter tuning, audit, robustness, report, broker connectivity, or live
trading, use the installed `open-xquant` skill first.

Do not run `oxq`, SDK code, scripts, or write report files until the
`open-xquant` skill routes the task to a more specific open-xquant skill.

Before any routed skill runs open-xquant commands in a new directory:
- Read `~/.config/open-xquant/agent.yaml`.
- Prefer `preferred_runner_argv` when your shell tool accepts argv; otherwise
  use `preferred_runner` in place of `oxq` or `uv run oxq`.
- These runners point at the cached SDK bundle under
  `~/.config/open-xquant/sdk-bundles/`, not the original source checkout.
- If runner metadata is needed, read `~/.config/open-xquant/agent-install.json`.
- Keep the shell in the user's research directory. Do not search unrelated
  home directories for another open-xquant checkout.

{SUBAGENT_POLICY_BLOCK}"""


CLAUDE_AGENT_BLOCK = f"""## open-xquant

When the user asks about quant strategy, backtest, factor evaluation,
parameter tuning, audit, robustness, report, broker connectivity, or live
trading, use the installed `open-xquant` skill first.

Do not run `oxq`, SDK code, scripts, or write report files until the
`open-xquant` skill routes the task to a more specific open-xquant skill.

Before any routed skill runs open-xquant commands in a new directory:
- Read `~/.config/open-xquant/agent.yaml`.
- Prefer `preferred_runner_argv` when your shell tool accepts argv; otherwise
  use `preferred_runner` in place of `oxq` or `uv run oxq`.
- These runners point at the cached SDK bundle under
  `~/.config/open-xquant/sdk-bundles/`, not the original source checkout.
- If runner metadata is needed, read `~/.config/open-xquant/agent-install.json`.
- Keep the shell in the user's research directory. Do not search unrelated
  home directories for another open-xquant checkout.

If this project has an `AGENTS.md`, also read it when it is relevant to
open-xquant work.

{SUBAGENT_POLICY_BLOCK}"""


GENERIC_AGENT_BLOCK = f"""## open-xquant

When the user asks about quant strategy, backtest, factor evaluation,
parameter tuning, audit, robustness, report, broker connectivity, or live
trading, use the installed `open-xquant` skill first.

Do not run `oxq`, SDK code, scripts, or write report files until the
`open-xquant` skill routes the task to a more specific open-xquant skill.

Before any routed skill runs open-xquant commands in a new directory:
- Read `~/.config/open-xquant/agent.yaml`.
- Use `preferred_runner` in place of `oxq` or `uv run oxq`.
- For generic installs, this runner is only valid where the open-xquant
  command is already available. To get a portable cached runner, rerun
  `oxq agent install` with a concrete target such as `codex`, `opencode`,
  `claude-code`, `cursor`, `openclaw`, or `trae`.
- Keep the shell in the user's research directory. Do not search unrelated
  home directories for another open-xquant checkout.

{SUBAGENT_POLICY_BLOCK}"""


@click.group()
def agent() -> None:
    """Manage long-lived Agent integration for open-xquant."""


@agent.command()
@click.option("--target", type=click.Choice(SUPPORTED_TARGETS), default=None)
@click.option("--all-targets", is_flag=True, help="Install every supported concrete target.")
@click.option("--from-local", "from_local", default=None, help="Path to an open-xquant checkout.")
@click.option(
    "--profile",
    "agent_profile",
    type=click.Choice(AGENT_PROFILES),
    default=None,
    help="Install profile: multi-agent or standalone-agent.",
)
@click.option("--dry-run", is_flag=True, help="Show planned writes without changing files.")
@click.option("--repair", is_flag=True, help="Reinstall missing managed files.")
@click.option("--yes", is_flag=True, help="Run non-interactively.")
@_serialized_agent_lifecycle
def install(
    target: str | None,
    all_targets: bool,
    from_local: str | None,
    agent_profile: str | None,
    dry_run: bool,
    repair: bool,
    yes: bool,
) -> None:
    """Install open-xquant skills into supported Agent homes."""

    if _recover_pending_lifecycle_transaction(dry_run=dry_run) and dry_run:
        return
    _recover_pending_purge_cleanup(dry_run=dry_run)
    target_ids = _select_targets(target, all_targets)
    if target_ids == ["generic"]:
        _print_generic()
        _ensure_agent_config(dry_run=dry_run, installed_targets=[])
        return
    selected_profile = _select_agent_profile(agent_profile, target_ids, yes=yes)

    source_root = resolve_source_root(from_local)
    skills = _filter_skills_for_profile(_discover_skills_or_raise(source_root), selected_profile)
    agent_roles = _filter_agent_roles_for_profile(_discover_agent_roles_or_raise(source_root), selected_profile)
    manifest = _load_manifest()
    now = _now()
    manifest.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("installed_at", now)
    manifest["updated_at"] = now
    manifest["source"] = _source_metadata(source_root, "local")
    manifest["agent_profile"] = selected_profile
    manifest.setdefault("targets", {})

    installed: list[str] = []
    operations: list[tuple[Path, Path | None]] = []
    staging_root = Path(tempfile.mkdtemp(prefix="open-xquant-agent-stage-"))
    bundle_roots_before = _sdk_bundle_roots()
    bundle_cleanup_records: list[dict[str, Any]] = []
    committed_successfully = False
    try:
        for index, target_id in enumerate(target_ids):
            target_obj = resolve_target(target_id)
            existing_state = manifest["targets"].get(target_id) if isinstance(manifest.get("targets"), dict) else None
            if isinstance(existing_state, dict) and existing_state.get("installed"):
                target_obj = _target_for_owned_state(target_obj, existing_state)
            owned_state = existing_state if isinstance(existing_state, dict) and existing_state.get("installed") else {}
            skipped, target_state = _stage_target_upgrade(
                target_obj,
                owned_state,
                skills,
                agent_roles,
                source_root,
                agent_profile=selected_profile,
                staging_root=staging_root / f"target-{index}-{target_id}",
                operations=operations,
            )
            manifest["targets"][target_id] = target_state
            installed.append(target_id)
            if skipped:
                click.echo(f"{target_id}: skipped modified managed files: {', '.join(skipped)}")

        sdk_bundle = build_sdk_bundle(source_root, config_dir(), dry_run=dry_run)
        if not dry_run:
            bundle_cleanup_records = _capture_lifecycle_cleanup_records(
                sorted(_sdk_bundle_roots() - bundle_roots_before)
            )
        _record_sdk_bundle(manifest, sdk_bundle)
        config = _agent_config_payload(
            installed_targets=installed,
            sdk_bundle=sdk_bundle,
            agent_profile=selected_profile,
        )
        if not dry_run:
            _stage_agent_state_files(staging_root, config, manifest, operations)
            committed = _commit_target_upgrade(
                operations,
                rollback_cleanup_paths=sorted(_sdk_bundle_roots() - bundle_roots_before),
                rollback_cleanup_records=bundle_cleanup_records,
            )
            committed_successfully = True
            _finish_committed_lifecycle_transaction(committed)
    except BaseException:
        if not dry_run and not committed_successfully:
            _remove_new_sdk_bundle_roots(
                bundle_roots_before,
                bundle_cleanup_records,
            )
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    click.echo(f"Installed open-xquant agent support ({selected_profile}): " + ", ".join(installed))


@agent.command()
@click.option("--target", type=click.Choice(CONCRETE_TARGETS), default=None)
@click.option("--all-targets", is_flag=True, help="Uninstall every manifest target.")
@click.option("--dry-run", is_flag=True)
@click.option("--purge-config", is_flag=True)
@click.option("--yes", is_flag=True)
@_serialized_agent_lifecycle
def uninstall(target: str | None, all_targets: bool, dry_run: bool, purge_config: bool, yes: bool) -> None:
    """Uninstall managed Agent skills."""

    del yes
    if all_targets and target:
        raise click.ClickException("Use --target or --all-targets, not both.")
    if target is None and not all_targets:
        raise click.ClickException("Use --target or --all-targets to uninstall managed Agent files.")
    if _recover_pending_lifecycle_transaction(dry_run=dry_run) and dry_run:
        return
    recovered_pending_cleanup = _recover_pending_purge_cleanup(dry_run=dry_run)
    if recovered_pending_cleanup and purge_config and all_targets and not (manifest_path().exists() or manifest_path().is_symlink()):
        return
    manifest = _require_manifest()
    targets = manifest.get("targets", {})
    selected = list(targets) if all_targets else [target]
    bundles_to_purge: list[dict[str, Any]] = []
    if purge_config and all_targets:
        bundles_to_purge = _validated_purge_sdk_bundles(manifest)
        active_bundles = [_bundle_label(bundle) for bundle in bundles_to_purge if sdk_bundle_contains_active_runner(bundle, config_dir())]
        if active_bundles:
            raise click.ClickException(
                "Refusing to purge config while running from the active cached SDK runner: "
                + ", ".join(active_bundles)
                + ". Re-run this command from a non-cached open-xquant checkout or installed Python environment."
            )
        failed = [_bundle_label(bundle) for bundle in bundles_to_purge if not sdk_bundle_can_be_removed(bundle, config_dir())]
        if failed:
            raise click.ClickException("Refusing to purge config because SDK bundle removal was not verified: " + ", ".join(failed))
    operations: list[tuple[Path, Path | None]] = []
    committed: list[tuple[Path, Path | None]] = []
    staging_root = Path(tempfile.mkdtemp(prefix="open-xquant-agent-uninstall-stage-"))
    try:
        for index, target_id in enumerate(selected):
            state = targets.get(target_id)
            if not isinstance(state, dict) or not state.get("installed"):
                click.echo(f"{target_id}: not installed")
                continue
            _stage_target_uninstall(
                target_id,
                state,
                staging_root=staging_root / f"target-{index}-{target_id}",
                operations=operations,
            )
            state["installed"] = False
            state["updated_at"] = _now()

        if not dry_run:
            if purge_config and all_targets:
                operations.extend(((agent_config_path(), None), (manifest_path(), None)))
                bundle_paths = {_expand_lexical_path(bundle["root"]) for bundle in bundles_to_purge if isinstance(bundle.get("root"), str)}
                operations.extend((path, None) for path in sorted(bundle_paths))
                sdk_cache = config_dir() / "sdk-cache"
                if sdk_cache.exists() or sdk_cache.is_symlink():
                    operations.append((sdk_cache, None))
            else:
                manifest["updated_at"] = _now()
                staged_manifest = staging_root / "agent-install.json"
                write_json_file(staged_manifest, manifest)
                operations.append((manifest_path(), staged_manifest))
                if purge_config:
                    operations.append((agent_config_path(), None))
                else:
                    config = _load_agent_config()
                    config["installed_targets"] = [
                        target_id for target_id, state in targets.items() if isinstance(state, dict) and state.get("installed")
                    ]
                    staged_config = staging_root / "agent.yaml"
                    write_yaml_file(staged_config, config)
                    operations.append((agent_config_path(), staged_config))
            committed = _commit_target_upgrade(operations)
            if purge_config and all_targets:
                failed = [_bundle_label(bundle) for bundle in bundles_to_purge if not remove_sdk_bundle(bundle, config_dir())]
                if failed:
                    raise click.ClickException("Refusing to purge config because SDK bundle removal was not verified: " + ", ".join(failed))
                _write_pending_purge_cleanup(committed)
                _remove_lifecycle_transaction_metadata()
                committed = []
                _finish_pending_purge_cleanup()
            else:
                cleanup_committed = committed
                committed = []
                cleanup_failures = _finish_committed_lifecycle_transaction(cleanup_committed)
                if cleanup_failures:
                    raise click.ClickException(
                        "Uninstall committed, but transaction backup cleanup failed: " + ", ".join(str(path) for path in cleanup_failures)
                    )
    except BaseException:
        if committed:
            _rollback_committed_lifecycle_transaction(committed)
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    click.echo("Uninstall complete")


@agent.command()
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def status(as_json: bool) -> None:
    """Show Agent installation status."""

    payload = _status_payload()
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo("open-xquant agent status")
    click.echo("")
    click.echo(f"Config:   {agent_config_path()}")
    click.echo(f"Manifest: {manifest_path()}")
    for target_id, target_state in payload["targets"].items():
        click.echo("")
        click.echo(f"Target: {target_id}")
        click.echo(f"Installed: {'yes' if target_state['installed'] else 'no'}")
        click.echo(f"Skills: {target_state['skills']['installed']}/{target_state['skills']['expected']}")
        click.echo(f"Agent roles: {target_state['agent_roles']['installed']}/{target_state['agent_roles']['expected']}")
        click.echo(f"Instruction block: {target_state['instruction_block']}")
        click.echo(f"Commit: {target_state.get('commit') or 'unknown'}")


@agent.command()
@click.option("--target", type=click.Choice(CONCRETE_TARGETS), default=None)
@click.option("--all-targets", is_flag=True)
@click.option("--from-local", "from_local", default=None)
@click.option("--repo", default="https://github.com/xingwudao/open-xquant")
@click.option("--ref", "git_ref", default="main")
@click.option(
    "--profile",
    "agent_profile",
    type=click.Choice(AGENT_PROFILES),
    default=None,
    help="Upgrade with a specific install profile.",
)
@click.option("--dry-run", is_flag=True)
@click.option("--yes", is_flag=True)
@_serialized_agent_lifecycle
def upgrade(
    target: str | None,
    all_targets: bool,
    from_local: str | None,
    repo: str,
    git_ref: str,
    agent_profile: str | None,
    dry_run: bool,
    yes: bool,
) -> None:
    """Upgrade managed Agent skills from a local checkout or GitHub ref."""

    del yes
    if _recover_pending_lifecycle_transaction(dry_run=dry_run) and dry_run:
        return
    recovered_pending_cleanup = _recover_pending_purge_cleanup(dry_run=dry_run)
    if recovered_pending_cleanup and not dry_run and not (manifest_path().exists() or manifest_path().is_symlink()):
        click.echo("Upgrade complete: ")
        return
    manifest = _require_manifest()
    targets = manifest.get("targets", {})
    selected = list(targets) if all_targets or target is None else [target]
    upgrade_ids: list[str] = []
    for target_id in selected:
        state = targets.get(target_id) if isinstance(targets, dict) else None
        if not isinstance(state, dict) or not state.get("installed"):
            click.echo(f"{target_id}: not installed")
            continue
        upgrade_ids.append(target_id)
    if not upgrade_ids:
        click.echo("Upgrade complete: ")
        return

    source_root = _upgrade_source(from_local, repo, git_ref)
    discovered_skills = _discover_skills_or_raise(source_root)
    discovered_agent_roles = _discover_agent_roles_or_raise(source_root)
    updated: list[str] = []
    target_profiles: dict[str, str] = {}
    operations: list[tuple[Path, Path | None]] = []
    staging_root = Path(tempfile.mkdtemp(prefix="open-xquant-agent-stage-"))
    bundle_roots_before = _sdk_bundle_roots()
    bundle_cleanup_records: list[dict[str, Any]] = []
    committed_successfully = False
    try:
        for index, target_id in enumerate(upgrade_ids):
            state = targets.get(target_id)
            assert isinstance(state, dict)
            selected_profile = (
                agent_profile
                or _target_agent_profile(state)
                or _manifest_agent_profile(manifest)
                or _recommended_agent_profile([target_id])
            )
            skills = _filter_skills_for_profile(discovered_skills, selected_profile)
            agent_roles = _filter_agent_roles_for_profile(discovered_agent_roles, selected_profile)
            target_obj = _target_for_owned_state(resolve_target(target_id), state)
            skipped, updated_state = _stage_target_upgrade(
                target_obj,
                state,
                skills,
                agent_roles,
                source_root,
                agent_profile=selected_profile,
                staging_root=staging_root / f"target-{index}-{target_id}",
                operations=operations,
            )
            targets[target_id] = updated_state
            target_profiles[target_id] = selected_profile
            updated.append(target_id)
            if skipped:
                click.echo(f"{target_id}: skipped modified managed files: {', '.join(skipped)}")

        sdk_bundle = build_sdk_bundle(source_root, config_dir(), dry_run=dry_run)
        if not dry_run:
            bundle_cleanup_records = _capture_lifecycle_cleanup_records(
                sorted(_sdk_bundle_roots() - bundle_roots_before)
            )
        config_profile = agent_profile
        manifest["updated_at"] = _now()
        manifest["source"] = _source_metadata(source_root, "local" if from_local else "git")
        if agent_profile is not None:
            manifest["agent_profile"] = agent_profile
        _record_sdk_bundle(manifest, sdk_bundle)
        config = _agent_config_payload(
            installed_targets=updated,
            sdk_bundle=sdk_bundle,
            agent_profile=config_profile,
        )
        if not dry_run:
            _stage_agent_state_files(staging_root, config, manifest, operations)
            committed = _commit_target_upgrade(
                operations,
                rollback_cleanup_paths=sorted(_sdk_bundle_roots() - bundle_roots_before),
                rollback_cleanup_records=bundle_cleanup_records,
            )
            committed_successfully = True
            _finish_committed_lifecycle_transaction(committed)
    except BaseException:
        if not dry_run and not committed_successfully:
            _remove_new_sdk_bundle_roots(
                bundle_roots_before,
                bundle_cleanup_records,
            )
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    display_profile = _upgrade_display_profile(agent_profile, target_profiles)
    click.echo(f"Upgrade complete ({display_profile}): " + ", ".join(updated))


def _select_targets(target: str | None, all_targets: bool) -> list[str]:
    if all_targets and target:
        raise click.ClickException("Use --target or --all-targets, not both.")
    if all_targets:
        return list(CONCRETE_TARGETS)
    if target:
        return [target]
    config = _load_agent_config()
    default_target = config.get("default_target")
    if isinstance(default_target, str) and default_target not in {"", "auto"}:
        return [default_target]
    return detect_targets()


def _discover_skills_or_raise(source_root: Path) -> list[Any]:
    try:
        return discover_skills(source_root)
    except SkillValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _discover_agent_roles_or_raise(source_root: Path) -> list[Any]:
    try:
        return discover_agent_roles(source_root)
    except SkillValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def _select_agent_profile(profile: str | None, target_ids: list[str], yes: bool) -> str:
    if profile is not None:
        return profile
    recommended = _recommended_agent_profile(target_ids)
    if yes:
        click.echo(f"Agent install profile: {recommended}")
        return recommended
    click.echo("Choose how open-xquant skills should be installed for this machine.")
    click.echo("- multi-agent: recommended when your Agent supports multi-Agent/subagent workflows.")
    click.echo("- standalone-agent: for a single Agent that orchestrates the same narrow phase skills itself.")
    return click.prompt(
        "Install profile",
        type=click.Choice(AGENT_PROFILES),
        default=recommended,
        show_choices=True,
    )


def _recommended_agent_profile(target_ids: list[str]) -> str:
    if any(target_id in MULTI_AGENT_RECOMMENDED_TARGETS for target_id in target_ids):
        return AGENT_PROFILE_MULTI
    return AGENT_PROFILE_STANDALONE


def _filter_skills_for_profile(skills: list[Any], profile: str) -> list[Any]:
    filtered = [skill for skill in skills if skill.name not in DEPRECATED_SKILLS]
    if profile == AGENT_PROFILE_STANDALONE:
        return filtered
    if profile == AGENT_PROFILE_MULTI:
        return filtered
    raise click.ClickException(f"Unsupported agent profile: {profile}")


def _filter_agent_roles_for_profile(agent_roles: list[Any], profile: str) -> list[Any]:
    if profile == AGENT_PROFILE_MULTI:
        return agent_roles
    if profile == AGENT_PROFILE_STANDALONE:
        return []
    raise click.ClickException(f"Unsupported agent profile: {profile}")


def _manifest_agent_profile(manifest: dict[str, Any]) -> str | None:
    value = manifest.get("agent_profile")
    return value if value in AGENT_PROFILES else None


def _upgrade_display_profile(explicit_profile: str | None, target_profiles: dict[str, str]) -> str:
    if explicit_profile is not None:
        return explicit_profile
    profiles = {profile for profile in target_profiles.values() if profile in AGENT_PROFILES}
    if len(profiles) == 1:
        return next(iter(profiles))
    return "mixed profiles"


def _target_agent_profile(state: dict[str, Any] | None) -> str | None:
    if not isinstance(state, dict):
        return None
    value = state.get("agent_profile")
    return value if value in AGENT_PROFILES else None


def _instruction_block_for_target(target_id: str, agent_profile: str) -> str:
    content = CLAUDE_AGENT_BLOCK if target_id == "claude-code" else GLOBAL_AGENT_BLOCK
    if agent_profile == AGENT_PROFILE_MULTI:
        return content
    return content.replace(f"\n\n{SUBAGENT_POLICY_BLOCK}", "")


def _render_skill_for_target_and_profile(skill: Any, target_id: str, agent_profile: str) -> str:
    content = render_skill_for_target(skill, target_id)
    if agent_profile != AGENT_PROFILE_MULTI:
        return content
    if skill.name == "open-xquant":
        content = content.replace("Studio Worker", "Multi-Agent worker")
    return content


def _install_target(
    target: AgentTarget,
    skills: list[Any],
    agent_roles: list[Any],
    source_root: Path,
    dry_run: bool,
    repair: bool = False,
    existing_state: dict[str, Any] | None = None,
    agent_profile: str = AGENT_PROFILE_MULTI,
) -> dict[str, Any]:
    if target.id == "generic":
        raise click.ClickException("generic target does not install files.")
    assert target.skills_dir is not None
    target_skills: list[dict[str, Any]] = []
    target_agent_roles: list[dict[str, Any]] = []
    installed_paths: list[str] = []
    existing_records = {
        record["name"]: record
        for record in (existing_state or {}).get("skills", [])
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    by_name = {skill.name: skill for skill in skills}
    removed_names: list[str] = []
    for name, record in existing_records.items():
        if name in by_name:
            continue
        dest = expand_path(record["dest"]) if isinstance(record.get("dest"), str) else None
        if dest is not None and dest.exists() and sha256_file(dest) != record.get("dest_sha256"):
            target_skills.append(record)
            installed_paths.append(str(dest.parent.resolve()))
            continue
        if dest is not None and _remove_managed_skill_dir(target.id, dest.parent, dry_run=dry_run):
            removed_names.append(name)
    removed_names.extend(_remove_deprecated_managed_skill_dirs(target, dry_run=dry_run))
    for skill in skills:
        content = _render_skill_for_target_and_profile(skill, target.id, agent_profile)
        dest_dir = _safe_skill_dest_dir(target, skill.name)
        dest_file = dest_dir / "SKILL.md"
        marker_file = dest_dir / MANAGED_MARKER
        if dest_dir.exists() and not marker_file.exists():
            click.echo(f"{target.id}: skip unmarked existing skill {dest_dir}")
            continue
        if repair and marker_file.exists() and dest_file.exists():
            marker_data = read_json_file(marker_file)
            if marker_data.get("managed_by") == "open-xquant" and sha256_file(dest_file) != marker_data.get("dest_sha256"):
                click.echo(f"{target.id}: skip modified managed skill {dest_dir}")
                existing_record = existing_records.get(skill.name)
                if existing_record is not None:
                    installed_paths.append(str(dest_dir.resolve()))
                    target_skills.append(existing_record)
                continue
        dest_sha = _sha256_text(content)
        if not dry_run:
            _replace_managed_skill(
                source_dir=skill.path.parent,
                dest_dir=dest_dir,
                content=content,
                target_id=target.id,
                skill_name=skill.name,
                source_sha=skill.source_sha256,
                dest_sha=dest_sha,
            )
        installed_paths.append(str(dest_dir.resolve()))
        target_skills.append(
            {
                "name": skill.name,
                "source": str(skill.path.relative_to(source_root)),
                "dest": str(dest_file.resolve()),
                "source_sha256": skill.source_sha256,
                "dest_sha256": dest_sha,
            }
        )
    existing_role_records = {
        record["name"]: record
        for record in (existing_state or {}).get("agent_roles", [])
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    if target.id in ROLE_TARGETS and target.agents_dir is not None:
        target_agent_roles = _install_agent_roles_for_target(
            target,
            agent_roles,
            source_root,
            dry_run=dry_run,
            repair=repair,
            existing_records=existing_role_records,
        )
    elif agent_roles:
        click.echo(f"{target.id}: skip agent roles; target has no supported multi-agent role directory")
    managed_blocks = []
    if target.instruction_file is not None:
        content = _instruction_block_for_target(target.id, agent_profile)
        if not dry_run:
            upsert_marker_block(target.instruction_file, "open-xquant", content)
        managed_blocks.append({"file": str(target.instruction_file.resolve()), "marker": "open-xquant"})
    if target.id == "openclaw":
        if removed_names and target.config_file is not None:
            _remove_openclaw_config(target.config_file, removed_names, dry_run=dry_run)
        _merge_openclaw_config(target, [skill["name"] for skill in target_skills], dry_run=dry_run)
    return {
        "installed": True,
        "installed_at": _now(),
        "updated_at": _now(),
        "agent_profile": agent_profile,
        "skills_dir": str(target.skills_dir.resolve()),
        "agents_dir": str(target.agents_dir.resolve()) if target.agents_dir else None,
        "instruction_file": str(target.instruction_file.resolve()) if target.instruction_file else None,
        "config_file": str(target.config_file.resolve()) if target.config_file else None,
        "installed_paths": installed_paths,
        "managed_blocks": managed_blocks,
        "skills": target_skills,
        "agent_roles": target_agent_roles,
    }


def _stage_target_uninstall(
    target_id: str,
    state: dict[str, Any],
    *,
    staging_root: Path,
    operations: list[tuple[Path, Path | None]],
) -> None:
    target = _target_for_owned_state(resolve_target(target_id), state)
    staging_root.mkdir(parents=True, exist_ok=True)
    skill_records: dict[Path, dict[str, Any]] = {}
    for record in state.get("skills", []):
        if not isinstance(record, dict) or not isinstance(record.get("dest"), str):
            continue
        dest = _managed_skill_dest_path(target, record["dest"])
        skill_records[dest.parent] = record
    for raw_path in state.get("installed_paths", []):
        path = _managed_skill_dir_path(target, raw_path)
        record = skill_records.get(path)
        if record is None:
            raise click.ClickException(f"{target_id}: refusing to uninstall managed skill without integrity record: {path}")
        if _managed_skill_record_is_modified(path, record):
            raise click.ClickException(f"{target_id}: refusing to uninstall modified managed skill: {path}")
        if _managed_skill_dir_is_removable(target_id, path):
            operations.append((path, None))
    role_records = [record for record in state.get("agent_roles", []) if isinstance(record, dict)]
    recorded_agents_dir = _validated_recorded_agents_dir(target, state) if role_records else target.agents_dir
    for record in role_records:
        dest = _managed_agent_role_dest(target_id, record, agents_dir=recorded_agents_dir)
        if dest is not None and _managed_agent_role_is_removable(
            target_id,
            record,
            agents_dir=recorded_agents_dir,
        ):
            operations.append((dest, None))
    for index, block in enumerate(state.get("managed_blocks", [])):
        if not isinstance(block, dict) or not isinstance(block.get("file"), str) or not isinstance(block.get("marker"), str):
            raise click.ClickException(f"{target_id}: malformed managed instruction block record")
        instruction_file = _recorded_target_file(
            target_id,
            block["file"],
            target.instruction_file,
            label="instruction file",
        )
        if instruction_file is None or not instruction_file.exists():
            continue
        staged_instruction = staging_root / f"instruction-{index}"
        shutil.copy2(instruction_file, staged_instruction)
        try:
            remove_marker_block(staged_instruction, block["marker"])
        except MarkerBlockError as exc:
            raise click.ClickException(str(exc)) from exc
        operations.append((instruction_file, staged_instruction))
    if target_id == "openclaw" and state.get("config_file"):
        config_file = _recorded_target_file(
            target_id,
            state["config_file"],
            target.config_file,
            label="config file",
        )
        if config_file is not None and config_file.exists():
            data = _read_json_or_yaml(config_file)
            entries = data.get("skills", {}).get("entries", {}) if isinstance(data.get("skills"), dict) else {}
            if isinstance(entries, dict):
                for name in _skill_names(state):
                    entries.pop(name, None)
                staged_config = staging_root / "openclaw-config"
                write_json_file(staged_config, data)
                operations.append((config_file, staged_config))


def _stage_target_upgrade(
    target: AgentTarget,
    state: dict[str, Any],
    skills: list[Any],
    agent_roles: list[Any],
    source_root: Path,
    *,
    agent_profile: str,
    staging_root: Path,
    operations: list[tuple[Path, Path | None]],
) -> tuple[list[str], dict[str, Any]]:
    assert target.skills_dir is not None
    staging_root.mkdir(parents=True, exist_ok=True)
    by_name = {skill.name: skill for skill in skills}
    old_records: dict[str, dict[str, Any]] = {
        record["name"]: record for record in state.get("skills", []) if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    skipped: list[str] = []
    new_skill_records: list[dict[str, Any]] = []
    removed_names: list[str] = []
    old_role_records: dict[str, dict[str, Any]] = {
        record["name"]: record
        for record in state.get("agent_roles", [])
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    new_role_records: list[dict[str, Any]] = []
    recorded_agents_dir = _validated_recorded_agents_dir(target, state) if old_role_records else target.agents_dir
    for name, record in old_records.items():
        if name in by_name:
            continue
        raw_dest = record.get("dest")
        dest = _managed_skill_dest_path(target, raw_dest) if isinstance(raw_dest, str) else None
        if dest is None:
            continue
        if _managed_skill_record_is_modified(dest.parent, record):
            skipped.append(name)
            new_skill_records.append(record)
            continue
        if _managed_skill_dir_is_removable(target.id, dest.parent):
            operations.append((dest.parent, None))
            removed_names.append(name)
        else:
            new_skill_records.append(record)

    deprecated_removed, deprecated_operations = _plan_deprecated_managed_skill_dirs(target)
    removed_names.extend(deprecated_removed)
    operations.extend((path, None) for path in deprecated_operations)

    for index, source_skill in enumerate(skills):
        name = source_skill.name
        existing_record = old_records.get(name)
        dest = (
            _managed_skill_dest_path(target, existing_record["dest"])
            if existing_record and isinstance(existing_record.get("dest"), str)
            else _safe_skill_dest_dir(target, name) / "SKILL.md"
        )
        marker = dest.parent / MANAGED_MARKER
        if dest.parent.exists() and not marker.exists():
            click.echo(f"{target.id}: skip unmarked existing skill {dest.parent}")
            continue
        if existing_record and _managed_skill_record_is_modified(dest.parent, existing_record):
            skipped.append(name)
            new_skill_records.append(existing_record)
            continue
        content = _render_skill_for_target_and_profile(source_skill, target.id, agent_profile)
        dest_sha = _sha256_text(content)
        staged = staging_root / f"skill-{index}"
        hashes = _stage_managed_skill(
            source_dir=source_skill.path.parent,
            staged_dir=staged,
            content=content,
            target_id=target.id,
            skill_name=name,
            source_sha=source_skill.source_sha256,
            dest_sha=dest_sha,
        )
        operations.append((dest.parent, staged))
        new_skill_records.append(
            {
                "name": name,
                "source": str(source_skill.path.relative_to(source_root)),
                "dest": str(dest),
                "source_sha256": source_skill.source_sha256,
                "dest_sha256": dest_sha,
                **hashes,
            }
        )

    if target.id in ROLE_TARGETS and target.agents_dir is not None:
        role_by_name = {role.name: role for role in agent_roles}
        for name, record in old_role_records.items():
            if name in role_by_name:
                continue
            if _managed_agent_role_is_removable(target.id, record, agents_dir=recorded_agents_dir):
                raw_dest = record.get("dest")
                if isinstance(raw_dest, str):
                    dest = _managed_agent_role_dest(target.id, record, agents_dir=recorded_agents_dir)
                    if dest is not None:
                        operations.append((dest, None))
            else:
                new_role_records.append(record)
        for index, role in enumerate(agent_roles):
            filename, content = render_agent_role_for_target(role, target.id)
            existing_record = old_role_records.get(role.name)
            dest = (
                _managed_agent_role_dest(target.id, existing_record, agents_dir=recorded_agents_dir)
                if existing_record
                else _safe_agent_role_dest_file(target, filename)
            )
            if dest is None:
                skipped.append(role.name)
                new_role_records.append(existing_record)
                continue
            if dest.exists() and existing_record is None:
                click.echo(f"{target.id}: skip existing agent role {dest}")
                continue
            if existing_record and dest.exists() and sha256_file(dest) != existing_record.get("dest_sha256"):
                skipped.append(role.name)
                new_role_records.append(existing_record)
                continue
            dest_sha = _sha256_text(content)
            staged = staging_root / f"role-{index}-{filename}"
            write_text_file(staged, content)
            operations.append((dest, staged))
            new_role_records.append(
                {
                    "name": role.name,
                    "source": str(role.path.relative_to(source_root)),
                    "dest": str(dest.resolve()),
                    "source_sha256": role.source_sha256,
                    "dest_sha256": dest_sha,
                }
            )
    elif agent_roles:
        click.echo(f"{target.id}: skip agent roles; target has no supported multi-agent role directory")

    if target.instruction_file is not None:
        staged_instruction = staging_root / "instruction-file"
        if target.instruction_file.exists():
            shutil.copy2(target.instruction_file, staged_instruction)
        content = _instruction_block_for_target(target.id, agent_profile)
        upsert_marker_block(staged_instruction, "open-xquant", content)
        operations.append((target.instruction_file, staged_instruction))

    if target.id == "openclaw":
        staged_config = _stage_openclaw_upgrade_config(
            target,
            state,
            removed_names=removed_names,
            skill_names=[record["name"] for record in new_skill_records],
            staging_root=staging_root,
        )
        if staged_config is not None:
            config_dest, staged_path = staged_config
            operations.append((config_dest, staged_path))

    managed_blocks = []
    if target.instruction_file is not None:
        managed_blocks.append({"file": str(target.instruction_file.resolve()), "marker": "open-xquant"})
    now = _now()
    updated_state = {
        **state,
        "installed": True,
        "installed_at": state.get("installed_at", now),
        "updated_at": now,
        "agent_profile": agent_profile,
        "skills_dir": str(target.skills_dir.resolve()),
        "agents_dir": str(target.agents_dir.resolve()) if target.agents_dir else None,
        "instruction_file": str(target.instruction_file.resolve()) if target.instruction_file else None,
        "config_file": str(target.config_file.resolve()) if target.config_file else None,
        "installed_paths": [str(_managed_skill_dest_path(target, record["dest"]).parent) for record in new_skill_records],
        "managed_blocks": managed_blocks,
        "skills": new_skill_records,
        "agent_roles": new_role_records,
    }
    return skipped, updated_state


def _stage_managed_skill(
    *,
    source_dir: Path,
    staged_dir: Path,
    content: str,
    target_id: str,
    skill_name: str,
    source_sha: str,
    dest_sha: str,
) -> dict[str, str]:
    staged_dir.mkdir(parents=True)
    write_text_file(staged_dir / "SKILL.md", content)
    _sync_skill_resources(source_dir, staged_dir)
    managed_tree_sha = _hash_managed_skill_tree(staged_dir)
    resources_sha = _hash_managed_skill_resources(staged_dir)
    assert managed_tree_sha is not None
    assert resources_sha is not None
    _write_managed_marker(
        staged_dir / MANAGED_MARKER,
        target_id=target_id,
        skill_name=skill_name,
        source_sha=source_sha,
        dest_sha=dest_sha,
        managed_tree_sha=managed_tree_sha,
        resources_sha=resources_sha,
    )
    return {
        "managed_tree_sha256": managed_tree_sha,
        "resources_sha256": resources_sha,
        "marker_sha256": sha256_file(staged_dir / MANAGED_MARKER),
    }


def _hash_managed_skill_tree(path: Path) -> str | None:
    return _hash_managed_skill_files(path, include_skill=True)


def _hash_managed_skill_resources(path: Path) -> str | None:
    return _hash_managed_skill_files(path, include_skill=False)


def _hash_managed_skill_files(path: Path, *, include_skill: bool) -> str | None:
    digest = hashlib.sha256()
    if not path.exists() or path.is_symlink():
        return None
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            return None
        if not child.is_file() or child.name == MANAGED_MARKER:
            continue
        relative = child.relative_to(path).as_posix()
        if not include_skill and relative == "SKILL.md":
            continue
        encoded_path = relative.encode("utf-8")
        content = child.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _managed_skill_record_is_modified(path: Path, record: dict[str, Any]) -> bool:
    if not path.exists() or path.is_symlink():
        return path.is_symlink()
    dest = path / "SKILL.md"
    has_resources = any(child.name not in {"SKILL.md", MANAGED_MARKER} for child in path.iterdir())
    if dest.is_symlink() or (dest.exists() and (not dest.is_file() or sha256_file(dest) != record.get("dest_sha256"))):
        return True
    if not dest.exists():
        expected_resources = record.get("resources_sha256")
        if isinstance(expected_resources, str):
            return _hash_managed_skill_resources(path) != expected_resources
        return has_resources
    marker = path / MANAGED_MARKER
    expected_marker = record.get("marker_sha256")
    if isinstance(expected_marker, str) and (not marker.is_file() or sha256_file(marker) != expected_marker):
        return True

    expected_tree = record.get("managed_tree_sha256")
    if isinstance(expected_tree, str):
        return _hash_managed_skill_tree(path) != expected_tree

    if _legacy_marker_proves_skill_ownership(path, record):
        return False
    return has_resources


def _legacy_marker_proves_skill_ownership(path: Path, record: dict[str, Any]) -> bool:
    marker = path / MANAGED_MARKER
    if not marker.is_file() or marker.is_symlink():
        return False
    marker_data = read_json_file(marker)
    shared_fields = ("name", "source_sha256", "dest_sha256")
    if marker_data.get("managed_by") != "open-xquant" or any(
        not isinstance(record.get(field), str) or marker_data.get(field) != record[field] for field in shared_fields
    ):
        return False
    expected_tree = marker_data.get("managed_tree_sha256")
    expected_resources = marker_data.get("resources_sha256")
    return (
        isinstance(expected_tree, str)
        and isinstance(expected_resources, str)
        and _hash_managed_skill_tree(path) == expected_tree
        and _hash_managed_skill_resources(path) == expected_resources
    )


def _managed_skill_dir_is_removable(target_id: str, path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return True
    marker = path / MANAGED_MARKER
    if not marker.exists():
        click.echo(f"{target_id}: skip unmarked path {path}")
        return False
    marker_data = read_json_file(marker)
    if marker_data.get("managed_by") != "open-xquant":
        click.echo(f"{target_id}: skip unmanaged path {path}")
        return False
    if path.is_symlink():
        click.echo(f"{target_id}: skip symlink path {path}")
        return False
    return True


def _plan_deprecated_managed_skill_dirs(target: AgentTarget) -> tuple[list[str], list[Path]]:
    assert target.skills_dir is not None
    removed_names: list[str] = []
    removals: list[Path] = []
    for name in sorted(DEPRECATED_SKILLS):
        path = target.skills_dir / name
        if not path.exists() and not path.is_symlink():
            removed_names.append(name)
            continue
        marker = path / MANAGED_MARKER
        dest = path / "SKILL.md"
        if marker.exists() and dest.exists():
            marker_data = read_json_file(marker)
            if marker_data.get("managed_by") == "open-xquant" and sha256_file(dest) != marker_data.get("dest_sha256"):
                click.echo(f"{target.id}: skip modified deprecated skill {path}")
                continue
        if _managed_skill_dir_is_removable(target.id, path):
            removed_names.append(name)
            removals.append(path)
    return removed_names, removals


def _expand_lexical_path(path: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    return Path(os.path.abspath(expanded))


def _target_for_owned_state(target: AgentTarget, state: dict[str, Any]) -> AgentTarget:
    if target.id != "codex" or not state.get("installed"):
        return target
    skills_dir = _validated_owned_target_path(target.id, state, "skills_dir")
    assert skills_dir is not None
    codex_home = skills_dir.parent
    expected_paths = {
        "skills_dir": codex_home / "skills",
        "agents_dir": codex_home / "agents",
        "instruction_file": codex_home / "AGENTS.md",
        "config_file": None,
    }
    recorded_paths = {
        key: _validated_owned_target_path(target.id, state, key, optional=expected is None) for key, expected in expected_paths.items()
    }
    for key, expected in expected_paths.items():
        if recorded_paths[key] != expected:
            raise click.ClickException(f"{target.id}: refusing invalid persisted target path for {key}: {recorded_paths[key]}")
    return AgentTarget(
        id=target.id,
        skills_dir=recorded_paths["skills_dir"],
        agents_dir=recorded_paths["agents_dir"],
        instruction_file=recorded_paths["instruction_file"],
        config_file=recorded_paths["config_file"],
    )


def _validated_owned_target_path(
    target_id: str,
    state: dict[str, Any],
    field: str,
    *,
    optional: bool = False,
) -> Path | None:
    raw_path = state.get(field)
    if raw_path is None and optional:
        return None
    if not isinstance(raw_path, str):
        raise click.ClickException(f"{target_id}: refusing invalid persisted target path for {field}")
    try:
        path = _validated_absolute_lexical_path(raw_path)
    except ValueError as exc:
        raise click.ClickException(f"{target_id}: refusing invalid persisted target path for {field}: {exc}") from exc
    if path.resolve(strict=False) != path:
        raise click.ClickException(f"{target_id}: refusing persisted target path with symlink component: {path}")
    return path


def _reject_symlink_components(target_id: str, root: Path, path: Path) -> None:
    candidates = [root]
    candidates.extend(root / relative for relative in path.relative_to(root).parents if relative != Path("."))
    candidates.append(path)
    for candidate in candidates:
        if candidate.is_symlink():
            raise click.ClickException(f"{target_id}: refusing managed skill path with symlink component: {candidate}")


def _managed_skill_dest_path(target: AgentTarget, raw_path: str | Path) -> Path:
    assert target.skills_dir is not None
    root = _expand_lexical_path(target.skills_dir)
    dest = _expand_lexical_path(raw_path)
    if not dest.is_relative_to(root) or dest.name != "SKILL.md" or dest.parent.parent != root:
        raise click.ClickException(f"{target.id}: managed skill destination is outside skills_dir: {dest}")
    _reject_symlink_components(target.id, root, dest)
    return dest


def _managed_skill_dir_path(target: AgentTarget, raw_path: str | Path) -> Path:
    assert target.skills_dir is not None
    root = _expand_lexical_path(target.skills_dir)
    path = _expand_lexical_path(raw_path)
    if not path.is_relative_to(root) or path.parent != root:
        raise click.ClickException(f"{target.id}: managed skill directory is outside skills_dir: {path}")
    _reject_symlink_components(target.id, root, path)
    return path


def _recorded_target_file(
    target_id: str,
    raw_path: str | Path,
    expected: Path | None,
    *,
    label: str,
) -> Path | None:
    if expected is None:
        return None
    recorded = _expand_lexical_path(raw_path)
    expected_path = _expand_lexical_path(expected)
    if recorded != expected_path or recorded.is_symlink():
        raise click.ClickException(f"{target_id}: refusing unexpected managed {label}: {recorded}")
    return recorded


def _validated_recorded_agents_dir(target: AgentTarget, state: dict[str, Any]) -> Path:
    raw_agents_dir = state.get("agents_dir")
    if target.agents_dir is None or not isinstance(raw_agents_dir, (str, Path)):
        raise click.ClickException(f"{target.id}: refusing invalid managed agents directory")
    recorded = _expand_lexical_path(raw_agents_dir)
    if recorded.is_symlink():
        raise click.ClickException(f"{target.id}: refusing managed agent role path with symlink component: {recorded}")
    agents_dir = _recorded_target_file(
        target.id,
        recorded,
        target.agents_dir,
        label="agents directory",
    )
    assert agents_dir is not None
    return agents_dir


def _managed_agent_role_dest(
    target_id: str,
    record: dict[str, Any],
    *,
    agents_dir: object,
) -> Path | None:
    raw_dest = record.get("dest")
    if not isinstance(raw_dest, str) or not isinstance(agents_dir, (str, Path)):
        return None
    dest = _expand_lexical_path(raw_dest)
    root = _expand_lexical_path(agents_dir)
    if not dest.is_relative_to(root) or dest.parent != root:
        raise click.ClickException(f"{target_id}: refusing managed agent role outside agents directory: {dest}")
    if root.is_symlink():
        raise click.ClickException(f"{target_id}: refusing managed agent role path with symlink component: {root}")
    if dest.is_symlink():
        click.echo(f"{target_id}: skip symlink agent role {dest}")
        return None
    return dest


def _managed_agent_role_is_removable(
    target_id: str,
    record: dict[str, Any],
    *,
    agents_dir: object,
) -> bool:
    dest = _managed_agent_role_dest(target_id, record, agents_dir=agents_dir)
    if dest is None:
        return False
    if not dest.exists() and not dest.is_symlink():
        return True
    if sha256_file(dest) != record.get("dest_sha256"):
        click.echo(f"{target_id}: skip modified managed agent role {dest}")
        return False
    return True


def _stage_openclaw_upgrade_config(
    target: AgentTarget,
    state: dict[str, Any],
    *,
    removed_names: list[str],
    skill_names: list[str],
    staging_root: Path,
) -> tuple[Path, Path] | None:
    raw_config = state.get("config_file")
    if raw_config is not None and not isinstance(raw_config, (str, Path)):
        raise click.ClickException(f"{target.id}: refusing invalid managed config file")
    config_file = (
        _recorded_target_file(target.id, raw_config, target.config_file, label="config file")
        if raw_config is not None
        else target.config_file
    )
    if config_file is None or not config_file.exists():
        return None
    data = _read_json_or_yaml(config_file)
    skills = data.setdefault("skills", {})
    if not isinstance(skills, dict):
        raise click.ClickException(f"{target.id}: invalid OpenClaw config: skills must be an object")
    entries = skills.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise click.ClickException(f"{target.id}: invalid OpenClaw config: skills.entries must be an object")
    for name in removed_names:
        entries.pop(name, None)
    for name in skill_names:
        entries.setdefault(name, {})["enabled"] = True
    staged = staging_root / "openclaw-config"
    write_json_file(staged, data)
    return config_file, staged


def _commit_target_upgrade(
    operations: list[tuple[Path, Path | None]],
    *,
    rollback_cleanup_paths: list[Path] | None = None,
    rollback_cleanup_records: list[dict[str, Any]] | None = None,
) -> list[tuple[Path, Path | None]]:
    cleanup_paths = rollback_cleanup_paths or []
    created_parents = _missing_operation_parents([*operations, (lifecycle_transaction_path(), lifecycle_transaction_path())])
    for parent in created_parents:
        parent.mkdir()
    mutation_paths = [
        lifecycle_transaction_path(),
        lifecycle_manifest_witness_path(),
        lifecycle_manifest_witness_digest_path(),
        *(destination for destination, _staged in operations),
        *created_parents,
        *cleanup_paths,
    ]
    committed: list[tuple[Path, Path | None]] = []
    transaction_operations: list[dict[str, Any]] = []
    cleanup_records: list[dict[str, Any]] = []
    staging_complete = False
    with _secure_recovery_mutations(mutation_paths) as mutations:
        try:
            transaction_operations = _prepare_lifecycle_operations(operations)
            if rollback_cleanup_records is None:
                cleanup_records = _prepare_lifecycle_cleanup_records(
                    cleanup_paths,
                    mutations,
                )
            else:
                cleanup_records = rollback_cleanup_records
                if [Path(record["path"]) for record in cleanup_records] != cleanup_paths:
                    raise click.ClickException(
                        "Lifecycle rollback cleanup paths do not match captured ownership records"
                    )
                for record in cleanup_records:
                    _assert_lifecycle_path_evidence(
                        mutations,
                        Path(record["path"]),
                        record["evidence"],
                        label="rollback cleanup generation before journaling",
                    )
            authoritative_manifest = _lifecycle_authoritative_manifest(transaction_operations)
            _write_lifecycle_manifest_witness(authoritative_manifest)
            trusted_roots = _lifecycle_trusted_root_identities(
                _purge_trusted_roots(authoritative_manifest)
            )
            _write_lifecycle_transaction(
                phase="prepared",
                staging_complete=False,
                operations=transaction_operations,
                created_parents=created_parents,
                rollback_cleanup_paths=cleanup_records,
                trusted_roots=trusted_roots,
            )
            mutations.verify_all()
            for operation, (_destination, staged) in zip(
                transaction_operations,
                operations,
                strict=True,
            ):
                raw_local_staged = operation.get("local_staged")
                if staged is None or not isinstance(raw_local_staged, str):
                    continue
                local_staged = Path(raw_local_staged)
                mutations.create_placeholder(staged, local_staged)
                operation["local_staged_evidence"] = mutations.path_evidence(
                    local_staged
                )
                _write_lifecycle_transaction(
                    phase="prepared",
                    staging_complete=False,
                    operations=transaction_operations,
                    created_parents=created_parents,
                    rollback_cleanup_paths=cleanup_records,
                    trusted_roots=trusted_roots,
                )
            for operation, (_dest, staged) in zip(
                transaction_operations,
                operations,
                strict=True,
            ):
                raw_local_staged = operation.get("local_staged")
                if staged is not None and isinstance(raw_local_staged, str):
                    local_staged = Path(raw_local_staged)
                    mutations.copy_from(staged, local_staged)
                    operation["replacement_evidence"] = mutations.path_evidence(local_staged)
                    _write_lifecycle_transaction(
                        phase="prepared",
                        staging_complete=False,
                        operations=transaction_operations,
                        created_parents=created_parents,
                        rollback_cleanup_paths=cleanup_records,
                        trusted_roots=trusted_roots,
                    )
            _write_lifecycle_transaction(
                phase="prepared",
                staging_complete=True,
                operations=transaction_operations,
                created_parents=created_parents,
                rollback_cleanup_paths=cleanup_records,
                trusted_roots=trusted_roots,
            )
            staging_complete = True
            for operation, (dest, staged) in zip(transaction_operations, operations, strict=True):
                raw_backup = operation.get("backup")
                backup = Path(raw_backup) if isinstance(raw_backup, str) else None
                raw_local_staged = operation.get("local_staged")
                local_staged = Path(raw_local_staged) if isinstance(raw_local_staged, str) else None
                if operation["had_destination"]:
                    assert backup is not None
                    _assert_lifecycle_path_evidence(
                        mutations,
                        dest,
                        operation["original_evidence"],
                        label="destination before backup rename",
                    )
                    _assert_lifecycle_path_evidence(
                        mutations,
                        backup,
                        {"kind": "absent"},
                        label="backup before destination rename",
                    )
                    mutations.replace(dest, backup)
                    mutations.sync_parent(dest)
                    _assert_lifecycle_path_evidence(
                        mutations,
                        backup,
                        operation["original_evidence"],
                        label="original backup after destination rename",
                    )
                    _assert_lifecycle_path_evidence(
                        mutations,
                        dest,
                        {"kind": "absent"},
                        label="destination after backup rename",
                    )
                else:
                    _assert_lifecycle_path_evidence(
                        mutations,
                        dest,
                        {"kind": "absent"},
                        label="new destination before install rename",
                    )
                if staged is not None:
                    assert local_staged is not None
                    assert operation["replacement_evidence"] is not None
                    _assert_lifecycle_path_evidence(
                        mutations,
                        local_staged,
                        operation["replacement_evidence"],
                        label="replacement before install rename",
                    )
                    _assert_lifecycle_path_evidence(
                        mutations,
                        dest,
                        {"kind": "absent"},
                        label="destination before install rename",
                    )
                    mutations.replace(local_staged, dest)
                    mutations.sync_parent(dest)
                    _assert_lifecycle_path_evidence(
                        mutations,
                        dest,
                        operation["replacement_evidence"],
                        label="installed destination",
                    )
                    _assert_lifecycle_path_evidence(
                        mutations,
                        local_staged,
                        {"kind": "absent"},
                        label="consumed replacement stage",
                    )
                committed.append((dest, backup))
            _write_lifecycle_transaction(
                phase="committed",
                staging_complete=True,
                operations=transaction_operations,
                created_parents=created_parents,
                rollback_cleanup_paths=cleanup_records,
                trusted_roots=trusted_roots,
            )
        except BaseException:
            try:
                if staging_complete:
                    _rollback_lifecycle_operations(transaction_operations)
                else:
                    _secure_cleanup_local_lifecycle_stages(
                        transaction_operations,
                        mutations,
                    )
                _remove_created_parents(created_parents)
                _remove_cleanup_paths(cleanup_records)
                _remove_lifecycle_transaction_metadata()
            except BaseException:
                pass
            raise
    return committed


def _prepare_lifecycle_operations(
    operations: list[tuple[Path, Path | None]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    destinations: set[tuple[int, int, str]] = set()
    backups: set[Path] = set()
    local_stages: set[Path] = set()
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    for dest, staged in operations:
        destination = _expand_lexical_path(dest)
        parent_identity = mutations.parent_identity(destination) if mutations is not None else _path_parent_identity(destination)
        location_key = (
            parent_identity["device"],
            parent_identity["inode"],
            _normalized_lifecycle_name(destination.name),
        )
        if location_key in destinations:
            raise click.ClickException(f"Duplicate lifecycle destination: {destination}")
        destinations.add(location_key)
        if mutations is None:
            with _secure_recovery_mutations([destination]) as evidence_mutations:
                original_evidence = evidence_mutations.path_evidence(destination)
        else:
            original_evidence = mutations.path_evidence(destination)
        had_destination = original_evidence["kind"] != "absent"
        backup = _unused_lifecycle_backup_path(destination, backups) if had_destination else None
        if backup is not None:
            backups.add(backup)
        local_staged = _unused_lifecycle_install_path(destination, local_stages) if staged is not None else None
        if local_staged is not None:
            local_stages.add(local_staged)
        prepared.append(
            {
                "destination": str(destination),
                "staged": str(_expand_lexical_path(staged)) if staged is not None else None,
                "local_staged": str(local_staged) if local_staged is not None else None,
                "backup": str(backup) if backup is not None else None,
                "had_destination": had_destination,
                "relative_name": destination.name,
                "parent_identity": parent_identity,
                "original_evidence": original_evidence,
                "local_staged_evidence": None,
                "replacement_evidence": None,
            }
        )
    return prepared


def _unused_lifecycle_backup_path(destination: Path, reserved: set[Path]) -> Path:
    while True:
        candidate = destination.parent / f".{destination.name}.backup-{secrets.token_hex(8)}"
        if candidate not in reserved and not candidate.exists() and not candidate.is_symlink():
            return candidate


def _unused_lifecycle_install_path(destination: Path, reserved: set[Path]) -> Path:
    while True:
        candidate = destination.parent / f".{destination.name}.install-{secrets.token_hex(8)}"
        if candidate not in reserved and not candidate.exists() and not candidate.is_symlink():
            return candidate


def _missing_operation_parents(operations: list[tuple[Path, Path | None]]) -> list[Path]:
    created: list[Path] = []
    seen: set[Path] = set()
    for destination, staged in operations:
        if staged is None:
            continue
        missing: list[Path] = []
        parent = destination.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for path in reversed(missing):
            normalized = _expand_lexical_path(path)
            if normalized not in seen:
                seen.add(normalized)
                created.append(normalized)
    return created


def _prepare_lifecycle_cleanup_records(
    paths: list[Path],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _expand_lexical_path(raw_path)
        evidence = mutations.path_evidence(path)
        if evidence == {"kind": "absent"}:
            raise click.ClickException(
                f"Lifecycle rollback cleanup path is unavailable before journaling: {path}"
            )
        if evidence["kind"] != "directory":
            raise click.ClickException(
                f"Lifecycle rollback cleanup path is not a directory: {path}"
            )
        records.append({"path": str(path), "evidence": evidence})
    return records


def _capture_lifecycle_cleanup_records(
    paths: list[Path],
) -> list[dict[str, Any]]:
    if not paths:
        return []
    with _secure_recovery_mutations(paths) as mutations:
        return _prepare_lifecycle_cleanup_records(paths, mutations)


def _write_lifecycle_transaction(
    *,
    phase: str,
    staging_complete: bool,
    operations: list[dict[str, Any]],
    created_parents: list[Path],
    rollback_cleanup_paths: list[dict[str, Any]],
    trusted_roots: dict[str, Any],
) -> None:
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    journal_parent_identity = (
        mutations.parent_identity(lifecycle_transaction_path())
        if mutations is not None
        else _path_parent_identity(lifecycle_transaction_path())
    )
    _write_lifecycle_json(
        lifecycle_transaction_path(),
        {
            "schema_version": LIFECYCLE_TRANSACTION_SCHEMA_VERSION,
            "transaction_type": "agent-lifecycle",
            "phase": phase,
            "staging_complete": staging_complete,
            "created_at": _now(),
            "journal_parent_identity": journal_parent_identity,
            "operations": operations,
            "created_parents": [str(path) for path in created_parents],
            "rollback_cleanup_paths": rollback_cleanup_paths,
            "trusted_roots": trusted_roots,
        },
    )
    if mutations is None:
        _fsync_directory(lifecycle_transaction_path().parent)
    else:
        mutations.sync_parent(lifecycle_transaction_path())


def _fsync_directory(path: Path) -> None:
    if _is_windows():
        # Windows does not expose a supported directory fsync through os.open.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_identity_payload(status: os.stat_result) -> dict[str, int]:
    return {"device": int(status.st_dev), "inode": int(status.st_ino)}


def _path_parent_identity(path: Path) -> dict[str, int]:
    return _directory_identity_payload(path.parent.stat(follow_symlinks=False))


def _validated_directory_identity(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"device", "inode"}:
        raise ValueError(f"{label} must contain device and inode")
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
        raise ValueError(f"{label} device and inode must be non-negative integers")
    return {"device": device, "inode": inode}


def _normalized_lifecycle_name(value: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFKC", value).casefold())


def _evidence_snapshot(status: os.stat_result) -> tuple[int, ...]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_nlink),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _hash_evidence_value(digest: Any, tag: bytes, value: bytes) -> None:
    digest.update(tag)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _hash_evidence_status(digest: Any, status: os.stat_result, *, kind: str, name: bytes) -> None:
    _hash_evidence_value(digest, b"name", name)
    _hash_evidence_value(digest, b"kind", kind.encode("ascii"))
    for field, value in (
        (b"device", status.st_dev),
        (b"inode", status.st_ino),
        (b"mode", stat.S_IMODE(status.st_mode)),
        (b"links", status.st_nlink),
        (b"size", status.st_size),
        (b"uid", getattr(status, "st_uid", 0)),
        (b"gid", getattr(status, "st_gid", 0)),
    ):
        _hash_evidence_value(digest, field, str(int(value)).encode("ascii"))


def _hash_posix_evidence_entry(
    parent_descriptor: int,
    name: str,
    digest: Any,
    *,
    evidence_name: bytes,
) -> os.stat_result:
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    close_on_error = getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if stat.S_ISREG(before.st_mode):
        descriptor = os.open(name, os.O_RDONLY | no_follow | close_on_error, dir_fd=parent_descriptor)
        kind = "file"
    elif stat.S_ISDIR(before.st_mode):
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | no_follow | close_on_error,
            dir_fd=parent_descriptor,
        )
        kind = "directory"
    else:
        raise click.ClickException(f"Unsupported or symlinked lifecycle artifact: {name}")
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode)) != (
            before.st_dev,
            before.st_ino,
            stat.S_IFMT(before.st_mode),
        ):
            raise click.ClickException(f"Lifecycle artifact changed while evidence was captured: {name}")
        snapshot = _evidence_snapshot(opened)
        _hash_evidence_status(digest, opened, kind=kind, name=evidence_name)
        if kind == "file":
            digest.update(b"content\0")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        else:
            names = sorted(os.listdir(descriptor), key=os.fsencode)
            _hash_evidence_value(digest, b"entries", str(len(names)).encode("ascii"))
            for child_name in names:
                _hash_posix_evidence_entry(
                    descriptor,
                    child_name,
                    digest,
                    evidence_name=os.fsencode(child_name),
                )
        if _evidence_snapshot(os.fstat(descriptor)) != snapshot:
            raise click.ClickException(f"Lifecycle artifact changed while evidence was captured: {name}")
        return opened
    finally:
        os.close(descriptor)


def _hash_path_evidence_entry(path: Path, digest: Any, *, evidence_name: bytes) -> os.stat_result:
    before = path.lstat()
    if _path_is_reparse_point(path):
        raise click.ClickException(f"Unsupported or symlinked lifecycle artifact: {path}")
    if stat.S_ISREG(before.st_mode):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        kind = "file"
    elif stat.S_ISDIR(before.st_mode):
        descriptor = None
        kind = "directory"
    else:
        raise click.ClickException(f"Unsupported or symlinked lifecycle artifact: {path}")
    _hash_evidence_status(digest, before, kind=kind, name=evidence_name)
    snapshot = _evidence_snapshot(before)
    if descriptor is not None:
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode)) != (
                before.st_dev,
                before.st_ino,
                stat.S_IFMT(before.st_mode),
            ):
                raise click.ClickException(f"Lifecycle artifact changed while evidence was captured: {path}")
            digest.update(b"content\0")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            if _evidence_snapshot(os.fstat(descriptor)) != snapshot:
                raise click.ClickException(f"Lifecycle artifact changed while evidence was captured: {path}")
        finally:
            os.close(descriptor)
    else:
        entries = sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
        _hash_evidence_value(digest, b"entries", str(len(entries)).encode("ascii"))
        for child in entries:
            _hash_path_evidence_entry(child, digest, evidence_name=os.fsencode(child.name))
        if _evidence_snapshot(path.lstat()) != snapshot:
            raise click.ClickException(f"Lifecycle artifact changed while evidence was captured: {path}")
    return before


def _present_path_evidence(status: os.stat_result, digest: Any) -> dict[str, Any]:
    return {
        "kind": "file" if stat.S_ISREG(status.st_mode) else "directory",
        "device": int(status.st_dev),
        "inode": int(status.st_ino),
        "mode": stat.S_IMODE(status.st_mode),
        "tree_sha256": digest.hexdigest(),
    }


def _validated_lifecycle_evidence(value: Any, *, label: str, allow_none: bool = False) -> dict[str, Any] | None:
    if value is None and allow_none:
        return None
    if value == {"kind": "absent"}:
        return {"kind": "absent"}
    if not isinstance(value, dict) or set(value) != {"kind", "device", "inode", "mode", "tree_sha256"}:
        raise ValueError(f"{label} is not complete lifecycle path evidence")
    kind = value.get("kind")
    device = value.get("device")
    inode = value.get("inode")
    mode = value.get("mode")
    tree_sha256 = value.get("tree_sha256")
    if kind not in {"file", "directory"}:
        raise ValueError(f"{label} has an invalid kind")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in (device, inode, mode)):
        raise ValueError(f"{label} has invalid filesystem identity or mode")
    if (
        not isinstance(tree_sha256, str)
        or len(tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tree_sha256)
    ):
        raise ValueError(f"{label} has an invalid tree digest")
    return {
        "kind": kind,
        "device": device,
        "inode": inode,
        "mode": mode,
        "tree_sha256": tree_sha256,
    }


def _is_windows() -> bool:
    return os.name == "nt"


def _copy_file_to_directory_descriptor(
    source: Path,
    destination_parent: int,
    destination_name: str,
    mode: int,
) -> None:
    source_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    source_descriptor = os.open(source, source_flags)
    try:
        destination_descriptor = os.open(
            destination_name,
            destination_flags,
            stat.S_IMODE(mode),
            dir_fd=destination_parent,
        )
        try:
            while chunk := os.read(source_descriptor, 1024 * 1024):
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(destination_descriptor, remaining)
                    remaining = remaining[written:]
            os.fchmod(destination_descriptor, stat.S_IMODE(mode))
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def _copy_directory_to_directory_descriptor(
    source: Path,
    destination_parent: int,
    destination_name: str,
    mode: int,
) -> None:
    os.mkdir(
        destination_name,
        stat.S_IMODE(mode),
        dir_fd=destination_parent,
    )
    destination_descriptor = os.open(
        destination_name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        dir_fd=destination_parent,
    )
    try:
        _copy_directory_contents_to_descriptor(source, destination_descriptor)
        os.fchmod(destination_descriptor, stat.S_IMODE(mode))
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)


def _copy_directory_contents_to_descriptor(
    source: Path,
    destination_descriptor: int,
) -> None:
    with os.scandir(source) as entries:
        for entry in entries:
            child = source / entry.name
            child_status = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(child_status.st_mode):
                _copy_file_to_directory_descriptor(
                    child,
                    destination_descriptor,
                    entry.name,
                    child_status.st_mode,
                )
            elif stat.S_ISDIR(child_status.st_mode):
                _copy_directory_to_directory_descriptor(
                    child,
                    destination_descriptor,
                    entry.name,
                    child_status.st_mode,
                )
            else:
                raise click.ClickException(
                    f"Refusing unsupported staged lifecycle artifact: {child}"
                )


def _copy_directory_contents_to_path(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        child_status = child.lstat()
        target = destination / child.name
        if _path_is_reparse_point(child):
            raise click.ClickException(
                f"Refusing unsupported staged lifecycle artifact: {child}"
            )
        if stat.S_ISREG(child_status.st_mode):
            shutil.copy2(child, target, follow_symlinks=False)
            with target.open("rb") as handle:
                os.fsync(handle.fileno())
        elif stat.S_ISDIR(child_status.st_mode):
            target.mkdir(mode=stat.S_IMODE(child_status.st_mode))
            _copy_directory_contents_to_path(child, target)
            target.chmod(stat.S_IMODE(child_status.st_mode))
        else:
            raise click.ClickException(
                f"Refusing unsupported staged lifecycle artifact: {child}"
            )


def _copy_staged_path(source: Path, destination: Path) -> None:
    source_status = source.lstat()
    if stat.S_ISREG(source_status.st_mode):
        shutil.copy2(source, destination, follow_symlinks=False)
    elif stat.S_ISDIR(source_status.st_mode):
        shutil.copytree(source, destination, symlinks=False)
    else:
        raise click.ClickException(f"Refusing unsupported staged lifecycle artifact: {source}")
    paths = [destination]
    if destination.is_dir():
        paths.extend(destination.rglob("*"))
    for path in reversed(paths):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise click.ClickException(f"Refusing symlinked staged lifecycle artifact: {path}")
        if stat.S_ISREG(status.st_mode):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif stat.S_ISDIR(status.st_mode) and not _is_windows():
            _fsync_directory(path)


class _PosixRecoveryMutations:
    def __init__(self, paths: list[Path]) -> None:
        self._directories: dict[Path, tuple[int, tuple[int, int]] | None] = {}
        try:
            for path in paths:
                self._snapshot_parent(path.parent)
        except BaseException:
            self.close()
            raise

    def _snapshot_parent(self, parent: Path) -> None:
        if parent in self._directories:
            return
        try:
            descriptor = self._open_directory(parent)
        except FileNotFoundError:
            self._directories[parent] = None
            return
        except OSError as exc:
            raise click.ClickException(f"Secure recovery rejected changed or symlinked directory: {parent} ({exc})") from exc
        info = os.fstat(descriptor)
        self._directories[parent] = (descriptor, (info.st_dev, info.st_ino))

    @staticmethod
    def _open_directory(path: Path) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(path.anchor, flags)
        try:
            for part in path.parts[1:]:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _parent_descriptor(self, path: Path) -> int | None:
        parent = path.parent
        snapshot = self._directories[parent]
        if snapshot is None:
            try:
                descriptor = self._open_directory(parent)
            except FileNotFoundError:
                return None
            else:
                os.close(descriptor)
                raise click.ClickException(f"Secure recovery directory identity changed after validation: {parent}")
        descriptor, identity = snapshot
        try:
            current = os.stat(parent, follow_symlinks=False)
        except OSError as exc:
            raise click.ClickException(f"Secure recovery directory identity changed after validation: {parent} ({exc})") from exc
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            raise click.ClickException(f"Secure recovery directory identity changed after validation: {parent}")
        return descriptor

    def _captured_parent_descriptor(self, path: Path) -> int:
        snapshot = self._directories[path.parent]
        if snapshot is None:
            raise click.ClickException(f"Secure lifecycle destination directory is unavailable: {path.parent}")
        return snapshot[0]

    def read_bytes(self, path: Path) -> bytes:
        descriptor = self._captured_parent_descriptor(path)
        file_descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
        try:
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise click.ClickException(f"Secure lifecycle metadata is not a regular file: {path}")
            chunks: list[bytes] = []
            while chunk := os.read(file_descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)

    def write_bytes(self, path: Path, content: bytes) -> None:
        descriptor = self._captured_parent_descriptor(path)
        temporary_name = f".{path.name}.write-{secrets.token_hex(8)}"
        file_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=descriptor,
        )
        try:
            remaining = memoryview(content)
            while remaining:
                written = os.write(file_descriptor, remaining)
                remaining = remaining[written:]
            os.fsync(file_descriptor)
        except BaseException:
            os.close(file_descriptor)
            os.unlink(temporary_name, dir_fd=descriptor)
            raise
        else:
            os.close(file_descriptor)
        try:
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            if not _is_windows():
                os.fsync(descriptor)
        except BaseException:
            try:
                os.unlink(temporary_name, dir_fd=descriptor)
            except FileNotFoundError:
                pass
            raise

    def exists(self, path: Path) -> bool:
        descriptor = self._parent_descriptor(path)
        if descriptor is None:
            return False
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def path_evidence(self, path: Path) -> dict[str, Any]:
        descriptor = self._parent_descriptor(path)
        if descriptor is None:
            return {"kind": "absent"}
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return {"kind": "absent"}
        digest = hashlib.sha256()
        status = _hash_posix_evidence_entry(
            descriptor,
            path.name,
            digest,
            evidence_name=b"",
        )
        return _present_path_evidence(status, digest)

    def parent_identity(self, path: Path) -> dict[str, int]:
        descriptor = self._parent_descriptor(path)
        if descriptor is None:
            raise click.ClickException(f"Secure lifecycle destination directory is unavailable: {path.parent}")
        return _directory_identity_payload(os.fstat(descriptor))

    def assert_parent_identity(
        self,
        path: Path,
        expected: dict[str, int],
    ) -> None:
        if self.parent_identity(path) != expected:
            raise click.ClickException(f"Secure recovery directory identity does not match journal: {path.parent}")

    def remove(self, path: Path) -> None:
        descriptor = self._parent_descriptor(path)
        if descriptor is None:
            return
        try:
            info = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISDIR(info.st_mode):
            if not shutil.rmtree.avoids_symlink_attacks:
                raise click.ClickException(f"Secure recovery cannot remove a directory on this platform: {path}")
            shutil.rmtree(path.name, dir_fd=descriptor)
        else:
            os.unlink(path.name, dir_fd=descriptor)

    def create_placeholder(self, source: Path, destination: Path) -> None:
        descriptor = self._parent_descriptor(destination)
        if descriptor is None:
            raise click.ClickException(
                f"Secure lifecycle destination directory is unavailable: {destination.parent}"
            )
        if self.exists(destination):
            raise click.ClickException(
                f"Transaction-owned lifecycle stage already exists: {destination}"
            )
        source_status = source.lstat()
        mode = stat.S_IMODE(source_status.st_mode)
        if stat.S_ISREG(source_status.st_mode):
            placeholder = os.open(
                destination.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                mode,
                dir_fd=descriptor,
            )
            try:
                os.fchmod(placeholder, mode)
                os.fsync(placeholder)
            finally:
                os.close(placeholder)
        elif stat.S_ISDIR(source_status.st_mode):
            os.mkdir(destination.name, mode, dir_fd=descriptor)
            placeholder = os.open(
                destination.name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                os.fchmod(placeholder, mode)
                os.fsync(placeholder)
            finally:
                os.close(placeholder)
        else:
            raise click.ClickException(
                f"Refusing unsupported staged lifecycle artifact: {source}"
            )
        self.sync_parent(destination)

    def copy_from(self, source: Path, destination: Path) -> None:
        descriptor = self._parent_descriptor(destination)
        if descriptor is None:
            raise click.ClickException(f"Secure lifecycle destination directory is unavailable: {destination.parent}")
        source_status = source.lstat()
        try:
            destination_status = os.stat(
                destination.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise click.ClickException(
                f"Transaction-owned lifecycle stage is unavailable: {destination}"
            ) from exc
        if stat.S_ISREG(source_status.st_mode) and stat.S_ISREG(
            destination_status.st_mode
        ):
            source_descriptor = os.open(
                source,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            )
            destination_descriptor = os.open(
                destination.name,
                os.O_WRONLY
                | os.O_TRUNC
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                while chunk := os.read(source_descriptor, 1024 * 1024):
                    remaining = memoryview(chunk)
                    while remaining:
                        written = os.write(destination_descriptor, remaining)
                        remaining = remaining[written:]
                os.fchmod(
                    destination_descriptor,
                    stat.S_IMODE(source_status.st_mode),
                )
                os.fsync(destination_descriptor)
            finally:
                os.close(destination_descriptor)
                os.close(source_descriptor)
        elif stat.S_ISDIR(source_status.st_mode) and stat.S_ISDIR(
            destination_status.st_mode
        ):
            destination_descriptor = os.open(
                destination.name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                if os.listdir(destination_descriptor):
                    raise click.ClickException(
                        f"Transaction-owned lifecycle stage is not empty: {destination}"
                    )
                _copy_directory_contents_to_descriptor(
                    source,
                    destination_descriptor,
                )
                os.fchmod(
                    destination_descriptor,
                    stat.S_IMODE(source_status.st_mode),
                )
                os.fsync(destination_descriptor)
            finally:
                os.close(destination_descriptor)
        else:
            raise click.ClickException(
                f"Lifecycle staging placeholder kind changed before copy: {destination}"
            )
        self.sync_parent(destination)

    def replace(self, source: Path, destination: Path) -> None:
        if source.parent != destination.parent:
            raise click.ClickException("Secure recovery replacement must remain within one directory")
        descriptor = self._parent_descriptor(source)
        if descriptor is None:
            raise click.ClickException(f"Secure recovery source directory is unavailable: {source.parent}")
        os.replace(
            source.name,
            destination.name,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    def quarantine(self, source: Path, destination: Path) -> None:
        if source.parent != destination.parent:
            raise click.ClickException(
                "Secure recovery quarantine must remain within one directory"
            )
        descriptor = self._parent_descriptor(source)
        if descriptor is None:
            raise click.ClickException(
                f"Secure recovery source directory is unavailable: {source.parent}"
            )
        os.replace(
            source.name,
            destination.name,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )

    def rmdir(self, path: Path) -> None:
        descriptor = self._parent_descriptor(path)
        if descriptor is None:
            return
        os.rmdir(path.name, dir_fd=descriptor)
        removed_snapshot = self._directories.get(path)
        if removed_snapshot is not None:
            os.close(removed_snapshot[0])
            self._directories[path] = None

    def sync_parent(self, path: Path) -> None:
        snapshot = self._directories[path.parent]
        if snapshot is None:
            self._parent_descriptor(path)
            return
        descriptor, _identity = snapshot
        os.fsync(descriptor)
        self._parent_descriptor(path)

    def verify_all(self) -> None:
        for parent in self._directories:
            self._parent_descriptor(parent / "__oxq_identity_check__")

    def close(self) -> None:
        for snapshot in self._directories.values():
            if snapshot is not None:
                os.close(snapshot[0])
        self._directories.clear()


class _WindowsRecoveryMutations:
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_EXISTING = 3

    def __init__(self, paths: list[Path]) -> None:
        import ctypes

        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._create_file = self._kernel32.CreateFileW
        self._create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._create_file.restype = ctypes.c_void_p
        self._close_handle = self._kernel32.CloseHandle
        self._close_handle.argtypes = [ctypes.c_void_p]
        self._close_handle.restype = ctypes.c_int
        self._get_file_attributes = self._kernel32.GetFileAttributesW
        self._get_file_attributes.argtypes = [ctypes.c_wchar_p]
        self._get_file_attributes.restype = ctypes.c_uint32
        self._handles: dict[Path, int] = {}
        self._identities: dict[Path, tuple[int, int]] = {}
        self._missing: set[Path] = set()
        try:
            for path in paths:
                self._snapshot_parent(path.parent)
        except BaseException:
            self.close()
            raise

    def _directory_chain(self, path: Path) -> list[Path]:
        current = Path(path.anchor)
        result = [current]
        for part in path.parts[1:]:
            current /= part
            result.append(current)
        return result

    def _open_directory(self, path: Path) -> int:
        ctypes = self._ctypes
        handle = self._create_file(
            str(path),
            self._FILE_READ_ATTRIBUTES,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            None,
            self._OPEN_EXISTING,
            self._FILE_FLAG_BACKUP_SEMANTICS | self._FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            error = ctypes.get_last_error()
            if error in {2, 3}:
                raise FileNotFoundError(error, os.strerror(error), str(path))
            raise OSError(error, os.strerror(error), str(path))
        attributes = self._get_file_attributes(str(path))
        if attributes == 0xFFFFFFFF or not attributes & self._FILE_ATTRIBUTE_DIRECTORY:
            self._close_handle(handle)
            raise OSError(f"recovery parent is not a directory: {path}")
        if attributes & self._FILE_ATTRIBUTE_REPARSE_POINT:
            self._close_handle(handle)
            raise OSError(f"recovery parent is a reparse point: {path}")
        return handle

    def _snapshot_parent(self, parent: Path) -> None:
        if parent in self._identities or parent in self._missing:
            return
        for component in self._directory_chain(parent):
            if component in self._handles:
                continue
            try:
                handle = self._open_directory(component)
            except FileNotFoundError:
                self._missing.add(parent)
                return
            except OSError as exc:
                raise click.ClickException(f"Secure recovery rejected changed or reparse directory: {component} ({exc})") from exc
            self._handles[component] = handle
        info = os.stat(parent, follow_symlinks=False)
        self._identities[parent] = (info.st_dev, info.st_ino)

    def _verify_parent(self, path: Path) -> bool:
        parent = path.parent
        if parent in self._missing:
            if parent.exists() or parent.is_symlink():
                raise click.ClickException(f"Secure recovery directory identity changed after validation: {parent}")
            return False
        try:
            info = os.stat(parent, follow_symlinks=False)
        except OSError as exc:
            raise click.ClickException(f"Secure recovery directory identity changed after validation: {parent} ({exc})") from exc
        if (info.st_dev, info.st_ino) != self._identities[parent]:
            raise click.ClickException(f"Secure recovery directory identity changed after validation: {parent}")
        return True

    def exists(self, path: Path) -> bool:
        return self._verify_parent(path) and (path.exists() or path.is_symlink())

    def path_evidence(self, path: Path) -> dict[str, Any]:
        if not self._verify_parent(path) or not (path.exists() or path.is_symlink()):
            return {"kind": "absent"}
        digest = hashlib.sha256()
        status = _hash_path_evidence_entry(path, digest, evidence_name=b"")
        self._verify_parent(path)
        return _present_path_evidence(status, digest)

    def read_bytes(self, path: Path) -> bytes:
        if not self._verify_parent(path):
            raise click.ClickException(f"Secure lifecycle destination directory is unavailable: {path.parent}")
        content = path.read_bytes()
        self._verify_parent(path)
        return content

    def write_bytes(self, path: Path, content: bytes) -> None:
        if not self._verify_parent(path):
            raise click.ClickException(f"Secure lifecycle destination directory is unavailable: {path.parent}")
        temporary = path.parent / f".{path.name}.write-{secrets.token_hex(8)}"
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self._verify_parent(path)
            os.replace(temporary, path)
            self._verify_parent(path)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()

    def parent_identity(self, path: Path) -> dict[str, int]:
        if not self._verify_parent(path):
            raise click.ClickException(f"Secure lifecycle destination directory is unavailable: {path.parent}")
        device, inode = self._identities[path.parent]
        return {"device": device, "inode": inode}

    def assert_parent_identity(
        self,
        path: Path,
        expected: dict[str, int],
    ) -> None:
        if self.parent_identity(path) != expected:
            raise click.ClickException(f"Secure recovery directory identity does not match journal: {path.parent}")

    def remove(self, path: Path) -> None:
        if not self._verify_parent(path):
            return
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    def create_placeholder(self, source: Path, destination: Path) -> None:
        if not self._verify_parent(destination):
            raise click.ClickException(
                f"Secure lifecycle destination directory is unavailable: {destination.parent}"
            )
        if destination.exists() or destination.is_symlink():
            raise click.ClickException(
                f"Transaction-owned lifecycle stage already exists: {destination}"
            )
        source_status = source.lstat()
        if _path_is_reparse_point(source):
            raise click.ClickException(
                f"Refusing unsupported staged lifecycle artifact: {source}"
            )
        if stat.S_ISREG(source_status.st_mode):
            with destination.open("xb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        elif stat.S_ISDIR(source_status.st_mode):
            destination.mkdir()
        else:
            raise click.ClickException(
                f"Refusing unsupported staged lifecycle artifact: {source}"
            )
        destination.chmod(stat.S_IMODE(source_status.st_mode))
        self._verify_parent(destination)

    def copy_from(self, source: Path, destination: Path) -> None:
        if not self._verify_parent(destination):
            raise click.ClickException(f"Secure lifecycle destination directory is unavailable: {destination.parent}")
        source_status = source.lstat()
        destination_status = destination.lstat()
        if _path_is_reparse_point(source) or _path_is_reparse_point(destination):
            raise click.ClickException(
                f"Refusing unsupported staged lifecycle artifact: {source}"
            )
        if stat.S_ISREG(source_status.st_mode) and stat.S_ISREG(
            destination_status.st_mode
        ):
            with source.open("rb") as source_handle, destination.open(
                "r+b"
            ) as destination_handle:
                destination_handle.truncate(0)
                shutil.copyfileobj(source_handle, destination_handle)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        elif stat.S_ISDIR(source_status.st_mode) and stat.S_ISDIR(
            destination_status.st_mode
        ):
            if any(destination.iterdir()):
                raise click.ClickException(
                    f"Transaction-owned lifecycle stage is not empty: {destination}"
                )
            _copy_directory_contents_to_path(source, destination)
        else:
            raise click.ClickException(
                f"Lifecycle staging placeholder kind changed before copy: {destination}"
            )
        destination.chmod(stat.S_IMODE(source_status.st_mode))
        self._verify_parent(destination)

    def replace(self, source: Path, destination: Path) -> None:
        if source.parent != destination.parent:
            raise click.ClickException("Secure recovery replacement must remain within one directory")
        if not self._verify_parent(source):
            raise click.ClickException(f"Secure recovery source directory is unavailable: {source.parent}")
        source.replace(destination)

    def quarantine(self, source: Path, destination: Path) -> None:
        if source.parent != destination.parent:
            raise click.ClickException(
                "Secure recovery quarantine must remain within one directory"
            )
        if not self._verify_parent(source):
            raise click.ClickException(
                f"Secure recovery source directory is unavailable: {source.parent}"
            )
        source.replace(destination)

    def rmdir(self, path: Path) -> None:
        if not self._verify_parent(path):
            return
        removed_handle = self._handles.pop(path, None)
        removed_identity = self._identities.pop(path, None)
        if removed_handle is not None:
            self._close_handle(removed_handle)
        try:
            path.rmdir()
        except BaseException:
            if removed_identity is not None:
                self._snapshot_parent(path)
            raise
        if removed_identity is not None:
            self._missing.add(path)

    def sync_parent(self, path: Path) -> None:
        self._verify_parent(path)

    def verify_all(self) -> None:
        for parent in self._identities:
            self._verify_parent(parent / "__oxq_identity_check__")
        for parent in self._missing:
            self._verify_parent(parent / "__oxq_identity_check__")

    def close(self) -> None:
        for handle in self._handles.values():
            self._close_handle(handle)
        self._handles.clear()


_ACTIVE_RECOVERY_MUTATIONS: ContextVar[_PosixRecoveryMutations | _WindowsRecoveryMutations | None] = ContextVar(
    "active_agent_recovery_mutations", default=None
)
_ACTIVE_VERIFIED_REMOVAL: ContextVar[
    tuple[
        _PosixRecoveryMutations | _WindowsRecoveryMutations,
        Path,
        dict[str, Any],
        bool,
        str,
    ]
    | None
] = ContextVar("active_agent_verified_removal", default=None)


@contextmanager
def _secure_recovery_mutations(
    paths: list[Path],
) -> Iterator[_PosixRecoveryMutations | _WindowsRecoveryMutations]:
    mutations = _WindowsRecoveryMutations(paths) if os.name == "nt" else _PosixRecoveryMutations(paths)
    token = _ACTIVE_RECOVERY_MUTATIONS.set(mutations)
    try:
        yield mutations
    finally:
        _ACTIVE_RECOVERY_MUTATIONS.reset(token)
        mutations.close()


def _lifecycle_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_lifecycle_bytes(path: Path, content: bytes) -> None:
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is None:
        write_text_file(path, content.decode("utf-8"))
        return
    mutations.write_bytes(path, content)


def _write_lifecycle_json(path: Path, payload: dict[str, Any]) -> None:
    _write_lifecycle_bytes(path, _lifecycle_json_bytes(payload))


def _read_lifecycle_json(path: Path) -> dict[str, Any]:
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    content = path.read_bytes() if mutations is None else mutations.read_bytes(path)
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError(f"lifecycle metadata must be a JSON object: {path}")
    return payload


def _assert_lifecycle_path_evidence(
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
    path: Path,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    actual = mutations.path_evidence(path)
    if actual != expected:
        raise click.ClickException(f"Lifecycle {label} changed from recorded transaction evidence: {path}")


def _assert_lifecycle_generation_evidence(
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
    path: Path,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    actual = mutations.path_evidence(path)
    identity_keys = ("kind", "device", "inode", "mode")
    if any(actual.get(key) != expected.get(key) for key in identity_keys):
        raise click.ClickException(
            f"Lifecycle {label} changed from recorded transaction evidence: {path}"
        )


def _lifecycle_quarantine_path(
    path: Path,
    evidence: dict[str, Any],
    *,
    generation_only: bool,
) -> Path:
    identity = json.dumps(
        {
            "name": _normalized_lifecycle_name(path.name),
            "evidence": evidence,
            "generation_only": generation_only,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return path.parent / f".oxq-quarantine-{token}"


def _assert_expected_removal_evidence(
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
    path: Path,
    evidence: dict[str, Any],
    *,
    generation_only: bool,
    label: str,
) -> None:
    if generation_only:
        _assert_lifecycle_generation_evidence(
            mutations,
            path,
            evidence,
            label=label,
        )
    else:
        _assert_lifecycle_path_evidence(
            mutations,
            path,
            evidence,
            label=label,
        )


def _secure_remove_evidenced_path(
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
    path: Path,
    evidence: dict[str, Any],
    *,
    generation_only: bool = False,
    label: str,
) -> None:
    token = _ACTIVE_VERIFIED_REMOVAL.set(
        (mutations, path, evidence, generation_only, label)
    )
    try:
        _remove_upgrade_path(path)
    finally:
        _ACTIVE_VERIFIED_REMOVAL.reset(token)


def _evidenced_removal_pending(
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
    path: Path,
    evidence: dict[str, Any],
    *,
    generation_only: bool = False,
) -> bool:
    quarantine = _lifecycle_quarantine_path(
        path,
        evidence,
        generation_only=generation_only,
    )
    return mutations.exists(path) or mutations.exists(quarantine)


def _quarantine_and_remove_evidenced_path(
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
    path: Path,
    evidence: dict[str, Any],
    *,
    generation_only: bool,
    label: str,
) -> None:
    quarantine = _lifecycle_quarantine_path(
        path,
        evidence,
        generation_only=generation_only,
    )
    if mutations.exists(quarantine):
        _assert_expected_removal_evidence(
            mutations,
            quarantine,
            evidence,
            generation_only=generation_only,
            label=f"{label} in quarantine",
        )
        mutations.remove(quarantine)
        mutations.sync_parent(quarantine)
        return
    if not mutations.exists(path):
        return
    _assert_expected_removal_evidence(
        mutations,
        path,
        evidence,
        generation_only=generation_only,
        label=f"{label} before quarantine",
    )
    quarantine_evidence = (
        mutations.path_evidence(path) if generation_only else evidence
    )
    mutations.quarantine(path, quarantine)
    mutations.sync_parent(path)
    try:
        _assert_expected_removal_evidence(
            mutations,
            quarantine,
            quarantine_evidence,
            generation_only=False,
            label=f"{label} after quarantine",
        )
    except BaseException:
        try:
            if mutations.exists(quarantine) and not mutations.exists(path):
                mutations.replace(quarantine, path)
                mutations.sync_parent(path)
        except BaseException:
            pass
        raise
    mutations.remove(quarantine)
    mutations.sync_parent(quarantine)


def _rollback_lifecycle_operations(operations: list[dict[str, Any]]) -> None:
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is not None:
        _secure_rollback_lifecycle_operations(operations, mutations)
        return
    paths: list[Path] = []
    for operation in operations:
        for key in ("destination", "backup", "local_staged"):
            raw_path = operation.get(key)
            if isinstance(raw_path, str):
                paths.append(Path(raw_path))
    with _secure_recovery_mutations(paths) as secure_mutations:
        _secure_rollback_lifecycle_operations(operations, secure_mutations)


def _remove_created_parents(created_parents: list[Path]) -> None:
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is not None:
        _secure_remove_created_parents(created_parents, mutations)
        return
    for parent in reversed(created_parents):
        try:
            parent.rmdir()
        except OSError:
            pass


def _remove_cleanup_paths(records: list[dict[str, Any]]) -> None:
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is not None:
        _secure_remove_cleanup_paths(records, mutations)
        return
    paths = [Path(record["path"]) for record in records]
    with _secure_recovery_mutations(paths) as secure_mutations:
        _secure_remove_cleanup_paths(records, secure_mutations)


def _lifecycle_recovery_paths(transaction: Path, payload: dict[str, Any]) -> list[Path]:
    paths = [
        transaction,
        lifecycle_manifest_witness_path(),
        lifecycle_manifest_witness_digest_path(),
    ]
    for operation in payload["operations"]:
        for key in ("destination", "backup", "local_staged"):
            raw_path = operation.get(key)
            if isinstance(raw_path, str):
                paths.append(Path(raw_path))
    paths.extend(Path(path) for path in payload["created_parents"])
    paths.extend(
        Path(record["path"]) for record in payload["rollback_cleanup_paths"]
    )
    return paths


def _secure_rollback_lifecycle_operations(
    operations: list[dict[str, Any]],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    _secure_cleanup_rollback_quarantines(operations, mutations)
    plans: list[dict[str, Any]] = []
    failures: list[BaseException] = []
    for operation in reversed(operations):
        try:
            plans.append(_lifecycle_rollback_plan(operation, mutations))
        except BaseException as exc:
            failures.append(exc)
    for plan in plans:
        try:
            _execute_lifecycle_rollback_plan(plan, mutations)
        except BaseException as exc:
            failures.append(exc)
    if failures:
        raise failures[0]


def _secure_cleanup_rollback_quarantines(
    operations: list[dict[str, Any]],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    for operation in operations:
        replacement_evidence = operation.get("replacement_evidence")
        if replacement_evidence is None:
            continue
        for key in ("destination", "local_staged"):
            raw_path = operation.get(key)
            if not isinstance(raw_path, str):
                continue
            path = Path(raw_path)
            quarantine = _lifecycle_quarantine_path(
                path,
                replacement_evidence,
                generation_only=False,
            )
            if mutations.exists(quarantine):
                _secure_remove_evidenced_path(
                    mutations,
                    path,
                    replacement_evidence,
                    label=f"interrupted {key} rollback cleanup",
                )


def _execute_lifecycle_rollback_plan(
    plan: dict[str, Any],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    destination = plan["destination"]
    backup = plan["backup"]
    local_staged = plan["local_staged"]
    replacement_evidence = plan["replacement_evidence"]
    original_evidence = plan["original_evidence"]
    if plan["remove_destination"]:
        assert replacement_evidence is not None
        _secure_remove_evidenced_path(
            mutations,
            destination,
            replacement_evidence,
            label="installed destination rollback removal",
        )
    if plan["remove_local_stage"]:
        assert local_staged is not None and replacement_evidence is not None
        _secure_remove_evidenced_path(
            mutations,
            local_staged,
            replacement_evidence,
            label="local replacement rollback removal",
        )
    if plan["restore_backup"]:
        assert backup is not None
        _assert_lifecycle_path_evidence(
            mutations,
            destination,
            {"kind": "absent"},
            label="destination before original restore",
        )
        _assert_lifecycle_path_evidence(
            mutations,
            backup,
            original_evidence,
            label="original backup before restore",
        )
        mutations.replace(backup, destination)
        mutations.sync_parent(destination)
        _assert_lifecycle_path_evidence(
            mutations,
            destination,
            original_evidence,
            label="restored original destination",
        )


def _lifecycle_rollback_plan(
    operation: dict[str, Any],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> dict[str, Any]:
    absent = {"kind": "absent"}
    destination = Path(operation["destination"])
    raw_backup = operation.get("backup")
    backup = Path(raw_backup) if isinstance(raw_backup, str) else None
    raw_local_staged = operation.get("local_staged")
    local_staged = Path(raw_local_staged) if isinstance(raw_local_staged, str) else None
    original_evidence = operation["original_evidence"]
    replacement_evidence = operation.get("replacement_evidence")
    destination_evidence = mutations.path_evidence(destination)
    backup_evidence = mutations.path_evidence(backup) if backup is not None else absent
    local_evidence = mutations.path_evidence(local_staged) if local_staged is not None else absent
    remove_destination = False
    remove_local_stage = False
    restore_backup = False

    if backup is not None:
        if backup_evidence == original_evidence:
            if destination_evidence == replacement_evidence and replacement_evidence is not None:
                remove_destination = True
            elif destination_evidence != absent:
                raise click.ClickException(
                    f"Lifecycle rollback found an unrecognized destination; journal preserved: {destination}"
                )
            restore_backup = True
        elif backup_evidence == absent:
            if destination_evidence != original_evidence:
                raise click.ClickException(
                    f"Lifecycle rollback cannot prove the recorded original destination; journal preserved: {destination}"
                )
        else:
            raise click.ClickException(f"Lifecycle rollback found an unrecognized backup; journal preserved: {backup}")
    else:
        if original_evidence != absent:
            raise click.ClickException(f"Lifecycle rollback evidence is internally inconsistent: {destination}")
        if destination_evidence == replacement_evidence and replacement_evidence is not None:
            remove_destination = True
        elif destination_evidence != absent:
            raise click.ClickException(
                f"Lifecycle rollback found an unrecognized destination; journal preserved: {destination}"
            )

    if local_staged is not None:
        if local_evidence == replacement_evidence and replacement_evidence is not None:
            remove_local_stage = True
        elif local_evidence != absent:
            raise click.ClickException(
                f"Lifecycle rollback found an unrecognized local stage; journal preserved: {local_staged}"
            )
    return {
        "destination": destination,
        "backup": backup,
        "local_staged": local_staged,
        "original_evidence": original_evidence,
        "replacement_evidence": replacement_evidence,
        "remove_destination": remove_destination,
        "remove_local_stage": remove_local_stage,
        "restore_backup": restore_backup,
    }


def _secure_cleanup_local_lifecycle_stages(
    operations: list[dict[str, Any]],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    for operation in operations:
        raw_local_staged = operation.get("local_staged")
        if not isinstance(raw_local_staged, str):
            continue
        local_staged = Path(raw_local_staged)
        replacement_evidence = operation.get("replacement_evidence")
        if replacement_evidence is not None:
            if _evidenced_removal_pending(
                mutations,
                local_staged,
                replacement_evidence,
            ):
                _secure_remove_evidenced_path(
                    mutations,
                    local_staged,
                    replacement_evidence,
                    label="local replacement cleanup",
                )
            continue
        local_staged_evidence = operation.get("local_staged_evidence")
        if local_staged_evidence is None:
            if mutations.exists(local_staged):
                raise click.ClickException(
                    "Lifecycle local staging path has no recorded ownership "
                    f"evidence; journal preserved: {local_staged}"
                )
            continue
        if _evidenced_removal_pending(
            mutations,
            local_staged,
            local_staged_evidence,
            generation_only=True,
        ):
            _secure_remove_evidenced_path(
                mutations,
                local_staged,
                local_staged_evidence,
                generation_only=True,
                label="incomplete local staging generation cleanup",
            )


def _secure_remove_created_parents(
    created_parents: list[Path],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    for parent in reversed(created_parents):
        try:
            mutations.rmdir(parent)
            mutations.sync_parent(parent)
        except click.ClickException:
            raise
        except OSError:
            pass


def _secure_remove_cleanup_paths(
    records: list[dict[str, Any]],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    for record in records:
        path = Path(record["path"])
        if _evidenced_removal_pending(
            mutations,
            path,
            record["evidence"],
        ):
            _secure_remove_evidenced_path(
                mutations,
                path,
                record["evidence"],
                label="rollback cleanup generation removal",
            )


def _secure_cleanup_upgrade_backups(
    committed: list[
        tuple[Path, Path | None]
        | tuple[Path, Path | None, dict[str, Any] | None]
    ],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> list[Path]:
    failures: list[Path] = []
    for record in committed:
        _destination, backup = record[:2]
        if backup is None:
            continue
        evidence = record[2] if len(record) == 3 else None
        if evidence is None and not mutations.exists(backup):
            continue
        if evidence is not None and not _evidenced_removal_pending(
            mutations,
            backup,
            evidence,
        ):
            continue
        try:
            if evidence is None:
                _remove_upgrade_path(backup)
            else:
                _secure_remove_evidenced_path(
                    mutations,
                    backup,
                    evidence,
                    label="retained purge backup cleanup",
                )
            mutations.sync_parent(backup)
        except click.ClickException:
            raise
        except Exception as exc:
            click.echo(f"retained backup after cleanup failure: {backup} ({exc})")
            failures.append(backup)
    return failures


def _secure_cleanup_lifecycle_backups(
    operations: list[dict[str, Any]],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> list[Path]:
    _assert_committed_lifecycle_state(operations, mutations)
    records = [
        (
            Path(operation["destination"]),
            Path(operation["backup"]) if isinstance(operation.get("backup"), str) else None,
            operation["original_evidence"],
        )
        for operation in operations
    ]
    manifest_records = [record for record in records if _same_lifecycle_location(record[0], manifest_path())]
    failures = _secure_cleanup_evidenced_backups(
        [record for record in records if record not in manifest_records],
        mutations,
    )
    if failures:
        return failures
    return _secure_cleanup_evidenced_backups(manifest_records, mutations)


def _secure_cleanup_evidenced_backups(
    records: list[tuple[Path, Path | None, dict[str, Any]]],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> list[Path]:
    failures: list[Path] = []
    for _destination, backup, original_evidence in records:
        if backup is None or not _evidenced_removal_pending(
            mutations,
            backup,
            original_evidence,
        ):
            continue
        try:
            _secure_remove_evidenced_path(
                mutations,
                backup,
                original_evidence,
                label="original backup committed cleanup",
            )
            mutations.sync_parent(backup)
        except click.ClickException:
            raise
        except Exception as exc:
            click.echo(f"retained backup after cleanup failure: {backup} ({exc})")
            failures.append(backup)
    return failures


def _assert_committed_lifecycle_state(
    operations: list[dict[str, Any]],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    absent = {"kind": "absent"}
    for operation in operations:
        destination = Path(operation["destination"])
        replacement_evidence = operation.get("replacement_evidence")
        expected_destination = replacement_evidence if replacement_evidence is not None else absent
        _assert_lifecycle_path_evidence(
            mutations,
            destination,
            expected_destination,
            label="committed destination",
        )
        raw_local_staged = operation.get("local_staged")
        if isinstance(raw_local_staged, str):
            _assert_lifecycle_path_evidence(
                mutations,
                Path(raw_local_staged),
                absent,
                label="committed local stage",
            )
        raw_backup = operation.get("backup")
        if isinstance(raw_backup, str):
            backup = Path(raw_backup)
            backup_evidence = mutations.path_evidence(backup)
            if backup_evidence not in (absent, operation["original_evidence"]):
                raise click.ClickException(
                    f"Lifecycle committed backup changed from recorded transaction evidence; journal preserved: {backup}"
                )


def _same_lifecycle_location(first: Path, second: Path) -> bool:
    if _normalized_lifecycle_name(first.name) != _normalized_lifecycle_name(second.name):
        return False
    try:
        return _path_parent_identity(first) == _path_parent_identity(second)
    except OSError:
        return False


def _secure_remove_transaction_metadata(
    transaction: Path,
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    mutations.verify_all()
    if mutations.exists(transaction):
        mutations.remove(transaction)
        mutations.sync_parent(transaction)


def _secure_remove_lifecycle_transaction_metadata(
    transaction: Path,
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    _secure_remove_transaction_metadata(transaction, mutations)
    for evidence in (
        lifecycle_manifest_witness_digest_path(),
        lifecycle_manifest_witness_path(),
    ):
        if mutations.exists(evidence):
            mutations.remove(evidence)
            mutations.sync_parent(evidence)


def _rollback_target_upgrade(committed: list[tuple[Path, Path | None]]) -> None:
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is not None:
        for dest, backup in reversed(committed):
            if mutations.exists(dest):
                mutations.remove(dest)
                mutations.sync_parent(dest)
            if backup is not None and mutations.exists(backup):
                mutations.replace(backup, dest)
                mutations.sync_parent(dest)
        return
    for dest, backup in reversed(committed):
        if dest.exists() or dest.is_symlink():
            _remove_upgrade_path(dest)
        if backup is not None and (backup.exists() or backup.is_symlink()):
            backup.replace(dest)
            _fsync_directory(dest.parent)


def _rollback_committed_lifecycle_transaction(
    committed: list[tuple[Path, Path | None]],
) -> None:
    transaction = lifecycle_transaction_path()
    payload = _validated_pending_lifecycle_transaction(transaction)
    _require_matching_committed_operations(payload, committed)
    with _secure_recovery_mutations(_lifecycle_recovery_paths(transaction, payload)) as mutations:
        _assert_lifecycle_parent_identities(transaction, payload, mutations)
        _mark_lifecycle_transaction_prepared()
        _secure_rollback_lifecycle_operations(payload["operations"], mutations)
        _remove_lifecycle_transaction_metadata()


def _finish_committed_lifecycle_transaction(
    committed: list[tuple[Path, Path | None]],
) -> list[Path]:
    transaction = lifecycle_transaction_path()
    payload = _validated_pending_lifecycle_transaction(transaction)
    _require_matching_committed_operations(payload, committed)
    with _secure_recovery_mutations(_lifecycle_recovery_paths(transaction, payload)) as mutations:
        _assert_lifecycle_parent_identities(transaction, payload, mutations)
        failures = _secure_cleanup_lifecycle_backups(payload["operations"], mutations)
        if not failures:
            _secure_remove_lifecycle_transaction_metadata(
                lifecycle_transaction_path(),
                mutations,
            )
        return failures


def _cleanup_lifecycle_backups(
    committed: list[tuple[Path, Path | None]],
) -> list[Path]:
    manifest_records = [record for record in committed if record[0] == manifest_path()]
    failures = _cleanup_upgrade_backups([record for record in committed if record[0] != manifest_path()])
    if failures:
        return failures
    return _cleanup_upgrade_backups(manifest_records)


def _mark_lifecycle_transaction_prepared() -> None:
    transaction = lifecycle_transaction_path()
    payload = _read_lifecycle_json(transaction)
    payload["phase"] = "prepared"
    _write_lifecycle_json(transaction, payload)
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is None:
        _fsync_directory(transaction.parent)
    else:
        mutations.sync_parent(transaction)


def _remove_lifecycle_transaction_metadata() -> None:
    transaction = lifecycle_transaction_path()
    evidence = [
        transaction,
        lifecycle_manifest_witness_path(),
        lifecycle_manifest_witness_digest_path(),
    ]
    if not any(path.exists() or path.is_symlink() for path in evidence):
        return
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is None:
        with _secure_recovery_mutations(evidence) as secure_mutations:
            _secure_remove_lifecycle_transaction_metadata(transaction, secure_mutations)
        return
    _secure_remove_lifecycle_transaction_metadata(transaction, mutations)


def _cleanup_upgrade_backups(
    committed: list[
        tuple[Path, Path | None]
        | tuple[Path, Path | None, dict[str, Any] | None]
    ],
) -> list[Path]:
    paths = [backup for record in committed if (backup := record[1]) is not None]
    if not paths:
        return []
    with _secure_recovery_mutations(paths) as mutations:
        return _secure_cleanup_upgrade_backups(committed, mutations)


def _recover_pending_lifecycle_transaction(*, dry_run: bool, announce: bool = True) -> bool:
    transaction = lifecycle_transaction_path()
    if not transaction.exists() and not transaction.is_symlink():
        return False
    payload = _validated_pending_lifecycle_transaction(transaction)
    if dry_run:
        click.echo("Found pending lifecycle transaction; dry-run made no changes")
        return True
    operations = payload["operations"]
    created_parents = [Path(path) for path in payload["created_parents"]]
    cleanup_records = payload["rollback_cleanup_paths"]
    with _secure_recovery_mutations(_lifecycle_recovery_paths(transaction, payload)) as mutations:
        _assert_lifecycle_parent_identities(transaction, payload, mutations)
        if payload["phase"] == "prepared":
            if payload["staging_complete"]:
                _secure_rollback_lifecycle_operations(operations, mutations)
            else:
                _secure_cleanup_local_lifecycle_stages(operations, mutations)
            _secure_remove_created_parents(created_parents, mutations)
            _secure_remove_cleanup_paths(cleanup_records, mutations)
        else:
            failures = _secure_cleanup_lifecycle_backups(operations, mutations)
            if failures:
                raise click.ClickException("Lifecycle transaction backup cleanup failed: " + ", ".join(str(path) for path in failures))
            _secure_cleanup_local_lifecycle_stages(operations, mutations)
        _secure_remove_lifecycle_transaction_metadata(transaction, mutations)
    if announce:
        click.echo(f"Recovered pending {payload['phase']} lifecycle transaction")
    return True


def _validated_pending_lifecycle_transaction(transaction: Path) -> dict[str, Any]:
    try:
        if transaction.is_symlink() or not transaction.is_file():
            raise ValueError("transaction record must be a regular file")
        payload = read_json_file(transaction)
        if not isinstance(payload, dict):
            raise ValueError("transaction record must be an object")
        if payload.get("schema_version") != LIFECYCLE_TRANSACTION_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if payload.get("transaction_type") != "agent-lifecycle":
            raise ValueError("unexpected transaction_type")
        if payload.get("phase") not in {"prepared", "committed"}:
            raise ValueError("unexpected transaction phase")
        staging_complete = payload.get("staging_complete", True)
        if not isinstance(staging_complete, bool):
            raise ValueError("staging_complete must be a boolean")
        if payload["phase"] == "committed" and not staging_complete:
            raise ValueError("committed transaction must have complete staging")
        raw_operations = payload.get("operations")
        if not isinstance(raw_operations, list) or not raw_operations:
            raise ValueError("operations must be a non-empty list")
        operations = _validated_lifecycle_operations(raw_operations)
        if staging_complete and any(
            operation["staged"] is not None and operation["replacement_evidence"] is None
            for operation in operations
        ):
            raise ValueError("complete staging requires replacement evidence for every staged operation")
        journal_parent_identity = _validated_directory_identity(
            payload.get("journal_parent_identity"),
            label="journal_parent_identity",
        )
        authoritative_manifest = _validated_lifecycle_manifest_witness()
        authoritative_paths = _purge_trusted_roots(authoritative_manifest)
        authoritative_roots = _lifecycle_trusted_root_identities(authoritative_paths)
        if payload.get("trusted_roots") != authoritative_roots:
            raise ValueError("recorded trusted root identity changed or does not match the authoritative manifest witness")
        trusted_destinations = _lifecycle_trusted_destinations(
            operations,
            recorded_roots=authoritative_paths,
        )
        for operation in operations:
            destination = Path(operation["destination"])
            backup = operation.get("backup")
            local_staged = operation.get("local_staged")
            if (
                payload["phase"] == "committed"
                and (not isinstance(backup, str) or not (Path(backup).exists() or Path(backup).is_symlink()))
                and (not isinstance(local_staged, str) or not (Path(local_staged).exists() or Path(local_staged).is_symlink()))
            ):
                continue
            owner_root = _lifecycle_trusted_owner(operation, trusted_destinations)
            if owner_root is None:
                raise ValueError(f"destination is not managed by agent lifecycle: {destination}")
            _reject_purge_cleanup_symlink_parents(owner_root, destination.parent)
        created_parents = _validated_created_parents(
            payload.get("created_parents"),
            trusted_destinations,
            operations,
        )
        cleanup_paths = _validated_lifecycle_cleanup_paths(
            payload.get("rollback_cleanup_paths"),
            trusted_destinations,
        )
        return {
            "phase": payload["phase"],
            "staging_complete": staging_complete,
            "journal_parent_identity": journal_parent_identity,
            "operations": operations,
            "created_parents": [str(path) for path in created_parents],
            "rollback_cleanup_paths": cleanup_paths,
            "trusted_roots": authoritative_roots,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Invalid pending lifecycle transaction metadata: {exc}") from exc


def _assert_lifecycle_parent_identities(
    transaction: Path,
    payload: dict[str, Any],
    mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations,
) -> None:
    mutations.assert_parent_identity(
        transaction,
        payload["journal_parent_identity"],
    )
    for operation in payload["operations"]:
        mutations.assert_parent_identity(
            Path(operation["destination"]),
            operation["parent_identity"],
        )
    mutations.verify_all()


def _require_matching_committed_operations(
    payload: dict[str, Any],
    committed: list[tuple[Path, Path | None]],
) -> None:
    expected = [
        (
            Path(operation["destination"]),
            Path(operation["backup"]) if isinstance(operation.get("backup"), str) else None,
        )
        for operation in payload["operations"]
    ]
    if payload["phase"] != "committed" or committed != expected:
        raise click.ClickException("Committed lifecycle operations do not match transaction evidence")


def _validated_lifecycle_operations(raw_operations: list[Any]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    destinations: set[tuple[int, int, str]] = set()
    artifact_locations: set[tuple[int, int, str]] = set()
    backups: set[Path] = set()
    local_stages: set[Path] = set()
    for raw_operation in raw_operations:
        if not isinstance(raw_operation, dict):
            raise ValueError("operation must be an object")
        raw_destination = raw_operation.get("destination")
        raw_staged = raw_operation.get("staged")
        raw_local_staged = raw_operation.get("local_staged")
        raw_backup = raw_operation.get("backup")
        had_destination = raw_operation.get("had_destination")
        relative_name = raw_operation.get("relative_name")
        parent_identity = _validated_directory_identity(
            raw_operation.get("parent_identity"),
            label="operation parent_identity",
        )
        original_evidence = _validated_lifecycle_evidence(
            raw_operation.get("original_evidence"),
            label="operation original_evidence",
        )
        local_staged_evidence = _validated_lifecycle_evidence(
            raw_operation.get("local_staged_evidence"),
            label="operation local_staged_evidence",
            allow_none=True,
        )
        replacement_evidence = _validated_lifecycle_evidence(
            raw_operation.get("replacement_evidence"),
            label="operation replacement_evidence",
            allow_none=True,
        )
        if not isinstance(raw_destination, str):
            raise ValueError("operation destination must be a string")
        if raw_staged is not None and not isinstance(raw_staged, str):
            raise ValueError("operation staged path must be a string or null")
        if raw_local_staged is not None and not isinstance(raw_local_staged, str):
            raise ValueError("operation local_staged path must be a string or null")
        if raw_backup is not None and not isinstance(raw_backup, str):
            raise ValueError("operation backup path must be a string or null")
        if not isinstance(had_destination, bool):
            raise ValueError("operation had_destination must be a boolean")
        if not isinstance(relative_name, str) or relative_name in {"", ".", ".."} or Path(relative_name).name != relative_name:
            raise ValueError("operation relative_name must be one path component")
        destination = _validated_absolute_lexical_path(raw_destination)
        staged = _validated_absolute_lexical_path(raw_staged) if raw_staged is not None else None
        local_staged = _validated_absolute_lexical_path(raw_local_staged) if raw_local_staged is not None else None
        backup = _validated_absolute_lexical_path(raw_backup) if raw_backup is not None else None
        if _normalized_lifecycle_name(relative_name) != _normalized_lifecycle_name(destination.name):
            raise ValueError(f"operation relative_name does not match destination: {destination}")
        destination_location = (
            parent_identity["device"],
            parent_identity["inode"],
            _normalized_lifecycle_name(relative_name),
        )
        if destination_location in destinations:
            raise ValueError(f"duplicate lifecycle destination: {destination}")
        destinations.add(destination_location)
        if had_destination != (backup is not None):
            raise ValueError(f"backup presence does not match destination state: {destination}")
        if had_destination != (original_evidence["kind"] != "absent"):
            raise ValueError(f"original evidence does not match destination state: {destination}")
        if "local_staged" in raw_operation and (staged is None) != (local_staged is None):
            raise ValueError(f"local lifecycle stage presence does not match source stage: {destination}")
        if staged is None and replacement_evidence is not None:
            raise ValueError(f"removal operation cannot carry replacement evidence: {destination}")
        if local_staged is None and local_staged_evidence is not None:
            raise ValueError(
                f"removal operation cannot carry local staging evidence: {destination}"
            )
        if local_staged_evidence == {"kind": "absent"}:
            raise ValueError(
                f"local staging evidence must identify an owned generation: {destination}"
            )
        if local_staged_evidence is not None and replacement_evidence is not None:
            identity_keys = ("kind", "device", "inode", "mode")
            if any(
                local_staged_evidence.get(key) != replacement_evidence.get(key)
                for key in identity_keys
            ):
                raise ValueError(
                    f"local staging generation changed before completion: {destination}"
                )
        if local_staged is not None:
            if local_staged in local_stages or local_staged.parent != destination.parent:
                raise ValueError(f"invalid local lifecycle stage path: {local_staged}")
            prefix = f".{destination.name}.install-"
            suffix = local_staged.name.removeprefix(prefix)
            if not local_staged.name.startswith(prefix) or not suffix or not all(character in "0123456789abcdef" for character in suffix):
                raise ValueError(f"local lifecycle stage name is not transaction-owned: {local_staged}")
            local_stages.add(local_staged)
            artifact_locations.add(
                (
                    parent_identity["device"],
                    parent_identity["inode"],
                    _normalized_lifecycle_name(local_staged.name),
                )
            )
        if backup is not None:
            if backup in backups or backup.parent != destination.parent:
                raise ValueError(f"invalid lifecycle backup path: {backup}")
            prefix = f".{destination.name}.backup-"
            suffix = backup.name.removeprefix(prefix)
            if not backup.name.startswith(prefix) or not suffix or not all(character in "0123456789abcdef" for character in suffix):
                raise ValueError(f"backup name is not transaction-owned: {backup}")
            backups.add(backup)
            artifact_locations.add(
                (
                    parent_identity["device"],
                    parent_identity["inode"],
                    _normalized_lifecycle_name(backup.name),
                )
            )
        operations.append(
            {
                "destination": str(destination),
                "staged": str(staged) if staged is not None else None,
                "local_staged": str(local_staged) if local_staged is not None else None,
                "backup": str(backup) if backup is not None else None,
                "had_destination": had_destination,
                "relative_name": relative_name,
                "parent_identity": parent_identity,
                "original_evidence": original_evidence,
                "local_staged_evidence": local_staged_evidence,
                "replacement_evidence": replacement_evidence,
            }
        )
    if destinations & artifact_locations or backups & local_stages:
        raise ValueError("lifecycle transaction artifact paths overlap")
    return operations


def _lifecycle_authoritative_manifest(
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    for operation in operations:
        if Path(operation["destination"]) != manifest_path():
            continue
        for key in ("staged", "destination", "backup"):
            raw_path = operation.get(key)
            if not isinstance(raw_path, str):
                continue
            candidate = Path(raw_path)
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                manifest = read_json_file(candidate)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(manifest, dict):
                return manifest
    return {
        "targets": {},
        "sdk_bundles": [],
    }


def _write_lifecycle_manifest_witness(manifest: dict[str, Any]) -> None:
    witness = lifecycle_manifest_witness_path()
    digest = lifecycle_manifest_witness_digest_path()
    witness_content = _lifecycle_json_bytes(
        {
            "schema_version": LIFECYCLE_MANIFEST_WITNESS_SCHEMA_VERSION,
            "witness_type": "agent-lifecycle-manifest",
            "manifest": manifest,
        }
    )
    _write_lifecycle_bytes(witness, witness_content)
    _write_lifecycle_bytes(digest, hashlib.sha256(witness_content).hexdigest().encode("ascii") + b"\n")
    mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if mutations is None:
        _fsync_directory(config_dir())
    else:
        mutations.sync_parent(witness)


def _validated_lifecycle_manifest_witness() -> dict[str, Any]:
    witness = lifecycle_manifest_witness_path()
    digest = lifecycle_manifest_witness_digest_path()
    if witness.is_symlink() or not witness.is_file():
        raise ValueError("authoritative manifest witness is unavailable")
    if digest.is_symlink() or not digest.is_file():
        raise ValueError("authoritative manifest witness digest is unavailable")
    expected_sha = digest.read_text(encoding="utf-8").strip()
    if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
        raise ValueError("authoritative manifest witness digest is invalid")
    if sha256_file(witness) != expected_sha:
        raise ValueError("authoritative manifest witness hash mismatch")
    payload = read_json_file(witness)
    if not isinstance(payload, dict):
        raise ValueError("authoritative manifest witness must be an object")
    if payload.get("schema_version") != LIFECYCLE_MANIFEST_WITNESS_SCHEMA_VERSION:
        raise ValueError("unsupported authoritative manifest witness schema_version")
    if payload.get("witness_type") != "agent-lifecycle-manifest":
        raise ValueError("unexpected authoritative manifest witness type")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("authoritative manifest witness must retain a manifest object")
    return manifest


def _lifecycle_trusted_destinations(
    operations: list[dict[str, Any]],
    *,
    recorded_roots: Any = None,
) -> dict[Path, Path]:
    trusted = {
        agent_config_path(): config_dir(),
        manifest_path(): config_dir(),
    }
    if recorded_roots is not None:
        trusted.update(
            _lifecycle_destinations_from_recorded_roots(
                operations,
                recorded_roots,
            )
        )
    candidate_paths: list[Path] = []
    for operation in operations:
        if Path(operation["destination"]) != manifest_path():
            continue
        for key in ("destination", "staged", "backup"):
            raw_path = operation.get(key)
            if isinstance(raw_path, str):
                candidate_paths.append(Path(raw_path))
    for candidate in candidate_paths:
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            manifest = read_json_file(candidate)
            if not isinstance(manifest, dict):
                continue
            trusted.update(_purge_trusted_destinations(manifest, _purge_trusted_roots(manifest)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return trusted


def _lifecycle_trusted_owner(
    operation: dict[str, Any],
    trusted_destinations: dict[Path, Path],
) -> Path | None:
    relative_name = _normalized_lifecycle_name(operation["relative_name"])
    parent_identity = operation["parent_identity"]
    for trusted_destination in trusted_destinations:
        if _normalized_lifecycle_name(trusted_destination.name) != relative_name:
            continue
        try:
            candidate_parent_identity = _path_parent_identity(trusted_destination)
        except OSError:
            continue
        if candidate_parent_identity == parent_identity:
            return Path(operation["destination"]).parent
    return None


def _lifecycle_destinations_from_recorded_roots(
    operations: list[dict[str, Any]],
    recorded_roots: Any,
) -> dict[Path, Path]:
    if not isinstance(recorded_roots, dict):
        raise ValueError("recorded lifecycle trusted roots must be an object")
    config_root = _validated_absolute_alias_path(recorded_roots.get("config_root"))
    if stable_path_location_identity(config_root) != stable_path_location_identity(config_dir()):
        raise ValueError("recorded config root does not match transaction location")
    target_roots = recorded_roots.get("targets")
    if not isinstance(target_roots, dict):
        raise ValueError("recorded lifecycle target roots are invalid")
    directory_roots: set[Path] = set()
    exact_destinations: set[Path] = set()
    for target_id, roots in target_roots.items():
        if target_id not in CONCRETE_TARGETS or not isinstance(roots, dict):
            raise ValueError(f"recorded target identity is invalid: {target_id}")
        for key in ("skills_dir", "agents_dir"):
            root = _optional_recorded_root(roots.get(key))
            if root is not None:
                directory_roots.add(root)
        for key in ("instruction_file", "config_file"):
            destination = _optional_recorded_root(roots.get(key))
            if destination is not None:
                exact_destinations.add(destination)
    sdk_bundles = recorded_roots.get("sdk_bundles")
    if not isinstance(sdk_bundles, list):
        raise ValueError("recorded lifecycle SDK bundle roots are invalid")
    sdk_root = config_root / "sdk-bundles"
    trusted = {
        config_root / "agent.yaml": config_root,
        config_root / "agent-install.json": config_root,
        config_root / "sdk-cache": config_root,
    }
    for bundle in sdk_bundles:
        if not isinstance(bundle, dict) or not isinstance(bundle.get("root"), str):
            raise ValueError("recorded lifecycle SDK bundle root is invalid")
        destination = _validated_absolute_lexical_path(bundle["root"])
        if destination.parent != sdk_root:
            raise ValueError(f"recorded SDK bundle root is outside managed cache: {destination}")
        trusted[destination] = sdk_root
    for operation in operations:
        destination = Path(operation["destination"])
        if destination in exact_destinations:
            trusted[destination] = destination.parent
            continue
        for root in directory_roots:
            if destination.parent == root:
                trusted[destination] = root
                break
    return trusted


def _validated_created_parents(
    raw_paths: Any,
    trusted_destinations: dict[Path, Path],
    operations: list[dict[str, Any]],
) -> list[Path]:
    if not isinstance(raw_paths, list):
        raise ValueError("created_parents must be a list")
    result: list[Path] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str):
            raise ValueError("created parent must be a string")
        path = _validated_absolute_lexical_path(raw_path)
        operation_destinations = [Path(operation["destination"]) for operation in operations]
        if not any(destination.is_relative_to(path) for destination in operation_destinations) and not any(
            destination.is_relative_to(path) for destination in trusted_destinations
        ):
            raise ValueError(f"created parent is outside managed destinations: {path}")
        result.append(path)
    return result


def _validated_lifecycle_cleanup_paths(
    raw_paths: Any,
    trusted_destinations: dict[Path, Path],
) -> list[dict[str, Any]]:
    if not isinstance(raw_paths, list):
        raise ValueError("rollback_cleanup_paths must be a list")
    sdk_root = config_dir() / "sdk-bundles"
    result: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_record in raw_paths:
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "path",
            "evidence",
        }:
            raise ValueError("rollback cleanup record must contain path and evidence")
        raw_path = raw_record.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("rollback cleanup path must be a string")
        path = _validated_absolute_lexical_path(raw_path)
        if path in seen:
            raise ValueError(f"duplicate rollback cleanup path: {path}")
        seen.add(path)
        evidence = _validated_lifecycle_evidence(
            raw_record.get("evidence"),
            label="rollback cleanup evidence",
        )
        if evidence == {"kind": "absent"}:
            raise ValueError("rollback cleanup evidence must identify an owned path")
        if evidence["kind"] != "directory":
            raise ValueError("rollback cleanup evidence must identify a directory")
        if (
            stable_path_location_identity(path.parent) != stable_path_location_identity(sdk_root)
            or path not in trusted_destinations
        ):
            raise ValueError(f"rollback cleanup path is outside managed SDK bundles: {path}")
        _reject_purge_cleanup_symlink_parents(path.parent, path.parent)
        result.append({"path": str(path), "evidence": evidence})
    return result


def _write_pending_purge_cleanup(committed: list[tuple[Path, Path | None]]) -> None:
    retained = [
        (destination, backup)
        for destination, backup in committed
        if backup is not None
    ]
    if not retained:
        return
    manifest_backup = next(
        (
            backup
            for destination, backup in retained
            if destination == manifest_path()
        ),
        None,
    )
    if manifest_backup is None:
        raise click.ClickException("Uninstall committed, but the retained manifest backup is unavailable for recovery")
    recovery_paths = [
        purge_transaction_path(),
        *(backup for _destination, backup in retained),
    ]
    with _secure_recovery_mutations(recovery_paths) as mutations:
        backups: list[dict[str, Any]] = []
        for destination, backup in retained:
            evidence = mutations.path_evidence(backup)
            if evidence == {"kind": "absent"}:
                raise click.ClickException(
                    "Uninstall committed, but a retained backup is unavailable "
                    f"for recovery: {backup}"
                )
            backups.append(
                {
                    "destination": str(destination),
                    "backup": str(backup),
                    "evidence": evidence,
                }
            )
        manifest_evidence = next(
            record["evidence"]
            for record in backups
            if record["destination"] == str(manifest_path())
        )
        if manifest_evidence["kind"] != "file":
            raise click.ClickException(
                "Uninstall committed, but the retained manifest backup is unavailable for recovery"
            )
        try:
            manifest_content = mutations.read_bytes(manifest_backup)
            retained_manifest = json.loads(manifest_content)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise click.ClickException(
                "Uninstall committed, but the retained manifest backup is invalid"
            ) from exc
        if not isinstance(retained_manifest, dict):
            raise click.ClickException(
                "Uninstall committed, but the retained manifest backup is invalid"
            )
        _write_lifecycle_json(
            purge_transaction_path(),
            {
                "schema_version": PURGE_TRANSACTION_SCHEMA_VERSION,
                "transaction_type": "agent-uninstall-purge",
                "phase": "committed",
                "created_at": _now(),
                "backups": backups,
                "manifest_backup_sha256": hashlib.sha256(
                    manifest_content
                ).hexdigest(),
                "trusted_roots": _purge_trusted_roots(retained_manifest),
            },
        )
        mutations.sync_parent(purge_transaction_path())


def _recover_pending_purge_cleanup(*, dry_run: bool, announce: bool = True) -> bool:
    transaction = purge_transaction_path()
    if not transaction.exists() and not transaction.is_symlink():
        return False
    committed = _validated_pending_purge_cleanup(transaction)
    if dry_run:
        click.echo("Found pending committed purge cleanup; dry-run made no changes")
        return True
    recovery_paths = [
        transaction,
        *(
            backup
            for _destination, backup, _evidence in committed
            if backup is not None
        ),
    ]
    with _secure_recovery_mutations(recovery_paths) as mutations:
        _finish_pending_purge_cleanup(committed, recovery_mutations=mutations)
    if announce:
        click.echo("Recovered pending committed purge cleanup")
    return True


def _finish_pending_purge_cleanup(
    committed: list[
        tuple[Path, Path | None, dict[str, Any] | None]
    ]
    | None = None,
    *,
    recovery_mutations: _PosixRecoveryMutations | _WindowsRecoveryMutations | None = None,
) -> None:
    transaction = purge_transaction_path()
    if committed is None:
        committed = _validated_pending_purge_cleanup(transaction)
    manifest_records = [record for record in committed if record[0] == manifest_path()]
    cleanup = (
        (lambda records: _secure_cleanup_upgrade_backups(records, recovery_mutations))
        if recovery_mutations is not None
        else _cleanup_upgrade_backups
    )
    cleanup_failures = cleanup([record for record in committed if record[0] != manifest_path()])
    if cleanup_failures:
        raise click.ClickException(
            "Uninstall committed, but transaction backup cleanup failed: " + ", ".join(str(path) for path in cleanup_failures)
        )
    cleanup_failures = cleanup(manifest_records)
    if cleanup_failures:
        raise click.ClickException(
            "Uninstall committed, but transaction backup cleanup failed: " + ", ".join(str(path) for path in cleanup_failures)
        )
    if recovery_mutations is not None:
        _secure_remove_transaction_metadata(transaction, recovery_mutations)
    else:
        try:
            transaction.unlink()
        except OSError as exc:
            raise click.ClickException(f"Uninstall committed, but transaction metadata cleanup failed: {transaction} ({exc})") from exc


def _purge_backup_artifact_exists(
    backup: Path,
    evidence: dict[str, Any] | None,
) -> bool:
    if backup.exists() or backup.is_symlink():
        return True
    if evidence is None:
        return False
    quarantine = _lifecycle_quarantine_path(
        backup,
        evidence,
        generation_only=False,
    )
    return quarantine.exists() or quarantine.is_symlink()


def _validated_pending_purge_cleanup(
    transaction: Path,
) -> list[tuple[Path, Path | None, dict[str, Any] | None]]:
    try:
        if transaction.is_symlink() or not transaction.is_file():
            raise ValueError("transaction record must be a regular file")
        payload = read_json_file(transaction)
        if not isinstance(payload, dict):
            raise ValueError("transaction record must be an object")
        schema_version = payload.get("schema_version")
        if schema_version not in {
            LEGACY_PURGE_TRANSACTION_SCHEMA_VERSION,
            RECORDED_ROOT_PURGE_TRANSACTION_SCHEMA_VERSION,
            PURGE_TRANSACTION_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported schema_version")
        if payload.get("transaction_type") != "agent-uninstall-purge":
            raise ValueError("unexpected transaction_type")
        if payload.get("phase") != "committed":
            raise ValueError("transaction is not committed")
        records = payload.get("backups")
        if not isinstance(records, list) or not records:
            raise ValueError("backups must be a non-empty list")
        committed: list[
            tuple[Path, Path | None, dict[str, Any] | None]
        ] = []
        parsed_records: list[
            tuple[Path, Path, dict[str, Any] | None]
        ] = []
        destinations: set[Path] = set()
        backups: set[Path] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("backup record must be an object")
            raw_destination = record.get("destination")
            raw_backup = record.get("backup")
            if not isinstance(raw_destination, str) or not isinstance(raw_backup, str):
                raise ValueError("backup paths must be strings")
            destination = _validated_absolute_lexical_path(raw_destination)
            backup = _validated_absolute_lexical_path(raw_backup)
            if destination in destinations or backup in backups:
                raise ValueError("duplicate backup record")
            destinations.add(destination)
            backups.add(backup)
            if backup.parent != destination.parent:
                raise ValueError(f"backup is not a sibling of destination: {backup}")
            prefix = f".{destination.name}.backup-"
            suffix = backup.name.removeprefix(prefix)
            if (
                not backup.name.startswith(prefix)
                or not suffix
                or not suffix.isascii()
                or not all(character.isalnum() or character == "_" for character in suffix)
            ):
                raise ValueError(f"backup name is not transaction-owned: {backup}")
            evidence = (
                _validated_lifecycle_evidence(
                    record.get("evidence"),
                    label="purge backup evidence",
                )
                if schema_version == PURGE_TRANSACTION_SCHEMA_VERSION
                else None
            )
            if evidence == {"kind": "absent"}:
                raise ValueError("purge backup evidence must identify an owned path")
            parsed_records.append((destination, backup, evidence))
        if destinations & backups:
            raise ValueError("backup path is also listed as a destination")
        if not any(
            _purge_backup_artifact_exists(backup, evidence)
            for _destination, backup, evidence in parsed_records
        ):
            return []
        path_records = [
            (destination, backup)
            for destination, backup, _evidence in parsed_records
        ]
        trusted_destinations = (
            _validated_recorded_purge_destinations(payload, parsed_records)
            if schema_version
            in {
                RECORDED_ROOT_PURGE_TRANSACTION_SCHEMA_VERSION,
                PURGE_TRANSACTION_SCHEMA_VERSION,
            }
            else _validated_legacy_purge_destinations(path_records)
        )
        for destination, backup, evidence in parsed_records:
            owner_root = (
                trusted_destinations.get(destination) if trusted_destinations is not None else _purge_destination_owner_root(destination)
            )
            if owner_root is None:
                raise ValueError(f"destination is not managed by agent purge: {destination}")
            _reject_purge_cleanup_symlink_parents(owner_root, backup.parent)
            committed.append((destination, backup, evidence))
        if schema_version == PURGE_TRANSACTION_SCHEMA_VERSION:
            with _secure_recovery_mutations(
                [backup for _destination, backup, _evidence in committed]
            ) as mutations:
                for _destination, backup, evidence in committed:
                    assert evidence is not None
                    quarantine = _lifecycle_quarantine_path(
                        backup,
                        evidence,
                        generation_only=False,
                    )
                    candidate = (
                        backup if mutations.exists(backup) else quarantine
                    )
                    if not mutations.exists(candidate):
                        continue
                    actual = mutations.path_evidence(candidate)
                    if actual != evidence:
                        raise ValueError(
                            "retained backup changed from recorded evidence: "
                            f"{candidate}"
                        )
        return committed
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(f"Invalid pending purge cleanup metadata: {exc}") from exc


def _validated_absolute_lexical_path(raw_path: str) -> Path:
    path = Path(raw_path)
    expanded = _expand_lexical_path(path)
    if not path.is_absolute() or str(path) != str(expanded):
        raise ValueError(f"path must be absolute and normalized: {raw_path}")
    return path


def _validated_absolute_alias_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str):
        raise ValueError("path must be a string")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"path must be absolute: {raw_path}")
    return _expand_lexical_path(path)


def _purge_destination_owner_root(destination: Path) -> Path | None:
    config_root = config_dir()
    if destination in {agent_config_path(), manifest_path(), config_root / "sdk-cache"}:
        return config_root
    sdk_bundles_root = config_root / "sdk-bundles"
    if destination.parent == sdk_bundles_root:
        return sdk_bundles_root
    for target_id in CONCRETE_TARGETS:
        target = resolve_target(target_id)
        if target.skills_dir is not None and destination.parent == _expand_lexical_path(target.skills_dir):
            return _expand_lexical_path(target.skills_dir)
        if target.agents_dir is not None and destination.parent == _expand_lexical_path(target.agents_dir):
            return _expand_lexical_path(target.agents_dir)
        for exact_path in (target.instruction_file, target.config_file):
            if exact_path is not None and destination == _expand_lexical_path(exact_path):
                return destination.parent
    return None


def _validated_legacy_purge_destinations(
    records: list[tuple[Path, Path]],
) -> dict[Path, Path] | None:
    manifest_record = next(
        ((destination, backup) for destination, backup in records if destination == manifest_path()),
        None,
    )
    if manifest_record is None:
        return None
    _manifest_destination, manifest_backup = manifest_record
    if manifest_backup.is_symlink() or not manifest_backup.is_file():
        return None
    try:
        retained_manifest = read_json_file(manifest_backup)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(retained_manifest, dict):
        return None
    trusted_roots = _purge_trusted_roots(retained_manifest)
    return _purge_trusted_destinations(retained_manifest, trusted_roots)


def _purge_trusted_roots(manifest: dict[str, Any]) -> dict[str, Any]:
    targets_payload = manifest.get("targets")
    targets: dict[str, dict[str, str | None]] = {}
    if isinstance(targets_payload, dict):
        for target_id, state in sorted(targets_payload.items()):
            if target_id not in CONCRETE_TARGETS or not isinstance(state, dict):
                continue
            targets[target_id] = {
                key: state.get(key) if isinstance(state.get(key), str) else None
                for key in ("skills_dir", "agents_dir", "instruction_file", "config_file")
            }
    sdk_bundles = [
        {"id": bundle.get("id"), "root": bundle.get("root")}
        for bundle in _manifest_sdk_bundles(manifest)
        if isinstance(bundle.get("id"), str) and isinstance(bundle.get("root"), str)
    ]
    return {
        "config_root": str(config_dir()),
        "targets": targets,
        "sdk_bundles": sdk_bundles,
    }


def _lifecycle_trusted_root_identities(trusted_paths: dict[str, Any]) -> dict[str, Any]:
    config_root = _validated_absolute_alias_path(trusted_paths["config_root"])
    raw_targets = trusted_paths.get("targets")
    raw_bundles = trusted_paths.get("sdk_bundles")
    if not isinstance(raw_targets, dict) or not isinstance(raw_bundles, list):
        raise ValueError("invalid lifecycle trusted path records")
    targets: dict[str, dict[str, Any]] = {}
    for target_id, paths in sorted(raw_targets.items()):
        if target_id not in CONCRETE_TARGETS or not isinstance(paths, dict):
            raise ValueError(f"invalid lifecycle target identity: {target_id}")
        target: dict[str, Any] = {}
        for key in ("skills_dir", "agents_dir"):
            path = _optional_recorded_root(paths.get(key))
            target[key] = None if path is None else {"filesystem_location": _lifecycle_root_identity(path)}
        for key in ("instruction_file", "config_file"):
            path = _optional_recorded_root(paths.get(key))
            target[key] = None if path is None else _lifecycle_relative_location_identity(path)
        targets[target_id] = target
    bundles: list[dict[str, Any]] = []
    for bundle in raw_bundles:
        if not isinstance(bundle, dict) or not isinstance(bundle.get("id"), str) or not isinstance(bundle.get("root"), str):
            raise ValueError("invalid lifecycle SDK bundle identity")
        root = _validated_absolute_lexical_path(bundle["root"])
        bundles.append({"id": bundle["id"], "root": _lifecycle_relative_location_identity(root)})
    return {
        "config_root": {"filesystem_location": _lifecycle_root_identity(config_root)},
        "targets": targets,
        "sdk_bundles": bundles,
    }


def _lifecycle_relative_location_identity(path: Path) -> dict[str, str]:
    return {
        "parent_filesystem_location": _lifecycle_root_identity(path.parent),
        "relative_name": _normalized_lifecycle_name(path.name),
    }


def _lifecycle_root_identity(path: Path) -> str:
    return (
        f"filesystem:{stable_filesystem_identity(path)}"
        if path.exists()
        else stable_path_location_identity(path)
    )


def _validated_recorded_purge_destinations(
    payload: dict[str, Any],
    records: list[tuple[Path, Path, dict[str, Any] | None]],
) -> dict[Path, Path]:
    manifest_record = next(
        (
            (destination, backup, evidence)
            for destination, backup, evidence in records
            if destination == manifest_path()
        ),
        None,
    )
    if manifest_record is None:
        raise ValueError("transaction requires the retained manifest backup")
    _manifest_destination, manifest_backup, manifest_evidence = manifest_record
    manifest_candidate = manifest_backup
    if (
        not manifest_candidate.is_file()
        and manifest_evidence is not None
    ):
        manifest_candidate = _lifecycle_quarantine_path(
            manifest_backup,
            manifest_evidence,
            generation_only=False,
        )
    if manifest_candidate.is_symlink() or not manifest_candidate.is_file():
        raise ValueError("retained manifest backup must be a regular file")
    expected_sha = payload.get("manifest_backup_sha256")
    if (
        not isinstance(expected_sha, str)
        or sha256_file(manifest_candidate) != expected_sha
    ):
        raise ValueError("retained manifest backup hash mismatch")
    retained_manifest = read_json_file(manifest_candidate)
    if not isinstance(retained_manifest, dict):
        raise ValueError("retained manifest backup must contain an object")
    recorded_roots = payload.get("trusted_roots")
    expected_roots = _purge_trusted_roots(retained_manifest)
    if recorded_roots != expected_roots:
        raise ValueError("recorded trusted roots do not match the retained manifest")
    return _purge_trusted_destinations(retained_manifest, expected_roots)


def _purge_trusted_destinations(
    manifest: dict[str, Any],
    trusted_roots: dict[str, Any],
) -> dict[Path, Path]:
    config_root = _validated_absolute_lexical_path(trusted_roots["config_root"])
    if config_root != config_dir():
        raise ValueError("recorded config root does not match transaction location")
    destinations = {
        config_root / "agent.yaml": config_root,
        config_root / "agent-install.json": config_root,
        config_root / "sdk-cache": config_root,
    }
    target_roots = trusted_roots.get("targets")
    target_states = manifest.get("targets")
    if not isinstance(target_roots, dict) or not isinstance(target_states, dict):
        raise ValueError("recorded target roots are invalid")
    for target_id, roots in target_roots.items():
        state = target_states.get(target_id)
        if target_id not in CONCRETE_TARGETS or not isinstance(roots, dict) or not isinstance(state, dict):
            raise ValueError(f"recorded target identity is invalid: {target_id}")
        skills_root = _optional_recorded_root(roots.get("skills_dir"))
        agents_root = _optional_recorded_root(roots.get("agents_dir"))
        instruction_file = _optional_recorded_root(roots.get("instruction_file"))
        config_file = _optional_recorded_root(roots.get("config_file"))
        for raw_path in state.get("installed_paths", []):
            if not isinstance(raw_path, str) or skills_root is None:
                raise ValueError(f"recorded skill destination is invalid for {target_id}")
            destination = _validated_absolute_lexical_path(raw_path)
            if destination.parent != skills_root:
                raise ValueError(f"recorded skill destination is outside trusted root: {destination}")
            destinations[destination] = skills_root
        for record in state.get("agent_roles", []):
            if not isinstance(record, dict) or not isinstance(record.get("dest"), str) or agents_root is None:
                raise ValueError(f"recorded agent role destination is invalid for {target_id}")
            destination = _validated_absolute_lexical_path(record["dest"])
            if destination.parent != agents_root:
                raise ValueError(f"recorded agent role destination is outside trusted root: {destination}")
            destinations[destination] = agents_root
        for block in state.get("managed_blocks", []):
            if not isinstance(block, dict) or not isinstance(block.get("file"), str):
                raise ValueError(f"recorded instruction destination is invalid for {target_id}")
            destination = _validated_absolute_lexical_path(block["file"])
            if instruction_file is None or destination != instruction_file:
                raise ValueError(f"recorded instruction destination does not match trusted identity: {destination}")
            destinations[destination] = destination.parent
        if target_id == "openclaw" and isinstance(state.get("config_file"), str):
            destination = _validated_absolute_lexical_path(state["config_file"])
            if config_file is None or destination != config_file:
                raise ValueError(f"recorded config destination does not match trusted identity: {destination}")
            destinations[destination] = destination.parent
    sdk_bundles = trusted_roots.get("sdk_bundles")
    if not isinstance(sdk_bundles, list):
        raise ValueError("recorded SDK bundle roots are invalid")
    sdk_root = config_root / "sdk-bundles"
    for bundle in sdk_bundles:
        if not isinstance(bundle, dict) or not isinstance(bundle.get("root"), str):
            raise ValueError("recorded SDK bundle root is invalid")
        destination = _validated_absolute_lexical_path(bundle["root"])
        if destination.parent != sdk_root:
            raise ValueError(f"recorded SDK bundle root is outside managed cache: {destination}")
        destinations[destination] = sdk_root
    return destinations


def _optional_recorded_root(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("recorded trusted root must be a path string or null")
    return _validated_absolute_lexical_path(value)


def _reject_purge_cleanup_symlink_parents(owner_root: Path, parent: Path) -> None:
    if not parent.is_relative_to(owner_root):
        raise ValueError(f"backup parent is outside managed root: {parent}")
    candidates = [owner_root]
    relative = parent.relative_to(owner_root)
    current = owner_root
    for part in relative.parts:
        current /= part
        candidates.append(current)
    for candidate in candidates:
        if _path_is_reparse_point(candidate):
            raise ValueError(f"backup parent has symlink or reparse component: {candidate}")


def _path_is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _remove_upgrade_path(path: Path) -> None:
    verified_removal = _ACTIVE_VERIFIED_REMOVAL.get()
    if verified_removal is not None and verified_removal[1] == path:
        mutations, _path, evidence, generation_only, label = verified_removal
        _quarantine_and_remove_evidenced_path(
            mutations,
            path,
            evidence,
            generation_only=generation_only,
            label=label,
        )
        return
    recovery_mutations = _ACTIVE_RECOVERY_MUTATIONS.get()
    if recovery_mutations is not None:
        recovery_mutations.remove(path)
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _remove_managed_skill_dir(target_id: str, path: Path, dry_run: bool) -> bool:
    marker = path / MANAGED_MARKER
    if not marker.exists():
        click.echo(f"{target_id}: skip unmarked path {path}")
        return False
    marker_data = read_json_file(marker)
    if marker_data.get("managed_by") != "open-xquant":
        click.echo(f"{target_id}: skip unmanaged path {path}")
        return False
    if path.is_symlink():
        click.echo(f"{target_id}: skip symlink path {path}")
        return False
    if not dry_run:
        shutil.rmtree(path)
    return True


def _sync_skill_resources(source_dir: Path, dest_dir: Path) -> None:
    """Copy bundled skill resources such as references, scripts, and assets."""
    for source in source_dir.rglob("*"):
        if source.is_symlink():
            raise click.ClickException(f"Refusing symlinked skill resource: {source}")
    for child in list(dest_dir.iterdir()):
        if child.name in {"SKILL.md", MANAGED_MARKER}:
            continue
        if child.is_symlink():
            raise click.ClickException(f"Refusing to remove symlinked skill resource: {child}")
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in source_dir.iterdir():
        if child.name in {"SKILL.md", MANAGED_MARKER}:
            continue
        if child.is_symlink():
            raise click.ClickException(f"Refusing symlinked skill resource: {child}")
        dest = dest_dir / child.name
        if child.is_dir():
            shutil.copytree(child, dest)
        elif child.is_file():
            shutil.copy2(child, dest)


def _replace_managed_skill(
    *,
    source_dir: Path,
    dest_dir: Path,
    content: str,
    target_id: str,
    skill_name: str,
    source_sha: str,
    dest_sha: str,
) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{dest_dir.name}.stage-", dir=dest_dir.parent))
    backup: Path | None = None
    try:
        write_text_file(staged / "SKILL.md", content)
        _sync_skill_resources(source_dir, staged)
        _write_managed_marker(
            staged / MANAGED_MARKER,
            target_id=target_id,
            skill_name=skill_name,
            source_sha=source_sha,
            dest_sha=dest_sha,
        )
        if dest_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{dest_dir.name}.backup-", dir=dest_dir.parent))
            backup.rmdir()
            dest_dir.replace(backup)
        try:
            staged.replace(dest_dir)
        except Exception:
            if backup is not None:
                backup.replace(dest_dir)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        if backup is not None and backup.exists() and not dest_dir.exists():
            backup.replace(dest_dir)


def _remove_deprecated_managed_skill_dirs(target: AgentTarget, dry_run: bool) -> list[str]:
    assert target.skills_dir is not None
    removed_names: list[str] = []
    for name in sorted(DEPRECATED_SKILLS):
        path = target.skills_dir / name
        if not path.exists():
            removed_names.append(name)
            continue
        marker = path / MANAGED_MARKER
        dest = path / "SKILL.md"
        if marker.exists() and dest.exists():
            marker_data = read_json_file(marker)
            if marker_data.get("managed_by") == "open-xquant" and sha256_file(dest) != marker_data.get("dest_sha256"):
                click.echo(f"{target.id}: skip modified deprecated skill {path}")
                continue
        if _remove_managed_skill_dir(target.id, path, dry_run=dry_run):
            removed_names.append(name)
    return removed_names


def _install_agent_roles_for_target(
    target: AgentTarget,
    agent_roles: list[Any],
    source_root: Path,
    dry_run: bool,
    repair: bool,
    existing_records: dict[str, dict[str, Any]],
    skipped: list[str] | None = None,
) -> list[dict[str, Any]]:
    if target.agents_dir is None:
        return []
    by_name = {role.name: role for role in agent_roles}
    records: list[dict[str, Any]] = []
    for name, record in existing_records.items():
        if name in by_name:
            continue
        if _remove_managed_agent_role_file(
            target.id,
            record,
            agents_dir=target.agents_dir,
            dry_run=dry_run,
        ):
            continue
        records.append(record)
    for role in agent_roles:
        filename, content = render_agent_role_for_target(role, target.id)
        existing_record = existing_records.get(role.name)
        dest = (
            _managed_agent_role_dest(target.id, existing_record, agents_dir=target.agents_dir)
            if existing_record
            else _safe_agent_role_dest_file(target, filename)
        )
        if dest is None:
            if skipped is not None:
                skipped.append(role.name)
            records.append(existing_record)
            continue
        if dest.exists() and existing_record is None:
            click.echo(f"{target.id}: skip existing agent role {dest}")
            continue
        if existing_record and dest.exists() and sha256_file(dest) != existing_record.get("dest_sha256"):
            if repair:
                click.echo(f"{target.id}: skip modified managed agent role {dest}")
            if skipped is not None:
                skipped.append(role.name)
            records.append(existing_record)
            continue
        dest_sha = _sha256_text(content)
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_text_file(dest, content)
        records.append(
            {
                "name": role.name,
                "source": str(role.path.relative_to(source_root)),
                "dest": str(dest.resolve()),
                "source_sha256": role.source_sha256,
                "dest_sha256": dest_sha,
            }
        )
    return records


def _remove_managed_agent_role_file(
    target_id: str,
    record: dict[str, Any],
    *,
    agents_dir: object,
    dry_run: bool,
) -> bool:
    dest = _managed_agent_role_dest(target_id, record, agents_dir=agents_dir)
    if dest is None:
        return False
    if not dest.exists():
        return True
    if sha256_file(dest) != record.get("dest_sha256"):
        click.echo(f"{target_id}: skip modified managed agent role {dest}")
        return False
    if not dry_run:
        dest.unlink()
    return True


def _safe_skill_dest_dir(target: AgentTarget, skill_name: str) -> Path:
    assert target.skills_dir is not None
    root = _expand_lexical_path(target.skills_dir)
    dest = _expand_lexical_path(target.skills_dir / skill_name)
    if not dest.is_relative_to(root) or dest.parent != root:
        raise click.ClickException(f"invalid skill name: {skill_name}")
    _reject_symlink_components(target.id, root, dest)
    return dest


def _safe_agent_role_dest_file(target: AgentTarget, filename: str) -> Path:
    if target.agents_dir is None:
        raise click.ClickException(f"{target.id} does not support managed agent roles")
    root = _expand_lexical_path(target.agents_dir)
    dest = _expand_lexical_path(target.agents_dir / filename)
    if not dest.is_relative_to(root) or dest.parent != root:
        raise click.ClickException(f"invalid agent role filename: {filename}")
    if root.is_symlink():
        raise click.ClickException(f"{target.id}: refusing managed agent role path with symlink component: {root}")
    return dest


def _merge_openclaw_config(target: AgentTarget, skill_names: list[str], dry_run: bool) -> None:
    if target.config_file is None or not target.config_file.exists() or dry_run:
        return
    data = _read_json_or_yaml(target.config_file)
    skills = data.setdefault("skills", {})
    if not isinstance(skills, dict):
        return
    entries = skills.setdefault("entries", {})
    if not isinstance(entries, dict):
        return
    for name in skill_names:
        entries.setdefault(name, {})["enabled"] = True
    write_json_file(target.config_file, data)


def _remove_openclaw_config(config_file: Path, skill_names: list[str], dry_run: bool) -> None:
    if not config_file.exists() or dry_run:
        return
    data = _read_json_or_yaml(config_file)
    entries = data.get("skills", {}).get("entries", {}) if isinstance(data.get("skills"), dict) else {}
    if isinstance(entries, dict):
        for name in skill_names:
            entries.pop(name, None)
        write_json_file(config_file, data)


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    try:
        return read_json_file(path)
    except json.JSONDecodeError:
        return read_yaml_file(path)


def _write_managed_marker(
    marker_file: Path,
    target_id: str,
    skill_name: str,
    source_sha: str,
    dest_sha: str,
    managed_tree_sha: str | None = None,
    resources_sha: str | None = None,
) -> None:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "managed_by": "open-xquant",
        "target": target_id,
        "name": skill_name,
        "installed_at": _now(),
        "source_commit": _current_commit(Path.cwd()),
        "source_sha256": source_sha,
        "dest_sha256": dest_sha,
    }
    if managed_tree_sha is not None:
        payload["managed_tree_sha256"] = managed_tree_sha
    if resources_sha is not None:
        payload["resources_sha256"] = resources_sha
    write_json_file(
        marker_file,
        payload,
    )


def _load_manifest() -> dict[str, Any]:
    return read_json_file(manifest_path()) if manifest_path().exists() else {}


def _sdk_bundle_roots() -> set[Path]:
    root = config_dir() / "sdk-bundles"
    if not root.is_dir():
        return set()
    return {_expand_lexical_path(path) for path in root.iterdir() if path.is_dir()}


def _remove_new_sdk_bundle_roots(
    existing: set[Path],
    records: list[dict[str, Any]],
) -> None:
    transaction = lifecycle_transaction_path()
    if transaction.exists() or transaction.is_symlink():
        return
    new_roots = _sdk_bundle_roots() - existing
    owned_records = [
        record for record in records if Path(record["path"]) in new_roots
    ]
    if not owned_records:
        return
    with _secure_recovery_mutations(
        [Path(record["path"]) for record in owned_records]
    ) as mutations:
        _secure_remove_cleanup_paths(owned_records, mutations)


def _require_manifest() -> dict[str, Any]:
    path = manifest_path()
    if not path.exists() and not path.is_symlink():
        raise click.ClickException("Missing manifest. Run `oxq agent install` first.")
    if path.is_symlink() or not path.is_file():
        raise click.ClickException(f"Invalid agent install manifest: {path} must be a regular file")
    try:
        payload = read_json_file(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Invalid agent install manifest: {path} must contain a JSON object") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"Invalid agent install manifest: {path} must contain a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise click.ClickException("Invalid agent install manifest: unsupported schema_version")
    if not isinstance(payload.get("targets"), dict):
        raise click.ClickException("Invalid agent install manifest: targets must be an object")
    return payload


def _load_agent_config() -> dict[str, Any]:
    if not agent_config_path().exists():
        return default_agent_config()
    loaded = read_yaml_file(agent_config_path())
    merged = default_agent_config()
    merged.update(loaded)
    if _should_drop_preferred_runner_argv(
        merged.get("preferred_runner"),
        merged.get("preferred_runner_argv"),
    ):
        merged.pop("preferred_runner_argv", None)
    return merged


def read_recovered_agent_profile() -> str:
    """Recover pending Agent state and return one validated profile snapshot."""

    with agent_lifecycle_lock():
        _recover_pending_lifecycle_transaction(dry_run=False, announce=False)
        _recover_pending_purge_cleanup(dry_run=False, announce=False)
        path = agent_config_path()
        if not path.exists() and not path.is_symlink():
            return AGENT_PROFILE_MULTI
        if path.is_symlink() or not path.is_file():
            raise click.ClickException(f"Invalid agent profile: {path} must be a regular file")
        try:
            import yaml

            raw_loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise click.ClickException(f"Invalid agent profile: cannot read {path}") from exc
        if raw_loaded is None:
            loaded: dict[str, Any] = {}
        elif isinstance(raw_loaded, dict):
            loaded = raw_loaded
        else:
            raise click.ClickException("Invalid agent profile: agent.yaml must contain a mapping")
        schema_version = loaded.get("schema_version", CONFIG_SCHEMA_VERSION)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != CONFIG_SCHEMA_VERSION:
            raise click.ClickException("Invalid agent profile: unsupported schema_version")
        profile = loaded.get("agent_profile", AGENT_PROFILE_MULTI)
        if profile not in AGENT_PROFILES:
            raise click.ClickException("Invalid agent profile: unsupported agent profile value")
        return profile


def _ensure_agent_config(
    dry_run: bool,
    installed_targets: list[str],
    sdk_bundle: dict[str, Any] | None = None,
    agent_profile: str | None = None,
) -> None:
    config = _agent_config_payload(
        installed_targets=installed_targets,
        sdk_bundle=sdk_bundle,
        agent_profile=agent_profile,
    )
    if not dry_run:
        write_yaml_file(agent_config_path(), config)


def _agent_config_payload(
    *,
    installed_targets: list[str],
    sdk_bundle: dict[str, Any] | None = None,
    agent_profile: str | None = None,
) -> dict[str, Any]:
    config = _load_agent_config()
    existing = config.get("installed_targets")
    target_set = set(existing if isinstance(existing, list) else [])
    target_set.update(installed_targets)
    config["installed_targets"] = sorted(target_set)
    if agent_profile is not None:
        config["agent_profile"] = agent_profile
    if sdk_bundle is not None and _should_update_preferred_runner(config.get("preferred_runner")):
        runner = sdk_bundle.get("runner", {})
        if isinstance(runner, dict) and isinstance(runner.get("oxq"), str):
            runner_oxq = runner["oxq"]
            config["preferred_runner"] = _quote_runner_for_shell(runner_oxq)
            argv = runner.get("argv")
            config["preferred_runner_argv"] = [item for item in argv if isinstance(item, str)] if isinstance(argv, list) else [runner_oxq]
    return config


def _stage_agent_state_files(
    staging_root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    operations: list[tuple[Path, Path | None]],
) -> None:
    staged_config = staging_root / "agent.yaml"
    staged_manifest = staging_root / "agent-install.json"
    write_yaml_file(staged_config, config)
    write_json_file(staged_manifest, manifest)
    operations.append((agent_config_path(), staged_config))
    operations.append((manifest_path(), staged_manifest))


def _quote_runner_for_shell(value: str) -> str:
    if _looks_like_windows_runner(value):
        quoted = subprocess.list2cmdline([value])
        return f"& {quoted}" if sys.platform == "win32" else quoted
    return shlex.quote(value)


def _looks_like_windows_runner(value: str) -> bool:
    has_drive = len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"}
    return sys.platform == "win32" or has_drive or ("\\" in value and value.lower().endswith(".exe"))


def _should_update_preferred_runner(value: Any) -> bool:
    if value in (None, "", "uv run oxq"):
        return True
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith("uv run --project ") and normalized.endswith(" oxq"):
        return True
    return "/sdk-bundles/" in normalized


def _should_drop_preferred_runner_argv(preferred_runner: Any, argv: Any) -> bool:
    if _should_update_preferred_runner(preferred_runner):
        return False
    return argv == ["uv", "run", "oxq"] or _runner_argv_points_to_sdk_bundle(argv)


def _runner_argv_points_to_sdk_bundle(argv: Any) -> bool:
    if not isinstance(argv, list):
        return False
    return any(isinstance(item, str) and "/sdk-bundles/" in item.replace("\\", "/") for item in argv)


def _record_sdk_bundle(manifest: dict[str, Any], sdk_bundle: dict[str, Any]) -> None:
    bundles = _manifest_sdk_bundles(manifest)
    bundles.append(sdk_bundle)
    manifest["sdk_bundle"] = sdk_bundle
    manifest["sdk_bundles"] = _dedupe_sdk_bundles(bundles)


def _manifest_sdk_bundles(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    current = manifest.get("sdk_bundle")
    if isinstance(current, dict):
        bundles.append(current)
    historical = manifest.get("sdk_bundles")
    if isinstance(historical, list):
        bundles.extend(bundle for bundle in historical if isinstance(bundle, dict))
    return _dedupe_sdk_bundles(bundles)


def _validated_purge_sdk_bundles(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    current = manifest.get("sdk_bundle")
    if current is not None and not isinstance(current, dict):
        raise click.ClickException("Invalid agent install manifest: sdk_bundle must be an object")
    historical = manifest.get("sdk_bundles")
    if historical is not None:
        if not isinstance(historical, list):
            raise click.ClickException("Invalid agent install manifest: sdk_bundles must be a list")
        if not all(isinstance(bundle, dict) for bundle in historical):
            raise click.ClickException("Invalid agent install manifest: sdk_bundles must contain only objects")
    return _manifest_sdk_bundles(manifest)


def _dedupe_sdk_bundles(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bundle in bundles:
        key = _bundle_label(bundle)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bundle)
    return deduped


def _bundle_label(bundle: dict[str, Any]) -> str:
    root = bundle.get("root")
    if isinstance(root, str) and root:
        return root
    bundle_id = bundle.get("id")
    return str(bundle_id) if bundle_id else "<unknown-sdk-bundle>"


def _status_payload() -> dict[str, Any]:
    with agent_lifecycle_lock():
        return _status_payload_locked()


def _status_payload_locked() -> dict[str, Any]:
    _recover_pending_lifecycle_transaction(dry_run=False, announce=False)
    _recover_pending_purge_cleanup(dry_run=False, announce=False)
    manifest = _load_manifest()
    targets_payload: dict[str, Any] = {}
    targets = manifest.get("targets", {}) if isinstance(manifest.get("targets"), dict) else {}
    for target_id, state in targets.items():
        if not isinstance(state, dict):
            continue
        skills = state.get("skills", []) if isinstance(state.get("skills"), list) else []
        agent_roles = state.get("agent_roles", []) if isinstance(state.get("agent_roles"), list) else []
        present = 0
        for record in skills:
            if isinstance(record, dict) and expand_path(record["dest"]).exists():
                present += 1
        present_roles = 0
        for record in agent_roles:
            if isinstance(record, dict) and expand_path(record["dest"]).exists():
                present_roles += 1
        targets_payload[target_id] = {
            "installed": bool(state.get("installed")),
            "agent_profile": state.get("agent_profile") or manifest.get("agent_profile"),
            "skills": {"installed": present, "expected": len(skills)},
            "agent_roles": {"installed": present_roles, "expected": len(agent_roles)},
            "missing_paths": [record["dest"] for record in skills if isinstance(record, dict) and not expand_path(record["dest"]).exists()]
            + [record["dest"] for record in agent_roles if isinstance(record, dict) and not expand_path(record["dest"]).exists()],
            "instruction_block": _instruction_block_state(state),
            "commit": manifest.get("source", {}).get("commit") if isinstance(manifest.get("source"), dict) else None,
        }
    return {
        "status": "ok" if targets_payload else "missing",
        "agent_profile": manifest.get("agent_profile"),
        "config": str(agent_config_path()),
        "manifest": str(manifest_path()),
        "targets": targets_payload,
    }


def _instruction_block_state(state: dict[str, Any]) -> str:
    blocks = state.get("managed_blocks", [])
    if not blocks:
        return "not-applicable"
    for block in blocks:
        path = expand_path(block["file"])
        if not path.exists() or f"{block['marker']}:begin" not in path.read_text(encoding="utf-8"):
            return "missing"
    return "present"


def _skill_names(state: dict[str, Any]) -> list[str]:
    return [record["name"] for record in state.get("skills", []) if isinstance(record, dict) and "name" in record]


def _source_metadata(source_root: Path, source_type: str) -> dict[str, Any]:
    return {
        "type": source_type,
        "repo": "xingwudao/open-xquant",
        "ref": "main",
        "commit": _current_commit(source_root),
        "path": str(source_root.resolve()),
    }


def _current_commit(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _upgrade_source(from_local: str | None, repo: str, git_ref: str) -> Path:
    if from_local:
        return resolve_source_root(from_local)
    cache_root = config_dir() / "cache" / "open-xquant"
    cache_key = hashlib.sha256(f"{repo}\0{git_ref}".encode()).hexdigest()[:16]
    cache = (cache_root / cache_key).resolve()
    if not cache.is_relative_to(cache_root.resolve()):
        raise click.ClickException("Invalid upgrade cache path")
    if cache.exists():
        shutil.rmtree(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", git_ref, repo, str(cache)], check=True)
    return resolve_source_root(str(cache))


def _sha256_text(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _print_generic() -> None:
    click.echo("Install these skills into your Agent's SKILL.md directory:")
    click.echo("agent/skills/<name>/SKILL.md")
    click.echo("")
    click.echo(GENERIC_AGENT_BLOCK)
