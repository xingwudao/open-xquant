"""Agent lifecycle commands for installing open-xquant skills."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
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

MANAGED_MARKER = ".open-xquant-managed.json"
CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
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
  - strategy builder worker
  - data inspection worker
  - spec audit worker
  - runtime audit worker
  - backtest runner worker
  - monitor/report worker
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

    target_ids = _select_targets(target, all_targets)
    if target_ids == ["generic"]:
        _print_generic()
        _ensure_agent_config(dry_run=dry_run, installed_targets=[])
        return
    selected_profile = _select_agent_profile(agent_profile, target_ids, yes=yes)

    source_root = resolve_source_root(from_local)
    skills = _filter_skills_for_profile(_discover_skills_or_raise(source_root), selected_profile)
    agent_roles = _filter_agent_roles_for_profile(_discover_agent_roles_or_raise(source_root), selected_profile)
    sdk_bundle = build_sdk_bundle(source_root, config_dir(), dry_run=dry_run)
    manifest = _load_manifest()
    now = _now()
    manifest.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("installed_at", now)
    manifest["updated_at"] = now
    manifest["source"] = _source_metadata(source_root, "local")
    manifest["agent_profile"] = selected_profile
    _record_sdk_bundle(manifest, sdk_bundle)
    manifest.setdefault("targets", {})

    installed: list[str] = []
    for target_id in target_ids:
        target_obj = resolve_target(target_id)
        existing_state = manifest["targets"].get(target_id) if isinstance(manifest.get("targets"), dict) else None
        target_state = _install_target(
            target_obj,
            skills,
            agent_roles,
            source_root,
            dry_run=dry_run,
            repair=repair,
            existing_state=existing_state if isinstance(existing_state, dict) else None,
            agent_profile=selected_profile,
        )
        manifest["targets"][target_id] = target_state
        installed.append(target_id)

    _ensure_agent_config(
        dry_run=dry_run,
        installed_targets=installed,
        sdk_bundle=sdk_bundle,
        agent_profile=selected_profile,
    )
    if not dry_run:
        write_json_file(manifest_path(), manifest)
    click.echo(f"Installed open-xquant agent support ({selected_profile}): " + ", ".join(installed))


@agent.command()
@click.option("--target", type=click.Choice(CONCRETE_TARGETS), default=None)
@click.option("--all-targets", is_flag=True, help="Uninstall every manifest target.")
@click.option("--dry-run", is_flag=True)
@click.option("--purge-config", is_flag=True)
@click.option("--yes", is_flag=True)
def uninstall(target: str | None, all_targets: bool, dry_run: bool, purge_config: bool, yes: bool) -> None:
    """Uninstall managed Agent skills."""

    del yes
    if all_targets and target:
        raise click.ClickException("Use --target or --all-targets, not both.")
    if target is None and not all_targets:
        raise click.ClickException("Use --target or --all-targets to uninstall managed Agent files.")
    manifest = _require_manifest()
    targets = manifest.get("targets", {})
    selected = list(targets) if all_targets else [target]
    bundles_to_purge: list[dict[str, Any]] = []
    if not dry_run and purge_config and all_targets:
        bundles_to_purge = _manifest_sdk_bundles(manifest)
        active_bundles = [
            _bundle_label(bundle)
            for bundle in bundles_to_purge
            if sdk_bundle_contains_active_runner(bundle, config_dir())
        ]
        if active_bundles:
            raise click.ClickException(
                "Refusing to purge config while running from the active cached SDK runner: "
                + ", ".join(active_bundles)
                + ". Re-run this command from a non-cached open-xquant checkout or installed Python environment."
            )
        failed = [
            _bundle_label(bundle)
            for bundle in bundles_to_purge
            if not sdk_bundle_can_be_removed(bundle, config_dir())
        ]
        if failed:
            raise click.ClickException(
                "Refusing to purge config because SDK bundle removal was not verified: "
                + ", ".join(failed)
            )
    for target_id in selected:
        state = targets.get(target_id)
        if not isinstance(state, dict) or not state.get("installed"):
            click.echo(f"{target_id}: not installed")
            continue
        _uninstall_target(target_id, state, dry_run=dry_run)
        if not dry_run:
            state["installed"] = False
            state["updated_at"] = _now()
    if not dry_run:
        if purge_config and all_targets:
            failed = [
                _bundle_label(bundle)
                for bundle in bundles_to_purge
                if not remove_sdk_bundle(bundle, config_dir())
            ]
            if failed:
                manifest["updated_at"] = _now()
                write_json_file(manifest_path(), manifest)
                raise click.ClickException(
                    "Refusing to purge config because SDK bundle removal was not verified: "
                    + ", ".join(failed)
                )
            sdk_cache = config_dir() / "sdk-cache"
            if sdk_cache.exists():
                shutil.rmtree(sdk_cache)
            if agent_config_path().exists():
                agent_config_path().unlink()
            if manifest_path().exists():
                manifest_path().unlink()
        else:
            if purge_config and agent_config_path().exists():
                agent_config_path().unlink()
            manifest["updated_at"] = _now()
            write_json_file(manifest_path(), manifest)
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
        click.echo(
            "Skills: "
            f"{target_state['skills']['installed']}/{target_state['skills']['expected']}"
        )
        click.echo(
            "Agent roles: "
            f"{target_state['agent_roles']['installed']}/{target_state['agent_roles']['expected']}"
        )
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
    sdk_bundle = build_sdk_bundle(source_root, config_dir(), dry_run=dry_run)
    updated: list[str] = []
    target_profiles: dict[str, str] = {}
    for target_id in upgrade_ids:
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
        target_obj = resolve_target(target_id)
        skipped = _upgrade_target(
            target_obj,
            state,
            skills,
            agent_roles,
            source_root,
            dry_run=dry_run,
            agent_profile=selected_profile,
        )
        state["agent_profile"] = selected_profile
        target_profiles[target_id] = selected_profile
        updated.append(target_id)
        if skipped:
            click.echo(f"{target_id}: skipped modified managed files: {', '.join(skipped)}")
    config_profile = agent_profile
    display_profile = _upgrade_display_profile(agent_profile, target_profiles)
    if not dry_run:
        manifest["updated_at"] = _now()
        manifest["source"] = _source_metadata(source_root, "local" if from_local else "git")
        if agent_profile is not None:
            manifest["agent_profile"] = agent_profile
        _record_sdk_bundle(manifest, sdk_bundle)
        write_json_file(manifest_path(), manifest)
    _ensure_agent_config(
        dry_run=dry_run,
        installed_targets=updated,
        sdk_bundle=sdk_bundle,
        agent_profile=config_profile,
    )
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
    click.echo("Choose how OpenXQuant skills should be installed for this machine.")
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
            dest_dir.mkdir(parents=True, exist_ok=True)
            write_text_file(dest_file, content)
            _write_managed_marker(
                marker_file,
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


def _uninstall_target(target_id: str, state: dict[str, Any], dry_run: bool) -> None:
    for raw_path in state.get("installed_paths", []):
        _remove_managed_skill_dir(target_id, expand_path(raw_path), dry_run=dry_run)
    for record in state.get("agent_roles", []):
        if isinstance(record, dict):
            _remove_managed_agent_role_file(target_id, record, dry_run=dry_run)
    for block in state.get("managed_blocks", []):
        try:
            if not dry_run:
                remove_marker_block(expand_path(block["file"]), block["marker"])
        except MarkerBlockError as exc:
            raise click.ClickException(str(exc)) from exc
    if target_id == "openclaw" and state.get("config_file"):
        _remove_openclaw_config(expand_path(state["config_file"]), _skill_names(state), dry_run=dry_run)


def _upgrade_target(
    target: AgentTarget,
    state: dict[str, Any],
    skills: list[Any],
    agent_roles: list[Any],
    source_root: Path,
    dry_run: bool,
    agent_profile: str,
) -> list[str]:
    assert target.skills_dir is not None
    by_name = {skill.name: skill for skill in skills}
    old_records: dict[str, dict[str, Any]] = {
        record["name"]: record
        for record in state.get("skills", [])
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    skipped: list[str] = []
    new_skill_records: list[dict[str, Any]] = []
    removed_names: list[str] = []
    for name, record in old_records.items():
        if name in by_name:
            continue
        dest = expand_path(record["dest"]) if isinstance(record.get("dest"), str) else None
        if dest is not None and dest.exists() and sha256_file(dest) != record.get("dest_sha256"):
            skipped.append(name)
            new_skill_records.append(record)
            continue
        if _remove_managed_skill_dir(target.id, expand_path(record["dest"]).parent, dry_run=dry_run):
            removed_names.append(name)
    removed_names.extend(_remove_deprecated_managed_skill_dirs(target, dry_run=dry_run))
    for source_skill in skills:
        name = source_skill.name
        existing_record = old_records.get(name)
        dest = (
            expand_path(existing_record["dest"])
            if existing_record and isinstance(existing_record.get("dest"), str)
            else _safe_skill_dest_dir(target, name) / "SKILL.md"
        )
        marker = dest.parent / MANAGED_MARKER
        if dest.parent.exists() and not marker.exists():
            click.echo(f"{target.id}: skip unmarked existing skill {dest.parent}")
            continue
        if existing_record and dest.exists() and sha256_file(dest) != existing_record.get("dest_sha256"):
            skipped.append(name)
            new_skill_records.append(existing_record)
            continue
        content = _render_skill_for_target_and_profile(source_skill, target.id, agent_profile)
        dest_sha = _sha256_text(content)
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_text_file(dest, content)
            _write_managed_marker(
                dest.parent / MANAGED_MARKER,
                target_id=target.id,
                skill_name=name,
                source_sha=source_skill.source_sha256,
                dest_sha=dest_sha,
            )
        new_skill_records.append(
            {
                "name": name,
                "source": str(source_skill.path.relative_to(source_root)),
                "dest": str(dest.resolve()),
                "source_sha256": source_skill.source_sha256,
                "dest_sha256": dest_sha,
            }
        )
    if target.instruction_file is not None and not dry_run:
        content = _instruction_block_for_target(target.id, agent_profile)
        upsert_marker_block(target.instruction_file, "open-xquant", content)
    if target.id == "openclaw":
        if removed_names:
            _remove_openclaw_config(expand_path(state["config_file"]), removed_names, dry_run=dry_run)
        _merge_openclaw_config(target, [record["name"] for record in new_skill_records], dry_run=dry_run)
    if not dry_run:
        state["skills"] = new_skill_records
        state["installed_paths"] = [str(expand_path(record["dest"]).parent) for record in new_skill_records]
        state["updated_at"] = _now()
    old_role_records: dict[str, dict[str, Any]] = {
        record["name"]: record
        for record in state.get("agent_roles", [])
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }
    if target.id in ROLE_TARGETS and target.agents_dir is not None:
        new_role_records = _install_agent_roles_for_target(
            target,
            agent_roles,
            source_root,
            dry_run=dry_run,
            repair=False,
            existing_records=old_role_records,
            skipped=skipped,
        )
        if not dry_run:
            state["agent_roles"] = new_role_records
            state["agents_dir"] = str(target.agents_dir.resolve())
    elif not dry_run:
        state["agent_roles"] = []
        state["agents_dir"] = str(target.agents_dir.resolve()) if target.agents_dir else None
    return skipped


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
        if _remove_managed_agent_role_file(target.id, record, dry_run=dry_run):
            continue
        records.append(record)
    for role in agent_roles:
        filename, content = render_agent_role_for_target(role, target.id)
        existing_record = existing_records.get(role.name)
        dest = (
            expand_path(existing_record["dest"])
            if existing_record and isinstance(existing_record.get("dest"), str)
            else _safe_agent_role_dest_file(target, filename)
        )
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


def _remove_managed_agent_role_file(target_id: str, record: dict[str, Any], dry_run: bool) -> bool:
    raw_dest = record.get("dest")
    if not isinstance(raw_dest, str):
        return False
    dest = expand_path(raw_dest)
    if not dest.exists():
        return True
    if dest.is_symlink():
        click.echo(f"{target_id}: skip symlink agent role {dest}")
        return False
    if sha256_file(dest) != record.get("dest_sha256"):
        click.echo(f"{target_id}: skip modified managed agent role {dest}")
        return False
    if not dry_run:
        dest.unlink()
    return True


def _safe_skill_dest_dir(target: AgentTarget, skill_name: str) -> Path:
    assert target.skills_dir is not None
    root = target.skills_dir.resolve()
    dest = (target.skills_dir / skill_name).resolve()
    if not dest.is_relative_to(root):
        raise click.ClickException(f"invalid skill name: {skill_name}")
    return dest


def _safe_agent_role_dest_file(target: AgentTarget, filename: str) -> Path:
    if target.agents_dir is None:
        raise click.ClickException(f"{target.id} does not support managed agent roles")
    root = target.agents_dir.resolve()
    dest = (target.agents_dir / filename).resolve()
    if not dest.is_relative_to(root):
        raise click.ClickException(f"invalid agent role filename: {filename}")
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


def _write_managed_marker(marker_file: Path, target_id: str, skill_name: str, source_sha: str, dest_sha: str) -> None:
    write_json_file(
        marker_file,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "managed_by": "open-xquant",
            "target": target_id,
            "name": skill_name,
            "installed_at": _now(),
            "source_commit": _current_commit(Path.cwd()),
            "source_sha256": source_sha,
            "dest_sha256": dest_sha,
        },
    )


def _load_manifest() -> dict[str, Any]:
    return read_json_file(manifest_path()) if manifest_path().exists() else {}


def _require_manifest() -> dict[str, Any]:
    if not manifest_path().exists():
        raise click.ClickException("Missing manifest. Run `oxq agent install` first.")
    return read_json_file(manifest_path())


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


def _ensure_agent_config(
    dry_run: bool,
    installed_targets: list[str],
    sdk_bundle: dict[str, Any] | None = None,
    agent_profile: str | None = None,
) -> None:
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
    if not dry_run:
        write_yaml_file(agent_config_path(), config)


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
            "missing_paths": [
                record["dest"]
                for record in skills
                if isinstance(record, dict) and not expand_path(record["dest"]).exists()
            ]
            + [
                record["dest"]
                for record in agent_roles
                if isinstance(record, dict) and not expand_path(record["dest"]).exists()
            ],
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
