"""Agent lifecycle commands for installing open-xquant skills."""

from __future__ import annotations

import json
import shutil
import subprocess
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
    SUPPORTED_TARGETS,
    AgentTarget,
    detect_targets,
    discover_skills,
    render_skill_for_target,
    resolve_source_root,
    resolve_target,
)

MANAGED_MARKER = ".open-xquant-managed.json"
CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


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
        "default_data_dir": "~/.oxq/data",
        "auto_init_workspace": True,
        "allow_auto_download": "ask",
        "preferred_runner": "uv run oxq",
    }


GLOBAL_AGENT_BLOCK = """## open-xquant

When the user asks about quant strategy, backtest, factor evaluation,
parameter tuning, audit, robustness, report, broker connectivity, or live
trading, use the installed open-xquant skills.

If the current directory has no `.open-xquant/workspace.yaml`, run
`oxq research init` before creating strategy artifacts.

Default workflow:
`strategy_spec.yaml` -> validate -> backtest -> audit -> robustness -> report."""


CLAUDE_AGENT_BLOCK = """## open-xquant

When the user asks about quant strategy, backtest, factor evaluation,
parameter tuning, audit, robustness, report, broker connectivity, or live
trading, use the installed open-xquant skills.

If the current directory has no `.open-xquant/workspace.yaml`, run
`oxq research init` before creating strategy artifacts.

If this project has an `AGENTS.md`, also read it when it is relevant to
open-xquant work."""


@click.group()
def agent() -> None:
    """Manage long-lived Agent integration for open-xquant."""


@agent.command()
@click.option("--target", type=click.Choice(SUPPORTED_TARGETS), default=None)
@click.option("--all-targets", is_flag=True, help="Install every supported concrete target.")
@click.option("--from-local", "from_local", default=None, help="Path to an open-xquant checkout.")
@click.option("--dry-run", is_flag=True, help="Show planned writes without changing files.")
@click.option("--repair", is_flag=True, help="Reinstall missing managed files.")
@click.option("--yes", is_flag=True, help="Run non-interactively.")
def install(target: str | None, all_targets: bool, from_local: str | None, dry_run: bool, repair: bool, yes: bool) -> None:
    """Install open-xquant skills into supported Agent homes."""

    del repair, yes
    target_ids = _select_targets(target, all_targets)
    if target_ids == ["generic"]:
        _print_generic()
        _ensure_agent_config(dry_run=dry_run, installed_targets=[])
        return

    source_root = resolve_source_root(from_local)
    skills = discover_skills(source_root)
    manifest = _load_manifest()
    now = _now()
    manifest.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("installed_at", now)
    manifest["updated_at"] = now
    manifest["source"] = _source_metadata(source_root, "local")
    manifest.setdefault("targets", {})

    installed: list[str] = []
    for target_id in target_ids:
        target_obj = resolve_target(target_id)
        target_state = _install_target(target_obj, skills, source_root, dry_run=dry_run)
        manifest["targets"][target_id] = target_state
        installed.append(target_id)

    _ensure_agent_config(dry_run=dry_run, installed_targets=installed)
    if not dry_run:
        write_json_file(manifest_path(), manifest)
    click.echo("Installed open-xquant agent support: " + ", ".join(installed))


@agent.command()
@click.option("--target", type=click.Choice(CONCRETE_TARGETS), default=None)
@click.option("--all-targets", is_flag=True, help="Uninstall every manifest target.")
@click.option("--dry-run", is_flag=True)
@click.option("--purge-config", is_flag=True)
@click.option("--yes", is_flag=True)
def uninstall(target: str | None, all_targets: bool, dry_run: bool, purge_config: bool, yes: bool) -> None:
    """Uninstall managed Agent skills."""

    del yes
    manifest = _require_manifest()
    targets = manifest.get("targets", {})
    selected = list(targets) if all_targets or target is None else [target]
    for target_id in selected:
        state = targets.get(target_id)
        if not isinstance(state, dict) or not state.get("installed"):
            click.echo(f"{target_id}: not installed")
            continue
        _uninstall_target(target_id, state, dry_run=dry_run)
        if not dry_run:
            state["installed"] = False
            state["updated_at"] = _now()
    if purge_config and not dry_run and agent_config_path().exists():
        agent_config_path().unlink()
    if not dry_run:
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
        click.echo(f"Instruction block: {target_state['instruction_block']}")
        click.echo(f"Commit: {target_state.get('commit') or 'unknown'}")


@agent.command()
@click.option("--target", type=click.Choice(CONCRETE_TARGETS), default=None)
@click.option("--all-targets", is_flag=True)
@click.option("--from-local", "from_local", default=None)
@click.option("--repo", default="https://github.com/xingwudao/open-xquant")
@click.option("--ref", "git_ref", default="main")
@click.option("--dry-run", is_flag=True)
@click.option("--yes", is_flag=True)
def upgrade(
    target: str | None,
    all_targets: bool,
    from_local: str | None,
    repo: str,
    git_ref: str,
    dry_run: bool,
    yes: bool,
) -> None:
    """Upgrade managed Agent skills from a local checkout or GitHub ref."""

    del yes
    manifest = _require_manifest()
    source_root = _upgrade_source(from_local, repo, git_ref)
    skills = discover_skills(source_root)
    targets = manifest.get("targets", {})
    selected = list(targets) if all_targets or target is None else [target]
    updated: list[str] = []
    for target_id in selected:
        state = targets.get(target_id)
        if not isinstance(state, dict) or not state.get("installed"):
            click.echo(f"{target_id}: not installed")
            continue
        target_obj = resolve_target(target_id)
        skipped = _upgrade_target(target_obj, state, skills, source_root, dry_run=dry_run)
        updated.append(target_id)
        if skipped:
            click.echo(f"{target_id}: skipped modified skills: {', '.join(skipped)}")
    if not dry_run:
        manifest["updated_at"] = _now()
        manifest["source"] = _source_metadata(source_root, "local" if from_local else "git")
        write_json_file(manifest_path(), manifest)
    click.echo("Upgrade complete: " + ", ".join(updated))


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


def _install_target(target: AgentTarget, skills: list[Any], source_root: Path, dry_run: bool) -> dict[str, Any]:
    if target.id == "generic":
        raise click.ClickException("generic target does not install files.")
    assert target.skills_dir is not None
    target_skills: list[dict[str, Any]] = []
    installed_paths: list[str] = []
    for skill in skills:
        content = render_skill_for_target(skill, target.id)
        dest_dir = target.skills_dir / skill.name
        dest_file = dest_dir / "SKILL.md"
        marker_file = dest_dir / MANAGED_MARKER
        if dest_dir.exists() and not marker_file.exists():
            click.echo(f"{target.id}: skip unmarked existing skill {dest_dir}")
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
    managed_blocks = []
    if target.instruction_file is not None:
        content = CLAUDE_AGENT_BLOCK if target.id == "claude-code" else GLOBAL_AGENT_BLOCK
        if not dry_run:
            upsert_marker_block(target.instruction_file, "open-xquant", content)
        managed_blocks.append({"file": str(target.instruction_file.resolve()), "marker": "open-xquant"})
    if target.id == "openclaw":
        _merge_openclaw_config(target, [skill["name"] for skill in target_skills], dry_run=dry_run)
    return {
        "installed": True,
        "installed_at": _now(),
        "updated_at": _now(),
        "skills_dir": str(target.skills_dir.resolve()),
        "instruction_file": str(target.instruction_file.resolve()) if target.instruction_file else None,
        "config_file": str(target.config_file.resolve()) if target.config_file else None,
        "installed_paths": installed_paths,
        "managed_blocks": managed_blocks,
        "skills": target_skills,
    }


def _uninstall_target(target_id: str, state: dict[str, Any], dry_run: bool) -> None:
    for raw_path in state.get("installed_paths", []):
        path = expand_path(raw_path)
        marker = path / MANAGED_MARKER
        if not marker.exists():
            click.echo(f"{target_id}: skip unmarked path {path}")
            continue
        marker_data = read_json_file(marker)
        if marker_data.get("managed_by") != "open-xquant":
            click.echo(f"{target_id}: skip unmanaged path {path}")
            continue
        if path.is_symlink():
            click.echo(f"{target_id}: skip symlink path {path}")
            continue
        if not dry_run:
            shutil.rmtree(path)
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
    source_root: Path,
    dry_run: bool,
) -> list[str]:
    assert target.skills_dir is not None
    by_name = {skill.name: skill for skill in skills}
    skipped: list[str] = []
    new_skill_records: list[dict[str, Any]] = []
    for record in state.get("skills", []):
        name = record["name"]
        source_skill = by_name.get(name)
        if source_skill is None:
            continue
        dest = expand_path(record["dest"])
        if dest.exists() and sha256_file(dest) != record.get("dest_sha256"):
            skipped.append(name)
            new_skill_records.append(record)
            continue
        content = render_skill_for_target(source_skill, target.id)
        dest_sha = _sha256_text(content)
        if not dry_run:
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
        content = CLAUDE_AGENT_BLOCK if target.id == "claude-code" else GLOBAL_AGENT_BLOCK
        upsert_marker_block(target.instruction_file, "open-xquant", content)
    if target.id == "openclaw":
        _merge_openclaw_config(target, [record["name"] for record in new_skill_records], dry_run=dry_run)
    if not dry_run:
        state["skills"] = new_skill_records
        state["updated_at"] = _now()
    return skipped


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
    return merged


def _ensure_agent_config(dry_run: bool, installed_targets: list[str]) -> None:
    config = _load_agent_config()
    existing = config.get("installed_targets")
    target_set = set(existing if isinstance(existing, list) else [])
    target_set.update(installed_targets)
    config["installed_targets"] = sorted(target_set)
    if not dry_run:
        write_yaml_file(agent_config_path(), config)


def _status_payload() -> dict[str, Any]:
    manifest = _load_manifest()
    targets_payload: dict[str, Any] = {}
    targets = manifest.get("targets", {}) if isinstance(manifest.get("targets"), dict) else {}
    for target_id, state in targets.items():
        if not isinstance(state, dict):
            continue
        skills = state.get("skills", []) if isinstance(state.get("skills"), list) else []
        present = 0
        for record in skills:
            if isinstance(record, dict) and expand_path(record["dest"]).exists():
                present += 1
        targets_payload[target_id] = {
            "installed": bool(state.get("installed")),
            "skills": {"installed": present, "expected": len(skills)},
            "missing_paths": [
                record["dest"]
                for record in skills
                if isinstance(record, dict) and not expand_path(record["dest"]).exists()
            ],
            "instruction_block": _instruction_block_state(state),
            "commit": manifest.get("source", {}).get("commit") if isinstance(manifest.get("source"), dict) else None,
        }
    return {
        "status": "ok" if targets_payload else "missing",
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
    cache = config_dir() / "cache" / "open-xquant" / git_ref.replace("/", "_")
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
    click.echo("agent/skills/*.md")
    click.echo("")
    click.echo(GLOBAL_AGENT_BLOCK)
