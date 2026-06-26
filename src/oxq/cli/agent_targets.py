"""Agent target adapters for installing open-xquant skills."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any

import yaml

from oxq.cli.agent_manifest import expand_path, sha256_file

CONCRETE_TARGETS = ("codex", "opencode", "claude-code", "cursor", "openclaw", "trae")
SUPPORTED_TARGETS = CONCRETE_TARGETS + ("generic",)
ROLE_TARGETS = ("codex", "opencode", "claude-code", "cursor")


@dataclass(frozen=True)
class AgentTarget:
    id: str
    skills_dir: Path | None
    instruction_file: Path | None = None
    config_file: Path | None = None
    agents_dir: Path | None = None


@dataclass(frozen=True)
class SkillSource:
    name: str
    description: str
    path: Path
    content: str
    body: str
    metadata: dict[str, object]
    source_sha256: str


@dataclass(frozen=True)
class AgentRoleSource:
    name: str
    description: str
    mode: str
    path: Path
    content: str
    body: str
    metadata: dict[str, Any]
    source_sha256: str


class SkillValidationError(ValueError):
    """Raised when a source skill cannot be installed for a target."""


def home_path(*parts: str) -> Path:
    return Path.home().joinpath(*parts).resolve()


def resolve_codex_target() -> AgentTarget:
    codex_home = expand_path(os.environ.get("CODEX_HOME", "~/.codex"))
    return AgentTarget("codex", codex_home / "skills", codex_home / "AGENTS.md", agents_dir=codex_home / "agents")


def resolve_opencode_target() -> AgentTarget:
    return AgentTarget(
        "opencode",
        home_path(".config", "opencode", "skills"),
        home_path(".config", "opencode", "AGENTS.md"),
        agents_dir=home_path(".config", "opencode", "agents"),
    )


def resolve_claude_code_target() -> AgentTarget:
    return AgentTarget(
        "claude-code",
        home_path(".claude", "skills"),
        home_path(".claude", "CLAUDE.md"),
        agents_dir=home_path(".claude", "agents"),
    )


def resolve_cursor_target() -> AgentTarget:
    return AgentTarget("cursor", home_path(".cursor", "skills"), agents_dir=home_path(".cursor", "agents"))


def resolve_openclaw_target() -> AgentTarget:
    return AgentTarget("openclaw", home_path(".openclaw", "skills"), config_file=home_path(".openclaw", "openclaw.json"))


def resolve_trae_target() -> AgentTarget:
    return AgentTarget("trae", home_path(".trae", "skills"))


def resolve_generic_target() -> AgentTarget:
    return AgentTarget("generic", None)


def resolve_target(target_id: str) -> AgentTarget:
    resolvers = {
        "codex": resolve_codex_target,
        "opencode": resolve_opencode_target,
        "claude-code": resolve_claude_code_target,
        "cursor": resolve_cursor_target,
        "openclaw": resolve_openclaw_target,
        "trae": resolve_trae_target,
        "generic": resolve_generic_target,
    }
    try:
        return resolvers[target_id]()
    except KeyError as exc:
        raise ValueError(f"Unsupported target: {target_id}") from exc


def detect_targets() -> list[str]:
    detected: list[str] = []
    if which("codex") or os.environ.get("CODEX_HOME") or home_path(".codex").exists():
        detected.append("codex")
    if which("opencode") or home_path(".config", "opencode").exists():
        detected.append("opencode")
    if which("claude") or home_path(".claude").exists():
        detected.append("claude-code")
    if which("cursor") or home_path(".cursor").exists():
        detected.append("cursor")
    if which("openclaw") or home_path(".openclaw").exists():
        detected.append("openclaw")
    if which("trae") or home_path(".trae").exists():
        detected.append("trae")
    return detected or ["generic"]


def resolve_source_root(from_local: str | None = None) -> Path:
    candidates: list[Path] = []
    if from_local:
        candidates.append(expand_path(from_local))
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    package_root = Path(__file__).resolve()
    candidates.extend(package_root.parents)
    for candidate in candidates:
        if (candidate / "agent" / "skills").is_dir():
            return candidate
    raise FileNotFoundError("Could not find agent/skills. Pass --from-local /path/to/open-xquant.")


def discover_skills(source_root: Path) -> list[SkillSource]:
    skills_dir = source_root / "agent" / "skills"
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"Missing skills directory: {skills_dir}")
    skills = [_read_skill(path) for path in sorted(skills_dir.glob("*/SKILL.md"))]
    if not skills:
        raise FileNotFoundError(f"No skill directories with SKILL.md found in {skills_dir}")
    return skills


def discover_agent_roles(source_root: Path) -> list[AgentRoleSource]:
    roles_dir = source_root / "agent" / "roles"
    if not roles_dir.is_dir():
        return []
    roles = [_read_agent_role(path) for path in sorted(roles_dir.glob("*.md"))]
    return roles


def _read_skill(path: Path) -> SkillSource:
    if path.is_symlink():
        raise SkillValidationError(f"Refusing symlinked skill file: {path}")
    content = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(content, path)
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillValidationError(f"Skill {path} is missing name frontmatter")
    if not _is_safe_skill_name(name.strip()):
        raise SkillValidationError(f"invalid skill name: {name}")
    if path.parent.name != name.strip():
        raise SkillValidationError(f"Skill {path} name must match its directory: {path.parent.name}")
    if not isinstance(description, str) or not description.strip():
        raise SkillValidationError(f"Skill {path} is missing description frontmatter")
    return SkillSource(
        name=name.strip(),
        description=" ".join(description.split()),
        path=path.resolve(),
        content=content,
        body=body,
        metadata=metadata,
        source_sha256=sha256_file(path),
    )


def _read_agent_role(path: Path) -> AgentRoleSource:
    if path.is_symlink():
        raise SkillValidationError(f"Refusing symlinked agent role file: {path}")
    content = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(content, path)
    name = metadata.get("name")
    description = metadata.get("description")
    mode = metadata.get("mode", "subagent")
    if not isinstance(name, str) or not name.strip():
        raise SkillValidationError(f"Agent role {path} is missing name frontmatter")
    if not _is_safe_skill_name(name.strip()):
        raise SkillValidationError(f"invalid agent role name: {name}")
    if not isinstance(description, str) or not description.strip():
        raise SkillValidationError(f"Agent role {path} is missing description frontmatter")
    if mode not in {"primary", "subagent"}:
        raise SkillValidationError(f"Agent role {path} has invalid mode: {mode}")
    return AgentRoleSource(
        name=name.strip(),
        description=" ".join(description.split()),
        mode=mode,
        path=path.resolve(),
        content=content,
        body=body,
        metadata=metadata,
        source_sha256=sha256_file(path),
    )


def _split_frontmatter(content: str, path: Path) -> tuple[dict[str, object], str]:
    if not content.startswith("---\n"):
        raise SkillValidationError(f"Skill {path} must start with YAML frontmatter")
    stop = content.find("\n---", 4)
    if stop == -1:
        raise SkillValidationError(f"Skill {path} has unterminated YAML frontmatter")
    raw = content[4:stop]
    body_start = stop + len("\n---")
    if content[body_start : body_start + 1] == "\n":
        body_start += 1
    metadata = yaml.safe_load(raw)
    if not isinstance(metadata, dict):
        raise SkillValidationError(f"Skill {path} frontmatter must be a mapping")
    return metadata, content[body_start:]


def _is_safe_skill_name(name: str) -> bool:
    if not name or "/" in name or "\\" in name:
        return False
    candidate = Path(name)
    return not candidate.is_absolute() and all(part not in {"", ".", ".."} for part in candidate.parts)


def validate_skill_for_target(skill: SkillSource, target_id: str) -> None:
    if target_id == "opencode" and not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", skill.name):
        raise SkillValidationError(f"OpenCode skill name is invalid: {skill.name}")
    if target_id == "openclaw":
        metadata = skill.metadata.get("metadata")
        if metadata is not None and not isinstance(metadata, (str, dict)):
            raise SkillValidationError(f"OpenClaw metadata must be renderable for {skill.name}")


def render_skill_for_target(skill: SkillSource, target_id: str) -> str:
    validate_skill_for_target(skill, target_id)
    if target_id != "openclaw":
        return skill.content
    rendered = {
        "name": skill.name,
        "description": skill.description,
    }
    metadata = skill.metadata.get("metadata")
    if metadata:
        rendered["metadata"] = metadata
    frontmatter = yaml.safe_dump(rendered, sort_keys=False, default_flow_style=False).strip()
    return f"---\n{frontmatter}\n---\n{skill.body}"


def render_agent_role_for_target(role: AgentRoleSource, target_id: str) -> tuple[str, str]:
    if target_id == "codex":
        return f"{role.name}.toml", _render_codex_role(role)
    if target_id == "opencode":
        return f"{role.name}.md", _render_opencode_role(role)
    if target_id == "claude-code":
        return f"{role.name}.md", _render_markdown_role(role)
    if target_id == "cursor":
        return f"{role.name}.md", _render_markdown_role(role)
    raise SkillValidationError(f"{target_id} does not support managed open-xquant agent roles")


def _render_codex_role(role: AgentRoleSource) -> str:
    import json

    body = _role_body_with_header(role)
    return (
        f"name = {json.dumps(role.name)}\n"
        f"description = {json.dumps(role.description)}\n"
        f"developer_instructions = {json.dumps(body)}\n"
    )


def _render_opencode_role(role: AgentRoleSource) -> str:
    rendered = {
        "description": role.description,
        "mode": role.mode,
    }
    if role.mode == "primary":
        rendered["permission"] = {"edit": "deny", "bash": "deny"}
    frontmatter = yaml.safe_dump(rendered, sort_keys=False, default_flow_style=False).strip()
    return f"---\n{frontmatter}\n---\n{_role_body_with_header(role)}"


def _render_markdown_role(role: AgentRoleSource) -> str:
    rendered = {
        "name": role.name,
        "description": role.description,
    }
    frontmatter = yaml.safe_dump(rendered, sort_keys=False, default_flow_style=False).strip()
    return f"---\n{frontmatter}\n---\n{_role_body_with_header(role)}"


def _role_body_with_header(role: AgentRoleSource) -> str:
    return f"# {role.name}\n\n{role.body.lstrip()}"
