from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from oxq.cli.agent import _quote_runner_for_shell, _should_update_preferred_runner
from oxq.cli.agent_manifest import read_yaml_file
from oxq.cli.main import main


def _write_source(root: Path, skills: dict[str, str] | None = None) -> None:
    skill_descriptions = skills or {
        "build-strategy-spec": "Build quant strategies",
        "run-authorized-backtest": "Run backtests",
    }
    skills = root / "agent" / "skills"
    skills.mkdir(parents=True)
    for name, description in skill_descriptions.items():
        title = name.replace("-", " ").title()
        skill_dir = skills / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: >-\n  {description}\n---\n\n# {title}\n",
            encoding="utf-8",
        )


def _snapshot_tree(root: Path) -> dict[str, tuple[str, bytes]]:
    if not root.exists():
        return {}
    snapshot: dict[str, tuple[str, bytes]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", str(path.readlink()).encode())
        elif path.is_dir():
            snapshot[relative] = ("dir", b"")
        else:
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


@pytest.fixture(autouse=True)
def fake_sdk_bundle(monkeypatch, tmp_path):
    calls: list[tuple[Path, Path, bool]] = []

    def build(source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict:
        calls.append((source_root, config_root, dry_run))
        root = config_root / "sdk-bundles" / "bundle-test"
        wheel = root / "dist" / "open_xquant-0.1.0-py3-none-any.whl"
        lock = root / "requirements.lock.txt"
        packages = root / "packages.json"
        python = root / "runner" / ".venv" / "bin" / "python"
        runner = root / "runner" / ".venv" / "bin" / "oxq"
        if not dry_run:
            wheel.parent.mkdir(parents=True, exist_ok=True)
            wheel.write_text("wheel", encoding="utf-8")
            lock.write_text("open-xquant @ file://wheel\n", encoding="utf-8")
            packages.write_text("[]\n", encoding="utf-8")
            runner.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
            runner.chmod(0o755)
        return {
            "id": "bundle-test",
            "root": str(root),
            "profile": "full-research",
            "extras": ["chart", "scipy", "yfinance", "akshare", "tushare", "live", "mcp", "agent"],
            "excluded_extras": ["dev", "docs", "talib"],
            "wheel": {
                "path": str(wheel),
                "sha256": "wheel-sha",
                "version": "0.1.0",
                "source_commit": "commit-sha",
            },
            "dependencies": {
                "lock_file": str(lock),
                "lock_sha256": "lock-sha",
                "packages_file": str(packages),
                "packages_count": 1,
            },
            "runner": {
                "venv": str(root / "runner" / ".venv"),
                "python": str(python),
                "oxq": str(runner),
                "argv": [str(runner)],
            },
            "uv_cache_dir": str(root / "uv-cache"),
        }

    monkeypatch.setattr("oxq.cli.agent.build_sdk_bundle", build, raising=False)
    return calls


def test_agent_install_all_targets_writes_managed_skills(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    codex_home = home / ".codex-profile"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--all-targets", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert (codex_home / "skills/build-strategy-spec/SKILL.md").exists()
    assert (home / ".config/opencode/skills/build-strategy-spec/SKILL.md").exists()
    assert (home / ".claude/skills/build-strategy-spec/SKILL.md").exists()
    assert (home / ".cursor/skills/build-strategy-spec/SKILL.md").exists()
    assert (home / ".openclaw/skills/build-strategy-spec/SKILL.md").exists()
    assert (home / ".trae/skills/build-strategy-spec/SKILL.md").exists()
    assert (codex_home / "skills/run-authorized-backtest/SKILL.md").exists()
    codex_instructions = (codex_home / "AGENTS.md").read_text(encoding="utf-8")
    opencode_instructions = (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    claude_instructions = (home / ".claude/CLAUDE.md").read_text(encoding="utf-8")
    assert codex_instructions.count("open-xquant:begin") == 1
    assert "open-xquant:begin" in opencode_instructions
    assert "open-xquant:begin" in claude_instructions
    assert "For open-xquant workflows, prefer SubAgents by default" in codex_instructions
    assert "For open-xquant workflows, prefer SubAgents by default" in opencode_instructions
    assert "For open-xquant workflows, prefer SubAgents by default" in claude_instructions

    marker = codex_home / "skills/build-strategy-spec/.open-xquant-managed.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["target"] == "codex"

    manifest = home / ".config/open-xquant/agent-install.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    targets = payload["targets"]
    assert set(targets) == {"codex", "opencode", "claude-code", "cursor", "openclaw", "trae"}
    assert payload["agent_profile"] == "multi-agent"
    assert targets["codex"]["agent_profile"] == "multi-agent"
    assert payload["sdk_bundle"]["id"] == "bundle-test"


def test_agent_install_copies_skill_references(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    codex_home = home / ".codex-profile"
    _write_source(source)
    reference_dir = source / "agent" / "skills" / "build-strategy-spec" / "references"
    reference_dir.mkdir()
    (reference_dir / "details.md").write_text("# Details\n\nReference body.\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    installed = codex_home / "skills/build-strategy-spec/references/details.md"
    assert installed.read_text(encoding="utf-8") == "# Details\n\nReference body.\n"


def test_agent_install_stages_all_skills_before_committing_target(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(
        source,
        {
            "build-strategy-spec": "Build quant strategies",
            "zz-second-skill": "Second staged skill",
        },
    )
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    (source / "agent/skills/zz-second-skill/linked.md").symlink_to(outside)
    monkeypatch.setenv("HOME", str(home))
    target_root = home / ".cursor"
    manifest = home / ".config/open-xquant/agent-install.json"

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 1
    assert "Refusing symlinked skill resource" in result.output
    assert not target_root.exists()
    assert not manifest.exists()


@pytest.mark.parametrize(
    ("malformed_config", "invalid_field"),
    [
        ({"skills": []}, "skills"),
        ({"skills": {"entries": []}}, "skills.entries"),
    ],
)
def test_agent_install_rejects_malformed_openclaw_config_before_mutation(
    monkeypatch,
    tmp_path,
    malformed_config: dict,
    invalid_field: str,
) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    config_file = home / ".openclaw/openclaw.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps(malformed_config, indent=2) + "\n", encoding="utf-8")
    home_before = _snapshot_tree(home)

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "openclaw", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 1
    assert invalid_field in result.output
    assert _snapshot_tree(home) == home_before


def test_agent_install_trae_writes_global_skills(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "trae", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    installed = home / ".trae/skills/build-strategy-spec/SKILL.md"
    assert installed.exists()
    assert "Build quant strategies" in installed.read_text(encoding="utf-8")
    assert (home / ".trae/skills/run-authorized-backtest/SKILL.md").exists()

    marker = home / ".trae/skills/build-strategy-spec/.open-xquant-managed.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["target"] == "trae"

    config = read_yaml_file(home / ".config/open-xquant/agent.yaml")
    assert config["agent_profile"] == "standalone-agent"
    assert config["preferred_runner"].endswith("/sdk-bundles/bundle-test/runner/.venv/bin/oxq")
    assert config["preferred_runner_argv"] == [config["preferred_runner"]]


def test_agent_install_writes_cached_runner(monkeypatch, tmp_path, fake_sdk_bundle) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    raw_config = (home / ".config/open-xquant/agent.yaml").read_text(encoding="utf-8")
    config = read_yaml_file(home / ".config/open-xquant/agent.yaml")
    assert fake_sdk_bundle == [(source.resolve(), home / ".config/open-xquant", False)]
    assert config["preferred_runner"].endswith("/sdk-bundles/bundle-test/runner/.venv/bin/oxq")
    assert config["preferred_runner_argv"] == [config["preferred_runner"]]
    assert "sdk-bundles/bundle-test" in raw_config
    assert str(source.resolve()) not in config["preferred_runner"]

    instructions = (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    assert "agent.yaml" in instructions
    assert "agent-install.json" in instructions
    assert "preferred_runner_argv" in instructions


def test_agent_install_interactive_profile_can_choose_standalone(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source)],
        input="standalone-agent\n",
    )

    assert result.exit_code == 0, result.output
    assert "Install profile" in result.output
    assert (home / ".config/opencode/skills/build-strategy-spec/SKILL.md").exists()
    assert (home / ".config/opencode/skills/run-authorized-backtest/SKILL.md").exists()
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    assert manifest["agent_profile"] == "standalone-agent"
    assert manifest["targets"]["opencode"]["agent_profile"] == "standalone-agent"
    instructions = (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    assert "For open-xquant workflows, prefer SubAgents by default" not in instructions


def test_agent_install_real_source_installs_open_xquant_router(monkeypatch, tmp_path) -> None:
    source = Path.cwd()
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    router = home / ".config/opencode/skills/open-xquant/SKILL.md"
    assert router.exists()
    router_text = router.read_text(encoding="utf-8")
    assert "Router Contract" in router_text
    assert "Multi-Agent workflows use narrow leaf skills only" in router_text
    assert "run-authorized-backtest" in router_text
    assert "strategy-builder-standalone" not in router_text
    assert "quant-research" not in router_text
    assert not (home / ".config/opencode/skills/strategy-builder-standalone").exists()
    assert not (home / ".config/opencode/skills/quant-research").exists()

    instructions = (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    assert "use the installed `open-xquant` skill first" in instructions
    assert "Do not run `oxq`" in instructions
    assert "Default workflow" not in instructions

    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert "open-xquant" in names
    assert "run-authorized-backtest" in names
    assert "strategy-builder-standalone" not in names
    assert "quant-research" not in names

    coordinator = home / ".config/opencode/agents/oxq-coordinator.md"
    builder = home / ".config/opencode/agents/oxq-strategy-builder-worker.md"
    assert coordinator.exists()
    assert builder.exists()
    assert "mode: primary" in coordinator.read_text(encoding="utf-8")
    assert "mode: subagent" in builder.read_text(encoding="utf-8")
    installed_roles = sorted((home / ".config/opencode/agents").glob("oxq-*.md"))
    assert installed_roles
    for installed_role in installed_roles:
        role_text = installed_role.read_text(encoding="utf-8")
        role_frontmatter = yaml.safe_load(role_text.split("---", 2)[1])
        assert role_frontmatter["permission"]["edit"] == "ask", installed_role.name
        assert role_frontmatter["permission"]["bash"] == "ask", installed_role.name
        if installed_role != coordinator:
            assert role_frontmatter["permission"]["task"] == {"*": "deny"}, installed_role.name

    coordinator_text = coordinator.read_text(encoding="utf-8")
    coordinator_frontmatter = yaml.safe_load(coordinator_text.split("---", 2)[1])
    coordinator_permissions = coordinator_frontmatter["permission"]
    assert coordinator_permissions["edit"] == "ask"
    assert coordinator_permissions["bash"] == "ask"
    worker_names = sorted(role.stem for role in installed_roles if role != coordinator)
    coordinator_task_permissions = coordinator_permissions["task"]
    assert list(coordinator_task_permissions) == ["*", *worker_names]
    assert coordinator_task_permissions["*"] == "deny"
    assert all(coordinator_task_permissions[name] == "allow" for name in worker_names)
    assert "general" not in coordinator_task_permissions
    assert "explore" not in coordinator_task_permissions
    assert "*/transcript.md" not in coordinator_text
    assert "*/confirmations.jsonl" not in coordinator_text
    assert "*/08_runtime_audit/backtest_authorization.json" not in coordinator_text
    assert "coordinator appends a confirmation event" in coordinator_text
    assert "coordinator writes only the small handoff file" in coordinator_text
    roles = {record["name"] for record in manifest["targets"]["opencode"]["agent_roles"]}
    assert roles == {
        "oxq-coordinator",
        "oxq-artifact-governor-worker",
        "oxq-component-author-worker",
        "oxq-data-inspection-worker",
        "oxq-experiment-comparator-worker",
        "oxq-final-selector-worker",
        "oxq-lineage-auditor-worker",
        "oxq-monitor-worker",
        "oxq-strategy-brainstorm-worker",
        "oxq-strategy-idea-auditor-worker",
        "oxq-strategy-builder-worker",
        "oxq-version-manager-worker",
        "oxq-spec-auditor-worker",
        "oxq-runtime-auditor-worker",
        "oxq-runner-worker",
        "oxq-report-writer-worker",
        "oxq-report-reviewer-worker",
    }


def test_agent_install_real_source_standalone_profile_uses_narrow_skills(monkeypatch, tmp_path) -> None:
    source = Path.cwd()
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "install",
            "--target",
            "opencode",
            "--from-local",
            str(source),
            "--profile",
            "standalone-agent",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (home / ".config/opencode/skills/run-authorized-backtest/SKILL.md").exists()
    assert (home / ".config/opencode/skills/audit-runtime-semantics/SKILL.md").exists()
    assert not (home / ".config/opencode/skills/strategy-builder-standalone").exists()
    assert not (home / ".config/opencode/skills/quant-research").exists()
    assert not (home / ".config/opencode/agents/oxq-coordinator.md").exists()
    router = (home / ".config/opencode/skills/open-xquant/SKILL.md").read_text(encoding="utf-8")
    assert "run-authorized-backtest" in router
    assert "audit-runtime-semantics" in router
    assert "strategy-builder-standalone" not in router
    assert "quant-research" not in router


def test_agent_install_generic_does_not_advertise_cached_runner(monkeypatch, tmp_path, fake_sdk_bundle) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(main, ["agent", "install", "--target", "generic", "--yes"])

    assert result.exit_code == 0, result.output
    assert fake_sdk_bundle == []
    assert "agent/skills/<name>/SKILL.md" in result.output
    assert "sdk-bundles" not in result.output
    assert "agent-install.json" not in result.output
    assert "preferred_runner_argv" not in result.output
    config = read_yaml_file(home / ".config/open-xquant/agent.yaml")
    assert config["preferred_runner"] == "uv run oxq"
    assert "preferred_runner_argv" not in config
    assert not (home / ".config/open-xquant/agent-install.json").exists()


def test_agent_install_quotes_shell_runner_but_keeps_raw_argv(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home with spaces"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    config = read_yaml_file(home / ".config/open-xquant/agent.yaml")
    assert config["preferred_runner"].startswith("'")
    assert config["preferred_runner"].endswith("'")
    assert config["preferred_runner_argv"] == [config["preferred_runner"].strip("'")]


def test_agent_install_preserves_custom_runner_without_default_argv(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    config_dir = home / ".config/open-xquant"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    config_dir.mkdir(parents=True)
    (config_dir / "agent.yaml").write_text(
        "schema_version: 1\npreferred_runner: custom-oxq\ninstalled_targets: []\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    config = read_yaml_file(config_dir / "agent.yaml")
    assert config["preferred_runner"] == "custom-oxq"
    assert "preferred_runner_argv" not in config


def test_agent_install_drops_stale_sdk_bundle_argv_for_custom_runner(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    config_dir = home / ".config/open-xquant"
    old_runner = home / ".config/open-xquant/sdk-bundles/old/runner/.venv/bin/oxq"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    config_dir.mkdir(parents=True)
    (config_dir / "agent.yaml").write_text(
        "schema_version: 1\n"
        "preferred_runner: custom-oxq\n"
        "preferred_runner_argv:\n"
        f"  - {old_runner}\n"
        "installed_targets: []\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    config = read_yaml_file(config_dir / "agent.yaml")
    assert config["preferred_runner"] == "custom-oxq"
    assert "preferred_runner_argv" not in config


def test_agent_runner_update_recognizes_windows_sdk_bundle_path() -> None:
    runner = r"C:\Users\Alice\.config\open-xquant\sdk-bundles\old\runner\.venv\Scripts\oxq.exe"

    assert _should_update_preferred_runner(runner) is True


def test_agent_runner_shell_quote_uses_windows_command_line_quotes() -> None:
    runner = r"C:\Users\Alice Smith\.config\open-xquant\sdk-bundles\bundle\runner\.venv\Scripts\oxq.exe"

    assert _quote_runner_for_shell(runner) == f'"{runner}"'


def test_agent_runner_shell_quote_uses_powershell_call_operator_on_windows(monkeypatch) -> None:
    runner = r"C:\Users\Alice Smith\.config\open-xquant\sdk-bundles\bundle\runner\.venv\Scripts\oxq.exe"
    monkeypatch.setattr("oxq.cli.agent.sys.platform", "win32")

    assert _quote_runner_for_shell(runner) == f'& "{runner}"'


def test_agent_status_json_reports_installed_targets(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(main, ["agent", "status", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["targets"]["opencode"]["installed"] is True
    assert payload["agent_profile"] == "multi-agent"
    assert payload["targets"]["opencode"]["agent_profile"] == "multi-agent"
    assert payload["targets"]["opencode"]["skills"]["installed"] == 2
    assert payload["targets"]["opencode"]["agent_roles"]["installed"] == 0


def test_agent_status_uses_persisted_codex_root_after_env_change(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    original_codex_home = tmp_path / "codex-original"
    replacement_codex_home = tmp_path / "codex-replacement"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(original_codex_home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    replacement_codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(replacement_codex_home))

    result = CliRunner().invoke(main, ["agent", "status", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["targets"]["codex"]["skills"]["installed"] == 2
    assert payload["targets"]["codex"]["instruction_block"] == "present"
    assert not any(replacement_codex_home.iterdir())


def test_agent_install_real_source_writes_codex_agent_roles(monkeypatch, tmp_path) -> None:
    source = Path.cwd()
    home = tmp_path / "home"
    codex_home = home / ".codex-profile"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code == 0, result.output
    coordinator = codex_home / "agents/oxq-coordinator.toml"
    runner = codex_home / "agents/oxq-runner-worker.toml"
    assert coordinator.exists()
    assert runner.exists()
    coordinator_payload = tomllib.loads(coordinator.read_text(encoding="utf-8"))
    assert coordinator_payload["name"] == "oxq-coordinator"
    assert "oxq-strategy-brainstorm-worker" in coordinator_payload["developer_instructions"]
    assert "oxq-strategy-idea-auditor-worker" in coordinator_payload["developer_instructions"]
    assert "oxq-strategy-builder-worker" in coordinator_payload["developer_instructions"]
    assert "oxq-data-inspection-worker" in coordinator_payload["developer_instructions"]
    assert "oxq-version-manager-worker" in coordinator_payload["developer_instructions"]
    assert "oxq-artifact-governor-worker" in coordinator_payload["developer_instructions"]
    assert "oxq-final-selector-worker" in coordinator_payload["developer_instructions"]
    assert "open-xquant SubAgent workflow" in coordinator_payload["developer_instructions"]
    assert "Main agent only coordinates" in coordinator_payload["developer_instructions"]
    runner_payload = tomllib.loads(runner.read_text(encoding="utf-8"))
    assert "run-authorized-backtest" in runner_payload["developer_instructions"]


def test_agent_install_skips_agent_roles_for_targets_without_subagents(monkeypatch, tmp_path) -> None:
    source = Path.cwd()
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        [
            "agent",
            "install",
            "--target",
            "trae",
            "--from-local",
            str(source),
            "--profile",
            "multi-agent",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "skip agent roles" in result.output
    assert not (home / ".trae/agents/oxq-coordinator.md").exists()
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    assert manifest["targets"]["trae"]["agent_roles"] == []


def test_agent_upgrade_reconciles_added_and_removed_managed_skills(monkeypatch, tmp_path) -> None:
    old_source = tmp_path / "old-source"
    new_source = tmp_path / "new-source"
    home = tmp_path / "home"
    _write_source(old_source, {"old-skill": "Old skill", "kept-skill": "Kept skill"})
    _write_source(new_source, {"kept-skill": "Kept skill updated", "new-skill": "New skill"})
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(old_source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    upgrade = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(new_source), "--yes"],
    )

    assert upgrade.exit_code == 0, upgrade.output
    skills_dir = home / ".config/opencode/skills"
    assert not (skills_dir / "old-skill").exists()
    assert (skills_dir / "kept-skill/SKILL.md").exists()
    assert (skills_dir / "new-skill/SKILL.md").exists()
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert names == {"kept-skill", "new-skill"}


def test_agent_upgrade_cleans_renamed_managed_skill_dirs(monkeypatch, tmp_path) -> None:
    new_source = tmp_path / "new-source"
    home = tmp_path / "home"
    _write_source(new_source)
    monkeypatch.setenv("HOME", str(home))
    skills_dir = home / ".config/opencode/skills"
    for name in ("strategy-builder", "backtest-runner"):
        skill_dir = skills_dir / name
        skill_file = skill_dir / "SKILL.md"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(
            f"---\nname: {name}\ndescription: Legacy skill\n---\n\n# Legacy\n",
            encoding="utf-8",
        )
        skill_sha = hashlib.sha256(skill_file.read_bytes()).hexdigest()
        (skill_dir / ".open-xquant-managed.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "managed_by": "open-xquant",
                    "target": "opencode",
                    "name": name,
                    "dest_sha256": skill_sha,
                }
            ),
            encoding="utf-8",
        )

    upgrade = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(new_source), "--yes"],
    )

    assert upgrade.exit_code == 0, upgrade.output
    assert not (skills_dir / "strategy-builder").exists()
    assert not (skills_dir / "backtest-runner").exists()
    assert (skills_dir / "build-strategy-spec/SKILL.md").exists()
    assert (skills_dir / "run-authorized-backtest/SKILL.md").exists()
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert names == {"build-strategy-spec", "run-authorized-backtest"}


def test_agent_upgrade_removes_openclaw_config_for_deprecated_skills(monkeypatch, tmp_path) -> None:
    new_source = tmp_path / "new-source"
    home = tmp_path / "home"
    _write_source(new_source)
    monkeypatch.setenv("HOME", str(home))
    skills_dir = home / ".openclaw" / "skills"
    config_file = home / ".openclaw" / "openclaw.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(
            {
                "skills": {
                    "entries": {
                        "strategy-builder": {"enabled": True},
                        "backtest-runner": {"enabled": True},
                        "spec-auditor": {"enabled": True},
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in ("strategy-builder", "backtest-runner"):
        skill_dir = skills_dir / name
        skill_file = skill_dir / "SKILL.md"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(
            f"---\nname: {name}\ndescription: Legacy skill\n---\n\n# Legacy\n",
            encoding="utf-8",
        )
        skill_sha = hashlib.sha256(skill_file.read_bytes()).hexdigest()
        (skill_dir / ".open-xquant-managed.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "managed_by": "open-xquant",
                    "target": "openclaw",
                    "name": name,
                    "dest_sha256": skill_sha,
                }
            ),
            encoding="utf-8",
        )

    upgrade = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "openclaw", "--from-local", str(new_source), "--yes"],
    )

    assert upgrade.exit_code == 0, upgrade.output
    assert not (skills_dir / "strategy-builder").exists()
    assert not (skills_dir / "backtest-runner").exists()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    entries = config["skills"]["entries"]
    assert "strategy-builder" not in entries
    assert "backtest-runner" not in entries
    assert "spec-auditor" not in entries
    assert entries["build-strategy-spec"]["enabled"] is True
    assert entries["run-authorized-backtest"]["enabled"] is True


def test_agent_upgrade_preserves_modified_removed_managed_skill(monkeypatch, tmp_path) -> None:
    old_source = tmp_path / "old-source"
    new_source = tmp_path / "new-source"
    home = tmp_path / "home"
    _write_source(old_source, {"old-skill": "Old skill", "kept-skill": "Kept skill"})
    _write_source(new_source, {"kept-skill": "Kept skill updated"})
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(old_source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    old_skill_file = home / ".config/opencode/skills/old-skill/SKILL.md"
    old_skill_file.write_text(old_skill_file.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")

    upgrade = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(new_source), "--yes"],
    )

    assert upgrade.exit_code == 0, upgrade.output
    assert old_skill_file.exists()
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert names == {"old-skill", "kept-skill"}


def test_agent_install_repair_restores_missing_managed_skill(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    skill_file = home / ".config/opencode/skills/build-strategy-spec/SKILL.md"
    skill_file.unlink()

    repair = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--repair", "--yes"],
    )

    assert repair.exit_code == 0, repair.output
    assert skill_file.exists()


def test_agent_install_repair_preserves_modified_skill_manifest_record(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    skill_file = home / ".config/opencode/skills/build-strategy-spec/SKILL.md"
    skill_file.write_text(skill_file.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")

    repair = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--repair", "--yes"],
    )

    assert repair.exit_code == 0, repair.output
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    names = {record["name"] for record in manifest["targets"]["opencode"]["skills"]}
    assert names == {"build-strategy-spec", "run-authorized-backtest"}


def test_agent_uninstall_requires_explicit_target_or_all_targets(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(main, ["agent", "uninstall", "--yes"])

    assert result.exit_code != 0
    assert "Use --target or --all-targets" in result.output
    assert (home / ".config/opencode/skills/build-strategy-spec/SKILL.md").exists()


def test_agent_uninstall_all_targets_rolls_back_when_later_marker_is_malformed(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        root = tmp_path / cwd
        source = root / "source"
        home = root / "home"
        _write_source(source)
        monkeypatch.setenv("HOME", str(home))
        for target in ("cursor", "opencode"):
            install = runner.invoke(
                main,
                ["agent", "install", "--target", target, "--from-local", str(source), "--yes"],
            )
            assert install.exit_code == 0, install.output

        cursor_root = home / ".cursor"
        opencode_root = home / ".config/opencode"
        manifest = home / ".config/open-xquant/agent-install.json"
        config = home / ".config/open-xquant/agent.yaml"
        instruction = opencode_root / "AGENTS.md"
        instruction.write_text(
            instruction.read_text(encoding="utf-8").replace("<!-- open-xquant:end -->", ""),
            encoding="utf-8",
        )
        cursor_before = _snapshot_tree(cursor_root)
        opencode_before = _snapshot_tree(opencode_root)
        manifest_before = manifest.read_bytes()
        config_before = config.read_bytes()

        result = runner.invoke(main, ["agent", "uninstall", "--all-targets", "--yes"])

        assert result.exit_code == 1
        assert "marker" in result.output.lower()
        assert _snapshot_tree(cursor_root) == cursor_before
        assert _snapshot_tree(opencode_root) == opencode_before
        assert manifest.read_bytes() == manifest_before
        assert config.read_bytes() == config_before


def test_agent_install_rejects_unsafe_skill_names(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    skills = source / "agent" / "skills"
    skill_dir = skills / "escape"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: ../AGENTS\ndescription: Escape target\n---\n\n# Escape\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    result = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert result.exit_code != 0
    assert "invalid skill name" in result.output
    assert not (home / ".config/opencode/AGENTS/SKILL.md").exists()
