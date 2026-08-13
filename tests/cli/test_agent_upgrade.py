from __future__ import annotations

import json
import stat
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import oxq.cli.agent as agent_module
from oxq.cli.agent import _upgrade_source
from oxq.cli.main import main


def _write_skill(root: Path, name: str, body: str) -> Path:
    skill_dir = root / "agent" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Build quant strategies\n---\n\n"
        f"# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_source(root: Path, body: str) -> None:
    _write_skill(root, "build-strategy-spec", body)


def _write_role(root: Path, name: str, body: str) -> None:
    role_dir = root / "agent" / "roles"
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Test managed role\nmode: subagent\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _snapshot_tree(root: Path) -> dict[str, tuple[str, bytes]]:
    if not root.exists():
        return {}
    result: dict[str, tuple[str, bytes]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = ("symlink", str(path.readlink()).encode())
        elif path.is_dir():
            result[relative] = ("dir", b"")
        else:
            result[relative] = ("file", path.read_bytes())
    return result


@pytest.fixture(autouse=True)
def fake_sdk_bundle(monkeypatch):
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
            "wheel": {"path": str(wheel), "sha256": "wheel-sha", "version": "0.1.0", "source_commit": "commit-sha"},
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

    monkeypatch.setattr("oxq.cli.agent.build_sdk_bundle", build)
    return calls


def test_agent_upgrade_replaces_unmodified_managed_skill(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    installed = home / ".cursor/skills/build-strategy-spec/SKILL.md"
    assert "new workflow" in installed.read_text(encoding="utf-8")


def test_agent_upgrade_uses_persisted_codex_root_after_env_change(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    original_codex_home = tmp_path / "codex-original"
    replacement_codex_home = tmp_path / "codex-replacement"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(original_codex_home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    replacement_codex_home.mkdir()
    replacement_sentinel = replacement_codex_home / "keep.txt"
    replacement_sentinel.write_text("keep me\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(replacement_codex_home))

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "codex", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    installed = original_codex_home / "skills/build-strategy-spec/SKILL.md"
    assert "new workflow" in installed.read_text(encoding="utf-8")
    assert replacement_sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_agent_upgrade_rejects_forged_openclaw_config_file(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "openclaw", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".openclaw"
    unrelated_config = tmp_path / "unrelated.json"
    unrelated_config.write_text(
        json.dumps({"skills": {"entries": {"unrelated": {"enabled": False}}}}) + "\n",
        encoding="utf-8",
    )
    manifest = home / ".config/open-xquant/agent-install.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["targets"]["openclaw"]["config_file"] = str(unrelated_config)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config = home / ".config/open-xquant/agent.yaml"
    target_before = _snapshot_tree(target_root)
    unrelated_before = unrelated_config.read_bytes()
    manifest_before = manifest.read_bytes()
    config_before = config.read_bytes()

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "openclaw", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 1
    assert "unexpected managed config file" in result.output
    assert _snapshot_tree(target_root) == target_before
    assert unrelated_config.read_bytes() == unrelated_before
    assert manifest.read_bytes() == manifest_before
    assert config.read_bytes() == config_before


@pytest.mark.parametrize(
    ("malformed_config", "invalid_field"),
    [
        ({"skills": []}, "skills"),
        ({"skills": {"entries": []}}, "skills.entries"),
    ],
)
def test_agent_upgrade_rejects_malformed_openclaw_config_before_mutation(
    monkeypatch,
    tmp_path,
    malformed_config: dict,
    invalid_field: str,
) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "openclaw", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    config_file = home / ".openclaw/openclaw.json"
    config_file.write_text(json.dumps(malformed_config, indent=2) + "\n", encoding="utf-8")
    home_before = _snapshot_tree(home)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "openclaw", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 1
    assert invalid_field in result.output
    assert _snapshot_tree(home) == home_before


def test_agent_upgrade_preserves_symlinked_managed_role_and_external_target(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    _write_role(source_v1, "oxq-test-worker", "old role")
    _write_role(source_v2, "oxq-test-worker", "new role")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    role_path = home / ".config/opencode/agents/oxq-test-worker.md"
    external = tmp_path / "external-role.md"
    external.write_bytes(role_path.read_bytes())
    external_before = external.read_bytes()
    role_path.unlink()
    role_path.symlink_to(external)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "skip symlink agent role" in result.output
    assert role_path.is_symlink()
    assert role_path.readlink() == external
    assert external.read_bytes() == external_before


def test_agent_uninstall_preserves_symlinked_managed_role_and_external_target(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source, "workflow")
    _write_role(source, "oxq-test-worker", "managed role")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    role_path = home / ".config/opencode/agents/oxq-test-worker.md"
    external = tmp_path / "external-role.md"
    external.write_bytes(role_path.read_bytes())
    external_before = external.read_bytes()
    role_path.unlink()
    role_path.symlink_to(external)

    result = CliRunner().invoke(main, ["agent", "uninstall", "--target", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert "skip symlink agent role" in result.output
    assert role_path.is_symlink()
    assert role_path.readlink() == external
    assert external.read_bytes() == external_before


def test_agent_uninstall_rejects_forged_recorded_agents_dir(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source, "workflow")
    _write_role(source, "oxq-test-worker", "managed role")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".config/opencode"
    role_path = target_root / "agents/oxq-test-worker.md"
    unrelated_dir = tmp_path / "unrelated-agents"
    unrelated_dir.mkdir()
    unrelated_role = unrelated_dir / "unrelated.md"
    unrelated_role.write_bytes(role_path.read_bytes())
    manifest = home / ".config/open-xquant/agent-install.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    state = payload["targets"]["opencode"]
    state["agents_dir"] = str(unrelated_dir)
    state["agent_roles"][0]["dest"] = str(unrelated_role)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config = home / ".config/open-xquant/agent.yaml"
    target_before = _snapshot_tree(target_root)
    unrelated_before = unrelated_role.read_bytes()
    manifest_before = manifest.read_bytes()
    config_before = config.read_bytes()

    result = CliRunner().invoke(main, ["agent", "uninstall", "--target", "opencode", "--yes"])

    assert result.exit_code == 1
    assert "agents directory" in result.output
    assert _snapshot_tree(target_root) == target_before
    assert unrelated_role.read_bytes() == unrelated_before
    assert manifest.read_bytes() == manifest_before
    assert config.read_bytes() == config_before


def test_agent_upgrade_rejects_symlinked_recorded_agents_dir(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    _write_role(source_v1, "oxq-test-worker", "old role")
    _write_role(source_v2, "oxq-test-worker", "new role")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    agents_dir = home / ".config/opencode/agents"
    external = tmp_path / "external-agents"
    agents_dir.replace(external)
    agents_dir.symlink_to(external, target_is_directory=True)
    external_before = _snapshot_tree(external)
    manifest = home / ".config/open-xquant/agent-install.json"
    manifest_before = manifest.read_bytes()

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 1
    assert "symlink" in result.output.lower()
    assert agents_dir.is_symlink()
    assert agents_dir.readlink() == external
    assert _snapshot_tree(external) == external_before
    assert manifest.read_bytes() == manifest_before


def test_agent_uninstall_rejects_symlinked_recorded_agents_dir(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source, "workflow")
    _write_role(source, "oxq-test-worker", "managed role")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    agents_dir = home / ".config/opencode/agents"
    external = tmp_path / "external-agents"
    agents_dir.replace(external)
    agents_dir.symlink_to(external, target_is_directory=True)
    external_before = _snapshot_tree(external)
    manifest = home / ".config/open-xquant/agent-install.json"
    manifest_before = manifest.read_bytes()

    result = CliRunner().invoke(main, ["agent", "uninstall", "--target", "opencode", "--yes"])

    assert result.exit_code == 1
    assert "symlink" in result.output.lower()
    assert agents_dir.is_symlink()
    assert agents_dir.readlink() == external
    assert _snapshot_tree(external) == external_before
    assert manifest.read_bytes() == manifest_before


def test_agent_upgrade_rejects_symlinked_managed_skill_directory(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        root = tmp_path / cwd
        source_v1 = root / "source-v1"
        source_v2 = root / "source-v2"
        home = root / "home"
        _write_source(source_v1, "old workflow")
        _write_source(source_v2, "new workflow")
        monkeypatch.setenv("HOME", str(home))

        install = runner.invoke(
            main,
            ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
        )
        assert install.exit_code == 0, install.output
        skill_dir = home / ".cursor/skills/build-strategy-spec"
        external = root / "external-skill"
        skill_dir.replace(external)
        skill_dir.symlink_to(external, target_is_directory=True)
        external_before = _snapshot_tree(external)
        manifest = home / ".config/open-xquant/agent-install.json"
        manifest_before = manifest.read_bytes()

        result = runner.invoke(
            main,
            ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
        )

        assert result.exit_code == 1
        assert "symlink" in result.output.lower()
        assert skill_dir.is_symlink()
        assert skill_dir.readlink() == external
        assert _snapshot_tree(external) == external_before
        assert manifest.read_bytes() == manifest_before


def test_agent_uninstall_rejects_symlinked_managed_skill_directory(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        root = tmp_path / cwd
        source = root / "source"
        home = root / "home"
        _write_source(source, "workflow")
        monkeypatch.setenv("HOME", str(home))

        install = runner.invoke(
            main,
            ["agent", "install", "--target", "cursor", "--from-local", str(source), "--yes"],
        )
        assert install.exit_code == 0, install.output
        skill_dir = home / ".cursor/skills/build-strategy-spec"
        external = root / "external-skill"
        skill_dir.replace(external)
        skill_dir.symlink_to(external, target_is_directory=True)
        external_before = _snapshot_tree(external)
        manifest = home / ".config/open-xquant/agent-install.json"
        manifest_before = manifest.read_bytes()

        result = runner.invoke(main, ["agent", "uninstall", "--target", "cursor", "--yes"])

        assert result.exit_code == 1
        assert "symlink" in result.output.lower()
        assert skill_dir.is_symlink()
        assert skill_dir.readlink() == external
        assert _snapshot_tree(external) == external_before
        assert manifest.read_bytes() == manifest_before


def test_agent_upgrade_syncs_skill_references(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    old_ref_dir = source_v1 / "agent/skills/build-strategy-spec/references"
    new_ref_dir = source_v2 / "agent/skills/build-strategy-spec/references"
    old_ref_dir.mkdir()
    new_ref_dir.mkdir()
    (old_ref_dir / "old.md").write_text("old reference\n", encoding="utf-8")
    (new_ref_dir / "new.md").write_text("new reference\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    installed_dir = home / ".cursor/skills/build-strategy-spec/references"
    assert not (installed_dir / "old.md").exists()
    assert (installed_dir / "new.md").read_text(encoding="utf-8") == "new reference\n"


def test_agent_upgrade_preserves_existing_skill_when_new_resource_is_symlinked(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    old_references = source_v1 / "agent/skills/build-strategy-spec/references"
    old_references.mkdir()
    (old_references / "stable.md").write_text("stable reference\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside reference\n", encoding="utf-8")
    (source_v2 / "agent/skills/build-strategy-spec/linked.md").symlink_to(outside)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    installed_dir = home / ".cursor/skills/build-strategy-spec"
    before = {
        path.relative_to(installed_dir).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(installed_dir.rglob("*"))
    }

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 1
    assert "Refusing symlinked skill resource" in result.output
    after = {
        path.relative_to(installed_dir).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(installed_dir.rglob("*"))
    }
    assert after == before
    assert "old workflow" in (installed_dir / "SKILL.md").read_text(encoding="utf-8")


def test_agent_upgrade_rolls_back_entire_target_when_second_skill_preflight_fails(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old first workflow")
    _write_source(source_v2, "new first workflow")
    _write_skill(source_v1, "zz-second-skill", "old second workflow")
    second_v2 = _write_skill(source_v2, "zz-second-skill", "new second workflow")
    outside = tmp_path / "outside.md"
    outside.write_text("outside reference\n", encoding="utf-8")
    (second_v2 / "linked.md").symlink_to(outside)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".cursor"
    manifest = home / ".config/open-xquant/agent-install.json"

    def snapshot(root: Path) -> dict[str, tuple[str, bytes]]:
        result: dict[str, tuple[str, bytes]] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[relative] = ("symlink", str(path.readlink()).encode())
            elif path.is_dir():
                result[relative] = ("dir", b"")
            else:
                result[relative] = ("file", path.read_bytes())
        return result

    target_before = snapshot(target_root)
    manifest_before = manifest.read_bytes()

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 1
    assert "Refusing symlinked skill resource" in result.output
    assert snapshot(target_root) == target_before
    assert manifest.read_bytes() == manifest_before


def test_agent_upgrade_skips_locally_modified_skill(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output

    installed = home / ".cursor/skills/build-strategy-spec/SKILL.md"
    installed.write_text(installed.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "local edit" in installed.read_text(encoding="utf-8")
    assert "new workflow" not in installed.read_text(encoding="utf-8")
    assert "modified" in result.output


def test_agent_upgrade_skips_skill_with_locally_modified_resource(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    skill_v1 = _write_skill(source_v1, "build-strategy-spec", "old workflow")
    skill_v2 = _write_skill(source_v2, "build-strategy-spec", "new workflow")
    for skill, body in ((skill_v1, "old reference\n"), (skill_v2, "new reference\n")):
        references = skill / "references"
        references.mkdir()
        (references / "ref.md").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    installed_dir = home / ".cursor/skills/build-strategy-spec"
    installed_ref = installed_dir / "references/ref.md"
    installed_ref.write_text("local reference edit\n", encoding="utf-8")
    before = _snapshot_tree(installed_dir)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "skipped modified managed files: build-strategy-spec" in result.output
    assert _snapshot_tree(installed_dir) == before
    assert installed_ref.read_bytes() == b"local reference edit\n"


def test_agent_upgrade_migrates_legacy_resource_skill_hashes(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        root = tmp_path / cwd
        source_v1 = root / "source-v1"
        source_v2 = root / "source-v2"
        home = root / "home"
        skill_v1 = _write_skill(source_v1, "build-strategy-spec", "old workflow")
        skill_v2 = _write_skill(source_v2, "build-strategy-spec", "new workflow")
        for skill, body in ((skill_v1, "old reference\n"), (skill_v2, "new reference\n")):
            references = skill / "references"
            references.mkdir()
            (references / "ref.md").write_text(body, encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))

        install = runner.invoke(
            main,
            ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
        )
        assert install.exit_code == 0, install.output
        manifest_path = home / ".config/open-xquant/agent-install.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["targets"]["cursor"]["skills"][0]
        installed = home / ".cursor/skills/build-strategy-spec"
        marker_path = installed / agent_module.MANAGED_MARKER
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["managed_tree_sha256"] = agent_module._hash_managed_skill_tree(installed)
        marker["resources_sha256"] = agent_module._hash_managed_skill_resources(installed)
        marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        for key in ("managed_tree_sha256", "resources_sha256", "marker_sha256"):
            record.pop(key)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        result = runner.invoke(
            main,
            ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
        )

        assert result.exit_code == 0, result.output
        assert "new workflow" in (installed / "SKILL.md").read_text(encoding="utf-8")
        assert (installed / "references/ref.md").read_text(encoding="utf-8") == "new reference\n"
        migrated = json.loads(manifest_path.read_text(encoding="utf-8"))["targets"]["cursor"]["skills"][0]
        assert all(migrated[key] for key in ("managed_tree_sha256", "resources_sha256", "marker_sha256"))


def test_agent_upgrade_preserves_user_file_in_unproven_legacy_resource_skill(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    for source, body, reference in (
        (source_v1, "old workflow", "old reference\n"),
        (source_v2, "new workflow", "new reference\n"),
    ):
        skill = _write_skill(source, "build-strategy-spec", body)
        (skill / "references").mkdir()
        (skill / "references/ref.md").write_text(reference, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    manifest_path = home / ".config/open-xquant/agent-install.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["targets"]["cursor"]["skills"][0]
    for key in ("managed_tree_sha256", "resources_sha256", "marker_sha256"):
        record.pop(key, None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    installed = home / ".cursor/skills/build-strategy-spec"
    notes = installed / "user-notes.md"
    notes.write_text("keep this\n", encoding="utf-8")
    before = _snapshot_tree(installed)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "skipped modified managed files: build-strategy-spec" in result.output
    assert _snapshot_tree(installed) == before
    assert notes.read_bytes() == b"keep this\n"


def test_agent_upgrade_preserves_modified_legacy_resource_skill(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        root = tmp_path / cwd
        source_v1 = root / "source-v1"
        source_v2 = root / "source-v2"
        home = root / "home"
        for source, body, reference in (
            (source_v1, "old workflow", "old reference\n"),
            (source_v2, "new workflow", "new reference\n"),
        ):
            skill = _write_skill(source, "build-strategy-spec", body)
            (skill / "references").mkdir()
            (skill / "references/ref.md").write_text(reference, encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))
        install = runner.invoke(
            main,
            ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
        )
        assert install.exit_code == 0, install.output
        manifest_path = home / ".config/open-xquant/agent-install.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["targets"]["cursor"]["skills"][0]
        for key in ("managed_tree_sha256", "resources_sha256", "marker_sha256"):
            record.pop(key)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        installed = home / ".cursor/skills/build-strategy-spec/SKILL.md"
        installed.write_text(installed.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")

        result = runner.invoke(
            main,
            ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
        )

        assert result.exit_code == 0, result.output
        assert "skipped modified managed files: build-strategy-spec" in result.output
        assert "local edit" in installed.read_text(encoding="utf-8")
        assert "new workflow" not in installed.read_text(encoding="utf-8")


def test_agent_upgrade_rolls_back_target_and_state_when_manifest_replace_fails(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".cursor"
    manifest = home / ".config/open-xquant/agent-install.json"
    config = home / ".config/open-xquant/agent.yaml"
    target_before = _snapshot_tree(target_root)
    manifest_before = manifest.read_bytes()
    config_before = config.read_bytes()
    mutation_type = (
        agent_module._WindowsRecoveryMutations
        if agent_module.os.name == "nt"
        else agent_module._PosixRecoveryMutations
    )
    original_replace = mutation_type.replace
    failed = False

    def fail_manifest_once(self, source: Path, target: Path) -> None:
        nonlocal failed
        if target == manifest and not failed:
            failed = True
            raise OSError("injected manifest replace failure")
        original_replace(self, source, target)

    monkeypatch.setattr(mutation_type, "replace", fail_manifest_once)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 1
    assert failed is True
    assert _snapshot_tree(target_root) == target_before
    assert manifest.read_bytes() == manifest_before
    assert config.read_bytes() == config_before


def test_agent_upgrade_preflights_all_targets_before_committing_any(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))
    for target in ("cursor", "opencode"):
        install = CliRunner().invoke(
            main,
            ["agent", "install", "--target", target, "--from-local", str(source_v1), "--yes"],
        )
        assert install.exit_code == 0, install.output
    cursor_root = home / ".cursor"
    opencode_root = home / ".config/opencode"
    manifest = home / ".config/open-xquant/agent-install.json"
    cursor_before = _snapshot_tree(cursor_root)
    opencode_before = _snapshot_tree(opencode_root)
    manifest_before = manifest.read_bytes()
    original_stage = agent_module._stage_managed_skill

    def fail_second_target(*args, **kwargs):
        if kwargs.get("target_id") == "opencode":
            raise click.ClickException("injected second target preflight failure")
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(agent_module, "_stage_managed_skill", fail_second_target)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--all-targets", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 1
    assert "injected second target preflight failure" in result.output
    assert _snapshot_tree(cursor_root) == cursor_before
    assert _snapshot_tree(opencode_root) == opencode_before
    assert manifest.read_bytes() == manifest_before


def test_agent_upgrade_removes_new_bundle_after_target_preflight_failure(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        root = tmp_path / cwd
        source_v1 = root / "source-v1"
        source_v2 = root / "source-v2"
        home = root / "home"
        _write_source(source_v1, "old workflow")
        _write_source(source_v2, "new workflow")
        monkeypatch.setenv("HOME", str(home))
        install = runner.invoke(
            main,
            ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
        )
        assert install.exit_code == 0, install.output
        manifest_path = home / ".config/open-xquant/agent-install.json"
        manifest_before = manifest_path.read_bytes()
        bundles_root = home / ".config/open-xquant/sdk-bundles"
        bundles_before = _snapshot_tree(bundles_root)

        def build_orphan(_source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict:
            bundle = config_root / "sdk-bundles/bundle-orphan"
            if not dry_run:
                (bundle / "runner/.venv/bin").mkdir(parents=True)
                (bundle / "runner/.venv/bin/oxq").write_text("runner\n", encoding="utf-8")
            return {"id": "bundle-orphan", "root": str(bundle), "runner": {"oxq": str(bundle / "runner/.venv/bin/oxq")}}

        monkeypatch.setattr(agent_module, "build_sdk_bundle", build_orphan)
        skill_dir = home / ".cursor/skills/build-strategy-spec"
        external = root / "external-skill"
        skill_dir.replace(external)
        skill_dir.symlink_to(external, target_is_directory=True)

        result = runner.invoke(
            main,
            ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
        )

        assert result.exit_code == 1
        assert manifest_path.read_bytes() == manifest_before
        assert _snapshot_tree(bundles_root) == bundles_before


def test_agent_upgrade_backup_cleanup_failure_is_post_commit_housekeeping(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    original_remove = agent_module._remove_upgrade_path

    def fail_backup_cleanup(path: Path) -> None:
        if ".backup-" in path.name:
            raise OSError("injected backup cleanup failure")
        original_remove(path)

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", fail_backup_cleanup)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "retained backup" in result.output
    installed = home / ".cursor/skills/build-strategy-spec/SKILL.md"
    assert "new workflow" in installed.read_text(encoding="utf-8")
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    assert manifest["targets"]["cursor"]["skills"][0]["managed_tree_sha256"]
    assert any(".backup-" in path.name for path in home.rglob("*"))

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", original_remove)
    status = CliRunner().invoke(main, ["agent", "status", "--json"])

    assert status.exit_code == 0, status.output
    json.loads(status.output)
    assert not agent_module.lifecycle_transaction_path().exists()
    assert not any(".backup-" in path.name for path in home.rglob("*"))


def test_agent_upgrade_dry_run_does_not_write_non_writable_target(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".cursor"
    target_root.chmod(stat.S_IRUSR | stat.S_IXUSR)

    def snapshot() -> dict[str, tuple[int, int, bytes | None]]:
        paths = [target_root, *sorted(target_root.rglob("*"))]
        return {
            path.relative_to(target_root.parent).as_posix(): (
                path.lstat().st_mode,
                path.lstat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in paths
        }

    before = snapshot()
    try:
        result = CliRunner().invoke(
            main,
            [
                "agent",
                "upgrade",
                "--target",
                "cursor",
                "--from-local",
                str(source_v2),
                "--dry-run",
                "--yes",
            ],
        )
        after = snapshot()
    finally:
        target_root.chmod(stat.S_IRWXU)

    assert result.exit_code == 0, result.output
    assert after == before


def test_agent_upgrade_single_target_preserves_existing_target_profile(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    opencode_install = CliRunner().invoke(
        main,
        [
            "agent",
            "install",
            "--target",
            "opencode",
            "--from-local",
            str(source_v1),
            "--profile",
            "multi-agent",
            "--yes",
        ],
    )
    assert opencode_install.exit_code == 0, opencode_install.output
    trae_install = CliRunner().invoke(
        main,
        [
            "agent",
            "install",
            "--target",
            "trae",
            "--from-local",
            str(source_v1),
            "--profile",
            "standalone-agent",
            "--yes",
        ],
    )
    assert trae_install.exit_code == 0, trae_install.output
    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    assert manifest["agent_profile"] == "standalone-agent"
    assert manifest["targets"]["opencode"]["agent_profile"] == "multi-agent"

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    upgraded_manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    assert upgraded_manifest["agent_profile"] == "standalone-agent"
    assert upgraded_manifest["targets"]["opencode"]["agent_profile"] == "multi-agent"
    agent_config = (home / ".config/open-xquant/agent.yaml").read_text(encoding="utf-8")
    assert "agent_profile: standalone-agent" in agent_config
    instructions = (home / ".config/opencode/AGENTS.md").read_text(encoding="utf-8")
    assert "For open-xquant workflows, prefer SubAgents by default" in instructions
    installed = home / ".config/opencode/skills/build-strategy-spec/SKILL.md"
    assert "new workflow" in installed.read_text(encoding="utf-8")


def test_agent_upgrade_missing_target_does_not_build_or_update_sdk_bundle(monkeypatch, tmp_path, fake_sdk_bundle) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    manifest_path = home / ".config/open-xquant/agent-install.json"
    config_path = home / ".config/open-xquant/agent.yaml"
    manifest_before = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_before = config_path.read_text(encoding="utf-8")
    build_calls_before = list(fake_sdk_bundle)

    result = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(source_v2), "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert "opencode: not installed" in result.output
    assert fake_sdk_bundle == build_calls_before
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest_before
    assert config_path.read_text(encoding="utf-8") == config_before


def test_agent_upgrade_tracks_previous_sdk_bundle(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    _write_source(source_v1, "old workflow")
    _write_source(source_v2, "new workflow")
    monkeypatch.setenv("HOME", str(home))

    def build(source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict:
        bundle_id = f"bundle-{source_root.name}"
        root = config_root / "sdk-bundles" / bundle_id
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
            "id": bundle_id,
            "root": str(root),
            "profile": "full-research",
            "extras": ["chart", "scipy", "yfinance", "akshare", "tushare", "live", "mcp", "agent"],
            "excluded_extras": ["dev", "docs", "talib"],
            "wheel": {"path": str(wheel), "sha256": "wheel-sha", "version": "0.1.0", "source_commit": "commit-sha"},
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

    monkeypatch.setattr("oxq.cli.agent.build_sdk_bundle", build)

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "cursor", "--from-local", str(source_v1), "--yes"],
    )
    assert install.exit_code == 0, install.output
    upgrade = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "cursor", "--from-local", str(source_v2), "--yes"],
    )
    assert upgrade.exit_code == 0, upgrade.output

    manifest = json.loads((home / ".config/open-xquant/agent-install.json").read_text(encoding="utf-8"))
    assert [bundle["id"] for bundle in manifest["sdk_bundles"]] == ["bundle-source-v1", "bundle-source-v2"]
    assert manifest["sdk_bundle"]["id"] == "bundle-source-v2"


def test_upgrade_source_uses_safe_cache_path_for_path_like_ref(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    source = tmp_path / "cloned"
    _write_source(source, "from git")
    clone_destinations: list[Path] = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("oxq.cli.agent.resolve_source_root", lambda _path: source)

    def fake_run(cmd, check):
        assert check is True
        clone_destinations.append(Path(cmd[-1]).resolve())

    monkeypatch.setattr("oxq.cli.agent.subprocess.run", fake_run)

    result = _upgrade_source(None, "https://example.invalid/repo.git", "..")

    cache_root = (home / ".config/open-xquant/cache/open-xquant").resolve()
    assert result == source
    assert clone_destinations
    assert clone_destinations[0].is_relative_to(cache_root)
    assert clone_destinations[0] != cache_root.parent
