from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from oxq.cli.agent import _upgrade_source
from oxq.cli.main import main


def _write_source(root: Path, body: str) -> None:
    skills = root / "agent" / "skills"
    skill_dir = skills / "build-strategy-spec"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: build-strategy-spec\ndescription: Build quant strategies\n---\n\n"
        f"# Strategy Builder\n\n{body}\n",
        encoding="utf-8",
    )


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
            "extras": ["chart", "scipy", "yfinance", "akshare", "live", "mcp", "agent"],
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
            "extras": ["chart", "scipy", "yfinance", "akshare", "live", "mcp", "agent"],
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
