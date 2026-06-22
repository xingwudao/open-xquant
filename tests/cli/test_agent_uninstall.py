from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from oxq.cli.main import main


def _write_source(root: Path) -> None:
    skills = root / "agent" / "skills"
    skills.mkdir(parents=True)
    (skills / "strategy-builder.md").write_text(
        "---\nname: strategy-builder\ndescription: Build quant strategies\n---\n\n# Strategy Builder\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def fake_sdk_bundle(monkeypatch):
    def build(source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict:
        del source_root
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
        wheel_sha = hashlib.sha256(b"wheel").hexdigest()
        lock_sha = hashlib.sha256(b"open-xquant @ file://wheel\n").hexdigest()
        return {
            "id": "bundle-test",
            "root": str(root),
            "profile": "full-research",
            "extras": ["chart", "scipy", "yfinance", "akshare", "live", "mcp", "agent"],
            "excluded_extras": ["dev", "docs", "talib"],
            "wheel": {"path": str(wheel), "sha256": wheel_sha, "version": "0.1.0", "source_commit": "commit-sha"},
            "dependencies": {
                "lock_file": str(lock),
                "lock_sha256": lock_sha,
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


def test_agent_uninstall_removes_only_managed_files(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    agents = home / ".config/opencode/AGENTS.md"
    agents.write_text(
        "user content\n" + agents.read_text(encoding="utf-8") + "\nmore user content\n",
        encoding="utf-8",
    )
    data_dir = home / ".oxq" / "data"
    data_dir.mkdir(parents=True)

    result = CliRunner().invoke(main, ["agent", "uninstall", "--target", "opencode", "--yes"])

    assert result.exit_code == 0, result.output
    assert not (home / ".config/opencode/skills/strategy-builder").exists()
    assert "open-xquant:begin" not in agents.read_text(encoding="utf-8")
    assert "user content" in agents.read_text(encoding="utf-8")
    assert data_dir.exists()
    assert (home / ".config/open-xquant/sdk-bundles/bundle-test").exists()
    assert (home / ".config/open-xquant/agent-install.json").exists()


def test_agent_uninstall_purge_config_removes_managed_sdk_bundle(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    bundle = home / ".config/open-xquant/sdk-bundles/bundle-test"
    assert bundle.exists()
    manifest = home / ".config/open-xquant/agent-install.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["sdk_bundle"]["root"] == str(bundle)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert not bundle.exists()
    assert not manifest.exists()
    assert not (home / ".config/open-xquant/agent.yaml").exists()


def test_agent_uninstall_keeps_manifest_when_sdk_bundle_purge_fails(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    bundle = home / ".config/open-xquant/sdk-bundles/bundle-test"
    manifest = home / ".config/open-xquant/agent-install.json"
    (bundle / "requirements.lock.txt").write_text("corrupted\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code != 0
    assert "Refusing to purge config" in result.output
    assert bundle.exists()
    assert manifest.exists()


def test_agent_uninstall_purge_refuses_active_cached_runner_before_mutating_targets(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    bundle = home / ".config/open-xquant/sdk-bundles/bundle-test"
    runner_python = bundle / "runner/.venv/bin/python"
    skill_dir = home / ".config/opencode/skills/strategy-builder"
    manifest = home / ".config/open-xquant/agent-install.json"
    monkeypatch.setattr("oxq.cli.sdk_bundle.sys.executable", str(runner_python))

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code != 0
    assert "active cached SDK runner" in result.output
    assert skill_dir.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["targets"]["opencode"]["installed"] is True
