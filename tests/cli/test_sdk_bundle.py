from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from oxq.cli.sdk_bundle import _uv_cmd, _verify_bundle, build_sdk_bundle, install_workspace_sdk, remove_sdk_bundle


def _write_valid_bundle(root) -> dict:
    wheel = root / "dist/open_xquant-0.1.0-py3-none-any.whl"
    lock = root / "requirements.lock.txt"
    packages = root / "packages.json"
    runner_python = root / "runner/.venv/bin/python"
    runner = root / "runner/.venv/bin/oxq"
    wheel.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    wheel.write_text("wheel", encoding="utf-8")
    lock.write_text("open-xquant @ file://wheel\n", encoding="utf-8")
    packages.write_text("[]\n", encoding="utf-8")
    runner_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner_python.chmod(0o755)
    runner.chmod(0o755)
    return {
        "id": root.name,
        "root": str(root),
        "profile": "full-research",
        "wheel": {
            "path": str(wheel),
            "sha256": hashlib.sha256(b"wheel").hexdigest(),
            "version": "0.1.0",
        },
        "dependencies": {
            "lock_file": str(lock),
            "lock_sha256": hashlib.sha256(b"open-xquant @ file://wheel\n").hexdigest(),
            "packages_file": str(packages),
        },
        "runner": {
            "python": str(runner_python),
            "oxq": str(runner),
        },
        "uv_cache_dir": str(root / "uv-cache"),
    }


def test_uv_cmd_isolates_from_caller_project(tmp_path) -> None:
    assert _uv_cmd(["pip", "compile", "requirements.in"], directory=tmp_path) == [
        "uv",
        "--directory",
        str(tmp_path),
        "--no-config",
        "pip",
        "compile",
        "requirements.in",
    ]


def _fake_build_run(commands: list[list[str]]) -> Any:
    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if cmd[0] == "uv" and "build" in cmd:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "open_xquant-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
        if cmd[0] == "uv" and "compile" in cmd:
            output = Path(cmd[cmd.index("--output-file") + 1])
            output.write_text("lock\n", encoding="utf-8")
        stdout = "[]" if cmd[0] == "uv" and "list" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return run


def test_build_sdk_bundle_resolves_lock_with_runner_python(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'open-xquant'\nversion = '0.1.0'\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    build_sdk_bundle(source, config_root)

    compile_cmd = next(cmd for cmd in commands if cmd[0] == "uv" and "compile" in cmd)
    assert compile_cmd[compile_cmd.index("--python") + 1] == sys.executable


def test_build_sdk_bundle_uses_installed_distribution_without_project_metadata(monkeypatch, tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    config_root = tmp_path / "config/open-xquant"
    (source / "agent/skills").mkdir(parents=True)
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    site = tmp_path / "fake-site"
    dist_info = site / "open_xquant-0.2.0.dist-info"
    (site / "oxq").mkdir(parents=True)
    (site / "agent/skills").mkdir(parents=True)
    dist_info.mkdir(parents=True)
    (site / "oxq/__init__.py").write_text("", encoding="utf-8")
    (site / "agent/skills/strategy-builder.md").write_text("# Strategy Builder\n", encoding="utf-8")
    (dist_info / "METADATA").write_text("Name: open-xquant\nVersion: 0.2.0\n", encoding="utf-8")
    (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    class FakeDistribution:
        version = "0.2.0"
        metadata = {"Name": "open-xquant"}
        files = [
            Path("oxq/__init__.py"),
            Path("agent/skills/strategy-builder.md"),
            Path("open_xquant-0.2.0.dist-info/METADATA"),
            Path("open_xquant-0.2.0.dist-info/WHEEL"),
            Path("open_xquant-0.2.0.dist-info/RECORD"),
        ]

        def locate_file(self, path: Path) -> Path:
            return site / path

    monkeypatch.setattr("oxq.cli.sdk_bundle.metadata.distribution", lambda _name: FakeDistribution())
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    payload = build_sdk_bundle(source, config_root)

    assert payload["wheel"]["version"] == "0.2.0"
    assert payload["id"] != bundle["id"]
    assert not any(cmd[0] == "uv" and "build" in cmd for cmd in commands)
    with zipfile.ZipFile(payload["wheel"]["path"]) as wheel:
        assert "agent/skills/strategy-builder.md" in wheel.namelist()


def test_build_sdk_bundle_requires_project_or_installed_distribution(monkeypatch, tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    (source / "agent/skills").mkdir(parents=True)
    monkeypatch.setattr(
        "oxq.cli.sdk_bundle.metadata.distribution",
        lambda _name: (_ for _ in ()).throw(metadata.PackageNotFoundError("open-xquant")),
    )

    with pytest.raises(Exception, match="not a project checkout"):
        build_sdk_bundle(source, tmp_path / "config/open-xquant")


def test_install_workspace_sdk_force_keeps_existing_virtualenv(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    config_root = home / ".config/open-xquant"
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    runner_python = venv / "bin/python"
    runner = venv / "bin/oxq"
    runner.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    (venv / "sentinel.txt").write_text("keep\n", encoding="utf-8")
    runner_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner_python.chmod(0o755)
    runner.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", lambda cmd: None)

    install_workspace_sdk(workspace, venv, force=True)

    assert (venv / "sentinel.txt").read_text(encoding="utf-8") == "keep\n"


def test_install_workspace_sdk_uses_copy_link_mode(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    config_root = home / ".config/open-xquant"
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    runner_python = venv / "bin/python"
    runner = venv / "bin/oxq"
    runner.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    runner_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner_python.chmod(0o755)
    runner.chmod(0o755)
    commands: list[list[str]] = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", lambda cmd: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0))

    install_workspace_sdk(workspace, venv)

    install_cmd = next(cmd for cmd in commands if cmd[0] == "uv" and "install" in cmd)
    assert install_cmd[install_cmd.index("--link-mode") + 1] == "copy"


def test_remove_sdk_bundle_refuses_active_cached_runner(monkeypatch, tmp_path) -> None:
    config_root = tmp_path / "config/open-xquant"
    root = config_root / "sdk-bundles/bundle-test"
    bundle = _write_valid_bundle(root)
    monkeypatch.setattr("oxq.cli.sdk_bundle.sys.executable", str(root / "runner/.venv/bin/python"))
    monkeypatch.setattr(
        "oxq.cli.sdk_bundle.shutil.rmtree",
        lambda _path: pytest.fail("active cached runner must not be deleted"),
    )

    assert remove_sdk_bundle(bundle, config_root) is False


def test_verify_bundle_requires_runner_python(tmp_path) -> None:
    root = tmp_path / "sdk-bundles/bundle"
    wheel = root / "dist/open_xquant-0.1.0-py3-none-any.whl"
    lock = root / "requirements.lock.txt"
    packages = root / "packages.json"
    runner = root / "runner/.venv/bin/oxq"
    wheel.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    wheel.write_text("wheel", encoding="utf-8")
    lock.write_text("open-xquant @ file://wheel\n", encoding="utf-8")
    packages.write_text("[]\n", encoding="utf-8")
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)

    bundle = {
        "root": str(root),
        "wheel": {
            "path": str(wheel),
            "sha256": hashlib.sha256(b"wheel").hexdigest(),
        },
        "dependencies": {
            "lock_file": str(lock),
            "lock_sha256": hashlib.sha256(b"open-xquant @ file://wheel\n").hexdigest(),
            "packages_file": str(packages),
        },
        "runner": {
            "python": str(root / "runner/.venv/bin/python"),
            "oxq": str(runner),
        },
    }

    with pytest.raises(Exception, match="SDK bundle file is missing"):
        _verify_bundle(bundle)


def test_verify_bundle_allows_runner_python_symlink(tmp_path) -> None:
    root = tmp_path / "sdk-bundles/bundle"
    wheel = root / "dist/open_xquant-0.1.0-py3-none-any.whl"
    lock = root / "requirements.lock.txt"
    packages = root / "packages.json"
    runner_python = root / "runner/.venv/bin/python"
    runner = root / "runner/.venv/bin/oxq"
    outside_python = tmp_path / "python-target"
    wheel.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    wheel.write_text("wheel", encoding="utf-8")
    lock.write_text("open-xquant @ file://wheel\n", encoding="utf-8")
    packages.write_text("[]\n", encoding="utf-8")
    outside_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    outside_python.chmod(0o755)
    runner.chmod(0o755)
    try:
        runner_python.symlink_to(outside_python)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    bundle = {
        "root": str(root),
        "wheel": {
            "path": str(wheel),
            "sha256": hashlib.sha256(b"wheel").hexdigest(),
        },
        "dependencies": {
            "lock_file": str(lock),
            "lock_sha256": hashlib.sha256(b"open-xquant @ file://wheel\n").hexdigest(),
            "packages_file": str(packages),
        },
        "runner": {
            "python": str(runner_python),
            "oxq": str(runner),
        },
    }

    _verify_bundle(bundle)
