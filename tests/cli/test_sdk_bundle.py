from __future__ import annotations

import hashlib
import json

import pytest

from oxq.cli.sdk_bundle import _uv_cmd, _verify_bundle, build_sdk_bundle, install_workspace_sdk


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


def test_build_sdk_bundle_reuses_cached_bundle_without_project_metadata(monkeypatch, tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    config_root = tmp_path / "config/open-xquant"
    (source / "agent/skills").mkdir(parents=True)
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    commands: list[list[str]] = []

    def run(cmd: list[str]) -> None:
        commands.append(cmd)
        assert cmd[0] != "uv"

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

    assert build_sdk_bundle(source, config_root) == bundle
    assert commands == [
        [bundle["runner"]["python"], "-c", "import oxq"],
        [bundle["runner"]["oxq"], "--help"],
    ]


def test_build_sdk_bundle_requires_project_or_cached_bundle(tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    (source / "agent/skills").mkdir(parents=True)

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
