from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from oxq.cli.sdk_bundle import (
    _build_installed_distribution_wheel,
    _is_safe_wheel_archive_name,
    _uv_cmd,
    _verify_bundle,
    build_sdk_bundle,
    install_workspace_sdk,
    remove_sdk_bundle,
)


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
        "extras": ["agent", "akshare", "chart", "live", "mcp", "scipy", "yfinance"],
        "excluded_extras": ["dev", "docs", "talib"],
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


def test_build_sdk_bundle_selects_all_research_extras_except_excluded(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "name = 'open-xquant'",
                "version = '0.1.0'",
                "",
                "[project.optional-dependencies]",
                "chart = ['matplotlib']",
                "researchx = ['duckdb']",
                "dev = ['pytest']",
                "docs = ['mkdocs-material']",
                "talib = ['TA-Lib']",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    payload = build_sdk_bundle(source, config_root)

    req = (Path(payload["root"]) / "requirements.in").read_text(encoding="utf-8")
    assert req.startswith("open-xquant[chart,researchx] @ ")
    assert "dev" not in req
    assert "docs" not in req
    assert "talib" not in req
    assert payload["extras"] == ["chart", "researchx"]
    assert payload["excluded_extras"] == ["dev", "docs", "talib"]


def test_build_sdk_bundle_uses_shared_uv_cache_across_bundle_roots(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'open-xquant'\nversion = '0.1.0'\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    payload = build_sdk_bundle(source, config_root)

    shared_cache = config_root / "sdk-cache" / "uv"
    bundle_root = Path(payload["root"])
    assert Path(payload["uv_cache_dir"]) == shared_cache
    assert not Path(payload["uv_cache_dir"]).is_relative_to(bundle_root)
    cache_args = [
        Path(cmd[cmd.index("--cache-dir") + 1])
        for cmd in commands
        if cmd[0] == "uv" and "--cache-dir" in cmd
    ]
    assert cache_args
    assert set(cache_args) == {shared_cache}


def test_build_sdk_bundle_reuses_runner_venv_when_dependency_lock_is_unchanged(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'open-xquant'\nversion = '0.1.0'\n", encoding="utf-8")
    bundle = _write_valid_bundle(config_root / "sdk-bundles/old-bundle")
    old_lock = "\n".join(
        [
            "dep==1.0 \\",
            "    --hash=sha256:old",
            "open-xquant @ file:///old/open_xquant-0.1.0-py3-none-any.whl \\",
            "    --hash=sha256:oldwheel",
            "zdep==2.0 \\",
            "    --hash=sha256:z",
            "",
        ]
    )
    new_lock = old_lock.replace("file:///old/open_xquant-0.1.0-py3-none-any.whl", "file:///new/open_xquant-0.1.0-py3-none-any.whl")
    lock_path = Path(bundle["dependencies"]["lock_file"])
    lock_path.write_text(old_lock, encoding="utf-8")
    bundle["dependencies"]["lock_sha256"] = hashlib.sha256(old_lock.encode("utf-8")).hexdigest()
    outside_venv = tmp_path / "outside-venv"
    (outside_venv / "bin").mkdir(parents=True)
    (outside_venv / "marker.txt").write_text("do not copy\n", encoding="utf-8")
    (outside_venv / "bin/python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (outside_venv / "bin/oxq").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bundle["runner"]["venv"] = str(outside_venv)
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    commands: list[list[str]] = []

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "-c" and "version_info" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 0, stdout="3.13\n", stderr="")
        if cmd[0] == "uv" and "build" in cmd:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "open_xquant-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
        if cmd[0] == "uv" and "compile" in cmd:
            output = Path(cmd[cmd.index("--output-file") + 1])
            output.write_text(new_lock, encoding="utf-8")
        if cmd[0] == "uv" and "sync" in cmd:
            pytest.fail(f"unchanged dependency lock should reuse the previous runner venv: {cmd}")
        stdout = "[]" if cmd[0] == "uv" and "list" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

    payload = build_sdk_bundle(source, config_root)

    install_cmd = next(cmd for cmd in commands if cmd[0] == "uv" and "install" in cmd)
    assert "--reinstall" in install_cmd
    assert "--no-deps" in install_cmd
    assert payload["id"] != bundle["id"]
    assert Path(payload["runner"]["python"]).exists()
    assert not (Path(payload["runner"]["venv"]) / "marker.txt").exists()


def test_build_sdk_bundle_rebuilds_runner_venv_when_python_version_changes(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'open-xquant'\nversion = '0.1.0'\n", encoding="utf-8")
    bundle = _write_valid_bundle(config_root / "sdk-bundles/old-bundle")
    old_lock = "open-xquant @ file:///old/open_xquant-0.1.0-py3-none-any.whl \\\n    --hash=sha256:oldwheel\n"
    new_lock = old_lock.replace("file:///old/open_xquant-0.1.0-py3-none-any.whl", "file:///new/open_xquant-0.1.0-py3-none-any.whl")
    lock_path = Path(bundle["dependencies"]["lock_file"])
    lock_path.write_text(old_lock, encoding="utf-8")
    bundle["dependencies"]["lock_sha256"] = hashlib.sha256(old_lock.encode("utf-8")).hexdigest()
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    commands: list[list[str]] = []

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "-c" and "version_info" in cmd[2]:
            version = "3.12\n" if cmd[0] == bundle["runner"]["python"] else "3.13\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=version, stderr="")
        if cmd[0] == "uv" and "build" in cmd:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "open_xquant-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
        if cmd[0] == "uv" and "compile" in cmd:
            output = Path(cmd[cmd.index("--output-file") + 1])
            output.write_text(new_lock, encoding="utf-8")
        stdout = "[]" if cmd[0] == "uv" and "list" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

    build_sdk_bundle(source, config_root)

    assert any(cmd[0] == "uv" and "sync" in cmd for cmd in commands)


def test_build_sdk_bundle_rebuilds_runner_venv_when_cached_packages_are_polluted(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'open-xquant'\nversion = '0.1.0'\n", encoding="utf-8")
    bundle = _write_valid_bundle(config_root / "sdk-bundles/old-bundle")
    old_lock = "open-xquant @ file:///old/open_xquant-0.1.0-py3-none-any.whl \\\n    --hash=sha256:oldwheel\n"
    new_lock = old_lock.replace("file:///old/open_xquant-0.1.0-py3-none-any.whl", "file:///new/open_xquant-0.1.0-py3-none-any.whl")
    lock_path = Path(bundle["dependencies"]["lock_file"])
    lock_path.write_text(old_lock, encoding="utf-8")
    bundle["dependencies"]["lock_sha256"] = hashlib.sha256(old_lock.encode("utf-8")).hexdigest()
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    commands: list[list[str]] = []

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "-c" and "version_info" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 0, stdout="3.13\n", stderr="")
        if cmd[0] == "uv" and "build" in cmd:
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "open_xquant-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
        if cmd[0] == "uv" and "compile" in cmd:
            output = Path(cmd[cmd.index("--output-file") + 1])
            output.write_text(new_lock, encoding="utf-8")
        stdout = '[{"name": "unexpected", "version": "1.0"}]' if cmd[0] == "uv" and "list" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

    build_sdk_bundle(source, config_root)

    assert any(cmd[0] == "uv" and "sync" in cmd for cmd in commands)


def test_build_sdk_bundle_rebuilds_malformed_existing_manifest(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'open-xquant'\nversion = '0.1.0'\n", encoding="utf-8")
    wheel_sha = hashlib.sha256(b"wheel").hexdigest()
    bundle_root = config_root / "sdk-bundles" / f"0.1.0-no-git-{wheel_sha[:12]}"
    bundle_root.mkdir(parents=True)
    (bundle_root / "manifest.json").write_text("{broken", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    payload = build_sdk_bundle(source, config_root)

    assert payload["id"] == bundle_root.name
    assert any(cmd[0] == "uv" and "compile" in cmd for cmd in commands)


def test_build_sdk_bundle_refuses_to_replace_active_existing_bundle(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    config_root = tmp_path / "config/open-xquant"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'open-xquant'\nversion = '0.1.0'\n", encoding="utf-8")
    wheel_sha = hashlib.sha256(b"wheel").hexdigest()
    bundle_root = config_root / "sdk-bundles" / f"0.1.0-no-git-{wheel_sha[:12]}"
    runner_python = bundle_root / "runner/.venv/bin/python"
    runner_python.parent.mkdir(parents=True)
    runner_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner_python.chmod(0o755)
    (bundle_root / "manifest.json").write_text("{broken", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))
    monkeypatch.setattr("oxq.cli.sdk_bundle.sys.executable", str(runner_python))
    real_rmtree = shutil.rmtree

    def rmtree(path: Path) -> None:
        if path == bundle_root:
            pytest.fail("active cached bundle must not be deleted during rebuild")
        real_rmtree(path)

    monkeypatch.setattr("oxq.cli.sdk_bundle.shutil.rmtree", rmtree)

    with pytest.raises(Exception, match="active cached SDK bundle"):
        build_sdk_bundle(source, config_root)


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
    (site / "agent/skills/build-strategy-spec").mkdir(parents=True)
    dist_info.mkdir(parents=True)
    escaped_script = tmp_path / "bin/oxq"
    escaped_script.parent.mkdir(parents=True)
    escaped_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (site / "oxq/__init__.py").write_text("", encoding="utf-8")
    (site / "agent/skills/build-strategy-spec/SKILL.md").write_text("# Strategy Builder\n", encoding="utf-8")
    (dist_info / "METADATA").write_text("Name: open-xquant\nVersion: 0.2.0\n", encoding="utf-8")
    (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    class FakeMetadata(dict):
        def get_all(self, key: str, default: list[str] | None = None) -> list[str]:
            if key == "Provides-Extra":
                return ["chart", "researchx", "dev", "docs", "talib"]
            return default or []

    class FakeDistribution:
        version = "0.2.0"
        metadata = FakeMetadata({"Name": "open-xquant"})
        files = [
            Path("oxq/__init__.py"),
            Path("agent/skills/build-strategy-spec/SKILL.md"),
            Path("../../../bin/oxq"),
            Path("open_xquant-0.2.0.dist-info/METADATA"),
            Path("open_xquant-0.2.0.dist-info/WHEEL"),
            Path("open_xquant-0.2.0.dist-info/RECORD"),
        ]

        def locate_file(self, path: Path) -> Path:
            if str(path).replace("\\", "/") == "../../../bin/oxq":
                return escaped_script
            return site / path

    monkeypatch.setattr("oxq.cli.sdk_bundle.metadata.distribution", lambda _name: FakeDistribution())
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    payload = build_sdk_bundle(source, config_root)

    assert payload["wheel"]["version"] == "0.2.0"
    assert payload["id"] != bundle["id"]
    assert not any(cmd[0] == "uv" and "build" in cmd for cmd in commands)
    req = (Path(payload["root"]) / "requirements.in").read_text(encoding="utf-8")
    assert req.startswith("open-xquant[chart,researchx] @ ")
    with zipfile.ZipFile(payload["wheel"]["path"]) as wheel:
        names = wheel.namelist()
        assert "agent/skills/build-strategy-spec/SKILL.md" in names
        assert "../../../bin/oxq" not in names


def test_build_installed_distribution_wheel_uses_deterministic_archive_metadata(monkeypatch, tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    site = tmp_path / "fake-site"
    dist_info = site / "open_xquant-0.2.0.dist-info"
    (source / "agent/skills").mkdir(parents=True)
    (site / "oxq").mkdir(parents=True)
    dist_info.mkdir(parents=True)
    (site / "oxq/a.py").write_text("a = 1\n", encoding="utf-8")
    (site / "oxq/z.py").write_text("z = 1\n", encoding="utf-8")
    (dist_info / "METADATA").write_text("Name: open-xquant\nVersion: 0.2.0\n", encoding="utf-8")
    (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")

    class FakeDistribution:
        version = "0.2.0"
        metadata = {"Name": "open-xquant"}
        files = [
            Path("oxq/z.py"),
            Path("open_xquant-0.2.0.dist-info/WHEEL"),
            Path("oxq/a.py"),
            Path("open_xquant-0.2.0.dist-info/METADATA"),
            Path("open_xquant-0.2.0.dist-info/RECORD"),
        ]

        def locate_file(self, path: Path) -> Path:
            return site / path

    monkeypatch.setattr("oxq.cli.sdk_bundle.metadata.distribution", lambda _name: FakeDistribution())

    wheel_path = _build_installed_distribution_wheel(source, tmp_path / "dist")

    with zipfile.ZipFile(wheel_path) as wheel:
        infos = wheel.infolist()
        names = [info.filename for info in infos]
        assert names == sorted(names)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)


def test_wheel_archive_name_rejects_drive_relative_paths() -> None:
    assert _is_safe_wheel_archive_name("C:pkg/file.py") is False
    assert _is_safe_wheel_archive_name("C:/pkg/file.py") is False


def test_build_sdk_bundle_reuses_matching_cached_bundle_for_installed_source(monkeypatch, tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    config_root = tmp_path / "config/open-xquant"
    (source / "agent/skills").mkdir(parents=True)
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    commands: list[list[str]] = []

    class FakeDistribution:
        version = "0.1.0"

    def build_installed_wheel(_source_root: Path, dist_tmp: Path) -> Path:
        dist_tmp.mkdir(parents=True, exist_ok=True)
        wheel = dist_tmp / "open_xquant-0.1.0-py3-none-any.whl"
        wheel.write_text("wheel", encoding="utf-8")
        return wheel

    monkeypatch.setattr("oxq.cli.sdk_bundle.metadata.distribution", lambda _name: FakeDistribution())
    monkeypatch.setattr("oxq.cli.sdk_bundle._build_installed_distribution_wheel", build_installed_wheel)
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", lambda cmd: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0))

    assert build_sdk_bundle(source, config_root) == bundle
    assert commands == [
        [bundle["runner"]["python"], "-c", "import oxq"],
        [bundle["runner"]["oxq"], "--help"],
    ]


def test_build_sdk_bundle_rebuilds_stale_same_version_installed_bundle(monkeypatch, tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    config_root = tmp_path / "config/open-xquant"
    (source / "agent/skills").mkdir(parents=True)
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    site = tmp_path / "fake-site"
    dist_info = site / "open_xquant-0.1.0.dist-info"
    (site / "oxq").mkdir(parents=True)
    dist_info.mkdir(parents=True)
    (site / "oxq/__init__.py").write_text("new installed content\n", encoding="utf-8")
    (dist_info / "METADATA").write_text("Name: open-xquant\nVersion: 0.1.0\n", encoding="utf-8")
    (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    class FakeDistribution:
        version = "0.1.0"
        metadata = {"Name": "open-xquant"}
        files = [
            Path("oxq/__init__.py"),
            Path("open_xquant-0.1.0.dist-info/METADATA"),
            Path("open_xquant-0.1.0.dist-info/WHEEL"),
            Path("open_xquant-0.1.0.dist-info/RECORD"),
        ]

        def locate_file(self, path: Path) -> Path:
            return site / path

    monkeypatch.setattr("oxq.cli.sdk_bundle.metadata.distribution", lambda _name: FakeDistribution())
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    payload = build_sdk_bundle(source, config_root)

    assert payload["id"] != bundle["id"]
    assert payload["wheel"]["sha256"] != bundle["wheel"]["sha256"]
    assert any(cmd[0] == "uv" and "compile" in cmd for cmd in commands)


def test_build_sdk_bundle_treats_invalid_cached_installed_bundle_as_cache_miss(monkeypatch, tmp_path) -> None:
    source = tmp_path / "site-packages/open_xquant"
    config_root = tmp_path / "config/open-xquant"
    (source / "agent/skills").mkdir(parents=True)
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    Path(bundle["dependencies"]["lock_file"]).unlink()
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    site = tmp_path / "fake-site"
    dist_info = site / "open_xquant-0.1.0.dist-info"
    (site / "oxq").mkdir(parents=True)
    dist_info.mkdir(parents=True)
    (site / "oxq/__init__.py").write_text("", encoding="utf-8")
    (dist_info / "METADATA").write_text("Name: open-xquant\nVersion: 0.1.0\nProvides-Extra: chart\n", encoding="utf-8")
    (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    class FakeMetadata(dict):
        def get_all(self, key: str, default: list[str] | None = None) -> list[str]:
            if key == "Provides-Extra":
                return ["chart"]
            return default or []

    class FakeDistribution:
        version = "0.1.0"
        metadata = FakeMetadata({"Name": "open-xquant"})
        files = [
            Path("oxq/__init__.py"),
            Path("open_xquant-0.1.0.dist-info/METADATA"),
            Path("open_xquant-0.1.0.dist-info/WHEEL"),
            Path("open_xquant-0.1.0.dist-info/RECORD"),
        ]

        def locate_file(self, path: Path) -> Path:
            return site / path

    monkeypatch.setattr("oxq.cli.sdk_bundle.metadata.distribution", lambda _name: FakeDistribution())
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", _fake_build_run(commands))

    payload = build_sdk_bundle(source, config_root)

    assert payload["id"] != bundle["id"]
    assert payload["extras"] == ["chart"]
    assert any(cmd[0] == "uv" and "compile" in cmd for cmd in commands)


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

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if len(cmd) >= 3 and cmd[1] == "-c" and "version_info" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 0, stdout="3.13\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

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

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        stdout = "3.13\n" if len(cmd) >= 3 and cmd[1] == "-c" and "version_info" in cmd[2] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

    install_workspace_sdk(workspace, venv)

    install_cmd = next(cmd for cmd in commands if cmd[0] == "uv" and "install" in cmd)
    assert install_cmd[install_cmd.index("--link-mode") + 1] == "copy"


def test_install_workspace_sdk_rejects_existing_venv_missing_python(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    config_root = home / ".config/open-xquant"
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    bundle = _write_valid_bundle(config_root / "sdk-bundles/bundle-test")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    (venv / "sentinel.txt").write_text("keep\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "uv" and "venv" in cmd:
            pytest.fail(f"broken existing venv must be rejected before running: {cmd}")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

    with pytest.raises(Exception, match="missing Python interpreter"):
        install_workspace_sdk(workspace, venv)

    assert (venv / "sentinel.txt").read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize(
    "sdk_venv",
    [
        ".open-xquant",
        "strategy_specs",
        "runs",
        "reports",
        "comparisons",
        "experiments.jsonl",
        "strategy_spec.yaml",
        "AGENTS.md",
    ],
)
def test_install_workspace_sdk_rejects_reserved_workspace_paths(tmp_path, sdk_venv) -> None:
    with pytest.raises(Exception, match="reserved workspace path"):
        install_workspace_sdk(tmp_path / "workspace", tmp_path / "workspace" / sdk_venv)


def test_chart_extra_includes_seaborn_for_statistical_report_charts() -> None:
    data = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "seaborn>=0.13" in data


def test_install_workspace_sdk_rejects_existing_venv_python_version_mismatch(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setenv("HOME", str(home))

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if len(cmd) >= 3 and cmd[1] == "-c" and "version_info" in cmd[2]:
            version = "3.12\n" if cmd[0] == bundle["runner"]["python"] else "3.13\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=version, stderr="")
        if cmd[0] == "uv" and "install" in cmd:
            pytest.fail(f"mismatched existing venv must be rejected before install: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)

    with pytest.raises(Exception, match="Python version"):
        install_workspace_sdk(workspace, venv)


def test_install_workspace_sdk_rejects_unmanaged_bundle_root(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    config_root = home / ".config/open-xquant"
    workspace = tmp_path / "workspace"
    bundle = _write_valid_bundle(tmp_path / "outside-bundle")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "agent-install.json").write_text(json.dumps({"sdk_bundle": bundle}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        "oxq.cli.sdk_bundle._run",
        lambda _cmd: pytest.fail("unmanaged bundle roots must be rejected before executing bundle commands"),
    )

    with pytest.raises(Exception, match="SDK bundle root escapes"):
        install_workspace_sdk(workspace, workspace / ".venv")


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


def test_remove_sdk_bundle_converts_verification_oserror_to_safe_failure(monkeypatch, tmp_path) -> None:
    config_root = tmp_path / "config/open-xquant"
    root = config_root / "sdk-bundles/bundle-test"
    bundle = _write_valid_bundle(root)

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd == [bundle["runner"]["python"], "-c", "import oxq"]:
            raise OSError("runner is no longer executable")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("oxq.cli.sdk_bundle._run", run)
    monkeypatch.setattr(
        "oxq.cli.sdk_bundle.shutil.rmtree",
        lambda _path: pytest.fail("corrupted bundle must not be deleted"),
    )

    assert remove_sdk_bundle(bundle, config_root) is False
    assert root.exists()


def test_verify_bundle_rejects_symlinked_non_runner_paths(tmp_path) -> None:
    root = tmp_path / "sdk-bundles/bundle"
    outside = tmp_path / "outside"
    bundle = _write_valid_bundle(root)
    outside.mkdir()
    outside_lock = outside / "requirements.lock.txt"
    outside_lock.write_text("outside-lock\n", encoding="utf-8")
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    bundle["dependencies"]["lock_file"] = str(link / "requirements.lock.txt")
    bundle["dependencies"]["lock_sha256"] = hashlib.sha256(b"outside-lock\n").hexdigest()

    with pytest.raises(Exception, match="escapes bundle root"):
        _verify_bundle(bundle)


def test_remove_sdk_bundle_refuses_active_symlinked_cached_runner(monkeypatch, tmp_path) -> None:
    config_root = tmp_path / "config/open-xquant"
    root = config_root / "sdk-bundles/bundle-test"
    bundle = _write_valid_bundle(root)
    base_python = tmp_path / "base-python"
    runner_python = root / "runner/.venv/bin/python"
    base_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    base_python.chmod(0o755)
    runner_python.unlink()
    try:
        runner_python.symlink_to(base_python)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    monkeypatch.setattr("oxq.cli.sdk_bundle.sys.executable", str(runner_python))
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", lambda cmd: subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(
        "oxq.cli.sdk_bundle.shutil.rmtree",
        lambda _path: pytest.fail("active symlinked cached runner must not be deleted"),
    )

    assert remove_sdk_bundle(bundle, config_root) is False


def test_remove_sdk_bundle_resolves_root_before_purge(monkeypatch, tmp_path) -> None:
    config_root = tmp_path / "config/open-xquant"
    escaped_root = config_root / "sdk-bundles/../outside"
    actual_root = config_root / "outside"
    bundle = _write_valid_bundle(escaped_root)
    monkeypatch.setattr(
        "oxq.cli.sdk_bundle.shutil.rmtree",
        lambda _path: pytest.fail("escaped bundle root must not be deleted"),
    )

    assert actual_root.exists()
    assert remove_sdk_bundle(bundle, config_root) is False
    assert actual_root.exists()


def test_remove_sdk_bundle_rejects_symlinked_root_before_purge(monkeypatch, tmp_path) -> None:
    config_root = tmp_path / "config/open-xquant"
    actual_root = tmp_path / "outside-bundle"
    bundle = _write_valid_bundle(actual_root)
    link_root = config_root / "sdk-bundles/link"
    link_root.parent.mkdir(parents=True)
    try:
        link_root.symlink_to(actual_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    bundle["root"] = str(link_root)
    bundle["wheel"]["path"] = str(link_root / "dist/open_xquant-0.1.0-py3-none-any.whl")
    bundle["dependencies"]["lock_file"] = str(link_root / "requirements.lock.txt")
    bundle["dependencies"]["packages_file"] = str(link_root / "packages.json")
    bundle["runner"]["python"] = str(link_root / "runner/.venv/bin/python")
    bundle["runner"]["oxq"] = str(link_root / "runner/.venv/bin/oxq")
    monkeypatch.setattr("oxq.cli.sdk_bundle._run", lambda cmd: subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(
        "oxq.cli.sdk_bundle.shutil.rmtree",
        lambda _path: pytest.fail("symlinked bundle root must not be deleted"),
    )

    assert remove_sdk_bundle(bundle, config_root) is False
    assert actual_root.exists()


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


def test_verify_bundle_wraps_runner_oserror(tmp_path) -> None:
    root = tmp_path / "sdk-bundles/bundle"
    bundle = _write_valid_bundle(root)
    Path(bundle["runner"]["python"]).chmod(0o644)

    with pytest.raises(Exception, match="Command failed"):
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
