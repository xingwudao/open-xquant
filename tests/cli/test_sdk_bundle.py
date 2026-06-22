from __future__ import annotations

import hashlib

import pytest

from oxq.cli.sdk_bundle import _uv_cmd, _verify_bundle


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
