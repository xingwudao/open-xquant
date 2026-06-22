"""SDK bundle helpers for Agent and research workspace setup."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import click

from oxq.cli.agent_manifest import expand_path, read_json_file, sha256_file, write_json_file, write_text_file

SDK_PROFILE = "full-research"
SDK_EXTRAS = ("chart", "scipy", "yfinance", "akshare", "live", "mcp", "agent")
EXCLUDED_EXTRAS = ("dev", "docs", "talib")


def default_config_dir() -> Path:
    return Path.home().joinpath(".config", "open-xquant").resolve()


def default_manifest_path() -> Path:
    return default_config_dir() / "agent-install.json"


def build_sdk_bundle(source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Build a cached open-xquant SDK bundle and runner environment."""

    source_root = source_root.resolve()
    config_root = config_root.resolve()
    version = _package_version(source_root)
    source_commit = _current_commit(source_root)
    if dry_run:
        bundle_root = config_root / "sdk-bundles" / f"{_slug(version)}-dry-run"
        oxq = _venv_executable(bundle_root / "runner" / ".venv", "oxq")
        return _bundle_payload(
            bundle_id=bundle_root.name,
            bundle_root=bundle_root,
            wheel_path=bundle_root / "dist" / f"open_xquant-{version}-py3-none-any.whl",
            wheel_sha="dry-run",
            version=version,
            source_commit=source_commit,
            lock_path=bundle_root / "requirements.lock.txt",
            lock_sha="dry-run",
            packages_path=bundle_root / "packages.json",
            packages_count=0,
            runner_venv=bundle_root / "runner" / ".venv",
            runner_python=_venv_executable(bundle_root / "runner" / ".venv", "python"),
            runner_oxq=oxq,
            uv_cache_dir=bundle_root / "uv-cache",
        )

    config_root.mkdir(parents=True, exist_ok=True)
    if not _is_buildable_source(source_root):
        cached_bundle = _installed_sdk_bundle(config_root)
        if cached_bundle is not None:
            return cached_bundle
        raise click.ClickException(
            "Cannot build the SDK bundle because the resolved open-xquant source "
            f"is not a project checkout: {source_root}. Re-run `oxq agent install` "
            "from an open-xquant checkout to refresh the cached SDK bundle."
        )

    tmp_root = config_root / "sdk-bundles" / ".build-tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    try:
        dist_tmp = tmp_root / "dist"
        _run(_uv_cmd(["build", "--wheel", "--out-dir", str(dist_tmp), "."], directory=source_root))
        wheel_tmp = _single_wheel(dist_tmp)
        wheel_sha = sha256_file(wheel_tmp)
        bundle_id = _bundle_id(version, source_commit, wheel_sha)
        bundle_root = config_root / "sdk-bundles" / bundle_id
        existing_manifest = bundle_root / "manifest.json"
        if existing_manifest.exists():
            existing_payload = read_json_file(existing_manifest)
            try:
                _verify_bundle(existing_payload)
            except click.ClickException:
                shutil.rmtree(bundle_root)
            else:
                return existing_payload
        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        bundle_root.mkdir(parents=True)

        dist_dir = bundle_root / "dist"
        dist_dir.mkdir()
        wheel_path = dist_dir / wheel_tmp.name
        shutil.copy2(wheel_tmp, wheel_path)

        uv_cache_dir = bundle_root / "uv-cache"
        lock_path = bundle_root / "requirements.lock.txt"
        req_in = bundle_root / "requirements.in"
        requirement = f"open-xquant[{','.join(SDK_EXTRAS)}] @ {wheel_path.as_uri()}\n"
        write_text_file(req_in, requirement)
        _run(
            _uv_cmd(
                [
                    "pip",
                    "compile",
                    str(req_in),
                    "--generate-hashes",
                    "--output-file",
                    str(lock_path),
                    "--no-header",
                    "--no-annotate",
                    "--cache-dir",
                    str(uv_cache_dir),
                ],
                directory=bundle_root,
            )
        )
        lock_sha = sha256_file(lock_path)

        runner_venv = bundle_root / "runner" / ".venv"
        _run(_uv_cmd(["venv", "--python", sys.executable, str(runner_venv)], directory=bundle_root))
        runner_python = _venv_executable(runner_venv, "python")
        _run(
            _uv_cmd(
                [
                    "pip",
                    "sync",
                    "--python",
                    str(runner_python),
                    "--require-hashes",
                    "--strict",
                    "--cache-dir",
                    str(uv_cache_dir),
                    str(lock_path),
                ],
                directory=bundle_root,
            )
        )
        _run([str(runner_python), "-c", "import oxq"])
        _run(_uv_cmd(["pip", "check", "--python", str(runner_python), "--cache-dir", str(uv_cache_dir)], directory=bundle_root))
        runner_oxq = _venv_executable(runner_venv, "oxq")
        _run([str(runner_oxq), "--help"])

        packages_path = bundle_root / "packages.json"
        packages = _run_json(
            _uv_cmd(
                [
                    "pip",
                    "list",
                    "--format",
                    "json",
                    "--python",
                    str(runner_python),
                    "--cache-dir",
                    str(uv_cache_dir),
                ],
                directory=bundle_root,
            )
        )
        write_json_file(packages_path, {"packages": packages})
        packages_count = len(packages) if isinstance(packages, list) else 0
        payload = _bundle_payload(
            bundle_id=bundle_id,
            bundle_root=bundle_root,
            wheel_path=wheel_path,
            wheel_sha=wheel_sha,
            version=version,
            source_commit=source_commit,
            lock_path=lock_path,
            lock_sha=lock_sha,
            packages_path=packages_path,
            packages_count=packages_count,
            runner_venv=runner_venv,
            runner_python=runner_python,
            runner_oxq=runner_oxq,
            uv_cache_dir=uv_cache_dir,
        )
        write_json_file(bundle_root / "manifest.json", payload)
        return payload
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


def install_workspace_sdk(cwd: Path, venv: Path, *, force: bool = False) -> dict[str, Any]:
    """Install the cached SDK bundle into a research workspace virtualenv."""

    del force
    cwd = cwd.resolve()
    venv = venv.resolve()
    _validate_workspace_venv(cwd, venv)
    manifest_path = default_manifest_path()
    if not manifest_path.exists():
        raise click.ClickException("Missing agent-install.json. Run `oxq agent install` first.")
    manifest = read_json_file(manifest_path)
    bundle = manifest.get("sdk_bundle")
    if not isinstance(bundle, dict):
        raise click.ClickException("agent-install.json has no sdk_bundle. Re-run `oxq agent install`.")
    _verify_bundle(bundle)

    python = _venv_executable(venv, "python")
    if not python.exists():
        runner = _require_dict(bundle, "runner")
        bundle_python = runner.get("python")
        python_arg = str(expand_path(bundle_python)) if isinstance(bundle_python, str) and bundle_python else "3.12"
        _run(_uv_cmd(["venv", "--python", python_arg, str(venv)], directory=cwd))

    lock_path = expand_path(bundle["dependencies"]["lock_file"])
    uv_cache_dir = expand_path(bundle["uv_cache_dir"])
    _run(
        _uv_cmd(
            [
                "pip",
                "install",
                "--python",
                str(python),
                "--requirements",
                str(lock_path),
                "--require-hashes",
                "--strict",
                "--cache-dir",
                str(uv_cache_dir),
            ],
            directory=cwd,
        )
    )
    _run([str(python), "-c", "import oxq"])
    _run(_uv_cmd(["pip", "check", "--python", str(python), "--cache-dir", str(uv_cache_dir)], directory=cwd))
    runner_oxq = _venv_executable(venv, "oxq")
    _run([str(runner_oxq), "--help"])

    return {
        "enabled": True,
        "bundle_id": str(bundle["id"]),
        "profile": str(bundle.get("profile", SDK_PROFILE)),
        "venv": _display_path(cwd, venv),
        "runner": _display_path(cwd, runner_oxq),
        "python": _display_path(cwd, python),
        "wheel_sha256": str(bundle["wheel"]["sha256"]),
        "lock_sha256": str(bundle["dependencies"]["lock_sha256"]),
    }


def remove_sdk_bundle(bundle: dict[str, Any], config_root: Path) -> bool:
    """Remove a managed SDK bundle after validating that it is under config_root."""

    root_value = bundle.get("root")
    if not isinstance(root_value, str):
        return False
    root = _stored_path(root_value)
    bundles_root = (config_root / "sdk-bundles").resolve()
    if not root.is_relative_to(bundles_root):
        return False
    if not root.exists():
        return True
    try:
        _verify_bundle(bundle)
    except click.ClickException:
        return False
    if root.exists():
        shutil.rmtree(root)
    return True


def _verify_bundle(bundle: dict[str, Any]) -> None:
    root = _stored_path(_require_str(bundle, "root"))
    wheel = _stored_path(_require_str(_require_dict(bundle, "wheel"), "path"))
    lock = _stored_path(_require_str(_require_dict(bundle, "dependencies"), "lock_file"))
    packages = _stored_path(_require_str(_require_dict(bundle, "dependencies"), "packages_file"))
    runner_meta = _require_dict(bundle, "runner")
    runner_python = _stored_path(_require_str(runner_meta, "python"))
    runner = _stored_path(_require_str(runner_meta, "oxq"))
    if not root.exists():
        raise click.ClickException(f"SDK bundle directory is missing: {root}")
    for path in (wheel, lock, packages, runner_python, runner):
        if not path.is_relative_to(root):
            raise click.ClickException(f"SDK bundle path escapes bundle root: {path}")
        if not path.exists():
            raise click.ClickException(f"SDK bundle file is missing: {path}")
    expected_wheel_sha = _require_str(_require_dict(bundle, "wheel"), "sha256")
    if sha256_file(wheel) != expected_wheel_sha:
        raise click.ClickException(f"SDK bundle wheel hash mismatch: {wheel}")
    expected_lock_sha = _require_str(_require_dict(bundle, "dependencies"), "lock_sha256")
    if sha256_file(lock) != expected_lock_sha:
        raise click.ClickException(f"SDK bundle lock hash mismatch: {lock}")
    _run([str(runner_python), "-c", "import oxq"])
    _run([str(runner), "--help"])


def _is_buildable_source(source_root: Path) -> bool:
    return (source_root / "pyproject.toml").is_file()


def _installed_sdk_bundle(config_root: Path) -> dict[str, Any] | None:
    manifest_path = config_root / "agent-install.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = read_json_file(manifest_path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise click.ClickException(f"Cannot read cached SDK bundle manifest: {manifest_path}") from exc
    bundle = manifest.get("sdk_bundle")
    if not isinstance(bundle, dict):
        return None
    try:
        _verify_bundle(bundle)
    except click.ClickException as exc:
        raise click.ClickException(f"Cached SDK bundle is invalid: {exc}") from exc
    return bundle


def _validate_workspace_venv(cwd: Path, venv: Path) -> None:
    if venv == cwd or cwd.is_relative_to(venv):
        raise click.ClickException(
            f"Refusing to use the research directory or a parent as the SDK virtualenv: {venv}"
        )
    if not venv.exists():
        return
    if not venv.is_dir():
        raise click.ClickException(f"SDK virtualenv path exists but is not a directory: {venv}")
    if not _is_virtualenv_dir(venv) and any(venv.iterdir()):
        raise click.ClickException(
            "Refusing to use or replace an existing non-virtualenv path for --sdk-venv: "
            f"{venv}"
        )


def _is_virtualenv_dir(path: Path) -> bool:
    return (path / "pyvenv.cfg").is_file()


def _require_dict(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise click.ClickException(f"Invalid sdk_bundle metadata: missing {key}")
    return value


def _require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise click.ClickException(f"Invalid sdk_bundle metadata: missing {key}")
    return value


def _bundle_payload(
    *,
    bundle_id: str,
    bundle_root: Path,
    wheel_path: Path,
    wheel_sha: str,
    version: str,
    source_commit: str,
    lock_path: Path,
    lock_sha: str,
    packages_path: Path,
    packages_count: int,
    runner_venv: Path,
    runner_python: Path,
    runner_oxq: Path,
    uv_cache_dir: Path,
) -> dict[str, Any]:
    return {
        "id": bundle_id,
        "root": str(bundle_root.resolve()),
        "profile": SDK_PROFILE,
        "extras": list(SDK_EXTRAS),
        "excluded_extras": list(EXCLUDED_EXTRAS),
        "wheel": {
            "path": str(wheel_path.resolve()),
            "sha256": wheel_sha,
            "version": version,
            "source_commit": source_commit,
        },
        "dependencies": {
            "lock_file": str(lock_path.resolve()),
            "lock_sha256": lock_sha,
            "packages_file": str(packages_path.resolve()),
            "packages_count": packages_count,
        },
        "runner": {
            "venv": _absolute_path(runner_venv),
            "python": _absolute_path(runner_python),
            "oxq": _absolute_path(runner_oxq),
            "argv": [_absolute_path(runner_oxq)],
        },
        "uv_cache_dir": str(uv_cache_dir.resolve()),
    }


def _single_wheel(dist: Path) -> Path:
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise click.ClickException(f"Expected exactly one built wheel in {dist}, found {len(wheels)}.")
    return wheels[0]


def _package_version(source_root: Path) -> str:
    pyproject = source_root / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version") if isinstance(project, dict) else None
    return str(version) if version else "unknown"


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


def _bundle_id(version: str, source_commit: str, wheel_sha: str) -> str:
    commit = source_commit[:12] if source_commit and source_commit != "unknown" else "no-git"
    return f"{_slug(version)}-{commit}-{wheel_sha[:12]}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "unknown"


def _venv_executable(venv: Path, name: str) -> Path:
    if sys.platform == "win32":
        suffix = ".exe" if name in {"python", "oxq"} else ""
        return venv / "Scripts" / f"{name}{suffix}"
    return venv / "bin" / name


def _display_path(cwd: Path, path: Path) -> str:
    path = Path(os.path.abspath(path))
    try:
        return path.relative_to(Path(os.path.abspath(cwd))).as_posix()
    except ValueError:
        return str(path)


def _absolute_path(path: Path) -> str:
    return str(Path(os.path.abspath(path)))


def _stored_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(str(path)))))


def _uv_cmd(args: list[str], *, directory: Path) -> list[str]:
    return ["uv", "--directory", str(directory), "--no-config", *args]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise click.ClickException(f"Required command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise click.ClickException(f"Command failed: {' '.join(cmd)}\n{detail}") from exc


def _run_json(cmd: list[str]) -> Any:
    result = _run(cmd)
    return json.loads(result.stdout or "null")
