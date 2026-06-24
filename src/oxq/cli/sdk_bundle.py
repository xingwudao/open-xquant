"""SDK bundle helpers for Agent and research workspace setup."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

import click

from oxq.cli.agent_manifest import expand_path, read_json_file, sha256_file, write_json_file, write_text_file

SDK_PROFILE = "full-research"
SDK_EXTRA_FALLBACK = ("chart", "scipy", "yfinance", "akshare", "live", "mcp", "agent")
EXCLUDED_EXTRAS = ("dev", "docs", "talib")
WHEEL_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
RESERVED_WORKSPACE_PATHS = (
    ".open-xquant",
    "strategy_specs",
    "runs",
    "reports",
    "comparisons",
    "experiments.jsonl",
    "strategy_spec.yaml",
    "AGENTS.md",
)


def default_config_dir() -> Path:
    return Path.home().joinpath(".config", "open-xquant").resolve()


def default_manifest_path() -> Path:
    return default_config_dir() / "agent-install.json"


def build_sdk_bundle(source_root: Path, config_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Build a cached open-xquant SDK bundle and runner environment."""

    source_root = source_root.resolve()
    config_root = config_root.resolve()
    uv_cache_dir = _shared_uv_cache_dir(config_root)
    buildable_source = _is_buildable_source(source_root)
    version = _package_version(source_root) if buildable_source else _installed_distribution_version(source_root)
    sdk_extras = _selected_sdk_extras(source_root, buildable_source=buildable_source)
    source_commit = _current_commit(source_root) if buildable_source else "installed-distribution"
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
            uv_cache_dir=uv_cache_dir,
            extras=sdk_extras,
            excluded_extras=EXCLUDED_EXTRAS,
        )

    config_root.mkdir(parents=True, exist_ok=True)

    tmp_root = config_root / "sdk-bundles" / ".build-tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    try:
        dist_tmp = tmp_root / "dist"
        wheel_tmp = _build_source_wheel(source_root, dist_tmp, buildable_source=buildable_source)
        wheel_sha = sha256_file(wheel_tmp)
        cached_bundle = _installed_sdk_bundle(config_root, invalid_as_miss=True)
        if not buildable_source:
            if (
                cached_bundle is not None
                and _bundle_version(cached_bundle) == version
                and _bundle_extras(cached_bundle) == sdk_extras
                and _bundle_wheel_sha(cached_bundle) == wheel_sha
            ):
                return cached_bundle
        bundle_id = _bundle_id(version, source_commit, wheel_sha)
        bundle_root = config_root / "sdk-bundles" / bundle_id
        existing_manifest = bundle_root / "manifest.json"
        if existing_manifest.exists():
            existing_payload = _read_bundle_manifest(existing_manifest)
            if existing_payload is None:
                _remove_bundle_root_for_rebuild(bundle_root)
            else:
                try:
                    _verify_bundle(existing_payload)
                except click.ClickException:
                    _remove_bundle_root_for_rebuild(bundle_root)
                else:
                    if _bundle_extras(existing_payload) == sdk_extras:
                        return existing_payload
                    _remove_bundle_root_for_rebuild(bundle_root)
        if bundle_root.exists():
            _remove_bundle_root_for_rebuild(bundle_root)
        bundle_root.mkdir(parents=True)

        dist_dir = bundle_root / "dist"
        dist_dir.mkdir()
        wheel_path = dist_dir / wheel_tmp.name
        shutil.copy2(wheel_tmp, wheel_path)

        lock_path = bundle_root / "requirements.lock.txt"
        req_in = bundle_root / "requirements.in"
        requirement_name = "open-xquant"
        if sdk_extras:
            requirement_name = f"{requirement_name}[{','.join(sdk_extras)}]"
        requirement = f"{requirement_name} @ {wheel_path.as_uri()}\n"
        write_text_file(req_in, requirement)
        _run(
            _uv_cmd(
                [
                    "pip",
                    "compile",
                    str(req_in),
                    "--python",
                    sys.executable,
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
        runner_python = _venv_executable(runner_venv, "python")
        runner_oxq = _venv_executable(runner_venv, "oxq")
        if not _try_reuse_runner_venv(cached_bundle, runner_venv, lock_path, wheel_path, uv_cache_dir, bundle_root):
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
            extras=sdk_extras,
            excluded_extras=EXCLUDED_EXTRAS,
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
    _verify_managed_bundle_root(bundle, default_config_dir())
    _verify_bundle(bundle)
    runner = _require_dict(bundle, "runner")
    bundle_python = _stored_path(_require_str(runner, "python"))

    python = _venv_executable(venv, "python")
    if not python.exists():
        if _is_virtualenv_dir(venv):
            raise click.ClickException(f"SDK virtualenv is missing Python interpreter: {python}")
        _run(_uv_cmd(["venv", "--python", str(bundle_python), str(venv)], directory=cwd))
    else:
        _verify_python_version_matches(bundle_python, python)

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
                "--link-mode",
                "copy",
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

    if not sdk_bundle_can_be_removed(bundle, config_root):
        return False
    root = _managed_bundle_root(bundle, config_root)
    if root is None:
        return False
    if root.exists():
        try:
            shutil.rmtree(root)
        except OSError:
            return False
    return True


def sdk_bundle_can_be_removed(bundle: dict[str, Any], config_root: Path) -> bool:
    """Return whether a managed SDK bundle passes purge preflight checks."""

    root = _managed_bundle_root(bundle, config_root)
    if root is None:
        return False
    if not root.exists():
        return True
    try:
        _verify_bundle(bundle)
    except (OSError, click.ClickException):
        return False
    return not _path_is_relative_to(_stored_path(sys.executable), root)


def _managed_bundle_root(bundle: dict[str, Any], config_root: Path) -> Path | None:
    root_value = bundle.get("root")
    if not isinstance(root_value, str):
        return None
    root = _stored_path(root_value).resolve()
    bundles_root = (config_root / "sdk-bundles").resolve()
    if not root.is_relative_to(bundles_root):
        return None
    return root


def _remove_bundle_root_for_rebuild(bundle_root: Path) -> None:
    if _path_is_relative_to(_stored_path(sys.executable), bundle_root):
        raise click.ClickException(
            "Refusing to replace the active cached SDK bundle. "
            "Re-run this command from a non-cached open-xquant checkout or installed Python environment."
        )
    shutil.rmtree(bundle_root)


def sdk_bundle_contains_active_runner(bundle: dict[str, Any], config_root: Path) -> bool:
    """Return whether the current Python executable is inside a managed SDK bundle."""

    root_value = bundle.get("root")
    if not isinstance(root_value, str):
        return False
    root = _stored_path(root_value).resolve()
    bundles_root = (config_root / "sdk-bundles").resolve()
    if not root.is_relative_to(bundles_root) or not root.exists():
        return False
    return _path_is_relative_to(_stored_path(sys.executable), root)


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
    for path in (wheel, lock, packages, runner):
        _verify_bundle_path(root, path)
    _verify_bundle_path(root, runner_python, allow_symlink_target=True)
    for path in (wheel, lock, packages, runner_python, runner):
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


def _bundle_version(bundle: dict[str, Any]) -> str | None:
    wheel = bundle.get("wheel")
    if not isinstance(wheel, dict):
        return None
    value = wheel.get("version")
    return str(value) if isinstance(value, str) and value else None


def _bundle_wheel_sha(bundle: dict[str, Any]) -> str | None:
    wheel = bundle.get("wheel")
    if not isinstance(wheel, dict):
        return None
    value = wheel.get("sha256")
    return str(value) if isinstance(value, str) and value else None


def _bundle_extras(bundle: dict[str, Any]) -> tuple[str, ...] | None:
    extras = bundle.get("extras")
    if not isinstance(extras, list) or not all(isinstance(extra, str) for extra in extras):
        return None
    return tuple(sorted(_normalize_extra(extra) for extra in extras))


def _selected_sdk_extras(source_root: Path, *, buildable_source: bool) -> tuple[str, ...]:
    extras = _project_optional_extras(source_root) if buildable_source else _installed_optional_extras()
    if not extras:
        extras = SDK_EXTRA_FALLBACK
    excluded = {_normalize_extra(extra) for extra in EXCLUDED_EXTRAS}
    return tuple(sorted({_normalize_extra(extra) for extra in extras if _normalize_extra(extra) not in excluded}))


def _project_optional_extras(source_root: Path) -> tuple[str, ...]:
    pyproject = source_root / "pyproject.toml"
    if not pyproject.exists():
        return ()
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    optional = project.get("optional-dependencies") if isinstance(project, dict) else None
    if not isinstance(optional, dict):
        return ()
    return tuple(str(extra) for extra in optional)


def _installed_optional_extras() -> tuple[str, ...]:
    try:
        dist = metadata.distribution("open-xquant")
    except metadata.PackageNotFoundError:
        return ()
    dist_metadata = getattr(dist, "metadata", None)
    if dist_metadata is None:
        return ()
    values = _metadata_get_all(dist_metadata, "Provides-Extra")
    return tuple(str(value) for value in values)


def _metadata_get_all(message: Any, key: str) -> list[Any]:
    get_all = getattr(message, "get_all", None)
    if callable(get_all):
        return list(get_all(key, []))
    value = message.get(key) if hasattr(message, "get") else None
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_extra(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _try_reuse_runner_venv(
    cached_bundle: dict[str, Any] | None,
    runner_venv: Path,
    lock_path: Path,
    wheel_path: Path,
    uv_cache_dir: Path,
    bundle_root: Path,
) -> bool:
    if cached_bundle is None:
        return False
    previous_lock = _stored_path(_require_str(_require_dict(cached_bundle, "dependencies"), "lock_file"))
    if not previous_lock.exists():
        return False
    if _lock_without_project_requirement(previous_lock) != _lock_without_project_requirement(lock_path):
        return False
    previous_runner_python = _runner_python_path(cached_bundle)
    if previous_runner_python is None or not previous_runner_python.exists():
        return False
    if _python_major_minor(previous_runner_python) != _python_major_minor(Path(sys.executable)):
        return False
    if not _runner_packages_match_manifest(cached_bundle, previous_runner_python, uv_cache_dir, bundle_root):
        return False
    previous_runner_venv = _runner_venv_path(cached_bundle)
    if previous_runner_venv is None or not previous_runner_venv.exists():
        return False
    if runner_venv.exists():
        shutil.rmtree(runner_venv)
    shutil.copytree(previous_runner_venv, runner_venv, symlinks=True)
    runner_python = _venv_executable(runner_venv, "python")
    _run(
        _uv_cmd(
            [
                "pip",
                "install",
                "--python",
                str(runner_python),
                "--reinstall",
                "--no-deps",
                "--cache-dir",
                str(uv_cache_dir),
                str(wheel_path),
            ],
            directory=bundle_root,
        )
    )
    return True


def _runner_venv_path(bundle: dict[str, Any]) -> Path | None:
    python = _runner_python_path(bundle)
    return python.parent.parent if python is not None else None


def _runner_python_path(bundle: dict[str, Any]) -> Path | None:
    runner = _require_dict(bundle, "runner")
    python = runner.get("python")
    if isinstance(python, str) and python:
        return _stored_path(python)
    return None


def _runner_packages_match_manifest(
    bundle: dict[str, Any],
    runner_python: Path,
    uv_cache_dir: Path,
    bundle_root: Path,
) -> bool:
    packages_path = _stored_path(_require_str(_require_dict(bundle, "dependencies"), "packages_file"))
    if not packages_path.exists():
        return False
    try:
        recorded = read_json_file(packages_path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    recorded_packages = recorded.get("packages") if isinstance(recorded, dict) else recorded
    if not isinstance(recorded_packages, list):
        return False
    current_packages = _run_json(
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
    if not isinstance(current_packages, list):
        return False
    return _package_identity_set(recorded_packages) == _package_identity_set(current_packages)


def _package_identity_set(packages: list[Any]) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for package in packages:
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        version = package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            identities.add((name.lower(), version))
    return identities


def _lock_without_project_requirement(lock_path: Path) -> str:
    kept: list[str] = []
    skipping_project = False
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("open-xquant @ "):
            skipping_project = True
            continue
        if skipping_project:
            if line.startswith((" ", "\t")):
                continue
            skipping_project = False
        kept.append(line)
    return "\n".join(kept)


def _build_source_wheel(source_root: Path, dist_tmp: Path, *, buildable_source: bool) -> Path:
    if buildable_source:
        _run(_uv_cmd(["build", "--wheel", "--out-dir", str(dist_tmp), "."], directory=source_root))
        return _single_wheel(dist_tmp)
    return _build_installed_distribution_wheel(source_root, dist_tmp)


def _build_installed_distribution_wheel(source_root: Path, dist_tmp: Path) -> Path:
    try:
        dist = metadata.distribution("open-xquant")
    except metadata.PackageNotFoundError as exc:
        raise click.ClickException(
            "Cannot build the SDK bundle because the resolved open-xquant source "
            f"is not a project checkout and the installed package metadata is unavailable: {source_root}. "
            "Re-run `oxq agent install` from an open-xquant checkout or a wheel-installed `oxq` command."
        ) from exc
    files = list(dist.files or [])
    if not files:
        raise click.ClickException(
            "Cannot build the SDK bundle because the installed open-xquant distribution "
            "does not expose installed file metadata."
        )
    dist_tmp.mkdir(parents=True, exist_ok=True)
    dist_name = _wheel_safe_name(str(dist.metadata.get("Name") or "open-xquant"))
    version = _wheel_safe_version(str(dist.version or "unknown"))
    wheel_path = dist_tmp / f"{dist_name}-{version}-py3-none-any.whl"
    entries: list[tuple[str, bytes]] = []
    dist_info_dir: str | None = None
    for file in sorted(files, key=lambda item: str(item).replace("\\", "/")):
        archive_name = str(file).replace("\\", "/")
        if not _is_safe_wheel_archive_name(archive_name) or archive_name.endswith("/RECORD"):
            continue
        parts = PurePosixPath(archive_name).parts
        for index, part in enumerate(parts):
            if part.endswith(".dist-info"):
                dist_info_dir = "/".join(parts[: index + 1])
                break
        source = Path(dist.locate_file(file))
        if not source.is_file():
            continue
        entries.append((archive_name, source.read_bytes()))
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        if dist_info_dir is None:
            raise click.ClickException("Cannot build the SDK bundle because installed distribution metadata is incomplete.")
        record_name = f"{dist_info_dir}/RECORD"
        records = [_wheel_record_line(archive_name, data) for archive_name, data in entries]
        record_data = ("\n".join([*records, _csv_line([record_name, "", ""])]) + "\n").encode("utf-8")
        for archive_name, data in sorted([*entries, (record_name, record_data)]):
            _write_wheel_entry(wheel, archive_name, data)
    return wheel_path


def _write_wheel_entry(wheel: zipfile.ZipFile, archive_name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(archive_name, WHEEL_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    wheel.writestr(info, data)


def _is_safe_wheel_archive_name(value: str) -> bool:
    if not value or re.match(r"^[A-Za-z]:", value):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _verify_managed_bundle_root(bundle: dict[str, Any], config_root: Path) -> None:
    root_value = _require_str(bundle, "root")
    root = _stored_path(root_value).resolve()
    bundles_root = (config_root / "sdk-bundles").resolve()
    if not root.is_relative_to(bundles_root):
        raise click.ClickException(f"SDK bundle root escapes managed cache: {root}")


def _verify_bundle_path(root: Path, path: Path, *, allow_symlink_target: bool = False) -> None:
    if not path.is_relative_to(root):
        raise click.ClickException(f"SDK bundle path escapes bundle root: {path}")
    if not allow_symlink_target and not _resolved_path_is_relative_to(path, root):
        raise click.ClickException(f"SDK bundle path escapes bundle root: {path}")


def _verify_python_version_matches(expected_python: Path, actual_python: Path) -> None:
    expected = _python_major_minor(expected_python)
    actual = _python_major_minor(actual_python)
    if expected != actual:
        raise click.ClickException(
            "Existing SDK virtualenv Python version does not match the cached SDK bundle: "
            f"{actual_python} uses {actual}, bundle runner uses {expected}. "
            "Use a matching --sdk-venv or remove the existing virtualenv before retrying."
        )


def _python_major_minor(python: Path) -> str:
    result = _run(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ]
    )
    version = result.stdout.strip()
    if not version:
        raise click.ClickException(f"Cannot determine Python version for SDK interpreter: {python}")
    return version


def _installed_distribution_version(source_root: Path) -> str:
    try:
        return str(metadata.distribution("open-xquant").version or "unknown")
    except metadata.PackageNotFoundError as exc:
        raise click.ClickException(
            "Cannot build the SDK bundle because the resolved open-xquant source "
            f"is not a project checkout and the installed package metadata is unavailable: {source_root}. "
            "Re-run `oxq agent install` from an open-xquant checkout or a wheel-installed `oxq` command."
        ) from exc


def _wheel_record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return _csv_line([path, f"sha256={digest}", str(len(data))])


def _csv_line(row: list[str]) -> str:
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="").writerow(row)
    return output.getvalue()


def _wheel_safe_name(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value).lower()


def _wheel_safe_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.!+]+", "_", value)


def _installed_sdk_bundle(config_root: Path, *, invalid_as_miss: bool = False) -> dict[str, Any] | None:
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
        if invalid_as_miss:
            return None
        raise click.ClickException(f"Cached SDK bundle is invalid: {exc}") from exc
    return bundle


def _shared_uv_cache_dir(config_root: Path) -> Path:
    return config_root / "sdk-cache" / "uv"


def _read_bundle_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json_file(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _validate_workspace_venv(cwd: Path, venv: Path) -> None:
    if venv == cwd or cwd.is_relative_to(venv):
        raise click.ClickException(
            f"Refusing to use the research directory or a parent as the SDK virtualenv: {venv}"
        )
    for reserved in RESERVED_WORKSPACE_PATHS:
        reserved_path = (cwd / reserved).resolve()
        if venv == reserved_path or venv.is_relative_to(reserved_path):
            raise click.ClickException(f"Refusing to use reserved workspace path for --sdk-venv: {venv}")
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
    extras: tuple[str, ...],
    excluded_extras: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "id": bundle_id,
        "root": str(bundle_root.resolve()),
        "profile": SDK_PROFILE,
        "extras": list(extras),
        "excluded_extras": list(excluded_extras),
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


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    if path.is_relative_to(parent):
        return True
    try:
        return path.resolve().is_relative_to(parent.resolve())
    except OSError:
        return path.is_relative_to(parent)


def _resolved_path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        return path.resolve().is_relative_to(parent.resolve())
    except OSError:
        return path.is_relative_to(parent)


def _uv_cmd(args: list[str], *, directory: Path) -> list[str]:
    return ["uv", "--directory", str(directory), "--no-config", *args]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise click.ClickException(f"Required command not found: {cmd[0]}") from exc
    except OSError as exc:
        raise click.ClickException(f"Command failed: {' '.join(cmd)}\n{exc}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise click.ClickException(f"Command failed: {' '.join(cmd)}\n{detail}") from exc


def _run_json(cmd: list[str]) -> Any:
    result = _run(cmd)
    return json.loads(result.stdout or "null")
