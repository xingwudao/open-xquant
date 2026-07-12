from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner

import oxq.cli.agent as agent_module
from oxq.cli.main import main


def _write_source(root: Path) -> None:
    skills = root / "agent" / "skills"
    skill_dir = skills / "build-strategy-spec"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: build-strategy-spec\ndescription: Build quant strategies\n---\n\n# Strategy Builder\n",
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
    assert not (home / ".config/opencode/skills/build-strategy-spec").exists()
    assert "open-xquant:begin" not in agents.read_text(encoding="utf-8")
    assert "user content" in agents.read_text(encoding="utf-8")
    assert data_dir.exists()
    assert (home / ".config/open-xquant/sdk-bundles/bundle-test").exists()
    assert (home / ".config/open-xquant/agent-install.json").exists()


def test_agent_uninstall_uses_persisted_codex_root_after_env_change(monkeypatch, tmp_path) -> None:
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
    replacement_sentinel = replacement_codex_home / "keep.txt"
    replacement_sentinel.write_text("keep me\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(replacement_codex_home))

    result = CliRunner().invoke(main, ["agent", "uninstall", "--target", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    assert not (original_codex_home / "skills/build-strategy-spec").exists()
    assert "open-xquant:begin" not in (original_codex_home / "AGENTS.md").read_text(
        encoding="utf-8"
    )
    assert replacement_sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_agent_reinstall_after_uninstall_uses_current_codex_root(monkeypatch, tmp_path) -> None:
    source_v1 = tmp_path / "source-v1"
    source_v2 = tmp_path / "source-v2"
    home = tmp_path / "home"
    codex_home_a = tmp_path / "codex-a"
    codex_home_b = tmp_path / "codex-b"
    _write_source(source_v1)
    _write_source(source_v2)
    source_v2_skill = source_v2 / "agent/skills/build-strategy-spec/SKILL.md"
    source_v2_skill.write_text(
        source_v2_skill.read_text(encoding="utf-8") + "\nupgraded workflow\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home_a))

    install_a = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source_v1), "--yes"],
    )
    assert install_a.exit_code == 0, install_a.output
    uninstall_a = CliRunner().invoke(main, ["agent", "uninstall", "--target", "codex", "--yes"])
    assert uninstall_a.exit_code == 0, uninstall_a.output

    monkeypatch.setenv("CODEX_HOME", str(codex_home_b))
    install_b = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source_v1), "--yes"],
    )

    assert install_b.exit_code == 0, install_b.output
    assert not (codex_home_a / "skills/build-strategy-spec").exists()
    assert (codex_home_b / "skills/build-strategy-spec/SKILL.md").exists()
    status = CliRunner().invoke(main, ["agent", "status", "--json"])
    assert status.exit_code == 0, status.output
    status_payload = json.loads(status.output)
    assert status_payload["targets"]["codex"]["installed"] is True
    assert status_payload["targets"]["codex"]["skills"]["installed"] == 1

    upgrade_b = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "codex", "--from-local", str(source_v2), "--yes"],
    )
    assert upgrade_b.exit_code == 0, upgrade_b.output
    installed_b = codex_home_b / "skills/build-strategy-spec/SKILL.md"
    assert "upgraded workflow" in installed_b.read_text(encoding="utf-8")

    uninstall_b = CliRunner().invoke(main, ["agent", "uninstall", "--target", "codex", "--yes"])
    assert uninstall_b.exit_code == 0, uninstall_b.output
    assert not (codex_home_b / "skills/build-strategy-spec").exists()


def test_agent_uninstall_refuses_modified_managed_skill_resource(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    references = source / "agent/skills/build-strategy-spec/references"
    references.mkdir()
    (references / "ref.md").write_text("managed reference\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".config/opencode"
    installed_ref = target_root / "skills/build-strategy-spec/references/ref.md"
    installed_ref.write_text("local reference edit\n", encoding="utf-8")
    manifest = home / ".config/open-xquant/agent-install.json"
    config = home / ".config/open-xquant/agent.yaml"
    target_before = _snapshot_tree(target_root)
    manifest_before = manifest.read_bytes()
    config_before = config.read_bytes()

    result = CliRunner().invoke(main, ["agent", "uninstall", "--target", "opencode", "--yes"])

    assert result.exit_code == 1
    assert "modified managed skill" in result.output
    assert _snapshot_tree(target_root) == target_before
    assert manifest.read_bytes() == manifest_before
    assert config.read_bytes() == config_before
    assert installed_ref.read_bytes() == b"local reference edit\n"


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
    sdk_cache = home / ".config/open-xquant/sdk-cache/uv"
    sdk_cache.mkdir(parents=True)
    (sdk_cache / "cached-wheel").write_text("cache\n", encoding="utf-8")
    manifest = home / ".config/open-xquant/agent-install.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["sdk_bundle"]["root"] == str(bundle)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert not bundle.exists()
    assert not sdk_cache.exists()
    assert not manifest.exists()
    assert not (home / ".config/open-xquant/agent.yaml").exists()
    assert not any(".backup-" in path.name for path in home.rglob("*"))


def test_agent_uninstall_purge_reports_post_commit_backup_cleanup_failure(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".config/opencode"
    instructions = target_root / "AGENTS.md"
    instructions.write_text(
        "user instructions\n" + instructions.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    user_file = target_root / "user-owned.txt"
    user_file.write_text("keep me\n", encoding="utf-8")
    sdk_cache = home / ".config/open-xquant/sdk-cache/uv"
    sdk_cache.mkdir(parents=True)
    (sdk_cache / "cached-wheel").write_text("cache\n", encoding="utf-8")
    attempted_backups: list[str] = []
    original_remove = agent_module._remove_upgrade_path

    def fail_backup_cleanup(path: Path) -> None:
        if ".backup-" in path.name:
            attempted_backups.append(path.name)
            raise OSError("injected backup cleanup failure")
        original_remove(path)

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", fail_backup_cleanup)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code == 1
    assert "committed, but transaction backup cleanup failed" in result.output
    assert "rolled back" not in result.output.lower()
    for backup_prefix in (
        ".build-strategy-spec.backup-",
        ".AGENTS.md.backup-",
        ".agent.yaml.backup-",
        ".bundle-test.backup-",
        ".sdk-cache.backup-",
    ):
        assert sum(name.startswith(backup_prefix) for name in attempted_backups) == 1
    assert not any(name.startswith(".agent-install.json.backup-") for name in attempted_backups)
    assert not (target_root / "skills/build-strategy-spec").exists()
    assert instructions.read_text(encoding="utf-8") == "user instructions\n"
    assert user_file.read_text(encoding="utf-8") == "keep me\n"
    assert not (home / ".config/open-xquant/agent.yaml").exists()
    assert not (home / ".config/open-xquant/agent-install.json").exists()
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    assert transaction.exists()
    transaction_payload = json.loads(transaction.read_text(encoding="utf-8"))
    assert transaction_payload["transaction_type"] == "agent-uninstall-purge"
    assert transaction_payload["phase"] == "committed"
    transaction_backups = {
        Path(record["backup"]).name for record in transaction_payload["backups"]
    }
    assert set(attempted_backups) < transaction_backups
    assert sum(
        name.startswith(".agent-install.json.backup-") for name in transaction_backups
    ) == 1

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", original_remove)
    before_dry_run = _snapshot_tree(home)
    dry_run = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--dry-run", "--yes"],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "pending committed purge cleanup" in dry_run.output
    assert _snapshot_tree(home) == before_dry_run

    retry = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert retry.exit_code == 0, retry.output
    assert "Recovered pending committed purge cleanup" in retry.output
    assert user_file.read_text(encoding="utf-8") == "keep me\n"
    assert not transaction.exists()
    assert not (home / ".config/open-xquant/agent.yaml").exists()
    assert not (home / ".config/open-xquant/agent-install.json").exists()
    assert not any(".backup-" in path.name for path in home.rglob("*"))


def test_pending_purge_cleanup_preflights_reinstall_then_current_uninstall(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install_args = [
        "agent",
        "install",
        "--target",
        "opencode",
        "--from-local",
        str(source),
        "--yes",
    ]
    install = CliRunner().invoke(main, install_args)
    assert install.exit_code == 0, install.output

    original_remove = agent_module._remove_upgrade_path

    def fail_backup_cleanup(path: Path) -> None:
        if ".backup-" in path.name:
            raise OSError("injected backup cleanup failure")
        original_remove(path)

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", fail_backup_cleanup)
    purge = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )
    assert purge.exit_code == 1

    config_root = home / ".config/open-xquant"
    transaction = config_root / "agent-uninstall-transaction.json"
    manifest = config_root / "agent-install.json"
    assert transaction.exists()
    assert not manifest.exists()

    blocked_reinstall = CliRunner().invoke(main, install_args)

    assert blocked_reinstall.exit_code == 1
    assert "transaction backup cleanup failed" in blocked_reinstall.output
    assert transaction.exists()
    assert not manifest.exists()

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", original_remove)
    reinstall = CliRunner().invoke(main, install_args)

    assert reinstall.exit_code == 0, reinstall.output
    assert "Recovered pending committed purge cleanup" in reinstall.output
    assert not transaction.exists()
    assert manifest.exists()
    installed_skill = home / ".config/opencode/skills/build-strategy-spec"
    assert installed_skill.exists()

    before_dry_run = _snapshot_tree(home)
    dry_run = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--dry-run", "--yes"],
    )

    assert dry_run.exit_code == 0, dry_run.output
    assert "pending committed purge cleanup" not in dry_run.output
    assert "Uninstall complete" in dry_run.output
    assert _snapshot_tree(home) == before_dry_run

    uninstall = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert uninstall.exit_code == 0, uninstall.output
    assert not manifest.exists()
    assert not installed_skill.exists()


def test_pending_purge_cleanup_preflights_upgrade(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    config_root = home / ".config/open-xquant"
    manifest = config_root / "agent-install.json"
    transaction = config_root / "agent-uninstall-transaction.json"
    transaction.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_type": "agent-uninstall-purge",
                "phase": "committed",
                "backups": [
                    {
                        "destination": str(manifest),
                        "backup": str(config_root / ".agent-install.json.backup-round13"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    upgrade = CliRunner().invoke(
        main,
        ["agent", "upgrade", "--target", "opencode", "--from-local", str(source), "--yes"],
    )

    assert upgrade.exit_code == 0, upgrade.output
    assert "Recovered pending committed purge cleanup" in upgrade.output
    assert not transaction.exists()
    assert manifest.exists()


@pytest.mark.parametrize(
    "uninstall_args",
    [
        ["--target", "opencode"],
        ["--target", "opencode", "--purge-config"],
        ["--all-targets"],
        ["--all-targets", "--purge-config"],
    ],
)
def test_every_uninstall_mode_recovers_pending_cleanup_before_current_mutation(
    monkeypatch,
    tmp_path,
    uninstall_args: list[str],
) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    config_root = home / ".config/open-xquant"
    backup = config_root / ".agent-install.json.backup-round14"
    backup.write_text("old manifest backup\n", encoding="utf-8")
    transaction = config_root / "agent-uninstall-transaction.json"
    transaction.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_type": "agent-uninstall-purge",
                "phase": "committed",
                "backups": [
                    {
                        "destination": str(config_root / "agent-install.json"),
                        "backup": str(backup),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["agent", "uninstall", *uninstall_args, "--yes"])

    assert result.exit_code == 0, result.output
    assert "Recovered pending committed purge cleanup" in result.output
    assert not transaction.exists()
    assert not backup.exists()


@pytest.mark.parametrize(
    "uninstall_args",
    [
        ["--target", "opencode"],
        ["--target", "opencode", "--purge-config"],
        ["--all-targets"],
        ["--all-targets", "--purge-config"],
    ],
)
def test_every_uninstall_mode_rejects_malformed_pending_cleanup_without_mutation(
    monkeypatch,
    tmp_path,
    uninstall_args: list[str],
) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    transaction.write_text("{malformed\n", encoding="utf-8")
    before = _snapshot_tree(home)

    result = CliRunner().invoke(main, ["agent", "uninstall", *uninstall_args, "--yes"])

    assert result.exit_code == 1
    assert "Invalid pending purge cleanup metadata" in result.output
    assert _snapshot_tree(home) == before


@pytest.mark.parametrize("failure", ["active_runner", "bundle_integrity", "manifest_schema"])
def test_agent_uninstall_purge_dry_run_performs_real_preflight_without_mutation(
    monkeypatch,
    tmp_path,
    failure: str,
) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    config_root = home / ".config/open-xquant"
    bundle = config_root / "sdk-bundles/bundle-test"
    manifest = config_root / "agent-install.json"
    if failure == "active_runner":
        monkeypatch.setattr(
            "oxq.cli.sdk_bundle.sys.executable",
            str(bundle / "runner/.venv/bin/python"),
        )
    elif failure == "bundle_integrity":
        (bundle / "requirements.lock.txt").write_text("corrupted\n", encoding="utf-8")
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["sdk_bundles"] = [payload["sdk_bundle"], "not-an-object"]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    before = _snapshot_tree(home)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--dry-run", "--yes"],
    )

    assert result.exit_code == 1
    if failure == "active_runner":
        assert "active cached SDK runner" in result.output
    elif failure == "bundle_integrity":
        assert "SDK bundle removal was not verified" in result.output
    else:
        assert "sdk_bundles must contain only objects" in result.output
    assert _snapshot_tree(home) == before


def test_agent_lifecycle_lock_serializes_contenders_and_tolerates_stale_file(
    monkeypatch,
    tmp_path,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    lock_path = agent_module.lifecycle_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale owner metadata\n", encoding="utf-8")
    acquired = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def hold_lock() -> None:
        try:
            with agent_module._agent_lifecycle_lock():
                acquired.set()
                assert release.wait(timeout=5)
        except BaseException as exc:
            failures.append(exc)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert acquired.wait(timeout=5)
    with lock_path.open("a+", encoding="utf-8") as contender:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    release.set()
    holder.join(timeout=5)

    assert not holder.is_alive()
    assert failures == []
    with agent_module._agent_lifecycle_lock():
        pass


@pytest.mark.parametrize(
    "recovery_command",
    [
        ["agent", "install", "--target", "codex"],
        ["agent", "upgrade", "--target", "codex"],
        ["agent", "uninstall", "--all-targets", "--purge-config"],
    ],
)
def test_legacy_pending_purge_cleanup_recovers_original_codex_root_after_env_change(
    monkeypatch,
    tmp_path,
    recovery_command: list[str],
) -> None:
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
    original_remove = agent_module._remove_upgrade_path

    def fail_backup_cleanup(path: Path) -> None:
        if ".backup-" in path.name:
            raise OSError("injected backup cleanup failure")
        original_remove(path)

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", fail_backup_cleanup)
    purge = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )
    assert purge.exit_code == 1
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    assert payload["trusted_roots"]["targets"]["codex"]["skills_dir"] == str(
        original_codex_home / "skills"
    )
    payload["schema_version"] = 1
    payload.pop("manifest_backup_sha256")
    payload.pop("trusted_roots")
    transaction.write_text(json.dumps(payload), encoding="utf-8")
    replacement_codex_home.mkdir(parents=True)
    replacement_sentinel = replacement_codex_home / "keep.txt"
    replacement_sentinel.write_text("keep me\n", encoding="utf-8")

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", original_remove)
    monkeypatch.setenv("CODEX_HOME", str(replacement_codex_home))
    command = [*recovery_command, "--yes"]
    if recovery_command[1] in {"install", "upgrade"}:
        command.extend(["--from-local", str(source)])
    recovery = CliRunner().invoke(
        main,
        command,
    )

    assert recovery.exit_code == 0, recovery.output
    assert "Recovered pending committed purge cleanup" in recovery.output
    assert not transaction.exists()
    assert replacement_sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert not any(".backup-" in path.name for path in original_codex_home.rglob("*"))


@pytest.mark.parametrize("schema_version", [1, 2])
def test_pending_purge_cleanup_rejects_tampered_recorded_target_root(
    monkeypatch,
    tmp_path,
    schema_version: int,
) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    codex_home = tmp_path / "codex"
    user_root = tmp_path / "documents"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "codex", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    original_remove = agent_module._remove_upgrade_path

    def fail_backup_cleanup(path: Path) -> None:
        if ".backup-" in path.name:
            raise OSError("injected backup cleanup failure")
        original_remove(path)

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", fail_backup_cleanup)
    purge = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )
    assert purge.exit_code == 1
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    if schema_version == 1:
        payload.pop("manifest_backup_sha256")
        payload.pop("trusted_roots")
    user_root.mkdir()
    user_destination = user_root / "notes"
    user_backup = user_root / ".notes.backup-forged"
    user_backup.write_text("keep me\n", encoding="utf-8")
    if schema_version == 2:
        payload["trusted_roots"]["targets"]["codex"]["skills_dir"] = str(user_root)
    skill_record = next(
        record
        for record in payload["backups"]
        if Path(record["destination"]).name == "build-strategy-spec"
    )
    skill_record["destination"] = str(user_destination)
    skill_record["backup"] = str(user_backup)
    transaction.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(agent_module, "_remove_upgrade_path", original_remove)
    before = _snapshot_tree(tmp_path)

    recovery = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert recovery.exit_code == 1
    assert "Invalid pending purge cleanup metadata" in recovery.output
    assert _snapshot_tree(tmp_path) == before
    assert user_backup.read_text(encoding="utf-8") == "keep me\n"


def test_agent_uninstall_purge_retry_finishes_partial_backup_cleanup(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    sdk_cache = home / ".config/open-xquant/sdk-cache/uv"
    sdk_cache.mkdir(parents=True)
    (sdk_cache / "cached-wheel").write_text("cache\n", encoding="utf-8")
    original_remove = agent_module._remove_upgrade_path
    failed_once = False

    def fail_one_backup_cleanup(path: Path) -> None:
        nonlocal failed_once
        if path.name.startswith(".agent-install.json.backup-") and not failed_once:
            failed_once = True
            raise OSError("injected partial cleanup failure")
        original_remove(path)

    monkeypatch.setattr(agent_module, "_remove_upgrade_path", fail_one_backup_cleanup)

    initial = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert initial.exit_code == 1
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    assert transaction.exists()
    remaining_backups = [path for path in home.rglob("*") if ".backup-" in path.name]
    assert len(remaining_backups) == 1
    assert remaining_backups[0].name.startswith(".agent-install.json.backup-")

    retry = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert retry.exit_code == 0, retry.output
    assert not transaction.exists()
    assert not any(".backup-" in path.name for path in home.rglob("*"))
    assert not (home / ".config/open-xquant/agent.yaml").exists()
    assert not (home / ".config/open-xquant/agent-install.json").exists()


def test_agent_uninstall_purge_retry_removes_sidecar_after_metadata_unlink_failure(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    original_unlink = Path.unlink

    def fail_transaction_unlink(path: Path, *args, **kwargs) -> None:
        if path == transaction:
            raise OSError("injected transaction unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_transaction_unlink)
    initial = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert initial.exit_code == 1
    assert "transaction metadata cleanup failed" in initial.output
    assert transaction.exists()
    assert not any(".backup-" in path.name for path in home.rglob("*"))

    monkeypatch.setattr(Path, "unlink", original_unlink)
    retry = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert retry.exit_code == 0, retry.output
    assert "Recovered pending committed purge cleanup" in retry.output
    assert not transaction.exists()


def test_legacy_purge_sidecar_without_remaining_backups_recovers_after_root_change(
    monkeypatch,
    tmp_path,
) -> None:
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
    transaction = home / ".config/open-xquant/agent-uninstall-transaction.json"
    original_unlink = Path.unlink

    def fail_transaction_unlink(path: Path, *args, **kwargs) -> None:
        if path == transaction:
            raise OSError("injected transaction unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_transaction_unlink)
    initial = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )
    assert initial.exit_code == 1
    assert transaction.exists()
    assert not any(".backup-" in path.name for path in tmp_path.rglob("*"))
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("manifest_backup_sha256")
    payload.pop("trusted_roots")
    transaction.write_text(json.dumps(payload), encoding="utf-8")
    replacement_codex_home.mkdir()
    replacement_sentinel = replacement_codex_home / "keep.txt"
    replacement_sentinel.write_text("keep me\n", encoding="utf-8")
    monkeypatch.setattr(Path, "unlink", original_unlink)
    monkeypatch.setenv("CODEX_HOME", str(replacement_codex_home))

    retry = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert retry.exit_code == 0, retry.output
    assert "Recovered pending committed purge cleanup" in retry.output
    assert not transaction.exists()
    assert replacement_sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_agent_uninstall_purge_recovery_rejects_malformed_unowned_backup_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    user_root = home / "documents"
    user_root.mkdir(parents=True)
    user_backup = user_root / ".notes.backup-forged"
    user_backup.write_text("keep me\n", encoding="utf-8")
    config_root = home / ".config/open-xquant"
    config_root.mkdir(parents=True)
    transaction = config_root / "agent-uninstall-transaction.json"
    transaction.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_type": "agent-uninstall-purge",
                "phase": "committed",
                "backups": [
                    {
                        "destination": str(user_root / "notes"),
                        "backup": str(user_backup),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = _snapshot_tree(home)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code == 1
    assert "Invalid pending purge cleanup metadata" in result.output
    assert _snapshot_tree(home) == before
    assert user_backup.read_text(encoding="utf-8") == "keep me\n"


def test_agent_uninstall_purge_recovery_rejects_symlinked_transaction_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_root = home / ".config/open-xquant"
    config_root.mkdir(parents=True)
    external_transaction = tmp_path / "external-transaction.json"
    external_transaction.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_type": "agent-uninstall-purge",
                "phase": "committed",
                "backups": [],
            }
        ),
        encoding="utf-8",
    )
    transaction = config_root / "agent-uninstall-transaction.json"
    transaction.symlink_to(external_transaction)
    before = _snapshot_tree(home)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code == 1
    assert "Invalid pending purge cleanup metadata" in result.output
    assert _snapshot_tree(home) == before
    assert external_transaction.exists()


def test_agent_uninstall_purge_recovery_unlinks_dangling_backup_symlink(
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_root = home / ".config/open-xquant"
    config_root.mkdir(parents=True)
    backup = config_root / ".agent-install.json.backup-abcdefgh"
    external_target = tmp_path / "missing-external-target"
    backup.symlink_to(external_target)
    transaction = config_root / "agent-uninstall-transaction.json"
    transaction.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_type": "agent-uninstall-purge",
                "phase": "committed",
                "backups": [
                    {
                        "destination": str(config_root / "agent-install.json"),
                        "backup": str(backup),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert not backup.is_symlink()
    assert not transaction.exists()
    assert not external_target.exists()


def test_agent_uninstall_purge_dry_run_does_not_clean_up_or_mutate(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".config/opencode"
    config_root = home / ".config/open-xquant"
    target_before = _snapshot_tree(target_root)
    config_before = _snapshot_tree(config_root)

    def unexpected_cleanup(_committed):
        raise AssertionError("dry-run invoked transaction backup cleanup")

    monkeypatch.setattr(agent_module, "_cleanup_upgrade_backups", unexpected_cleanup)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--dry-run", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert _snapshot_tree(target_root) == target_before
    assert _snapshot_tree(config_root) == config_before


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


def test_agent_uninstall_rolls_back_when_sdk_bundle_remove_fails(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    target_root = home / ".config/opencode"
    config_root = home / ".config/open-xquant"
    manifest = config_root / "agent-install.json"
    config = config_root / "agent.yaml"
    bundle = config_root / "sdk-bundles/bundle-test"
    target_before = _snapshot_tree(target_root)
    manifest_before = manifest.read_bytes()
    config_before = config.read_bytes()
    bundle_before = _snapshot_tree(bundle)
    calls = 0

    def fail_remove(_bundle, _config_root) -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(agent_module, "remove_sdk_bundle", fail_remove)

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code == 1
    assert calls == 1
    assert _snapshot_tree(target_root) == target_before
    assert manifest.read_bytes() == manifest_before
    assert config.read_bytes() == config_before
    assert _snapshot_tree(bundle) == bundle_before


def test_agent_uninstall_purge_preflights_all_sdk_bundles_before_deleting(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output
    config_root = home / ".config/open-xquant"
    current_bundle = config_root / "sdk-bundles/bundle-test"
    broken_bundle = config_root / "sdk-bundles/broken-bundle"
    broken_wheel = broken_bundle / "dist/open_xquant-0.1.0-py3-none-any.whl"
    broken_lock = broken_bundle / "requirements.lock.txt"
    broken_packages = broken_bundle / "packages.json"
    broken_python = broken_bundle / "runner/.venv/bin/python"
    broken_oxq = broken_bundle / "runner/.venv/bin/oxq"
    broken_wheel.parent.mkdir(parents=True)
    broken_python.parent.mkdir(parents=True)
    broken_wheel.write_text("wheel", encoding="utf-8")
    broken_lock.write_text("corrupted\n", encoding="utf-8")
    broken_packages.write_text("[]\n", encoding="utf-8")
    broken_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    broken_oxq.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    broken_python.chmod(0o755)
    broken_oxq.chmod(0o755)
    manifest = config_root / "agent-install.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    broken_payload = dict(payload["sdk_bundle"])
    broken_payload["id"] = "broken-bundle"
    broken_payload["root"] = str(broken_bundle)
    broken_payload["wheel"] = {
        **broken_payload["wheel"],
        "path": str(broken_wheel),
        "sha256": hashlib.sha256(b"wheel").hexdigest(),
    }
    broken_payload["dependencies"] = {
        **broken_payload["dependencies"],
        "lock_file": str(broken_lock),
        "lock_sha256": hashlib.sha256(b"expected-lock\n").hexdigest(),
        "packages_file": str(broken_packages),
    }
    broken_payload["runner"] = {
        **broken_payload["runner"],
        "venv": str(broken_bundle / "runner/.venv"),
        "python": str(broken_python),
        "oxq": str(broken_oxq),
        "argv": [str(broken_oxq)],
    }
    payload["sdk_bundles"] = [payload["sdk_bundle"], broken_payload]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["agent", "uninstall", "--all-targets", "--purge-config", "--yes"],
    )

    assert result.exit_code != 0
    assert "Refusing to purge config" in result.output
    assert current_bundle.exists()
    assert broken_bundle.exists()
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
    skill_dir = home / ".config/opencode/skills/build-strategy-spec"
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
