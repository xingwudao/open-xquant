from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from oxq.cli.doctor import _check_data, _check_deps, _check_workspace
from oxq.cli.main import main


def _write_source(root: Path) -> None:
    skills = root / "agent" / "skills"
    skill_dir = skills / "build-strategy-spec"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: build-strategy-spec\ndescription: Build quant strategies\n---\n\n# Strategy Builder\n",
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


def test_doctor_json_reports_missing_workspace_fix(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    _write_source(source)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)

    install = CliRunner().invoke(
        main,
        ["agent", "install", "--target", "opencode", "--from-local", str(source), "--yes"],
    )
    assert install.exit_code == 0, install.output

    result = CliRunner().invoke(main, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checks"]["agent"]["status"] == "ok"
    assert payload["checks"]["workspace"]["status"] == "missing"
    assert "oxq research init" in payload["fixes"]


def test_doctor_json_fix_outputs_only_json(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["doctor", "--json", "--fix"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checks"]["workspace"]["status"] == "ok"
    assert (work / ".open-xquant" / "workspace.yaml").exists()


def test_doctor_data_check_uses_market_data_directory(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    (home / ".oxq/data").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    result = _check_data()

    assert result["status"] == "warn"
    assert result["path"].endswith(".oxq/data/market")


def test_doctor_accepts_legacy_workspace_layout(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  specs_dir: strategy_specs",
                "  runs_dir: runs",
                "  reports_dir: reports",
                "  experiment_registry: experiments.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    (work / "strategy_specs").mkdir()
    (work / "runs").mkdir()
    (work / "reports").mkdir()
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "ok"
    assert result["missing"] == []


def test_doctor_warns_when_configured_comparison_registry_is_missing(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  runs_dir: runs",
                "  final_dir: runs/final",
                "  comparisons_dir: comparisons",
                "  experiment_registry: experiments.jsonl",
                "  comparison_registry: comparisons/comparisons.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    (work / "runs/final").mkdir(parents=True)
    (work / "comparisons").mkdir()
    (work / "experiments.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(work)

    result = _check_workspace()

    assert result["status"] == "warn"
    assert str(work / "comparisons" / "comparisons.jsonl") in result["missing"]


def test_doctor_json_reports_malformed_workspace_config(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    (work / ".open-xquant").mkdir(parents=True)
    (work / ".open-xquant" / "workspace.yaml").write_text("paths: [broken", encoding="utf-8")
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checks"]["workspace"]["status"] == "fail"
    assert "workspace.yaml" in payload["checks"]["workspace"]["path"]
    assert "oxq research init --force" in payload["fixes"]


def test_doctor_deps_separates_core_and_optional_missing(monkeypatch) -> None:
    missing = {"pyarrow", "pandas", "numpy", "yaml", "scipy", "matplotlib", "yfinance"}

    def fake_find_spec(module: str):
        return None if module in missing else object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    result = _check_deps()

    assert result["status"] == "fail"
    assert "pyarrow" in result["missing_core"]
    assert "pandas" in result["missing_core"]
    assert "numpy" in result["missing_core"]
    assert "yaml" in result["missing_core"]
    assert "scipy" in result["missing_optional"]
    assert "matplotlib" in result["missing_optional"]
    assert "yfinance" in result["missing_optional"]
    assert "uv sync --all-extras" in result["fixes"]


def test_doctor_deps_warns_when_only_optional_missing(monkeypatch) -> None:
    missing = {"scipy", "mplfinance", "seaborn"}

    def fake_find_spec(module: str):
        return None if module in missing else object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    result = _check_deps()

    assert result["status"] == "warn"
    assert result["missing_core"] == []
    assert result["missing_optional"] == ["mplfinance", "scipy", "seaborn"]
    assert "uv sync --extra scipy" in result["fixes"]
    assert "uv sync --extra chart" in result["fixes"]
