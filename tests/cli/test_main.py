from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

import oxq.cli.main as main_module
import oxq.run_digests as run_digests_module
from oxq.cli.main import _run_comparability_signature, main
from oxq.core.component_catalog import build_component_catalog, component_catalog_json
from oxq.core.component_manifest import compute_component_bundle_hash
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.run_digests import (
    publish_run_artifacts,
    replace_run_digest_entry,
    require_current_run_digest,
    run_digest_transaction,
)
from oxq.spec.compiler import _append_run_digest, _hash_file, _hash_json_file, _write_artifacts
from oxq.spec.schema import StrategySpec


def _write_experiment_identity(run_dir: Path, strategy_id: str) -> None:
    spec = StrategySpec.template(strategy_id=strategy_id, hypothesis="experiment identity fixture")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")


def _spec_audit_context() -> dict[str, object]:
    return {
        "strategy_idea_brief": "versions/v001/01_brainstorm/strategy_idea_brief.json",
        "strategy_idea_audit": "versions/v001/02_idea_audit/strategy_idea_audit.json",
        "strategy_idea_brief_hash": "sha256:" + "5" * 16,
        "strategy_idea_audit_hash": "sha256:" + "6" * 16,
        "unsupported_mappings": [],
    }


def _confirmation_event(
    artifact_path: str = "versions/v001/06_spec_audit/spec_confirmation_table.md",
    artifact_hash: str = "sha256:" + "4" * 16,
) -> dict[str, object]:
    return {
        "path": "conversations/demo/confirmations.jsonl",
        "event_id": "spec-confirmation-1",
        "line_number": 1,
        "event_hash": "sha256:" + "7" * 16,
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "spec_audit_path": "spec_audit.json",
        "spec_audit_hash": "sha256:" + "8" * 16,
    }


def _write_confirmation_event(
    workspace_root: Path,
    *,
    artifact_path: str,
    artifact_hash: str,
    spec_audit_hash: str = "sha256:" + "8" * 16,
) -> dict[str, object]:
    event_reference = Path("conversations/demo/confirmations.jsonl")
    path = workspace_root / event_reference
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": "2026-07-07T08:00:00Z",
        "phase": "spec_confirmation",
        "field_scope": "full_spec_table",
        "decision": "confirmed",
        "event_id": "spec-confirmation-1",
        "user_text": "确认",
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "spec_audit_path": "spec_audit.json",
        "spec_audit_hash": spec_audit_hash,
    }
    line = json.dumps(event, sort_keys=True, ensure_ascii=False)
    line_number = len(path.read_text(encoding="utf-8").splitlines()) + 1 if path.exists() else 1
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    return {
        "path": event_reference.as_posix(),
        "event_id": event["event_id"],
        "decision": event["decision"],
        "line_number": line_number,
        "event_hash": f"sha256:{hashlib.sha256(line.encode('utf-8')).hexdigest()}",
        "artifact_path": artifact_path,
        "artifact_hash": artifact_hash,
        "spec_audit_path": "spec_audit.json",
        "spec_audit_hash": spec_audit_hash,
    }


def _pre_confirmation_spec_audit_hash(payload: dict) -> str:
    candidate = json.loads(json.dumps(payload, default=str))
    candidate.pop("confirmation_event", None)
    candidate["status"] = "block"
    candidate["user_confirmation_status"] = "pending"
    canonical = json.dumps(candidate, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _attach_confirmation_artifacts(tmp_path: Path, audit: dict) -> None:
    confirmation_table = tmp_path / "spec_confirmation_table.md"
    confirmation_table.write_text(
        "| Field | Confirmed Value |\n| --- | --- |\n| spec_hash | confirmed |\n",
        encoding="utf-8",
    )
    table_hash = _hash_file(confirmation_table)
    audit["spec_confirmation_table"] = {
        "path": str(confirmation_table),
        "hash": table_hash,
        "hash_type": "sha256",
    }
    audit["confirmation_event"] = _write_confirmation_event(
        tmp_path,
        artifact_path=str(confirmation_table),
        artifact_hash=table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(audit),
    )


def test_robustness_run_exits_nonzero_for_error(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run_robustness(_run_dir: str) -> dict:
        return {"status": "error", "tests": [], "message": "missing data"}

    monkeypatch.setattr("oxq.robustness.run_robustness", fake_run_robustness)

    result = CliRunner().invoke(main, ["robustness", "run", str(run_dir), "--json"])

    assert result.exit_code == 1
    assert "missing data" in result.output


def test_robustness_run_exits_nonzero_for_fragile(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run_robustness(_run_dir: str) -> dict:
        return {
            "status": "fragile",
            "tests": [{"name": "cost_x2", "status": "fail", "message": "sharpe collapsed"}],
            "baseline_sharpe": 1.0,
        }

    monkeypatch.setattr("oxq.robustness.run_robustness", fake_run_robustness)

    result = CliRunner().invoke(main, ["robustness", "run", str(run_dir)])

    assert result.exit_code == 1
    assert "Status: FRAGILE" in result.output


def test_spec_init_generates_path_safe_strategy_id(tmp_path) -> None:
    out = tmp_path / "strategy_spec.yaml"

    result = CliRunner().invoke(main, ["spec", "init", "SMA/RSI crossover!!!", "--out", str(out)])

    assert result.exit_code == 0, result.output
    spec = yaml.safe_load(out.read_text(encoding="utf-8"))
    template = StrategySpec.template()
    assert spec["schema_version"] == "0.1"
    assert spec["strategy_id"] == "sma_rsi_crossover"
    assert spec["required_oxq_version"] == template.required_oxq_version
    assert spec["required_oxq_version"] != spec["schema_version"]


def test_spec_init_can_generate_china_a_share_candidate_template(tmp_path) -> None:
    out = tmp_path / "strategy_spec.yaml"

    result = CliRunner().invoke(
        main,
        [
            "spec",
            "init",
            "A-share momentum TopN",
            "--market-preset",
            "cn_a_share",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    spec = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert spec["market"] == {
        "asset_class": "equity",
        "region": "cn",
        "currency": "CNY",
        "calendar": "XSHG",
    }
    assert spec["universe"]["type"] == "static"
    assert spec["universe"]["symbols"] == ["600519.SH", "000001.SZ"]
    assert spec["benchmark"]["symbols"] == ["000300.SH"]
    assert spec["execution"]["lot_size"] == 100
    assert spec["execution"]["lot_size_config"]["default"] == 100
    assert "candidate values; Agent workflows must still collect user confirmation" in result.output


def test_spec_init_defaults_to_active_version_spec_path(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": {"current_manifest": "current.json"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "v007"}), encoding="utf-8")

        result = runner.invoke(main, ["spec", "init", "version governed spec"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / "versions/v007/04_spec_build/strategy_spec.yaml").exists()
        assert not (cwd_path / "strategy_spec.yaml").exists()


def test_spec_init_treats_versions_dir_workspace_as_version_governed(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {"paths": {"versions_dir": "versions", "current_manifest": "current.json"}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "v009"}), encoding="utf-8")

        result = runner.invoke(main, ["spec", "init", "versions dir governed"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / "versions/v009/04_spec_build/strategy_spec.yaml").exists()
        assert not (cwd_path / "strategy_spec.yaml").exists()


def test_spec_init_honors_custom_versions_dir_workspace(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {"paths": {"versions_dir": "research_versions", "current_manifest": "current.json"}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "v010"}), encoding="utf-8")

        result = runner.invoke(main, ["spec", "init", "custom versions dir"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / "research_versions/v010/04_spec_build/strategy_spec.yaml").exists()
        assert not (cwd_path / "versions/v010/04_spec_build/strategy_spec.yaml").exists()
        assert not (cwd_path / "strategy_spec.yaml").exists()


def test_spec_init_rejects_symlinked_versions_dir_escape(tmp_path) -> None:
    runner = CliRunner()
    outside = tmp_path / "outside"
    outside.mkdir()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / "versions_link").symlink_to(outside, target_is_directory=True)
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {"paths": {"versions_dir": "versions_link", "current_manifest": "current.json"}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "v011"}), encoding="utf-8")

        result = runner.invoke(main, ["spec", "init", "symlink escape"])

        assert result.exit_code == 1
        assert "paths.versions_dir must not contain symlink components" in result.output
        assert not (outside / "v011").exists()


def test_spec_init_rejects_symlinked_active_version_dir_escape_before_mutation(tmp_path) -> None:
    runner = CliRunner()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("unchanged\n", encoding="utf-8")
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / "versions").mkdir()
        (cwd_path / "versions/v011").symlink_to(outside, target_is_directory=True)
        (cwd_path / ".open-xquant/workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": {"versions_dir": "versions", "current_manifest": "current.json"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v011"}),
            encoding="utf-8",
        )
        before = {
            path.relative_to(outside).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(outside.rglob("*"))
        }

        result = runner.invoke(main, ["spec", "init", "active version symlink escape"])

        assert result.exit_code == 1
        assert "active version directory must stay within the workspace" in result.output
        assert {
            path.relative_to(outside).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(outside.rglob("*"))
        } == before


def test_spec_init_legacy_fallback_rejects_symlinked_spec_phase_before_mutation(tmp_path) -> None:
    runner = CliRunner()
    outside = tmp_path / "outside"
    outside.mkdir()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        workspace = Path(cwd)
        version_dir = workspace / "versions/v012"
        (workspace / ".open-xquant").mkdir()
        version_dir.mkdir(parents=True)
        (version_dir / "04_spec_build").symlink_to(outside, target_is_directory=True)
        (workspace / ".open-xquant/workspace.yaml").write_text(
            "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
            encoding="utf-8",
        )
        (workspace / "current.json").write_text(json.dumps({"active_version": "v012"}), encoding="utf-8")

        result = runner.invoke(main, ["spec", "init", "legacy symlink phase"])

        assert result.exit_code != 0
        assert "04_spec_build" in result.output
        assert "symlink" in result.output
        assert list(outside.iterdir()) == []


def test_experiment_add_honors_custom_workspace_versions_dir(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    config_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"versions_dir": "research_versions"},
                "workflow": {"layout": "version_governed"},
            }
        ),
        encoding="utf-8",
    )
    run_dir = work / "research_versions/v003/09_backtests/run_1"
    run_dir.mkdir(parents=True)
    (work / "current.json").write_text(
        json.dumps({"active_version": "v003"}),
        encoding="utf-8",
    )
    (work / "research_versions/v003/version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v003",
                "phase_paths": {
                    "09_backtests": "research_versions/v003/09_backtests",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_root", "run_id": "run_1"}),
        encoding="utf-8",
    )
    _write_experiment_identity(run_dir, "custom_root")
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["experiment", "add", str(run_dir)])

    assert result.exit_code == 0, result.output
    entry = json.loads((work / "experiments.jsonl").read_text(encoding="utf-8"))
    assert entry["version_id"] == "v003"
    assert not (work / "versions").exists()


@pytest.mark.parametrize("governance_state", ["missing_current", "missing_active_version", "missing_manifest"])
def test_experiment_add_fails_closed_for_incomplete_governance(
    monkeypatch,
    tmp_path,
    governance_state: str,
) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    run_dir = work / "versions/v999/09_backtests/run_1"
    config_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"versions_dir": "versions"},
                "workflow": {"layout": "version_governed"},
            }
        ),
        encoding="utf-8",
    )
    if governance_state != "missing_current":
        current = {"active_version": "v999"} if governance_state != "missing_active_version" else {}
        (work / "current.json").write_text(json.dumps(current), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "fail_closed", "run_id": "run_1"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["experiment", "add", str(run_dir)])

    assert result.exit_code == 1
    assert "version-governed workspace" in result.output
    assert not (work / "experiments.jsonl").exists()
    assert not (run_dir / "research_bias_audit.json").exists()


def test_experiment_add_keeps_structural_fallback_for_legacy_workspace(monkeypatch, tmp_path) -> None:
    work = tmp_path / "legacy"
    run_dir = work / "versions/v003/09_backtests/run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "legacy", "run_id": "run_1"}),
        encoding="utf-8",
    )
    _write_experiment_identity(run_dir, "legacy")
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["experiment", "add", str(run_dir)])

    assert result.exit_code == 0, result.output
    entry = json.loads((work / "experiments.jsonl").read_text(encoding="utf-8"))
    assert entry["version_id"] == "v003"


def test_experiment_add_rejects_arbitrary_structural_version_in_governed_workspace(
    monkeypatch,
    tmp_path,
) -> None:
    work = tmp_path / "work"
    run_dir = work / "versions/v999/09_backtests/run_1"
    (work / ".open-xquant").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (work / ".open-xquant/workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"versions_dir": "versions"},
                "workflow": {"layout": "version_governed"},
            }
        ),
        encoding="utf-8",
    )
    (work / "current.json").write_text(json.dumps({"active_version": "v003"}), encoding="utf-8")
    active_version_dir = work / "versions/v003"
    active_version_dir.mkdir(parents=True)
    (active_version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v003",
                "phase_paths": {"09_backtests": "versions/v003/09_backtests"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "arbitrary_version", "run_id": "run_1"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["experiment", "add", str(run_dir)])

    assert result.exit_code == 1
    assert "resolved backtest phase directory" in result.output
    assert not (work / "experiments.jsonl").exists()
    assert not (run_dir / "research_bias_audit.json").exists()


def test_experiment_add_honors_custom_workspace_registry(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    config_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump({"paths": {"experiment_registry": "governance/registry.jsonl"}}),
        encoding="utf-8",
    )
    run_dir = work / "run_1"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_registry", "run_id": "run_1"}),
        encoding="utf-8",
    )
    _write_experiment_identity(run_dir, "custom_registry")
    default_registry = work / "experiments.jsonl"
    default_registry.write_bytes(b"existing-default\n")
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["experiment", "add", str(run_dir)])

    assert result.exit_code == 0, result.output
    custom_registry = work / "governance/registry.jsonl"
    entry = json.loads(custom_registry.read_text(encoding="utf-8"))
    assert entry["strategy_id"] == "custom_registry"
    assert default_registry.read_bytes() == b"existing-default\n"


def test_experiment_add_honors_manifest_backtest_phase_path(monkeypatch, tmp_path) -> None:
    work = tmp_path / "work"
    config_dir = work / ".open-xquant"
    version_dir = work / "research_versions/v003"
    backtest_dir = version_dir / "artifacts/backtests"
    run_dir = backtest_dir / "run_1_cost_x2"
    config_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "versions_dir": "research_versions",
                    "current_manifest": "current.json",
                },
                "workflow": {"layout": "version_governed"},
            }
        ),
        encoding="utf-8",
    )
    (work / "current.json").write_text(
        json.dumps({"active_version": "v003"}),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v003",
                "phase_paths": {
                    "09_backtests": "research_versions/v003/artifacts/backtests"
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_phase", "run_id": "run_1_cost_x2"}),
        encoding="utf-8",
    )
    _write_experiment_identity(run_dir, "custom_phase")
    monkeypatch.chdir(work)

    result = CliRunner().invoke(main, ["experiment", "add", str(run_dir)])

    assert result.exit_code == 0, result.output
    entry = json.loads((work / "experiments.jsonl").read_text(encoding="utf-8"))
    assert entry["version_id"] == "v003"
    assert entry["run_role"] == "robustness_cost_x2"


def test_spec_init_fails_when_version_workspace_lacks_current_manifest(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": {"current_manifest": "current.json"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["spec", "init", "missing current"])

        assert result.exit_code == 1
        assert "run `oxq research init` to repair manifests" in result.output
        assert not (cwd_path / "strategy_spec.yaml").exists()


def test_spec_init_fails_for_unsafe_active_version(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": {"current_manifest": "current.json"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(json.dumps({"active_version": "../escape"}), encoding="utf-8")

        result = runner.invoke(main, ["spec", "init", "unsafe current"])

        assert result.exit_code == 1
        assert "requires a safe current.json active_version" in result.output
        assert not (cwd_path / "strategy_spec.yaml").exists()
        assert not (cwd_path.parent / "escape").exists()


def test_spec_init_fails_for_hidden_current_manifest_path(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        (cwd_path / ".open-xquant").mkdir()
        (cwd_path / ".open-xquant" / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": {"current_manifest": ".open-xquant/current.json"},
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / ".open-xquant/current.json").write_text(
            json.dumps({"active_version": "v007"}),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["spec", "init", "hidden current"])

        assert result.exit_code == 1
        assert "requires root current.json active_version" in result.output
        assert not (cwd_path / "strategy_spec.yaml").exists()


def test_spec_init_uses_manifest_spec_phase_path_with_custom_versions_root(tmp_path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "research_versions/v003"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "versions_dir": "research_versions",
                        "current_manifest": "current.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v003"}),
            encoding="utf-8",
        )
        spec_phase = "research_versions/v003/artifacts/specs"
        (version_dir / "version_manifest.json").write_text(
            json.dumps(
                {
                    "version_id": "v003",
                    "phase_paths": {"04_spec_build": spec_phase},
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["spec", "init", "custom spec phase"])

        assert result.exit_code == 0, result.output
        assert (cwd_path / spec_phase / "strategy_spec.yaml").is_file()
        assert not (version_dir / "04_spec_build").exists()


@pytest.mark.parametrize(
    "phase_paths",
    [
        {},
        {"04_spec_build": "../escape"},
    ],
)
def test_spec_init_rejects_invalid_manifest_spec_phase_path_before_mutation(
    tmp_path,
    phase_paths: dict[str, str],
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        config_dir = cwd_path / ".open-xquant"
        version_dir = cwd_path / "research_versions/v003"
        config_dir.mkdir()
        version_dir.mkdir(parents=True)
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "versions_dir": "research_versions",
                        "current_manifest": "current.json",
                    },
                    "workflow": {"layout": "version_governed"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (cwd_path / "current.json").write_text(
            json.dumps({"active_version": "v003"}),
            encoding="utf-8",
        )
        (version_dir / "version_manifest.json").write_text(
            json.dumps({"version_id": "v003", "phase_paths": phase_paths}),
            encoding="utf-8",
        )
        before = {
            path.relative_to(cwd_path).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(cwd_path.rglob("*"))
        }

        result = runner.invoke(main, ["spec", "init", "invalid spec phase"])

        assert result.exit_code == 1
        assert "phase_paths.04_spec_build" in result.output
        assert {
            path.relative_to(cwd_path).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(cwd_path.rglob("*"))
        } == before


def test_registry_export_writes_component_catalog(tmp_path) -> None:
    out = tmp_path / "component_catalog.json"

    result = CliRunner().invoke(main, ["registry", "export", "--out", str(out)])

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["catalog_hash"].startswith("sha256:")
    assert payload["recipe_catalog_hash"].startswith("sha256:")
    assert {item["name"] for item in payload["indicators"]} >= {"NdayReturn", "RollingVolatility", "Ratio"}
    assert {item["name"] for item in payload["recipes"]} >= {
        "roc_timing",
        "sma_golden_cross",
        "threshold_then_rank_top_n",
        "top_n_normalized_weights",
        "top_n_positive_momentum_rotation",
        "volatility_adjusted_momentum",
    }
    assert "Catalog hash:" in result.output


def test_component_manifest_loads_workspace_indicator_and_updates_catalog(tmp_path) -> None:
    manifest = _write_custom_indicator_extension(tmp_path)

    hash_result = CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"])

    assert hash_result.exit_code == 0, hash_result.output
    digest = json.loads(hash_result.output)["component_bundle_hash"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    validate_result = CliRunner().invoke(main, ["component-manifest", "validate", str(manifest), "--json"])

    assert validate_result.exit_code == 0, validate_result.output
    assert json.loads(validate_result.output)["computed_bundle_hash"] == digest

    catalog_path = tmp_path / "component_catalog.json"
    export_result = CliRunner().invoke(
        main,
        ["registry", "export", "--component-manifest", str(manifest), "--out", str(catalog_path)],
    )

    assert export_result.exit_code == 0, export_result.output
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    custom = next(item for item in catalog["indicators"] if item["name"] == "WorkspaceConstantIndicator")
    assert custom["source"] == "workspace_extension"
    assert custom["bundle_hash"] == digest
    assert custom["manifest_path"] == str(manifest.resolve())
    assert custom["manifest_parameters"] == {"value": 1.0}
    assert custom["params"]["value"]["default"] == 1.0
    assert custom["params"]["value"]["manifest_value"] == 1.0
    assert custom["params"]["value"]["required"] is False


def test_component_manifest_hash_includes_non_python_resources(tmp_path) -> None:
    manifest = _write_custom_indicator_extension(tmp_path)
    resource = tmp_path / "custom_components" / "resources" / "lookup.json"
    resource.parent.mkdir(parents=True)
    resource.write_text(json.dumps({"threshold": 1}), encoding="utf-8")

    first = CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"])
    resource.write_text(json.dumps({"threshold": 2}), encoding="utf-8")
    second = CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output)["component_bundle_hash"] != json.loads(second.output)["component_bundle_hash"]


def test_component_manifest_hash_rejects_symlinked_bundle_file(tmp_path) -> None:
    manifest = _write_custom_indicator_extension(tmp_path)
    outside = tmp_path / "outside_lookup.json"
    outside.write_text(json.dumps({"threshold": 1}), encoding="utf-8")
    resource = tmp_path / "custom_components" / "resources" / "lookup.json"
    resource.parent.mkdir(parents=True)
    resource.symlink_to(outside)

    result = CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"])

    assert result.exit_code == 1
    assert "must not be a symlink" in str(result.exception)


def test_component_manifest_hash_rejects_symlink_to_manifest_in_bundle_root(tmp_path) -> None:
    module = tmp_path / "workspace_indicator.py"
    module.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "class WorkspaceRootIndicator:",
                "    name = 'WorkspaceRootIndicator'",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "workspace_root",
                "extension_root": ".",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "WorkspaceRootIndicator",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "workspace_indicator",
                        "class": "WorkspaceRootIndicator",
                        "protocol": "Indicator",
                        "source_hash": "sha256:" + hashlib.sha256(module.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    alias = tmp_path / "manifest_alias.json"
    alias.symlink_to(manifest)

    result = CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"])

    assert result.exit_code == 1
    assert "must not be a symlink" in str(result.exception)


def test_component_manifest_hash_skips_manifest_when_extension_root_is_workspace(tmp_path) -> None:
    module = tmp_path / "workspace_indicator.py"
    module.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "class WorkspaceRootIndicator:",
                "    name = 'WorkspaceRootIndicator'",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "workspace_root",
                "extension_root": ".",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "WorkspaceRootIndicator",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "workspace_indicator",
                        "class": "WorkspaceRootIndicator",
                        "protocol": "Indicator",
                        "source_hash": "sha256:" + hashlib.sha256(module.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    result = CliRunner().invoke(main, ["component-manifest", "validate", str(manifest), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_component_manifest_validate_does_not_persist_workspace_component(tmp_path) -> None:
    manifest = _write_custom_indicator_extension(tmp_path)
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    spec = StrategySpec.template(strategy_id="custom_indicator_scope", hypothesis="custom indicators are scoped")
    spec.signal.indicators = {
        "custom_score": {
            "type": "WorkspaceConstantIndicator",
            "params": {"value": 2.0},
        }
    }
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {"score_col": "custom_score", "n": 1}
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")

    validate_manifest = CliRunner().invoke(main, ["component-manifest", "validate", str(manifest), "--json"])
    without_manifest = CliRunner().invoke(main, ["spec", "validate", str(spec_path), "--json"])

    assert validate_manifest.exit_code == 0, validate_manifest.output
    assert without_manifest.exit_code == 1
    assert "WorkspaceConstantIndicator" in without_manifest.output


def test_component_manifest_rejects_declared_name_mismatch(tmp_path) -> None:
    manifest = _write_custom_indicator_extension(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["components"][0]["name"] = "DeclaredButNotRegistered"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    result = CliRunner().invoke(main, ["component-manifest", "validate", str(manifest), "--json"])

    assert result.exit_code == 1
    assert "must match registered class name" in result.output


def test_component_manifest_rejects_duplicate_registered_name(tmp_path) -> None:
    root = tmp_path / "custom_components"
    root.mkdir()
    source = root / "workspace_roc.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "class ROC:",
                "    name = 'ROC'",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "ROC",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "workspace_roc",
                        "class": "ROC",
                        "protocol": "Indicator",
                        "source_hash": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = compute_component_bundle_hash(manifest)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    result = CliRunner().invoke(main, ["component-manifest", "validate", str(manifest), "--json"])

    assert result.exit_code == 1
    assert "already exists in the Indicator registry" in result.output


def test_component_manifest_rejects_module_outside_extension_root(tmp_path) -> None:
    import sys

    root = tmp_path / "custom_components"
    root.mkdir()
    manifest = tmp_path / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "JSONDecoder",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "json",
                        "class": "JSONDecoder",
                        "protocol": "Indicator",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    digest = json.loads(CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output)[
        "component_bundle_hash"
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    result = CliRunner().invoke(main, ["component-manifest", "validate", str(manifest), "--json"])

    assert result.exit_code == 1
    assert "must resolve inside the component extension root" in result.output
    assert sys.modules["json"] is json


def test_component_manifest_clears_single_module_cache(tmp_path) -> None:
    import sys
    import types

    from oxq.core.component_manifest import _clear_extension_module_cache

    root = tmp_path / "custom_components"
    root.mkdir()
    module_file = root / "workspace_indicator.py"
    module_file.write_text("", encoding="utf-8")
    module = types.ModuleType("workspace_indicator")
    module.__file__ = str(module_file)
    sys.modules["workspace_indicator"] = module

    _clear_extension_module_cache(
        {
            "schema_version": 1,
            "extension_id": "custom_components",
            "components": [
                {
                    "name": "WorkspaceIndicator",
                    "kind": "Indicator",
                    "module": "workspace_indicator",
                    "class": "WorkspaceIndicator",
                }
            ],
        },
        root,
    )

    assert "workspace_indicator" not in sys.modules


def test_component_manifest_clears_helper_modules_from_previous_extension(tmp_path) -> None:
    import sys

    from oxq.core.component_manifest import load_component_manifest, scoped_component_registries

    def write_extension(root_name: str, class_name: str) -> Path:
        root = tmp_path / root_name
        root.mkdir()
        (root / "helpers.py").write_text(f"CLASS_NAME = {class_name!r}\n", encoding="utf-8")
        (root / "workspace_indicator.py").write_text(
            "\n".join(
                [
                    "from __future__ import annotations",
                    "import pandas as pd",
                    "import helpers",
                    "class WorkspaceIndicator:",
                    "    name = helpers.CLASS_NAME",
                    "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                    "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest = tmp_path / f"{root_name}_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "extension_id": root_name,
                    "extension_root": root_name,
                    "bundle_hash": "",
                    "components": [
                        {
                            "name": class_name,
                            "kind": "Indicator",
                            "source": "workspace_extension",
                            "module": "workspace_indicator",
                            "class": "WorkspaceIndicator",
                            "protocol": "Indicator",
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["bundle_hash"] = compute_component_bundle_hash(manifest)
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    first = write_extension("first_components", "FirstIndicator")
    second = write_extension("second_components", "SecondIndicator")

    with scoped_component_registries():
        load_component_manifest(first)
        assert sys.modules["helpers"].__file__.startswith(str(tmp_path / "first_components"))
        load_component_manifest(second)

        assert sys.modules["helpers"].__file__.startswith(str(tmp_path / "second_components"))


def test_component_manifest_clears_helper_modules_when_reloading_same_extension(tmp_path) -> None:
    import sys

    from oxq.core.component_manifest import load_component_manifest, scoped_component_registries

    root = tmp_path / "custom_components"
    root.mkdir()
    helper = root / "helpers.py"
    component = root / "workspace_indicator.py"
    manifest = tmp_path / "component_manifest.json"
    component.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "import helpers",
                "class WorkspaceIndicator:",
                "    name = helpers.CLASS_NAME",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def write_manifest(class_name: str) -> None:
        helper.write_text(f"CLASS_NAME = {class_name!r}\n", encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "extension_id": "custom_components",
                    "extension_root": "custom_components",
                    "bundle_hash": "",
                    "components": [
                        {
                            "name": class_name,
                            "kind": "Indicator",
                            "source": "workspace_extension",
                            "module": "workspace_indicator",
                            "class": "WorkspaceIndicator",
                            "protocol": "Indicator",
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["bundle_hash"] = compute_component_bundle_hash(manifest)
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with scoped_component_registries():
        write_manifest("FirstIndicator")
        load_component_manifest(manifest)
        assert sys.modules["helpers"].CLASS_NAME == "FirstIndicator"

        write_manifest("SecondReloadedIndicator")
        load_component_manifest(manifest)

        assert sys.modules["helpers"].CLASS_NAME == "SecondReloadedIndicator"


def test_spec_validate_loads_workspace_component_manifest(tmp_path) -> None:
    manifest = _write_custom_indicator_extension(tmp_path)
    digest = json.loads(
        CliRunner().invoke(main, ["component-manifest", "hash", str(manifest), "--json"]).output
    )["component_bundle_hash"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = digest
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    spec = StrategySpec.template(strategy_id="custom_indicator", hypothesis="custom indicators must load")
    spec.signal.indicators = {
        "custom_score": {
            "type": "WorkspaceConstantIndicator",
            "params": {"value": 2.0},
        }
    }
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {"score_col": "custom_score", "n": 1}
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")

    missing = CliRunner().invoke(main, ["spec", "validate", str(spec_path), "--json"])
    loaded = CliRunner().invoke(
        main,
        ["spec", "validate", str(spec_path), "--component-manifest", str(manifest), "--json"],
    )
    missing_after_loaded = CliRunner().invoke(main, ["spec", "validate", str(spec_path), "--json"])

    assert missing.exit_code == 1
    assert loaded.exit_code == 0, loaded.output
    assert json.loads(loaded.output)["status"] == "pass"
    assert missing_after_loaded.exit_code == 1
    assert "WorkspaceConstantIndicator" in missing_after_loaded.output


def test_spec_hash_and_fields_are_deterministic(tmp_path) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    init_result = CliRunner().invoke(main, ["spec", "init", "SMA crossover", "--out", str(spec_path)])
    assert init_result.exit_code == 0, init_result.output

    hash_result = CliRunner().invoke(main, ["spec", "hash", str(spec_path), "--json"])
    fields_result = CliRunner().invoke(main, ["spec", "fields", str(spec_path), "--json"])

    assert hash_result.exit_code == 0, hash_result.output
    assert fields_result.exit_code == 0, fields_result.output
    digest = json.loads(hash_result.output)["spec_hash"]
    fields = json.loads(fields_result.output)
    assert digest.startswith("sha256:")
    assert fields["spec_hash"] == digest
    assert {"path": "research.hypothesis", "value": "SMA crossover"} in fields["fields"]
    assert {"path": "execution.initial_cash", "value": 100000.0} in fields["fields"]
    assert {"path": "execution.lot_size_config.default", "value": 1} in fields["fields"]


def _write_custom_indicator_extension(tmp_path):
    root = tmp_path / "custom_components"
    source_dir = root / "oxq_components" / "indicators"
    tests_dir = root / "tests"
    source_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (root / "oxq_components" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    source = source_dir / "workspace_constant_indicator.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import pandas as pd",
                "",
                "",
                "class WorkspaceConstantIndicator:",
                "    name = 'WorkspaceConstantIndicator'",
                "",
                "    def compute(self, mktdata: pd.DataFrame, value: float = 1.0) -> pd.Series:",
                "        return pd.Series(float(value), index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_file = tests_dir / "test_workspace_constant_indicator.py"
    test_file.write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )
    source_hash = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    test_hash = "sha256:" + hashlib.sha256(test_file.read_bytes()).hexdigest()
    manifest = tmp_path / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "WorkspaceConstantIndicator",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "oxq_components.indicators.workspace_constant_indicator",
                        "class": "WorkspaceConstantIndicator",
                        "protocol": "Indicator",
                        "parameters": {"value": 1.0},
                        "tests": ["custom_components/tests/test_workspace_constant_indicator.py"],
                        "source_path": "oxq_components/indicators/workspace_constant_indicator.py",
                        "source_hash": source_hash,
                        "test_hash": test_hash,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _write_hashed_custom_indicator_extension(tmp_path) -> Path:
    manifest = _write_custom_indicator_extension(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = compute_component_bundle_hash(manifest)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _write_hashed_preview_indicator_extension(
    workspace: Path,
    *,
    root_name: str,
    indicator_name: str,
    marker: str,
) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    root = workspace / root_name
    root.mkdir(parents=True, exist_ok=True)
    module_name = f"{root_name}_indicator"
    source = root / f"{module_name}.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                f"class {indicator_name}:",
                f"    name = {indicator_name!r}",
                f"    marker = {marker!r}",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = workspace / f"{root_name}_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": root_name,
                "extension_root": root_name,
                "bundle_hash": "",
                "components": [
                    {
                        "name": indicator_name,
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": module_name,
                        "class": indicator_name,
                        "protocol": "Indicator",
                        "parameters": {},
                        "source_path": source.name,
                        "source_hash": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _rehash_preview_indicator_extension(manifest)
    return manifest


def _rehash_preview_indicator_extension(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = manifest.parent / payload["extension_root"] / payload["components"][0]["source_path"]
    payload["components"][0]["source_hash"] = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    payload["bundle_hash"] = ""
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["bundle_hash"] = compute_component_bundle_hash(manifest)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compile_preview(spec_path: Path, out_dir: Path, *manifests: Path):
    args = ["strategy", "compile", str(spec_path), "--out", str(out_dir)]
    for manifest in manifests:
        args.extend(["--component-manifest", str(manifest)])
    return CliRunner().invoke(main, args)


def _directory_file_snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_formal_backtest_rejects_component_bundle_swap_after_authorization_before_import_or_execution(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _write_hashed_custom_indicator_extension(tmp_path)
    source = tmp_path / "custom_components/oxq_components/indicators/workspace_constant_indicator.py"
    replacement_import_marker = tmp_path / "replacement_imported"
    spec = StrategySpec.template(strategy_id="component_swap", hypothesis="authorized component snapshot")
    spec.signal.indicators = {
        "custom_score": {
            "type": "WorkspaceConstantIndicator",
            "params": {"value": 1.0},
        }
    }
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    gate_paths = {}
    for name in ("spec_audit.json", "runtime_audit.json", "component_catalog.json", "backtest_authorization.json"):
        gate_paths[name] = tmp_path / name
        gate_paths[name].write_text("{}\n", encoding="utf-8")
    compile_called = False

    def swap_to_valid_replacement(*args, **kwargs) -> None:
        replacement_source = source.read_text(encoding="utf-8").replace(
            "import pandas as pd",
            f"import pandas as pd\nfrom pathlib import Path\nPath({str(replacement_import_marker)!r}).write_text('imported')",
        )
        source.write_text(replacement_source, encoding="utf-8")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["components"][0]["source_hash"] = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["bundle_hash"] = compute_component_bundle_hash(manifest)
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def fake_compile_run(*args, **kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("backtest must not execute after a component source swap")

    monkeypatch.setattr("oxq.cli.main._require_pre_backtest_spec_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr("oxq.cli.main._require_pre_backtest_runtime_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr("oxq.cli.main._require_component_catalog_before_import", lambda *args, **kwargs: None)
    monkeypatch.setattr("oxq.cli.main._require_backtest_authorization", swap_to_valid_replacement)
    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "run",
            str(spec_path),
            "--spec-audit",
            str(gate_paths["spec_audit.json"]),
            "--runtime-audit",
            str(gate_paths["runtime_audit.json"]),
            "--component-catalog",
            str(gate_paths["component_catalog.json"]),
            "--authorization",
            str(gate_paths["backtest_authorization.json"]),
            "--component-manifest",
            str(manifest),
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "component bundle changed after its authorized snapshot was staged" in result.output
    assert not replacement_import_marker.exists()
    assert compile_called is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", None, "metrics.json run_id is required for comparison"),
        ("run_id", "wrong-run", "metrics.json run_id does not match run directory"),
        ("strategy_id", None, "metrics.json strategy_id is required for comparison"),
        ("strategy_id", "wrong-strategy", "metrics.json strategy_id does not match strategy_spec.yaml"),
    ],
    ids=["missing-run-id", "wrong-run-id", "missing-strategy-id", "wrong-strategy-id"],
)
def test_backtest_compare_runs_rejects_missing_or_wrong_metrics_identity(
    tmp_path,
    field: str,
    value: str | None,
    message: str,
) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    left_run = _write_minimal_cli_run(left_root)
    right_run = _write_minimal_cli_run(right_root)
    metrics_path = left_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if value is None:
        metrics.pop(field)
    else:
        metrics[field] = value
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    artifact_hashes_path = left_run / "artifact_hashes.json"
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    artifact_hashes["metrics.json"] = _hash_json_file(metrics_path, exclude_keys={"run_id"})
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8")
    replace_run_digest_entry(left_run, _hash_json_file(artifact_hashes_path))

    result = CliRunner().invoke(
        main,
        ["backtest", "compare-runs", str(left_run), str(right_run), "--json"],
    )

    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize("boundary", ["artifact:metrics.json.replace", "manifest.replace"])
def test_backtest_compare_runs_waits_for_paused_publication(monkeypatch, tmp_path, boundary: str) -> None:
    left_root = tmp_path / "left"
    left_root.mkdir()
    left_run = _write_minimal_cli_run(left_root)
    metrics = json.loads((left_run / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] = 0.25
    publication_paused = threading.Event()
    allow_publication = threading.Event()
    reader_attempted = threading.Event()
    reader_completed = threading.Event()
    publisher_failures: list[BaseException] = []
    reader_results: list[dict[str, object]] = []
    reader_failures: list[BaseException] = []
    original_boundary = run_digests_module._publication_boundary

    def pause_publication(label: str) -> None:
        if threading.current_thread().name == "compare-publisher" and label == boundary:
            publication_paused.set()
            assert allow_publication.wait(timeout=5)
        original_boundary(label)

    @contextmanager
    def observed_transaction(run_path):
        if threading.current_thread().name == "compare-reader":
            reader_attempted.set()
        with run_digest_transaction(run_path):
            yield

    def publish_metrics() -> None:
        try:
            publish_run_artifacts(left_run, {"metrics.json": json.dumps(metrics).encode()})
        except BaseException as exc:
            publisher_failures.append(exc)

    def compare() -> None:
        try:
            reader_results.append(_run_comparability_signature(left_run))
        except BaseException as exc:
            reader_failures.append(exc)
        finally:
            reader_completed.set()

    monkeypatch.setattr(run_digests_module, "_publication_boundary", pause_publication)
    monkeypatch.setattr("oxq.cli.main.run_digest_transaction", observed_transaction, raising=False)
    publisher = threading.Thread(target=publish_metrics, name="compare-publisher")
    reader = threading.Thread(target=compare, name="compare-reader")
    publisher.start()
    try:
        assert publication_paused.wait(timeout=5)
        reader.start()
        assert reader_attempted.wait(timeout=5)
        assert not reader_completed.is_set()
    finally:
        allow_publication.set()
    publisher.join(timeout=5)
    reader.join(timeout=5)

    assert not publisher.is_alive()
    assert not reader.is_alive()
    assert publisher_failures == []
    assert reader_failures == []
    assert reader_results[0]["spec_hash"]


def test_backtest_compare_runs_holds_left_lock_while_right_publisher_is_paused(monkeypatch, tmp_path) -> None:
    left_root = tmp_path / "a-left"
    right_root = tmp_path / "z-right"
    left_root.mkdir()
    right_root.mkdir()
    left_run = _write_minimal_cli_run(left_root)
    right_run = _write_minimal_cli_run(right_root)
    right_metrics = json.loads((right_run / "metrics.json").read_text(encoding="utf-8"))
    left_metrics = json.loads((left_run / "metrics.json").read_text(encoding="utf-8"))
    right_metrics["total_return"] = 0.25
    left_metrics["total_return"] = 0.25
    right_paused = threading.Event()
    allow_right = threading.Event()
    compare_waiting_for_right = threading.Event()
    left_completed = threading.Event()
    failures: list[BaseException] = []
    compare_results = []
    original_boundary = run_digests_module._publication_boundary
    original_lock = run_digests_module.ProcessFileLock
    right_lock_path = (right_run.parent / "run_digests.jsonl.lock").resolve(strict=False)

    def pause_right_publication(label: str) -> None:
        if threading.current_thread().name == "right-publisher" and label == "artifact:metrics.json.replace":
            right_paused.set()
            assert allow_right.wait(timeout=5)
        original_boundary(label)

    class ObservedLock:
        def __init__(self, path: str | Path) -> None:
            self.path = Path(path).resolve(strict=False)
            self.delegate = original_lock(path)
            if threading.current_thread().name == "compare-reader" and self.path == right_lock_path:
                compare_waiting_for_right.set()

        def __enter__(self):
            self.delegate.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            return self.delegate.__exit__(exc_type, exc, traceback)

    def publish_right() -> None:
        try:
            publish_run_artifacts(right_run, {"metrics.json": json.dumps(right_metrics).encode()})
        except BaseException as exc:
            failures.append(exc)

    def publish_left() -> None:
        try:
            publish_run_artifacts(left_run, {"metrics.json": json.dumps(left_metrics).encode()})
        except BaseException as exc:
            failures.append(exc)
        finally:
            left_completed.set()

    def compare() -> None:
        try:
            compare_results.append(
                CliRunner().invoke(
                    main,
                    ["backtest", "compare-runs", str(left_run), str(right_run), "--json"],
                )
            )
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(run_digests_module, "_publication_boundary", pause_right_publication)
    monkeypatch.setattr(run_digests_module, "ProcessFileLock", ObservedLock)
    right_publisher = threading.Thread(target=publish_right, name="right-publisher")
    compare_reader = threading.Thread(target=compare, name="compare-reader")
    left_publisher = threading.Thread(target=publish_left, name="left-publisher")
    right_publisher.start()
    try:
        assert right_paused.wait(timeout=5)
        compare_reader.start()
        assert compare_waiting_for_right.wait(timeout=5)
        left_publisher.start()
        assert not left_completed.wait(timeout=0.2)
    finally:
        allow_right.set()
    for thread in (right_publisher, compare_reader, left_publisher):
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in (right_publisher, compare_reader, left_publisher))
    assert failures == []
    assert compare_results[0].exit_code in {0, 1}, compare_results[0].output
    assert json.loads(compare_results[0].output)["status"] in {"pass", "fail"}


def _write_governed_compare_runs(tmp_path: Path, *, distinct_workspaces: bool) -> tuple[Path, Path]:
    if distinct_workspaces:
        left_workspace = tmp_path / "a-workspace"
        right_workspace = tmp_path / "z-workspace"
        left_version = right_version = "v001"
    else:
        left_workspace = right_workspace = tmp_path / "workspace"
        left_version, right_version = "v001", "v002"

    for workspace in {left_workspace, right_workspace}:
        config = workspace / ".open-xquant" / "workspace.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(
            "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
            encoding="utf-8",
        )

    left_fixture_root = left_workspace / "versions" / left_version / "09_backtests"
    right_fixture_root = right_workspace / "versions" / right_version / "09_backtests"
    left_fixture_root.mkdir(parents=True)
    right_fixture_root.mkdir(parents=True)
    left_run = _write_minimal_cli_run(left_fixture_root)
    right_run = _write_minimal_cli_run(right_fixture_root)
    synchronized = {
        name: (left_run / name).read_bytes()
        for name in ("compiled_plan.json", "data_manifest.json", "execution_assumptions.json")
    }
    publish_run_artifacts(right_run, synchronized)

    stale_plan = json.loads((right_run / "compiled_plan.json").read_text(encoding="utf-8"))
    stale_plan["execution"]["fill_price_mode"] = "stale-before-publication"
    publish_run_artifacts(
        right_run,
        {"compiled_plan.json": (json.dumps(stale_plan, indent=2) + "\n").encode()},
    )
    return left_run, right_run


def _subprocess_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}


def _start_paused_plan_publisher(
    run_dir: Path,
    source_plan: Path,
    ready: Path,
    release: Path,
) -> subprocess.Popen[str]:
    script = r"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import oxq.run_digests as run_digests

run_dir = Path(sys.argv[1])
source_plan = Path(sys.argv[2])
ready = Path(sys.argv[3])
release = Path(sys.argv[4])
original_hold = run_digests.hold_final_selection_lock

@contextmanager
def paused_hold(lock_path):
    ready.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 15
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("publisher release was not signaled")
        time.sleep(0.01)
    with original_hold(lock_path):
        yield

run_digests.hold_final_selection_lock = paused_hold
run_digests.publish_run_artifacts(
    run_dir,
    {"compiled_plan.json": source_plan.read_bytes()},
)
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(run_dir), str(source_plan), str(ready), str(release)],
        cwd=Path.cwd(),
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _start_observed_compare(
    left_run: Path,
    right_run: Path,
    left_locked: Path,
) -> subprocess.Popen[str]:
    script = r"""
import sys
from pathlib import Path

from click.testing import CliRunner
import oxq.run_digests as run_digests
from oxq.cli.main import main

left_run = Path(sys.argv[1])
right_run = Path(sys.argv[2])
left_locked = Path(sys.argv[3])
left_lock_path = (left_run.parent / "run_digests.jsonl.lock").resolve(strict=False)
OriginalLock = run_digests.ProcessFileLock

class ObservedLock:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=False)
        self.delegate = OriginalLock(path)

    def __enter__(self):
        self.delegate.__enter__()
        if self.path == left_lock_path:
            left_locked.write_text("locked\n", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self.delegate.__exit__(exc_type, exc, traceback)

run_digests.ProcessFileLock = ObservedLock
result = CliRunner().invoke(
    main,
    ["backtest", "compare-runs", str(left_run), str(right_run), "--json"],
)
sys.stdout.write(result.output)
raise SystemExit(result.exit_code)
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(left_run), str(right_run), str(left_locked)],
        cwd=Path.cwd(),
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_process_marker(path: Path, process: subprocess.Popen[str], timeout: float = 8) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"process exited before {path.name}: code={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
            )
        time.sleep(0.01)
    process.kill()
    stdout, stderr = process.communicate()
    raise AssertionError(f"process did not reach {path.name}: stdout={stdout!r}, stderr={stderr!r}")


def _assert_publisher_compare_complete_without_deadlock(tmp_path: Path, *, distinct_workspaces: bool) -> None:
    left_run, right_run = _write_governed_compare_runs(
        tmp_path,
        distinct_workspaces=distinct_workspaces,
    )
    publisher_ready = tmp_path / "publisher-ready"
    release_publisher = tmp_path / "release-publisher"
    left_locked = tmp_path / "compare-left-locked"
    publisher = _start_paused_plan_publisher(
        right_run,
        left_run / "compiled_plan.json",
        publisher_ready,
        release_publisher,
    )
    compare: subprocess.Popen[str] | None = None
    try:
        _wait_for_process_marker(publisher_ready, publisher)
        compare = _start_observed_compare(left_run, right_run, left_locked)
        _wait_for_process_marker(left_locked, compare)
        release_publisher.write_text("release\n", encoding="utf-8")
        publisher_stdout, publisher_stderr = publisher.communicate(timeout=10)
        compare_stdout, compare_stderr = compare.communicate(timeout=10)
    except BaseException:
        publisher.kill()
        publisher.communicate()
        if compare is not None:
            compare.kill()
            compare.communicate()
        raise

    assert publisher.returncode == 0, (publisher_stdout, publisher_stderr)
    assert compare.returncode == 0, (compare_stdout, compare_stderr)
    payload = json.loads(compare_stdout)
    assert payload["comparable"] is True
    assert payload["differences"] == []


def test_backtest_compare_runs_process_avoids_two_version_same_workspace_publisher_deadlock(tmp_path) -> None:
    _assert_publisher_compare_complete_without_deadlock(tmp_path, distinct_workspaces=False)


def test_backtest_compare_runs_process_keeps_distinct_workspace_signatures_coherent(tmp_path) -> None:
    _assert_publisher_compare_complete_without_deadlock(tmp_path, distinct_workspaces=True)


def test_backtest_compare_runs_deduplicates_same_run_path_alias(monkeypatch, tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    alias = tmp_path / "run-alias"
    alias.symlink_to(run_dir, target_is_directory=True)
    observed: list[tuple[Path, ...]] = []
    original_transaction = main_module.multi_run_digest_read_transaction

    @contextmanager
    def observe_transaction(run_dirs):
        observed.append(tuple(Path(path) for path in run_dirs))
        with original_transaction(run_dirs) as resolved:
            assert resolved == (run_dir.resolve(), run_dir.resolve())
            yield resolved

    monkeypatch.setattr(main_module, "multi_run_digest_read_transaction", observe_transaction)

    result = CliRunner().invoke(
        main,
        ["backtest", "compare-runs", str(run_dir), str(alias), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["comparable"] is True
    assert payload["differences"] == []
    assert observed == [(run_dir, alias)]


def test_backtest_compare_runs_releases_all_locks_when_signature_read_fails(monkeypatch, tmp_path) -> None:
    left_run, right_run = _write_governed_compare_runs(tmp_path, distinct_workspaces=False)
    original_signature = main_module._run_comparability_signature_locked

    def fail_right_signature(run_dir: Path) -> dict[str, object]:
        if run_dir == right_run.resolve():
            raise main_module.click.ClickException("injected comparison read failure")
        return original_signature(run_dir)

    monkeypatch.setattr(main_module, "_run_comparability_signature_locked", fail_right_signature)

    result = CliRunner().invoke(
        main,
        ["backtest", "compare-runs", str(left_run), str(right_run), "--json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["errors"][0]["message"] == "injected comparison read failure"
    for run_dir in (left_run, right_run):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "from oxq.run_digests import publish_run_artifacts; "
                    "run = Path(sys.argv[1]); "
                    "publish_run_artifacts(run, {'compiled_plan.json': (run / 'compiled_plan.json').read_bytes()})"
                ),
                str(run_dir),
            ],
            cwd=Path.cwd(),
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        assert completed.returncode == 0, (completed.stdout, completed.stderr)


def test_run_component_manifest_writer_publishes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    from oxq.cli.main import _write_run_component_manifest_artifacts

    run_dir = _write_minimal_cli_run(tmp_path)
    manifest_path = _write_hashed_custom_indicator_extension(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    publications: list[dict[str, bytes]] = []

    def capture_publication(run_path, artifacts, *, replacement_paths, remove_artifacts) -> str:
        assert run_path == run_dir
        assert not (run_dir / "component_manifests.json").exists()
        assert not (run_dir / "component_manifest.json").exists()
        assert not (run_dir / "component_bundle_hash.txt").exists()
        assert not (run_dir / "component_extensions").exists()
        assert not (run_dir / "custom_components").exists()
        publications.append(dict(artifacts))
        assert set(replacement_paths) == {"component_extensions", "custom_components"}
        assert remove_artifacts == set()
        return "sha256:" + "1" * 16

    monkeypatch.setattr("oxq.cli.main.publish_run_artifacts", capture_publication, raising=False)

    _write_run_component_manifest_artifacts(run_dir, [manifest])

    assert set(publications[0]) == {
        "component_manifests.json",
        "component_manifest.json",
        "component_bundle_hash.txt",
    }


@pytest.mark.parametrize("boundary", ["path:component_extensions.replace", "manifest.replace"])
def test_run_component_manifest_writer_rolls_back_interrupted_manifest_publication(
    monkeypatch,
    tmp_path,
    boundary: str,
) -> None:
    from oxq.cli.main import _write_run_component_manifest_artifacts

    run_dir = _write_minimal_cli_run(tmp_path)
    manifest_path = _write_hashed_custom_indicator_extension(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    original_hashes = (run_dir / "artifact_hashes.json").read_bytes()
    original_boundary = run_digests_module._publication_boundary

    def interrupt_manifest(label: str) -> None:
        if label == boundary:
            raise OSError("interrupted component manifest publication")
        original_boundary(label)

    monkeypatch.setattr(run_digests_module, "_publication_boundary", interrupt_manifest)

    with pytest.raises(OSError, match="interrupted component manifest publication"):
        _write_run_component_manifest_artifacts(run_dir, [manifest])

    assert (run_dir / "artifact_hashes.json").read_bytes() == original_hashes
    assert not (run_dir / "component_manifests.json").exists()
    assert not (run_dir / "component_manifest.json").exists()
    assert not (run_dir / "component_bundle_hash.txt").exists()
    assert not (run_dir / "component_extensions").exists()
    assert not (run_dir / "custom_components").exists()
    require_current_run_digest(run_dir)


def test_run_component_manifest_writer_replaces_changed_bundle_without_stale_files(tmp_path) -> None:
    from oxq.cli.main import _write_run_component_manifest_artifacts

    run_dir = _write_minimal_cli_run(tmp_path)
    manifest_path = _write_hashed_custom_indicator_extension(tmp_path)
    source_root = tmp_path / "custom_components"
    stale_file = source_root / "stale_module.py"
    stale_file.write_text("STALE = True\n", encoding="utf-8")
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_manifest["bundle_hash"] = ""
    manifest_path.write_text(json.dumps(first_manifest, indent=2, sort_keys=True), encoding="utf-8")
    first_manifest["bundle_hash"] = compute_component_bundle_hash(manifest_path)
    manifest_path.write_text(json.dumps(first_manifest, indent=2, sort_keys=True), encoding="utf-8")
    first_manifest["_manifest_path"] = str(manifest_path)

    _write_run_component_manifest_artifacts(run_dir, [first_manifest])

    archived_stale = run_dir / "component_extensions/00_custom_components/custom_components/stale_module.py"
    legacy_stale = run_dir / "custom_components/stale_module.py"
    assert archived_stale.exists()
    assert legacy_stale.exists()

    stale_file.unlink()
    source = source_root / "oxq_components/indicators/workspace_constant_indicator.py"
    source.write_text(source.read_text(encoding="utf-8") + "\nUPDATED = True\n", encoding="utf-8")
    second_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    second_manifest["components"][0]["source_hash"] = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    second_manifest["bundle_hash"] = ""
    manifest_path.write_text(json.dumps(second_manifest, indent=2, sort_keys=True), encoding="utf-8")
    second_manifest["bundle_hash"] = compute_component_bundle_hash(manifest_path)
    manifest_path.write_text(json.dumps(second_manifest, indent=2, sort_keys=True), encoding="utf-8")
    second_manifest["_manifest_path"] = str(manifest_path)

    _write_run_component_manifest_artifacts(run_dir, [second_manifest])

    assert not archived_stale.exists()
    assert not legacy_stale.exists()
    require_current_run_digest(run_dir)


def test_run_component_manifest_writer_atomically_switches_inventory_variants(tmp_path) -> None:
    from oxq.cli.main import _write_run_component_manifest_artifacts

    run_dir = _write_minimal_cli_run(tmp_path)
    first_path = _write_hashed_custom_indicator_extension(tmp_path / "first")
    second_path = _write_hashed_custom_indicator_extension(tmp_path / "second")
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    first["_manifest_path"] = str(first_path)
    second["_manifest_path"] = str(second_path)

    _write_run_component_manifest_artifacts(run_dir, [first])
    assert (run_dir / "component_manifest.json").exists()
    assert (run_dir / "component_bundle_hash.txt").exists()

    _write_run_component_manifest_artifacts(run_dir, [first, second])

    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "component_manifests.json" in hashes
    assert "component_manifest.json" not in hashes
    assert "component_bundle_hash.txt" not in hashes
    assert not (run_dir / "component_manifest.json").exists()
    assert not (run_dir / "component_bundle_hash.txt").exists()
    assert {path.name for path in (run_dir / "component_extensions").iterdir()} == {
        "00_custom_components",
        "01_custom_components",
    }
    require_current_run_digest(run_dir)


def test_component_and_provenance_publications_serialize_without_lost_updates(monkeypatch, tmp_path) -> None:
    from oxq.cli.main import _publish_provenance_artifacts, _write_run_component_manifest_artifacts

    run_dir = _write_minimal_cli_run(tmp_path)
    manifest_path = _write_hashed_custom_indicator_extension(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    component_paused = threading.Event()
    allow_component = threading.Event()
    provenance_attempted = threading.Event()
    provenance_completed = threading.Event()
    failures: list[BaseException] = []
    original_boundary = run_digests_module._publication_boundary
    original_transaction = run_digests_module.run_digest_transaction

    def pause_component_manifest(label: str) -> None:
        if threading.current_thread().name == "component-publisher" and label == "manifest.replace":
            component_paused.set()
            assert allow_component.wait(timeout=5)
        original_boundary(label)

    @contextmanager
    def observed_transaction(run_path):
        if threading.current_thread().name == "provenance-publisher":
            provenance_attempted.set()
        with original_transaction(run_path):
            yield

    def publish_components() -> None:
        try:
            _write_run_component_manifest_artifacts(run_dir, [manifest])
        except BaseException as exc:
            failures.append(exc)

    def publish_provenance() -> None:
        try:
            _publish_provenance_artifacts(
                run_dir,
                spec_audit_content=b'{"status":"pass"}',
                runtime_audit_content=b'{"status":"pass"}',
                conversation_hash="sha256:" + "a" * 64,
                catalog_hash="sha256:" + "b" * 64,
                recipe_catalog_hash="sha256:" + "c" * 64,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            provenance_completed.set()

    monkeypatch.setattr(run_digests_module, "_publication_boundary", pause_component_manifest)
    monkeypatch.setattr(run_digests_module, "run_digest_transaction", observed_transaction)
    component_thread = threading.Thread(target=publish_components, name="component-publisher")
    provenance_thread = threading.Thread(target=publish_provenance, name="provenance-publisher")
    component_thread.start()
    try:
        assert component_paused.wait(timeout=5)
        provenance_thread.start()
        assert provenance_attempted.wait(timeout=5)
        assert not provenance_completed.is_set()
    finally:
        allow_component.set()
    component_thread.join(timeout=5)
    provenance_thread.join(timeout=5)

    assert not component_thread.is_alive()
    assert not provenance_thread.is_alive()
    assert failures == []
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert {
        "component_manifests.json",
        "component_manifest.json",
        "component_bundle_hash.txt",
        "spec_audit.json",
        "conversation_hash.txt",
        "component_catalog_hash.txt",
        "recipe_catalog_hash.txt",
    } <= hashes.keys()
    require_current_run_digest(run_dir)


def test_strategy_compile_writes_compile_preview(monkeypatch, tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="compile_preview", hypothesis="compile preview should be auditable")
    spec.data.data_dir = "data"
    spec.portfolio.rules["rebalance"] = {
        "type": "RebalanceFrequencyRule",
        "params": {"interval_days": 10},
    }
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "compile_preview"
    monkeypatch.chdir(tmp_path)

    explicit_data_dir = tmp_path / "formal_data"
    result = CliRunner().invoke(
        main,
        [
            "strategy",
            "compile",
            "strategy_spec.yaml",
            "--data-dir",
            str(explicit_data_dir),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    compiled_plan = json.loads((out_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    assert compiled_plan["open_xquant_version"]
    assert compiled_plan["execution"]["rebalance"]["interval_days"] == 10
    assert compiled_plan["execution"]["rebalance"]["source"] == "portfolio.rules.rebalance"
    assert compiled_plan["data"]["spec_data_dir"] == "data"
    assert compiled_plan["data"]["effective_data_dir"] == str(explicit_data_dir.resolve())
    assert (out_dir / "spec_hash.txt").read_text(encoding="utf-8").strip() == compiled_plan["spec_hash"]
    assert "Effective data dir:" in result.output
    assert "included in compiled_plan.json and its hash" in result.output


@pytest.mark.parametrize("out_value", [".", ".."], ids=["cwd", "ancestor"])
def test_strategy_compile_preview_rejects_cwd_and_ancestors_without_deleting_sentinel(
    monkeypatch,
    tmp_path,
    out_value: str,
) -> None:
    work_dir = tmp_path / "work"
    cwd = work_dir / "nested"
    cwd.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="compile_preview_unsafe_out",
        hypothesis="compile preview must not replace its working directory",
    )
    spec_path = cwd / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    protected_dir = cwd if out_value == "." else work_dir
    sentinel = protected_dir / "sentinel.txt"
    sentinel.write_text("keep me\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    result = CliRunner().invoke(
        main,
        ["strategy", "compile", "strategy_spec.yaml", "--out", out_value],
    )

    assert result.exit_code == 1
    assert "current working directory or one of its ancestors" in result.output
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_strategy_compile_preview_rejects_unowned_nonempty_directory_without_deleting_sentinel(
    monkeypatch,
    tmp_path,
) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_unowned",
        hypothesis="compile preview must preserve unowned directories",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "compile_preview"
    out_dir.mkdir()
    sentinel = out_dir / "sentinel.txt"
    sentinel.write_text("keep me\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["strategy", "compile", "strategy_spec.yaml", "--out", str(out_dir)],
    )

    assert result.exit_code == 1
    assert "not an open-xquant-managed compile preview" in result.output
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_strategy_compile_preview_adopts_empty_directory_then_replaces_managed_preview(
    monkeypatch,
    tmp_path,
) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_managed_transition",
        hypothesis="only managed compile previews may be replaced",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "compile_preview"
    out_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    first = CliRunner().invoke(
        main,
        ["strategy", "compile", "strategy_spec.yaml", "--out", str(out_dir)],
    )

    assert first.exit_code == 0, first.output
    marker = json.loads((out_dir / ".oxq-compile-preview.json").read_text(encoding="utf-8"))
    assert marker == {
        "artifact": "strategy-compile-preview",
        "managed_by": "open-xquant",
        "schema_version": 1,
    }
    stale = out_dir / "stale.txt"
    stale.write_text("remove me\n", encoding="utf-8")

    second = CliRunner().invoke(
        main,
        ["strategy", "compile", "strategy_spec.yaml", "--out", str(out_dir)],
    )

    assert second.exit_code == 0, second.output
    assert not stale.exists()
    assert (out_dir / ".oxq-compile-preview.json").is_file()


def test_strategy_compile_preview_rejects_symlink_target_without_mutating_target(
    monkeypatch,
    tmp_path,
) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_symlink",
        hypothesis="compile preview targets must not be symlinks",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("keep me\n", encoding="utf-8")
    out_dir = tmp_path / "compile_preview"
    out_dir.symlink_to(external, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["strategy", "compile", "strategy_spec.yaml", "--out", str(out_dir)],
    )

    assert result.exit_code == 1
    assert "must not contain symlink components" in result.output
    assert out_dir.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_strategy_compile_preview_rejects_parent_replacement_before_target_mutation(
    monkeypatch,
    tmp_path,
) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_parent_change",
        hypothesis="compile preview publication pins its output parent",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    parent = tmp_path / "preview_parent"
    parent.mkdir()
    out_dir = parent / "compile_preview"
    sentinel = out_dir / "sentinel.txt"
    monkeypatch.chdir(tmp_path)
    first = CliRunner().invoke(
        main,
        ["strategy", "compile", "strategy_spec.yaml", "--out", str(out_dir)],
    )
    assert first.exit_code == 0, first.output
    sentinel.write_text("keep me\n", encoding="utf-8")
    displaced_parent = tmp_path / "displaced_preview_parent"
    original_replace = main_module._replace_compile_preview

    def replace_after_parent_swap(target, staging_root, **kwargs):
        parent.rename(displaced_parent)
        parent.mkdir()
        try:
            return original_replace(
                target,
                staging_root,
                **kwargs,
            )
        finally:
            shutil.rmtree(parent)
            displaced_parent.rename(parent)

    monkeypatch.setattr(main_module, "_replace_compile_preview", replace_after_parent_swap)

    result = CliRunner().invoke(
        main,
        ["strategy", "compile", "strategy_spec.yaml", "--out", str(out_dir)],
    )

    assert result.exit_code == 1
    assert "output parent changed during publication" in result.output
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"


def test_strategy_compile_preview_writes_strategy_py_for_user_review(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_strategy_py",
        hypothesis="compile preview should produce source code for user review",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "compile_preview"
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "strategy",
            "compile",
            "strategy_spec.yaml",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    strategy_py = (out_dir / "strategy.py").read_text(encoding="utf-8")
    assert (out_dir / "strategy_spec.yaml").read_text(encoding="utf-8") == spec_path.read_text(encoding="utf-8")
    assert "def define_strategy() -> StrategySpec:" in strategy_py
    assert "def run_backtest(" in strategy_py
    assert "def main(dry_run: bool = False)" in strategy_py
    assert "Python source preview:" in result.output
    module_spec = importlib.util.spec_from_file_location("compiled_strategy_preview", out_dir / "strategy.py")
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    preview = module.main(dry_run=True)
    assert preview["status"] == "dry_run"
    assert preview["strategy"]["strategy_id"] == "compile_preview_strategy_py"


def test_strategy_compile_preview_archives_workspace_component_manifest(tmp_path, monkeypatch) -> None:
    manifest = _write_hashed_custom_indicator_extension(tmp_path)
    spec = StrategySpec.template(
        strategy_id="compile_preview_custom_component",
        hypothesis="compile preview should carry custom component source",
    )
    spec.signal.indicators = {
        "custom_score": {
            "type": "WorkspaceConstantIndicator",
            "params": {"value": 2.0},
        }
    }
    spec.portfolio.type = "TopNRanking"
    spec.portfolio.params = {"score_col": "custom_score", "n": 1}
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "compile_preview"
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "strategy",
            "compile",
            "strategy_spec.yaml",
            "--component-manifest",
            str(manifest),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "component_manifests.json").exists()
    assert list(
        out_dir.glob("component_extensions/*/custom_components/oxq_components/indicators/workspace_constant_indicator.py")
    )
    shutil.rmtree(tmp_path / "custom_components")
    manifest.unlink()

    module_spec = importlib.util.spec_from_file_location("compiled_strategy_preview_custom", out_dir / "strategy.py")
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    preview = module.main(dry_run=True)
    assert preview["status"] == "dry_run"
    assert preview["indicators"]["custom_score"]["type"] == "WorkspaceConstantIndicator"


def test_strategy_compile_preview_replaces_one_many_one_and_absent_component_states(
    tmp_path,
    monkeypatch,
) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_replacement",
        hypothesis="compile preview is a complete replacement",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    alpha = _write_hashed_preview_indicator_extension(
        tmp_path / "alpha_workspace",
        root_name="alpha_components",
        indicator_name="AlphaPreviewIndicator",
        marker="alpha",
    )
    beta = _write_hashed_preview_indicator_extension(
        tmp_path / "beta_workspace",
        root_name="beta_components",
        indicator_name="BetaPreviewIndicator",
        marker="beta",
    )
    gamma = _write_hashed_preview_indicator_extension(
        tmp_path / "gamma_workspace",
        root_name="gamma_components",
        indicator_name="GammaPreviewIndicator",
        marker="gamma",
    )
    out_dir = tmp_path / "compile_preview"
    monkeypatch.chdir(tmp_path)

    first = _compile_preview(spec_path, out_dir, alpha)
    assert first.exit_code == 0, first.output
    (out_dir / "stale-preview-file.txt").write_text("stale", encoding="utf-8")

    one_to_many = _compile_preview(spec_path, out_dir, beta, gamma)

    assert one_to_many.exit_code == 0, one_to_many.output
    assert not (out_dir / "alpha_components").exists()
    assert not (out_dir / "component_manifest.json").exists()
    assert not (out_dir / "component_bundle_hash.txt").exists()
    assert not (out_dir / "stale-preview-file.txt").exists()
    summary = json.loads((out_dir / "component_manifests.json").read_text(encoding="utf-8"))
    assert [item["extension_id"] for item in summary] == ["beta_components", "gamma_components"]
    assert len(list((out_dir / "component_extensions").iterdir())) == 2

    many_to_one = _compile_preview(spec_path, out_dir, alpha)

    assert many_to_one.exit_code == 0, many_to_one.output
    assert (out_dir / "alpha_components").is_dir()
    assert (out_dir / "component_manifest.json").is_file()
    assert (out_dir / "component_bundle_hash.txt").is_file()
    assert len(list((out_dir / "component_extensions").iterdir())) == 1
    assert not list(out_dir.rglob("*beta_components*"))
    assert not list(out_dir.rglob("*gamma_components*"))

    absent = _compile_preview(spec_path, out_dir)

    assert absent.exit_code == 0, absent.output
    assert {path.name for path in out_dir.iterdir()} == {
        ".oxq-compile-preview.json",
        "compiled_plan.json",
        "spec_hash.txt",
        "strategy.py",
        "strategy_spec.yaml",
    }


def test_strategy_compile_preview_replaces_changed_same_root_without_stale_files(
    tmp_path,
    monkeypatch,
) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_same_root",
        hypothesis="same-root changes replace the old tree",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    manifest = _write_hashed_preview_indicator_extension(
        tmp_path / "component_workspace",
        root_name="shared_components",
        indicator_name="SharedPreviewIndicator",
        marker="old-generation",
    )
    source_root = manifest.parent / "shared_components"
    obsolete = source_root / "obsolete.py"
    obsolete.write_text("GENERATION = 'obsolete'\n", encoding="utf-8")
    _rehash_preview_indicator_extension(manifest)
    out_dir = tmp_path / "compile_preview"
    monkeypatch.chdir(tmp_path)
    first = _compile_preview(spec_path, out_dir, manifest)
    assert first.exit_code == 0, first.output
    assert list(out_dir.rglob("obsolete.py"))

    source = source_root / "shared_components_indicator.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace("old-generation", "new-generation"),
        encoding="utf-8",
    )
    obsolete.unlink()
    _rehash_preview_indicator_extension(manifest)

    second = _compile_preview(spec_path, out_dir, manifest)

    assert second.exit_code == 0, second.output
    assert not list(out_dir.rglob("obsolete.py"))
    published_sources = list(out_dir.rglob("shared_components_indicator.py"))
    assert published_sources
    assert all("new-generation" in path.read_text(encoding="utf-8") for path in published_sources)


def test_strategy_compile_preview_failed_publish_preserves_previous_state_and_retry_replaces_it(
    tmp_path,
    monkeypatch,
) -> None:
    spec = StrategySpec.template(
        strategy_id="compile_preview_retry",
        hypothesis="failed preview publication can be retried",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    manifest = _write_hashed_preview_indicator_extension(
        tmp_path / "component_workspace",
        root_name="retry_components",
        indicator_name="RetryPreviewIndicator",
        marker="old-generation",
    )
    source = manifest.parent / "retry_components/retry_components_indicator.py"
    out_dir = tmp_path / "compile_preview"
    monkeypatch.chdir(tmp_path)
    initial = _compile_preview(spec_path, out_dir, manifest)
    assert initial.exit_code == 0, initial.output
    original_snapshot = _directory_file_snapshot(out_dir)
    source.write_text(
        source.read_text(encoding="utf-8").replace("old-generation", "new-generation"),
        encoding="utf-8",
    )
    _rehash_preview_indicator_extension(manifest)
    original_write_bytes = Path.write_bytes

    def fail_summary_write(path: Path, content: bytes) -> int:
        if path.name == "component_manifests.json":
            raise OSError("injected preview publication failure")
        return original_write_bytes(path, content)

    with monkeypatch.context() as failure_patch:
        failure_patch.setattr(Path, "write_bytes", fail_summary_write)
        failed = _compile_preview(spec_path, out_dir, manifest)

    assert failed.exit_code == 1
    assert "injected preview publication failure" in str(failed.exception)
    assert _directory_file_snapshot(out_dir) == original_snapshot

    retried = _compile_preview(spec_path, out_dir, manifest)

    assert retried.exit_code == 0, retried.output
    assert _directory_file_snapshot(out_dir) != original_snapshot
    assert all(
        "new-generation" in path.read_text(encoding="utf-8")
        for path in out_dir.rglob("retry_components_indicator.py")
    )


def test_backtest_default_out_uses_version_governed_workspace_default(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  current_manifest: current.json",
                "workflow:",
                "  layout: version_governed",
                "  default_output_dir: versions/{active_version}/09_backtests",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v001", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "versions" / "v001" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="version_default_out",
        hypothesis="workspace default output directory should be version-governed",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_compile_run(spec, data_dir=None, out_dir=None):
        captured["out_dir"] = out_dir
        run_dir = Path(str(out_dir)) / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return object(), run_dir

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["out_dir"] == "versions/v001/09_backtests"
    payload = json.loads(result.output)
    assert payload["run_dir"] == "versions/v001/09_backtests/run_001"


def test_backtest_default_out_uses_versions_dir_workspace_default(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions",
                "  current_manifest: current.json",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v002", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "versions" / "v002" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="versions_dir_default_out",
        hypothesis="versions_dir-only workspace should still use governed backtest output",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_compile_run(spec, data_dir=None, out_dir=None):
        captured["out_dir"] = out_dir
        run_dir = Path(str(out_dir)) / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return object(), run_dir

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["out_dir"] == "versions/v002/09_backtests"


def test_backtest_default_out_honors_custom_versions_dir_workspace(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: research_versions",
                "  current_manifest: current.json",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v003", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "research_versions" / "v003" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="custom_versions_dir_default_out",
        hypothesis="versions_dir-only workspace should use configured versions root",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_compile_run(spec, data_dir=None, out_dir=None):
        captured["out_dir"] = out_dir
        run_dir = Path(str(out_dir)) / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return object(), run_dir

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["out_dir"] == "research_versions/v003/09_backtests"


@pytest.mark.parametrize("requested_out", ["../escape", "research_versions/v001/09_backtests"])
def test_backtest_rejects_output_outside_active_governed_phase(
    monkeypatch,
    tmp_path,
    requested_out: str,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    config_dir = workspace / ".open-xquant"
    version_dir = workspace / "research_versions/v003"
    config_dir.mkdir(parents=True)
    version_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "paths": {"versions_dir": "research_versions"},
                "workflow": {"layout": "version_governed"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(
        json.dumps({"active_version": "v003", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v003",
                "phase_paths": {"09_backtests": "research_versions/v003/09_backtests"},
            }
        ),
        encoding="utf-8",
    )
    spec = StrategySpec.template(strategy_id="reject_outside", hypothesis="governed output only")
    spec_path = workspace / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    called = False

    def fake_compile_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("compile_run must not be called")

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    entry_cwd = Path.cwd()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with monkeypatch.context() as cwd_patch:
            cwd_patch.chdir(workspace)
            result = runner.invoke(
                main,
                ["backtest", "run", str(spec_path), "--allow-unaudited", "--out", requested_out, "--json"],
            )

    assert Path.cwd() == entry_cwd
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "output_dir_failed"
    assert called is False
    assert not (workspace.parent / "escape").exists()


def test_backtest_rejects_configured_output_outside_active_governed_phase(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / ".open-xquant"
    version_dir = workspace / "versions/v002"
    config_dir.mkdir(parents=True)
    version_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "paths": {"versions_dir": "versions"},
                "workflow": {
                    "layout": "version_governed",
                    "default_output_dir": "unrelated/backtests",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(json.dumps({"active_version": "v002"}), encoding="utf-8")
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v002",
                "phase_paths": {"09_backtests": "versions/v002/09_backtests"},
            }
        ),
        encoding="utf-8",
    )
    spec = StrategySpec.template(strategy_id="reject_config", hypothesis="governed output only")
    spec_path = workspace / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.output)["errors"][0]["check"] == "output_dir_failed"
    assert not (workspace / "unrelated").exists()


def test_backtest_default_uses_active_version_manifest_backtest_phase(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / ".open-xquant"
    version_dir = workspace / "research_versions/v004"
    config_dir.mkdir(parents=True)
    version_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "paths": {"versions_dir": "research_versions"},
                "workflow": {"layout": "version_governed"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(json.dumps({"active_version": "v004"}), encoding="utf-8")
    governed_output = "research_versions/v004/09_backtests/formal"
    (version_dir / "version_manifest.json").write_text(
        json.dumps({"version_id": "v004", "phase_paths": {"09_backtests": governed_output}}),
        encoding="utf-8",
    )
    spec = StrategySpec.template(strategy_id="manifest_output", hypothesis="manifest path is authoritative")
    spec_path = workspace / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_compile_run(spec, data_dir=None, out_dir=None):
        captured["out_dir"] = out_dir
        run_dir = Path(str(out_dir)) / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return object(), run_dir

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(main, ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"])

    assert result.exit_code == 0, result.output
    assert captured["out_dir"] == governed_output


def test_backtest_default_out_rewrites_legacy_default_for_custom_versions_dir(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: research_versions",
                "  current_manifest: current.json",
                "workflow:",
                "  layout: version_governed",
                "  default_output_dir: versions/{active_version}/09_backtests",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v004", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "research_versions" / "v004" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="legacy_default_custom_versions_dir",
        hypothesis="legacy default_output_dir should follow configured versions root",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_compile_run(spec, data_dir=None, out_dir=None):
        captured["out_dir"] = out_dir
        run_dir = Path(str(out_dir)) / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return object(), run_dir

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["out_dir"] == "research_versions/v004/09_backtests"
    assert not (tmp_path / "versions/v004/09_backtests").exists()


def test_backtest_default_out_rewrites_stale_runs_auto_for_custom_versions_dir(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: research_versions",
                "  current_manifest: current.json",
                "workflow:",
                "  layout: version_governed",
                "  default_output_dir: runs/auto",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v005", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "research_versions" / "v005" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="stale_runs_auto_custom_versions_dir",
        hypothesis="stale runs/auto default_output_dir should follow configured versions root",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_compile_run(spec, data_dir=None, out_dir=None):
        captured["out_dir"] = out_dir
        run_dir = Path(str(out_dir)) / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return object(), run_dir

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["out_dir"] == "research_versions/v005/09_backtests"
    assert not (tmp_path / "runs" / "auto").exists()


def test_backtest_default_out_rewrites_nested_stale_runs_auto_for_custom_versions_dir(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: research_versions",
                "  current_manifest: current.json",
                "workflow:",
                "  layout: version_governed",
                "  default_output_dir: runs/auto/runs/runs/{active_version}",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v006", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "research_versions" / "v006" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="nested_stale_runs_auto_custom_versions_dir",
        hypothesis="nested stale runs/auto default_output_dir should follow configured versions root",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_compile_run(spec, data_dir=None, out_dir=None):
        captured["out_dir"] = out_dir
        run_dir = Path(str(out_dir)) / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return object(), run_dir

    monkeypatch.setattr("oxq.spec.compiler.compile_run", fake_compile_run)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["out_dir"] == "research_versions/v006/09_backtests"
    assert not (tmp_path / "runs" / "auto" / "runs" / "runs").exists()


def test_backtest_default_out_rejects_symlinked_versions_dir_escape(monkeypatch, tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".open-xquant").mkdir()
    (workspace / "versions_link").symlink_to(outside, target_is_directory=True)
    (workspace / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  versions_dir: versions_link",
                "  current_manifest: current.json",
                "workflow:",
                "  layout: version_governed",
            ]
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v012", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec = StrategySpec.template(
        strategy_id="symlink_versions_dir_default_out",
        hypothesis="versions_dir symlinks must not escape workspace",
    )
    spec_path = workspace / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "output_dir_failed"
    assert "paths.versions_dir must not contain symlink components" in payload["errors"][0]["message"]
    assert not (outside / "v012").exists()


def test_backtest_legacy_fallback_rejects_symlinked_backtest_phase_before_mutation(monkeypatch, tmp_path) -> None:
    outside = tmp_path / "outside"
    version_dir = tmp_path / "versions/v013"
    (tmp_path / ".open-xquant").mkdir()
    version_dir.mkdir(parents=True)
    outside.mkdir()
    (version_dir / "09_backtests").symlink_to(outside, target_is_directory=True)
    (tmp_path / ".open-xquant/workspace.yaml").write_text(
        "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(json.dumps({"active_version": "v013"}), encoding="utf-8")
    spec = StrategySpec.template(strategy_id="legacy_symlink_out", hypothesis="reject symlinked fallback")
    spec_path = version_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "output_dir_failed"
    assert "09_backtests" in payload["errors"][0]["message"]
    assert "symlink" in payload["errors"][0]["message"]
    assert list(outside.iterdir()) == []


def test_backtest_default_out_rejects_unsafe_active_version(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  current_manifest: current.json",
                "workflow:",
                "  layout: version_governed",
                "  default_output_dir: versions/{active_version}/09_backtests",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "../escape", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "versions" / "v001" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="unsafe_active_version",
        hypothesis="workspace active version must be path safe",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "output_dir_failed"
    assert "active_version is unsafe" in payload["errors"][0]["message"]


def test_backtest_default_out_rejects_hidden_current_manifest_path(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "paths:",
                "  current_manifest: .open-xquant/current.json",
                "workflow:",
                "  layout: version_governed",
                "  default_output_dir: versions/{active_version}/09_backtests",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".open-xquant" / "current.json").write_text(
        json.dumps({"schema_version": 1, "active_version": "v001", "active_phase": "09_backtests"}),
        encoding="utf-8",
    )
    spec_dir = tmp_path / "versions" / "v001" / "04_spec_build"
    spec_dir.mkdir(parents=True)
    spec = StrategySpec.template(
        strategy_id="hidden_current_manifest",
        hypothesis="workspace current manifest must be root-local",
    )
    spec_path = spec_dir / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "output_dir_failed"
    assert "paths.current_manifest must be current.json" in payload["errors"][0]["message"]


def test_backtest_default_out_rejects_malformed_workspace_yaml(monkeypatch, tmp_path) -> None:
    (tmp_path / ".open-xquant").mkdir()
    (tmp_path / ".open-xquant" / "workspace.yaml").write_text("workflow: [\n", encoding="utf-8")
    spec = StrategySpec.template(
        strategy_id="malformed_workspace",
        hypothesis="workspace config parse errors should not silently use runs auto",
    )
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["backtest", "run", str(spec_path), "--allow-unaudited", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["errors"][0]["check"] == "output_dir_failed"
    assert "workspace config is invalid" in payload["errors"][0]["message"]


@pytest.mark.parametrize(
    ("command", "audit_name", "artifact_name"),
    [
        ("reproducibility", "audit_reproducibility", "reproducibility_audit.json"),
        ("research", "audit_research", "research_bias_audit.json"),
    ],
)
def test_audit_publish_atomically_binds_result_without_changing_json_stdout(
    monkeypatch,
    tmp_path,
    command: str,
    audit_name: str,
    artifact_name: str,
) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    audit_result = {
        "status": "pass",
        "checks": [{"id": command, "status": "pass", "severity": "info", "message": "ok"}],
        "fatal_count": 0,
        "warning_count": 0,
    }
    monkeypatch.setattr(f"oxq.audit.{audit_name}", lambda _run_dir: audit_result)

    result = CliRunner().invoke(
        main,
        ["audit", command, str(run_dir), "--json", "--publish"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == audit_result
    assert json.loads((run_dir / artifact_name).read_text(encoding="utf-8")) == audit_result
    artifact_hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert artifact_name in artifact_hashes
    require_current_run_digest(run_dir)


def test_audit_publish_does_not_truncate_existing_artifact_before_audit_succeeds(monkeypatch, tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    artifact_name = "reproducibility_audit.json"
    original = b'{"status":"pass","generation":"original"}\n'
    publish_run_artifacts(run_dir, {artifact_name: original})

    def fail_audit(_run_dir):
        raise RuntimeError("audit validation failed")

    monkeypatch.setattr("oxq.audit.audit_reproducibility", fail_audit)

    result = CliRunner().invoke(
        main,
        ["audit", "reproducibility", str(run_dir), "--json", "--publish"],
    )

    assert result.exit_code == 1
    assert (run_dir / artifact_name).read_bytes() == original
    require_current_run_digest(run_dir)


def test_spec_audit_validate_accepts_required_schema(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "block",
        "audit_conclusion": "blocked",
        "user_confirmation_status": "pending",
        "spec_confirmation_table": None,
        "spec_provenance_pass": False,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [
            {
                "recipe": "sma_golden_cross",
                "status": "not_applicable",
                "evidence": [],
                "canonical": False,
            }
        ],
        "field_audits": [
                {
                    "field_path": "portfolio.type",
                    "spec_value": "EqualWeight",
                    "status": "confirmed",
                    "material_category": "portfolio_construction",
                    "evidence": [],
                    "blocking": False,
                }
        ],
        "component_audits": [
            {
                "component_path": "portfolio.type",
                "component_type": "EqualWeight",
                "status": "catalog",
                "evidence": [],
                "blocking": False,
            }
        ],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [{"message": "confirm cost assumptions"}],
    }
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_spec_audit_validate_accepts_blocked_audit_without_confirmation_table(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "block",
        "audit_conclusion": "blocked",
        "user_confirmation_status": "pending",
        "spec_provenance_pass": False,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [
                {
                    "field_path": "execution.initial_cash",
                    "spec_value": 100000,
                    "status": "contradiction",
                    "material_category": "execution_assumption",
                    "evidence": ["Builder mapped confirmed initial cash to a non-operative YAML field."],
                    "blocking": True,
                }
        ],
        "component_audits": [],
        "unsupported_mappings": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [
            {
                "field_path": "execution.initial_cash",
                "message": "effective StrategySpec value differs from user-confirmed source value",
            }
        ],
        "blocking_findings": [{"message": "return to build; do not show a placeholder confirmation table"}],
    }
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_spec_audit_validate_rejects_missing_required_fields(tmp_path) -> None:
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "spec_hash" for error in payload["errors"])
    assert any(error["path"] == "schema_version" for error in payload["errors"])
    assert any(error["path"] == "spec_provenance_pass" for error in payload["errors"])


def test_spec_audit_validate_rejects_legacy_v1_after_gate_breaking_change(tmp_path) -> None:
    audit = {
        "schema_version": 1,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "schema_version" and "must be 4" in error["message"] for error in payload["errors"])


def test_spec_audit_validate_rejects_malformed_entries(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "blocked",
        "audit_conclusion": "blocked",
        "user_confirmation_status": "pending",
        "spec_confirmation_table": None,
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": ["volatility_adjusted_momentum"],
        "field_audits": [{"field_path": "portfolio.type", "status": "ok", "evidence": []}],
        "component_audits": [{"component_path": "portfolio.type", "component_type": "EqualWeight", "status": "unknown"}],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "status" for error in payload["errors"])
    assert any(error["path"] == "recipe_matches[0]" for error in payload["errors"])
    assert any(error["path"] == "field_audits[0].status" for error in payload["errors"])
    assert any(error["path"] == "field_audits[0].spec_value" for error in payload["errors"])
    assert any(error["path"] == "component_audits[0].status" for error in payload["errors"])


def test_spec_audit_validate_rejects_confirmed_when_evidence_denies_user_confirmation(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
                "evidence": ["用户只给了回测时间，未指定训练/测试期划分"],
                "blocking": False,
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    _attach_confirmation_artifacts(tmp_path, audit)
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "field_audits[0].status" for error in payload["errors"])


def test_spec_audit_validate_rejects_confirmed_when_same_evidence_denies_field_confirmation(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
                "evidence": ["用户确认了完整回测区间，但未指定训练/测试期划分"],
                "blocking": False,
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    _attach_confirmation_artifacts(tmp_path, audit)
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "field_audits[0].status" for error in payload["errors"])


def test_spec_audit_validate_rejects_confirmation_for_other_field_before_denial(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
                "material_category": "validation_assumption",
                "evidence": ["User confirmed the full backtest range in turn 5, but did not specify the train/test split."],
                "blocking": False,
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    _attach_confirmation_artifacts(tmp_path, audit)
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "field_audits[0].status" for error in payload["errors"])


def test_spec_audit_validate_allows_confirmed_after_later_confirmation(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
                "material_category": "validation_assumption",
                "evidence": ["The split was initially not specified; user confirmed it in turn 5."],
                "blocking": False,
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    _attach_confirmation_artifacts(tmp_path, audit)
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_spec_audit_validate_allows_later_confirmation_in_separate_evidence_entry(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
                "material_category": "validation_assumption",
                "evidence": ["initially not specified", "user confirmed it in turn 5"],
                "blocking": False,
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    _attach_confirmation_artifacts(tmp_path, audit)
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_spec_audit_validate_allows_confirmation_before_historical_negative_context(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
                "material_category": "validation_assumption",
                "evidence": ["User confirmed in turn 5 after it was initially not specified."],
                "blocking": False,
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    _attach_confirmation_artifacts(tmp_path, audit)
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output


def test_spec_audit_validate_rejects_pass_with_false_provenance_gate(tmp_path) -> None:
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
            "hash": "sha256:" + "4" * 16,
            "hash_type": "sha256",
        },
        "confirmation_event": _confirmation_event(),
        "spec_provenance_pass": False,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": [],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "spec_provenance_pass" for error in payload["errors"])


def test_spec_audit_validate_strict_confirmed_requires_spec(tmp_path) -> None:
    path = tmp_path / "spec_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "pass",
                "audit_conclusion": "all_pass",
                "user_confirmation_status": "confirmed",
                "spec_confirmation_table": {
                    "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
                    "hash": "sha256:" + "4" * 16,
                    "hash_type": "sha256",
                },
                "confirmation_event": _confirmation_event(),
                "spec_provenance_pass": True,
                "spec_hash": "sha256:" + "1" * 16,
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                **_spec_audit_context(),
                "recipe_matches": [],
                "field_audits": [],
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--strict-confirmed", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "spec" for error in payload["errors"])


def test_spec_audit_validate_checks_mapping_contract(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "block",
                "audit_conclusion": "blocked",
                "user_confirmation_status": "pending",
                "spec_confirmation_table": None,
                "spec_provenance_pass": False,
                "spec_hash": "sha256:" + "1" * 16,
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                **_spec_audit_context(),
                "recipe_matches": [],
                "field_audits": [],
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [{"message": "builder contract is invalid"}],
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "spec_mapping_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_format": "ebacktestcraft_yaml",
                "field_mappings": [
                    {
                        "source_field": "rebalance.day",
                        "target_field": "",
                        "semantic": "strategy",
                        "status": "unsupported",
                        "confirmation_required": False,
                        "blocking": False,
                        "reason": "Calendar-aware week-end rebalance is not executable.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["spec-audit", "validate", str(audit_path), "--mapping-contract", str(contract_path), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "mapping_contract.field_mappings[0].blocking" for error in payload["errors"])


def test_spec_audit_validate_binds_mapping_contract_to_supplied_spec(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        root = tmp_path / cwd
        spec = StrategySpec.template(strategy_id="equal_weight", hypothesis="actual mapping fields")
        spec_path = root / "strategy_spec.yaml"
        spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
        audit_path = root / "spec_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "status": "block",
                    "audit_conclusion": "blocked",
                    "user_confirmation_status": "pending",
                    "spec_confirmation_table": None,
                    "spec_provenance_pass": False,
                    "spec_hash": "sha256:" + "1" * 16,
                    "conversation_hash": "sha256:" + "2" * 16,
                    "catalog_hash": "sha256:" + "3" * 16,
                    **_spec_audit_context(),
                    "recipe_matches": [],
                    "field_audits": [],
                    "component_audits": [],
                    "missing_user_requirements": [],
                    "agent_added_fields": [],
                    "contradictions": [],
                    "blocking_findings": [{"message": "mapping is invalid"}],
                }
            ),
            encoding="utf-8",
        )
        contract_path = root / "spec_mapping_contract.json"
        contract_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_format": "external_strategy",
                    "field_mappings": [
                        {
                            "source_field": "portfolio.nonexistent",
                            "target_field": "portfolio.params.nonexistent",
                            "semantic": "strategy",
                            "status": "mapped",
                            "confirmation_required": False,
                            "blocking": False,
                            "reason": "Must exist in this EqualWeight candidate.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            main,
            [
                "spec-audit",
                "validate",
                str(audit_path),
                "--spec",
                str(spec_path),
                "--mapping-contract",
                str(contract_path),
                "--json",
            ],
        )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "mapping_contract.field_mappings[0].target_field" for error in payload["errors"])


def test_spec_audit_validate_rejects_pass_with_blocked_strategy_mapping(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "pass",
                "audit_conclusion": "all_pass",
                "user_confirmation_status": "confirmed",
                "spec_confirmation_table": {
                    "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
                    "hash": "sha256:" + "4" * 16,
                    "hash_type": "sha256",
                },
                "confirmation_event": _confirmation_event(),
                "spec_provenance_pass": True,
                "spec_hash": "sha256:" + "1" * 16,
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                **_spec_audit_context(),
                "recipe_matches": [],
                "field_audits": [],
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [],
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "spec_mapping_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_format": "ebacktestcraft_yaml",
                "field_mappings": [
                    {
                        "source_field": "market.cross_calendar_policy",
                        "target_field": "",
                        "semantic": "strategy",
                        "status": "blocked",
                        "confirmation_required": False,
                        "blocking": True,
                        "reason": "Mixed-calendar execution still needs runtime support confirmation.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["spec-audit", "validate", str(audit_path), "--mapping-contract", str(contract_path), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(
        error["path"] == "mapping_contract.builder_pass.field_mappings[0].status"
        and "builder pass requires mappings to be mapped or excluded_non_material and non-blocking"
        in error["message"]
        for error in payload["errors"]
    )


def test_spec_audit_validate_accepts_blocked_audit_with_blocking_mapping(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "block",
                "audit_conclusion": "blocked",
                "user_confirmation_status": "pending",
                "spec_confirmation_table": None,
                "spec_provenance_pass": False,
                "spec_hash": "sha256:" + "1" * 16,
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                **_spec_audit_context(),
                "recipe_matches": [],
                "field_audits": [],
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [{"message": "The source mapping is not executable."}],
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "spec_mapping_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_format": "external_strategy",
                "field_mappings": [
                    {
                        "source_field": "market.cross_calendar_policy",
                        "target_field": "",
                        "semantic": "strategy",
                        "status": "blocked",
                        "confirmation_required": False,
                        "blocking": True,
                        "reason": "The runtime cannot preserve the source calendar policy.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["spec-audit", "validate", str(audit_path), "--mapping-contract", str(contract_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["errors"] == []


def test_spec_audit_validate_checks_strategy_confirmation_blocking(tmp_path) -> None:
    audit_path = tmp_path / "spec_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "block",
                "audit_conclusion": "blocked",
                "user_confirmation_status": "pending",
                "spec_confirmation_table": None,
                "spec_provenance_pass": False,
                "spec_hash": "sha256:" + "1" * 16,
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                **_spec_audit_context(),
                "recipe_matches": [],
                "field_audits": [],
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [{"message": "builder contract is invalid"}],
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "spec_mapping_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_format": "ebacktestcraft_yaml",
                "field_mappings": [
                    {
                        "source_field": "market.calendar_mixed_regions",
                        "target_field": "market.calendar",
                        "semantic": "strategy",
                        "status": "needs_user_confirmation",
                        "confirmation_required": True,
                        "blocking": False,
                        "reason": "Single-calendar execution needs user confirmation.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["spec-audit", "validate", str(audit_path), "--mapping-contract", str(contract_path), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(
        error["path"] == "mapping_contract.field_mappings[0].blocking"
        and "strategy semantics needing user confirmation require blocking=true" in error["message"]
        for error in payload["errors"]
    )


def test_spec_audit_validate_strict_confirmed_rejects_missing_effective_fields(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="strict_audit", hypothesis="strict audit")
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    path = tmp_path / "spec_audit.json"
    confirmation_table = tmp_path / "spec_confirmation_table.md"
    confirmation_table.write_text(_spec_confirmation_table_text(spec_path), encoding="utf-8")
    confirmation_event = _write_confirmation_event(
        tmp_path,
        artifact_path=str(confirmation_table),
        artifact_hash=_hash_file(confirmation_table),
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "pass",
                "audit_conclusion": "all_pass",
                "user_confirmation_status": "confirmed",
                "spec_confirmation_table": {
                    "path": str(confirmation_table),
                    "hash": _hash_file(confirmation_table),
                    "hash_type": "sha256",
                },
                "confirmation_event": confirmation_event,
                "spec_provenance_pass": True,
                "spec_hash": spec.compute_hash(),
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                **_spec_audit_context(),
                "recipe_matches": [],
                "field_audits": [],
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["spec-audit", "validate", str(path), "--spec", str(spec_path), "--strict-confirmed", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "field_audits[execution.initial_cash]" for error in payload["errors"])
    assert any("missing confirmed audit row" in error["message"] for error in payload["errors"])


def test_spec_audit_validate_strict_confirmed_rejects_default_status(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="strict_audit", hypothesis="strict audit")
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    field_audits = _confirmed_field_audits(spec_path)
    for item in field_audits:
        if item["field_path"] == "execution.initial_cash":
            item["status"] = "default"
            item["evidence"] = ["documented template default"]
            break
    path = tmp_path / "spec_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "pass",
                "audit_conclusion": "all_pass",
                "user_confirmation_status": "confirmed",
                "spec_confirmation_table": {
                    "path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
                    "hash": "sha256:" + "4" * 16,
                    "hash_type": "sha256",
                },
                "confirmation_event": _confirmation_event(),
                "spec_provenance_pass": True,
                "spec_hash": spec.compute_hash(),
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                **_spec_audit_context(),
                "recipe_matches": [],
                "field_audits": field_audits,
                "component_audits": [],
                "missing_user_requirements": [],
                "agent_added_fields": [],
                "contradictions": [],
                "blocking_findings": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["spec-audit", "validate", str(path), "--spec", str(spec_path), "--strict-confirmed", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "field_audits[execution.initial_cash].status" for error in payload["errors"])


def test_spec_audit_validate_strict_confirmed_accepts_full_effective_field_confirmation(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="strict_audit", hypothesis="strict audit")
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    path = tmp_path / "spec_audit.json"
    confirmation_table = tmp_path / "spec_confirmation_table.md"
    confirmation_table.write_text(_spec_confirmation_table_text(spec_path), encoding="utf-8")
    idea_brief_path = tmp_path / "strategy_idea_brief.json"
    conversation_hash = "sha256:" + "2" * 16
    idea_brief_path.write_text(
        json.dumps(
            {
                "strategy_name": "strict_audit",
                "conversation_hash": conversation_hash,
            }
        ),
        encoding="utf-8",
    )
    idea_audit_path = tmp_path / "strategy_idea_audit.json"
    idea_audit_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "idea_workflow_pass": True,
                "strategy_idea_brief": str(idea_brief_path),
                "strategy_idea_brief_hash": _hash_json_file(idea_brief_path),
                "conversation_hash": conversation_hash,
                "next_required_phase": "build",
            }
        ),
        encoding="utf-8",
    )
    mapping_contract_path = tmp_path / "spec_mapping_contract.json"
    mapping_contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_format": "strategy_idea_brief",
                "source_fields": ["strategy_name"],
                "field_mappings": [
                    {
                        "source_field": "strategy_name",
                        "target_field": "strategy_id",
                        "semantic": "strategy",
                        "status": "mapped",
                        "confirmation_required": False,
                        "blocking": False,
                        "reason": "The confirmed name supplies the strategy id.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": str(confirmation_table),
            "hash": _hash_file(confirmation_table),
            "hash_type": "sha256",
        },
        "spec_provenance_pass": True,
        "spec_hash": spec.compute_hash(),
        "conversation_hash": conversation_hash,
        "catalog_hash": "sha256:" + "3" * 16,
        **_spec_audit_context(),
        "strategy_idea_brief": str(idea_brief_path),
        "strategy_idea_audit": str(idea_audit_path),
        "strategy_idea_brief_hash": _hash_json_file(idea_brief_path),
        "strategy_idea_audit_hash": _hash_json_file(idea_audit_path),
        "spec_mapping_contract": str(mapping_contract_path),
        "spec_mapping_contract_hash": _hash_json_file(mapping_contract_path),
        "spec_mapping_contract_status": "pass",
        "recipe_matches": [],
        "field_audits": _confirmed_field_audits(spec_path),
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    audit["confirmation_event"] = _write_confirmation_event(
        tmp_path,
        artifact_path=str(confirmation_table),
        artifact_hash=_hash_file(confirmation_table),
        spec_audit_hash=_pre_confirmation_spec_audit_hash(audit),
    )
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "spec-audit",
            "validate",
            str(path),
            "--spec",
            str(spec_path),
            "--mapping-contract",
            str(mapping_contract_path),
            "--strict-confirmed",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_spec_audit_validate_rejects_missing_confirmation_event_artifact_without_strict(tmp_path) -> None:
    audit_path = _write_pass_spec_audit(
        tmp_path,
        "sha256:" + "1" * 16,
        "sha256:" + "3" * 16,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    (tmp_path / audit["confirmation_event"]["path"]).unlink()

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(audit_path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "confirmation_event.path" for error in payload["errors"])


def test_runtime_audit_validate_accepts_required_schema(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": True,
        "strategy_source_path": "compile_preview/strategy.py",
        "strategy_source_hash": "sha256:" + "4" * 64,
        "spec_hash": "sha256:" + "1" * 16,
        "spec_audit_hash": "sha256:" + "2" * 16,
        "compiled_plan_hash": "sha256:" + "3" * 16,
        "component_bundle_hashes": ["sha256:" + "4" * 16],
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": [
            {
                "field_path": "portfolio.rules.rebalance",
                "spec_value": {"params": {"interval_days": 10}},
                "runtime_path": "execution.rebalance",
                "runtime_value": {"interval_days": 10},
                "status": "preserved",
                "evidence": ["compiled plan preserved interval_days"],
                "blocking": False,
            }
        ],
        "blocking_findings": [],
    }
    path = tmp_path / "runtime_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["runtime-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_runtime_audit_validate_rejects_pass_with_false_runtime_gate(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": False,
        "strategy_source_path": "compile_preview/strategy.py",
        "strategy_source_hash": "sha256:" + "4" * 64,
        "spec_hash": "sha256:" + "1" * 16,
        "spec_audit_hash": "sha256:" + "2" * 16,
        "compiled_plan_hash": "sha256:" + "3" * 16,
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": [],
        "blocking_findings": [],
    }
    path = tmp_path / "runtime_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["runtime-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "runtime_semantics_pass" for error in payload["errors"])


def test_runtime_audit_validate_rejects_invalid_component_bundle_hashes(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": True,
        "strategy_source_path": "compile_preview/strategy.py",
        "strategy_source_hash": "sha256:" + "4" * 64,
        "spec_hash": "sha256:" + "1" * 16,
        "spec_audit_hash": "sha256:" + "2" * 16,
        "compiled_plan_hash": "sha256:" + "3" * 16,
        "component_bundle_hashes": ["not-a-hash"],
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": [],
        "blocking_findings": [],
    }
    path = tmp_path / "runtime_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["runtime-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "component_bundle_hashes[0]" for error in payload["errors"])


def test_runtime_audit_validate_rejects_pass_with_blocking_findings(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": True,
        "strategy_source_path": "compile_preview/strategy.py",
        "strategy_source_hash": "sha256:" + "4" * 64,
        "spec_hash": "sha256:" + "1" * 16,
        "spec_audit_hash": "sha256:" + "2" * 16,
        "compiled_plan_hash": "sha256:" + "3" * 16,
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": [],
        "blocking_findings": [{"message": "compiled plan omitted trade_time"}],
    }
    path = tmp_path / "runtime_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["runtime-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "blocking_findings" for error in payload["errors"])


def test_runtime_audit_validate_rejects_pass_with_mismatch_material_field(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": True,
        "strategy_source_path": "compile_preview/strategy.py",
        "strategy_source_hash": "sha256:" + "4" * 64,
        "spec_hash": "sha256:" + "1" * 16,
        "spec_audit_hash": "sha256:" + "2" * 16,
        "compiled_plan_hash": "sha256:" + "3" * 16,
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": [
            {
                "field_path": "execution.trade_time",
                "spec_value": "next_open",
                "runtime_path": "execution.fill_price_mode",
                "runtime_value": "close",
                "status": "mismatch",
                "evidence": ["compile preview changed fill timing"],
                "blocking": False,
            }
        ],
        "blocking_findings": [],
    }
    path = tmp_path / "runtime_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["runtime-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "material_field_audits[0].status" for error in payload["errors"])


def test_runtime_audit_validate_rejects_pass_with_blocking_material_field(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": True,
        "strategy_source_path": "compile_preview/strategy.py",
        "strategy_source_hash": "sha256:" + "4" * 64,
        "spec_hash": "sha256:" + "1" * 16,
        "spec_audit_hash": "sha256:" + "2" * 16,
        "compiled_plan_hash": "sha256:" + "3" * 16,
        "compiled_plan_path": "compile_preview/compiled_plan.json",
        "material_field_audits": [
            {
                "field_path": "execution.trade_time",
                "spec_value": "next_open",
                "runtime_path": "execution.fill_price_mode",
                "runtime_value": "next_open",
                "status": "preserved",
                "evidence": ["compiled plan preserved fill timing"],
                "blocking": True,
            }
        ],
        "blocking_findings": [],
    }
    path = tmp_path / "runtime_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["runtime-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "material_field_audits[0].blocking" for error in payload["errors"])


def test_backtest_attach_provenance_preserves_run_digest(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    artifact_hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    run_digest_lines = (run_dir.parent / "run_digests.jsonl").read_text(encoding="utf-8").splitlines()
    last_digest = json.loads(run_digest_lines[-1])
    assert payload["status"] == "pass"
    assert "spec_audit.json" in artifact_hashes
    assert "conversation_hash.txt" in artifact_hashes
    assert (run_dir / "conversation_hash.txt").read_text(encoding="utf-8").strip() == audit["conversation_hash"]
    assert last_digest["run_id"] == run_dir.name
    assert last_digest["artifact_hashes"] == payload["artifact_hashes_digest"]


def test_backtest_attach_provenance_accepts_full_spec_confirmation_table_hash(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))
    table_path = Path(audit["spec_confirmation_table"]["path"])
    full_table_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    audit["spec_confirmation_table"]["hash"] = full_table_hash
    audit["confirmation_event"] = _write_confirmation_event(
        tmp_path,
        artifact_path=str(table_path),
        artifact_hash=full_table_hash,
        spec_audit_hash=_pre_confirmation_spec_audit_hash(audit),
    )
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"


def test_backtest_attach_provenance_rejects_missing_runtime_audit(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--component-catalog",
            str(component_catalog),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "Missing option '--runtime-audit'" in result.output


def test_backtest_attach_provenance_rejects_spec_audit_without_user_confirmation(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))
    audit["user_confirmation_status"] = "pending"
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "user_confirmation_status" in result.output


def test_backtest_attach_provenance_rejects_tampered_spec_confirmation_table(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))
    table_path = Path(audit["spec_confirmation_table"]["path"])
    table_path.write_text(table_path.read_text(encoding="utf-8") + "\n| tampered | yes |\n", encoding="utf-8")
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "spec_confirmation_table.hash" in result.output


def test_backtest_attach_provenance_rejects_tampered_confirmation_event(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))
    event_path = tmp_path / str(audit["confirmation_event"]["path"])
    event_path.write_text(
        event_path.read_text(encoding="utf-8").replace('"field_scope": "full_spec_table"', '"field_scope": "partial"'),
        encoding="utf-8",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "confirmation_event" in result.output


def test_backtest_attach_provenance_rejects_mismatched_strategy_source_hash(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    payload = json.loads(runtime_audit.read_text(encoding="utf-8"))
    payload["strategy_source_hash"] = "sha256:" + "f" * 64
    runtime_audit.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "strategy_source_hash mismatch" in result.output


def test_backtest_attach_provenance_rejects_blocking_runtime_audit(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit, blocking=True)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "blocking material field row" in result.output


def test_backtest_attach_provenance_rejects_runtime_audit_fail_status(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    payload = json.loads(runtime_audit.read_text(encoding="utf-8"))
    payload["status"] = "fail"
    runtime_audit.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "runtime audit status must be pass before attaching provenance" in result.output


def test_backtest_attach_provenance_rejects_stale_runtime_audit_hashes(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    payload = json.loads(runtime_audit.read_text(encoding="utf-8"))
    payload["compiled_plan_hash"] = "sha256:" + "0" * 16
    runtime_audit.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "compiled_plan_hash mismatch" in result.output


def test_backtest_attach_provenance_rejects_runtime_audit_component_bundle_mismatch(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    run_bundle_hash = _attach_component_bundle_artifacts(run_dir)
    catalog = build_component_catalog(
        [
            {
                "_manifest_path": str(tmp_path / "run_component_manifest.json"),
                "bundle_hash": run_bundle_hash,
                "components": [
                    {
                        "name": "WorkspaceRunComponent",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "workspace_run_component",
                        "class": "WorkspaceRunComponent",
                    }
                ],
            }
        ]
    )
    component_catalog = tmp_path / "component_catalog.json"
    component_catalog.write_text(component_catalog_json(catalog), encoding="utf-8")
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "component_bundle_hashes" in result.output


def test_backtest_attach_provenance_rejects_component_bundle_not_in_catalog(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    run_bundle_hash = _attach_component_bundle_artifacts(run_dir)
    catalog = build_component_catalog(
        [
            {
                "_manifest_path": str(tmp_path / "component_manifest.json"),
                "bundle_hash": "sha256:" + "2" * 64,
                "components": [
                    {
                        "name": "WorkspaceCatalogOnly",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "workspace_catalog_only",
                        "class": "WorkspaceCatalogOnly",
                    }
                ],
            }
        ]
    )
    component_catalog = tmp_path / "component_catalog.json"
    component_catalog.write_text(component_catalog_json(catalog), encoding="utf-8")
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit, component_bundle_hashes=[run_bundle_hash])

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "component bundle hash mismatch" in result.output


def test_backtest_attach_provenance_allows_catalog_component_bundle_superset(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    run_bundle_hash = _attach_component_bundle_artifacts(run_dir)
    catalog = build_component_catalog(
        [
            {
                "_manifest_path": str(tmp_path / "run_component_manifest.json"),
                "bundle_hash": run_bundle_hash,
                "components": [
                    {
                        "name": "WorkspaceRunComponent",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "workspace_run_component",
                        "class": "WorkspaceRunComponent",
                    }
                ],
            },
            {
                "_manifest_path": str(tmp_path / "unused_component_manifest.json"),
                "bundle_hash": "sha256:" + "2" * 64,
                "components": [
                    {
                        "name": "WorkspaceUnusedComponent",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "workspace_unused_component",
                        "class": "WorkspaceUnusedComponent",
                    }
                ],
            },
        ]
    )
    component_catalog = tmp_path / "component_catalog.json"
    component_catalog.write_text(component_catalog_json(catalog), encoding="utf-8")
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit, component_bundle_hashes=[run_bundle_hash])

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Status: PASS" in result.output


def test_backtest_attach_provenance_rejects_blocking_audit(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))
    audit["status"] = "block"
    audit["audit_conclusion"] = "blocked"
    audit["user_confirmation_status"] = "rejected"
    audit["spec_confirmation_table"] = None
    audit["spec_provenance_pass"] = False
    audit["blocking_findings"] = [{"message": "confirm allocation"}]
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "spec audit status must be pass" in result.output


def test_backtest_attach_provenance_rejects_hash_mismatch(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        "sha256:" + "1" * 16,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    component_catalog = tmp_path / "component_catalog.json"
    component_catalog.write_text(
        json.dumps({"catalog_hash": "sha256:" + "4" * 64, "recipe_catalog_hash": "sha256:" + "5" * 64}),
        encoding="utf-8",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "invalid spec audit" in result.output
    assert "must match strategy spec hash" in result.output

    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        "sha256:" + "9" * 64,
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    catalog_result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert catalog_result.exit_code == 1
    assert "catalog hash mismatch" in catalog_result.output


def test_backtest_attach_provenance_rejects_nested_blockers(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))
    for item in audit["field_audits"]:
        if item["field_path"] == "execution.initial_cash":
            item["status"] = "unconfirmed"
            item["evidence"] = []
            item["blocking"] = True
            break
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "field_audits[execution.initial_cash].status" in result.output
    assert "must be confirmed before formal backtest" in result.output


def test_backtest_attach_provenance_rejects_tampered_catalog_body(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    tampered_catalog = dict(catalog)
    tampered_catalog["recipes"] = []
    component_catalog.write_text(json.dumps(tampered_catalog), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "component catalog hash mismatch" in result.output


def test_backtest_attach_provenance_rejects_non_reproducible_run(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] = 999
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
        ],
    )

    assert result.exit_code == 1
    assert "run reproducibility must pass before attaching provenance" in result.output


def _write_minimal_cli_run(tmp_path):
    spec = StrategySpec.template(strategy_id="attach_provenance", hypothesis="attach provenance")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    source_df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": source_df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "runs" / "run_1"
    run_dir.mkdir(parents=True)
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))
    return run_dir


def _write_pass_spec_audit(
    tmp_path: Path,
    spec_hash: str,
    catalog_hash: str,
    *,
    spec_path: Path | None = None,
) -> Path:
    confirmation_table = tmp_path / "spec_confirmation_table.md"
    if spec_path is not None:
        table_text = _spec_confirmation_table_text(spec_path)
    else:
        table_text = "| Field | Confirmed Value |\n| --- | --- |\n| spec_hash | confirmed |\n"
    confirmation_table.write_text(table_text, encoding="utf-8")
    audit = {
        "schema_version": 4,
        "status": "pass",
        "audit_conclusion": "all_pass",
        "user_confirmation_status": "confirmed",
        "spec_confirmation_table": {
            "path": str(confirmation_table),
            "hash": _hash_file(confirmation_table),
            "hash_type": "sha256",
        },
        "spec_provenance_pass": True,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog_hash,
        **_spec_audit_context(),
        "recipe_matches": [],
        "field_audits": _confirmed_field_audits(spec_path) if spec_path is not None else [],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    audit["confirmation_event"] = _write_confirmation_event(
        tmp_path,
        artifact_path=str(confirmation_table),
        artifact_hash=_hash_file(confirmation_table),
        spec_audit_hash=_pre_confirmation_spec_audit_hash(audit),
    )
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def _spec_confirmation_table_text(spec_path: Path) -> str:
    spec = StrategySpec.from_yaml(spec_path)
    rows = [
        "| Section | Field path | Spec value | Source | Audit status | Impact |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for field_path, value in _flatten_effective_fields(spec.to_effective_dict()):
        section = field_path.split(".", 1)[0]
        rows.append(
            "| "
            + " | ".join(
                [
                    section,
                    field_path,
                    json.dumps(value, sort_keys=True, default=str),
                    "User confirmed full SPEC table",
                    "confirmed",
                    "material",
                ]
            )
            + " |"
        )
    return "\n".join(rows) + "\n"


def _confirmed_field_audits(spec_path: Path) -> list[dict]:
    spec = StrategySpec.from_yaml(spec_path)
    return [
        {
            "field_path": field_path,
            "spec_value": value,
            "status": "confirmed",
            "material_category": _material_category_for_field_path(field_path),
            "evidence": [f"user confirmed {field_path} = {json.dumps(value, sort_keys=True, default=str)}"],
            "blocking": False,
        }
        for field_path, value in _flatten_effective_fields(spec.to_effective_dict())
    ]


def _material_category_for_field_path(field_path: str) -> str:
    if field_path.startswith(("signal.", "rules.")):
        return "strategy_logic"
    if field_path.startswith("portfolio."):
        return "portfolio_construction"
    if field_path.startswith("execution."):
        return "execution_assumption"
    if field_path.startswith("cost."):
        return "cost_assumption"
    if field_path.startswith(("data.", "universe.", "market.", "benchmark.")):
        return "data_assumption"
    if field_path.startswith("validation."):
        return "validation_assumption"
    if field_path.startswith(("metrics.", "decision_policy.")):
        return "metric_assumption"
    if field_path.startswith(("robustness.", "risk.")):
        return "risk_assumption"
    if field_path == "required_oxq_version":
        return "system_provenance"
    return "backtest_assumption"


def _flatten_effective_fields(value: object, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        if not value and prefix:
            return [(prefix, {})]
        fields: list[tuple[str, object]] = []
        for key in sorted(value):
            child_path = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(_flatten_effective_fields(value[key], child_path))
        return fields
    return [(prefix, value)]


def _write_pass_runtime_audit(
    run_dir: Path,
    spec_audit: Path,
    *,
    blocking: bool = False,
    component_bundle_hashes: list[str] | None = None,
) -> Path:
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    spec = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml").to_effective_dict()
    compiled_plan = json.loads((run_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    material_paths = (
        ("required_oxq_version", "open_xquant_version"),
        ("market", "market"),
        ("universe", "universe"),
        ("data", "data"),
        ("signal", "signals"),
        ("portfolio", "portfolio"),
        ("execution", "execution"),
        ("cost", "cost"),
        ("benchmark", "benchmark"),
        ("validation", "validation"),
        ("metrics", "metrics"),
    )
    path = spec_audit.with_name("runtime_audit.json")
    payload = {
        "schema_version": 2,
        "status": "pass",
        "runtime_semantics_pass": True,
        "strategy_source_path": str(run_dir / "strategy.py"),
        "strategy_source_hash": "sha256:" + hashlib.sha256((run_dir / "strategy.py").read_bytes()).hexdigest(),
        "spec_hash": spec_hash,
        "spec_audit_hash": _hash_json_file(spec_audit),
        "compiled_plan_hash": _hash_json_file(run_dir / "compiled_plan.json"),
        "compiled_plan_path": str(run_dir / "compiled_plan.json"),
        "material_field_audits": [
            {
                "field_path": field_path,
                "spec_value": spec[field_path],
                "runtime_path": runtime_path,
                "runtime_value": compiled_plan[runtime_path],
                "status": "mismatch" if blocking else "preserved",
                "evidence": ["test fixture"],
                "blocking": blocking,
            }
            for field_path, runtime_path in material_paths
        ],
        "blocking_findings": [],
    }
    payload["component_bundle_hashes"] = component_bundle_hashes or []
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_backtest_attach_provenance_rejects_empty_runtime_material_coverage(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    payload = json.loads(runtime_audit.read_text(encoding="utf-8"))
    payload["material_field_audits"] = []
    runtime_audit.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "missing material field audit row" in result.output


def test_backtest_attach_provenance_rejects_omitted_execution_runtime_coverage(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    payload = json.loads(runtime_audit.read_text(encoding="utf-8"))
    payload["material_field_audits"] = [
        row for row in payload["material_field_audits"] if row["field_path"] != "execution"
    ]
    runtime_audit.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "missing material field audit row for execution" in result.output


def test_backtest_attach_provenance_rejects_duplicate_runtime_material_row(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    payload = json.loads(runtime_audit.read_text(encoding="utf-8"))
    execution_row = next(row for row in payload["material_field_audits"] if row["field_path"] == "execution")
    payload["material_field_audits"].append(execution_row)
    runtime_audit.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate material field path" in result.output


def test_backtest_attach_provenance_rejects_tampered_runtime_material_value(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        spec_hash,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    runtime_audit = _write_pass_runtime_audit(run_dir, spec_audit)
    payload = json.loads(runtime_audit.read_text(encoding="utf-8"))
    execution_row = next(row for row in payload["material_field_audits"] if row["field_path"] == "execution")
    execution_row["runtime_value"]["fill_price_mode"] = "close"
    runtime_audit.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
            "--runtime-audit",
            str(runtime_audit),
            "--component-catalog",
            str(component_catalog),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "does not match compiled runtime field execution" in result.output


def _attach_component_bundle_artifacts(run_dir: Path) -> str:
    archive_base = run_dir / "component_extensions" / "00_custom_components"
    source_dir = archive_base / "custom_components" / "oxq_components" / "indicators"
    source_dir.mkdir(parents=True)
    (archive_base / "custom_components" / "oxq_components" / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    source = source_dir / "workspace_run_component.py"
    source.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import pandas as pd",
                "class WorkspaceRunComponent:",
                "    name = 'WorkspaceRunComponent'",
                "    def compute(self, mktdata: pd.DataFrame) -> pd.Series:",
                "        return pd.Series(1.0, index=mktdata.index, name=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = archive_base / "component_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extension_id": "custom_components",
                "extension_root": "custom_components",
                "bundle_hash": "",
                "components": [
                    {
                        "name": "WorkspaceRunComponent",
                        "kind": "Indicator",
                        "source": "workspace_extension",
                        "module": "oxq_components.indicators.workspace_run_component",
                        "class": "WorkspaceRunComponent",
                        "protocol": "Indicator",
                        "source_path": "oxq_components/indicators/workspace_run_component.py",
                        "source_hash": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bundle_hash = compute_component_bundle_hash(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_hash"] = bundle_hash
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "component_manifests.json").write_text(
        json.dumps(
            [
                {
                    "manifest_path": "/deleted/component_manifest.json",
                    "archived_manifest_path": "component_extensions/00_custom_components/component_manifest.json",
                    "archived_extension_root": "component_extensions/00_custom_components/custom_components",
                    "extension_id": "custom_components",
                    "bundle_hash": bundle_hash,
                    "components": [
                        {
                            "name": "WorkspaceRunComponent",
                            "kind": "Indicator",
                            "module": "oxq_components.indicators.workspace_run_component",
                            "class": "WorkspaceRunComponent",
                        }
                    ],
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "component_bundle_hash.txt").write_text(bundle_hash + "\n", encoding="utf-8")
    artifact_hashes_path = run_dir / "artifact_hashes.json"
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    artifact_hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    artifact_hashes["component_bundle_hash.txt"] = _hash_file(run_dir / "component_bundle_hash.txt")
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8")
    _append_run_digest(run_dir, _hash_json_file(artifact_hashes_path))
    return bundle_hash


def _write_component_catalog(tmp_path):
    catalog = build_component_catalog()
    component_catalog = tmp_path / "component_catalog.json"
    component_catalog.write_text(component_catalog_json(catalog), encoding="utf-8")
    return component_catalog, catalog


@pytest.mark.parametrize("versions_dir", [None, "", 7])
def test_spec_init_rejects_malformed_configured_versions_dir_as_invalid_governed(
    tmp_path,
    versions_dir: object,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        cwd_path = Path(cwd)
        config_dir = cwd_path / ".open-xquant"
        config_dir.mkdir()
        (config_dir / "workspace.yaml").write_text(
            yaml.safe_dump(
                {"paths": {"versions_dir": versions_dir}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        before = {
            path.relative_to(cwd_path).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(cwd_path.rglob("*"))
        }

        result = runner.invoke(main, ["spec", "init", "invalid governed root"])

        assert result.exit_code == 1
        assert "workspace paths.versions_dir must be a non-empty string" in result.output
        assert {
            path.relative_to(cwd_path).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(cwd_path.rglob("*"))
        } == before
