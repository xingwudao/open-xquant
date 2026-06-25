from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd
import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.core.component_catalog import build_component_catalog, component_catalog_json
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _write_artifacts
from oxq.spec.schema import StrategySpec


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
    assert spec["strategy_id"] == "sma_rsi_crossover"


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
        "top_n_normalized_weights",
        "top_n_positive_momentum_rotation",
        "volatility_adjusted_momentum",
    }
    assert "Catalog hash:" in result.output


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


def test_strategy_compile_writes_compile_preview(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="compile_preview", hypothesis="compile preview should be auditable")
    spec.portfolio.rules["rebalance"] = {
        "type": "RebalanceFrequencyRule",
        "params": {"interval_days": 10},
    }
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "compile_preview"

    result = CliRunner().invoke(main, ["strategy", "compile", str(spec_path), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    compiled_plan = json.loads((out_dir / "compiled_plan.json").read_text(encoding="utf-8"))
    assert compiled_plan["execution"]["rebalance"]["interval_days"] == 10
    assert compiled_plan["execution"]["rebalance"]["source"] == "portfolio.rules.rebalance"
    assert (out_dir / "spec_hash.txt").read_text(encoding="utf-8").strip() == compiled_plan["spec_hash"]


def test_spec_audit_validate_accepts_required_schema(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "block",
        "spec_provenance_pass": False,
        "runtime_semantics_pass": False,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
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
    assert any(error["path"] == "runtime_semantics_pass" for error in payload["errors"])


def test_spec_audit_validate_rejects_legacy_v1_after_gate_breaking_change(tmp_path) -> None:
    audit = {
        "schema_version": 1,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
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
    assert any(error["path"] == "schema_version" and "must be 2" in error["message"] for error in payload["errors"])


def test_spec_audit_validate_rejects_malformed_entries(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "blocked",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
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
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": False,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
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
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "field_audits[0].status" for error in payload["errors"])
    assert any(error["path"] == "runtime_semantics_pass" for error in payload["errors"])


def test_spec_audit_validate_rejects_confirmed_when_same_evidence_denies_field_confirmation(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
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
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "field_audits[0].status" for error in payload["errors"])


def test_spec_audit_validate_allows_confirmed_after_later_confirmation(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
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
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_spec_audit_validate_allows_confirmation_before_historical_negative_context(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
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
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output


def test_spec_audit_validate_rejects_pass_with_false_gate(tmp_path) -> None:
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": False,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
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
    assert any(error["path"] == "runtime_semantics_pass" for error in payload["errors"])


def test_backtest_attach_provenance_preserves_run_digest(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog["catalog_hash"],
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "portfolio.type",
                "spec_value": "EqualWeight",
                "status": "confirmed",
                "evidence": [],
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    spec_audit = tmp_path / "spec_audit.json"
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")

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


def test_backtest_attach_provenance_rejects_blocking_audit(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    audit = {
        "schema_version": 2,
        "status": "block",
        "spec_provenance_pass": False,
        "runtime_semantics_pass": False,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog["catalog_hash"],
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "portfolio.type",
                "spec_value": "EqualWeight",
                "status": "unconfirmed",
                "evidence": [],
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [{"message": "confirm allocation"}],
    }
    spec_audit = tmp_path / "spec_audit.json"
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")

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
        ],
    )

    assert result.exit_code == 1
    assert "spec audit status must be pass" in result.output


def test_backtest_attach_provenance_rejects_hash_mismatch(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog["catalog_hash"],
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "portfolio.type",
                "spec_value": "EqualWeight",
                "status": "confirmed",
                "evidence": [],
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    spec_audit = tmp_path / "spec_audit.json"
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")
    component_catalog = tmp_path / "component_catalog.json"
    component_catalog.write_text(
        json.dumps({"catalog_hash": "sha256:" + "4" * 64, "recipe_catalog_hash": "sha256:" + "5" * 64}),
        encoding="utf-8",
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
        ],
    )

    assert result.exit_code == 1
    assert "spec audit hash mismatch" in result.output

    audit["spec_hash"] = spec_hash
    audit["catalog_hash"] = "sha256:" + "9" * 64
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")

    catalog_result = CliRunner().invoke(
        main,
        [
            "backtest",
            "attach-provenance",
            str(run_dir),
            "--spec-audit",
            str(spec_audit),
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
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog["catalog_hash"],
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "execution.initial_cash",
                "spec_value": 100000,
                "status": "unconfirmed",
                "evidence": [],
                "blocking": True,
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    spec_audit = tmp_path / "spec_audit.json"
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")

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
        ],
    )

    assert result.exit_code == 1
    assert "blocking field audit row" in result.output


def test_backtest_attach_provenance_rejects_tampered_catalog_body(tmp_path) -> None:
    run_dir = _write_minimal_cli_run(tmp_path)
    spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    component_catalog, catalog = _write_component_catalog(tmp_path)
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog["catalog_hash"],
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "portfolio.type",
                "spec_value": "EqualWeight",
                "status": "confirmed",
                "evidence": [],
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    spec_audit = tmp_path / "spec_audit.json"
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")
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
    audit = {
        "schema_version": 2,
        "status": "pass",
        "spec_provenance_pass": True,
        "runtime_semantics_pass": True,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog["catalog_hash"],
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "portfolio.type",
                "spec_value": "EqualWeight",
                "status": "confirmed",
                "evidence": [],
            }
        ],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    spec_audit = tmp_path / "spec_audit.json"
    spec_audit.write_text(json.dumps(audit), encoding="utf-8")
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


def _write_component_catalog(tmp_path):
    catalog = build_component_catalog()
    component_catalog = tmp_path / "component_catalog.json"
    component_catalog.write_text(component_catalog_json(catalog), encoding="utf-8")
    return component_catalog, catalog
