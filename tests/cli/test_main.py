from __future__ import annotations

import json

import yaml
from click.testing import CliRunner

from oxq.cli.main import main


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


def test_spec_audit_validate_accepts_required_schema(tmp_path) -> None:
    audit = {
        "schema_version": 1,
        "status": "block",
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
        "field_audits": [{"field_path": "portfolio.type", "status": "confirmed", "evidence": [], "blocking": False}],
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


def test_spec_audit_validate_rejects_malformed_entries(tmp_path) -> None:
    audit = {
        "schema_version": 1,
        "status": "blocked",
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
    assert any(error["path"] == "component_audits[0].status" for error in payload["errors"])
