from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.core.component_catalog import build_component_catalog, component_catalog_json
from oxq.core.component_manifest import compute_component_bundle_hash
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _append_run_digest, _hash_file, _hash_json_file, _write_artifacts
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
    assert compiled_plan["execution"]["rebalance"]["interval_days"] == 10
    assert compiled_plan["execution"]["rebalance"]["source"] == "portfolio.rules.rebalance"
    assert compiled_plan["data"]["spec_data_dir"] == "data"
    assert compiled_plan["data"]["effective_data_dir"] == str(explicit_data_dir.resolve())
    assert (out_dir / "spec_hash.txt").read_text(encoding="utf-8").strip() == compiled_plan["spec_hash"]
    assert "Effective data dir:" in result.output
    assert "included in compiled_plan.json and its hash" in result.output


def test_spec_audit_validate_accepts_required_schema(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "block",
        "spec_provenance_pass": False,
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


def test_spec_audit_validate_rejects_legacy_v1_after_gate_breaking_change(tmp_path) -> None:
    audit = {
        "schema_version": 1,
        "status": "pass",
        "spec_provenance_pass": True,
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
    assert any(error["path"] == "schema_version" and "must be 3" in error["message"] for error in payload["errors"])


def test_spec_audit_validate_rejects_malformed_entries(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "blocked",
        "spec_provenance_pass": True,
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
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
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


def test_spec_audit_validate_rejects_confirmed_when_same_evidence_denies_field_confirmation(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
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


def test_spec_audit_validate_rejects_confirmation_for_other_field_before_denial(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
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
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert any(error["path"] == "field_audits[0].status" for error in payload["errors"])


def test_spec_audit_validate_allows_confirmed_after_later_confirmation(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
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


def test_spec_audit_validate_allows_later_confirmation_in_separate_evidence_entry(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
        "spec_hash": "sha256:" + "1" * 16,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": "sha256:" + "3" * 16,
        "recipe_matches": [],
        "field_audits": [
            {
                "field_path": "validation.train_period",
                "spec_value": ["2025-01-01", "2025-12-31"],
                "status": "confirmed",
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
    path.write_text(json.dumps(audit), encoding="utf-8")

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_spec_audit_validate_allows_confirmation_before_historical_negative_context(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
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


def test_spec_audit_validate_rejects_pass_with_false_provenance_gate(tmp_path) -> None:
    audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": False,
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
    assert any(error["path"] == "spec_provenance_pass" for error in payload["errors"])


def test_spec_audit_validate_strict_confirmed_requires_spec(tmp_path) -> None:
    path = tmp_path / "spec_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "pass",
                "spec_provenance_pass": True,
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
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["spec-audit", "validate", str(path), "--strict-confirmed", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert any(error["path"] == "spec" for error in payload["errors"])


def test_spec_audit_validate_strict_confirmed_rejects_missing_effective_fields(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="strict_audit", hypothesis="strict audit")
    spec_path = tmp_path / "strategy_spec.yaml"
    spec_path.write_text(yaml.dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    path = tmp_path / "spec_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "pass",
                "spec_provenance_pass": True,
                "spec_hash": spec.compute_hash(),
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
                "schema_version": 3,
                "status": "pass",
                "spec_provenance_pass": True,
                "spec_hash": spec.compute_hash(),
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
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
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "pass",
                "spec_provenance_pass": True,
                "spec_hash": spec.compute_hash(),
                "conversation_hash": "sha256:" + "2" * 16,
                "catalog_hash": "sha256:" + "3" * 16,
                "recipe_matches": [],
                "field_audits": _confirmed_field_audits(spec_path),
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

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "pass"


def test_runtime_audit_validate_accepts_required_schema(tmp_path) -> None:
    audit = {
        "schema_version": 1,
        "status": "pass",
        "runtime_semantics_pass": True,
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
        "schema_version": 1,
        "status": "pass",
        "runtime_semantics_pass": False,
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
        "schema_version": 1,
        "status": "pass",
        "runtime_semantics_pass": True,
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
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))

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
    _attach_component_bundle_artifacts(run_dir)
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
    audit["spec_provenance_pass"] = False
    audit["blocking_findings"] = [{"message": "confirm allocation"}]
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
    spec_audit = _write_pass_spec_audit(
        tmp_path,
        "sha256:" + "1" * 16,
        catalog["catalog_hash"],
        spec_path=run_dir / "strategy_spec.yaml",
    )
    audit = json.loads(spec_audit.read_text(encoding="utf-8"))
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
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
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


def _write_pass_spec_audit(
    tmp_path: Path,
    spec_hash: str,
    catalog_hash: str,
    *,
    spec_path: Path | None = None,
) -> Path:
    audit = {
        "schema_version": 3,
        "status": "pass",
        "spec_provenance_pass": True,
        "spec_hash": spec_hash,
        "conversation_hash": "sha256:" + "2" * 16,
        "catalog_hash": catalog_hash,
        "recipe_matches": [],
        "field_audits": _confirmed_field_audits(spec_path) if spec_path is not None else [],
        "component_audits": [],
        "missing_user_requirements": [],
        "agent_added_fields": [],
        "contradictions": [],
        "blocking_findings": [],
    }
    path = tmp_path / "spec_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def _confirmed_field_audits(spec_path: Path) -> list[dict]:
    spec = StrategySpec.from_yaml(spec_path)
    return [
        {
            "field_path": field_path,
            "spec_value": value,
            "status": "confirmed",
            "evidence": [f"user confirmed {field_path} = {json.dumps(value, sort_keys=True, default=str)}"],
            "blocking": False,
        }
        for field_path, value in _flatten_effective_fields(spec.to_effective_dict())
    ]


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
    path = spec_audit.with_name("runtime_audit.json")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "runtime_semantics_pass": True,
        "spec_hash": spec_hash,
        "spec_audit_hash": _hash_json_file(spec_audit),
        "compiled_plan_hash": _hash_json_file(run_dir / "compiled_plan.json"),
        "compiled_plan_path": str(run_dir / "compiled_plan.json"),
        "material_field_audits": [
            {
                "field_path": "portfolio.rules.rebalance",
                "spec_value": {"type": "RebalanceFrequencyRule"},
                "runtime_path": "execution.rebalance",
                "runtime_value": {},
                "status": "mismatch" if blocking else "preserved",
                "evidence": ["test fixture"],
                "blocking": blocking,
            }
        ],
        "blocking_findings": [],
    }
    if component_bundle_hashes is not None:
        payload["component_bundle_hashes"] = component_bundle_hashes
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


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
